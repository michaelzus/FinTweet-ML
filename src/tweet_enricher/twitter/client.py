"""TwitterAPI.io HTTP client with rate limiting."""

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Optional

import requests
import urllib3

from tweet_enricher.config import TWITTER_API_KEY, TWITTER_RATE_LIMIT_DELAY, TWITTER_TWEETS_PER_REQUEST

# Suppress SSL warnings (known issue with TwitterAPI.io)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger(__name__)


@dataclass
class Tweet:
    """Represents a tweet from the API."""

    id: str
    text: str
    created_at: str  # UTC timestamp: "Wed Dec 17 15:01:11 +0000 2025"
    url: str
    author_name: str
    author_username: str
    symbols: list[str]  # Extracted tickers from entities.symbols
    raw_json: dict  # Full API response for storage


class TwitterClient:
    """Client for TwitterAPI.io with rate limiting and pagination."""

    BASE_URL = "https://api.twitterapi.io"
    DEFAULT_COUNT = TWITTER_TWEETS_PER_REQUEST  # 100 for paid tier
    TIMEOUT = 30  # Reduced from 60 - fail fast if API is slow

    def __init__(self, api_key: Optional[str] = None, rate_limit_delay: Optional[float] = None):
        """
        Initialize the Twitter client.

        Args:
            api_key: TwitterAPI.io API key (defaults to env var TWITTER_API_KEY)
            rate_limit_delay: Seconds between requests (defaults to config value)
        """
        self.api_key = api_key or TWITTER_API_KEY
        self.rate_limit_delay = rate_limit_delay or TWITTER_RATE_LIMIT_DELAY
        self._last_request_time: Optional[float] = None

        if not self.api_key:
            raise ValueError("Twitter API key not provided. Set TWITTER_API_KEY environment variable " "or pass api_key parameter.")

    def _wait_for_rate_limit(self) -> None:
        """Enforce rate limiting between requests."""
        if self._last_request_time is not None:
            elapsed = time.time() - self._last_request_time
            if elapsed < self.rate_limit_delay:
                wait_time = self.rate_limit_delay - elapsed
                logger.debug(f"Rate limiting: waiting {wait_time:.1f}s")
                time.sleep(wait_time)

    def _make_request(self, endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
        """
        Make a rate-limited request to the API.

        Args:
            endpoint: API endpoint path
            params: Query parameters

        Returns:
            JSON response data

        Raises:
            requests.RequestException: On network/API errors
        """
        self._wait_for_rate_limit()

        url = f"{self.BASE_URL}{endpoint}"
        headers = {"X-API-Key": self.api_key}

        logger.debug(f"Requesting {url} with params {params}")

        response = requests.get(
            url,
            headers=headers,
            params=params,
            verify=False,  # Known SSL issue with TwitterAPI.io
            timeout=self.TIMEOUT,
        )
        self._last_request_time = time.time()

        response.raise_for_status()
        return response.json()

    def _parse_tweet_timestamp(self, timestamp_str: str) -> Optional[datetime]:
        """
        Parse Twitter timestamp string to datetime.

        Args:
            timestamp_str: Twitter format "Wed Dec 17 15:01:11 +0000 2025"

        Returns:
            datetime object (timezone-aware UTC) or None if parsing fails
        """
        try:
            return datetime.strptime(timestamp_str, "%a %b %d %H:%M:%S %z %Y")
        except (ValueError, TypeError):
            return None

    def _parse_tweet(self, tweet_data: dict[str, Any]) -> Tweet:
        """Parse raw API tweet data into Tweet object."""
        author = tweet_data.get("author", {})

        # Extract symbols from entities
        entities = tweet_data.get("entities", {})
        symbols = [s.get("text", "") for s in entities.get("symbols", [])]

        return Tweet(
            id=tweet_data.get("id", ""),
            text=tweet_data.get("text", ""),
            created_at=tweet_data.get("createdAt", ""),
            url=tweet_data.get("twitterUrl") or tweet_data.get("url", ""),
            author_name=author.get("name", ""),
            author_username=author.get("userName", ""),
            symbols=symbols,
            raw_json=tweet_data,
        )

    def fetch_tweets(
        self,
        username: str,
        count: int = DEFAULT_COUNT,
        cursor: Optional[str] = None,
        include_replies: bool = False,
    ) -> tuple[list[Tweet], Optional[str], bool]:
        """
        Fetch tweets for a user.

        Args:
            username: Twitter username (without @)
            count: Number of tweets to fetch per request
            cursor: Pagination cursor from previous request
            include_replies: Whether to include reply tweets

        Returns:
            Tuple of (tweets, next_cursor, has_next_page)
        """
        params: dict[str, Any] = {
            "userName": username,
            "count": count,
            "includeReplies": "true" if include_replies else "false",
        }
        if cursor:
            params["cursor"] = cursor

        try:
            data = self._make_request("/twitter/user/last_tweets", params)

            # Parse response structure: data.tweets[]
            tweets_data = data.get("data", {}).get("tweets", [])
            tweets = [self._parse_tweet(t) for t in tweets_data]

            next_cursor = data.get("next_cursor")
            has_next_page = data.get("has_next_page", False)

            logger.info(f"Fetched {len(tweets)} tweets for @{username}")
            return tweets, next_cursor, has_next_page

        except requests.RequestException as e:
            logger.error(f"Failed to fetch tweets for @{username}: {e}")
            raise

    def fetch_tweets_incremental(
        self,
        username: str,
        exists_check: Callable[[str], bool],
        max_tweets: Optional[int] = None,
        until_date: Optional[datetime] = None,
        include_replies: bool = False,
        progress_callback: Optional[Callable[[dict], None]] = None,
        start_cursor: Optional[str] = None,
        save_callback: Optional[Callable[["Tweet"], bool]] = None,
    ) -> tuple[list[Tweet], int, Optional[str]]:
        """
        Fetch tweets incrementally, stopping as soon as we hit existing data or target date.

        This is cost-optimized: stops pagination immediately when reaching
        tweets we already have or when tweets are older than until_date.

        Args:
            username: Twitter username (without @)
            exists_check: Callable(tweet_id) -> bool, returns True if tweet exists
            max_tweets: Maximum number of tweets to fetch (None = no limit)
            until_date: Stop fetching when tweets are older than this date (UTC)
            include_replies: Whether to include reply tweets
            progress_callback: Optional callback for progress updates
            start_cursor: Optional cursor to resume from
            save_callback: Optional callback to save tweet IMMEDIATELY (for crash safety)
                          Returns True if tweet was saved (new), False if skipped (duplicate)

        Returns:
            Tuple of (new_tweets, api_calls_made, last_cursor)
        """
        cursor = start_cursor
        new_tweets: list[Tweet] = []
        api_calls = 0
        last_cursor = None
        skipped_existing = 0  # Track duplicates during backfill

        while True:
            tweets, next_cursor, has_next_page = self.fetch_tweets(
                username=username,
                cursor=cursor,
                include_replies=include_replies,
            )
            api_calls += 1
            last_cursor = cursor

            # Check first tweet - if it exists, we have no new data (only for incremental, not backfill)
            if tweets and until_date is None and exists_check(tweets[0].id):
                logger.info(f"First tweet {tweets[0].id} already exists - no new data")
                return new_tweets, api_calls, last_cursor

            # Process tweets until we hit existing data or target date
            found_existing = False
            reached_target_date = False

            for tweet in tweets:
                # Check if tweet is older than target date
                if until_date:
                    tweet_dt = self._parse_tweet_timestamp(tweet.created_at)
                    if tweet_dt and tweet_dt < until_date:
                        logger.info(f"Reached target date {until_date.date()}, stopping")
                        reached_target_date = True
                        break

                # For non-backfill: stop at existing data
                # For backfill: skip existing tweets but continue (to fill gaps)
                if exists_check(tweet.id):
                    if until_date is None:
                        # Regular incremental: stop here
                        logger.info(f"Reached existing tweet {tweet.id}, stopping pagination")
                        found_existing = True
                        break
                    else:
                        # Backfill mode: skip but continue
                        skipped_existing += 1
                        continue

                # Save immediately if callback provided (crash safety)
                if save_callback:
                    save_callback(tweet)

                new_tweets.append(tweet)

                # Progress callback
                if progress_callback:
                    tweet_dt = self._parse_tweet_timestamp(tweet.created_at)
                    progress_callback(
                        {
                            "tweets_fetched": len(new_tweets),
                            "api_calls": api_calls,
                            "current_date": tweet_dt.strftime("%Y-%m-%d") if tweet_dt else "unknown",
                            "skipped_existing": skipped_existing,
                        }
                    )

                # Stop if we've reached max tweets
                if max_tweets and len(new_tweets) >= max_tweets:
                    logger.info(f"Reached max tweets limit ({max_tweets})")
                    return new_tweets, api_calls, next_cursor

            # If we found existing data or reached target date, stop pagination
            if found_existing or reached_target_date:
                if skipped_existing > 0:
                    logger.info(f"Skipped {skipped_existing} existing tweets during backfill")
                return new_tweets, api_calls, last_cursor

            # Check if there are more pages
            if not has_next_page or not next_cursor:
                logger.info(f"No more pages, fetched {len(new_tweets)} new tweets")
                if skipped_existing > 0:
                    logger.info(f"Skipped {skipped_existing} existing tweets during backfill")
                return new_tweets, api_calls, None

            cursor = next_cursor

    def search_tweets_by_date(
        self,
        username: str,
        since_date: str,
        until_date: str,
        cursor: Optional[str] = None,
    ) -> tuple[list[Tweet], Optional[str], bool]:
        """
        Search tweets for a user within a specific date range.

        Uses the advanced_search endpoint with Twitter search syntax.

        Args:
            username: Twitter username (without @)
            since_date: Start date (YYYY-MM-DD format, inclusive)
            until_date: End date (YYYY-MM-DD format, exclusive)
            cursor: Pagination cursor from previous request

        Returns:
            Tuple of (tweets, next_cursor, has_next_page)
        """
        import time as _time

        start = _time.time()

        # Build Twitter search query with date filters
        query = f"from:{username} since:{since_date} until:{until_date}"

        params: dict[str, Any] = {
            "query": query,
            "queryType": "Latest",
        }
        if cursor:
            params["cursor"] = cursor

        try:
            data = self._make_request("/twitter/tweet/advanced_search", params)
            elapsed = _time.time() - start

            # Parse response - may be different structure than last_tweets
            tweets_data = data.get("tweets", [])
            if not tweets_data:
                tweets_data = data.get("data", {}).get("tweets", [])

            tweets = [self._parse_tweet(t) for t in tweets_data]

            next_cursor = data.get("next_cursor")
            has_next_page = data.get("has_next_page", False)

            logger.debug(f"Search {since_date}: {len(tweets)} tweets in {elapsed:.1f}s")
            return tweets, next_cursor, has_next_page

        except requests.RequestException as e:
            elapsed = _time.time() - start
            logger.error(f"Failed to search {since_date} after {elapsed:.1f}s: {e}")
            raise

    def fetch_tweets_by_date_range(
        self,
        username: str,
        since_date: str,
        until_date: str,
        exists_check: Callable[[str], bool],
        save_callback: Optional[Callable[["Tweet"], bool]] = None,
    ) -> tuple[list[Tweet], int, int]:
        """
        Fetch all tweets for a user from a specific date range.

        Args:
            username: Twitter username (without @)
            since_date: Start date (YYYY-MM-DD)
            until_date: End date (YYYY-MM-DD, exclusive)
            exists_check: Callable(tweet_id) -> bool
            save_callback: Optional callback to save tweet immediately

        Returns:
            Tuple of (new_tweets, api_calls_made, skipped_count)
        """
        cursor = None
        new_tweets: list[Tweet] = []
        api_calls = 0
        skipped = 0

        while True:
            tweets, next_cursor, has_next_page = self.search_tweets_by_date(
                username=username,
                since_date=since_date,
                until_date=until_date,
                cursor=cursor,
            )
            api_calls += 1

            if not tweets:
                break

            for tweet in tweets:
                if exists_check(tweet.id):
                    skipped += 1
                    continue

                if save_callback:
                    save_callback(tweet)

                new_tweets.append(tweet)

            if not has_next_page or not next_cursor:
                break

            cursor = next_cursor

        return new_tweets, api_calls, skipped

    def test_connection(self) -> bool:
        """
        Test the API connection.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            tweets, _, _ = self.fetch_tweets("StockMKTNewz", count=1)
            return len(tweets) > 0
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False

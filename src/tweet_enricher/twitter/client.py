"""TwitterAPI.io HTTP client with rate limiting."""

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterator, Optional

import requests
import urllib3

from tweet_enricher.config import TWITTER_API_KEY, TWITTER_RATE_LIMIT_DELAY

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
    DEFAULT_COUNT = 20
    TIMEOUT = 60

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
            raise ValueError(
                "Twitter API key not provided. Set TWITTER_API_KEY environment variable "
                "or pass api_key parameter."
            )

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
        exists_check: callable,
        max_tweets: Optional[int] = None,
        include_replies: bool = False,
    ) -> tuple[list[Tweet], int]:
        """
        Fetch tweets incrementally, stopping as soon as we hit existing data.
        
        This is cost-optimized: stops pagination immediately when reaching
        tweets we already have, minimizing API calls.

        Args:
            username: Twitter username (without @)
            exists_check: Callable(tweet_id) -> bool, returns True if tweet exists
            max_tweets: Maximum number of tweets to fetch (None = no limit)
            include_replies: Whether to include reply tweets

        Returns:
            Tuple of (new_tweets, api_calls_made)
        """
        cursor = None
        new_tweets: list[Tweet] = []
        api_calls = 0

        while True:
            tweets, next_cursor, has_next_page = self.fetch_tweets(
                username=username,
                cursor=cursor,
                include_replies=include_replies,
            )
            api_calls += 1

            # Check first tweet - if it exists, we have no new data
            if tweets and exists_check(tweets[0].id):
                logger.info(f"First tweet {tweets[0].id} already exists - no new data")
                return new_tweets, api_calls

            # Process tweets until we hit existing data
            found_existing = False
            for tweet in tweets:
                if exists_check(tweet.id):
                    logger.info(f"Reached existing tweet {tweet.id}, stopping pagination")
                    found_existing = True
                    break

                new_tweets.append(tweet)

                # Stop if we've reached max tweets
                if max_tweets and len(new_tweets) >= max_tweets:
                    logger.info(f"Reached max tweets limit ({max_tweets})")
                    return new_tweets, api_calls

            # If we found existing data, stop pagination (don't make more API calls)
            if found_existing:
                return new_tweets, api_calls

            # Check if there are more pages
            if not has_next_page or not next_cursor:
                logger.info(f"No more pages, fetched {len(new_tweets)} new tweets")
                return new_tweets, api_calls

            cursor = next_cursor

    def fetch_all_tweets(
        self,
        username: str,
        max_tweets: Optional[int] = None,
        since_id: Optional[str] = None,
        include_replies: bool = False,
    ) -> Iterator[Tweet]:
        """
        Fetch all tweets for a user with pagination.
        
        Note: For incremental sync with cost optimization, use fetch_tweets_incremental().

        Args:
            username: Twitter username (without @)
            max_tweets: Maximum number of tweets to fetch (None = no limit)
            since_id: Stop fetching when reaching this tweet ID
            include_replies: Whether to include reply tweets

        Yields:
            Tweet objects
        """
        cursor = None
        total_fetched = 0

        while True:
            tweets, next_cursor, has_next_page = self.fetch_tweets(
                username=username,
                cursor=cursor,
                include_replies=include_replies,
            )

            for tweet in tweets:
                # Stop if we've reached a tweet we already have
                if since_id and tweet.id == since_id:
                    logger.info(f"Reached existing tweet {since_id}, stopping")
                    return

                yield tweet
                total_fetched += 1

                # Stop if we've reached max tweets
                if max_tweets and total_fetched >= max_tweets:
                    logger.info(f"Reached max tweets limit ({max_tweets})")
                    return

            # Check if there are more pages
            if not has_next_page or not next_cursor:
                logger.info(f"No more pages, fetched {total_fetched} total tweets")
                return

            cursor = next_cursor

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


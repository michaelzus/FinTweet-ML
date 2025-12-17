"""Incremental tweet sync service."""

import logging
from datetime import datetime
from typing import Optional

import pytz

from tweet_enricher.config import TWITTER_ACCOUNTS
from tweet_enricher.parsers.discord import MessageCategorizer, MessageProcessor
from tweet_enricher.twitter.client import Tweet, TwitterClient
from tweet_enricher.twitter.database import ProcessedTweet, TweetDatabase

logger = logging.getLogger(__name__)


class SyncService:
    """Service for incrementally syncing tweets from Twitter accounts."""

    def __init__(
        self,
        client: Optional[TwitterClient] = None,
        database: Optional[TweetDatabase] = None,
        accounts: Optional[list[str]] = None,
        lazy_client: bool = False,
    ):
        """
        Initialize sync service.

        Args:
            client: TwitterClient instance (creates new one if not provided)
            database: TweetDatabase instance (creates new one if not provided)
            accounts: List of Twitter accounts to sync (defaults to config)
            lazy_client: If True, don't create client until needed (for read-only ops)
        """
        self._client = client
        self._lazy_client = lazy_client
        self.database = database or TweetDatabase()
        self.accounts = accounts or TWITTER_ACCOUNTS
        self.categorizer = MessageCategorizer()
        self.processor = MessageProcessor(min_text_length=30)  # Lower threshold for tweets

    @property
    def client(self) -> TwitterClient:
        """Get or create the Twitter client."""
        if self._client is None:
            self._client = TwitterClient()
        return self._client

    def _convert_utc_to_eastern(self, twitter_ts: str) -> str:
        """
        Convert Twitter UTC timestamp to US Eastern time.

        Args:
            twitter_ts: Twitter timestamp format "Wed Dec 17 15:01:11 +0000 2025"

        Returns:
            Eastern time string "2025-12-17 10:01:11"
        """
        try:
            # Parse Twitter format (already has timezone info +0000)
            dt = datetime.strptime(twitter_ts, "%a %b %d %H:%M:%S %z %Y")

            # Convert to US Eastern
            eastern_tz = pytz.timezone("America/New_York")
            dt_eastern = dt.astimezone(eastern_tz)

            return dt_eastern.strftime("%Y-%m-%d %H:%M:%S")
        except ValueError as e:
            logger.warning(f"Failed to parse timestamp '{twitter_ts}': {e}")
            return twitter_ts

    def _extract_tickers(self, tweet: Tweet) -> list[str]:
        """
        Extract tickers from tweet using API entities and text parsing.

        Args:
            tweet: Tweet object

        Returns:
            List of unique ticker symbols
        """
        tickers = set()

        # First, use API-provided symbols (most reliable)
        for symbol in tweet.symbols:
            if symbol:
                tickers.add(symbol.upper())

        # Also extract from text as backup
        text_tickers = self.processor.extract_tickers(tweet.text)
        for ticker in text_tickers:
            tickers.add(ticker.upper())

        return list(tickers)

    def _process_tweet(self, tweet: Tweet) -> list[ProcessedTweet]:
        """
        Process a tweet into one or more ProcessedTweet objects.

        Each ticker gets its own row (same format as Discord converter).

        Args:
            tweet: Raw Tweet from API

        Returns:
            List of ProcessedTweet objects (one per ticker)
        """
        tickers = self._extract_tickers(tweet)

        if not tickers:
            logger.debug(f"No tickers found in tweet {tweet.id}")
            return []

        # Clean the text
        clean_text = self.processor.clean_text(tweet.text)

        # Skip if text is too short or not English
        if not self.processor.is_valid(clean_text):
            logger.debug(f"Tweet {tweet.id} failed validation")
            return []

        # Categorize
        category = self.categorizer.categorize(clean_text)

        # Convert timestamp
        timestamp_et = self._convert_utc_to_eastern(tweet.created_at)

        # Create one row per ticker
        processed = []
        for ticker in tickers:
            processed.append(
                ProcessedTweet(
                    id=tweet.id,
                    timestamp_utc=tweet.created_at,
                    timestamp_et=timestamp_et,
                    author=tweet.author_username,
                    ticker=ticker,
                    tweet_url=tweet.url,
                    category=category,
                    text=clean_text,
                )
            )

        return processed

    def sync_account(
        self,
        account: str,
        full_sync: bool = False,
        max_tweets: Optional[int] = None,
    ) -> dict:
        """
        Sync tweets for a single account.
        
        Uses smart incremental fetching: stops API pagination immediately
        when reaching tweets we already have, minimizing API costs.

        Args:
            account: Twitter username
            full_sync: If True, ignore cursor and fetch all available tweets
            max_tweets: Maximum tweets to fetch (None = no limit)

        Returns:
            Dict with sync results including API calls made
        """
        logger.info(f"Starting sync for @{account}")

        # Get current sync state
        state = self.database.get_sync_state(account)
        
        if full_sync:
            logger.info("Full sync requested (ignoring existing data)")
            # For full sync, don't check existing - but still avoid duplicates on insert
            exists_check = lambda tweet_id: False
        else:
            if state and state.last_tweet_id:
                logger.info(f"Incremental sync (last tweet: {state.last_tweet_id})")
            else:
                logger.info("First sync for this account")
            # Check database for existing tweets
            exists_check = self.database.tweet_exists

        # Fetch tweets with smart incremental logic
        raw_count = 0
        processed_count = 0
        api_calls = 0
        new_tweets: list[ProcessedTweet] = []
        first_tweet_id = None

        try:
            # Use cost-optimized incremental fetch
            fetched_tweets, api_calls = self.client.fetch_tweets_incremental(
                username=account,
                exists_check=exists_check,
                max_tweets=max_tweets,
                include_replies=False,
            )

            for tweet in fetched_tweets:
                # Track first (most recent) tweet ID
                if first_tweet_id is None:
                    first_tweet_id = tweet.id

                # Store raw tweet (skip if already exists for full_sync)
                if self.database.insert_raw_tweet(tweet.id, account, tweet.raw_json):
                    raw_count += 1

                # Process tweet
                processed = self._process_tweet(tweet)
                new_tweets.extend(processed)

        except Exception as e:
            logger.error(f"Error fetching tweets for @{account}: {e}")
            raise

        # Batch insert processed tweets
        if new_tweets:
            processed_count = self.database.insert_processed_tweets(new_tweets)

        # Update sync state
        if first_tweet_id:
            self.database.update_sync_state(
                account=account,
                last_tweet_id=first_tweet_id,
                tweets_added=processed_count,
            )

        result = {
            "account": account,
            "raw_tweets_fetched": raw_count,
            "processed_tweets_added": processed_count,
            "api_calls": api_calls,
            "last_tweet_id": first_tweet_id,
        }

        logger.info(
            f"Sync complete for @{account}: "
            f"{raw_count} raw, {processed_count} processed ({api_calls} API calls)"
        )

        return result

    def sync_all(
        self,
        full_sync: bool = False,
        max_tweets_per_account: Optional[int] = None,
    ) -> list[dict]:
        """
        Sync all configured accounts.

        Args:
            full_sync: If True, ignore cursors and fetch all available tweets
            max_tweets_per_account: Maximum tweets to fetch per account

        Returns:
            List of sync results per account
        """
        logger.info(f"Starting sync for {len(self.accounts)} accounts")
        results = []

        for account in self.accounts:
            try:
                result = self.sync_account(
                    account=account,
                    full_sync=full_sync,
                    max_tweets=max_tweets_per_account,
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to sync @{account}: {e}")
                results.append({
                    "account": account,
                    "error": str(e),
                })

        # Summary
        total_raw = sum(r.get("raw_tweets_fetched", 0) for r in results)
        total_processed = sum(r.get("processed_tweets_added", 0) for r in results)
        total_api_calls = sum(r.get("api_calls", 0) for r in results)
        logger.info(
            f"Sync complete: {total_raw} raw tweets, {total_processed} processed "
            f"({total_api_calls} API calls)"
        )

        return results

    def get_status(self) -> dict:
        """
        Get sync status for all accounts.

        Returns:
            Dict with status information
        """
        states = self.database.get_all_sync_states()
        stats = self.database.get_stats()

        # Format sync states
        accounts_status = []
        for state in states:
            accounts_status.append({
                "account": state.account,
                "last_sync": state.last_sync_at.isoformat() if state.last_sync_at else None,
                "total_tweets": state.total_tweets,
                "last_tweet_id": state.last_tweet_id,
            })

        # Add accounts that haven't been synced yet
        synced_accounts = {s.account for s in states}
        for account in self.accounts:
            if account not in synced_accounts:
                accounts_status.append({
                    "account": account,
                    "last_sync": None,
                    "total_tweets": 0,
                    "last_tweet_id": None,
                })

        return {
            "database": stats,
            "accounts": accounts_status,
        }

    def export_csv(
        self,
        output_path: str,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        account: Optional[str] = None,
        ticker: Optional[str] = None,
    ) -> int:
        """
        Export tweets to CSV file.

        Args:
            output_path: Path to output CSV file
            since: Start date filter
            until: End date filter
            account: Filter by account
            ticker: Filter by ticker

        Returns:
            Number of rows exported
        """
        import csv

        tweets = self.database.get_processed_tweets(
            since=since,
            until=until,
            account=account,
            ticker=ticker,
        )

        if not tweets:
            logger.warning("No tweets found matching criteria")
            return 0

        # Sort by timestamp ascending for export
        tweets.sort(key=lambda t: t.timestamp_et)

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            # Header matching existing pipeline format
            writer.writerow(["timestamp", "author", "ticker", "tweet_url", "category", "text"])

            for tweet in tweets:
                writer.writerow([
                    tweet.timestamp_et,
                    tweet.author,
                    tweet.ticker,
                    tweet.tweet_url,
                    tweet.category,
                    tweet.text,
                ])

        logger.info(f"Exported {len(tweets)} tweets to {output_path}")
        return len(tweets)


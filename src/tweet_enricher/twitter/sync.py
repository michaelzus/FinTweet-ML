"""Incremental tweet sync service."""

import logging
import sys
from datetime import datetime, timedelta
from typing import Callable, Optional

import pytz

from tweet_enricher.config import TWITTER_ACCOUNTS, TWITTER_RATE_LIMIT_DELAY
from tweet_enricher.parsers.discord import MessageCategorizer, MessageProcessor
from tweet_enricher.text.cleaner import clean_for_finbert
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

    def sync_account_by_month(
        self,
        account: str,
        months_back: int = 6,
        show_progress: bool = False,
    ) -> dict:
        """
        Sync tweets for an account using day-by-day date-based search.

        This is efficient for historical backfill as it uses the
        advanced_search endpoint with date filters. Day-by-day is faster
        because each query returns fewer results.

        Args:
            account: Twitter username
            months_back: Number of months to fetch (default 6)
            show_progress: Show real-time progress updates

        Returns:
            Dict with sync results including API calls made
        """
        logger.info(f"Starting day-by-day sync for @{account} ({months_back} months)")

        # Calculate days to fetch
        now = datetime.now(pytz.UTC)
        total_days = months_back * 30

        raw_count = 0
        total_api_calls = 0
        total_skipped = 0
        first_tweet_id: Optional[str] = None
        new_tweets: list[ProcessedTweet] = []

        exists_check = self.database.tweet_exists
        days_skipped_journal = 0

        # Fetch day by day (oldest first for better progress display)
        # Skip today (day_offset=0) since tweets are still being posted
        for day_offset in range(total_days - 1, 0, -1):
            target_date = now - timedelta(days=day_offset)
            since_date = target_date.strftime("%Y-%m-%d")
            until_date = (target_date + timedelta(days=1)).strftime("%Y-%m-%d")

            days_done = total_days - day_offset
            pct = (days_done / total_days) * 100

            if show_progress:
                sys.stdout.write(f"\r  [{pct:5.1f}%] {since_date} | {raw_count:,} tweets | {total_api_calls} calls    ")
                sys.stdout.flush()

            # Check fetch journal - skip if already fetched
            if self.database.is_day_fetched(account, since_date):
                days_skipped_journal += 1
                if show_progress:
                    sys.stdout.write("S")  # Skipped (already in journal)
                    sys.stdout.flush()
                continue

            # Retry up to 3 times on failure
            max_retries = 3
            day_tweets_count = 0
            day_api_calls = 0
            success = False
            day_raw_tweets: list[Tweet] = []  # Collect tweets, only save on success

            for attempt in range(max_retries):
                try:
                    # Don't use save_callback here - collect tweets first, save only on success
                    day_tweets, api_calls, skipped = self.client.fetch_tweets_by_date_range(
                        username=account,
                        since_date=since_date,
                        until_date=until_date,
                        exists_check=exists_check,
                        save_callback=None,  # Don't save during fetch!
                    )
                    day_raw_tweets = day_tweets
                    day_tweets_count = len(day_tweets)
                    day_api_calls = api_calls
                    total_api_calls += api_calls
                    total_skipped += skipped
                    success = True

                    # Show dot for each day completed
                    if show_progress:
                        sys.stdout.write(".")
                        sys.stdout.flush()
                    break  # Success, move to next day

                except Exception as e:
                    if attempt < max_retries - 1:
                        # Retry after short delay
                        if show_progress:
                            sys.stdout.write("R")  # Mark retry
                            sys.stdout.flush()
                        import time

                        time.sleep(2)  # Wait 2s before retry
                    else:
                        # Final failure
                        if show_progress:
                            sys.stdout.write("X")  # Mark failed day
                            sys.stdout.flush()
                        logger.warning(f"Failed to fetch {since_date} after {max_retries} attempts: {e}")

            # Only save tweets AFTER the entire day's fetch succeeded
            if success and day_raw_tweets:
                for tweet in day_raw_tweets:
                    saved = self.database.insert_raw_tweet(tweet.id, account, tweet.raw_json)
                    if saved:
                        raw_count += 1
                        if first_tweet_id is None:
                            first_tweet_id = tweet.id

                    processed = self._process_tweet(tweet)
                    if processed:
                        self.database.insert_processed_tweets(processed)
                        new_tweets.extend(processed)

            # Mark day as fetched in journal (only on success)
            if success:
                self.database.mark_day_fetched(
                    account=account,
                    date=since_date,
                    tweets_count=day_tweets_count,
                    api_calls=day_api_calls,
                )

        if show_progress:
            print()  # New line after progress

        processed_count = len(new_tweets)

        # Always update sync state (even if no new tweets, to record the sync happened)
        self.database.update_sync_state(
            account=account,
            last_tweet_id=first_tweet_id,  # May be None if no new tweets
            tweets_added=raw_count,  # Use raw_count not processed (which is per-ticker)
        )

        result = {
            "account": account,
            "raw_tweets_fetched": raw_count,
            "processed_tweets_added": processed_count,
            "api_calls": total_api_calls,
            "skipped_existing": total_skipped,
            "days_fetched": total_days,
            "days_skipped_journal": days_skipped_journal,
        }

        skip_msg = f", {days_skipped_journal} days skipped" if days_skipped_journal > 0 else ""
        logger.info(
            f"Day-by-day sync complete for @{account}: "
            f"{raw_count} raw, {processed_count} processed ({total_api_calls} API calls{skip_msg})"
        )

        return result

    def sync_account(
        self,
        account: str,
        full_sync: bool = False,
        max_tweets: Optional[int] = None,
        months_back: Optional[int] = None,
        show_progress: bool = False,
        resume: bool = False,
        use_date_search: bool = False,
    ) -> dict:
        """
        Sync tweets for a single account.

        Uses smart incremental fetching: stops API pagination immediately
        when reaching tweets we already have, minimizing API costs.

        Args:
            account: Twitter username
            full_sync: If True, ignore cursor and fetch all available tweets
            max_tweets: Maximum tweets to fetch (None = no limit)
            months_back: Fetch tweets going back N months from now
            show_progress: Show real-time progress updates
            resume: Resume from last backfill cursor
            use_date_search: Use month-by-month date-based search (better for historical)

        Returns:
            Dict with sync results including API calls made
        """
        # Use date-based search for historical backfill if requested
        if use_date_search and months_back:
            return self.sync_account_by_month(
                account=account,
                months_back=months_back,
                show_progress=show_progress,
            )

        logger.info(f"Starting sync for @{account}")

        # Get current sync state
        state = self.database.get_sync_state(account)

        # Calculate target date for historical backfill
        until_date = None
        if months_back:
            until_date = datetime.now(pytz.UTC) - timedelta(days=months_back * 30)
            logger.info(f"Historical backfill: fetching back to {until_date.date()}")

        # Determine exists_check behavior
        if full_sync:
            logger.info("Full sync (will re-fetch all tweets)")
            # For full sync, don't check existing tweets at all
            exists_check: Callable[[str], bool] = lambda tweet_id: False
        elif months_back:
            logger.info("Historical backfill (will skip existing tweets, fill gaps)")
            # For backfill: check database to skip existing, but continue pagination
            exists_check = self.database.tweet_exists
        else:
            if state and state.last_tweet_id:
                logger.info(f"Incremental sync (last tweet: {state.last_tweet_id})")
            else:
                logger.info("First sync for this account")
            # Check database for existing tweets
            exists_check = self.database.tweet_exists

        # Resume support
        start_cursor = None
        if resume and state and state.backfill_cursor:
            start_cursor = state.backfill_cursor
            logger.info(f"Resuming from cursor: {start_cursor[:20]}...")

        # Track fetch progress
        raw_count = 0
        processed_count = 0
        api_calls = 0
        new_tweets: list[ProcessedTweet] = []
        first_tweet_id = None
        last_cursor = None
        skipped_existing = 0

        # IMMEDIATE SAVE callback - saves raw tweet right away (crash-safe)
        def save_callback(tweet: Tweet) -> bool:
            nonlocal raw_count, first_tweet_id, new_tweets

            # Track first (most recent) tweet ID
            if first_tweet_id is None:
                first_tweet_id = tweet.id

            # Save raw tweet IMMEDIATELY (INSERT OR IGNORE for duplicates)
            saved = self.database.insert_raw_tweet(tweet.id, account, tweet.raw_json)
            if saved:
                raw_count += 1

            # Process and save processed tweets immediately too
            processed = self._process_tweet(tweet)
            if processed:
                self.database.insert_processed_tweets(processed)
                new_tweets.extend(processed)

            return saved

        # Progress callback
        progress_callback = None
        if show_progress:
            target_date_str = until_date.strftime("%Y-%m-%d") if until_date else "latest"

            def progress_callback(progress: dict) -> None:
                nonlocal skipped_existing
                skipped_existing = progress.get("skipped_existing", 0)
                skip_info = f" | Skipped: {skipped_existing}" if skipped_existing > 0 else ""
                sys.stdout.write(
                    f"\r  Progress: {progress['tweets_fetched']:,} tweets | "
                    f"{progress['api_calls']} API calls | "
                    f"Currently at: {progress['current_date']}{skip_info}    "
                )
                sys.stdout.flush()

        try:
            # Use cost-optimized incremental fetch with immediate save
            fetched_tweets, api_calls, last_cursor = self.client.fetch_tweets_incremental(
                username=account,
                exists_check=exists_check,
                max_tweets=max_tweets,
                until_date=until_date,
                include_replies=False,
                progress_callback=progress_callback,
                start_cursor=start_cursor,
                save_callback=save_callback,  # IMMEDIATE SAVE
            )

            if show_progress:
                sys.stdout.write("\n")  # New line after progress

        except Exception as e:
            logger.error(f"Error fetching tweets for @{account}: {e}")
            if show_progress:
                sys.stdout.write("\n")  # New line after progress
            # Save cursor for resume if we have one
            if last_cursor and months_back:
                self.database.update_backfill_state(
                    account=account,
                    cursor=last_cursor,
                    target_date=until_date.isoformat() if until_date else None,
                    complete=False,
                )
                logger.info(f"Saved cursor for resume (saved {raw_count} tweets before error)")
            raise

        # Processed count is number of processed tweet rows created
        processed_count = len(new_tweets)

        # Update sync state
        if first_tweet_id:
            self.database.update_sync_state(
                account=account,
                last_tweet_id=first_tweet_id,
                tweets_added=processed_count,
            )

        # Update backfill state
        if months_back:
            # Mark complete if we reached the target date (no more cursor)
            complete = last_cursor is None
            self.database.update_backfill_state(
                account=account,
                cursor=last_cursor,
                target_date=until_date.isoformat() if until_date else None,
                complete=complete,
            )

        result = {
            "account": account,
            "raw_tweets_fetched": raw_count,
            "processed_tweets_added": processed_count,
            "api_calls": api_calls,
            "last_tweet_id": first_tweet_id,
            "skipped_existing": skipped_existing,
        }

        skip_msg = f", {skipped_existing} skipped" if skipped_existing > 0 else ""
        logger.info(f"Sync complete for @{account}: " f"{raw_count} raw, {processed_count} processed{skip_msg} ({api_calls} API calls)")

        return result

    def sync_all(
        self,
        full_sync: bool = False,
        max_tweets_per_account: Optional[int] = None,
        months_back: Optional[int] = None,
        show_progress: bool = False,
        resume: bool = False,
        use_date_search: bool = False,
    ) -> list[dict]:
        """
        Sync all configured accounts.

        Args:
            full_sync: If True, ignore cursors and fetch all available tweets
            max_tweets_per_account: Maximum tweets to fetch per account
            months_back: Fetch tweets going back N months
            show_progress: Show real-time progress updates
            resume: Resume from last backfill cursor
            use_date_search: Use month-by-month date-based search (better for historical)

        Returns:
            List of sync results per account
        """
        logger.info(f"Starting sync for {len(self.accounts)} accounts")

        if months_back:
            target_date = datetime.now(pytz.UTC) - timedelta(days=months_back * 30)
            logger.info(f"Historical backfill to {target_date.date()} ({months_back} months)")

        results = []

        for i, account in enumerate(self.accounts, 1):
            if show_progress:
                print(f"\n[{i}/{len(self.accounts)}] Fetching @{account}...")

            try:
                result = self.sync_account(
                    account=account,
                    full_sync=full_sync,
                    max_tweets=max_tweets_per_account,
                    months_back=months_back,
                    show_progress=show_progress,
                    resume=resume,
                    use_date_search=use_date_search,
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to sync @{account}: {e}")
                results.append(
                    {
                        "account": account,
                        "error": str(e),
                    }
                )

        # Summary
        total_raw = sum(r.get("raw_tweets_fetched", 0) for r in results)
        total_processed = sum(r.get("processed_tweets_added", 0) for r in results)
        total_api_calls = sum(r.get("api_calls", 0) for r in results)
        logger.info(f"Sync complete: {total_raw} raw tweets, {total_processed} processed " f"({total_api_calls} API calls)")

        return results

    def estimate_backfill(self, months_back: int = 6) -> dict:
        """
        Estimate time and API calls for historical backfill.

        Args:
            months_back: Number of months to fetch

        Returns:
            Dict with estimates
        """
        # Assumptions based on typical finance account activity
        tweets_per_day = 30  # Average across accounts
        days = months_back * 30
        total_tweets = tweets_per_day * days * len(self.accounts)

        # API pagination
        tweets_per_request = 20
        api_calls = total_tweets // tweets_per_request

        # Time estimate (with rate limiting)
        seconds = api_calls * TWITTER_RATE_LIMIT_DELAY
        hours = seconds / 3600

        return {
            "accounts": len(self.accounts),
            "months": months_back,
            "estimated_tweets": total_tweets,
            "estimated_api_calls": api_calls,
            "estimated_time_seconds": seconds,
            "estimated_time_hours": round(hours, 2),
            "rate_limit_delay": TWITTER_RATE_LIMIT_DELAY,
        }

    def get_status(self) -> dict:
        """
        Get sync status for all accounts.

        Returns:
            Dict with status information
        """
        states = self.database.get_all_sync_states()
        stats = self.database.get_stats()

        # Get ACTUAL raw tweet counts from database (not from sync_state)
        raw_counts = self.database.get_raw_tweet_counts_by_account()
        journal_days = self.database.get_journal_days_by_account()

        # Format sync states - use actual counts from database
        accounts_status = []
        synced_accounts = {s.account for s in states}

        for account in self.accounts:
            state = next((s for s in states if s.account == account), None)
            accounts_status.append(
                {
                    "account": account,
                    "last_sync": state.last_sync_at.isoformat() if state and state.last_sync_at else None,
                    "raw_tweets": raw_counts.get(account, 0),  # From actual database
                    "days_fetched": journal_days.get(account, 0),  # From journal
                }
            )

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
                # Apply FinBERT-optimized cleaning (Unicode normalization, emoji mapping)
                cleaned_text = clean_for_finbert(tweet.text)
                writer.writerow(
                    [
                        tweet.timestamp_et,
                        tweet.author,
                        tweet.ticker,
                        tweet.tweet_url,
                        tweet.category,
                        cleaned_text,
                    ]
                )

        logger.info(f"Exported {len(tweets)} tweets to {output_path}")
        return len(tweets)

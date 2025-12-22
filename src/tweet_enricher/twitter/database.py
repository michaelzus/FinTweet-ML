"""SQLite database manager for tweet storage."""

import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from tweet_enricher.config import TWITTER_DB_PATH
from tweet_enricher.utils.timezone import ET

logger = logging.getLogger(__name__)


@dataclass
class SyncState:
    """Sync state for a Twitter account."""

    account: str
    last_tweet_id: Optional[str]
    last_cursor: Optional[str]
    last_sync_at: Optional[datetime]
    total_tweets: int
    # Backfill state
    backfill_cursor: Optional[str] = None
    backfill_target_date: Optional[str] = None
    backfill_complete: bool = False


@dataclass
class ProcessedTweet:
    """A processed tweet ready for export."""

    id: str
    timestamp_utc: str
    timestamp_et: str
    author: str
    ticker: str
    tweet_url: str
    category: Optional[str]
    text: str


class TweetDatabase:
    """SQLite database manager for tweets."""

    SCHEMA = """
    -- Track sync progress per account
    CREATE TABLE IF NOT EXISTS sync_state (
        account TEXT PRIMARY KEY,
        last_tweet_id TEXT,
        last_cursor TEXT,
        last_sync_at TEXT,
        total_tweets INTEGER DEFAULT 0,
        backfill_cursor TEXT,
        backfill_target_date TEXT,
        backfill_complete INTEGER DEFAULT 0
    );

    -- Raw API responses (for debugging/replay)
    CREATE TABLE IF NOT EXISTS tweets_raw (
        id TEXT PRIMARY KEY,
        account TEXT NOT NULL,
        json_data TEXT NOT NULL,
        fetched_at TEXT NOT NULL
    );

    -- Processed tweets (ready for export)
    CREATE TABLE IF NOT EXISTS tweets_processed (
        id TEXT NOT NULL,
        timestamp_utc TEXT NOT NULL,
        timestamp_et TEXT NOT NULL,
        author TEXT NOT NULL,
        ticker TEXT NOT NULL,
        tweet_url TEXT NOT NULL,
        category TEXT,
        text TEXT NOT NULL,
        PRIMARY KEY (id, ticker)
    );

    -- Index for efficient date-based queries
    CREATE INDEX IF NOT EXISTS idx_tweets_timestamp ON tweets_processed(timestamp_et);
    CREATE INDEX IF NOT EXISTS idx_tweets_author ON tweets_processed(author);
    CREATE INDEX IF NOT EXISTS idx_tweets_ticker ON tweets_processed(ticker);

    -- Fetch journal: tracks which (account, date) pairs have been fetched
    CREATE TABLE IF NOT EXISTS fetch_journal (
        account TEXT NOT NULL,
        fetch_date TEXT NOT NULL,      -- YYYY-MM-DD
        fetched_at TEXT NOT NULL,      -- ISO timestamp when fetched
        tweets_count INTEGER DEFAULT 0,
        api_calls INTEGER DEFAULT 0,
        PRIMARY KEY (account, fetch_date)
    );
    CREATE INDEX IF NOT EXISTS idx_journal_account ON fetch_journal(account);
    """

    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize the database connection.

        Args:
            db_path: Path to SQLite database file (defaults to config value)
        """
        self.db_path = db_path or TWITTER_DB_PATH
        self._ensure_directory()
        self._init_schema()

    def _ensure_directory(self) -> None:
        """Ensure the database directory exists."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        """Initialize the database schema."""
        with self._get_connection() as conn:
            conn.executescript(self.SCHEMA)
            conn.commit()

        # Run migrations for existing databases
        self._migrate_schema()
        logger.debug(f"Database initialized at {self.db_path}")

    def _migrate_schema(self) -> None:
        """Add new columns to existing databases."""
        with self._get_connection() as conn:
            # Check if backfill columns exist
            cursor = conn.execute("PRAGMA table_info(sync_state)")
            columns = {row["name"] for row in cursor.fetchall()}

            # Add backfill columns if missing
            if "backfill_cursor" not in columns:
                conn.execute("ALTER TABLE sync_state ADD COLUMN backfill_cursor TEXT")
            if "backfill_target_date" not in columns:
                conn.execute("ALTER TABLE sync_state ADD COLUMN backfill_target_date TEXT")
            if "backfill_complete" not in columns:
                conn.execute("ALTER TABLE sync_state ADD COLUMN backfill_complete INTEGER DEFAULT 0")

            conn.commit()

    # =========================================================================
    # Sync State Operations
    # =========================================================================

    def get_sync_state(self, account: str) -> Optional[SyncState]:
        """
        Get sync state for an account.

        Args:
            account: Twitter username

        Returns:
            SyncState if exists, None otherwise
        """
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM sync_state WHERE account = ?", (account,)).fetchone()

            if row:
                last_sync_at = None
                if row["last_sync_at"]:
                    last_sync_at = datetime.fromisoformat(row["last_sync_at"])

                return SyncState(
                    account=row["account"],
                    last_tweet_id=row["last_tweet_id"],
                    last_cursor=row["last_cursor"],
                    last_sync_at=last_sync_at,
                    total_tweets=row["total_tweets"] or 0,
                    backfill_cursor=row["backfill_cursor"] if "backfill_cursor" in row.keys() else None,
                    backfill_target_date=row["backfill_target_date"] if "backfill_target_date" in row.keys() else None,
                    backfill_complete=bool(row["backfill_complete"]) if "backfill_complete" in row.keys() else False,
                )
            return None

    def update_sync_state(
        self,
        account: str,
        last_tweet_id: Optional[str] = None,
        last_cursor: Optional[str] = None,
        tweets_added: int = 0,
    ) -> None:
        """
        Update sync state for an account.

        Args:
            account: Twitter username
            last_tweet_id: ID of the most recent tweet
            last_cursor: Pagination cursor for next sync
            tweets_added: Number of tweets added in this sync
        """
        now = datetime.now(timezone.utc).isoformat()

        with self._get_connection() as conn:
            # Get current state
            current = self.get_sync_state(account)
            current_total = current.total_tweets if current else 0

            conn.execute(
                """
                INSERT INTO sync_state (account, last_tweet_id, last_cursor, last_sync_at, total_tweets)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(account) DO UPDATE SET
                    last_tweet_id = COALESCE(excluded.last_tweet_id, last_tweet_id),
                    last_cursor = excluded.last_cursor,
                    last_sync_at = excluded.last_sync_at,
                    total_tweets = total_tweets + ?
                """,
                (account, last_tweet_id, last_cursor, now, current_total + tweets_added, tweets_added),
            )
            conn.commit()

    def get_all_sync_states(self) -> list[SyncState]:
        """Get sync states for all accounts."""
        with self._get_connection() as conn:
            rows = conn.execute("SELECT * FROM sync_state ORDER BY account").fetchall()

            states = []
            for row in rows:
                last_sync_at = None
                if row["last_sync_at"]:
                    last_sync_at = datetime.fromisoformat(row["last_sync_at"])

                states.append(
                    SyncState(
                        account=row["account"],
                        last_tweet_id=row["last_tweet_id"],
                        last_cursor=row["last_cursor"],
                        last_sync_at=last_sync_at,
                        total_tweets=row["total_tweets"] or 0,
                        backfill_cursor=row["backfill_cursor"] if "backfill_cursor" in row.keys() else None,
                        backfill_target_date=row["backfill_target_date"] if "backfill_target_date" in row.keys() else None,
                        backfill_complete=bool(row["backfill_complete"]) if "backfill_complete" in row.keys() else False,
                    )
                )
            return states

    def update_backfill_state(
        self,
        account: str,
        cursor: Optional[str] = None,
        target_date: Optional[str] = None,
        complete: bool = False,
    ) -> None:
        """
        Update backfill state for an account.

        Args:
            account: Twitter username
            cursor: Pagination cursor for resuming
            target_date: Target date for backfill (ISO format)
            complete: Whether backfill is complete
        """
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO sync_state (account, backfill_cursor, backfill_target_date, backfill_complete)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(account) DO UPDATE SET
                    backfill_cursor = excluded.backfill_cursor,
                    backfill_target_date = COALESCE(excluded.backfill_target_date, backfill_target_date),
                    backfill_complete = excluded.backfill_complete
                """,
                (account, cursor, target_date, 1 if complete else 0),
            )
            conn.commit()

    # =========================================================================
    # Raw Tweet Operations
    # =========================================================================

    def insert_raw_tweet(self, tweet_id: str, account: str, json_data: dict) -> bool:
        """
        Insert a raw tweet into the database.

        Args:
            tweet_id: Tweet ID
            account: Twitter username
            json_data: Raw API response data

        Returns:
            True if inserted, False if already exists
        """
        with self._get_connection() as conn:
            try:
                conn.execute(
                    """
                    INSERT INTO tweets_raw (id, account, json_data, fetched_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        tweet_id,
                        account,
                        json.dumps(json_data),
                        datetime.now(timezone.utc).isoformat(),
                    ),
                )
                conn.commit()
                return True
            except sqlite3.IntegrityError:
                # Already exists
                return False

    def tweet_exists(self, tweet_id: str) -> bool:
        """Check if a tweet already exists in the database."""
        with self._get_connection() as conn:
            row = conn.execute("SELECT 1 FROM tweets_raw WHERE id = ?", (tweet_id,)).fetchone()
            return row is not None

    # =========================================================================
    # Processed Tweet Operations
    # =========================================================================

    def insert_processed_tweet(self, tweet: ProcessedTweet) -> bool:
        """
        Insert a processed tweet.

        Args:
            tweet: ProcessedTweet object

        Returns:
            True if inserted, False if already exists
        """
        with self._get_connection() as conn:
            try:
                conn.execute(
                    """
                    INSERT INTO tweets_processed 
                    (id, timestamp_utc, timestamp_et, author, ticker, tweet_url, category, text)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        tweet.id,
                        tweet.timestamp_utc,
                        tweet.timestamp_et,
                        tweet.author,
                        tweet.ticker,
                        tweet.tweet_url,
                        tweet.category,
                        tweet.text,
                    ),
                )
                conn.commit()
                return True
            except sqlite3.IntegrityError:
                return False

    def insert_processed_tweets(self, tweets: list[ProcessedTweet]) -> int:
        """
        Batch insert processed tweets.

        Args:
            tweets: List of ProcessedTweet objects

        Returns:
            Number of tweets inserted
        """
        inserted = 0
        with self._get_connection() as conn:
            for tweet in tweets:
                try:
                    conn.execute(
                        """
                        INSERT INTO tweets_processed 
                        (id, timestamp_utc, timestamp_et, author, ticker, tweet_url, category, text)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            tweet.id,
                            tweet.timestamp_utc,
                            tweet.timestamp_et,
                            tweet.author,
                            tweet.ticker,
                            tweet.tweet_url,
                            tweet.category,
                            tweet.text,
                        ),
                    )
                    inserted += 1
                except sqlite3.IntegrityError:
                    pass  # Skip duplicates
            conn.commit()
        return inserted

    def _format_timestamp_for_query(self, dt: datetime) -> str:
        """
        Format datetime for SQLite query to match stored format.

        Timestamps are stored WITH timezone offset (e.g., "2025-01-15 10:00:00-0500").
        Query parameters must use the same format for correct string comparison.

        Args:
            dt: Datetime to format (should be timezone-aware in ET)

        Returns:
            Formatted timestamp string with timezone offset
        """
        # Ensure timezone-aware (assume ET if naive)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=ET)
        # Convert to ET and format with offset
        dt_et = dt.astimezone(ET)
        return dt_et.strftime("%Y-%m-%d %H:%M:%S%z")

    def _parse_timestamp_et(self, ts_str: str) -> datetime:
        """
        Parse timestamp string from database, handling both formats.

        Supports both:
        - Old format without offset: "2025-01-15 10:00:00"
        - New format with offset: "2025-01-15 10:00:00-0500"

        Args:
            ts_str: Timestamp string from database

        Returns:
            Timezone-aware datetime in ET
        """
        # Try format with timezone offset first (new format)
        try:
            dt = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S%z")
            return dt.astimezone(ET)
        except ValueError:
            pass

        # Try format without offset (old format, assume ET)
        try:
            dt = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
            return dt.replace(tzinfo=ET)
        except ValueError:
            pass

        # Try ISO format as fallback
        dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        if dt.tzinfo is not None:
            return dt.astimezone(ET)
        return dt.replace(tzinfo=ET)

    def get_processed_tweets(
        self,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        account: Optional[str] = None,
        ticker: Optional[str] = None,
    ) -> list[ProcessedTweet]:
        """
        Get processed tweets with optional filters.

        Args:
            since: Start date filter (Eastern Time)
            until: End date filter (Eastern Time)
            account: Filter by author username
            ticker: Filter by ticker symbol

        Returns:
            List of ProcessedTweet objects
        """
        query = "SELECT * FROM tweets_processed WHERE 1=1"
        params: list = []

        if since:
            query += " AND timestamp_et >= ?"
            params.append(self._format_timestamp_for_query(since))

        if until:
            query += " AND timestamp_et <= ?"
            params.append(self._format_timestamp_for_query(until))

        if account:
            query += " AND author = ?"
            params.append(account)

        if ticker:
            query += " AND ticker = ?"
            params.append(ticker)

        query += " ORDER BY timestamp_et DESC"

        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()

            return [
                ProcessedTweet(
                    id=row["id"],
                    timestamp_utc=row["timestamp_utc"],
                    timestamp_et=row["timestamp_et"],
                    author=row["author"],
                    ticker=row["ticker"],
                    tweet_url=row["tweet_url"],
                    category=row["category"],
                    text=row["text"],
                )
                for row in rows
            ]

    def get_tickers_with_first_date(self) -> dict[str, datetime]:
        """
        Get unique tickers with their first appearance date.

        Returns:
            Dictionary mapping ticker symbol to first appearance datetime (ET, timezone-aware)
        """
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT ticker, MIN(timestamp_et) as first_date 
                FROM tweets_processed 
                GROUP BY ticker 
                ORDER BY ticker
                """
            ).fetchall()

            result = {}
            for row in rows:
                ticker = row["ticker"]
                first_date_str = row["first_date"]
                if first_date_str:
                    result[ticker] = self._parse_timestamp_et(first_date_str)
            return result

    def get_tickers_with_date_range(self) -> dict[str, tuple[datetime, datetime]]:
        """
        Get unique tickers with their first and last appearance dates.

        Returns:
            Dictionary mapping ticker symbol to (first_date, last_date) tuple (ET, timezone-aware)
        """
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT ticker, MIN(timestamp_et) as first_date, MAX(timestamp_et) as last_date
                FROM tweets_processed 
                GROUP BY ticker 
                ORDER BY ticker
                """
            ).fetchall()

            result: dict[str, tuple[datetime, datetime]] = {}
            for row in rows:
                ticker = row["ticker"]
                first_date_str = row["first_date"]
                last_date_str = row["last_date"]

                if first_date_str and last_date_str:
                    first_dt = self._parse_timestamp_et(first_date_str)
                    last_dt = self._parse_timestamp_et(last_date_str)
                    result[ticker] = (first_dt, last_dt)
            return result

    def get_raw_tweet_counts_by_account(self) -> dict[str, int]:
        """Get raw tweet counts per account from the database."""
        with self._get_connection() as conn:
            rows = conn.execute("SELECT account, COUNT(*) as cnt FROM tweets_raw GROUP BY account").fetchall()
            return {row["account"]: row["cnt"] for row in rows}

    def get_journal_days_by_account(self) -> dict[str, int]:
        """Get number of days fetched per account from the journal."""
        with self._get_connection() as conn:
            rows = conn.execute("SELECT account, COUNT(*) as cnt FROM fetch_journal GROUP BY account").fetchall()
            return {row["account"]: row["cnt"] for row in rows}

    def get_stats(self) -> dict:
        """Get database statistics."""
        with self._get_connection() as conn:
            total = conn.execute("SELECT COUNT(*) FROM tweets_processed").fetchone()[0]
            unique_tweets = conn.execute("SELECT COUNT(DISTINCT id) FROM tweets_processed").fetchone()[0]
            unique_tickers = conn.execute("SELECT COUNT(DISTINCT ticker) FROM tweets_processed").fetchone()[0]
            unique_authors = conn.execute("SELECT COUNT(DISTINCT author) FROM tweets_processed").fetchone()[0]

            # Get date range
            date_range = conn.execute(
                "SELECT MIN(timestamp_et) as min_date, MAX(timestamp_et) as max_date FROM tweets_processed"
            ).fetchone()

            return {
                "total_rows": total,
                "unique_tweets": unique_tweets,
                "unique_tickers": unique_tickers,
                "unique_authors": unique_authors,
                "date_range": (
                    {
                        "min": date_range["min_date"],
                        "max": date_range["max_date"],
                    }
                    if date_range["min_date"]
                    else None
                ),
            }

    # =========================================================================
    # Fetch Journal Operations
    # =========================================================================

    def is_day_fetched(self, account: str, date: str) -> bool:
        """
        Check if a specific day has already been fetched for an account.

        Args:
            account: Twitter username
            date: Date string in YYYY-MM-DD format

        Returns:
            True if day was already fetched, False otherwise
        """
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT 1 FROM fetch_journal WHERE account = ? AND fetch_date = ?",
                (account, date),
            ).fetchone()
            return row is not None

    def mark_day_fetched(
        self,
        account: str,
        date: str,
        tweets_count: int = 0,
        api_calls: int = 0,
    ) -> None:
        """
        Mark a day as successfully fetched for an account.

        Args:
            account: Twitter username
            date: Date string in YYYY-MM-DD format
            tweets_count: Number of tweets fetched for this day
            api_calls: Number of API calls made for this day
        """
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO fetch_journal (account, fetch_date, fetched_at, tweets_count, api_calls)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(account, fetch_date) DO UPDATE SET
                    fetched_at = excluded.fetched_at,
                    tweets_count = excluded.tweets_count,
                    api_calls = excluded.api_calls
                """,
                (
                    account,
                    date,
                    datetime.now(timezone.utc).isoformat(),
                    tweets_count,
                    api_calls,
                ),
            )
            conn.commit()

    def get_fetched_days(self, account: str) -> list[str]:
        """
        Get all dates that have been fetched for an account.

        Args:
            account: Twitter username

        Returns:
            List of date strings (YYYY-MM-DD format)
        """
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT fetch_date FROM fetch_journal WHERE account = ? ORDER BY fetch_date",
                (account,),
            ).fetchall()
            return [row["fetch_date"] for row in rows]

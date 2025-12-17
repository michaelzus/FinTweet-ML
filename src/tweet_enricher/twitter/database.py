"""SQLite database manager for tweet storage."""

import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import pytz

from tweet_enricher.config import TWITTER_DB_PATH

logger = logging.getLogger(__name__)


@dataclass
class SyncState:
    """Sync state for a Twitter account."""

    account: str
    last_tweet_id: Optional[str]
    last_cursor: Optional[str]
    last_sync_at: Optional[datetime]
    total_tweets: int


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
        total_tweets INTEGER DEFAULT 0
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
        logger.debug(f"Database initialized at {self.db_path}")

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
            row = conn.execute(
                "SELECT * FROM sync_state WHERE account = ?", (account,)
            ).fetchone()

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
        now = datetime.now(pytz.UTC).isoformat()

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
                    )
                )
            return states

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
                        datetime.now(pytz.UTC).isoformat(),
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
            row = conn.execute(
                "SELECT 1 FROM tweets_raw WHERE id = ?", (tweet_id,)
            ).fetchone()
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
            params.append(since.strftime("%Y-%m-%d %H:%M:%S"))

        if until:
            query += " AND timestamp_et <= ?"
            params.append(until.strftime("%Y-%m-%d %H:%M:%S"))

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

    def get_tweet_count(self) -> int:
        """Get total number of processed tweets."""
        with self._get_connection() as conn:
            row = conn.execute("SELECT COUNT(*) as cnt FROM tweets_processed").fetchone()
            return row["cnt"] if row else 0

    def get_unique_tickers(self) -> list[str]:
        """Get list of unique tickers in the database."""
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT DISTINCT ticker FROM tweets_processed ORDER BY ticker"
            ).fetchall()
            return [row["ticker"] for row in rows]

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
                "date_range": {
                    "min": date_range["min_date"],
                    "max": date_range["max_date"],
                } if date_range["min_date"] else None,
            }


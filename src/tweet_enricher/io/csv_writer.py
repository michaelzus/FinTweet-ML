"""CSV writing utilities for tweet data."""

import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

import pytz


class CSVWriter:
    """Writes messages to CSV with optional filtering and deduplication."""

    FIELDNAMES = ["timestamp", "author", "ticker", "tweet_url", "category", "text"]

    def __init__(self, deduplicate: bool = True, ticker_filter: Optional[Set[str]] = None):
        """
        Initialize CSV writer.

        Args:
            deduplicate: Remove duplicate messages
            ticker_filter: Optional set of allowed tickers
        """
        self.deduplicate = deduplicate
        self.ticker_filter = ticker_filter

    def write(self, messages: List[Dict[str, str]], output_path: Path) -> Dict[str, int]:
        """
        Write messages to CSV file.

        Args:
            messages: List of message dictionaries
            output_path: Output CSV file path

        Returns:
            Statistics dictionary
        """
        stats = {
            "total_input": len(messages),
            "filtered": 0,
            "duplicates": 0,
            "written": 0,
        }

        seen_messages: Set[tuple] = set()

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
            writer.writeheader()

            for msg in messages:
                # Apply ticker filter
                if self.ticker_filter and msg["ticker"].upper() not in self.ticker_filter:
                    stats["filtered"] += 1
                    continue

                # Check for duplicates
                if self.deduplicate:
                    unique_key = (msg["timestamp"], msg["ticker"].upper(), msg["text"])
                    if unique_key in seen_messages:
                        stats["duplicates"] += 1
                        continue
                    seen_messages.add(unique_key)

                # Write row
                writer.writerow(
                    {
                        "timestamp": self._convert_timestamp(msg["timestamp"]),
                        "author": msg["author"],
                        "ticker": msg["ticker"],
                        "tweet_url": msg["tweet_url"],
                        "category": msg["category"],
                        "text": msg["text"],
                    }
                )
                stats["written"] += 1

        return stats

    def _convert_timestamp(self, timestamp_str: str) -> str:
        """
        Convert timestamp from Jerusalem time to US Eastern time.

        Handles daylight saving time (DST) transitions correctly for both timezones.

        Args:
            timestamp_str: Timestamp in format 'DD/MM/YYYY HH:MM' (Jerusalem time)

        Returns:
            Timestamp in ISO format 'YYYY-MM-DD HH:MM:SS' (US Eastern time)
        """
        try:
            # Parse timestamp as naive datetime
            dt_naive = datetime.strptime(timestamp_str, "%d/%m/%Y %H:%M")

            # Define timezones
            jerusalem_tz = pytz.timezone("Asia/Jerusalem")
            eastern_tz = pytz.timezone("America/New_York")

            # Localize to Jerusalem time (this handles IST/IDT automatically)
            dt_jerusalem = jerusalem_tz.localize(dt_naive)

            # Convert to US Eastern time (handles EST/EDT automatically)
            dt_eastern = dt_jerusalem.astimezone(eastern_tz)

            # Return in ISO format
            return dt_eastern.strftime("%Y-%m-%d %H:%M:%S")

        except ValueError:
            return timestamp_str


class TickerFilter:
    """Loads and manages ticker filter lists."""

    @staticmethod
    def load_from_csv(file_path: Path) -> Set[str]:
        """
        Load ticker symbols from CSV file.

        Args:
            file_path: Path to CSV file with 'symbol' column

        Returns:
            Set of ticker symbols (uppercase)
        """
        tickers: Set[str] = set()
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ticker = row.get("symbol", "").strip().upper()
                if ticker:
                    tickers.add(ticker)
        return tickers


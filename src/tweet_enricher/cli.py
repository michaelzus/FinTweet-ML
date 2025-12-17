"""Unified CLI for tweet_enricher package.

Provides subcommands for all tweet enrichment operations.
"""

import argparse
import asyncio
import logging
import sys
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pandas_market_calendars as mcal

from tweet_enricher.config import (
    DAILY_DATA_DIR,
    ET,
    EXCLUDED_TICKERS,
    MARKET_CLOSE,
    TWITTER_ACCOUNTS,
)
from tweet_enricher.core.enricher import TweetEnricher
from tweet_enricher.core.indicators import TechnicalIndicators
from tweet_enricher.data.cache import DataCache
from tweet_enricher.data.ib_fetcher import IBHistoricalDataFetcher
from tweet_enricher.data.stock_metadata import StockMetadataCache
from tweet_enricher.data.tickers import (
    fetch_russell1000_tickers,
    fetch_sp500_tickers,
    filter_tickers_by_volume,
)
from tweet_enricher.io.feather import save_daily_data
from tweet_enricher.parsers.discord import DiscordToCSVConverter
from tweet_enricher.twitter.client import TwitterClient
from tweet_enricher.twitter.database import TweetDatabase
from tweet_enricher.twitter.sync import SyncService


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the CLI."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# ============================================================================
# Subcommand: convert
# ============================================================================
def cmd_convert(args: argparse.Namespace) -> int:
    """Convert Discord export to CSV."""
    setup_logging(args.verbose)

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        return 1

    filter_path = Path(args.filter) if args.filter else None
    if filter_path and not filter_path.exists():
        print(f"Error: Filter file not found: {filter_path}", file=sys.stderr)
        return 1

    try:
        converter = DiscordToCSVConverter(
            min_text_length=args.min_length,
            deduplicate=not args.no_dedup,
        )

        converter.convert(
            input_file=input_path,
            output_file=output_path,
            ticker_filter_file=filter_path,
            verbose=not args.quiet,
        )

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


# ============================================================================
# Subcommand: fetch
# ============================================================================
async def _fetch_data(args: argparse.Namespace) -> int:
    """Async implementation of fetch command."""
    logger = logging.getLogger(__name__)

    # Determine which symbols to fetch
    if args.all:
        logger.info("Fetching combined S&P 500 and Russell 1000 ticker lists...")
        try:
            sp500_symbols = fetch_sp500_tickers()
            logger.info(f"Found {len(sp500_symbols)} S&P 500 tickers")

            russell1000_symbols = fetch_russell1000_tickers()
            logger.info(f"Found {len(russell1000_symbols)} Russell 1000 tickers")

            symbols = list(set(sp500_symbols + russell1000_symbols))
            logger.info(f"Combined total: {len(symbols)} unique tickers")
        except Exception as e:
            logger.error(f"Error: {e}")
            return 1
    elif args.sp500:
        logger.info("Fetching S&P 500 ticker list...")
        try:
            symbols = fetch_sp500_tickers()
            logger.info(f"Found {len(symbols)} S&P 500 tickers")
        except Exception as e:
            logger.error(f"Error: {e}")
            return 1
    elif args.russell1000:
        logger.info("Fetching Russell 1000 ticker list...")
        try:
            symbols = fetch_russell1000_tickers()
            logger.info(f"Found {len(symbols)} Russell 1000 tickers")
        except Exception as e:
            logger.error(f"Error: {e}")
            return 1
    else:
        symbols = args.symbols

    fetcher = IBHistoricalDataFetcher(host=args.host, port=args.port, client_id=args.client_id)

    connected = await fetcher.connect()
    if not connected:
        return 1

    try:
        logger.info(f"Fetching historical data for {len(symbols)} symbols...")
        data_dict = await fetcher.fetch_multiple_stocks(
            symbols=symbols,
            exchange=args.exchange,
            currency=args.currency,
            duration=args.duration,
            bar_size=args.bar_size,
            batch_size=args.batch_size,
            delay_between_batches=args.batch_delay,
        )

        if data_dict:
            output_dir = Path(args.output_dir)
            for symbol, df in data_dict.items():
                save_daily_data(symbol, df, output_dir)

            logger.info("Summary:")
            total_records = sum(len(df) for df in data_dict.values())
            logger.info(f"Total records: {total_records}")
            logger.info(f"Files saved to: {args.output_dir}/")
        else:
            logger.warning("No data fetched")

    finally:
        await fetcher.disconnect()

    return 0


def cmd_fetch(args: argparse.Namespace) -> int:
    """Fetch historical data from Interactive Brokers."""
    setup_logging(args.verbose)
    return asyncio.run(_fetch_data(args))


# ============================================================================
# Subcommand: enrich
# ============================================================================
async def _enrich_data(args: argparse.Namespace) -> int:
    """Async implementation of enrich command."""
    logger = logging.getLogger(__name__)

    # Read tweets CSV
    logger.info(f"Reading tweets from {args.input}")

    try:
        tweets_df = pd.read_csv(args.input)
        logger.info(f"Loaded {len(tweets_df)} tweets")
    except Exception as e:
        logger.error(f"Error reading tweets file: {e}")
        return 1

    # Calculate max date from dataset
    tweets_df["timestamp"] = pd.to_datetime(tweets_df["timestamp"])
    max_date = tweets_df["timestamp"].max() + timedelta(days=2)

    # Ensure max_date is timezone-aware
    if max_date.tzinfo is None:
        max_date = ET.localize(max_date.to_pydatetime())
    elif max_date.tzinfo != ET:
        max_date = max_date.tz_convert(ET)

    # Get last market trading day
    nyse = mcal.get_calendar("NYSE")
    now_et = datetime.now(ET)
    today = now_et.date()
    schedule = nyse.schedule(start_date=today - timedelta(days=10), end_date=today)

    if not schedule.empty:
        last_market_day = schedule.index[-1].date()

        # If today is a trading day but market hasn't closed yet, use previous trading day
        if last_market_day == today:
            current_time_minutes = now_et.hour * 60 + now_et.minute
            if current_time_minutes < MARKET_CLOSE:
                logger.info(f"Market not closed yet (current time: {now_et.strftime('%H:%M')} ET)")
                if len(schedule) > 1:
                    last_market_day = schedule.index[-2].date()
                    logger.info(f"Using previous trading day: {last_market_day}")
                else:
                    extended_schedule = nyse.schedule(start_date=today - timedelta(days=30), end_date=today)
                    if len(extended_schedule) > 1:
                        last_market_day = extended_schedule.index[-2].date()
                        logger.info(f"Using previous trading day: {last_market_day}")

        last_market_datetime = ET.localize(datetime.combine(last_market_day, datetime.max.time()))
        max_date = min(max_date, last_market_datetime)
        logger.info(f"Last closed market trading day: {last_market_day}")
    else:
        max_date = min(max_date, datetime.now(ET))
        logger.warning("Could not determine last market day, using current date")

    min_date = tweets_df["timestamp"].min()
    logger.info(f"Dataset date range: {min_date.date()} to {max_date.date()}")

    # Extract unique tickers
    unique_tickers = tweets_df["ticker"].unique().tolist()
    unique_tickers.append("SPY")  # Add SPY for market adjustment
    unique_tickers = list(set(unique_tickers))

    # Filter out excluded tickers
    excluded_count = len([t for t in unique_tickers if t in EXCLUDED_TICKERS])
    if excluded_count > 0:
        logger.info(f"Excluding {excluded_count} problematic tickers")
    unique_tickers = [t for t in unique_tickers if t not in EXCLUDED_TICKERS]

    logger.info(f"Found {len(unique_tickers)} unique tickers in dataset")

    # Initialize components
    ib_fetcher = IBHistoricalDataFetcher(host=args.host, port=args.port, client_id=args.client_id)
    cache = DataCache(ib_fetcher)
    indicators = TechnicalIndicators()
    metadata_cache = StockMetadataCache()
    enricher = TweetEnricher(ib_fetcher, cache, indicators, metadata_cache)

    # Connect to IB (if connection fails, try to continue with cached data only)
    connected = await enricher.connect()
    if not connected:
        logger.warning("Failed to connect to IB - will attempt to use cached data only")
        logger.warning("If enrichment fails, please start TWS/Gateway and try again")

    try:
        # Pre-fetch all ticker data
        logger.info("=" * 80)
        logger.info("PHASE 1: Pre-fetching ALL ticker data (daily + intraday)")
        logger.info("=" * 80)

        logger.info("Step 1/2: Fetching daily data...")
        await cache.prefetch_all_daily_data(unique_tickers, max_date)

        logger.info("Step 2/2: Fetching intraday data...")
        await cache.prefetch_all_intraday_data(unique_tickers, max_date)

        logger.info(f"Phase 1 complete! Cached data for {len(unique_tickers)} tickers")
        logger.info("=" * 80)

        # Process all tweets
        logger.info("PHASE 2: Processing tweets")
        logger.info("=" * 80)

        output_df = await enricher.enrich_dataframe(tweets_df, max_date)

        # Save results
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(output_df)} enriched tweets to: {output_path}")

        # Summary
        logger.info("")
        logger.info("=" * 80)
        logger.info("SUMMARY")
        logger.info("=" * 80)
        successful = sum(1 for _, r in output_df.iterrows() if r["entry_price"] is not None)
        logger.info(f"Total processed: {len(output_df)}")
        logger.info(f"Successful: {successful}")
        logger.info(f"Failed: {len(output_df) - successful}")

        # Label distribution
        labels = output_df["label_5class"].dropna().tolist()
        if labels:
            label_counts = Counter(labels)
            logger.info("\nLabel distribution:")
            for label, count in sorted(label_counts.items()):
                logger.info(f"  {label}: {count}")

    finally:
        await enricher.disconnect()

    return 0


def cmd_enrich(args: argparse.Namespace) -> int:
    """Enrich tweets with market data."""
    setup_logging(args.verbose)
    return asyncio.run(_enrich_data(args))


# ============================================================================
# Subcommand: filter-volume
# ============================================================================
def cmd_filter_volume(args: argparse.Namespace) -> int:
    """Filter tickers by average daily volume."""
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    try:
        logger.info("=" * 80)
        logger.info(f"FILTERING TICKERS (min avg volume: {args.min_volume:,.0f})")
        logger.info("=" * 80)

        filtered_tickers = filter_tickers_by_volume(args.data_dir, args.min_volume)

        if filtered_tickers:
            logger.info(f"Found {len(filtered_tickers)} tickers meeting criteria")

            if args.output:
                df = pd.DataFrame({"symbol": filtered_tickers})
                df.to_csv(args.output, index=False)
                logger.info(f"Filtered tickers saved to: {args.output}")
        else:
            logger.warning(f"No tickers found with average volume >= {args.min_volume:,.0f}")

        return 0

    except FileNotFoundError as e:
        logger.error(f"Error: {e}")
        logger.error(f"Make sure the data directory '{args.data_dir}' exists and contains feather files.")
        return 1
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1


# ============================================================================
# Subcommand: twitter sync
# ============================================================================
def cmd_twitter_sync(args: argparse.Namespace) -> int:
    """Sync tweets from Twitter accounts."""
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    try:
        # Initialize sync service
        accounts = [args.account] if args.account else None
        sync_service = SyncService(accounts=accounts)

        # Handle estimate mode
        months = getattr(args, 'months', None)
        if getattr(args, 'estimate', False):
            estimate = sync_service.estimate_backfill(months_back=months or 6)
            print("\n" + "=" * 60)
            print("BACKFILL ESTIMATE")
            print("=" * 60)
            print(f"  Accounts:           {estimate['accounts']}")
            print(f"  Months back:        {estimate['months']}")
            print(f"  Estimated tweets:   ~{estimate['estimated_tweets']:,}")
            print(f"  Estimated API calls: ~{estimate['estimated_api_calls']:,}")
            print(f"  Estimated time:     ~{estimate['estimated_time_hours']:.1f} hours")
            print(f"  Rate limit delay:   {estimate['rate_limit_delay']}s per request")
            print("=" * 60)
            print("\nTo run the backfill:")
            print(f"  tweet-enricher twitter sync --months {estimate['months']}")
            print("\nFor long backfills, run in background:")
            print(f"  nohup tweet-enricher twitter sync --months {estimate['months']} -v > backfill.log 2>&1 &")
            return 0

        # Test connection first
        logger.info("Testing Twitter API connection...")
        if not sync_service.client.test_connection():
            logger.error("Failed to connect to Twitter API")
            logger.error("Make sure TWITTER_API_KEY environment variable is set")
            return 1
        logger.info("Connection successful!")

        # Check for resume mode
        resume = getattr(args, 'resume', False)
        show_progress = months is not None  # Show progress for historical backfill

        # Perform sync
        if args.account:
            logger.info(f"Syncing account: @{args.account}")
            if months:
                logger.info(f"Historical backfill: {months} months")
            result = sync_service.sync_account(
                account=args.account,
                full_sync=args.full,
                max_tweets=args.max_tweets,
                months_back=months,
                show_progress=show_progress,
                resume=resume,
                use_date_search=getattr(args, 'date_search', False),
            )
            results = [result]
        else:
            logger.info(f"Syncing all {len(TWITTER_ACCOUNTS)} configured accounts")
            if months:
                logger.info(f"Historical backfill: {months} months")
            results = sync_service.sync_all(
                full_sync=args.full,
                max_tweets_per_account=args.max_tweets,
                months_back=months,
                show_progress=show_progress,
                resume=resume,
                use_date_search=getattr(args, 'date_search', False),
            )

        # Print summary
        print("\n" + "=" * 60)
        print("SYNC SUMMARY")
        print("=" * 60)
        total_raw = 0
        total_processed = 0
        total_api_calls = 0
        for r in results:
            if "error" in r:
                print(f"  @{r['account']}: ERROR - {r['error']}")
            else:
                raw = r.get("raw_tweets_fetched", 0)
                processed = r.get("processed_tweets_added", 0)
                api_calls = r.get("api_calls", 0)
                total_raw += raw
                total_processed += processed
                total_api_calls += api_calls
                print(f"  @{r['account']}: {raw} raw, {processed} processed ({api_calls} API calls)")

        print("-" * 60)
        print(f"  TOTAL: {total_raw} raw, {total_processed} processed ({total_api_calls} API calls)")
        print("=" * 60)

        return 0

    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        return 1
    except Exception as e:
        logger.error(f"Sync failed: {e}")
        return 1


# ============================================================================
# Subcommand: twitter status
# ============================================================================
def cmd_twitter_status(args: argparse.Namespace) -> int:
    """Show Twitter sync status."""
    setup_logging(args.verbose)

    try:
        database = TweetDatabase()
        sync_service = SyncService(database=database, lazy_client=True)
        status = sync_service.get_status()

        print("\n" + "=" * 60)
        print("TWITTER SYNC STATUS")
        print("=" * 60)

        # Database stats
        db_stats = status["database"]
        print("\nDatabase Statistics:")
        print(f"  Total rows:      {db_stats['total_rows']:,}")
        print(f"  Unique tweets:   {db_stats['unique_tweets']:,}")
        print(f"  Unique tickers:  {db_stats['unique_tickers']:,}")
        print(f"  Unique authors:  {db_stats['unique_authors']:,}")

        if db_stats["date_range"]:
            print(f"  Date range:      {db_stats['date_range']['min']} to {db_stats['date_range']['max']}")

        # Account status
        print("\nAccount Status:")
        print("-" * 70)
        print(f"  {'Account':<20} {'Raw Tweets':>12} {'Days':>8} {'Last Sync':<20}")
        print("-" * 70)

        for acc in status["accounts"]:
            last_sync = acc["last_sync"][:19] if acc["last_sync"] else "Never"
            raw_tweets = acc["raw_tweets"]
            days = acc["days_fetched"]
            print(f"  @{acc['account']:<19} {raw_tweets:>12,} {days:>8} {last_sync:<20}")

        print("=" * 60)

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


# ============================================================================
# Subcommand: twitter export
# ============================================================================
def cmd_twitter_export(args: argparse.Namespace) -> int:
    """Export tweets to CSV."""
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    try:
        database = TweetDatabase()
        sync_service = SyncService(database=database, lazy_client=True)

        # Parse date filters
        since = None
        until = None

        if args.since:
            since = datetime.strptime(args.since, "%Y-%m-%d")
            since = ET.localize(since)
            logger.info(f"Filtering from: {args.since}")

        if args.until:
            until = datetime.strptime(args.until, "%Y-%m-%d")
            until = ET.localize(until.replace(hour=23, minute=59, second=59))
            logger.info(f"Filtering until: {args.until}")

        # Export
        count = sync_service.export_csv(
            output_path=args.output,
            since=since,
            until=until,
            account=args.account,
            ticker=args.ticker,
        )

        if count > 0:
            print(f"\nExported {count:,} tweets to {args.output}")
        else:
            print("\nNo tweets found matching criteria")

        return 0

    except Exception as e:
        logger.error(f"Export failed: {e}")
        return 1


# ============================================================================
# Main entry point
# ============================================================================
def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="tweet-enricher",
        description="Tweet Enrichment CLI - Convert, fetch, and enrich tweet data with market information",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ============ convert subcommand ============
    convert_parser = subparsers.add_parser(
        "convert",
        help="Convert Discord channel export to CSV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  tweet-enricher convert -i discord_data.txt -o output.csv
  tweet-enricher convert -i discord_data.txt -o output.csv -f tickers.csv
  tweet-enricher convert -i discord_data.txt -o output.csv --no-dedup
        """,
    )
    convert_parser.add_argument("-i", "--input", type=str, required=True, help="Input Discord export file")
    convert_parser.add_argument("-o", "--output", type=str, required=True, help="Output CSV file")
    convert_parser.add_argument("-f", "--filter", type=str, help="Ticker filter CSV file (with 'symbol' column)")
    convert_parser.add_argument("--no-dedup", action="store_true", help="Disable duplicate message removal")
    convert_parser.add_argument("--min-length", type=int, default=60, help="Minimum text length to keep (default: 60)")
    convert_parser.add_argument("-q", "--quiet", action="store_true", help="Suppress progress output")
    convert_parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    convert_parser.set_defaults(func=cmd_convert)

    # ============ fetch subcommand ============
    fetch_parser = subparsers.add_parser(
        "fetch",
        help="Fetch historical OHLCV data from Interactive Brokers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  tweet-enricher fetch --symbols AAPL MSFT GOOGL
  tweet-enricher fetch --sp500
  tweet-enricher fetch --russell1000
  tweet-enricher fetch --all --duration "1 Y"
        """,
    )
    fetch_parser.add_argument("--symbols", nargs="+", help="List of stock symbols")
    fetch_parser.add_argument("--sp500", action="store_true", help="Fetch data for all S&P 500 stocks")
    fetch_parser.add_argument("--russell1000", action="store_true", help="Fetch data for all Russell 1000 stocks")
    fetch_parser.add_argument("--all", action="store_true", help="Fetch data for both S&P 500 and Russell 1000")
    fetch_parser.add_argument("--duration", default="1 Y", help="Duration string (default: 1 Y)")
    fetch_parser.add_argument("--bar-size", default="1 day", help="Bar size (default: 1 day)")
    fetch_parser.add_argument("--output-dir", default=str(DAILY_DATA_DIR), help=f"Output directory (default: {DAILY_DATA_DIR})")
    fetch_parser.add_argument("--batch-size", type=int, default=200, help="Batch size (default: 200)")
    fetch_parser.add_argument("--batch-delay", type=float, default=2.0, help="Delay between batches (default: 2.0)")
    fetch_parser.add_argument("--host", default="127.0.0.1", help="TWS/Gateway host (default: 127.0.0.1)")
    fetch_parser.add_argument("--port", type=int, default=7497, help="TWS/Gateway port (default: 7497)")
    fetch_parser.add_argument("--client-id", type=int, default=1, help="Client ID (default: 1)")
    fetch_parser.add_argument("--exchange", default="SMART", help="Exchange (default: SMART)")
    fetch_parser.add_argument("--currency", default="USD", help="Currency (default: USD)")
    fetch_parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    fetch_parser.set_defaults(func=cmd_fetch)

    # ============ enrich subcommand ============
    enrich_parser = subparsers.add_parser(
        "enrich",
        help="Enrich tweets with market data and technical indicators",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  tweet-enricher enrich -i tweets.csv -o enriched.csv
  tweet-enricher enrich -i tweets.csv -o enriched.csv --port 4002
        """,
    )
    enrich_parser.add_argument("-i", "--input", type=str, required=True, help="Input tweets CSV file")
    enrich_parser.add_argument("-o", "--output", type=str, required=True, help="Output enriched CSV file")
    enrich_parser.add_argument("--host", default="127.0.0.1", help="TWS/Gateway host (default: 127.0.0.1)")
    enrich_parser.add_argument("--port", type=int, default=7497, help="TWS/Gateway port (default: 7497)")
    enrich_parser.add_argument("--client-id", type=int, default=1, help="Client ID (default: 1)")
    enrich_parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    enrich_parser.set_defaults(func=cmd_enrich)

    # ============ filter-volume subcommand ============
    filter_parser = subparsers.add_parser(
        "filter-volume",
        help="Filter tickers by average daily trading volume",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  tweet-enricher filter-volume --min-volume 1000000
  tweet-enricher filter-volume --min-volume 500000 --output filtered.csv
        """,
    )
    filter_parser.add_argument("--data-dir", default=str(DAILY_DATA_DIR), help=f"Data directory (default: {DAILY_DATA_DIR})")
    filter_parser.add_argument("--min-volume", type=float, default=1_000_000, help="Minimum avg volume (default: 1,000,000)")
    filter_parser.add_argument("--output", help="Output CSV file (optional)")
    filter_parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    filter_parser.set_defaults(func=cmd_filter_volume)

    # ============ twitter subcommand group ============
    twitter_parser = subparsers.add_parser(
        "twitter",
        help="Twitter API operations (sync, status, export)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    twitter_subparsers = twitter_parser.add_subparsers(dest="twitter_command", help="Twitter commands")

    # twitter sync
    twitter_sync_parser = twitter_subparsers.add_parser(
        "sync",
        help="Sync tweets from configured Twitter accounts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  tweet-enricher twitter sync                    # Sync all accounts (incremental)
  tweet-enricher twitter sync --account StockMKTNewz  # Sync specific account
  tweet-enricher twitter sync --months 6         # Historical backfill (6 months)
  tweet-enricher twitter sync --months 6 --estimate  # Estimate time/cost only
  tweet-enricher twitter sync --resume           # Resume interrupted backfill
  tweet-enricher twitter sync --full             # Full re-sync (ignore cursor)
  tweet-enricher twitter sync --max-tweets 100   # Limit tweets per account
        """,
    )
    twitter_sync_parser.add_argument("--account", help="Specific account to sync (default: all)")
    twitter_sync_parser.add_argument("--months", type=int, help="Historical backfill: fetch N months of tweets")
    twitter_sync_parser.add_argument("--estimate", action="store_true", help="Show time/cost estimate without fetching")
    twitter_sync_parser.add_argument("--resume", action="store_true", help="Resume interrupted backfill from last cursor")
    twitter_sync_parser.add_argument("--full", action="store_true", help="Full re-sync, ignore previous cursor")
    twitter_sync_parser.add_argument("--max-tweets", type=int, help="Max tweets to fetch per account")
    twitter_sync_parser.add_argument("--date-search", action="store_true", help="Use date-based search (better for 6+ month history)")
    twitter_sync_parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    twitter_sync_parser.set_defaults(func=cmd_twitter_sync)

    # twitter status
    twitter_status_parser = twitter_subparsers.add_parser(
        "status",
        help="Show sync status for all accounts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    twitter_status_parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    twitter_status_parser.set_defaults(func=cmd_twitter_status)

    # twitter export
    twitter_export_parser = twitter_subparsers.add_parser(
        "export",
        help="Export tweets to CSV file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  tweet-enricher twitter export -o tweets.csv
  tweet-enricher twitter export -o tweets.csv --since 2025-12-01
  tweet-enricher twitter export -o tweets.csv --account StockMKTNewz
  tweet-enricher twitter export -o tweets.csv --ticker NVDA
        """,
    )
    twitter_export_parser.add_argument("-o", "--output", required=True, help="Output CSV file")
    twitter_export_parser.add_argument("--since", help="Start date filter (YYYY-MM-DD)")
    twitter_export_parser.add_argument("--until", help="End date filter (YYYY-MM-DD)")
    twitter_export_parser.add_argument("--account", help="Filter by account")
    twitter_export_parser.add_argument("--ticker", help="Filter by ticker")
    twitter_export_parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    twitter_export_parser.set_defaults(func=cmd_twitter_export)

    # Parse arguments
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 0

    # Validate fetch arguments
    if args.command == "fetch":
        if not args.symbols and not args.sp500 and not args.russell1000 and not args.all:
            fetch_parser.error("Either --symbols, --sp500, --russell1000, or --all must be provided")
        if sum([bool(args.symbols), args.sp500, args.russell1000, args.all]) > 1:
            fetch_parser.error("Please provide only one of: --symbols, --sp500, --russell1000, or --all")

    # Handle twitter subcommand group
    if args.command == "twitter":
        if args.twitter_command is None:
            twitter_parser.print_help()
            return 0

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())


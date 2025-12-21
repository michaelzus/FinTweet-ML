"""Unified CLI for FinTweet-ML pipeline.

Provides subcommands for all pipeline operations:
- ohlcv: OHLCV data collection from IB API
- twitter: Tweet collection from Twitter API
- prepare: Dataset preparation (offline, no API calls)
- train: Model training
- evaluate: Model evaluation
- convert: Discord export conversion
"""

import argparse
import asyncio
import logging
import re
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
    INTRADAY_CACHE_DIR,
    INTRADAY_FETCH_DELAY,
    MARKET_CLOSE,
    TWITTER_ACCOUNTS,
    TWITTER_DB_PATH,
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
from tweet_enricher.parsers.discord import DiscordToCSVConverter
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


def normalize_duration(duration: str) -> str:
    """
    Normalize IB duration string to required format: integer{SPACE}unit.

    IB API requires duration format like "1 Y", "6 M", "30 D".
    This handles user input without space like "1Y" -> "1 Y".

    Args:
        duration: Duration string (e.g., "1Y", "1 Y", "6M", "30D")

    Returns:
        Normalized duration string with space between number and unit
    """
    duration = duration.strip()
    # If already has space in correct format, return as-is
    if re.match(r"^\d+\s+[SDWMY]$", duration, re.IGNORECASE):
        return duration.upper()
    # Handle no-space format like "1Y" or "30D"
    match = re.match(r"^(\d+)([SDWMY])$", duration, re.IGNORECASE)
    if match:
        return f"{match.group(1)} {match.group(2).upper()}"
    # Return original if format is unrecognized (let IB API report error)
    return duration


def duration_to_days(duration: str) -> int:
    """
    Convert IB duration string to approximate number of calendar days.

    Args:
        duration: Duration string (e.g., "1 Y", "6 M", "30 D")

    Returns:
        Approximate number of calendar days
    """
    duration = normalize_duration(duration)
    match = re.match(r"^(\d+)\s+([SDWMY])$", duration, re.IGNORECASE)
    if not match:
        return 200  # Default fallback

    value = int(match.group(1))
    unit = match.group(2).upper()

    # Convert to calendar days (approximate)
    if unit == "Y":
        return value * 365
    elif unit == "M":
        return value * 30
    elif unit == "W":
        return value * 7
    elif unit == "D":
        return value
    elif unit == "S":  # Seconds - unusual but handle it
        return max(1, value // 86400)
    return 200


# ============================================================================
# Subcommand: convert (Utility)
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
# FLOW 1: OHLCV Commands (IB API)
# ============================================================================
async def _ohlcv_sync(args: argparse.Namespace) -> int:
    """
    Sync OHLCV data using smart caching with incremental updates and backfill.

    This command:
    1. Fetches daily data for the full requested duration
    2. Fetches intraday data with smart caching (checks existing cache first)
    3. Automatically backfills intraday data to match daily duration
    """
    logger = logging.getLogger(__name__)

    # Determine which symbols to fetch
    if args.all:
        logger.info("Fetching combined S&P 500 and Russell 1000 ticker lists...")
        try:
            sp500_symbols = fetch_sp500_tickers()
            logger.info(f"Found {len(sp500_symbols)} S&P 500 tickers")

            russell1000_symbols = fetch_russell1000_tickers()
            logger.info(f"Found {len(russell1000_symbols)} Russell 1000 tickers")

            symbols = list(set(sp500_symbols + russell1000_symbols + ["SPY"]))
            logger.info(f"Combined total: {len(symbols)} unique tickers (including SPY)")
        except Exception as e:
            logger.error(f"Error: {e}")
            return 1
    elif args.sp500:
        logger.info("Fetching S&P 500 ticker list...")
        try:
            symbols = fetch_sp500_tickers()
            if "SPY" not in symbols:
                symbols.append("SPY")
            logger.info(f"Found {len(symbols)} S&P 500 tickers (including SPY)")
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
        symbols = args.tickers

    # Calculate target start date from duration
    duration = normalize_duration(args.duration)
    total_days = duration_to_days(duration)
    now = datetime.now(ET)
    target_start_date = (now - timedelta(days=total_days)).date()

    logger.info("")
    logger.info("=" * 80)
    logger.info(f"OHLCV SYNC: {len(symbols)} symbols, duration={duration} ({total_days} days)")
    logger.info(f"Target date range: {target_start_date} to {now.date()}")
    logger.info("=" * 80)

    # Initialize IB fetcher and DataCache
    fetcher = IBHistoricalDataFetcher(host=args.host, port=args.port, client_id=args.client_id)
    cache = DataCache(
        ib_fetcher=fetcher,
        daily_dir=Path(args.output_dir),
        intraday_dir=Path(args.intraday_dir),
    )

    connected = await fetcher.connect()
    if not connected:
        return 1

    try:
        # Step 1: Fetch daily data (smart caching with incremental updates)
        logger.info("")
        logger.info("=" * 80)
        logger.info(f"STEP 1: Syncing DAILY data ({duration})...")
        logger.info("=" * 80)

        await cache.prefetch_all_daily_data(
            symbols=symbols,
            max_date=now,
            duration=duration,
            batch_size=args.batch_size,
        )

        # Step 2: Fetch intraday data (unless --daily-only)
        if not args.daily_only:
            # Step 2a: Initial intraday fetch (up to 200 days with smart caching)
            logger.info("")
            logger.info("=" * 80)
            logger.info("STEP 2: Syncing INTRADAY data (initial fetch)...")
            logger.info("=" * 80)

            await cache.prefetch_all_intraday_data(
                symbols=symbols,
                max_date=now,
                total_days=min(total_days, 200),  # IB limit per request
                delay_between_symbols=INTRADAY_FETCH_DELAY,
            )

            # Step 2b: Backfill intraday to match daily duration (if needed)
            if total_days > 200:
                logger.info("")
                logger.info("=" * 80)
                logger.info(f"STEP 3: Backfilling INTRADAY data to {target_start_date}...")
                logger.info("=" * 80)

                await cache.backfill_intraday_data(
                    symbols=symbols,
                    target_start_date=target_start_date,
                    delay_between_symbols=INTRADAY_FETCH_DELAY,
                )

        # Final Summary
        logger.info("")
        logger.info("=" * 80)
        logger.info("SYNC COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Daily data:    {args.output_dir}/")
        if not args.daily_only:
            logger.info(f"Intraday data: {args.intraday_dir}/")

    finally:
        await fetcher.disconnect()

    return 0


def cmd_ohlcv_sync(args: argparse.Namespace) -> int:
    """Sync OHLCV data from Interactive Brokers."""
    setup_logging(args.verbose)
    return asyncio.run(_ohlcv_sync(args))


async def _ohlcv_backfill(args: argparse.Namespace) -> int:
    """Run intraday data backfill."""
    logger = logging.getLogger(__name__)

    # Parse target start date
    try:
        target_start_date = datetime.strptime(args.start_date, "%Y-%m-%d").date()
    except ValueError:
        logger.error(f"Invalid date format: {args.start_date}. Use YYYY-MM-DD")
        return 1

    # Get symbols to backfill
    symbols: list = []

    if args.tickers:
        symbols = args.tickers
    elif args.from_file:
        # Read symbols from a file (one per line or CSV with 'symbol' column)
        file_path = Path(args.from_file)
        if not file_path.exists():
            logger.error(f"File not found: {args.from_file}")
            return 1
        if file_path.suffix == ".csv":
            df = pd.read_csv(file_path)
            if "symbol" in df.columns:
                symbols = df["symbol"].dropna().unique().tolist()
            elif "ticker" in df.columns:
                symbols = df["ticker"].dropna().unique().tolist()
            else:
                logger.error("CSV must have 'symbol' or 'ticker' column")
                return 1
        else:
            with open(file_path) as f:
                symbols = [line.strip() for line in f if line.strip()]
    elif args.all_cached:
        # Get all symbols from existing intraday cache
        intraday_dir = Path(INTRADAY_CACHE_DIR)
        if intraday_dir.exists():
            symbols = [f.stem for f in intraday_dir.glob("*.feather")]
        else:
            logger.error(f"Intraday cache directory not found: {intraday_dir}")
            return 1

    if not symbols:
        logger.error("No symbols specified. Use --tickers, --from-file, or --all-cached")
        return 1

    # Filter excluded tickers
    excluded_count = sum(1 for t in symbols if t in EXCLUDED_TICKERS)
    if excluded_count > 0:
        logger.info(f"Excluding {excluded_count} problematic tickers")
    symbols = [t for t in symbols if t not in EXCLUDED_TICKERS]

    logger.info("=" * 80)
    logger.info(f"BACKFILL INTRADAY DATA: {len(symbols)} symbols back to {target_start_date}")
    logger.info("=" * 80)

    # Initialize IB fetcher and cache
    ib_fetcher = IBHistoricalDataFetcher(host=args.host, port=args.port, client_id=args.client_id)
    cache = DataCache(ib_fetcher)

    # Connect to IB
    connected = await ib_fetcher.connect()
    if not connected:
        logger.error("Failed to connect to IB. Please start TWS/Gateway.")
        return 1

    try:
        await cache.backfill_intraday_data(
            symbols=symbols,
            target_start_date=target_start_date,
            delay_between_symbols=args.delay,
        )
    finally:
        await ib_fetcher.disconnect()

    return 0


def cmd_ohlcv_backfill(args: argparse.Namespace) -> int:
    """Backfill historical intraday data."""
    setup_logging(args.verbose)
    return asyncio.run(_ohlcv_backfill(args))


def cmd_ohlcv_status(args: argparse.Namespace) -> int:
    """Show OHLCV cache status."""
    setup_logging(args.verbose)

    from tweet_enricher.data.cache_reader import CacheReader

    cache = CacheReader(
        daily_dir=Path(args.daily_dir),
        intraday_dir=Path(args.intraday_dir),
    )

    stats = cache.get_cache_stats()
    daily_symbols = cache.list_available_symbols("daily")
    intraday_symbols = cache.list_available_symbols("intraday")

    print("\n" + "=" * 60)
    print("OHLCV CACHE STATUS")
    print("=" * 60)
    print(f"\nDaily data:    {stats['daily_symbols']} symbols")
    print(f"Intraday data: {stats['intraday_symbols']} symbols")
    print(f"\nDaily dir:     {stats['daily_dir']}")
    print(f"Intraday dir:  {stats['intraday_dir']}")

    # Show coverage for a few symbols if requested
    if args.details:
        print("\n" + "-" * 60)
        print("Sample Coverage (first 10 symbols):")
        print("-" * 60)

        sample_symbols = daily_symbols[:10] if daily_symbols else []
        for symbol in sample_symbols:
            info = cache.get_coverage(symbol)
            daily_range = f"{info.daily_start} - {info.daily_end}" if info.has_daily else "N/A"
            intraday_range = f"{info.intraday_start} - {info.intraday_end}" if info.has_intraday else "N/A"
            print(f"  {symbol:6s}  Daily: {daily_range:25s}  Intraday: {intraday_range}")

    print("=" * 60)

    return 0


# ============================================================================
# FLOW 2: Twitter Commands
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
        months = getattr(args, "months", None)
        if getattr(args, "estimate", False):
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
            print(f"  fintweet-ml twitter sync --months {estimate['months']}")
            print("\nFor long backfills, run in background:")
            print(f"  nohup fintweet-ml twitter sync --months {estimate['months']} -v > backfill.log 2>&1 &")
            return 0

        # Test connection first
        logger.info("Testing Twitter API connection...")
        if not sync_service.client.test_connection():
            logger.error("Failed to connect to Twitter API")
            logger.error("Make sure TWITTER_API_KEY environment variable is set")
            return 1
        logger.info("Connection successful!")

        # Check for resume mode
        resume = getattr(args, "resume", False)
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
                use_date_search=getattr(args, "date_search", False),
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
                use_date_search=getattr(args, "date_search", False),
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
# FLOW 3: Dataset Preparation (No API calls)
# ============================================================================
def cmd_prepare(args: argparse.Namespace) -> int:
    """Prepare training dataset from cached data (no API calls)."""
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    from tweet_enricher.core.dataset_builder import DatasetBuilder
    from tweet_enricher.data.cache_reader import CacheReader

    # Load tweets from DB or CSV
    tweets_path = Path(args.tweets)

    if not tweets_path.exists():
        logger.error(f"Tweets file not found: {tweets_path}")
        return 1

    logger.info(f"Loading tweets from {tweets_path}")

    if tweets_path.suffix == ".db":
        # Load from SQLite database
        database = TweetDatabase(db_path=tweets_path)

        since = None
        until = None
        if args.since:
            since = ET.localize(datetime.strptime(args.since, "%Y-%m-%d"))
        if args.until:
            until = ET.localize(datetime.strptime(args.until, "%Y-%m-%d").replace(hour=23, minute=59, second=59))

        tweets = database.get_processed_tweets(since=since, until=until)
        tweets_df = pd.DataFrame([
            {
                "timestamp": t.timestamp_et,
                "ticker": t.ticker,
                "author": t.author,
                "category": t.category,
                "tweet_url": t.tweet_url,
                "text": t.text,
            }
            for t in tweets
        ])
        logger.info(f"Loaded {len(tweets_df)} tweets from database")
    else:
        # Load from CSV
        tweets_df = pd.read_csv(tweets_path)
        logger.info(f"Loaded {len(tweets_df)} tweets from CSV")

        # Apply date filters if provided
        if args.since or args.until:
            tweets_df["timestamp"] = pd.to_datetime(tweets_df["timestamp"])
            if args.since:
                since_date = pd.to_datetime(args.since)
                tweets_df = tweets_df[tweets_df["timestamp"] >= since_date]
            if args.until:
                until_date = pd.to_datetime(args.until)
                tweets_df = tweets_df[tweets_df["timestamp"] <= until_date]
            logger.info(f"After date filter: {len(tweets_df)} tweets")

    if tweets_df.empty:
        logger.error("No tweets found matching criteria")
        return 1

    # Filter out excluded tickers
    excluded_count = len(tweets_df[tweets_df["ticker"].isin(EXCLUDED_TICKERS)])
    if excluded_count > 0:
        logger.info(f"Excluding {excluded_count} tweets with problematic tickers")
        tweets_df = tweets_df[~tweets_df["ticker"].isin(EXCLUDED_TICKERS)]

    # Initialize components (no IB connection needed!)
    cache = CacheReader(
        daily_dir=Path(args.daily_dir),
        intraday_dir=Path(args.intraday_dir),
    )
    indicators = TechnicalIndicators()
    metadata_cache = StockMetadataCache()
    builder = DatasetBuilder(cache, indicators, metadata_cache)

    # Validate data coverage
    logger.info("Validating data coverage...")
    report = builder.validate_coverage(tweets_df, require_intraday=True)

    if report.missing_tickers:
        logger.warning(f"Missing OHLCV data for {len(report.missing_tickers)} tickers:")
        for ticker in report.missing_tickers[:10]:
            logger.warning(f"  - {ticker}")
        if len(report.missing_tickers) > 10:
            logger.warning(f"  ... and {len(report.missing_tickers) - 10} more")
        logger.warning("")
        logger.warning("To fetch missing data, run:")
        logger.warning(f"  fintweet-ml ohlcv sync --tickers {' '.join(report.missing_tickers[:5])}")

    if report.partial_coverage:
        logger.warning(f"{len(report.partial_coverage)} tickers have partial coverage (missing intraday)")

    logger.info(f"Data available for {report.available_tickers}/{report.total_tickers} tickers")

    # Build dataset
    logger.info("=" * 80)
    logger.info("Building dataset...")
    logger.info("=" * 80)

    output_df = builder.build(tweets_df, skip_missing=True, verbose=True)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)

    # Summary
    logger.info("")
    logger.info("=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total samples: {len(output_df)}")

    # Label distribution
    label_counts = Counter(output_df["label_1d_3class"].tolist())
    logger.info("\nLabel distribution:")
    for label, count in sorted(label_counts.items()):
        pct = 100 * count / len(output_df) if len(output_df) > 0 else 0
        logger.info(f"  {label}: {count} ({pct:.1f}%)")

    logger.info(f"\nDataset saved to: {output_path}")

    return 0


# ============================================================================
# FLOW 3 (Legacy): Enrich with API calls
# ============================================================================
async def _enrich_data(args: argparse.Namespace) -> int:
    """Async implementation of enrich command (legacy - with IB API calls)."""
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
        labels = output_df["label_1d_3class"].dropna().tolist()
        if labels:
            label_counts = Counter(labels)
            logger.info("\nLabel distribution (1-day 3-class):")
            for label, count in sorted(label_counts.items()):
                logger.info(f"  {label}: {count}")

    finally:
        await enricher.disconnect()

    return 0


def cmd_enrich(args: argparse.Namespace) -> int:
    """Enrich tweets with market data (legacy - requires IB connection)."""
    setup_logging(args.verbose)
    return asyncio.run(_enrich_data(args))


# ============================================================================
# FLOW 4: Training Commands
# ============================================================================
def cmd_train(args: argparse.Namespace) -> int:
    """Train FinBERT model."""
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    try:
        from tweet_classifier.train import train

        logger.info("Starting model training...")

        train(
            data_path=Path(args.data) if args.data else None,
            output_dir=Path(args.output_dir) if args.output_dir else None,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            freeze_bert=args.freeze_bert,
            dropout=args.dropout,
            evaluate_test=args.evaluate_test,
            temporal_split=args.temporal_split,
            early_stopping_patience=args.early_stopping_patience,
            buy_weight_boost=args.buy_weight_boost,
        )

        return 0

    except ImportError as e:
        logger.error(f"Failed to import training module: {e}")
        logger.error("Make sure tweet_classifier package is installed")
        return 1
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return 1


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Evaluate trained model."""
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    try:
        from pathlib import Path

        from tweet_classifier.evaluate import evaluate_model

        logger.info("Starting model evaluation...")

        evaluate_model(
            model_path=Path(args.model),
            data_path=Path(args.data) if args.data else None,
            output_dir=Path(args.output_dir) if args.output_dir else None,
        )

        return 0

    except ImportError as e:
        logger.error(f"Failed to import evaluation module: {e}")
        logger.error("Make sure tweet_classifier package is installed")
        return 1
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return 1


# ============================================================================
# Utility Commands
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
# Main entry point
# ============================================================================
def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="fintweet-ml",
        description="FinTweet-ML Pipeline - End-to-end ML pipeline for financial tweet analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  ohlcv      OHLCV data collection from IB API (Flow 1)
  twitter    Tweet collection from Twitter API (Flow 2)
  prepare    Dataset preparation - offline, no API calls (Flow 3)
  train      Model training (Flow 4)
  evaluate   Model evaluation (Flow 4)
  convert    Discord export conversion (utility)

Examples:
  # Collect OHLCV data
  fintweet-ml ohlcv sync --sp500

  # Collect tweets
  fintweet-ml twitter sync --months 6

  # Prepare dataset (offline)
  fintweet-ml prepare --tweets data/tweets.db --output output/dataset.csv

  # Train model
  fintweet-ml train --data output/dataset.csv --epochs 5
        """,
    )

    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--version", action="version", version="%(prog)s 0.2.0")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ============ ohlcv subcommand group ============
    ohlcv_parser = subparsers.add_parser(
        "ohlcv",
        help="OHLCV data collection from Interactive Brokers (Flow 1)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ohlcv_subparsers = ohlcv_parser.add_subparsers(dest="ohlcv_command", help="OHLCV commands")

    # ohlcv sync
    ohlcv_sync_parser = ohlcv_subparsers.add_parser(
        "sync",
        help="Sync OHLCV data from Interactive Brokers (daily + intraday)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  fintweet-ml ohlcv sync --tickers AAPL MSFT GOOGL   # Fetches daily + intraday
  fintweet-ml ohlcv sync --sp500                     # All S&P 500 stocks
  fintweet-ml ohlcv sync --tickers NVDA --daily-only # Daily only, skip intraday
        """,
    )
    ohlcv_sync_parser.add_argument("--tickers", nargs="+", help="List of stock symbols")
    ohlcv_sync_parser.add_argument("--sp500", action="store_true", help="Fetch data for all S&P 500 stocks")
    ohlcv_sync_parser.add_argument("--russell1000", action="store_true", help="Fetch data for all Russell 1000 stocks")
    ohlcv_sync_parser.add_argument("--all", action="store_true", help="Fetch data for both S&P 500 and Russell 1000")
    ohlcv_sync_parser.add_argument("--daily-only", action="store_true", help="Fetch only daily data (skip intraday)")
    ohlcv_sync_parser.add_argument("--duration", default="1 Y", help="Duration for daily data (default: 1 Y)")
    ohlcv_sync_parser.add_argument("--bar-size", default="1 day", help="Bar size for daily data (default: 1 day)")
    ohlcv_sync_parser.add_argument("--output-dir", default=str(DAILY_DATA_DIR), help=f"Daily data directory (default: {DAILY_DATA_DIR})")
    ohlcv_sync_parser.add_argument(
        "--intraday-dir", default=str(INTRADAY_CACHE_DIR), help=f"Intraday data directory (default: {INTRADAY_CACHE_DIR})"
    )
    ohlcv_sync_parser.add_argument("--batch-size", type=int, default=200, help="Batch size for daily fetch (default: 200)")
    ohlcv_sync_parser.add_argument("--batch-delay", type=float, default=2.0, help="Delay between batches (default: 2.0)")
    ohlcv_sync_parser.add_argument("--host", default="127.0.0.1", help="TWS/Gateway host (default: 127.0.0.1)")
    ohlcv_sync_parser.add_argument("--port", type=int, default=7497, help="TWS/Gateway port (default: 7497)")
    ohlcv_sync_parser.add_argument("--client-id", type=int, default=1, help="Client ID (default: 1)")
    ohlcv_sync_parser.add_argument("--exchange", default="SMART", help="Exchange (default: SMART)")
    ohlcv_sync_parser.add_argument("--currency", default="USD", help="Currency (default: USD)")
    ohlcv_sync_parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    ohlcv_sync_parser.set_defaults(func=cmd_ohlcv_sync)

    # ohlcv backfill
    ohlcv_backfill_parser = ohlcv_subparsers.add_parser(
        "backfill",
        help="Backfill historical intraday data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  fintweet-ml ohlcv backfill --start-date 2024-12-01 --all-cached
  fintweet-ml ohlcv backfill --start-date 2024-12-01 --tickers AAPL NVDA TSLA
        """,
    )
    ohlcv_backfill_parser.add_argument("--start-date", required=True, help="Target start date (YYYY-MM-DD)")
    ohlcv_backfill_parser.add_argument("--tickers", nargs="+", help="List of symbols to backfill")
    ohlcv_backfill_parser.add_argument("--from-file", help="File with symbols (CSV or text, one per line)")
    ohlcv_backfill_parser.add_argument("--all-cached", action="store_true", help="Backfill all symbols in existing cache")
    ohlcv_backfill_parser.add_argument("--delay", type=float, default=2.0, help="Delay between symbols (default: 2.0)")
    ohlcv_backfill_parser.add_argument("--host", default="127.0.0.1", help="TWS/Gateway host (default: 127.0.0.1)")
    ohlcv_backfill_parser.add_argument("--port", type=int, default=7497, help="TWS/Gateway port (default: 7497)")
    ohlcv_backfill_parser.add_argument("--client-id", type=int, default=1, help="Client ID (default: 1)")
    ohlcv_backfill_parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    ohlcv_backfill_parser.set_defaults(func=cmd_ohlcv_backfill)

    # ohlcv status
    ohlcv_status_parser = ohlcv_subparsers.add_parser(
        "status",
        help="Show OHLCV cache status",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ohlcv_status_parser.add_argument("--daily-dir", default=str(DAILY_DATA_DIR), help=f"Daily data directory (default: {DAILY_DATA_DIR})")
    ohlcv_status_parser.add_argument(
        "--intraday-dir", default=str(INTRADAY_CACHE_DIR), help=f"Intraday data directory (default: {INTRADAY_CACHE_DIR})"
    )
    ohlcv_status_parser.add_argument("--details", action="store_true", help="Show detailed coverage for sample symbols")
    ohlcv_status_parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    ohlcv_status_parser.set_defaults(func=cmd_ohlcv_status)

    # ============ twitter subcommand group ============
    twitter_parser = subparsers.add_parser(
        "twitter",
        help="Twitter API operations (Flow 2)",
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
  fintweet-ml twitter sync                    # Sync all accounts (incremental)
  fintweet-ml twitter sync --account StockMKTNewz  # Sync specific account
  fintweet-ml twitter sync --months 6         # Historical backfill (6 months)
  fintweet-ml twitter sync --months 6 --estimate  # Estimate time/cost only
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
  fintweet-ml twitter export -o tweets.csv
  fintweet-ml twitter export -o tweets.csv --since 2025-01-01
        """,
    )
    twitter_export_parser.add_argument("-o", "--output", required=True, help="Output CSV file")
    twitter_export_parser.add_argument("--since", help="Start date filter (YYYY-MM-DD)")
    twitter_export_parser.add_argument("--until", help="End date filter (YYYY-MM-DD)")
    twitter_export_parser.add_argument("--account", help="Filter by account")
    twitter_export_parser.add_argument("--ticker", help="Filter by ticker")
    twitter_export_parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    twitter_export_parser.set_defaults(func=cmd_twitter_export)

    # ============ prepare subcommand (Flow 3 - NEW) ============
    prepare_parser = subparsers.add_parser(
        "prepare",
        help="Prepare training dataset from cached data (Flow 3 - no API calls)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  fintweet-ml prepare --tweets data/tweets.db --output output/dataset.csv
  fintweet-ml prepare --tweets tweets.csv --output dataset.csv --since 2025-01-01
        """,
    )
    prepare_parser.add_argument("--tweets", required=True, help="Input tweets file (DB or CSV)")
    prepare_parser.add_argument("-o", "--output", required=True, help="Output CSV file")
    prepare_parser.add_argument("--since", help="Start date filter (YYYY-MM-DD)")
    prepare_parser.add_argument("--until", help="End date filter (YYYY-MM-DD)")
    prepare_parser.add_argument("--daily-dir", default=str(DAILY_DATA_DIR), help=f"Daily OHLCV cache directory (default: {DAILY_DATA_DIR})")
    prepare_parser.add_argument(
        "--intraday-dir", default=str(INTRADAY_CACHE_DIR), help=f"Intraday cache directory (default: {INTRADAY_CACHE_DIR})"
    )
    prepare_parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    prepare_parser.set_defaults(func=cmd_prepare)

    # ============ enrich subcommand (Legacy - kept for backward compatibility) ============
    enrich_parser = subparsers.add_parser(
        "enrich",
        help="[Legacy] Enrich tweets with market data (requires IB connection)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
NOTE: This is a legacy command. For offline dataset preparation, use 'prepare' instead.

Examples:
  fintweet-ml enrich -i tweets.csv -o enriched.csv
        """,
    )
    enrich_parser.add_argument("-i", "--input", type=str, required=True, help="Input tweets CSV file")
    enrich_parser.add_argument("-o", "--output", type=str, required=True, help="Output enriched CSV file")
    enrich_parser.add_argument("--host", default="127.0.0.1", help="TWS/Gateway host (default: 127.0.0.1)")
    enrich_parser.add_argument("--port", type=int, default=7497, help="TWS/Gateway port (default: 7497)")
    enrich_parser.add_argument("--client-id", type=int, default=1, help="Client ID (default: 1)")
    enrich_parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    enrich_parser.set_defaults(func=cmd_enrich)

    # ============ train subcommand (Flow 4) ============
    train_parser = subparsers.add_parser(
        "train",
        help="Train FinBERT model (Flow 4)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  fintweet-ml train --data output/dataset.csv --epochs 5
  fintweet-ml train --data dataset.csv --epochs 5 --freeze-bert --evaluate-test
        """,
    )
    train_parser.add_argument("--data", help="Path to enriched CSV data file")
    train_parser.add_argument("--output-dir", help="Directory to save model and artifacts")
    train_parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs (default: 5)")
    train_parser.add_argument("--batch-size", type=int, default=16, help="Batch size per device (default: 16)")
    train_parser.add_argument("--learning-rate", type=float, default=2e-5, help="Initial learning rate (default: 2e-5)")
    train_parser.add_argument("--freeze-bert", action="store_true", help="Freeze BERT parameters (faster training)")
    train_parser.add_argument("--dropout", type=float, default=0.3, help="Dropout probability (default: 0.3)")
    train_parser.add_argument("--evaluate-test", action="store_true", help="Run evaluation on test set after training")
    train_parser.add_argument("--temporal-split", action="store_true", help="Use temporal split instead of random hash split")
    train_parser.add_argument("--early-stopping-patience", type=int, default=2, help="Early stopping patience (default: 2)")
    train_parser.add_argument("--buy-weight-boost", type=float, default=1.0, help="BUY class weight multiplier (default: 1.0)")
    train_parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    train_parser.set_defaults(func=cmd_train)

    # ============ evaluate subcommand (Flow 4) ============
    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate trained model (Flow 4)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  fintweet-ml evaluate --model models/final --data output/dataset.csv
        """,
    )
    evaluate_parser.add_argument("--model", required=True, help="Path to trained model directory")
    evaluate_parser.add_argument("--data", help="Path to evaluation data CSV")
    evaluate_parser.add_argument("--output-dir", help="Directory to save evaluation results")
    evaluate_parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    evaluate_parser.set_defaults(func=cmd_evaluate)

    # ============ convert subcommand (Utility) ============
    convert_parser = subparsers.add_parser(
        "convert",
        help="Convert Discord channel export to CSV (utility)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  fintweet-ml convert -i discord_data.txt -o output.csv
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

    # ============ filter-volume subcommand (Utility) ============
    filter_parser = subparsers.add_parser(
        "filter-volume",
        help="Filter tickers by average daily trading volume (utility)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  fintweet-ml filter-volume --min-volume 1000000
        """,
    )
    filter_parser.add_argument("--data-dir", default=str(DAILY_DATA_DIR), help=f"Data directory (default: {DAILY_DATA_DIR})")
    filter_parser.add_argument("--min-volume", type=float, default=1_000_000, help="Minimum avg volume (default: 1,000,000)")
    filter_parser.add_argument("--output", help="Output CSV file (optional)")
    filter_parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    filter_parser.set_defaults(func=cmd_filter_volume)

    # Parse arguments
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 0

    # Validate ohlcv sync arguments
    if args.command == "ohlcv" and getattr(args, "ohlcv_command", None) == "sync":
        if not args.tickers and not args.sp500 and not args.russell1000 and not args.all:
            ohlcv_sync_parser.error("Either --tickers, --sp500, --russell1000, or --all must be provided")
        if sum([bool(args.tickers), args.sp500, args.russell1000, args.all]) > 1:
            ohlcv_sync_parser.error("Please provide only one of: --tickers, --sp500, --russell1000, or --all")

    # Handle subcommand groups
    if args.command == "ohlcv":
        if getattr(args, "ohlcv_command", None) is None:
            ohlcv_parser.print_help()
            return 0

    if args.command == "twitter":
        if getattr(args, "twitter_command", None) is None:
            twitter_parser.print_help()
            return 0

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

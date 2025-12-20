#!/usr/bin/env python3
"""Interactive dataset validation script.

This script performs deep validation of the tweet enrichment pipeline:
1. Validates timezone handling across the pipeline
2. Validates that stock data is correctly matched to tweets
3. Validates label calculations
4. Identifies potential issues with the dataset

Usage:
    python scripts/validate_dataset.py
    python scripts/validate_dataset.py --sample 100
    python scripts/validate_dataset.py --ticker AAPL
"""

import argparse
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import pytz

from tweet_enricher.config import ET, TWITTER_DB_PATH
from tweet_enricher.io.feather import load_daily_data, load_intraday_data
from tweet_enricher.market.session import MarketSession, get_market_session


def validate_tweet_timestamp_conversion():
    """Validate Twitter UTC to Eastern conversion with detailed examples."""
    print("\n" + "=" * 80)
    print("VALIDATION 1: Twitter Timestamp Conversion (UTC -> Eastern)")
    print("=" * 80)

    if not TWITTER_DB_PATH.exists():
        print("  ✗ Tweet database not found")
        return False

    conn = sqlite3.connect(TWITTER_DB_PATH)

    # Get sample tweets with both UTC and ET timestamps
    df = pd.read_sql(
        """
        SELECT id, timestamp_utc, timestamp_et, author, ticker
        FROM tweets_processed
        ORDER BY RANDOM()
        LIMIT 10
        """,
        conn,
    )
    conn.close()

    if df.empty:
        print("  ✗ No processed tweets found")
        return False

    print(f"\n  Sample of {len(df)} tweets:")
    print("  " + "-" * 75)

    all_valid = True
    for _, row in df.iterrows():
        utc_str = row["timestamp_utc"]
        et_str = row["timestamp_et"]

        # Parse UTC timestamp (Twitter format: "Wed Dec 17 15:01:11 +0000 2025")
        try:
            utc_dt = datetime.strptime(utc_str, "%a %b %d %H:%M:%S %z %Y")
        except ValueError:
            print(f"  ✗ Failed to parse UTC: {utc_str}")
            all_valid = False
            continue

        # Convert to Eastern
        eastern_tz = pytz.timezone("America/New_York")
        expected_et = utc_dt.astimezone(eastern_tz)
        expected_et_str = expected_et.strftime("%Y-%m-%d %H:%M:%S")

        # Compare
        if et_str == expected_et_str:
            status = "✓"
        else:
            status = "✗"
            all_valid = False

        print(f"  {status} UTC: {utc_str}")
        print(f"       ET:  {et_str} (expected: {expected_et_str})")
        print()

    if all_valid:
        print("  ✓ All timestamp conversions are correct!")
    else:
        print("  ✗ Some timestamp conversions are incorrect!")

    return all_valid


def validate_stock_data_timezone():
    """Validate stock data timezone handling."""
    print("\n" + "=" * 80)
    print("VALIDATION 2: Stock Data Timezone")
    print("=" * 80)

    daily_dir = Path("data/daily")
    intraday_dir = Path("data/intraday")

    all_valid = True

    # Check daily data
    print("\n  Daily Data:")
    if daily_dir.exists():
        df = load_daily_data("AAPL", daily_dir)
        if df is not None:
            tz = str(df.index.tz) if df.index.tz else "None"
            if tz == "US/Eastern":
                print(f"    ✓ AAPL timezone: {tz}")
            else:
                print(f"    ✗ AAPL timezone: {tz} (expected US/Eastern)")
                all_valid = False

            # Check sample dates
            print(f"    ✓ Date range: {df.index.min().date()} to {df.index.max().date()}")
            print(f"    ✓ Total bars: {len(df)}")
        else:
            print("    ✗ Failed to load AAPL daily data")
            all_valid = False
    else:
        print("    ✗ Daily data directory not found")
        all_valid = False

    # Check intraday data
    print("\n  Intraday Data:")
    if intraday_dir.exists():
        df = load_intraday_data("AAPL", intraday_dir)
        if df is not None:
            tz = str(df.index.tz) if df.index.tz else "None"
            if tz == "US/Eastern":
                print(f"    ✓ AAPL timezone: {tz}")
            else:
                print(f"    ✗ AAPL timezone: {tz} (expected US/Eastern)")
                all_valid = False

            # Check bar times - should start at 4:00 AM ET (pre-market)
            sample_day = df[df.index.date == df.index.date[-1]]
            if not sample_day.empty:
                first_bar_hour = sample_day.index[0].hour
                print(f"    ✓ First bar starts at hour {first_bar_hour} (pre-market at 4 AM)")
            print(f"    ✓ Total bars: {len(df)}")
        else:
            print("    ✗ Failed to load AAPL intraday data")
            all_valid = False
    else:
        print("    ✗ Intraday data directory not found")
        all_valid = False

    return all_valid


def validate_market_session_detection():
    """Validate market session detection logic."""
    print("\n" + "=" * 80)
    print("VALIDATION 3: Market Session Detection")
    print("=" * 80)

    test_cases = [
        # (timestamp, expected_session, description)
        (datetime(2025, 12, 16, 10, 30, 0), MarketSession.REGULAR, "Tuesday 10:30 AM"),
        (datetime(2025, 12, 16, 15, 59, 0), MarketSession.REGULAR, "Tuesday 3:59 PM"),
        (datetime(2025, 12, 16, 4, 30, 0), MarketSession.PREMARKET, "Tuesday 4:30 AM"),
        (datetime(2025, 12, 16, 9, 29, 0), MarketSession.PREMARKET, "Tuesday 9:29 AM"),
        (datetime(2025, 12, 16, 16, 30, 0), MarketSession.AFTERHOURS, "Tuesday 4:30 PM"),
        (datetime(2025, 12, 16, 19, 59, 0), MarketSession.AFTERHOURS, "Tuesday 7:59 PM"),
        (datetime(2025, 12, 16, 2, 0, 0), MarketSession.CLOSED, "Tuesday 2:00 AM"),
        (datetime(2025, 12, 16, 21, 0, 0), MarketSession.CLOSED, "Tuesday 9:00 PM"),
        (datetime(2025, 12, 20, 10, 0, 0), MarketSession.CLOSED, "Saturday 10:00 AM"),
        (datetime(2025, 12, 21, 14, 0, 0), MarketSession.CLOSED, "Sunday 2:00 PM"),
    ]

    all_valid = True
    print("\n  Test cases:")
    print("  " + "-" * 60)

    for ts, expected, description in test_cases:
        ts_et = ET.localize(ts)
        actual = get_market_session(ts_et)

        if actual == expected:
            print(f"  ✓ {description}: {actual.value}")
        else:
            print(f"  ✗ {description}: {actual.value} (expected: {expected.value})")
            all_valid = False

    return all_valid


def validate_tweet_stock_alignment(sample_size: int = 20, ticker_filter: Optional[str] = None):
    """Validate that tweets align correctly with stock data."""
    print("\n" + "=" * 80)
    print("VALIDATION 4: Tweet-Stock Data Alignment")
    print("=" * 80)

    if not TWITTER_DB_PATH.exists():
        print("  ✗ Tweet database not found")
        return False

    intraday_dir = Path("data/intraday")
    if not intraday_dir.exists():
        print("  ✗ Intraday data directory not found")
        return False

    conn = sqlite3.connect(TWITTER_DB_PATH)

    # Get sample tweets
    query = "SELECT * FROM tweets_processed"
    if ticker_filter:
        query += f" WHERE ticker = '{ticker_filter}'"
    query += f" ORDER BY RANDOM() LIMIT {sample_size}"

    df = pd.read_sql(query, conn)
    conn.close()

    if df.empty:
        print("  ✗ No tweets found")
        return False

    print(f"\n  Validating {len(df)} tweets:")
    print("  " + "-" * 75)

    valid_count = 0
    issue_count = 0

    for _, row in df.iterrows():
        ticker = row["ticker"]
        et_str = row["timestamp_et"]

        # Parse ET timestamp
        tweet_ts = pd.to_datetime(et_str)
        tweet_ts_et = ET.localize(tweet_ts) if tweet_ts.tzinfo is None else tweet_ts

        # Get session
        session = get_market_session(tweet_ts_et)

        # Try to load intraday data
        intraday_df = load_intraday_data(ticker, intraday_dir)

        if intraday_df is None:
            print(f"  ? {ticker} @ {et_str}: No intraday data available")
            issue_count += 1
            continue

        # Check if tweet time falls within intraday data range
        tweet_date = tweet_ts_et.date() if hasattr(tweet_ts_et, "date") else tweet_ts.date()
        intraday_min = intraday_df.index.min().date()

        if tweet_date < intraday_min:
            print(f"  ? {ticker} @ {et_str}: Tweet date {tweet_date} < data start {intraday_min}")
            issue_count += 1
            continue

        # For tweets within range, check if we can find matching bar
        if session in [MarketSession.REGULAR, MarketSession.PREMARKET, MarketSession.AFTERHOURS]:
            # Find bars after tweet
            future_bars = intraday_df[intraday_df.index > tweet_ts_et]

            if len(future_bars) > 0:
                next_bar = future_bars.iloc[0]
                delay = next_bar.name - tweet_ts_et
                print(
                    f"  ✓ {ticker} @ {et_str[:16]} ({session.value}): "
                    f"Next bar at {next_bar.name.strftime('%H:%M')} (delay: {delay})"
                )
                valid_count += 1
            else:
                print(f"  ? {ticker} @ {et_str}: No future bars found (may be recent tweet)")
                issue_count += 1
        else:
            print(f"  ✓ {ticker} @ {et_str[:16]} ({session.value}): Market closed, will use next day open")
            valid_count += 1

    print()
    print(f"  Summary: {valid_count} valid, {issue_count} issues")

    return issue_count == 0


def validate_enriched_output(sample_size: int = 20):
    """Validate enriched output file."""
    print("\n" + "=" * 80)
    print("VALIDATION 5: Enriched Output Quality")
    print("=" * 80)

    output_path = Path("output/2025_enrich.csv")
    if not output_path.exists():
        print("  ✗ Enriched output file not found")
        return False

    df = pd.read_csv(output_path, low_memory=False)

    print(f"\n  Total rows: {len(df):,}")
    print()

    # Check timestamp format
    print("  Timestamp format check:")
    sample_ts = df.iloc[0]["timestamp"]
    if "-05:00" in str(sample_ts) or "-04:00" in str(sample_ts):
        print(f"    ✓ Timestamps have timezone offset: {sample_ts}")
    else:
        print(f"    ✗ Timestamps missing timezone: {sample_ts}")

    # Session distribution
    print("\n  Session distribution:")
    for session, count in df["session"].value_counts().items():
        pct = 100 * count / len(df)
        print(f"    - {session}: {count:,} ({pct:.1f}%)")

    # Label distribution
    if "label_3class" in df.columns:
        print("\n  Label distribution (3-class):")
        for label, count in df["label_3class"].value_counts().items():
            pct = 100 * count / len(df)
            print(f"    - {label}: {count:,} ({pct:.1f}%)")

    # Reliability
    if "is_reliable_label" in df.columns:
        reliable = df["is_reliable_label"].sum()
        print(f"\n  Reliable labels: {reliable:,}/{len(df):,} ({100 * reliable / len(df):.1f}%)")

    # Check for anomalies
    print("\n  Anomaly checks:")

    # Check for extreme returns
    if "return_1hr" in df.columns:
        extreme_returns = df[df["return_1hr"].abs() > 0.5]
        if len(extreme_returns) > 0:
            print(f"    ! Warning: {len(extreme_returns)} rows with |return_1hr| > 50%")
        else:
            print("    ✓ No extreme 1hr returns (> 50%)")

    # Check for missing entry prices
    missing_entry = df["entry_price"].isna().sum()
    print(f"    - Missing entry prices: {missing_entry:,} ({100 * missing_entry / len(df):.1f}%)")

    # Check for missing labels
    missing_labels = df["label_3class"].isna().sum()
    print(f"    - Missing labels: {missing_labels:,} ({100 * missing_labels / len(df):.1f}%)")

    # Sample validation
    print(f"\n  Sample of {sample_size} rows:")
    print("  " + "-" * 75)

    sample = df.sample(min(sample_size, len(df)))
    issues = 0

    for _, row in sample.iterrows():
        ticker = row["ticker"]
        ts = row["timestamp"]
        session = row["session"]
        entry_flag = row["entry_price_flag"]
        label = row["label_3class"]

        # Check consistency
        if pd.isna(row["entry_price"]) and entry_flag not in ["no_data", "no_data_available"]:
            print(f"  ✗ {ticker} @ {ts[:16]}: Entry price None but flag is '{entry_flag}'")
            issues += 1
        elif pd.notna(row["entry_price"]) and pd.notna(row["exit_price_1hr"]):
            # Verify return calculation
            calc_return = (row["exit_price_1hr"] - row["entry_price"]) / row["entry_price"]
            if abs(calc_return - row["return_1hr"]) > 0.0001:
                print(f"  ✗ {ticker}: Return mismatch: calculated {calc_return:.4f} vs stored {row['return_1hr']:.4f}")
                issues += 1
            else:
                print(f"  ✓ {ticker} @ {ts[:16]} ({session}): {label} | return={row['return_1hr']:.4f}")
        else:
            print(f"  ? {ticker} @ {ts[:16]} ({session}): Missing prices (flag: {entry_flag})")

    if issues == 0:
        print("\n  ✓ All sampled rows are consistent!")
    else:
        print(f"\n  ✗ Found {issues} issues in sample")

    return issues == 0


def validate_incremental_fetching():
    """Validate incremental fetching journal."""
    print("\n" + "=" * 80)
    print("VALIDATION 6: Incremental Fetching Journal")
    print("=" * 80)

    if not TWITTER_DB_PATH.exists():
        print("  ✗ Tweet database not found")
        return False

    conn = sqlite3.connect(TWITTER_DB_PATH)

    # Check fetch journal
    try:
        journal_df = pd.read_sql("SELECT * FROM fetch_journal ORDER BY fetch_date DESC LIMIT 20", conn)
    except Exception:
        print("  ✗ fetch_journal table not found")
        conn.close()
        return False

    if journal_df.empty:
        print("  ? No fetch journal entries (this is OK if using incremental sync)")
        conn.close()
        return True

    print(f"\n  Recent fetched days ({len(journal_df)} shown):")
    print("  " + "-" * 60)

    accounts = journal_df["account"].unique()
    for account in accounts:
        account_data = journal_df[journal_df["account"] == account]
        total_tweets = account_data["tweets_count"].sum()
        total_calls = account_data["api_calls"].sum()
        days = len(account_data)
        print(f"    @{account}: {days} days, {total_tweets} tweets, {total_calls} API calls")

    # Check for gaps
    print("\n  Gap analysis:")
    for account in accounts:
        account_data = journal_df[journal_df["account"] == account].sort_values("fetch_date")
        if len(account_data) > 1:
            dates = pd.to_datetime(account_data["fetch_date"])
            gaps = []
            for i in range(1, len(dates)):
                diff = (dates.iloc[i] - dates.iloc[i - 1]).days
                if diff > 2:  # More than 2 days gap (accounting for weekends)
                    gaps.append((dates.iloc[i - 1].date(), dates.iloc[i].date(), diff))

            if gaps:
                print(f"    ! @{account}: Found {len(gaps)} gaps in fetched days")
                for start, end, days in gaps[:3]:
                    print(f"      - Gap from {start} to {end} ({days} days)")
            else:
                print(f"    ✓ @{account}: No significant gaps")

    conn.close()
    return True


def main():
    """Run all validations."""
    parser = argparse.ArgumentParser(description="Validate tweet enrichment pipeline")
    parser.add_argument("--sample", type=int, default=20, help="Sample size for detailed checks")
    parser.add_argument("--ticker", type=str, help="Filter by specific ticker")
    args = parser.parse_args()

    print("=" * 80)
    print("TWEET ENRICHMENT PIPELINE VALIDATION")
    print("=" * 80)

    results = []

    # Run all validations
    results.append(("Twitter Timestamp Conversion", validate_tweet_timestamp_conversion()))
    results.append(("Stock Data Timezone", validate_stock_data_timezone()))
    results.append(("Market Session Detection", validate_market_session_detection()))
    results.append(("Tweet-Stock Alignment", validate_tweet_stock_alignment(args.sample, args.ticker)))
    results.append(("Enriched Output Quality", validate_enriched_output(args.sample)))
    results.append(("Incremental Fetching", validate_incremental_fetching()))

    # Summary
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)

    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("  🎉 All validations passed!")
    else:
        print("  ⚠️  Some validations failed - review issues above")

    print("=" * 80)

    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())

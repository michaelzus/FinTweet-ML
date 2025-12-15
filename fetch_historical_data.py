"""
Interactive Brokers Historical Data Fetcher CLI.

Command-line interface for fetching historical OHLCV data from Interactive Brokers.
"""

import argparse
import asyncio
import logging

from helpers import fetch_sp500_tickers, fetch_russell1000_tickers, save_daily_data, DAILY_DATA_DIR
from ib_fetcher import IBHistoricalDataFetcher

# Module-level logger
logger = logging.getLogger(__name__)


async def main() -> None:
    """Main execution function."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    parser = argparse.ArgumentParser(description="Fetch historical OHLCV data from Interactive Brokers")

    parser.add_argument("--symbols", nargs="+", help="List of stock symbols (e.g., AAPL MSFT GOOGL)")

    parser.add_argument("--sp500", action="store_true", help="Fetch data for all S&P 500 stocks")

    parser.add_argument("--russell1000", action="store_true", help="Fetch data for all Russell 1000 stocks")

    parser.add_argument("--all", action="store_true", help="Fetch data for both S&P 500 and Russell 1000 stocks combined")

    parser.add_argument(
        "--duration", default="1 Y", help="Duration string (e.g., '1 Y', '6 M', '30 D', '1 W'). Default: 1 Y"
    )

    parser.add_argument(
        "--bar-size",
        default="1 day",
        help="Bar size (e.g., '1 day', '1 hour', '30 mins', '5 mins', '1 min'). Default: 1 day",
    )

    parser.add_argument(
        "--output-dir", default=str(DAILY_DATA_DIR), help=f"Output directory for feather files. Default: {DAILY_DATA_DIR}"
    )

    parser.add_argument("--batch-size", type=int, default=200, help="Number of symbols to fetch per batch. Default: 200")

    parser.add_argument("--batch-delay", type=float, default=2.0, help="Delay in seconds between batches. Default: 2.0")

    parser.add_argument("--host", default="127.0.0.1", help="TWS/Gateway host. Default: 127.0.0.1")

    parser.add_argument("--port", type=int, default=7497, help="TWS/Gateway port. Default: 7497 (TWS), use 4002 for Gateway")

    parser.add_argument("--client-id", type=int, default=1, help="Client ID. Default: 1")

    parser.add_argument("--exchange", default="SMART", help="Exchange. Default: SMART")

    parser.add_argument("--currency", default="USD", help="Currency. Default: USD")

    args = parser.parse_args()

    # Validate arguments
    if not args.symbols and not args.sp500 and not args.russell1000 and not args.all:
        parser.error("Either --symbols, --sp500, --russell1000, or --all must be provided")

    if sum([bool(args.symbols), args.sp500, args.russell1000, args.all]) > 1:
        parser.error("Please provide only one of: --symbols, --sp500, --russell1000, or --all")

    # Determine which symbols to fetch
    if args.all:
        logger.info("Fetching combined S&P 500 and Russell 1000 ticker lists from Wikipedia...")
        try:
            sp500_symbols = fetch_sp500_tickers()
            logger.info(f"Found {len(sp500_symbols)} S&P 500 tickers")

            russell1000_symbols = fetch_russell1000_tickers()
            logger.info(f"Found {len(russell1000_symbols)} Russell 1000 tickers")

            # Combine and deduplicate
            symbols = list(set(sp500_symbols + russell1000_symbols))
            logger.info(f"Combined total: {len(symbols)} unique tickers (after deduplication)")
        except Exception as e:
            logger.error(f"Error: {e}")
            return
    elif args.sp500:
        logger.info("Fetching S&P 500 ticker list from Wikipedia...")
        try:
            symbols = fetch_sp500_tickers()
            logger.info(f"Found {len(symbols)} S&P 500 tickers")
        except Exception as e:
            logger.error(f"Error: {e}")
            return
    elif args.russell1000:
        logger.info("Fetching Russell 1000 ticker list from Wikipedia...")
        try:
            symbols = fetch_russell1000_tickers()
            logger.info(f"Found {len(symbols)} Russell 1000 tickers")
        except Exception as e:
            logger.error(f"Error: {e}")
            return
    else:
        symbols = args.symbols

    fetcher = IBHistoricalDataFetcher(host=args.host, port=args.port, client_id=args.client_id)

    # Connect to IB
    connected = await fetcher.connect()
    if not connected:
        return

    try:
        # Fetch data for all symbols
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
            # Save to feather files (one per ticker)
            from pathlib import Path

            output_dir = Path(args.output_dir)
            for symbol, df in data_dict.items():
                save_daily_data(symbol, df, output_dir)

            logger.info("Summary:")
            total_records = sum(len(df) for df in data_dict.values())
            logger.info(f"Total records: {total_records}")
            logger.info(f"Files saved to: {args.output_dir}/")
            for symbol, df in data_dict.items():
                logger.debug(f"  - {symbol}.feather ({len(df)} bars)")
        else:
            logger.warning("No data fetched")

    finally:
        # Always disconnect
        await fetcher.disconnect()


if __name__ == "__main__":
    asyncio.run(main())

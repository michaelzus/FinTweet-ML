"""
Filter tickers by volume.

This script demonstrates how to filter tickers based on average daily trading volume
and optionally save the filtered list to a CSV file.
"""

import argparse
import logging

from helpers import filter_tickers_by_volume, DAILY_DATA_DIR
import pandas as pd

logger = logging.getLogger(__name__)


def main() -> None:
    """Main execution function."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    parser = argparse.ArgumentParser(description="Filter tickers by average daily trading volume")

    parser.add_argument(
        "--data-dir",
        default=str(DAILY_DATA_DIR),
        help=f"Directory containing feather files with historical data. Default: {DAILY_DATA_DIR}"
    )

    parser.add_argument(
        "--min-volume",
        type=float,
        default=1_000_000,
        help="Minimum average daily volume (default: 1,000,000)"
    )

    parser.add_argument(
        "--output",
        help="Output CSV file to save filtered tickers (optional)"
    )

    args = parser.parse_args()

    try:
        # Filter tickers by volume
        logger.info("=" * 80)
        logger.info(f"FILTERING TICKERS (min avg volume: {args.min_volume:,.0f})")
        logger.info("=" * 80)

        filtered_tickers = filter_tickers_by_volume(args.data_dir, args.min_volume)

        if filtered_tickers:
            logger.info(f"Found {len(filtered_tickers)} tickers meeting criteria:")
            for ticker in sorted(filtered_tickers):
                logger.debug(f"  - {ticker}")

            # Save to CSV if output file specified
            if args.output:
                df = pd.DataFrame({"symbol": filtered_tickers})
                df.to_csv(args.output, index=False)
                logger.info(f"Filtered tickers saved to: {args.output}")

        else:
            logger.warning(f"No tickers found with average volume >= {args.min_volume:,.0f}")

    except FileNotFoundError as e:
        logger.error(f"Error: {e}")
        logger.error(f"Make sure the data directory '{args.data_dir}' exists and contains feather files.")
    except Exception as e:
        logger.error(f"Error: {e}")


if __name__ == "__main__":
    main()

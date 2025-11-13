"""
Helper utilities for fetching stock data.

This module contains utility functions for working with stock symbols
and other helper operations.
"""

import logging
from pathlib import Path
from typing import Dict, List

import pandas as pd

logger = logging.getLogger(__name__)


def fetch_sp500_tickers() -> List[str]:
    """
    Fetch the list of S&P 500 ticker symbols from Wikipedia.

    Returns:
        List of S&P 500 ticker symbols

    Raises:
        Exception: If unable to fetch the ticker list
    """
    try:
        # Read S&P 500 table from Wikipedia with proper headers
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

        # Add headers to avoid 403 Forbidden error
        headers = {"User-Agent": "Mozilla/5.0 (compatible; Python/pandas)"}

        tables = pd.read_html(url, storage_options=headers, header=0)

        # Find the table with the most rows (usually the main S&P 500 list)
        # Skip empty tables and look for one with Symbol column
        sp500_table = None
        for table in tables:
            if not table.empty and len(table) > 100 and "Symbol" in table.columns:
                sp500_table = table
                break

        if sp500_table is None:
            raise Exception("Could not find S&P 500 table with Symbol column")

        # Extract ticker symbols from Symbol column
        tickers = sp500_table["Symbol"].tolist()

        # Clean tickers (remove any special characters that IB might not recognize)
        # Also exclude problematic tickers that always fail
        excluded_tickers = {"BRK-B", "BF-B"}

        cleaned_tickers = []
        for ticker in tickers:
            if pd.notna(ticker):  # Check if not NaN
                ticker_str = str(ticker).replace(".", "-").strip()
                if ticker_str and ticker_str.upper() != "NAN":  # Filter empty and NaN strings
                    # Exclude problematic tickers
                    if ticker_str not in excluded_tickers:
                        cleaned_tickers.append(ticker_str)
                    else:
                        logger.debug(f"Excluding problematic ticker: {ticker_str}")

        return cleaned_tickers

    except Exception as e:
        raise Exception(f"Failed to fetch S&P 500 tickers: {e}")


def fetch_russell1000_tickers() -> List[str]:
    """
    Fetch the list of Russell 1000 ticker symbols from Wikipedia.

    Returns:
        List of Russell 1000 ticker symbols

    Raises:
        Exception: If unable to fetch the ticker list
    """
    try:
        # Read Russell 1000 table from Wikipedia
        url = "https://en.wikipedia.org/wiki/Russell_1000_Index"

        # Add headers to avoid 403 Forbidden error
        headers = {"User-Agent": "Mozilla/5.0 (compatible; Python/pandas)"}

        tables = pd.read_html(url, storage_options=headers, header=0)

        # Find the table with tickers (look for one with Symbol or Ticker column and many rows)
        component_table = None
        for table in tables:
            if not table.empty and len(table) > 50:
                # Check for Symbol or Ticker column
                if "Symbol" in table.columns or "Ticker" in table.columns:
                    component_table = table
                    break

        if component_table is None:
            raise Exception("Could not find Russell 1000 table with Symbol column")

        # Extract ticker symbols
        symbol_column = "Symbol" if "Symbol" in component_table.columns else "Ticker"
        tickers = component_table[symbol_column].tolist()

        # Exclude problematic tickers (Class A/B shares and others that fail)
        excluded_tickers = {
            "BRK-B", "BF-B", "BF-A", "CWEN-A", "HEI-A", "LEN-B", "UHAL-B",
            "LNW", "DNB", "CCCS", "COOP"
        }

        # Clean tickers
        cleaned_tickers = []
        for ticker in tickers:
            if pd.notna(ticker):
                ticker_str = str(ticker).replace(".", "-").strip()
                if ticker_str and ticker_str.upper() != "NAN":
                    # Exclude problematic tickers and anything with -A or -B suffix
                    if ticker_str not in excluded_tickers and not ticker_str.endswith(("-A", "-B")):
                        cleaned_tickers.append(ticker_str)
                    else:
                        logger.debug(f"Excluding problematic ticker: {ticker_str}")

        logger.info(f"Fetched {len(cleaned_tickers)} Russell 1000 tickers from Wikipedia")
        return cleaned_tickers

    except Exception as e:
        raise Exception(f"Failed to fetch Russell 1000 tickers: {e}")


def save_to_csv(data_dict: Dict[str, pd.DataFrame], output_dir: str) -> None:
    """
    Save DataFrames to separate CSV files per ticker.

    Args:
        data_dict: Dictionary mapping symbols to DataFrames
        output_dir: Output directory path for CSV files
    """
    try:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for symbol, df in data_dict.items():
            output_file = output_path / f"{symbol}.csv"
            df.to_csv(output_file, index=False)
            logger.debug(f"Data for {symbol} saved to {output_file}")
    except Exception as e:
        logger.error(f"Error saving to CSV: {e}")


def read_tickers_from_csv(csv_file: str, column_name: str = "symbol") -> List[str]:
    """
    Read a list of ticker symbols from a CSV file.

    Args:
        csv_file: Path to CSV file containing ticker symbols
        column_name: Name of the column containing ticker symbols (default: "symbol")

    Returns:
        List of ticker symbols

    Raises:
        FileNotFoundError: If CSV file doesn't exist
        KeyError: If column_name doesn't exist in CSV
        Exception: For other errors during reading
    """
    try:
        csv_path = Path(csv_file)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_file}")

        df = pd.read_csv(csv_path)

        if column_name not in df.columns:
            available_columns = df.columns.tolist()
            raise KeyError(f"Column '{column_name}' not found. Available columns: {available_columns}")

        tickers = df[column_name].dropna().unique().tolist()
        tickers = [str(ticker).strip() for ticker in tickers if str(ticker).strip()]

        logger.info(f"Read {len(tickers)} unique tickers from {csv_file}")
        return tickers

    except Exception as e:
        logger.error(f"Error reading tickers from CSV: {e}")
        raise


def filter_tickers_by_volume(data_dir: str, min_avg_volume: float = 1_000_000) -> List[str]:
    """
    Filter tickers based on average daily trading volume.

    Reads CSV files from the specified directory, calculates average daily volume
    for each ticker, and returns tickers with average volume above threshold.

    Args:
        data_dir: Directory containing CSV files with historical data
        min_avg_volume: Minimum average daily volume threshold (default: 1,000,000)

    Returns:
        List of ticker symbols meeting the volume criteria

    Raises:
        FileNotFoundError: If data directory doesn't exist
    """
    try:
        data_path = Path(data_dir)
        if not data_path.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

        filtered_tickers = []
        csv_files = list(data_path.glob("*.csv"))

        if not csv_files:
            logger.warning(f"No CSV files found in {data_dir}")
            return []

        logger.info(f"Analyzing {len(csv_files)} CSV files in {data_dir}...")

        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)

                # Check if volume column exists
                if "volume" not in df.columns:
                    logger.warning(f"No 'volume' column in {csv_file.name}, skipping")
                    continue

                # Calculate average volume
                avg_volume = df["volume"].mean()

                # Extract symbol from filename (e.g., AAPL.csv -> AAPL)
                symbol = csv_file.stem

                if avg_volume >= min_avg_volume:
                    filtered_tickers.append(symbol)
                    logger.debug(f"{symbol}: avg volume = {avg_volume:,.0f} (PASS)")
                else:
                    logger.debug(f"{symbol}: avg volume = {avg_volume:,.0f} (FAIL)")

            except Exception as e:
                logger.error(f"Error processing {csv_file.name}: {e}")
                continue

        logger.info(f"Found {len(filtered_tickers)} tickers with avg volume >= {min_avg_volume:,.0f}")
        return filtered_tickers

    except Exception as e:
        logger.error(f"Error filtering tickers by volume: {e}")
        raise

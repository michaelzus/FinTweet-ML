"""
Interactive Brokers data fetcher.

This module handles all interactions with Interactive Brokers TWS/Gateway
for fetching historical OHLCV data.
"""

import asyncio
import logging
from typing import Dict, Optional

import pandas as pd
from ib_async import IB, Stock, util


class IBHistoricalDataFetcher:
    """
    Fetches historical OHLCV data from Interactive Brokers.

    Attributes:
        ib: IB connection instance
        host: TWS/Gateway host address
        port: TWS/Gateway port number
        client_id: Unique client identifier
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 7497, client_id: int = 1):
        """
        Initialize the IBHistoricalDataFetcher.

        Args:
            host: TWS/Gateway host address (default: 127.0.0.1)
            port: TWS/Gateway port number (default: 7497 for TWS, 4002 for Gateway)
            client_id: Unique client identifier (default: 1)
        """
        self.ib = IB()
        self.host = host
        self.port = port
        self.client_id = client_id
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        """
        Set up logging configuration.

        Returns:
            Configured logger instance
        """
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    async def connect(self) -> bool:
        """
        Establish connection to Interactive Brokers TWS/Gateway.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            await self.ib.connectAsync(self.host, self.port, clientId=self.client_id)
            self.logger.info(f"Successfully connected to IB at {self.host}:{self.port}")
            return True
        except ConnectionRefusedError:
            self.logger.error("Connection refused. Ensure TWS/Gateway is running and API is enabled.")
            return False
        except Exception as e:
            self.logger.error(f"Connection error: {e}")
            return False

    async def disconnect(self) -> None:
        """Disconnect from Interactive Brokers with proper cleanup."""
        if self.ib.isConnected():
            await asyncio.sleep(1)  # Allow pending data to flush
            self.ib.disconnect()
            self.logger.info("Disconnected from IB")

    async def fetch_historical_data(
        self,
        symbol: str,
        exchange: str = "SMART",
        currency: str = "USD",
        duration: str = "1 Y",
        bar_size: str = "1 day",
        use_rth: bool = True,
        end_datetime: str = "",
    ) -> Optional[pd.DataFrame]:
        """
        Fetch historical OHLCV data for a single stock.

        Args:
            symbol: Stock ticker symbol
            exchange: Exchange (default: SMART for automatic routing)
            currency: Currency (default: USD)
            duration: Duration string (e.g., '1 Y', '6 M', '30 D')
            bar_size: Bar size (e.g., '1 day', '1 hour', '5 mins', '1 min')
            use_rth: Use regular trading hours only (default: True)
            end_datetime: End date/time for historical data (default: "" for now)

        Returns:
            DataFrame with OHLCV data or None if error occurs
        """
        try:
            contract = Stock(symbol, exchange, currency)

            # Qualify the contract to ensure it's valid
            qualified = await self.ib.qualifyContractsAsync(contract)
            if not qualified:
                self.logger.error(f"Could not qualify contract for {symbol}")
                return None

            self.logger.debug(f"Fetching historical data for {symbol}...")

            # Request historical data
            bars = await self.ib.reqHistoricalDataAsync(
                contract,
                endDateTime=end_datetime,
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow="TRADES",
                useRTH=use_rth,
                formatDate=1,
            )

            if not bars:
                self.logger.warning(f"No data returned for {symbol}")
                return None

            # Convert to DataFrame
            df = util.df(bars)
            df.columns = df.columns.str.lower()

            # Set date column as index if it's not already
            if not isinstance(df.index, pd.DatetimeIndex):
                if "date" in df.columns:
                    df = df.set_index("date")
                else:
                    self.logger.error(f"No date column found for {symbol}")
                    return None

            # Ensure datetime index
            df.index = pd.to_datetime(df.index)

            self.logger.debug(f"Successfully fetched {len(df)} bars for {symbol}")
            return df

        except Exception as e:
            self.logger.error(f"Error fetching data for {symbol}: {e}")
            return None

    async def fetch_multiple_stocks(
        self,
        symbols: list[str],
        exchange: str = "SMART",
        currency: str = "USD",
        duration: str = "1 Y",
        bar_size: str = "1 day",
        batch_size: int = 200,
        delay_between_batches: float = 2.0,
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical data for multiple stocks asynchronously with batching.

        Args:
            symbols: List of stock ticker symbols
            exchange: Exchange (default: SMART)
            currency: Currency (default: USD)
            duration: Duration string
            bar_size: Bar size
            batch_size: Number of symbols to process per batch (default: 200)
            delay_between_batches: Delay in seconds between batches (default: 2.0)

        Returns:
            Dictionary mapping symbols to their DataFrames
        """
        data_dict = {}
        total_symbols = len(symbols)

        # Split symbols into batches
        for batch_num, i in enumerate(range(0, total_symbols, batch_size), 1):
            batch = symbols[i: i + batch_size]
            batch_end = min(i + batch_size, total_symbols)

            self.logger.info(f"Processing batch {batch_num} ({i + 1}-{batch_end} of {total_symbols} symbols)...")

            # Process current batch
            tasks = [self.fetch_historical_data(symbol, exchange, currency, duration, bar_size) for symbol in batch]
            results = await asyncio.gather(*tasks)

            # Build dictionary for this batch
            for symbol, df in zip(batch, results):
                if df is not None:
                    if not df.empty:
                        data_dict[symbol] = df
                    else:
                        self.logger.warning(f"Empty DataFrame returned for {symbol}")

            self.logger.info(f"Batch {batch_num} complete: {len([r for r in results if r is not None])} successful out of {len(batch)}")

            # Delay between batches (except after the last batch)
            if batch_end < total_symbols:
                self.logger.debug(f"Waiting {delay_between_batches}s before next batch...")
                await asyncio.sleep(delay_between_batches)

        if not data_dict:
            self.logger.error("No data fetched for any symbols")
        else:
            self.logger.info(f"Total: {len(data_dict)} symbols successfully fetched out of {total_symbols}")

        return data_dict

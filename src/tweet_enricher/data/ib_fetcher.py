"""Interactive Brokers data fetcher.

This module handles all interactions with Interactive Brokers TWS/Gateway
for fetching historical OHLCV data.
"""

import asyncio
import logging
from typing import Callable, Dict, List, Optional

import pandas as pd
from ib_async import IB, Stock, util

from tweet_enricher.config import (
    DEFAULT_BATCH_DELAY,
    DEFAULT_BATCH_SIZE,
    DEFAULT_CLIENT_ID,
    DEFAULT_IB_HOST,
    DEFAULT_IB_PORT,
    INTRADAY_FETCH_DELAY,
    INTRADAY_TOTAL_DAYS,
)


class IBHistoricalDataFetcher:
    """
    Fetches historical OHLCV data from Interactive Brokers.

    Attributes:
        ib: IB connection instance
        host: TWS/Gateway host address
        port: TWS/Gateway port number
        client_id: Unique client identifier
    """

    def __init__(
        self,
        host: str = DEFAULT_IB_HOST,
        port: int = DEFAULT_IB_PORT,
        client_id: int = DEFAULT_CLIENT_ID,
    ):
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
        self.logger = logging.getLogger(__name__)

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

    async def validate_contracts(
        self,
        symbols: List[str],
        exchange: str = "SMART",
        currency: str = "USD",
        batch_size: int = 50,
    ) -> tuple[List[str], List[str]]:
        """
        Validate which symbols have valid IB contracts.

        Args:
            symbols: List of stock ticker symbols to validate
            exchange: Exchange (default: SMART)
            currency: Currency (default: USD)
            batch_size: Number of contracts to validate per batch

        Returns:
            Tuple of (valid_symbols, invalid_symbols)
        """
        valid_symbols: List[str] = []
        invalid_symbols: List[str] = []
        total = len(symbols)

        if total == 0:
            return valid_symbols, invalid_symbols

        self.logger.info(f"Validating {total} contracts...")

        # Process in batches
        for batch_start in range(0, total, batch_size):
            batch_end = min(batch_start + batch_size, total)
            batch_symbols = symbols[batch_start:batch_end]

            # Create contracts for this batch
            contracts = [Stock(symbol, exchange, currency) for symbol in batch_symbols]

            # Qualify all contracts in batch
            try:
                qualified = await self.ib.qualifyContractsAsync(*contracts)

                # Check which contracts were successfully qualified
                for symbol, contract in zip(batch_symbols, contracts):
                    if contract.conId and contract.conId > 0:
                        valid_symbols.append(symbol)
                    else:
                        invalid_symbols.append(symbol)
                        self.logger.debug(f"Invalid contract: {symbol}")

            except Exception as e:
                self.logger.warning(f"Error validating batch: {e}")
                # Mark entire batch as invalid on error
                invalid_symbols.extend(batch_symbols)

            # Small delay between batches
            if batch_end < total:
                await asyncio.sleep(0.5)

        self.logger.info(f"Validation complete: {len(valid_symbols)} valid, {len(invalid_symbols)} invalid")
        return valid_symbols, invalid_symbols

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
        symbols: List[str],
        exchange: str = "SMART",
        currency: str = "USD",
        duration: str = "1 Y",
        bar_size: str = "1 day",
        use_rth: bool = True,
        batch_size: int = DEFAULT_BATCH_SIZE,
        delay_between_batches: float = DEFAULT_BATCH_DELAY,
        on_batch_complete: Optional[Callable[[Dict[str, pd.DataFrame]], None]] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical data for multiple stocks with batching.

        Args:
            symbols: List of stock ticker symbols
            exchange: Exchange (default: SMART)
            currency: Currency (default: USD)
            duration: Duration string (e.g., '1 Y', '6 M', '30 D')
            bar_size: Bar size (e.g., '1 day', '1 hour', '5 mins')
            use_rth: Use regular trading hours only (default: True)
            batch_size: Number of symbols per batch (default: 50)
            delay_between_batches: Delay in seconds between batches (default: 2.0)
            on_batch_complete: Optional callback called after each batch with results

        Returns:
            Dictionary mapping symbols to their DataFrames
        """
        results: Dict[str, pd.DataFrame] = {}
        total = len(symbols)

        if total == 0:
            return results

        total_batches = (total + batch_size - 1) // batch_size

        for batch_start in range(0, total, batch_size):
            batch_end = min(batch_start + batch_size, total)
            batch_symbols = symbols[batch_start:batch_end]
            batch_num = (batch_start // batch_size) + 1

            self.logger.info(f"Batch {batch_num}/{total_batches}: Fetching {len(batch_symbols)} symbols...")

            # Create and execute tasks in parallel
            tasks = [
                self.fetch_historical_data(
                    symbol=symbol,
                    exchange=exchange,
                    currency=currency,
                    duration=duration,
                    bar_size=bar_size,
                    use_rth=use_rth,
                )
                for symbol in batch_symbols
            ]
            batch_results = await asyncio.gather(*tasks)

            # Process results
            batch_data: Dict[str, pd.DataFrame] = {}
            successful = 0
            failed = 0

            for symbol, df in zip(batch_symbols, batch_results):
                if df is not None and not df.empty:
                    results[symbol] = df
                    batch_data[symbol] = df
                    successful += 1
                else:
                    failed += 1

            # Log batch status
            if failed > 0:
                self.logger.warning(f"Batch {batch_num}/{total_batches}: {successful} success, {failed} failed")
            else:
                self.logger.info(f"Batch {batch_num}/{total_batches}: {successful} success")

            # Call callback for immediate processing (e.g., save to disk)
            if on_batch_complete and batch_data:
                on_batch_complete(batch_data)

            # Delay between batches (except after the last batch)
            if batch_end < total:
                await asyncio.sleep(delay_between_batches)

        self.logger.info(f"Total: {len(results)}/{total} symbols fetched successfully")
        return results

    async def fetch_multiple_stocks_intraday(
        self,
        symbols: List[str],
        total_days: int = INTRADAY_TOTAL_DAYS,
        bar_size: str = "15 mins",
        exchange: str = "SMART",
        currency: str = "USD",
        use_rth: bool = False,
        delay_between_symbols: float = INTRADAY_FETCH_DELAY,
        on_symbol_complete: Optional[Callable[[str, pd.DataFrame], None]] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch intraday historical data for multiple stocks sequentially.

        Fetches data in single 200-day requests per symbol. IB throttles parallel
        requests, so sequential fetching is most efficient.

        Args:
            symbols: List of stock ticker symbols
            total_days: Days of history to fetch (default: 200)
            bar_size: Bar size (default: 15 mins)
            exchange: Exchange (default: SMART)
            currency: Currency (default: USD)
            use_rth: Use regular trading hours only (default: False for extended hours)
            delay_between_symbols: Seconds between symbol requests (default: 2.0)
            on_symbol_complete: Callback called after each symbol with (symbol, df)

        Returns:
            Dictionary mapping symbols to their DataFrames
        """
        results: Dict[str, pd.DataFrame] = {}
        total = len(symbols)

        if total == 0:
            return results

        successful = 0
        failed = 0

        for i, symbol in enumerate(symbols):
            self.logger.info(f"Symbol {i + 1}/{total}: {symbol}")

            df = await self.fetch_historical_data(
                symbol=symbol,
                exchange=exchange,
                currency=currency,
                duration=f"{total_days} D",
                bar_size=bar_size,
                use_rth=use_rth,
            )

            if df is not None and not df.empty:
                results[symbol] = df
                successful += 1
                if on_symbol_complete:
                    on_symbol_complete(symbol, df)
            else:
                failed += 1

            # Delay between symbols (except after the last one)
            if i < total - 1:
                await asyncio.sleep(delay_between_symbols)

        self.logger.info(f"Total: {successful}/{total} symbols fetched successfully, {failed} failed")
        return results

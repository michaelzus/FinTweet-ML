# Interactive Brokers Historical Data Fetcher

A modular Python application that connects to Interactive Brokers TWS/Gateway and fetches historical OHLCV (Open, High, Low, Close, Volume) data for multiple stocks using asynchronous programming.

## Features

- ✨ Asynchronous data fetching for multiple stocks
- 🔒 Robust error handling and connection management
- 📊 Exports data to CSV format (separate file per ticker)
- ⚙️ Configurable bar sizes and durations
- 🚀 Efficient batched data requests to avoid rate limiting
- 📈 Built-in S&P 500 and Russell 1000 ticker list fetching
- 🔧 Modular architecture for easy integration and testing
- ⚡ Automatic rate limiting with configurable batch sizes

## Prerequisites

1. **Interactive Brokers Account**: You need an active IB account
2. **TWS or IB Gateway**: Must be running and configured
3. **API Settings**: Enable API connections in TWS/Gateway
   - TWS: File → Global Configuration → API → Settings
   - Enable "ActiveX and Socket Clients"
   - Note your Socket Port (default: 7497 for TWS, 4002 for Gateway)

## Project Structure

```
TimeWaste2/
├── fetch_historical_data.py  # CLI entry point
├── ib_fetcher.py             # Interactive Brokers client
├── helpers.py                # Utility functions (S&P 500, CSV saving)
├── verify_connection.py      # Connection verification tool
├── requirements.txt
├── pyproject.toml
└── README.md
```

### Modules

- **`fetch_historical_data.py`**: Command-line interface for fetching data
- **`ib_fetcher.py`**: `IBHistoricalDataFetcher` class for IBKR interactions with batching support
- **`helpers.py`**: Utility functions (`fetch_sp500_tickers()`, `fetch_russell1000_tickers()`, `save_to_csv()`, `filter_tickers_by_volume()`)
- **`filter_by_volume.py`**: Standalone script to filter tickers by trading volume

## Installation

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Usage

Fetch 1 year of daily data for Apple and Microsoft:

```bash
source venv/bin/activate
python fetch_historical_data.py --symbols AAPL MSFT
```

### Fetch S&P 500 Data

Fetch data for all S&P 500 stocks:

```bash
python fetch_historical_data.py --sp500 --duration "1 M" --bar-size "1 day"
```

### Fetch Russell 1000 Data

Fetch data for all Russell 1000 stocks:

```bash
python fetch_historical_data.py --russell1000 --duration "1 M" --bar-size "1 day"
```

### Fetch Both S&P 500 and Russell 1000

Fetch data for both S&P 500 and Russell 1000 stocks combined (automatically deduplicated):

```bash
python fetch_historical_data.py --all --duration "1 M" --bar-size "1 day"
```

**Note**: Fetching large lists of stocks (500-1000+) will take time. The script automatically batches requests (default: 200 stocks per batch with 2-second delays) to avoid IBKR rate limits. You can customize batching with `--batch-size` and `--batch-delay` arguments.

### Advanced Usage

Fetch 6 months of hourly data for multiple stocks:

```bash
python fetch_historical_data.py \
  --symbols AAPL MSFT GOOGL TSLA \
  --duration "6 M" \
  --bar-size "1 hour" \
  --output-dir my_data
```

### Command Line Arguments

| Argument | Description | Default | Example |
|----------|-------------|---------|---------|
| `--symbols` | Stock ticker symbols | - | `AAPL MSFT GOOGL` |
| `--sp500` | Fetch all S&P 500 stocks | `False` | `--sp500` |
| `--russell1000` | Fetch all Russell 1000 stocks | `False` | `--russell1000` |
| `--all` | Fetch both S&P 500 and Russell 1000 | `False` | `--all` |
| `--duration` | Historical data duration | `1 Y` | `6 M`, `30 D`, `1 W` |
| `--bar-size` | Bar size/timeframe | `1 day` | `1 hour`, `5 mins`, `1 min` |
| `--output-dir` | Output directory for CSV files | `data` | `my_stocks` |
| `--batch-size` | Symbols per batch | `200` | `50`, `100`, `300` |
| `--batch-delay` | Delay between batches (seconds) | `2.0` | `1.0`, `5.0` |
| `--host` | TWS/Gateway host | `127.0.0.1` | `192.168.1.100` |
| `--port` | TWS/Gateway port | `7497` | `4002` |
| `--client-id` | Unique client ID | `1` | `2` |
| `--exchange` | Exchange | `SMART` | `NYSE`, `NASDAQ` |
| `--currency` | Currency | `USD` | `EUR`, `GBP` |

**Note**: One of `--symbols`, `--sp500`, `--russell1000`, or `--all` must be provided (but not multiple).

### Duration Strings

Valid duration formats:
- `S` - Seconds
- `D` - Days
- `W` - Weeks
- `M` - Months
- `Y` - Years

Examples: `30 S`, `1 D`, `2 W`, `6 M`, `1 Y`

### Bar Size Options

Common bar sizes:
- `1 sec`, `5 secs`, `15 secs`, `30 secs`
- `1 min`, `2 mins`, `3 mins`, `5 mins`, `15 mins`, `30 mins`
- `1 hour`, `2 hours`, `3 hours`, `4 hours`, `8 hours`
- `1 day`, `1 week`, `1 month`

## Output Format

The script generates **separate CSV files per ticker** in the specified output directory.

Each CSV file contains the following columns:

| Column | Description |
|--------|-------------|
| `date` | Date/time of the bar |
| `open` | Opening price |
| `high` | Highest price |
| `low` | Lowest price |
| `close` | Closing price |
| `volume` | Trading volume |
| `average` | Average price |
| `barCount` | Number of trades |

**Example output structure:**
```
data/
  ├── AAPL.csv
  ├── MSFT.csv
  └── GOOGL.csv
```

## Examples

### Example 1: Daily Data for Tech Stocks

```bash
python fetch_historical_data.py \
  --symbols AAPL MSFT GOOGL AMZN META \
  --duration "1 Y" \
  --bar-size "1 day" \
  --output-dir tech_stocks
```

### Example 2: Intraday 5-Minute Bars

```bash
python fetch_historical_data.py \
  --symbols SPY QQQ IWM \
  --duration "5 D" \
  --bar-size "5 mins" \
  --output-dir intraday_data
```

### Example 3: Fetch All S&P 500 Stocks

```bash
python fetch_historical_data.py \
  --sp500 \
  --duration "1 M" \
  --bar-size "1 day" \
  --output-dir sp500_data
```

### Example 4: Using IB Gateway

```bash
python fetch_historical_data.py \
  --symbols AAPL \
  --port 4002 \
  --duration "1 M" \
  --bar-size "1 hour"
```

### Example 5: Fetch Russell 1000 with Custom Batching

```bash
# Use smaller batches and longer delays for more conservative rate limiting
python fetch_historical_data.py \
  --russell1000 \
  --duration "1 M" \
  --bar-size "1 day" \
  --batch-size 25 \
  --batch-delay 5.0 \
  --output-dir russell1000_data
```

### Example 6: Fetch Combined S&P 500 and Russell 1000

```bash
# Fetch both indices in one go (automatically deduplicated)
python fetch_historical_data.py \
  --all \
  --duration "1 M" \
  --bar-size "1 day" \
  --output-dir combined_data
```

## Programmatic Usage

The modular design allows you to use the components programmatically in your own scripts:

```python
import asyncio
from ib_fetcher import IBHistoricalDataFetcher
from helpers import save_to_csv, fetch_sp500_tickers

async def my_script():
    # Initialize fetcher
    fetcher = IBHistoricalDataFetcher(host="127.0.0.1", port=7497, client_id=1)
    
    # Connect
    connected = await fetcher.connect()
    if not connected:
        return
    
    try:
        # Fetch data for specific symbols
        symbols = ["AAPL", "MSFT", "GOOGL"]
        data_dict = await fetcher.fetch_multiple_stocks(
            symbols=symbols,
            duration="1 M",
            bar_size="1 day",
            batch_size=200,  # Process 200 symbols per batch (default)
            delay_between_batches=2.0  # Wait 2 seconds between batches
        )
        
        # Or fetch S&P 500 tickers
        # sp500_symbols = fetch_sp500_tickers()
        # Or fetch Russell 1000 tickers
        # russell1000_symbols = fetch_russell1000_tickers()
        
        # Save to CSV
        if data_dict:
            save_to_csv(data_dict, "my_output_dir")
            
            # Or process the data directly
            for symbol, df in data_dict.items():
                print(f"{symbol}: {len(df)} bars")
                # Your custom processing here
    
    finally:
        await fetcher.disconnect()

# Run the async function
asyncio.run(my_script())
```

## Troubleshooting

### Connection Refused Error

**Error**: `Connection refused. Ensure TWS/Gateway is running and API is enabled.`

**Solution**:
1. Ensure TWS or IB Gateway is running
2. Check API settings are enabled
3. Verify the correct port (7497 for TWS, 4002 for Gateway)
4. Ensure no firewall is blocking the connection

### No Data Returned

**Error**: `No data returned for SYMBOL`

**Solution**:
1. Verify the symbol is correct
2. Check you have market data permissions for that symbol
3. Try using `--exchange NYSE` or `--exchange NASDAQ` instead of SMART
4. Reduce the duration or change the bar size

### Contract Not Qualified

**Error**: `Could not qualify contract for SYMBOL`

**Solution**:
1. Verify the symbol exists and is tradable
2. Specify the correct exchange
3. Check currency is correct

### Timeout or Rate Limiting Issues

**Error**: `reqHistoricalData: Timeout for Stock(...)` or many failed requests

**Solution**:
1. Reduce batch size: `--batch-size 50` or `--batch-size 25`
2. Increase delay between batches: `--batch-delay 5.0`
3. Ensure you're not running other IBKR API clients simultaneously
4. Check your IBKR market data subscription includes the requested symbols

**Example with conservative rate limiting:**
```bash
python fetch_historical_data.py --russell1000 --batch-size 50 --batch-delay 3.0
```

**Note**: The default batch size is 200, which works well for most scenarios. If you encounter issues, try smaller batch sizes (50-100) with longer delays (3-5 seconds).

## Code Quality

This project follows these Python quality standards:
- **Black**: Code formatting (line length: 140)
- **Flake8**: Linting
- **MyPy**: Type checking
- **Pydocstyle**: Docstring style checking
- **Darglint**: Docstring validation

### Running Quality Checks

```bash
# Activate virtual environment
source venv/bin/activate

# Format code
black *.py

# Lint
flake8 *.py

# Type check
mypy *.py

# Check docstrings
pydocstyle *.py
```

## Architecture

The project follows a clean, modular architecture:

- **Separation of Concerns**: CLI, business logic, and utilities are separated
- **Reusability**: Core components can be imported and used in other scripts
- **Testability**: Each module can be tested independently
- **Maintainability**: Clear structure makes it easy to find and modify code
- **Extensibility**: Easy to add new data sources or export formats

### Module Dependencies

```
fetch_historical_data.py (CLI)
    ├── ib_fetcher.py (IBKR client)
    │   └── ib_async (external library)
    └── helpers.py (utilities)
        └── pandas (external library)
```

## License

MIT License

## Support

For issues related to:
- **Interactive Brokers API**: Check [IB API Documentation](https://interactivebrokers.github.io/)
- **ib_async Library**: Visit [ib_async GitHub](https://github.com/ib-api-reloaded/ib_async)

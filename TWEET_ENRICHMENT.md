# Tweet Enrichment System

## Overview

This system enriches tweet data with financial indicators and price information from Interactive Brokers. It's designed to prepare tweet data for machine learning models that predict price movements based on social media sentiment.

## Architecture

### Components

1. **`technical_indicators.py`** - OOP class for calculating technical indicators using pandas-ta
2. **`enrich_tweets.py`** - Main enrichment engine with market session awareness
3. **`test_enrichment.py`** - Demo/test script for single tweet enrichment
4. **`ib_fetcher.py`** - (Existing) IBKR data fetching infrastructure

### Data Flow

```
tweets.csv → TweetEnricher → IBKR API → Technical Indicators → Enriched Output
                    ↓
            Market Session Logic
            (premarket/regular/afterhours/closed)
```

## Features

### 📊 Calculated Features

| Feature | Description | Type |
|---------|-------------|------|
| `price_at_tweet` | Stock price at tweet time | Float |
| `return_1d` | 1-day return | Float (%) |
| `volatility_7d` | 7-day historical volatility | Float |
| `relative_volume` | Volume vs 20-day average | Float (ratio) |
| `rsi_14` | 14-period RSI | Float (0-100) |
| `distance_from_ma_20` | Distance from 20-day MA | Float (%) |
| `spy_return_1d` | S&P 500 1-day return (market context) | Float (%) |
| `price_1hr_after` | Price 1 hour after tweet | Float |
| `return_1hr_adjusted` | **Market-adjusted 1-hour return** | Float (%) |
| `label_5class` | Classification label | Categorical |

### 🕐 Market Session Handling

The system intelligently handles different trading sessions:

#### 1. Regular Hours (9:30 AM - 4:00 PM ET)
- Uses 1-minute intraday bars for precise pricing
- Calculates exact price at tweet timestamp

#### 2. Pre-Market (4:00 AM - 9:30 AM ET)
- Attempts to use pre-market intraday data
- Falls back to previous day's close if unavailable
- Flags data quality appropriately

#### 3. After-Hours (4:00 PM - 8:00 PM ET)
- Uses after-hours intraday data
- Falls back to regular session close if needed
- Maintains data quality flags

#### 4. Overnight & Weekends (8:00 PM - 4:00 AM, Sat-Sun)
- Uses previous trading day's close
- Flags as `market_closed`
- Not tradeable periods

### 🏷️ Classification Labels

5-class classification based on market-adjusted 1-hour returns:

| Label | Return Range | Description |
|-------|--------------|-------------|
| `strong_sell` | < -2% | Strong downward movement |
| `sell` | -2% to -0.5% | Moderate downward movement |
| `hold` | -0.5% to +0.5% | Minimal movement |
| `buy` | +0.5% to +2% | Moderate upward movement |
| `strong_buy` | > +2% | Strong upward movement |

### 🎯 Market-Adjusted Returns

The system calculates **market-adjusted returns** by subtracting SPY (S&P 500) returns:

```python
return_1hr_adjusted = return_1hr - spy_return_1d
```

This removes market-wide movements and isolates ticker-specific alpha.

## Installation

### Prerequisites

1. **Interactive Brokers TWS/Gateway** running and connected
2. **Python 3.11+**
3. **Required packages:**

```bash
pip install -r requirements.txt
```

### New Dependencies

Added to `requirements.txt`:
- `pandas-ta>=0.3.14b` - Technical indicators library

## Usage

### Quick Test

Test the enrichment on a single tweet:

```bash
python test_enrichment.py
```

### Full Enrichment

Process all tweets in your dataset:

```bash
python enrich_tweets.py
```

This will:
1. Read `output/tweets.csv`
2. Extract unique tickers and date ranges
3. Fetch required historical data (daily + intraday)
4. Calculate all technical indicators
5. Enrich each tweet with features
6. Output results

### Programmatic Usage

```python
import asyncio
from enrich_tweets import TweetEnricher
import pandas as pd

async def enrich_my_tweets():
    # Load tweets
    tweets_df = pd.read_csv("output/tweets.csv")
    
    # Initialize enricher
    enricher = TweetEnricher(host="127.0.0.1", port=7497)
    
    await enricher.connect()
    
    try:
        results = []
        for _, tweet_row in tweets_df.iterrows():
            result = await enricher.enrich_tweet(tweet_row)
            results.append(result)
            await asyncio.sleep(0.5)  # Rate limiting
        
        # Convert to DataFrame
        enriched_df = pd.DataFrame(results)
        enriched_df.to_csv("enriched_tweets.csv", index=False)
        
    finally:
        await enricher.disconnect()

asyncio.run(enrich_my_tweets())
```

## Data Quality Flags

The system provides data quality flags for transparency:

### Price at Tweet Flags
- `regular_intraday` - Exact price from regular hours 1-min bars
- `premarket_intraday` - Price from pre-market data
- `afterhours_intraday` - Price from after-hours data
- `no_premarket_data_used_daily` - Pre-market unavailable, used daily close
- `no_afterhours_data_used_daily` - After-hours unavailable, used daily close
- `market_closed_used_prev_close` - Market closed, used previous close

### Price 1hr After Flags
- `1hr_after_intraday` - Found exact 1-hour-later price
- `1hr_after_unavailable_used_close` - 1hr data unavailable, used close

## Configuration

### Customize Market Hours

Edit `enrich_tweets.py`:

```python
# Market hours definition (all in ET)
MARKET_OPEN = 9 * 60 + 30      # 9:30 AM
MARKET_CLOSE = 16 * 60          # 4:00 PM
PREMARKET_START = 4 * 60        # 4:00 AM
AFTERHOURS_END = 20 * 60        # 8:00 PM
```

### Customize Classification Thresholds

Edit `_classify_return()` method:

```python
def _classify_return(self, return_value: Optional[float]) -> Optional[str]:
    if return_value is None:
        return None
    
    if return_value < -0.02:        # < -2%
        return "strong_sell"
    elif return_value < -0.005:     # -2% to -0.5%
        return "sell"
    elif return_value < 0.005:      # -0.5% to 0.5%
        return "hold"
    elif return_value < 0.02:       # 0.5% to 2%
        return "buy"
    else:                           # > 2%
        return "strong_buy"
```

### Batch Processing

For large datasets, the enricher uses caching:

```python
# Cache for fetched data
self.daily_data_cache: Dict[str, pd.DataFrame] = {}
self.intraday_data_cache: Dict[str, pd.DataFrame] = {}
```

Daily data is cached per ticker, intraday data per ticker-date combination.

## Technical Indicators Details

### Return 1D
- **Formula**: `(close_today - close_yesterday) / close_yesterday`
- **Lookback**: 1 day
- **Use**: Recent momentum

### Volatility 7D
- **Formula**: `std(daily_returns_last_7_days)`
- **Lookback**: 7 days
- **Use**: Risk/uncertainty measure

### Relative Volume
- **Formula**: `current_volume / avg_volume_20d`
- **Lookback**: 20 days
- **Use**: Unusual activity detection

### RSI 14
- **Formula**: `100 - (100 / (1 + RS))` where RS = avg_gain / avg_loss
- **Lookback**: 14 periods
- **Use**: Overbought/oversold levels (>70 overbought, <30 oversold)

### Distance from MA 20
- **Formula**: `(current_close - sma_20) / sma_20`
- **Lookback**: 20 days
- **Use**: Trend strength and mean reversion

## Performance Considerations

### Rate Limiting
- Built-in delays between requests (configurable)
- Batch processing with caching
- Reuses daily data across multiple tweets for same ticker

### Data Efficiency
- Fetches intraday data in 2-day windows
- Caches all fetched data during session
- Minimizes redundant API calls

### IBKR Limitations
- Historical data pacing violations: max ~60 requests per 10 minutes
- Solution: Use `await asyncio.sleep(1)` between tweets
- For large datasets, consider breaking into batches

## Example Output

```
================================================================================
TWEET #1
================================================================================
  ticker                   : DHR
  timestamp                : 2025-10-21 02:16:00
  session                  : closed
  price_at_tweet           : 245.320000
  price_at_tweet_flag      : market_closed_used_prev_close
  return_1d                : 0.012500
  volatility_7d            : 0.018200
  relative_volume          : 1.120000
  rsi_14                   : 62.450000
  distance_from_ma_20      : 0.034500
  spy_return_1d            : 0.008000
  price_1hr_after          : 245.890000
  price_1hr_after_flag     : 1hr_after_unavailable_used_close
  return_1hr               : 0.002323
  return_1hr_adjusted      : -0.005677
  label_5class             : sell
```

## Troubleshooting

### Issue: "No security definition found"
- **Cause**: Ticker symbol invalid or not available in IBKR
- **Solution**: Verify ticker symbol, check if it's available for your market data subscription

### Issue: "No intraday data available"
- **Cause**: IBKR may not have extended hours data for all symbols
- **Solution**: System automatically falls back to daily data with appropriate flags

### Issue: "Connection refused"
- **Cause**: TWS/Gateway not running
- **Solution**: Start TWS or Gateway, ensure API is enabled in settings

### Issue: "Pacing violation"
- **Cause**: Too many requests too quickly
- **Solution**: Increase delays between requests, reduce batch size

## Best Practices

1. **Always check data quality flags** - Don't assume all prices are from exact timestamps
2. **Filter by session if needed** - Consider removing overnight/weekend tweets
3. **Monitor for missing values** - Some indicators require minimum lookback periods
4. **Use market-adjusted returns** - Better for ML models than raw returns
5. **Cache data when possible** - Reuse fetched data for multiple analyses

## Future Enhancements

Potential improvements:
- [ ] Support for multiple data sources (Alpaca, Polygon) for extended hours
- [ ] More technical indicators (MACD, Bollinger Bands, etc.)
- [ ] Sentiment analysis integration
- [ ] Real-time streaming mode
- [ ] Multi-timeframe analysis (5min, 15min, etc.)
- [ ] Sector/industry context features

## License

Part of the TimeWaste2 project.


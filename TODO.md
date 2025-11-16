# TODO - Future Improvements

## 🎯 Prioritization

- **P0** - Critical bugs or blockers
- **P1** - Important improvements (post-POC)
- **P2** - Nice to have
- **P3** - Future considerations

---

## 🏗️ Architecture Improvements

### P1: Refactor Caching Layer
**Status:** Deferred until multiple ML pipelines exist

**Current State:**
- Caching is implemented in `enrich_tweets.py`
- Date-aware cache: `cache[f"{symbol}_{date}"]`
- Works well but not reusable

**Proposed Solution:**
Create a separate caching layer:
```python
# cached_fetcher.py
class CachedDataFetcher:
    def __init__(self, fetcher: IBHistoricalDataFetcher, cache_strategy="date_aware"):
        self.fetcher = fetcher
        self.cache = {}
        self.strategy = cache_strategy
    
    async def fetch_daily(self, symbol, end_date):
        cache_key = self._make_cache_key(symbol, end_date, self.strategy)
        # ... caching logic
```

**Benefits:**
- Reusable across multiple ML pipelines
- `ib_fetcher.py` stays generic
- Clean separation of concerns
- Pluggable cache strategies

**When to implement:**
- When building 2nd or 3rd ML pipeline
- When cache strategy needs to be configurable

---

## 🚀 Performance Optimizations

### P2: Batch Intraday Data Fetching
**Status:** Not needed for POC

**Issue:**
- Currently fetches intraday data per tweet
- Could batch by date ranges for same ticker

**Proposed Solution:**
- Group tweets by ticker and date range
- Fetch larger intraday windows
- Slice data per tweet

**When to implement:**
- When processing 10k+ tweets
- When IBKR rate limiting becomes an issue

---

### P2: Parallel Tweet Processing
**Status:** Not needed for POC

**Current:** Sequential processing with `asyncio`
**Proposed:** True parallelization with `asyncio.gather()` in batches

```python
# Process tweets in parallel batches of 50
batch_size = 50
for i in range(0, len(tweets), batch_size):
    batch = tweets[i:i+batch_size]
    results = await asyncio.gather(*[enrich_tweet(t) for t in batch])
```

**When to implement:**
- When dataset > 10k tweets
- When speed becomes critical

---

## 📊 Feature Engineering

### P1: Multi-Timeframe Returns
**Status:** Infrastructure ready, not implemented

**Current:** Only 1hr return calculated
**Proposed:** Calculate multiple target variables

```python
# Already have generic method: get_price_n_hr_after(hours_after)
returns = {
    'return_15min': get_price_n_hr_after(ts, hours_after=0.25),
    'return_30min': get_price_n_hr_after(ts, hours_after=0.5),
    'return_1hr': get_price_n_hr_after(ts, hours_after=1.0),
    'return_2hr': get_price_n_hr_after(ts, hours_after=2.0),
    'return_4hr': get_price_n_hr_after(ts, hours_after=4.0),
    'return_24hr': get_price_n_hr_after(ts, hours_after=24.0),
}
```

**Benefits:**
- Train models for different timeframes
- Better strategy selection
- More comprehensive backtesting

**When to implement:**
- After POC shows promise
- When building strategy-specific models

---

### P2: Additional Technical Indicators
**Status:** Easy to add with pandas-ta

**Current Indicators:**
- return_1d, volatility_7d, relative_volume, rsi_14, distance_from_ma_20

**Potential Additions:**
```python
# Trend indicators
- MACD (Moving Average Convergence Divergence)
- ADX (Average Directional Index)
- Aroon

# Volatility indicators
- ATR (Average True Range)
- Bollinger Bands
- Keltner Channels

# Volume indicators
- OBV (On-Balance Volume)
- VWAP (Volume Weighted Average Price)
- MFI (Money Flow Index)

# Multi-timeframe
- RSI on different periods (7, 14, 21)
- MA distances (10, 20, 50, 200)
- Volatility over different windows
```

**When to implement:**
- After feature importance analysis
- When model needs more signals

---

### P2: Sentiment Features
**Status:** Future enhancement

**Proposed:**
```python
def extract_sentiment_signals(text: str) -> dict:
    return {
        'bullish_words': count_bullish_keywords(text),
        'bearish_words': count_bearish_keywords(text),
        'has_earnings': 'earnings' in text.lower(),
        'has_numbers': bool(re.search(r'\d+', text)),
        'text_length': len(text),
        'exclamation_count': text.count('!'),
        'sentiment_score': TextBlob(text).sentiment.polarity,
    }
```

**When to implement:**
- When text analysis becomes important
- After basic model is working

---

### P3: Sector/Market Context
**Status:** Future consideration

**Proposed Features:**
```python
- Sector performance (vs SPY)
- Industry group momentum
- Market regime (bull/bear/sideways)
- VIX level
- Correlation with sector ETF
```

**When to implement:**
- When building more sophisticated models
- When sector rotation matters

---

## 🔧 Code Quality

### P2: Unit Tests
**Status:** Not written yet

**Areas to Test:**
- `TechnicalIndicators` class methods
- Date-aware caching logic
- Market session detection
- Timezone normalization
- Return calculations

```python
# tests/test_technical_indicators.py
def test_calculate_return():
    tech = TechnicalIndicators()
    df = create_test_dataframe()
    
    # Test 1-day return
    ret = tech.calculate_return(df, idx=10, periods=1)
    assert abs(ret - expected) < 0.0001
    
    # Test 5-day return
    ret = tech.calculate_return(df, idx=10, periods=5)
    assert abs(ret - expected) < 0.0001
```

**When to implement:**
- Before production deployment
- When refactoring critical paths

---

### P2: Type Checking with mypy
**Status:** Partial (type hints exist)

**Current:** Type hints in code but not enforced
**Proposed:** Add mypy to CI/CD

```bash
mypy enrich_tweets.py ib_fetcher.py technical_indicators.py --strict
```

**When to implement:**
- Before team expansion
- Before production deployment

---

### P3: Integration Tests
**Status:** Future consideration

**Proposed:**
- End-to-end enrichment test with mock IBKR data
- Cache behavior tests
- Look-ahead bias validation tests

**When to implement:**
- When system is stable
- Before production

---

## 📈 Data Quality

### P1: Data Quality Monitoring
**Status:** Basic flags exist, no monitoring

**Current:** Data quality flags like `regular_intraday`, `market_closed_used_prev_close`
**Proposed:** Track and report data quality metrics

```python
quality_report = {
    'total_tweets': len(tweets),
    'missing_price_at_tweet': count_where(price_at_tweet is None),
    'missing_1hr_after': count_where(price_1hr_after is None),
    'used_fallback_daily': count_where('used_daily' in flag),
    'market_closed_tweets': count_where(session == 'closed'),
    'quality_score': calculate_overall_quality(),
}
```

**When to implement:**
- After first full dataset enrichment
- When evaluating model reliability

---

### P2: Handle Corporate Actions
**Status:** Not handled

**Issue:**
- Stock splits affect price continuity
- Dividends affect returns
- Mergers/acquisitions invalidate data

**Proposed:** Filter or adjust for corporate actions

**When to implement:**
- When seeing anomalies in data
- Before production backtesting

---

## 🗄️ Data Management

### P2: Database Integration
**Status:** Currently CSV-based

**Current:** Read from CSV, save to CSV
**Proposed:** Use database (SQLite, PostgreSQL, or TimescaleDB)

**Benefits:**
- Faster queries
- Better for large datasets
- Incremental updates
- Multi-user access

**When to implement:**
- When dataset > 100k tweets
- When needing real-time updates

---

### P2: Incremental Processing
**Status:** Always reprocesses everything

**Proposed:**
```python
# Track processed tweets
processed_ids = load_processed_ids()
new_tweets = [t for t in tweets if t.id not in processed_ids]
# Only process new tweets
```

**When to implement:**
- When adding new tweets regularly
- When reprocessing is too slow

---

## 🔍 Monitoring & Observability

### P2: Structured Logging
**Status:** Basic logging exists

**Proposed:**
- JSON structured logs
- Log aggregation (e.g., ELK stack)
- Performance metrics
- Error tracking (e.g., Sentry)

**When to implement:**
- When debugging becomes difficult
- Before production deployment

---

### P3: Metrics Dashboard
**Status:** No dashboard

**Proposed:**
- Grafana dashboard showing:
  - Processing speed
  - Error rates
  - Cache hit rates
  - Data quality metrics
  - IBKR API usage

**When to implement:**
- When system is production-critical
- When optimizing performance

---

## 📚 Documentation

### P2: API Documentation
**Status:** Docstrings exist, no generated docs

**Proposed:** Generate Sphinx or mkdocs documentation

**When to implement:**
- When onboarding new team members
- When API becomes stable

---

### P2: Usage Examples
**Status:** Basic test script exists

**Proposed:** Comprehensive examples for:
- Different timeframes
- Custom indicators
- Batch processing
- Error handling

**When to implement:**
- After POC validation
- When building user community

---

## 🌐 Infrastructure

### P3: Dockerization
**Status:** Not containerized

**Proposed:**
```dockerfile
FROM python:3.12
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "enrich_tweets.py"]
```

**When to implement:**
- When deploying to cloud
- When sharing with team

---

### P3: CI/CD Pipeline
**Status:** No automation

**Proposed:**
- GitHub Actions for:
  - Linting (flake8, mypy)
  - Testing (pytest)
  - Code formatting (black)
  - Documentation generation

**When to implement:**
- When team size > 1
- Before production

---

## 🎯 ML Pipeline Enhancements

### P1: Model Training Pipeline
**Status:** Not built yet

**Proposed:** Separate script for model training
```python
# train_model.py
- Load enriched tweets
- Feature engineering
- Train/validation/test split (temporal!)
- Model training
- Evaluation metrics
- Model persistence
```

**When to implement:**
- After POC data enrichment complete
- Next immediate priority

---

### P2: Backtesting Framework
**Status:** Not built yet

**Proposed:**
- Walk-forward validation
- Transaction costs
- Slippage modeling
- Position sizing
- Risk management

**When to implement:**
- After initial model is trained
- Before live trading

---

### P3: Real-Time Inference
**Status:** Batch processing only

**Proposed:**
- Stream tweets in real-time
- Real-time enrichment
- Model inference
- Trading signals

**When to implement:**
- After backtesting is successful
- When going live

---

## 🐛 Known Issues

### P0: None currently
All critical bugs fixed! ✅

### P1: None currently

### P2: Potential Issues to Monitor

**Timezone edge cases:**
- DST transitions
- Pre-market hours on market holidays
- International markets (if expanding beyond US)

**IBKR API issues:**
- Rate limiting on large datasets
- Connection timeouts
- Data gaps for low-volume stocks

**Data quality:**
- Extended hours data availability
- Price discrepancies between sources
- Corporate actions not adjusted

---

## 📋 Decision Log

### 2024-11-16: Caching Architecture
**Decision:** Keep caching in `enrich_tweets.py` for now
**Rationale:** POC stage, avoid over-engineering
**Revisit:** When building 2nd ML pipeline

### 2024-11-16: Look-Ahead Bias Fix
**Decision:** Implemented date-aware caching
**Impact:** Critical for ML model validity
**Status:** ✅ Fixed

### 2024-11-16: Generic Time Horizons
**Decision:** Made `get_price_n_hr_after()` generic
**Rationale:** Enables multi-timeframe analysis
**Status:** ✅ Implemented

---

## 🎓 Learning & Research

### P2: Research Topics
- Alternative data sources (news, options flow, social sentiment)
- Advanced ML techniques (transformers for time series)
- Risk management strategies
- Portfolio optimization
- Market microstructure

### P3: Competitor Analysis
- How do other sentiment trading systems work?
- What features do they use?
- What are their edge cases?

---

## Notes

- **Focus on POC first** - Don't over-engineer before validating concept
- **Iterate based on reality** - Let actual usage drive decisions
- **Document but defer** - Capture ideas without immediate implementation
- **Technical debt is okay** - If it's intentional and tracked

---

**Last Updated:** 2024-11-16
**Next Review:** After POC completion


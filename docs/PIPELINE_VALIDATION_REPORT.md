# Pipeline Validation Report

**Date**: December 19, 2025  
**Scope**: Tweet enrichment pipeline - timezone handling, data validation, incremental fetching

---

## Executive Summary

A comprehensive review of the tweet enrichment pipeline identified **1 critical issue** and **2 moderate issues** related to timezone handling, market session detection, and data consistency. The core data flow is sound, but the holiday detection issue needs fixing to ensure data integrity.

### Overall Assessment

| Component | Status | Notes |
|-----------|--------|-------|
| Tweet UTC → ET Conversion | ✅ Working | Correctly uses pytz for DST handling |
| IB Data Fetching | ✅ Working | Data correctly stored as `US/Eastern` timezone-aware |
| Session Detection | 🚨 Critical | Ignores market holidays (121 tweets mislabeled on Jan 1) |
| Incremental Sync | ⚠️ Minor | UTC/ET date boundary edge case |
| Data Quality Flags | ✅ Working | Properly identifies fallback prices |
| Label Reliability | ✅ Working | `is_reliable_label` flag correct (56.2% reliable) |

### Automated Test Results

```
======================== 28 passed in 4.50s ========================
- TestTwitterTimestampConversion: 5/5 passed
- TestStockDataTimezone: 4/4 passed  
- TestMarketSessionDetection: 5/5 passed
- TestTweetStockCrossReference: 3/3 passed
- TestIncrementalFetching: 3/3 passed
- TestDataConsistency: 3/3 passed
- TestEnrichmentOutput: 5/5 passed
```

---

## ✅ What's Working Correctly

### 1. Tweet UTC to Eastern Conversion
**Location**: `src/tweet_enricher/twitter/sync.py:51-72`

The `_convert_utc_to_eastern()` function correctly:
- Parses Twitter's UTC timestamp format (`"Wed Dec 17 15:01:11 +0000 2025"`)
- Converts to `America/New_York` timezone using pytz
- Automatically handles DST transitions

```python
def _convert_utc_to_eastern(self, twitter_ts: str) -> str:
    dt = datetime.strptime(twitter_ts, "%a %b %d %H:%M:%S %z %Y")
    eastern_tz = pytz.timezone("America/New_York")
    dt_eastern = dt.astimezone(eastern_tz)
    return dt_eastern.strftime("%Y-%m-%d %H:%M:%S")
```

### 2. Output Timestamps Include Timezone
The enriched CSV correctly outputs timestamps with timezone offset:
```
2025-01-01 04:15:34-05:00
```

### 3. Data Quality Flags
Entry/exit price flags provide transparency about data sources:
- `regular_next_bar_open` - Real intraday data
- `premarket_next_day_open` - Fallback to next day's open
- `1.0hr_after_unavailable_used_close` - Fallback to daily close

### 4. Label Reliability Detection
The `is_reliable_label` flag correctly identifies when labels are based on actual intraday prices vs fallbacks.

### 5. Incremental Sync Journal
The fetch journal (`fetch_journal` table) correctly prevents re-fetching already-processed days:
- Tracks `(account, date)` pairs that have been fetched
- Allows resuming interrupted syncs

---

## 🚨 Critical Issues

### Issue 1: Session Detection Ignores Market Holidays

**Location**: `src/tweet_enricher/market/session.py:59-86`

**Problem**: `get_market_session()` only checks time-of-day and weekday, NOT actual trading holidays.

**Current Code**:
```python
def get_market_session(timestamp: datetime) -> MarketSession:
    timestamp = normalize_timestamp(timestamp)

    # Check if it's a weekend
    if timestamp.weekday() >= 5:  # Saturday = 5, Sunday = 6
        return MarketSession.CLOSED

    # Convert to minutes since midnight
    minutes_since_midnight = timestamp.hour * 60 + timestamp.minute
    # ... only checks time after this
```

**Evidence from Data** (January 1, 2025 - New Year's Day, market CLOSED):

Automated validation found **121 tweets on Jan 1, 2025** with incorrect session labels:
| Session (Current) | Count | Session (Correct) |
|-------------------|-------|-------------------|
| `regular` | 87 | `closed` |
| `afterhours` | 29 | `closed` |
| `premarket` | 5 | `closed` |

**NYSE 2025 Holidays Affected**:
- Jan 1 (New Year's Day) - **121 tweets mislabeled**
- Jan 9 (National Day of Mourning)
- Jan 20 (MLK Day)
- Feb 17 (Presidents Day)
- Apr 18 (Good Friday)
- May 26 (Memorial Day)
- Jun 19 (Juneteenth)
- Jul 4 (Independence Day)
- Sep 1 (Labor Day)
- Nov 27 (Thanksgiving)
- Dec 25 (Christmas)

**Impact**: 
- ~11 market holidays per year labeled incorrectly
- Estimated 500-1000 tweets with wrong session labels
- Training data may have incorrect session labels
- Could affect model performance for session-based features

**Recommended Fix**:
```python
import pandas_market_calendars as mcal

_nyse = mcal.get_calendar("NYSE")
_schedule_cache = {}

def is_trading_day(date) -> bool:
    """Check if the given date is a trading day."""
    date_key = date.date() if hasattr(date, 'date') else date
    if date_key not in _schedule_cache:
        schedule = _nyse.schedule(start_date=date_key, end_date=date_key)
        _schedule_cache[date_key] = not schedule.empty
    return _schedule_cache[date_key]

def get_market_session(timestamp: datetime) -> MarketSession:
    timestamp = normalize_timestamp(timestamp)

    # Check if it's a trading day (includes weekends AND holidays)
    if not is_trading_day(timestamp):
        return MarketSession.CLOSED

    # Then check time of day...
    minutes_since_midnight = timestamp.hour * 60 + timestamp.minute
    # ... rest of existing logic
```

---

### ~~Issue 2: IB Data Timestamps Are Timezone-Naive~~ ✅ VERIFIED WORKING

**Location**: `src/tweet_enricher/data/ib_fetcher.py` + `src/tweet_enricher/data/cache.py`

**Status**: ✅ **NOT AN ISSUE** - Automated tests confirm data IS timezone-aware.

**Verification**:
```python
# Daily data
>>> df = pd.read_feather('data/daily/AAPL.feather').set_index('date')
>>> df.index.tz
<DstTzInfo 'US/Eastern' EST-1 day, 19:00:00 STD>

# Sample timestamps
DatetimeIndex(['2024-12-17 00:00:00-05:00', '2024-12-18 00:00:00-05:00', ...])

# Intraday data
>>> df = pd.read_feather('data/intraday/AAPL.feather').set_index('date')
>>> df.index.tz
<DstTzInfo 'US/Eastern' EST-1 day, 19:00:00 STD>

# Sample timestamps (15-min bars)
DatetimeIndex(['2025-03-06 04:00:00-05:00', '2025-03-06 04:15:00-05:00', ...])
```

**Why It Works**: The `normalize_dataframe_timezone()` function in `cache.py` correctly localizes all data to `US/Eastern` before saving to disk.

---

### Issue 3: Database Stores Eastern Timestamps Without TZ Indicator

**Location**: `src/tweet_enricher/twitter/sync.py:69`

**Problem**: Eastern time is stored as a plain string without timezone information:

```python
return dt_eastern.strftime("%Y-%m-%d %H:%M:%S")  # "2025-12-17 10:01:11" - no TZ!
```

**Impact**:
- If data is read without context, timezone is ambiguous
- Future code changes could misinterpret the timezone
- Makes debugging timezone issues harder

**Recommended Fix** (Option A - Add TZ abbreviation):
```python
return dt_eastern.strftime("%Y-%m-%d %H:%M:%S %Z")  # "2025-12-17 10:01:11 EST"
```

**Recommended Fix** (Option B - ISO format with offset):
```python
return dt_eastern.isoformat()  # "2025-12-17T10:01:11-05:00"
```

---

### Issue 4: Day-by-Day Sync Uses UTC for Date Calculation

**Location**: `src/tweet_enricher/twitter/sync.py:172-189`

**Problem**: Date calculations use UTC, but tweets are stored in Eastern time:

```python
now = datetime.now(pytz.UTC)  # UTC!
target_date = now - timedelta(days=day_offset)
since_date = target_date.strftime("%Y-%m-%d")  # This is a UTC date!
```

**Edge Case Example**:
- Tweet at 11:00 PM ET on Dec 15 = 4:00 AM UTC on Dec 16
- The sync might mark Dec 16 (UTC) as fetched
- But the tweet's `timestamp_et` is Dec 15
- Could cause tweets near midnight to be in wrong day's journal

**Recommended Fix**:
```python
eastern_tz = pytz.timezone("America/New_York")
now = datetime.now(eastern_tz)  # Use Eastern Time consistently
target_date = now - timedelta(days=day_offset)
since_date = target_date.strftime("%Y-%m-%d")
```

---

## ⚠️ Moderate Issues

### Issue 5: Missing Technical Indicators for Early Data

**Observation**: Many rows in the enriched CSV have empty values for:
- `rsi_14`
- `distance_from_ma_20`
- `above_ma_20`
- `slope_ma_20`

**Cause**: These indicators require lookback periods:
- RSI(14) needs 14+ days
- MA(20) needs 20 days
- MA slope needs 25 days (20 for MA + 5 for slope calculation)

**Assessment**: This is **expected behavior** for early data in the dataset. However:
- Should be documented
- Could add logging to warn when indicators can't be calculated
- Consider filling with neutral values or flagging these rows

---

### Issue 6: Holiday Data Uses Fallback Prices

**Observation**: All January 1, 2025 tweets use fallback prices:
- Entry: `premarket_next_day_open`, `regular_next_day_open`, `afterhours_next_day_open`
- Exit: `1.0hr_after_unavailable_used_close`

**Assessment**: This is **working correctly** - the system properly falls back when no intraday data exists. The `is_reliable_label=False` flag correctly indicates these labels shouldn't be trusted for training.

---

## 📊 Data Validation Checklist

### Verified ✓
- [x] Tweet timestamps convert correctly from UTC to Eastern (5 DST tests passed)
- [x] Output CSV includes timezone offset (`2025-01-01 04:15:34-05:00`)
- [x] IB data is timezone-aware (`US/Eastern`) - both daily and intraday
- [x] Fallback prices are flagged appropriately
- [x] `is_reliable_label` correctly identifies unreliable labels (56.2% reliable)
- [x] Incremental sync prevents duplicate fetches
- [x] Return calculations are mathematically correct

### Needs Fixing ✗
- [ ] Session detection should check trading calendar (121 Jan 1 tweets affected)
- [ ] Database timestamps should include TZ indicator (cosmetic)
- [ ] Day-by-day sync should use Eastern Time for date math (edge case)

---

## 📈 Automated Validation Results

### Dataset Statistics
```
Raw tweets:        65,288
Processed tweets:  60,734
Enriched rows:     34,899
Daily data files:  1,008 tickers
Intraday files:    475 tickers
```

### Label Reliability Analysis
```
                                              Reliable  Count  % Reliable
Entry Flag               Exit Flag                                        
regular_next_bar_open    1.0hr_after_intraday    9,081  9,081      100.0%
premarket_next_bar_open  1.0hr_after_intraday    6,380  6,380      100.0%
afterhours_next_bar_open 1.0hr_after_intraday    4,145  4,145      100.0%
closed_next_day_open     ...unavailable_close        0  6,582        0.0%
regular_next_day_open    ...unavailable_close        0  3,380        0.0%
premarket_next_day_open  ...unavailable_close        0  2,751        0.0%
afterhours_next_day_open ...unavailable_close        0  1,690        0.0%
```

**Total reliable labels**: 19,606 / 34,899 = **56.2%**

### Session Distribution
```
regular:     12,461 (35.7%)
premarket:    9,131 (26.2%)
closed:       6,970 (20.0%)
afterhours:   6,319 (18.1%)
```

### Label Distribution (3-class)
```
HOLD: 13,822 (39.6%)
SELL: 11,083 (31.8%)
BUY:   9,972 (28.6%)
```

### Extreme Returns Check
Found **5 rows** with |return_1hr| > 50% - all correctly marked as `is_reliable_label=False`:
- BILL (Feb 7, 2025): +52.8% - earnings reaction, fallback prices
- SRPT (May 6, 2025): +59.2% - afterhours gap, fallback prices
- TTD (Aug 8, 2025): +64.2% - overnight gap, fallback prices

---

## 🔧 Implementation Priority

### High Priority (Data Correctness)
1. **Fix Session Detection** - Affects ~500-1000 tweets on market holidays

### Medium Priority (Data Quality)
2. **Fix Day-by-Day Sync TZ** - Edge case for midnight tweets
3. **Fix DB Timestamp Format** - Improves debuggability

### Low Priority (Documentation)
4. **Document indicator lookback requirements**
5. **Add validation logging**

### ~~Resolved~~ ✅
- ~~IB Timezone~~ - Verified working correctly with `US/Eastern` timezone

---

## 📁 Files Requiring Changes

| File | Priority | Issue |
|------|----------|-------|
| `src/tweet_enricher/market/session.py` | High | Add market calendar check |
| `src/tweet_enricher/twitter/sync.py:69` | Medium | Add TZ to stored timestamp |
| `src/tweet_enricher/twitter/sync.py:172` | Medium | Use ET for date calculations |

---

## 🧪 Implemented Validation Tests

The following tests are now implemented in `tests/test_pipeline_validation.py`:

```python
# Twitter Timestamp Conversion Tests (5 tests)
def test_utc_to_eastern_basic_conversion(): ...
def test_utc_to_eastern_summer_dst(): ...      # EDT (UTC-4)
def test_utc_to_eastern_winter_standard(): ... # EST (UTC-5)
def test_utc_to_eastern_dst_transition_spring(): ...  # March DST start
def test_utc_to_eastern_dst_transition_fall(): ...    # November DST end

# Stock Data Timezone Tests (4 tests)
def test_daily_data_has_eastern_timezone(): ...   # ✅ PASSED
def test_intraday_data_has_eastern_timezone(): ...# ✅ PASSED
def test_intraday_bars_cover_extended_hours(): ...
def test_daily_bar_timestamps_at_midnight(): ...

# Market Session Detection Tests (5 tests)
def test_regular_hours_detection(): ...    # 9:30 AM - 4:00 PM
def test_premarket_detection(): ...        # 4:00 AM - 9:30 AM
def test_afterhours_detection(): ...       # 4:00 PM - 8:00 PM
def test_closed_detection_overnight(): ... # Before 4 AM, after 8 PM
def test_closed_detection_weekend(): ...   # Saturday, Sunday

# NOTE: Holiday test not yet added - requires fix to session.py first
def test_session_detection_holidays():  # TODO
    """Will verify holidays are detected as CLOSED after fix."""
    pass
```

**Run all tests**: `pytest tests/test_pipeline_validation.py -v`

---

## Appendix: Sample Data Analysis

### Holiday Tweet Example (Jan 1, 2025)
```csv
timestamp,ticker,session,entry_price_flag,exit_price_1hr_flag,is_reliable_label
2025-01-01 04:15:34-05:00,TSLA,premarket,premarket_next_day_open,1.0hr_after_unavailable_used_close,False
2025-01-01 11:39:33-05:00,PLTR,regular,regular_next_day_open,1.0hr_after_unavailable_used_close,False
2025-01-01 17:07:01-05:00,AXP,afterhours,afterhours_next_day_open,1.0hr_after_unavailable_used_close,False
```

**Issue**: `session` should be `closed` for all these rows since Jan 1 is a market holiday.

### Regular Trading Day Example (Jan 2, 2025)
```csv
timestamp,ticker,session,entry_price_flag,exit_price_1hr_flag,is_reliable_label
2025-01-02 04:01:12-05:00,U,premarket,premarket_next_day_open,1.0hr_after_unavailable_used_close,False
```

**Note**: Even on trading days, premarket tweets fall back to next day's open because intraday data may not be available for extended hours.

---

## 🧪 Validation Test Files Created

### `tests/test_pipeline_validation.py`
Comprehensive pytest test suite with 28 tests covering:
- Twitter UTC → Eastern conversion (including DST transitions)
- Stock data timezone verification
- Market session detection
- Tweet-stock data alignment
- Incremental fetching journal
- Data consistency checks
- Enriched output validation

**Run with**: `pytest tests/test_pipeline_validation.py -v`

### `scripts/validate_dataset.py`
Interactive validation script for manual inspection:

**Run with**: `python scripts/validate_dataset.py --sample 20`

Sample output:
```
================================================================================
VALIDATION SUMMARY
================================================================================
  ✓ PASS: Twitter Timestamp Conversion
  ✓ PASS: Stock Data Timezone
  ✓ PASS: Market Session Detection
  ✗ FAIL: Tweet-Stock Alignment (some tweets predate intraday data)
  ✓ PASS: Enriched Output Quality
  ✓ PASS: Incremental Fetching
================================================================================
```

---

*Report generated by pipeline validation analysis*
*Test suite: 28 tests passing*


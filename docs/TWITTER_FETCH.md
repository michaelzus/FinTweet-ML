# Twitter Data Collection System

This document describes the TwitterAPI.io integration for collecting and storing tweets locally with incremental sync support.

## Overview

The system fetches tweets from curated Twitter accounts focused on stock market news, stores them in SQLite with full raw data preservation, and exports in a format compatible with the tweet enricher pipeline.

**Key Features:**
- Incremental sync with smart cost optimization
- Filters out replies at API level to save costs
- UTC → Eastern Time conversion for market alignment
- Automatic ticker extraction and categorization
- Full raw JSON storage for future feature extraction

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         CLI Commands                            │
│   tweet-enricher twitter sync|status|export                     │
└─────────────────────┬───────────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────────┐
│                      SyncService                                 │
│   - Orchestrates fetch → process → store                        │
│   - Smart incremental logic                                      │
│   - Reuses MessageCategorizer & MessageProcessor                │
└─────────────────────┬───────────────────────────────────────────┘
                      │
        ┌─────────────┴─────────────┐
        │                           │
┌───────▼───────┐          ┌────────▼────────┐
│ TwitterClient │          │  TweetDatabase  │
│               │          │                 │
│ - Rate limit  │          │ - sync_state    │
│ - Pagination  │          │ - tweets_raw    │
│ - API calls   │          │ - tweets_proc   │
└───────┬───────┘          └────────┬────────┘
        │                           │
        ▼                           ▼
   TwitterAPI.io              SQLite (data/tweets.db)
```

## Configuration

Settings in `src/tweet_enricher/config.py`:

```python
# TwitterAPI.io settings
TWITTER_API_KEY = os.environ.get("TWITTER_API_KEY", "")
TWITTER_DB_PATH = Path("data/tweets.db")
TWITTER_RATE_LIMIT_DELAY = 5.0  # seconds between requests (free tier)
TWITTER_ACCOUNTS = [
    "StockMKTNewz",
    "wallstengine",
    "amitisinvesting",
    "AIStockSavvy",
    "fiscal_ai",
    "EconomyApp",
]
```

### Environment Variable

Set your API key:
```bash
export TWITTER_API_KEY="your_api_key_here"
```

## CLI Commands

### Sync Tweets

```bash
# Sync all configured accounts
tweet-enricher twitter sync

# Sync specific account
tweet-enricher twitter sync --account StockMKTNewz

# Full re-sync (ignore existing data, fetch everything)
tweet-enricher twitter sync --full

# Limit tweets per account
tweet-enricher twitter sync --max-tweets 100

# Verbose output
tweet-enricher twitter sync -v
```

### Check Status

```bash
tweet-enricher twitter status
```

Output:
```
============================================================
TWITTER SYNC STATUS
============================================================

Database Statistics:
  Total rows:      5
  Unique tweets:   4
  Unique tickers:  5
  Unique authors:  1
  Date range:      2025-12-17 10:32:52 to 2025-12-17 11:11:17

Account Status:
------------------------------------------------------------
  Account              Last Sync                Tweets
------------------------------------------------------------
  @StockMKTNewz        2025-12-17T16:18:58           5
  @wallstengine        Never                         0
  ...
============================================================
```

### Export to CSV

```bash
# Export all tweets
tweet-enricher twitter export -o output/tweets.csv

# Filter by date range
tweet-enricher twitter export -o tweets.csv --since 2025-12-01 --until 2025-12-15

# Filter by account
tweet-enricher twitter export -o tweets.csv --account StockMKTNewz

# Filter by ticker
tweet-enricher twitter export -o tweets.csv --ticker NVDA
```

## Database Schema

Location: `data/tweets.db`

### sync_state
Tracks sync progress per account.

| Column | Type | Description |
|--------|------|-------------|
| account | TEXT (PK) | Twitter username |
| last_tweet_id | TEXT | Most recent tweet ID fetched |
| last_cursor | TEXT | API pagination cursor |
| last_sync_at | TEXT | ISO timestamp of last sync |
| total_tweets | INTEGER | Running count of processed tweets |

### tweets_raw
Stores complete API responses for debugging and future feature extraction.

| Column | Type | Description |
|--------|------|-------------|
| id | TEXT (PK) | Tweet ID |
| account | TEXT | Source account |
| json_data | TEXT | Full API response as JSON |
| fetched_at | TEXT | When tweet was fetched |

### tweets_processed
Ready-to-export tweets with extracted features.

| Column | Type | Description |
|--------|------|-------------|
| id | TEXT | Tweet ID |
| timestamp_utc | TEXT | Original UTC timestamp |
| timestamp_et | TEXT | Eastern Time (for market alignment) |
| author | TEXT | Twitter username |
| ticker | TEXT | Extracted ticker symbol |
| tweet_url | TEXT | Link to tweet |
| category | TEXT | Auto-categorized (Earnings, M&A, etc.) |
| text | TEXT | Cleaned tweet text |

**Primary Key:** (id, ticker) - one row per ticker mentioned in tweet

## Smart Incremental Sync

The system minimizes API costs by stopping pagination as soon as it detects existing data.

### How It Works

```
1. Get last_tweet_id from sync_state
2. Make API Call #1 (fetch 20 tweets)
3. Check FIRST tweet in response:
   - If exists in DB → STOP (no new data, cost: 1 API call)
   - If new → continue processing
4. Process tweets until hitting existing one
5. Found existing? → STOP (don't make more API calls)
6. More pages & all new? → Continue to next page
```

### Cost Comparison

| Scenario | Naive Approach | Smart Incremental |
|----------|---------------|-------------------|
| No new tweets | Fetch all, check each | **1 API call**, stop immediately |
| 5 new tweets (in first batch) | Multiple calls | **1 API call** |
| 100 new tweets | All pages fetched | Stops when hitting existing |

### Example Output

```
$ tweet-enricher twitter sync --account StockMKTNewz

Starting sync for @StockMKTNewz
Incremental sync (last tweet: 2001324330014376353)
First tweet 2001324330014376353 already exists - no new data
Sync complete for @StockMKTNewz: 0 raw, 0 processed (1 API calls)
```

## API Details

### Endpoint Used
```
GET https://api.twitterapi.io/twitter/user/last_tweets
```

### Parameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| userName | e.g., "StockMKTNewz" | Twitter username |
| count | 20 | Tweets per page |
| includeReplies | false | Filter out replies (saves cost) |
| cursor | (pagination) | For fetching next page |

### Rate Limiting
- Free tier: 1 request per 5 seconds
- Built-in delay enforced by `TwitterClient`

### Response Structure
```json
{
  "status": "success",
  "has_next_page": true,
  "next_cursor": "...",
  "data": {
    "tweets": [
      {
        "id": "2001324330014376353",
        "createdAt": "Wed Dec 17 15:01:11 +0000 2025",
        "text": "Google $GOOGL CEO...",
        "twitterUrl": "https://twitter.com/...",
        "author": {
          "userName": "StockMKTNewz",
          "name": "Stock Market News"
        },
        "entities": {
          "symbols": [{"text": "GOOGL"}]
        },
        "likeCount": 123,
        "retweetCount": 45,
        "replyCount": 12,
        "viewCount": 5000
      }
    ]
  }
}
```

## Timestamp Handling

Twitter API returns UTC timestamps:
```
"Wed Dec 17 15:01:11 +0000 2025"
```

System converts to Eastern Time for market alignment:
```
"2025-12-17 10:01:11"
```

This matches the existing enricher pipeline which uses Eastern Time for all market-related operations.

## Output Format

Export produces CSV compatible with the tweet enricher:

```csv
timestamp,author,ticker,tweet_url,category,text
2025-12-17 10:32:52,StockMKTNewz,NFLX,https://twitter.com/...,Breaking News,"Netflix $NFLX..."
2025-12-17 11:07:02,StockMKTNewz,AAPL,https://twitter.com/...,Other,"Apple $AAPL is now..."
2025-12-17 11:07:02,StockMKTNewz,NVDA,https://twitter.com/...,Other,"Apple $AAPL is now..."
```

Note: Tweets with multiple tickers create multiple rows (same as Discord converter).

## Categories

Tweets are auto-categorized using the same `MessageCategorizer` as Discord:

- Earnings
- Mergers & Acquisitions
- Guidance & Forecasts
- Regulatory & Legal
- Product Launch
- Market Data
- Breaking News
- Personnel Changes
- Stock Movements
- Macro & Economic
- Other

## Module Structure

```
src/tweet_enricher/twitter/
├── __init__.py       # Module exports
├── client.py         # TwitterAPI.io HTTP client
├── database.py       # SQLite manager
└── sync.py           # Incremental sync service
```

## Future Enhancements

### Potential Additions

1. **Engagement Metrics**: Extract `likeCount`, `retweetCount`, `viewCount` from raw JSON as features

2. **Cashtag Search**: Add endpoint for searching by `$TICKER` directly (broader coverage, more noise)

3. **Historical Backfill**: Support fetching 6+ months of history for ML training

4. **Webhook Support**: Real-time updates instead of polling

### Data Already Captured

The `tweets_raw` table stores complete API responses, so additional fields can be extracted later without re-fetching:

```python
import sqlite3, json
conn = sqlite3.connect('data/tweets.db')
for row in conn.execute('SELECT json_data FROM tweets_raw'):
    tweet = json.loads(row[0])
    likes = tweet.get('likeCount', 0)
    retweets = tweet.get('retweetCount', 0)
    # Use for new features...
```

## Troubleshooting

### Connection Timeout
The API may timeout from certain networks. Try:
1. Run directly from terminal (not IDE sandbox)
2. Check if IP is rate-limited
3. Increase timeout in `client.py` (default: 60s)

### Missing API Key
```
Error: Twitter API key not provided. Set TWITTER_API_KEY environment variable
```

Solution:
```bash
export TWITTER_API_KEY="your_key_here"
```

### SSL Warnings
SSL verification is disabled due to known API issues. Warnings are suppressed automatically.


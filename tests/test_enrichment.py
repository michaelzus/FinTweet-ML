"""
Quick test script for tweet enrichment.

This script demonstrates the enrichment functionality on a single tweet.
"""

import asyncio
import logging

import pandas as pd

from tweet_enricher.core.enricher import TweetEnricher
from tweet_enricher.core.indicators import TechnicalIndicators
from tweet_enricher.data.cache import DataCache
from tweet_enricher.data.ib_fetcher import IBHistoricalDataFetcher


async def main():
    """Test enrichment on a single tweet."""
    print("=" * 80)
    print("TWEET ENRICHMENT TEST")
    print("=" * 80)
    print()

    # Create a test tweet
    test_tweet = pd.Series(
        {
            "timestamp": "2025-10-21 11:06:00",
            "author": "Wall St Engine • TweetShift",
            "ticker": "GOOGL",
            "tweet_url": "https://twitter.com/wallstengine/status/1980651869052760512",
            "category": "Company Strategy",
            "text": "OpenAI hosting livestream at 1 PM ET to potentially announce...",
        }
    )

    test_tweet = pd.Series(
        {
            "timestamp": "2025-10-22 16:39:00",
            "author": "Wall St Engine • TweetShift",
            "ticker": "QS",
            "tweet_url": "https://twitter.com/wallstengine/status/1981097982692118863",
            "category": "Earnings",
            "text": (
                "QuantumScape $QS posted a Q3 EPS loss of ($0.18), beating estimates by $0.02. "
                "The company shipped its first B1 samples of the QSE-5 solid-state battery during the quarter, meeting a key 2025 goal. "
                "The cells use the new Cobra separator process and are part of QuantumScape's Volkswagen Ducati V21L program. "
                "Q3 adjusted EBITDA loss was $61.4M with capex of $9.6M, mainly for its Eagle Line pilot line in San Jose. "
                "Full-year EBITDA loss guidance was improved to $245M–$260M, and capex lowered to $30M–$40M. "
                "Liquidity stood at $1B, extending its cash runway through 2030. "
                "The company will shift from reporting runway updates to customer billing metrics going forward."
            ),
        }
    )

    print("Test Tweet:")
    print(f"  Ticker: {test_tweet['ticker']}")
    print(f"  Timestamp: {test_tweet['timestamp']}")
    print(f"  Author: {test_tweet['author']}")
    print()

    # Initialize components with dependency injection
    ib_fetcher = IBHistoricalDataFetcher(host="127.0.0.1", port=7497, client_id=1)
    cache = DataCache(ib_fetcher)
    indicators = TechnicalIndicators()
    enricher = TweetEnricher(ib_fetcher, cache, indicators)

    # Enable DEBUG logging for detailed output
    logging.getLogger("tweet_enricher").setLevel(logging.DEBUG)

    print("Connecting to Interactive Brokers...")
    connected = await enricher.connect()

    if not connected:
        print("Failed to connect to IB. Make sure TWS/Gateway is running.")
        return

    print("Connected to IB")
    print()

    try:
        # Enrich the tweet
        print("Enriching tweet...")
        from datetime import datetime, timedelta

        from tweet_enricher.config import ET

        max_date = datetime.now(ET) + timedelta(days=2)
        result = await enricher.enrich_tweet(test_tweet, max_date)

        print()
        print("=" * 80)
        print("ENRICHMENT RESULTS")
        print("=" * 80)
        print()

        # Print results in organized sections
        print("BASIC INFO:")
        print(f"  Ticker:                  {result['ticker']}")
        print(f"  Timestamp:               {result['timestamp']}")
        print(f"  Market Session:          {result['session']}")
        print()

        print("PRICE DATA:")
        if result["entry_price"]:
            print(f"  Entry Price:             ${result['entry_price']:.2f}")
        else:
            print("  Entry Price:             N/A")
        print(f"  Entry Price Flag:        {result['entry_price_flag']}")
        if result["exit_price_1hr"]:
            print(f"  Exit Price 1hr:          ${result['exit_price_1hr']:.2f}")
        else:
            print("  Exit Price 1hr:          N/A")
        print(f"  Exit Price 1hr Flag:     {result['exit_price_1hr_flag']}")
        print()

        print("TECHNICAL INDICATORS:")
        if result["return_1d"] is not None:
            print(f"  Return 1D:               {result['return_1d']:.4f}")
        else:
            print("  Return 1D:               N/A")
        print(
            f"  Volatility 7D:           {result['volatility_7d']:.4f}"
            if result["volatility_7d"] is not None
            else "  Volatility 7D:           N/A"
        )
        print(
            f"  Relative Volume:         {result['relative_volume']:.2f}x"
            if result["relative_volume"] is not None
            else "  Relative Volume:         N/A"
        )
        print(f"  RSI (14):                {result['rsi_14']:.2f}" if result["rsi_14"] is not None else "  RSI (14):                N/A")
        print(
            f"  Distance from MA(20):    {result['distance_from_ma_20']:.2%}"
            if result["distance_from_ma_20"] is not None
            else "  Distance from MA(20):    N/A"
        )
        print()

        print("MARKET CONTEXT:")
        if result["spy_return_1d"] is not None:
            print(f"  SPY Return 1D:           {result['spy_return_1d']:.6f} ({result['spy_return_1d']:.2%})")
        else:
            print("  SPY Return 1D:           N/A")
        if result["spy_return_1hr"] is not None:
            print(f"  SPY Return 1hr:          {result['spy_return_1hr']:.6f} ({result['spy_return_1hr']:.2%})")
        else:
            print("  SPY Return 1hr:          N/A")
        print()

        print("PREDICTION TARGETS:")
        print(
            f"  Return 1hr:              {result['return_1hr']:.4f}"
            if result["return_1hr"] is not None
            else "  Return 1hr:              N/A"
        )
        print(
            f"  Return 1hr Adjusted:     {result['return_1hr_adjusted']:.4f}"
            if result["return_1hr_adjusted"] is not None
            else "  Return 1hr Adjusted:     N/A"
        )
        print(f"  Label (5-class):         {result['label_5class']}" if result["label_5class"] else "  Label (5-class):         N/A")
        print()

    finally:
        await enricher.disconnect()
        print("Disconnected from IB")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        raise

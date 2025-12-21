#!/usr/bin/env python3
"""Stock Deal Scanner - Analyzes all stocks and finds best buying opportunities."""

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pyarrow.feather as feather

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class TradeSetup:
    """Complete trade setup with entry, stop, and targets."""

    direction: str  # LONG or SHORT
    entry_price: float
    stop_loss: float
    target_1: float
    target_2: float
    target_3: float
    risk_per_share: float
    reward_1: float
    reward_2: float
    reward_3: float
    risk_reward_1: float
    risk_reward_2: float
    risk_reward_3: float
    risk_pct: float
    position_size_100: int
    position_size_500: int
    position_size_1000: int


@dataclass
class SentimentInfo:
    """Sentiment analysis results from tweets with time-weighted decay."""

    # Time-weighted metrics (primary)
    weighted_sentiment: float = 0.0  # Time-weighted score (-1 to +1)
    momentum: float = 0.0  # Sentiment trend (+ improving, - deteriorating)

    # Raw metrics
    raw_sentiment: float = 0.0  # Unweighted average sentiment
    tweet_count_30d: int = 0  # Tweets in extended lookback
    tweet_count_7d: int = 0  # Tweets in last 7 days

    # Distribution
    positive_pct: float = 0.0  # % positive tweets
    negative_pct: float = 0.0  # % negative tweets
    neutral_pct: float = 0.0  # % neutral tweets

    # Additional signals
    volume_spike: bool = False  # 2x normal tweet volume
    dominant_category: str = ""  # Most common category
    avg_weight: float = 0.0  # Average decay weight (diagnostic)


@dataclass
class PatternInfo:
    """Pattern detection results."""

    cup_and_handle: bool = False
    cup_depth_pct: float = 0.0
    handle_depth_pct: float = 0.0
    cup_length_days: int = 0

    channel_type: str = ""  # ASCENDING, DESCENDING, HORIZONTAL, or ""
    channel_support: float = 0.0
    channel_resistance: float = 0.0
    channel_width_pct: float = 0.0
    channel_position_pct: float = 0.0  # Where price is in channel (0=bottom, 100=top)


@dataclass
class LLMTradeSetup:
    """LLM-refined trade setup with reasoning and confidence."""

    # Entry
    entry_price: float = 0.0
    entry_type: str = "limit"  # limit or market
    entry_reasoning: str = ""

    # Stop loss
    stop_loss: float = 0.0
    stop_reasoning: str = ""

    # Targets
    target_1: float = 0.0
    target_2: float = 0.0
    target_3: float = 0.0
    target_reasoning: str = ""

    # Assessment
    confidence: int = 0  # 1-10 scale
    risk_reward: str = ""  # e.g., "2.3R to T1, 4.2R to T2"
    key_risks: list = field(default_factory=list)
    thesis: str = ""  # Plain English trade thesis

    # Metadata
    model_used: str = ""
    analysis_timestamp: str = ""


@dataclass
class StockAnalysis:
    """Analysis results for a single stock."""

    ticker: str
    direction: str  # LONG or SHORT
    current_price: float
    ath: float
    atl: float
    ath_discount_pct: float
    atl_premium_pct: float
    rsi: float
    bb_position_pct: float
    ma20: Optional[float]
    ma50: Optional[float]
    ma100: Optional[float]
    ma200: Optional[float]
    above_ma20: bool
    above_ma50: bool
    above_ma100: bool
    above_ma200: bool
    atr: float
    atr_pct: float
    price_percentile: float
    trading_days: int
    avg_volume: float
    volume_trend: float
    deal_score: int
    deal_rating: str
    nearest_support: float
    support_distance_pct: float
    data_source: str
    date_range: str
    # Enhanced scoring fields
    momentum_5d: float = 0.0
    momentum_20d: float = 0.0
    volume_at_support: float = 0.0
    resistance: float = 0.0
    resistance_distance_pct: float = 0.0
    consolidation_score: float = 0.0
    setup_type: str = ""  # BREAKOUT, PULLBACK, CUP_HANDLE, CHANNEL, SHORT_BREAKOUT, SHORT_PULLBACK
    key_supports: list = field(default_factory=list)
    # Pattern info
    pattern: Optional[PatternInfo] = None
    # Trade setup
    trade_setup: Optional[TradeSetup] = None
    # Sentiment info
    sentiment: Optional[SentimentInfo] = None
    sentiment_aligned: bool = False  # True if sentiment matches direction
    # LLM-refined analysis (optional, only for top candidates)
    llm_analysis: Optional[LLMTradeSetup] = None


class SentimentEnricher:
    """Enrich stock analysis with pre-computed tweet sentiment using time-weighted decay.

    Expects tweets CSV with pre-computed sentiment_score column from FinBERT.
    Use scripts/preprocess_tweet_sentiment.py to generate the enriched CSV.
    """

    def __init__(self, tweets_path: Path, lookback_days: int = 30, half_life: float = 7.0):
        """
        Initialize sentiment enricher with time-weighted decay.

        Args:
            tweets_path: Path to tweets CSV with pre-computed sentiment_score column
            lookback_days: Number of days to look back for tweets (default: 30)
            half_life: Days until weight drops to 50% (default: 7.0)
        """
        self.lookback_days = lookback_days
        self.half_life = half_life
        self.has_sentiment = False
        self._load_tweets(tweets_path)

    def _load_tweets(self, tweets_path: Path) -> None:
        """Load pre-computed tweets DataFrame."""
        logger.info(f"Loading pre-computed tweets from {tweets_path}...")
        self.tweets = pd.read_csv(tweets_path)
        self.tweets["timestamp"] = pd.to_datetime(self.tweets["timestamp"])

        # Check for pre-computed sentiment
        if "sentiment_score" in self.tweets.columns:
            self.has_sentiment = True
            # Fill NaN sentiment scores with 0 (neutral)
            self.tweets["sentiment_score"] = self.tweets["sentiment_score"].fillna(0.0)
            logger.info("Found pre-computed sentiment_score column")
        else:
            logger.warning("No sentiment_score column found! Run preprocess_tweet_sentiment.py first.")
            self.has_sentiment = False

        # Build ticker -> tweets index for fast lookup
        self.ticker_tweets = {}
        for ticker in self.tweets["ticker"].unique():
            ticker_df = self.tweets[self.tweets["ticker"] == ticker].copy()
            # Sort by timestamp descending for efficient recent lookups
            ticker_df = ticker_df.sort_values("timestamp", ascending=False)
            self.ticker_tweets[ticker] = ticker_df

        logger.info(f"Loaded {len(self.tweets)} tweets for {len(self.ticker_tweets)} tickers")
        logger.info(f"Time-weighted decay: half_life={self.half_life}d, lookback={self.lookback_days}d")

    def get_ticker_sentiment(self, ticker: str) -> SentimentInfo:
        """
        Get time-weighted sentiment for a ticker from pre-computed tweets.

        Uses exponential decay weighting where:
        - weight = exp(-age_days / half_life)
        - At age=0: weight=1.0
        - At age=half_life: weight=0.5
        - At age=2*half_life: weight=0.25

        Args:
            ticker: Stock ticker symbol

        Returns:
            SentimentInfo with time-weighted sentiment and momentum
        """
        # Default empty sentiment
        sentiment = SentimentInfo()

        if ticker not in self.ticker_tweets:
            return sentiment

        ticker_df = self.ticker_tweets[ticker]
        now = datetime.now()

        # Filter tweets within lookback period
        cutoff_30d = now - timedelta(days=self.lookback_days)
        cutoff_7d = now - timedelta(days=7)
        cutoff_14d = now - timedelta(days=14)

        recent_30d = ticker_df[ticker_df["timestamp"] >= cutoff_30d].copy()

        if len(recent_30d) == 0:
            return sentiment

        # Calculate age in days for each tweet
        recent_30d["age_days"] = (now - recent_30d["timestamp"]).dt.total_seconds() / 86400

        # Calculate exponential decay weights
        recent_30d["weight"] = np.exp(-recent_30d["age_days"] / self.half_life)

        # =====================================================================
        # TIME-WEIGHTED SENTIMENT
        # =====================================================================
        if self.has_sentiment:
            weighted_sum = (recent_30d["sentiment_score"] * recent_30d["weight"]).sum()
            weight_sum = recent_30d["weight"].sum()
            weighted_sentiment = weighted_sum / weight_sum if weight_sum > 0 else 0.0

            # Raw (unweighted) sentiment for comparison
            raw_sentiment = recent_30d["sentiment_score"].mean()
        else:
            weighted_sentiment = 0.0
            raw_sentiment = 0.0

        # =====================================================================
        # SENTIMENT MOMENTUM (7d vs 7-14d comparison)
        # =====================================================================
        recent_7d = recent_30d[recent_30d["age_days"] <= 7]
        older_7_14d = recent_30d[(recent_30d["age_days"] > 7) & (recent_30d["age_days"] <= 14)]

        if self.has_sentiment and len(recent_7d) > 0 and len(older_7_14d) > 0:
            recent_avg = recent_7d["sentiment_score"].mean()
            older_avg = older_7_14d["sentiment_score"].mean()
            momentum = recent_avg - older_avg
        else:
            momentum = 0.0

        # =====================================================================
        # SENTIMENT DISTRIBUTION
        # =====================================================================
        if "sentiment_label" in recent_30d.columns:
            label_counts = recent_30d["sentiment_label"].value_counts()
            total = len(recent_30d)
            positive_pct = (label_counts.get("positive", 0) / total * 100) if total > 0 else 0
            negative_pct = (label_counts.get("negative", 0) / total * 100) if total > 0 else 0
            neutral_pct = (label_counts.get("neutral", 0) / total * 100) if total > 0 else 0
        else:
            # Estimate from scores if no labels
            if self.has_sentiment:
                positive_pct = (recent_30d["sentiment_score"] > 0.3).sum() / len(recent_30d) * 100
                negative_pct = (recent_30d["sentiment_score"] < -0.3).sum() / len(recent_30d) * 100
                neutral_pct = 100 - positive_pct - negative_pct
            else:
                positive_pct = negative_pct = neutral_pct = 0.0

        # =====================================================================
        # VOLUME SPIKE DETECTION
        # =====================================================================
        # Compare 7-day volume to average weekly volume
        tweet_count_7d = len(recent_7d)
        total_days = max((now - ticker_df["timestamp"].min()).days, 1)
        avg_weekly = len(ticker_df) / (total_days / 7) if total_days > 7 else len(ticker_df)
        volume_spike = tweet_count_7d > avg_weekly * 2

        # =====================================================================
        # DOMINANT CATEGORY
        # =====================================================================
        if "category" in recent_30d.columns and len(recent_30d) > 0:
            mode_result = recent_30d["category"].mode()
            dominant_category = mode_result.iloc[0] if len(mode_result) > 0 else ""
        else:
            dominant_category = ""

        return SentimentInfo(
            weighted_sentiment=round(weighted_sentiment, 3),
            momentum=round(momentum, 3),
            raw_sentiment=round(raw_sentiment, 3),
            tweet_count_30d=len(recent_30d),
            tweet_count_7d=tweet_count_7d,
            positive_pct=round(positive_pct, 1),
            negative_pct=round(negative_pct, 1),
            neutral_pct=round(neutral_pct, 1),
            volume_spike=volume_spike,
            dominant_category=dominant_category,
            avg_weight=round(recent_30d["weight"].mean(), 3),
        )

    def get_all_tickers(self) -> set:
        """Get all tickers with tweets."""
        return set(self.ticker_tweets.keys())


class LLMAnalyzer:
    """Analyze top stock candidates using GPT-5.2 for refined trade setups."""

    SYSTEM_PROMPT = """You are an expert technical analyst specializing in trade setup analysis. 
Analyze the provided stock data and generate a precise trade setup with entry, stop loss, and profit targets.
Be specific with price levels and explain your reasoning based on the technical data provided.

Output ONLY valid JSON with this exact structure (no markdown, no explanation outside JSON):
{
  "entry": {"price": <float>, "type": "limit|market", "reasoning": "<string>"},
  "stop_loss": {"price": <float>, "reasoning": "<string>"},
  "targets": [
    {"price": <float>, "label": "T1", "reasoning": "<string>"},
    {"price": <float>, "label": "T2", "reasoning": "<string>"},
    {"price": <float>, "label": "T3", "reasoning": "<string>"}
  ],
  "confidence": <int 1-10>,
  "risk_reward": "<string like '2.3R to T1, 4.2R to T2'>",
  "key_risks": ["<risk1>", "<risk2>"],
  "thesis": "<1-2 sentence trade thesis>"
}"""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize LLM analyzer.

        Args:
            api_key: NVIDIA API key. If not provided, reads from NVIDIA_API_KEY env var.
        """
        self.api_key = api_key or os.environ.get("NVIDIA_API_KEY")
        if not self.api_key:
            raise ValueError("NVIDIA_API_KEY not provided and not found in environment")

        self.base_url = "https://inference-api.nvidia.com"
        self.model = "azure/openai/gpt-5.2"
        self.client = None  # Lazy initialization

    def _get_client(self):
        """Get or create async OpenAI client."""
        if self.client is None:
            try:
                from openai import AsyncOpenAI

                self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
            except ImportError:
                raise ImportError("openai package required. Install with: pip install openai")
        return self.client

    def _build_prompt(self, analysis: StockAnalysis) -> str:
        """Build comprehensive prompt with all stock data."""
        # Pattern info
        pattern_str = "None detected"
        if analysis.pattern:
            if analysis.pattern.cup_and_handle:
                pattern_str = f"Cup and Handle (cup depth: {analysis.pattern.cup_depth_pct}%, handle: {analysis.pattern.handle_depth_pct}%)"
            elif analysis.pattern.channel_type:
                pattern_str = f"{analysis.pattern.channel_type} Channel (support: ${analysis.pattern.channel_support}, resistance: ${analysis.pattern.channel_resistance}, width: {analysis.pattern.channel_width_pct}%)"

        # Sentiment info
        sentiment_str = "No sentiment data"
        if analysis.sentiment and analysis.sentiment.tweet_count_30d > 0:
            sentiment_str = f"Score: {analysis.sentiment.weighted_sentiment:+.2f}, Momentum: {analysis.sentiment.momentum:+.2f}, Tweets (30d): {analysis.sentiment.tweet_count_30d}"

        # Trade setup info (if exists)
        setup_str = "No rule-based setup"
        if analysis.trade_setup:
            ts = analysis.trade_setup
            setup_str = f"Entry: ${ts.entry_price}, Stop: ${ts.stop_loss}, T1: ${ts.target_1}, R:R: {ts.risk_reward_1}"

        prompt = f"""Analyze this {analysis.direction} trade setup for {analysis.ticker}:

SETUP TYPE: {analysis.setup_type}
DEAL SCORE: {analysis.deal_score}/100 ({analysis.deal_rating})

PRICE DATA:
- Current Price: ${analysis.current_price}
- All-Time High: ${analysis.ath} ({analysis.ath_discount_pct}% below)
- All-Time Low: ${analysis.atl}
- 20MA: ${analysis.ma20 or 'N/A'}
- 50MA: ${analysis.ma50 or 'N/A'}
- 200MA: ${analysis.ma200 or 'N/A'}

TECHNICAL INDICATORS:
- RSI (14): {analysis.rsi}
- ATR: ${analysis.atr} ({analysis.atr_pct}% of price)
- Bollinger Position: {analysis.bb_position_pct}%
- Momentum (5d): {analysis.momentum_5d}%
- Momentum (20d): {analysis.momentum_20d}%

KEY LEVELS:
- Nearest Support: ${analysis.nearest_support} ({analysis.support_distance_pct}% below)
- Nearest Resistance: ${analysis.resistance} ({analysis.resistance_distance_pct}% above)
- Key Supports: {analysis.key_supports}
- Volume at Support: {analysis.volume_at_support}%
- Consolidation Score: {analysis.consolidation_score}/100

PATTERN DETECTED: {pattern_str}

SENTIMENT: {sentiment_str}

RULE-BASED SETUP: {setup_str}

Based on this data, provide your refined trade setup with specific price levels and reasoning."""

        return prompt

    async def analyze_stock(self, analysis: StockAnalysis) -> Optional[LLMTradeSetup]:
        """
        Get LLM-refined trade setup for a single stock.

        Args:
            analysis: StockAnalysis object with all technical data

        Returns:
            LLMTradeSetup or None if analysis fails
        """
        client = self._get_client()
        prompt = self._build_prompt(analysis)

        try:
            response = await client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": self.SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
                temperature=0.3,  # Lower temperature for more consistent outputs
                max_tokens=1024,
            )

            content = response.choices[0].message.content.strip()

            # Parse JSON response
            # Handle potential markdown code blocks
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()

            data = json.loads(content)

            return LLMTradeSetup(
                entry_price=float(data["entry"]["price"]),
                entry_type=data["entry"].get("type", "limit"),
                entry_reasoning=data["entry"]["reasoning"],
                stop_loss=float(data["stop_loss"]["price"]),
                stop_reasoning=data["stop_loss"]["reasoning"],
                target_1=float(data["targets"][0]["price"]),
                target_2=float(data["targets"][1]["price"]),
                target_3=float(data["targets"][2]["price"]),
                target_reasoning="; ".join([t["reasoning"] for t in data["targets"]]),
                confidence=int(data["confidence"]),
                risk_reward=data["risk_reward"],
                key_risks=data.get("key_risks", []),
                thesis=data["thesis"],
                model_used=self.model,
                analysis_timestamp=datetime.now().isoformat(),
            )

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM response for {analysis.ticker}: {e}")
            return None
        except Exception as e:
            logger.warning(f"LLM analysis failed for {analysis.ticker}: {e}")
            return None

    async def analyze_batch(self, candidates: list[StockAnalysis], max_concurrent: int = 5) -> dict[str, Optional[LLMTradeSetup]]:
        """
        Analyze multiple stocks in parallel with rate limiting.

        Args:
            candidates: List of StockAnalysis objects
            max_concurrent: Maximum concurrent API calls

        Returns:
            Dict mapping ticker to LLMTradeSetup (or None if failed)
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def analyze_with_semaphore(analysis: StockAnalysis):
            async with semaphore:
                return await self.analyze_stock(analysis)

        logger.info(f"Starting LLM analysis for {len(candidates)} stocks (max {max_concurrent} concurrent)...")

        tasks = [analyze_with_semaphore(a) for a in candidates]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        output = {}
        for candidate, result in zip(candidates, results):
            if isinstance(result, Exception):
                logger.warning(f"LLM analysis exception for {candidate.ticker}: {result}")
                output[candidate.ticker] = None
            else:
                output[candidate.ticker] = result

        successful = sum(1 for v in output.values() if v is not None)
        logger.info(f"LLM analysis complete: {successful}/{len(candidates)} successful")

        return output


class StockDealScanner:
    """Scans stock data files and calculates deal scores."""

    # Trade setup quality thresholds
    MIN_RR = 1.5  # Minimum risk/reward ratio for valid setup
    MAX_RISK_PCT = 5.0  # Maximum risk percentage per trade

    def __init__(
        self,
        data_dir: Path,
        tweets_path: Optional[Path] = None,
        lookback_days: int = 30,
        half_life: float = 7.0,
        llm_api_key: Optional[str] = None,
        llm_top_n: int = 20,
    ):
        """
        Initialize scanner with data directory.

        Args:
            data_dir: Path to data directory containing daily/ and intraday/ folders
            tweets_path: Optional path to tweets CSV for sentiment analysis (pre-computed)
            lookback_days: Number of days to look back for sentiment analysis (default: 30)
            half_life: Days until sentiment weight drops to 50% (default: 7.0)
            llm_api_key: Optional NVIDIA API key for LLM analysis
            llm_top_n: Number of top candidates to analyze with LLM (default: 20)
        """
        self.data_dir = data_dir
        self.daily_dir = data_dir / "daily"
        self.intraday_dir = data_dir / "intraday"
        self.results: list[StockAnalysis] = []

        # Initialize sentiment enricher if tweets provided
        self.sentiment_enricher: Optional[SentimentEnricher] = None
        if tweets_path and tweets_path.exists():
            self.sentiment_enricher = SentimentEnricher(tweets_path, lookback_days, half_life)

        # Initialize LLM analyzer if API key provided
        self.llm_analyzer: Optional[LLMAnalyzer] = None
        self.llm_top_n = llm_top_n
        if llm_api_key:
            try:
                self.llm_analyzer = LLMAnalyzer(api_key=llm_api_key)
                logger.info(f"LLM analysis enabled for top {llm_top_n} candidates")
            except Exception as e:
                logger.warning(f"Failed to initialize LLM analyzer: {e}")

    def scan_all(self, prefer_intraday: bool = True) -> list[StockAnalysis]:
        """
        Scan all stock files and analyze them.

        Args:
            prefer_intraday: If both exist, prefer intraday data (more granular)

        Returns:
            List of StockAnalysis objects
        """
        daily_tickers = {f.stem for f in self.daily_dir.glob("*.feather")} if self.daily_dir.exists() else set()
        intraday_tickers = {f.stem for f in self.intraday_dir.glob("*.feather")} if self.intraday_dir.exists() else set()

        all_tickers = daily_tickers | intraday_tickers
        logger.info(f"Found {len(all_tickers)} unique tickers ({len(daily_tickers)} daily, {len(intraday_tickers)} intraday)")

        for i, ticker in enumerate(sorted(all_tickers)):
            if (i + 1) % 50 == 0:
                logger.info(f"Processing {i + 1}/{len(all_tickers)}...")

            try:
                has_intraday = ticker in intraday_tickers
                has_daily = ticker in daily_tickers

                if prefer_intraday and has_intraday:
                    file_path = self.intraday_dir / f"{ticker}.feather"
                    source = "intraday"
                elif has_daily:
                    file_path = self.daily_dir / f"{ticker}.feather"
                    source = "daily"
                else:
                    file_path = self.intraday_dir / f"{ticker}.feather"
                    source = "intraday"

                analysis = self._analyze_stock(ticker, file_path, source)
                if analysis:
                    self.results.append(analysis)

            except Exception as e:
                logger.warning(f"Error analyzing {ticker}: {e}")

        self.results.sort(key=lambda x: x.deal_score, reverse=True)
        logger.info(f"Successfully analyzed {len(self.results)} stocks")

        # Run LLM analysis on top candidates if enabled
        if self.llm_analyzer:
            self._run_llm_analysis()

        return self.results

    def _run_llm_analysis(self) -> None:
        """Run LLM analysis on top candidates."""
        # Select top candidates by score (LLM can create setups from scratch)
        candidates = [r for r in self.results if r.deal_score >= 70][: self.llm_top_n]

        if not candidates:
            logger.warning("No candidates meet criteria for LLM analysis")
            return

        logger.info(f"Running LLM analysis on {len(candidates)} top candidates...")

        # Run async analysis
        try:
            llm_results = asyncio.run(self.llm_analyzer.analyze_batch(candidates, max_concurrent=5))

            # Attach results to stock analysis objects
            for analysis in self.results:
                if analysis.ticker in llm_results:
                    analysis.llm_analysis = llm_results[analysis.ticker]

        except Exception as e:
            logger.error(f"LLM analysis batch failed: {e}")

    def _analyze_stock(self, ticker: str, file_path: Path, source: str) -> Optional[StockAnalysis]:
        """Analyze a single stock file."""
        df = feather.read_feather(file_path)

        if len(df) < 20:
            return None

        # Aggregate intraday to daily if needed
        if source == "intraday" and "date" in df.columns:
            df["day"] = df["date"].dt.date
            daily = df.groupby("day").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).reset_index()
        else:
            daily = df.copy()
            if "date" in daily.columns:
                daily["day"] = daily["date"]

        if len(daily) < 20:
            return None

        # Current price and extremes
        current_price = float(daily["close"].iloc[-1])
        ath = float(daily["high"].max())
        atl = float(daily["low"].min())

        if current_price <= 0 or ath <= 0:
            return None

        ath_discount_pct = (ath - current_price) / ath * 100
        atl_premium_pct = (current_price - atl) / atl * 100

        # RSI (14-day)
        delta = daily["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        daily["rsi"] = 100 - (100 / (1 + rs))
        rsi = float(daily["rsi"].iloc[-1]) if pd.notna(daily["rsi"].iloc[-1]) else 50

        # Bollinger Bands
        daily["bb_middle"] = daily["close"].rolling(20).mean()
        daily["bb_std"] = daily["close"].rolling(20).std()
        daily["bb_upper"] = daily["bb_middle"] + (2 * daily["bb_std"])
        daily["bb_lower"] = daily["bb_middle"] - (2 * daily["bb_std"])

        bb_upper = daily["bb_upper"].iloc[-1]
        bb_lower = daily["bb_lower"].iloc[-1]
        if pd.notna(bb_upper) and pd.notna(bb_lower) and bb_upper != bb_lower:
            bb_position_pct = (current_price - bb_lower) / (bb_upper - bb_lower) * 100
        else:
            bb_position_pct = 50

        # Moving Averages
        daily["ma20"] = daily["close"].rolling(20).mean()
        daily["ma50"] = daily["close"].rolling(50).mean()
        daily["ma100"] = daily["close"].rolling(100).mean()
        daily["ma200"] = daily["close"].rolling(200).mean()

        ma20 = float(daily["ma20"].iloc[-1]) if pd.notna(daily["ma20"].iloc[-1]) else None
        ma50 = float(daily["ma50"].iloc[-1]) if pd.notna(daily["ma50"].iloc[-1]) else None
        ma100 = float(daily["ma100"].iloc[-1]) if pd.notna(daily["ma100"].iloc[-1]) else None
        ma200 = float(daily["ma200"].iloc[-1]) if pd.notna(daily["ma200"].iloc[-1]) else None

        # ATR
        daily["tr"] = np.maximum(
            daily["high"] - daily["low"],
            np.maximum(abs(daily["high"] - daily["close"].shift(1)), abs(daily["low"] - daily["close"].shift(1))),
        )
        daily["atr"] = daily["tr"].rolling(14).mean()
        atr = float(daily["atr"].iloc[-1]) if pd.notna(daily["atr"].iloc[-1]) else 0
        atr_pct = (atr / current_price * 100) if current_price > 0 else 0

        # Price percentile
        price_percentile = (daily["close"] < current_price).sum() / len(daily) * 100

        # Volume analysis
        avg_volume = float(daily["volume"].mean())
        recent_volume = float(daily["volume"].tail(5).mean())
        volume_trend = recent_volume / avg_volume if avg_volume > 0 else 1

        # Find key supports (swing lows)
        key_supports = self._find_swing_lows(daily, current_price)

        # Nearest support
        supports_below = [s for s in key_supports if s < current_price]
        if supports_below:
            nearest_support = max(supports_below)
            support_distance_pct = (current_price - nearest_support) / current_price * 100
        else:
            nearest_support = atl
            support_distance_pct = (current_price - atl) / current_price * 100

        # Find resistance levels (swing highs)
        key_resistances = self._find_swing_highs(daily, current_price)
        resistances_above = [r for r in key_resistances if r > current_price]
        if resistances_above:
            resistance = min(resistances_above)
            resistance_distance_pct = (resistance - current_price) / current_price * 100
        else:
            resistance = ath
            resistance_distance_pct = (ath - current_price) / current_price * 100

        # Calculate momentum
        if len(daily) >= 5:
            momentum_5d = (current_price - daily["close"].iloc[-5]) / daily["close"].iloc[-5] * 100
        else:
            momentum_5d = 0

        if len(daily) >= 20:
            momentum_20d = (current_price - daily["close"].iloc[-20]) / daily["close"].iloc[-20] * 100
        else:
            momentum_20d = 0

        # Volume at support
        volume_at_support = self._calculate_volume_at_price(daily, nearest_support)

        # Consolidation score
        consolidation_score = self._calculate_consolidation_score(daily)

        # Determine direction and trend status
        above_ma200 = current_price > ma200 if ma200 else False
        direction = "LONG" if above_ma200 else "SHORT"

        # Detect patterns
        pattern = PatternInfo()
        if above_ma200:  # Only detect bullish patterns for longs
            cup_pattern = self._detect_cup_and_handle(daily, current_price)
            if cup_pattern.cup_and_handle:
                pattern = cup_pattern

        channel_pattern = self._detect_channel(daily, current_price)
        if channel_pattern.channel_type:
            pattern.channel_type = channel_pattern.channel_type
            pattern.channel_support = channel_pattern.channel_support
            pattern.channel_resistance = channel_pattern.channel_resistance
            pattern.channel_width_pct = channel_pattern.channel_width_pct
            pattern.channel_position_pct = channel_pattern.channel_position_pct

        # Determine setup type (includes pattern-based setups)
        setup_type = self._determine_setup_type(current_price, resistance, nearest_support, momentum_5d, above_ma200, pattern)

        # Get sentiment if enricher available
        sentiment: Optional[SentimentInfo] = None
        sentiment_aligned = False
        if self.sentiment_enricher:
            sentiment = self.sentiment_enricher.get_ticker_sentiment(ticker)
            # Check if sentiment aligns with direction (using weighted sentiment + positive momentum)
            if sentiment.tweet_count_30d > 0:
                ws = sentiment.weighted_sentiment
                mom = sentiment.momentum
                if direction == "LONG" and ws > 0.1 and mom >= 0:
                    sentiment_aligned = True
                elif direction == "SHORT" and ws < -0.1 and mom <= 0:
                    sentiment_aligned = True

        # Calculate deal score (different for longs vs shorts)
        deal_score = self._calculate_deal_score(
            current_price=current_price,
            ma20=ma20,
            ma50=ma50,
            ma200=ma200,
            rsi=rsi,
            momentum_5d=momentum_5d,
            momentum_20d=momentum_20d,
            support_distance_pct=support_distance_pct,
            resistance_distance_pct=resistance_distance_pct,
            volume_at_support=volume_at_support,
            consolidation_score=consolidation_score,
            volume_trend=volume_trend,
            setup_type=setup_type,
            pattern=pattern,
            direction=direction,
            sentiment=sentiment,
        )

        # Calculate trade setup
        trade_setup = self._calculate_trade_setup(
            current_price=current_price,
            nearest_support=nearest_support,
            resistance=resistance,
            key_supports=key_supports,
            key_resistances=key_resistances,
            atr=atr,
            ma200=ma200,
            setup_type=setup_type,
            direction=direction,
            pattern=pattern,
        )

        # Deal rating
        if deal_score >= 70:
            deal_rating = "EXCELLENT"
        elif deal_score >= 55:
            deal_rating = "GOOD"
        elif deal_score >= 40:
            deal_rating = "FAIR"
        else:
            deal_rating = "POOR"

        # Date range
        if "day" in daily.columns:
            date_range = f"{daily['day'].iloc[0]} to {daily['day'].iloc[-1]}"
        elif "date" in df.columns:
            date_range = f"{df['date'].iloc[0].date()} to {df['date'].iloc[-1].date()}"
        else:
            date_range = "N/A"

        return StockAnalysis(
            ticker=ticker,
            direction=direction,
            current_price=round(current_price, 2),
            ath=round(ath, 2),
            atl=round(atl, 2),
            ath_discount_pct=round(ath_discount_pct, 1),
            atl_premium_pct=round(atl_premium_pct, 1),
            rsi=round(rsi, 1),
            bb_position_pct=round(bb_position_pct, 1),
            ma20=round(ma20, 2) if ma20 else None,
            ma50=round(ma50, 2) if ma50 else None,
            ma100=round(ma100, 2) if ma100 else None,
            ma200=round(ma200, 2) if ma200 else None,
            above_ma20=current_price > ma20 if ma20 else False,
            above_ma50=current_price > ma50 if ma50 else False,
            above_ma100=current_price > ma100 if ma100 else False,
            above_ma200=above_ma200,
            atr=round(atr, 2),
            atr_pct=round(atr_pct, 2),
            price_percentile=round(price_percentile, 1),
            trading_days=len(daily),
            avg_volume=round(avg_volume, 0),
            volume_trend=round(volume_trend, 2),
            deal_score=deal_score,
            deal_rating=deal_rating,
            nearest_support=round(nearest_support, 2),
            support_distance_pct=round(support_distance_pct, 1),
            data_source=source,
            date_range=date_range,
            momentum_5d=round(momentum_5d, 2),
            momentum_20d=round(momentum_20d, 2),
            volume_at_support=round(volume_at_support, 1),
            resistance=round(resistance, 2),
            resistance_distance_pct=round(resistance_distance_pct, 1),
            consolidation_score=round(consolidation_score, 1),
            setup_type=setup_type,
            key_supports=[round(s, 2) for s in key_supports[:5]],
            pattern=pattern,
            trade_setup=trade_setup,
            sentiment=sentiment,
            sentiment_aligned=sentiment_aligned,
        )

    def _find_swing_lows(self, daily: pd.DataFrame, current_price: float, window: int = 5) -> list:
        """Find significant swing lows."""
        swing_lows = []
        for i in range(window, len(daily) - window):
            current_low = daily.iloc[i]["low"]
            left_lows = daily.iloc[i - window : i]["low"].values
            right_lows = daily.iloc[i + 1 : i + window + 1]["low"].values

            if all(current_low <= left_lows) and all(current_low <= right_lows):
                swing_lows.append(float(current_low))

        # Cluster similar levels
        clustered = []
        for low in sorted(set(swing_lows)):
            if not clustered or abs(low - clustered[-1]) > current_price * 0.02:
                clustered.append(low)

        return sorted(clustered, reverse=True)

    def _find_swing_highs(self, daily: pd.DataFrame, current_price: float, window: int = 5) -> list:
        """Find significant swing highs (resistance levels)."""
        swing_highs = []
        for i in range(window, len(daily) - window):
            current_high = daily.iloc[i]["high"]
            left_highs = daily.iloc[i - window : i]["high"].values
            right_highs = daily.iloc[i + 1 : i + window + 1]["high"].values

            if all(current_high >= left_highs) and all(current_high >= right_highs):
                swing_highs.append(float(current_high))

        # Cluster similar levels
        clustered = []
        for high in sorted(set(swing_highs), reverse=True):
            if not clustered or abs(high - clustered[-1]) > current_price * 0.02:
                clustered.append(high)

        return sorted(clustered, reverse=True)

    def _get_supports_below(self, key_supports: list, price: float, count: int = 3) -> list:
        """
        Get the next N support levels below a given price.

        Args:
            key_supports: List of support levels (sorted descending)
            price: Reference price
            count: Number of levels to return

        Returns:
            List of support levels below price, sorted by distance (nearest first)
        """
        below = [s for s in key_supports if s < price]
        return sorted(below, reverse=True)[:count]

    def _get_resistances_above(self, key_resistances: list, price: float, count: int = 3) -> list:
        """
        Get the next N resistance levels above a given price.

        Args:
            key_resistances: List of resistance levels (sorted descending)
            price: Reference price
            count: Number of levels to return

        Returns:
            List of resistance levels above price, sorted by distance (nearest first)
        """
        above = [r for r in key_resistances if r > price]
        return sorted(above)[:count]

    def _calculate_volume_at_price(self, daily: pd.DataFrame, price_level: float, tolerance_pct: float = 2.0) -> float:
        """Calculate volume concentration around a price level."""
        tolerance = price_level * (tolerance_pct / 100)
        mask = (daily["low"] <= price_level + tolerance) & (daily["high"] >= price_level - tolerance)
        volume_at_level = daily.loc[mask, "volume"].sum()
        total_volume = daily["volume"].sum()
        return (volume_at_level / total_volume * 100) if total_volume > 0 else 0

    def _calculate_consolidation_score(self, daily: pd.DataFrame, lookback: int = 10) -> float:
        """
        Calculate how tight the recent consolidation is.

        Returns a score 0-100 where higher = tighter consolidation (better for breakout).
        """
        if len(daily) < lookback:
            return 0

        recent = daily.tail(lookback)
        price_range = recent["high"].max() - recent["low"].min()
        avg_price = recent["close"].mean()
        range_pct = (price_range / avg_price) * 100

        # Tighter range = higher score
        # < 5% range = 100, > 20% range = 0
        if range_pct < 5:
            return 100
        elif range_pct > 20:
            return 0
        else:
            return max(0, 100 - (range_pct - 5) * (100 / 15))

    def _determine_setup_type(
        self,
        current_price: float,
        resistance: float,
        nearest_support: float,
        momentum_5d: float,
        above_ma200: bool,
        pattern: Optional[PatternInfo] = None,
    ) -> str:
        """Determine the type of setup."""
        resistance_dist = (resistance - current_price) / current_price * 100 if resistance > current_price else 0
        support_dist = (current_price - nearest_support) / current_price * 100 if nearest_support < current_price else 0

        # Check for pattern-based setups first
        if pattern:
            if pattern.cup_and_handle:
                return "CUP_HANDLE"
            if pattern.channel_type:
                return f"CHANNEL_{pattern.channel_type}"

        # Long setups (above 200MA)
        if above_ma200:
            if resistance_dist < 3 and momentum_5d > 0:
                return "BREAKOUT"
            elif support_dist < 5:
                return "PULLBACK"
            else:
                return "NEUTRAL"

        # Short setups (below 200MA)
        else:
            resistance_dist_from_above = (resistance - current_price) / current_price * 100
            if resistance_dist_from_above < 5 and momentum_5d < 0:
                return "SHORT_PULLBACK"  # Rally to resistance in downtrend
            elif support_dist < 3 and momentum_5d < 0:
                return "SHORT_BREAKOUT"  # Breaking down through support
            else:
                return "SHORT_NEUTRAL"

    def _detect_cup_and_handle(self, daily: pd.DataFrame, current_price: float, lookback: int = 60) -> PatternInfo:
        """
        Detect Cup and Handle pattern.

        Cup and Handle criteria:
        - Cup: U-shaped price formation (drop, round bottom, recovery)
        - Handle: Small pullback after cup completion (5-15% of cup depth)
        - Breakout: Price near or above cup's rim (left high)
        """
        pattern = PatternInfo()

        if len(daily) < lookback:
            return pattern

        recent = daily.tail(lookback).copy()
        closes = recent["close"].values
        highs = recent["high"].values
        lows = recent["low"].values

        # Find the highest point in first third (left rim of cup)
        third = lookback // 3
        left_rim_idx = np.argmax(highs[:third])
        left_rim = highs[left_rim_idx]

        # Find the lowest point in middle (cup bottom)
        cup_bottom_idx = third + np.argmin(lows[third : 2 * third])
        cup_bottom = lows[cup_bottom_idx]

        # Find highest point in last third (right rim / current area)
        right_section = highs[2 * third :]
        if len(right_section) == 0:
            return pattern
        right_rim_idx = 2 * third + np.argmax(right_section)
        right_rim = highs[right_rim_idx]

        # Cup depth calculation
        cup_depth = left_rim - cup_bottom
        cup_depth_pct = (cup_depth / left_rim) * 100

        # Validate cup shape (15-50% depth is typical)
        if not (10 <= cup_depth_pct <= 60):
            return pattern

        # Check if right rim recovered to near left rim (within 10%)
        rim_recovery = right_rim / left_rim
        if rim_recovery < 0.85:
            return pattern

        # Detect handle (small pullback from right rim)
        handle_section = closes[right_rim_idx:]
        if len(handle_section) < 3:
            return pattern

        handle_low = np.min(handle_section)
        handle_depth = right_rim - handle_low
        handle_depth_pct = (handle_depth / right_rim) * 100

        # Handle should be 5-20% of cup depth and price should be near rim
        handle_valid = handle_depth_pct < cup_depth_pct * 0.5 and handle_depth_pct < 15

        # Current price should be near or above the rim for breakout
        price_near_rim = current_price >= right_rim * 0.95

        if handle_valid and price_near_rim:
            pattern.cup_and_handle = True
            pattern.cup_depth_pct = round(cup_depth_pct, 1)
            pattern.handle_depth_pct = round(handle_depth_pct, 1)
            pattern.cup_length_days = lookback

        return pattern

    def _detect_channel(self, daily: pd.DataFrame, current_price: float, lookback: int = 30) -> PatternInfo:
        """
        Detect channel patterns (ascending, descending, horizontal).

        Channel criteria:
        - At least 2 touches on support and resistance trendlines
        - Parallel or near-parallel lines
        - Price bouncing between the lines
        """
        pattern = PatternInfo()

        if len(daily) < lookback:
            return pattern

        recent = daily.tail(lookback).copy()
        recent = recent.reset_index(drop=True)
        closes = recent["close"].values
        highs = recent["high"].values
        lows = recent["low"].values
        x = np.arange(len(closes))

        # Linear regression on highs (resistance line)
        resist_slope, resist_intercept = np.polyfit(x, highs, 1)
        resist_line = resist_slope * x + resist_intercept

        # Linear regression on lows (support line)
        support_slope, support_intercept = np.polyfit(x, lows, 1)
        support_line = support_slope * x + support_intercept

        # Current channel levels
        channel_resistance = resist_slope * (len(closes) - 1) + resist_intercept
        channel_support = support_slope * (len(closes) - 1) + support_intercept

        # Channel width
        channel_width = channel_resistance - channel_support
        channel_width_pct = (channel_width / current_price) * 100

        # Check if channel is valid (parallel lines, reasonable width)
        slope_diff = abs(resist_slope - support_slope) / max(abs(resist_slope), abs(support_slope), 0.001)

        # Lines should be roughly parallel (slope difference < 50%)
        if slope_diff > 0.5:
            return pattern

        # Channel should have reasonable width (3-20% of price)
        if not (2 <= channel_width_pct <= 25):
            return pattern

        # Count touches (price coming within 1% of trendline)
        touch_threshold = current_price * 0.01
        resist_touches = sum(1 for i, h in enumerate(highs) if abs(h - resist_line[i]) < touch_threshold)
        support_touches = sum(1 for i, l in enumerate(lows) if abs(l - support_line[i]) < touch_threshold)

        # Need at least 2 touches on each line
        if resist_touches < 2 or support_touches < 2:
            return pattern

        # Determine channel type based on slope
        avg_slope = (resist_slope + support_slope) / 2
        slope_pct = (avg_slope / current_price) * 100 * lookback  # Normalize to % change over period

        if slope_pct > 3:
            channel_type = "ASCENDING"
        elif slope_pct < -3:
            channel_type = "DESCENDING"
        else:
            channel_type = "HORIZONTAL"

        # Where is price in the channel? (0 = support, 100 = resistance)
        if channel_width > 0:
            channel_position = ((current_price - channel_support) / channel_width) * 100
        else:
            channel_position = 50

        pattern.channel_type = channel_type
        pattern.channel_support = round(channel_support, 2)
        pattern.channel_resistance = round(channel_resistance, 2)
        pattern.channel_width_pct = round(channel_width_pct, 1)
        pattern.channel_position_pct = round(channel_position, 1)

        return pattern

    def _calculate_trade_setup(
        self,
        current_price: float,
        nearest_support: float,
        resistance: float,
        key_supports: list,
        key_resistances: list,
        atr: float,
        ma200: Optional[float],
        setup_type: str,
        direction: str = "LONG",
        pattern: Optional[PatternInfo] = None,
    ) -> Optional[TradeSetup]:
        """
        Calculate complete trade setup with smart entry, ATR-based stops, and level-based targets.

        Entry Strategy:
        - PULLBACK: Limit order at nearest support (buy the dip)
        - BREAKOUT: Breakout confirmation price (resistance + small buffer)
        - CUP_HANDLE: Breakout above rim
        - CHANNEL_ASC: Limit at channel support
        - SHORT setups: Mirror logic for shorts

        Stop Strategy:
        - ATR-based with structural awareness
        - Never more than MAX_RISK_PCT from entry

        Target Strategy:
        - Use actual resistance/support levels, not arbitrary extensions
        - Minimum R:R of MIN_RR required
        """
        if current_price <= 0 or atr <= 0:
            return None

        # Get next support/resistance levels for level-based targets
        supports_below = self._get_supports_below(key_supports, current_price, count=3)
        resistances_above = self._get_resistances_above(key_resistances, current_price, count=3)

        # =====================================================================
        # LONG SETUPS
        # =====================================================================
        if direction == "LONG":
            if setup_type == "PULLBACK":
                # Entry: At or near support (limit order)
                entry_price = nearest_support * 1.005  # Slightly above support for fill

                # Stop: Below support with ATR buffer
                stop_loss = nearest_support - (atr * 1.0)
                # Ensure stop is at least 1% below entry, max 5%
                min_stop = entry_price * (1 - self.MAX_RISK_PCT / 100)
                max_stop = entry_price * 0.99
                stop_loss = max(min_stop, min(stop_loss, max_stop))

                # Targets: Next resistance levels
                if resistances_above:
                    target_1 = resistances_above[0]
                    target_2 = resistances_above[1] if len(resistances_above) > 1 else target_1 * 1.03
                    target_3 = resistances_above[2] if len(resistances_above) > 2 else target_2 * 1.03
                else:
                    # Fallback: Use R-multiple targets
                    risk = entry_price - stop_loss
                    target_1 = entry_price + risk * 1.5
                    target_2 = entry_price + risk * 2.5
                    target_3 = entry_price + risk * 4.0

            elif setup_type == "BREAKOUT":
                # Entry: Confirmation above resistance
                entry_price = resistance * 1.005  # Breakout confirmation

                # Stop: Below breakout level with ATR buffer
                stop_loss = resistance - (atr * 1.5)
                # Ensure stop is at least 2% below entry, max 5%
                min_stop = entry_price * (1 - self.MAX_RISK_PCT / 100)
                max_stop = entry_price * 0.98
                stop_loss = max(min_stop, min(stop_loss, max_stop))

                # Targets: Measured move + next levels
                measured_move = resistance - nearest_support
                target_1 = entry_price + measured_move * 0.5
                target_2 = entry_price + measured_move
                target_3 = entry_price + measured_move * 1.5
                # Use actual resistance levels if better
                if resistances_above:
                    for i, r in enumerate(resistances_above[:3]):
                        if i == 0 and r > target_1:
                            target_1 = r
                        elif i == 1 and r > target_2:
                            target_2 = r
                        elif i == 2 and r > target_3:
                            target_3 = r

            elif setup_type == "CUP_HANDLE" and pattern:
                # Entry: Breakout above cup rim (resistance)
                entry_price = resistance * 1.005

                # Stop: Below handle (support) with tight ATR buffer
                stop_loss = nearest_support - (atr * 1.0)
                min_stop = entry_price * (1 - self.MAX_RISK_PCT / 100)
                max_stop = entry_price * 0.97
                stop_loss = max(min_stop, min(stop_loss, max_stop))

                # Targets: Cup depth projected above rim (classic C&H target)
                cup_depth_move = entry_price * (pattern.cup_depth_pct / 100)
                target_1 = entry_price + cup_depth_move * 0.5
                target_2 = entry_price + cup_depth_move
                target_3 = entry_price + cup_depth_move * 1.5

            elif setup_type.startswith("CHANNEL") and pattern:
                # Entry: At channel support (buy bottom of channel)
                entry_price = pattern.channel_support * 1.005

                # Stop: Below channel support with tight ATR
                stop_loss = pattern.channel_support - (atr * 0.5)
                min_stop = entry_price * (1 - self.MAX_RISK_PCT / 100)
                max_stop = entry_price * 0.98
                stop_loss = max(min_stop, min(stop_loss, max_stop))

                # Targets: Channel resistance and extensions
                target_1 = pattern.channel_resistance
                channel_range = pattern.channel_resistance - pattern.channel_support
                target_2 = pattern.channel_resistance + channel_range * 0.5
                target_3 = pattern.channel_resistance + channel_range

            else:  # NEUTRAL
                entry_price = current_price

                # Stop: Below nearest support
                stop_loss = nearest_support - (atr * 1.0)
                min_stop = entry_price * (1 - self.MAX_RISK_PCT / 100)
                stop_loss = max(min_stop, stop_loss)

                # Targets: Resistance levels
                if resistances_above:
                    target_1 = resistances_above[0]
                    target_2 = resistances_above[1] if len(resistances_above) > 1 else target_1 * 1.03
                    target_3 = resistances_above[2] if len(resistances_above) > 2 else target_2 * 1.03
                else:
                    risk = entry_price - stop_loss
                    target_1 = entry_price + risk * 1.5
                    target_2 = entry_price + risk * 2.5
                    target_3 = entry_price + risk * 4.0

        # =====================================================================
        # SHORT SETUPS
        # =====================================================================
        else:  # direction == "SHORT"
            if setup_type == "SHORT_PULLBACK":
                # Entry: At or near resistance (short the rally)
                entry_price = resistance * 0.995

                # Stop: Above resistance with ATR buffer
                stop_loss = resistance + (atr * 1.0)
                max_stop = entry_price * (1 + self.MAX_RISK_PCT / 100)
                min_stop = entry_price * 1.01
                stop_loss = min(max_stop, max(stop_loss, min_stop))

                # Targets: Next support levels below
                if supports_below:
                    target_1 = supports_below[0]
                    target_2 = supports_below[1] if len(supports_below) > 1 else target_1 * 0.97
                    target_3 = supports_below[2] if len(supports_below) > 2 else target_2 * 0.97
                else:
                    risk = stop_loss - entry_price
                    target_1 = entry_price - risk * 1.5
                    target_2 = entry_price - risk * 2.5
                    target_3 = entry_price - risk * 4.0

            elif setup_type == "SHORT_BREAKOUT":
                # Entry: Breakdown confirmation below support
                entry_price = nearest_support * 0.995

                # Stop: Above broken support with ATR buffer
                stop_loss = nearest_support + (atr * 1.5)
                max_stop = entry_price * (1 + self.MAX_RISK_PCT / 100)
                min_stop = entry_price * 1.02
                stop_loss = min(max_stop, max(stop_loss, min_stop))

                # Targets: Measured move down
                measured_move = resistance - nearest_support
                target_1 = entry_price - measured_move * 0.5
                target_2 = entry_price - measured_move
                target_3 = entry_price - measured_move * 1.5
                # Use actual support levels if better
                if supports_below:
                    for i, s in enumerate(supports_below[:3]):
                        if i == 0 and s < target_1:
                            target_1 = s
                        elif i == 1 and s < target_2:
                            target_2 = s
                        elif i == 2 and s < target_3:
                            target_3 = s

            elif setup_type.startswith("CHANNEL") and pattern and pattern.channel_type == "DESCENDING":
                # Entry: At channel resistance (sell top of channel)
                entry_price = pattern.channel_resistance * 0.995

                # Stop: Above channel resistance with tight ATR
                stop_loss = pattern.channel_resistance + (atr * 0.5)
                max_stop = entry_price * (1 + self.MAX_RISK_PCT / 100)
                min_stop = entry_price * 1.02
                stop_loss = min(max_stop, max(stop_loss, min_stop))

                # Targets: Channel support and extensions below
                target_1 = pattern.channel_support
                channel_range = pattern.channel_resistance - pattern.channel_support
                target_2 = pattern.channel_support - channel_range * 0.5
                target_3 = pattern.channel_support - channel_range

            else:  # SHORT_NEUTRAL
                entry_price = current_price

                # Stop: Above nearest resistance
                stop_loss = resistance + (atr * 1.0)
                max_stop = entry_price * (1 + self.MAX_RISK_PCT / 100)
                stop_loss = min(max_stop, stop_loss)

                # Targets: Support levels
                if supports_below:
                    target_1 = supports_below[0]
                    target_2 = supports_below[1] if len(supports_below) > 1 else target_1 * 0.97
                    target_3 = supports_below[2] if len(supports_below) > 2 else target_2 * 0.97
                else:
                    risk = stop_loss - entry_price
                    target_1 = entry_price - risk * 1.5
                    target_2 = entry_price - risk * 2.5
                    target_3 = entry_price - risk * 4.0

        # =====================================================================
        # Calculate risk and reward
        # =====================================================================
        if direction == "LONG":
            risk_per_share = entry_price - stop_loss
            reward_1 = target_1 - entry_price
            reward_2 = target_2 - entry_price
            reward_3 = target_3 - entry_price
        else:  # SHORT
            risk_per_share = stop_loss - entry_price
            reward_1 = entry_price - target_1
            reward_2 = entry_price - target_2
            reward_3 = entry_price - target_3

        # Sanity check on risk
        if risk_per_share <= 0:
            return None

        risk_reward_1 = reward_1 / risk_per_share if risk_per_share > 0 else 0
        risk_reward_2 = reward_2 / risk_per_share if risk_per_share > 0 else 0
        risk_reward_3 = reward_3 / risk_per_share if risk_per_share > 0 else 0

        risk_pct = (risk_per_share / entry_price) * 100

        # =====================================================================
        # R:R FILTERING - Skip poor setups
        # =====================================================================
        if risk_reward_1 < self.MIN_RR:
            return None  # Reject setup with poor R:R
        if risk_pct > self.MAX_RISK_PCT:
            return None  # Reject setup with excessive risk

        position_size_100 = int(100 / risk_per_share) if risk_per_share > 0 else 0
        position_size_500 = int(500 / risk_per_share) if risk_per_share > 0 else 0
        position_size_1000 = int(1000 / risk_per_share) if risk_per_share > 0 else 0

        return TradeSetup(
            direction=direction,
            entry_price=round(entry_price, 2),
            stop_loss=round(stop_loss, 2),
            target_1=round(target_1, 2),
            target_2=round(target_2, 2),
            target_3=round(target_3, 2),
            risk_per_share=round(risk_per_share, 2),
            reward_1=round(reward_1, 2),
            reward_2=round(reward_2, 2),
            reward_3=round(reward_3, 2),
            risk_reward_1=round(risk_reward_1, 2),
            risk_reward_2=round(risk_reward_2, 2),
            risk_reward_3=round(risk_reward_3, 2),
            risk_pct=round(risk_pct, 2),
            position_size_100=position_size_100,
            position_size_500=position_size_500,
            position_size_1000=position_size_1000,
        )

    def _calculate_deal_score(
        self,
        current_price: float,
        ma20: Optional[float],
        ma50: Optional[float],
        ma200: Optional[float],
        rsi: float,
        momentum_5d: float,
        momentum_20d: float,
        support_distance_pct: float,
        resistance_distance_pct: float,
        volume_at_support: float,
        consolidation_score: float,
        volume_trend: float,
        setup_type: str,
        pattern: Optional[PatternInfo] = None,
        direction: str = "LONG",
        sentiment: Optional[SentimentInfo] = None,
    ) -> int:
        """
        Calculate deal score (0-100).

        Long Scoring:
        - Trend Strength (25 pts): MA alignment, distance above 200MA
        - Momentum (25 pts): 5d/20d momentum, recovering from pullback
        - Volume at Support (20 pts): High volume = stronger support
        - Breakout/Pattern Potential (20 pts): Near resistance + consolidation
        - RSI Bonus (10 pts): Oversold in uptrend = opportunity

        Short Scoring:
        - Trend Weakness (25 pts): MA alignment bearish, distance below 200MA
        - Momentum (25 pts): Negative momentum, failing rallies
        - Volume at Resistance (20 pts): High volume = stronger resistance
        - Breakdown Potential (20 pts): Near support breakdown
        - RSI Bonus (10 pts): Overbought in downtrend = opportunity
        """
        score = 0

        # =====================================================================
        # LONG SCORING (Above 200MA)
        # =====================================================================
        if direction == "LONG":
            if not ma200 or current_price <= ma200:
                return 0  # Not in uptrend - reject for longs

            # === TREND STRENGTH (max 25 pts) ===
            dist_above_200 = (current_price - ma200) / ma200 * 100

            if 5 <= dist_above_200 <= 15:
                score += 10
            elif 2 <= dist_above_200 < 5:
                score += 7
            elif 15 < dist_above_200 <= 25:
                score += 5
            elif dist_above_200 > 25:
                score += 2

            if ma20 and ma50:
                if current_price > ma20 > ma50 > ma200:
                    score += 15
                elif current_price > ma50 > ma200:
                    score += 10
                elif current_price < ma20 and current_price > ma200:
                    score += 8

            # === MOMENTUM (max 25 pts) ===
            if momentum_5d > 3:
                score += 10
            elif momentum_5d > 0:
                score += 7
            elif momentum_5d > -3:
                score += 4
            elif momentum_5d > -7:
                score += 2

            if momentum_20d > 5:
                score += 10
            elif momentum_20d > 0:
                score += 7
            elif momentum_20d > -5:
                score += 4

            if momentum_20d < 0 < momentum_5d:
                score += 5

            # === VOLUME AT SUPPORT (max 20 pts) ===
            if volume_at_support > 15:
                score += 20
            elif volume_at_support > 10:
                score += 15
            elif volume_at_support > 5:
                score += 10
            elif volume_at_support > 2:
                score += 5

            # === PATTERN/BREAKOUT POTENTIAL (max 20 pts) ===
            if pattern and pattern.cup_and_handle:
                score += 15  # Cup and handle is a strong pattern
            elif pattern and pattern.channel_type == "ASCENDING":
                score += 12

            if resistance_distance_pct < 3:
                if consolidation_score > 70:
                    score += 20
                elif consolidation_score > 50:
                    score += 15
                else:
                    score += 10
            elif resistance_distance_pct < 7:
                if consolidation_score > 70:
                    score += 12
                elif consolidation_score > 50:
                    score += 8
                else:
                    score += 5

            if volume_trend > 1.5:
                score += 5
            elif volume_trend > 1.2:
                score += 3

            # === RSI BONUS (max 10 pts) ===
            if rsi < 35:
                score += 10
            elif rsi < 45:
                score += 7
            elif rsi < 55:
                score += 4

        # =====================================================================
        # SHORT SCORING (Below 200MA)
        # =====================================================================
        else:  # direction == "SHORT"
            if not ma200 or current_price >= ma200:
                return 0  # Not in downtrend - reject for shorts

            # === TREND WEAKNESS (max 25 pts) ===
            dist_below_200 = (ma200 - current_price) / ma200 * 100

            if 5 <= dist_below_200 <= 15:
                score += 10
            elif 2 <= dist_below_200 < 5:
                score += 7
            elif 15 < dist_below_200 <= 25:
                score += 5
            elif dist_below_200 > 25:
                score += 2

            if ma20 and ma50:
                if current_price < ma20 < ma50 < ma200:
                    score += 15  # Perfect bearish alignment
                elif current_price < ma50 < ma200:
                    score += 10
                elif current_price > ma20 and current_price < ma200:
                    score += 8  # Rally to 20MA in downtrend

            # === BEARISH MOMENTUM (max 25 pts) ===
            if momentum_5d < -3:
                score += 10
            elif momentum_5d < 0:
                score += 7
            elif momentum_5d < 3:
                score += 4
            elif momentum_5d < 7:
                score += 2

            if momentum_20d < -5:
                score += 10
            elif momentum_20d < 0:
                score += 7
            elif momentum_20d < 5:
                score += 4

            # Failed rally bonus (was up, now down)
            if momentum_20d > 0 > momentum_5d:
                score += 5

            # === VOLUME AT RESISTANCE (max 20 pts) ===
            # For shorts, high volume at current level indicates selling pressure
            if resistance_distance_pct < 5 and volume_at_support > 10:
                score += 15
            elif resistance_distance_pct < 10:
                score += 10
            elif volume_at_support > 5:
                score += 5

            # === BREAKDOWN POTENTIAL (max 20 pts) ===
            if pattern and pattern.channel_type == "DESCENDING":
                score += 12

            if support_distance_pct < 3:
                if consolidation_score > 70:
                    score += 20
                elif consolidation_score > 50:
                    score += 15
                else:
                    score += 10
            elif support_distance_pct < 7:
                if consolidation_score > 70:
                    score += 12
                elif consolidation_score > 50:
                    score += 8
                else:
                    score += 5

            if volume_trend > 1.5:
                score += 5
            elif volume_trend > 1.2:
                score += 3

            # === RSI BONUS for shorts (max 10 pts) ===
            # Overbought in downtrend = short opportunity
            if rsi > 65:
                score += 10
            elif rsi > 55:
                score += 7
            elif rsi > 45:
                score += 4

        # =====================================================================
        # SENTIMENT ADJUSTMENT (max +/- 20 pts)
        # Uses time-weighted sentiment and momentum for more robust signals
        # =====================================================================
        if sentiment and sentiment.tweet_count_30d > 0:
            # === WEIGHTED SENTIMENT ALIGNMENT (max ±12 pts) ===
            ws = sentiment.weighted_sentiment
            if direction == "LONG":
                if ws > 0.4:
                    score += 12  # Strong bullish sentiment supports long
                elif ws > 0.2:
                    score += 8
                elif ws > 0.1:
                    score += 4
                elif ws < -0.4:
                    score -= 12  # Strong bearish sentiment contradicts long
                elif ws < -0.2:
                    score -= 8
                elif ws < -0.1:
                    score -= 4
            else:  # SHORT
                if ws < -0.4:
                    score += 12  # Strong bearish sentiment supports short
                elif ws < -0.2:
                    score += 8
                elif ws < -0.1:
                    score += 4
                elif ws > 0.4:
                    score -= 12  # Strong bullish sentiment contradicts short
                elif ws > 0.2:
                    score -= 8
                elif ws > 0.1:
                    score -= 4

            # === SENTIMENT MOMENTUM BONUS (max ±5 pts) ===
            # Momentum > 0 means sentiment is improving (recent > older)
            mom = sentiment.momentum
            if direction == "LONG":
                if mom > 0.15:
                    score += 5  # Sentiment improving, good for longs
                elif mom > 0.05:
                    score += 3
                elif mom < -0.15:
                    score -= 3  # Sentiment deteriorating, bad for longs
            else:  # SHORT
                if mom < -0.15:
                    score += 5  # Sentiment deteriorating, good for shorts
                elif mom < -0.05:
                    score += 3
                elif mom > 0.15:
                    score -= 3  # Sentiment improving, bad for shorts

            # === VOLUME SPIKE BONUS (max +3 pts) ===
            # High tweet volume = more attention = potential move
            if sentiment.volume_spike:
                score += 3

        return min(max(score, 0), 100)  # Clamp between 0-100

    def export_to_csv(self, output_path: Path) -> None:
        """Export results to CSV."""
        if not self.results:
            logger.warning("No results to export")
            return

        rows = []
        for r in self.results:
            row = {
                "ticker": r.ticker,
                "direction": r.direction,
                "deal_score": r.deal_score,
                "deal_rating": r.deal_rating,
                "setup_type": r.setup_type,
                "current_price": r.current_price,
                # Trade Setup
                "entry_price": r.trade_setup.entry_price if r.trade_setup else None,
                "stop_loss": r.trade_setup.stop_loss if r.trade_setup else None,
                "target_1": r.trade_setup.target_1 if r.trade_setup else None,
                "target_2": r.trade_setup.target_2 if r.trade_setup else None,
                "target_3": r.trade_setup.target_3 if r.trade_setup else None,
                "risk_%": r.trade_setup.risk_pct if r.trade_setup else None,
                "risk_$": r.trade_setup.risk_per_share if r.trade_setup else None,
                "R:R_T1": r.trade_setup.risk_reward_1 if r.trade_setup else None,
                "R:R_T2": r.trade_setup.risk_reward_2 if r.trade_setup else None,
                "R:R_T3": r.trade_setup.risk_reward_3 if r.trade_setup else None,
                "shares_$100_risk": r.trade_setup.position_size_100 if r.trade_setup else None,
                "shares_$500_risk": r.trade_setup.position_size_500 if r.trade_setup else None,
                "shares_$1000_risk": r.trade_setup.position_size_1000 if r.trade_setup else None,
                # Pattern Info
                "cup_and_handle": r.pattern.cup_and_handle if r.pattern else False,
                "cup_depth_%": r.pattern.cup_depth_pct if r.pattern and r.pattern.cup_and_handle else None,
                "channel_type": r.pattern.channel_type if r.pattern else None,
                "channel_support": r.pattern.channel_support if r.pattern and r.pattern.channel_type else None,
                "channel_resistance": r.pattern.channel_resistance if r.pattern and r.pattern.channel_type else None,
                # Sentiment Info (time-weighted)
                "weighted_sentiment": r.sentiment.weighted_sentiment if r.sentiment else None,
                "sentiment_momentum": r.sentiment.momentum if r.sentiment else None,
                "raw_sentiment": r.sentiment.raw_sentiment if r.sentiment else None,
                "tweet_count_30d": r.sentiment.tweet_count_30d if r.sentiment else 0,
                "tweet_count_7d": r.sentiment.tweet_count_7d if r.sentiment else 0,
                "positive_pct": r.sentiment.positive_pct if r.sentiment else None,
                "negative_pct": r.sentiment.negative_pct if r.sentiment else None,
                "sentiment_aligned": r.sentiment_aligned,
                "volume_spike": r.sentiment.volume_spike if r.sentiment else False,
                # Analysis
                "ath": r.ath,
                "ath_discount_%": r.ath_discount_pct,
                "rsi": r.rsi,
                "momentum_5d_%": r.momentum_5d,
                "momentum_20d_%": r.momentum_20d,
                "ma200": r.ma200,
                "nearest_support": r.nearest_support,
                "support_dist_%": r.support_distance_pct,
                "volume_at_support_%": r.volume_at_support,
                "resistance": r.resistance,
                "resistance_dist_%": r.resistance_distance_pct,
                "consolidation_score": r.consolidation_score,
                "atr_%": r.atr_pct,
                "key_supports": str(r.key_supports),
                # LLM Analysis (if available)
                "llm_entry": r.llm_analysis.entry_price if r.llm_analysis else None,
                "llm_stop": r.llm_analysis.stop_loss if r.llm_analysis else None,
                "llm_t1": r.llm_analysis.target_1 if r.llm_analysis else None,
                "llm_t2": r.llm_analysis.target_2 if r.llm_analysis else None,
                "llm_t3": r.llm_analysis.target_3 if r.llm_analysis else None,
                "llm_confidence": r.llm_analysis.confidence if r.llm_analysis else None,
                "llm_thesis": r.llm_analysis.thesis if r.llm_analysis else None,
            }
            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        logger.info(f"Exported {len(rows)} stocks to {output_path}")

    def export_to_json(self, output_path: Path) -> None:
        """Export results to JSON."""
        if not self.results:
            return

        valid_results = [r for r in self.results if r.deal_score > 0]

        def build_trade_setup_dict(r):
            """Build trade setup dictionary for a result."""
            if not r.trade_setup:
                return None
            return {
                "direction": r.trade_setup.direction,
                "entry": r.trade_setup.entry_price,
                "stop_loss": r.trade_setup.stop_loss,
                "target_1": r.trade_setup.target_1,
                "target_2": r.trade_setup.target_2,
                "target_3": r.trade_setup.target_3,
                "risk_%": r.trade_setup.risk_pct,
                "risk_$": r.trade_setup.risk_per_share,
                "R:R_T1": r.trade_setup.risk_reward_1,
                "R:R_T2": r.trade_setup.risk_reward_2,
                "R:R_T3": r.trade_setup.risk_reward_3,
                "position_size": {
                    "$100_risk": r.trade_setup.position_size_100,
                    "$500_risk": r.trade_setup.position_size_500,
                    "$1000_risk": r.trade_setup.position_size_1000,
                },
            }

        def build_pattern_dict(r):
            """Build pattern dictionary for a result."""
            if not r.pattern:
                return None
            return {
                "cup_and_handle": r.pattern.cup_and_handle,
                "cup_depth_%": r.pattern.cup_depth_pct if r.pattern.cup_and_handle else None,
                "channel_type": r.pattern.channel_type or None,
                "channel_support": r.pattern.channel_support if r.pattern.channel_type else None,
                "channel_resistance": r.pattern.channel_resistance if r.pattern.channel_type else None,
                "channel_width_%": r.pattern.channel_width_pct if r.pattern.channel_type else None,
            }

        def build_sentiment_dict(r):
            """Build sentiment dictionary for a result."""
            if not r.sentiment:
                return None
            return {
                "weighted_sentiment": r.sentiment.weighted_sentiment,
                "momentum": r.sentiment.momentum,
                "raw_sentiment": r.sentiment.raw_sentiment,
                "tweet_count_30d": r.sentiment.tweet_count_30d,
                "tweet_count_7d": r.sentiment.tweet_count_7d,
                "positive_pct": r.sentiment.positive_pct,
                "negative_pct": r.sentiment.negative_pct,
                "neutral_pct": r.sentiment.neutral_pct,
                "volume_spike": r.sentiment.volume_spike,
                "dominant_category": r.sentiment.dominant_category,
                "sentiment_aligned": r.sentiment_aligned,
            }

        def build_llm_dict(r):
            """Build LLM analysis dictionary for a result."""
            if not r.llm_analysis:
                return None
            return {
                "entry": {
                    "price": r.llm_analysis.entry_price,
                    "type": r.llm_analysis.entry_type,
                    "reasoning": r.llm_analysis.entry_reasoning,
                },
                "stop_loss": {
                    "price": r.llm_analysis.stop_loss,
                    "reasoning": r.llm_analysis.stop_reasoning,
                },
                "targets": {
                    "t1": r.llm_analysis.target_1,
                    "t2": r.llm_analysis.target_2,
                    "t3": r.llm_analysis.target_3,
                    "reasoning": r.llm_analysis.target_reasoning,
                },
                "confidence": r.llm_analysis.confidence,
                "risk_reward": r.llm_analysis.risk_reward,
                "key_risks": r.llm_analysis.key_risks,
                "thesis": r.llm_analysis.thesis,
                "model": r.llm_analysis.model_used,
                "timestamp": r.llm_analysis.analysis_timestamp,
            }

        long_results = [r for r in valid_results if r.direction == "LONG"]
        short_results = [r for r in valid_results if r.direction == "SHORT"]

        data = {
            "scan_date": datetime.now().isoformat(),
            "total_stocks": len(self.results),
            "long_setups": len(long_results),
            "short_setups": len(short_results),
            "rejected_stocks": len(self.results) - len(valid_results),
            "top_deals": [
                {
                    "ticker": r.ticker,
                    "direction": r.direction,
                    "deal_score": r.deal_score,
                    "deal_rating": r.deal_rating,
                    "setup_type": r.setup_type,
                    "current_price": r.current_price,
                    "trade_setup": build_trade_setup_dict(r),
                    "pattern": build_pattern_dict(r),
                    "sentiment": build_sentiment_dict(r),
                    "llm_analysis": build_llm_dict(r),
                    "analysis": {
                        "momentum_5d": r.momentum_5d,
                        "momentum_20d": r.momentum_20d,
                        "rsi": r.rsi,
                        "resistance": r.resistance,
                        "resistance_distance_%": r.resistance_distance_pct,
                        "consolidation_score": r.consolidation_score,
                        "nearest_support": r.nearest_support,
                        "volume_at_support_%": r.volume_at_support,
                        "key_supports": r.key_supports,
                    },
                }
                for r in valid_results[:50]
            ],
            "breakout_setups": [
                {
                    "ticker": r.ticker,
                    "deal_score": r.deal_score,
                    "setup_type": r.setup_type,
                    "trade_setup": build_trade_setup_dict(r),
                }
                for r in long_results
                if r.setup_type == "BREAKOUT"
            ][:20],
            "pullback_setups": [
                {
                    "ticker": r.ticker,
                    "deal_score": r.deal_score,
                    "setup_type": r.setup_type,
                    "trade_setup": build_trade_setup_dict(r),
                }
                for r in long_results
                if r.setup_type == "PULLBACK"
            ][:20],
            "cup_and_handle_setups": [
                {
                    "ticker": r.ticker,
                    "deal_score": r.deal_score,
                    "pattern": build_pattern_dict(r),
                    "trade_setup": build_trade_setup_dict(r),
                }
                for r in long_results
                if r.pattern and r.pattern.cup_and_handle
            ][:20],
            "channel_setups": [
                {
                    "ticker": r.ticker,
                    "direction": r.direction,
                    "deal_score": r.deal_score,
                    "pattern": build_pattern_dict(r),
                    "trade_setup": build_trade_setup_dict(r),
                }
                for r in valid_results
                if r.pattern and r.pattern.channel_type
            ][:20],
            "short_setups": [
                {
                    "ticker": r.ticker,
                    "deal_score": r.deal_score,
                    "setup_type": r.setup_type,
                    "trade_setup": build_trade_setup_dict(r),
                }
                for r in short_results
            ][:20],
        }

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Exported to {output_path}")

    def print_summary(self, top_n: int = 20) -> None:
        """Print summary of best deals."""
        valid_results = [r for r in self.results if r.deal_score > 0]
        long_results = [r for r in valid_results if r.direction == "LONG"]
        short_results = [r for r in valid_results if r.direction == "SHORT"]

        # Check if sentiment is available (using 30d count for time-weighted analysis)
        has_sentiment = any(r.sentiment and r.sentiment.tweet_count_30d > 0 for r in valid_results)

        print("\n" + "=" * 150)
        print("TOP STOCK DEALS - COMPLETE TRADE SETUPS (LONG & SHORT)")
        if has_sentiment:
            print("📊 SENTIMENT ANALYSIS ENABLED")
        print("=" * 150)
        print(f"\nScanned {len(self.results)} stocks | {len(long_results)} LONG | {len(short_results)} SHORT | Top {top_n} shown\n")

        # =====================================================================
        # TOP LONG SETUPS
        # =====================================================================
        print("🟢 TOP LONG SETUPS")
        print("-" * 150)
        if has_sentiment:
            header = (
                f"{'Rank':<4} {'Ticker':<6} {'Score':<5} {'Setup':<16} {'Price':>8} "
                f"{'Entry':>8} {'Stop':>8} {'T1':>8} {'R:R':>5} {'Sent':>6} {'Mom':>6} {'Twts':>5} {'Pattern':<12}"
            )
        else:
            header = (
                f"{'Rank':<4} {'Ticker':<6} {'Score':<5} {'Setup':<16} {'Price':>8} "
                f"{'Entry':>8} {'Stop':>8} {'T1':>8} {'T2':>8} {'Risk%':>6} {'R:R':>5} {'Pattern':<12}"
            )
        print(header)
        print("-" * 155)

        for i, r in enumerate(long_results[:top_n], 1):
            ts = r.trade_setup
            pattern_str = ""
            if r.pattern:
                if r.pattern.cup_and_handle:
                    pattern_str = "CUP&HANDLE"
                elif r.pattern.channel_type:
                    pattern_str = f"CH:{r.pattern.channel_type[:3]}"

            sent_str = ""
            mom_str = ""
            tweets_str = ""
            if has_sentiment and r.sentiment:
                sent_str = f"{r.sentiment.weighted_sentiment:>+5.2f}" if r.sentiment.tweet_count_30d > 0 else "  N/A"
                mom_str = f"{r.sentiment.momentum:>+5.2f}" if r.sentiment.tweet_count_30d > 0 else "  N/A"
                tweets_str = f"{r.sentiment.tweet_count_30d:>4}" if r.sentiment.tweet_count_30d > 0 else "   0"

            if ts:
                if has_sentiment:
                    print(
                        f"{i:<4} {r.ticker:<6} {r.deal_score:<5} {r.setup_type:<16} ${r.current_price:>6.2f} "
                        f"${ts.entry_price:>6.2f} ${ts.stop_loss:>6.2f} ${ts.target_1:>6.2f} "
                        f"{ts.risk_reward_1:>4.1f}R {sent_str:>6} {mom_str:>6} {tweets_str:>5} {pattern_str:<12}"
                    )
                else:
                    print(
                        f"{i:<4} {r.ticker:<6} {r.deal_score:<5} {r.setup_type:<16} ${r.current_price:>6.2f} "
                        f"${ts.entry_price:>6.2f} ${ts.stop_loss:>6.2f} ${ts.target_1:>6.2f} ${ts.target_2:>6.2f} "
                        f"{ts.risk_pct:>5.1f}% {ts.risk_reward_1:>4.1f}R {pattern_str:<12}"
                    )
            else:
                if has_sentiment:
                    print(
                        f"{i:<4} {r.ticker:<6} {r.deal_score:<5} {r.setup_type:<16} ${r.current_price:>6.2f} "
                        f"{'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>5} {sent_str:>6} {mom_str:>6} {tweets_str:>5} {pattern_str:<12}"
                    )
                else:
                    print(
                        f"{i:<4} {r.ticker:<6} {r.deal_score:<5} {r.setup_type:<16} ${r.current_price:>6.2f} "
                        f"{'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>6} {'N/A':>5} {pattern_str:<12}"
                    )

        # =====================================================================
        # TOP SHORT SETUPS
        # =====================================================================
        if short_results:
            print("\n" + "=" * 155)
            print("🔴 TOP SHORT SETUPS")
            print("-" * 155)
            print(header)
            print("-" * 155)

            for i, r in enumerate(short_results[:top_n], 1):
                ts = r.trade_setup
                pattern_str = ""
                if r.pattern and r.pattern.channel_type:
                    pattern_str = f"CH:{r.pattern.channel_type[:3]}"

                sent_str = ""
                mom_str = ""
                tweets_str = ""
                if has_sentiment and r.sentiment:
                    sent_str = f"{r.sentiment.weighted_sentiment:>+5.2f}" if r.sentiment.tweet_count_30d > 0 else "  N/A"
                    mom_str = f"{r.sentiment.momentum:>+5.2f}" if r.sentiment.tweet_count_30d > 0 else "  N/A"
                    tweets_str = f"{r.sentiment.tweet_count_30d:>4}" if r.sentiment.tweet_count_30d > 0 else "   0"

                if ts:
                    if has_sentiment:
                        print(
                            f"{i:<4} {r.ticker:<6} {r.deal_score:<5} {r.setup_type:<16} ${r.current_price:>6.2f} "
                            f"${ts.entry_price:>6.2f} ${ts.stop_loss:>6.2f} ${ts.target_1:>6.2f} "
                            f"{ts.risk_reward_1:>4.1f}R {sent_str:>6} {mom_str:>6} {tweets_str:>5} {pattern_str:<12}"
                        )
                    else:
                        print(
                            f"{i:<4} {r.ticker:<6} {r.deal_score:<5} {r.setup_type:<16} ${r.current_price:>6.2f} "
                            f"${ts.entry_price:>6.2f} ${ts.stop_loss:>6.2f} ${ts.target_1:>6.2f} ${ts.target_2:>6.2f} "
                            f"{ts.risk_pct:>5.1f}% {ts.risk_reward_1:>4.1f}R {pattern_str:<12}"
                        )
                else:
                    if has_sentiment:
                        print(
                            f"{i:<4} {r.ticker:<6} {r.deal_score:<5} {r.setup_type:<16} ${r.current_price:>6.2f} "
                            f"{'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>5} {sent_str:>6} {mom_str:>6} {tweets_str:>5} {pattern_str:<12}"
                        )
                    else:
                        print(
                            f"{i:<4} {r.ticker:<6} {r.deal_score:<5} {r.setup_type:<16} ${r.current_price:>6.2f} "
                            f"{'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>6} {'N/A':>5} {pattern_str:<12}"
                        )

        # Category breakdown
        print("\n" + "=" * 140)
        print("DEAL DISTRIBUTION")
        print("=" * 140)

        excellent = len([r for r in valid_results if r.deal_score >= 70])
        good = len([r for r in valid_results if 55 <= r.deal_score < 70])
        fair = len([r for r in valid_results if 40 <= r.deal_score < 55])
        below = len([r for r in valid_results if r.deal_score < 40])
        rejected = len([r for r in self.results if r.deal_score == 0])

        print(f"   EXCELLENT (70+): {excellent} stocks")
        print(f"   GOOD (55-69):    {good} stocks")
        print(f"   FAIR (40-54):    {fair} stocks")
        print(f"   BELOW 40:        {below} stocks")
        print(f"   REJECTED (no trend): {rejected} stocks")

        # Pattern summary
        cup_handles = [r for r in valid_results if r.pattern and r.pattern.cup_and_handle]
        channels = [r for r in valid_results if r.pattern and r.pattern.channel_type]

        if cup_handles or channels:
            print("\n" + "=" * 140)
            print("PATTERN SETUPS")
            print("=" * 140)
            if cup_handles:
                print(f"\n>>> CUP & HANDLE ({len(cup_handles)} stocks)")
                for r in sorted(cup_handles, key=lambda x: x.deal_score, reverse=True)[:5]:
                    ts = r.trade_setup
                    if ts:
                        print(
                            f"    {r.ticker:<6} Score:{r.deal_score:<3} Cup:{r.pattern.cup_depth_pct}% Entry:${ts.entry_price:.2f} R:R:{ts.risk_reward_1:.1f}"
                        )

            if channels:
                asc_channels = [r for r in channels if r.pattern.channel_type == "ASCENDING"]
                desc_channels = [r for r in channels if r.pattern.channel_type == "DESCENDING"]
                horiz_channels = [r for r in channels if r.pattern.channel_type == "HORIZONTAL"]

                if asc_channels:
                    print(f"\n>>> ASCENDING CHANNELS ({len(asc_channels)} stocks)")
                    for r in sorted(asc_channels, key=lambda x: x.deal_score, reverse=True)[:5]:
                        ts = r.trade_setup
                        if ts:
                            print(
                                f"    {r.ticker:<6} Score:{r.deal_score:<3} Width:{r.pattern.channel_width_pct:.1f}% Pos:{r.pattern.channel_position_pct:.0f}% R:R:{ts.risk_reward_1:.1f}"
                            )

                if desc_channels:
                    print(f"\n>>> DESCENDING CHANNELS ({len(desc_channels)} stocks) - SHORT CANDIDATES")
                    for r in sorted(desc_channels, key=lambda x: x.deal_score, reverse=True)[:5]:
                        ts = r.trade_setup
                        if ts:
                            print(
                                f"    {r.ticker:<6} Score:{r.deal_score:<3} Width:{r.pattern.channel_width_pct:.1f}% Pos:{r.pattern.channel_position_pct:.0f}% R:R:{ts.risk_reward_1:.1f}"
                            )

        # Detailed trade setups by type
        print("\n" + "=" * 140)
        print("DETAILED LONG SETUPS BY TYPE")
        print("=" * 140)

        breakouts = [r for r in long_results if r.setup_type == "BREAKOUT"]
        pullbacks = [r for r in long_results if r.setup_type == "PULLBACK"]
        cup_handle_setups = [r for r in long_results if r.setup_type == "CUP_HANDLE"]

        print(f"\n>>> BREAKOUT SETUPS ({len(breakouts)} stocks)")
        print("-" * 140)
        print(
            f"    {'Ticker':<6} {'Score':<5} {'Entry':>8} {'Stop':>8} {'T1':>8} {'T2':>8} {'Risk%':>6} {'R:R':>5} {'Resist':>9} {'Consol':>7}"
        )
        print("-" * 140)
        for r in sorted(breakouts, key=lambda x: x.deal_score, reverse=True)[:10]:
            ts = r.trade_setup
            if ts:
                print(
                    f"    {r.ticker:<6} {r.deal_score:<5} ${ts.entry_price:>6.2f} ${ts.stop_loss:>6.2f} "
                    f"${ts.target_1:>6.2f} ${ts.target_2:>6.2f} {ts.risk_pct:>5.1f}% {ts.risk_reward_1:>4.1f}R "
                    f"${r.resistance:>7.2f} {r.consolidation_score:>6.0f}"
                )

        print(f"\n>>> PULLBACK SETUPS ({len(pullbacks)} stocks)")
        print("-" * 130)
        print(
            f"    {'Ticker':<6} {'Score':<5} {'Entry':>8} {'Stop':>8} {'T1':>8} {'T2':>8} {'Risk%':>6} {'R:R':>5} {'Support':>9} {'Vol@Sup':>8}"
        )
        print("-" * 130)
        for r in sorted(pullbacks, key=lambda x: x.deal_score, reverse=True)[:10]:
            ts = r.trade_setup
            if ts:
                print(
                    f"    {r.ticker:<6} {r.deal_score:<5} ${ts.entry_price:>6.2f} ${ts.stop_loss:>6.2f} "
                    f"${ts.target_1:>6.2f} ${ts.target_2:>6.2f} {ts.risk_pct:>5.1f}% {ts.risk_reward_1:>4.1f}R "
                    f"${r.nearest_support:>7.2f} {r.volume_at_support:>7.1f}%"
                )

        # Best Risk/Reward setups
        print("\n" + "=" * 130)
        print("BEST RISK/REWARD SETUPS (R:R >= 2.0)")
        print("=" * 130)

        best_rr = sorted(
            [r for r in valid_results if r.trade_setup and r.trade_setup.risk_reward_1 >= 2.0],
            key=lambda x: x.trade_setup.risk_reward_1,
            reverse=True,
        )[:10]

        print(
            f"\n    {'Ticker':<6} {'Score':<5} {'Setup':<10} {'Entry':>8} {'Stop':>8} {'T1':>8} {'Risk$':>7} {'R:R':>5} {'Shares@$500':>12}"
        )
        print("-" * 130)
        for r in best_rr:
            ts = r.trade_setup
            print(
                f"    {r.ticker:<6} {r.deal_score:<5} {r.setup_type:<10} ${ts.entry_price:>6.2f} ${ts.stop_loss:>6.2f} "
                f"${ts.target_1:>6.2f} ${ts.risk_per_share:>6.2f} {ts.risk_reward_1:>4.1f}R {ts.position_size_500:>11}"
            )

        # Lowest Risk setups (tight stops)
        print("\n" + "=" * 130)
        print("LOWEST RISK SETUPS (Risk < 3%)")
        print("=" * 130)

        low_risk = sorted(
            [r for r in valid_results if r.trade_setup and r.trade_setup.risk_pct < 3 and r.trade_setup.risk_reward_1 >= 1.5],
            key=lambda x: x.trade_setup.risk_pct,
        )[:10]

        print(f"\n    {'Ticker':<6} {'Score':<5} {'Setup':<10} {'Entry':>8} {'Stop':>8} {'T1':>8} {'Risk%':>6} {'R:R':>5}")
        print("-" * 130)
        for r in low_risk:
            ts = r.trade_setup
            print(
                f"    {r.ticker:<6} {r.deal_score:<5} {r.setup_type:<10} ${ts.entry_price:>6.2f} ${ts.stop_loss:>6.2f} "
                f"${ts.target_1:>6.2f} {ts.risk_pct:>5.1f}% {ts.risk_reward_1:>4.1f}R"
            )

        # Sentiment aligned setups (only if sentiment enabled)
        if has_sentiment:
            sentiment_aligned = [r for r in valid_results if r.sentiment_aligned and r.trade_setup and r.trade_setup.risk_reward_1 >= 1.5]

            if sentiment_aligned:
                print("\n" + "=" * 130)
                print("📊 SENTIMENT-ALIGNED SETUPS (Technical + Social Confirmation)")
                print("=" * 130)

                long_aligned = [r for r in sentiment_aligned if r.direction == "LONG"]
                short_aligned = [r for r in sentiment_aligned if r.direction == "SHORT"]

                if long_aligned:
                    print(f"\n>>> BULLISH SENTIMENT + LONG SETUP ({len(long_aligned)} stocks)")
                    print(
                        f"    {'Ticker':<6} {'Score':<5} {'Setup':<12} {'Entry':>8} {'T1':>8} {'R:R':>5} {'Sent':>6} {'Mom':>6} {'Twts':>5}"
                    )
                    print("-" * 110)
                    for r in sorted(long_aligned, key=lambda x: x.deal_score, reverse=True)[:10]:
                        ts = r.trade_setup
                        print(
                            f"    {r.ticker:<6} {r.deal_score:<5} {r.setup_type:<12} ${ts.entry_price:>6.2f} "
                            f"${ts.target_1:>6.2f} {ts.risk_reward_1:>4.1f}R "
                            f"{r.sentiment.weighted_sentiment:>+5.2f} {r.sentiment.momentum:>+5.2f} {r.sentiment.tweet_count_30d:>4}"
                        )

                if short_aligned:
                    print(f"\n>>> BEARISH SENTIMENT + SHORT SETUP ({len(short_aligned)} stocks)")
                    print(
                        f"    {'Ticker':<6} {'Score':<5} {'Setup':<12} {'Entry':>8} {'T1':>8} {'R:R':>5} {'Sent':>6} {'Mom':>6} {'Twts':>5}"
                    )
                    print("-" * 110)
                    for r in sorted(short_aligned, key=lambda x: x.deal_score, reverse=True)[:10]:
                        ts = r.trade_setup
                        print(
                            f"    {r.ticker:<6} {r.deal_score:<5} {r.setup_type:<12} ${ts.entry_price:>6.2f} "
                            f"${ts.target_1:>6.2f} {ts.risk_reward_1:>4.1f}R "
                            f"{r.sentiment.weighted_sentiment:>+5.2f} {r.sentiment.momentum:>+5.2f} {r.sentiment.tweet_count_30d:>4}"
                        )

                # Top tickers by tweet volume
                high_volume_tweets = sorted(
                    [r for r in valid_results if r.sentiment and r.sentiment.tweet_count_30d >= 5],
                    key=lambda x: x.sentiment.tweet_count_30d,
                    reverse=True,
                )[:10]

                if high_volume_tweets:
                    print(f"\n>>> HIGH TWEET VOLUME (most discussed stocks)")
                    print(
                        f"    {'Ticker':<6} {'Dir':<5} {'Score':<5} {'Twts':>5} {'Sent':>6} {'Mom':>6} {'Pos%':>6} {'Neg%':>6} {'Category':<15}"
                    )
                    print("-" * 110)
                    for r in high_volume_tweets:
                        print(
                            f"    {r.ticker:<6} {r.direction:<5} {r.deal_score:<5} "
                            f"{r.sentiment.tweet_count_30d:>4} {r.sentiment.weighted_sentiment:>+5.2f} "
                            f"{r.sentiment.momentum:>+5.2f} {r.sentiment.positive_pct:>5.1f}% {r.sentiment.negative_pct:>5.1f}% "
                            f"{r.sentiment.dominant_category:<15}"
                        )

        # =====================================================================
        # LLM-REFINED TRADE SETUPS
        # =====================================================================
        llm_analyzed = [r for r in valid_results if r.llm_analysis is not None]

        if llm_analyzed:
            print("\n" + "=" * 140)
            print("LLM-REFINED TRADE SETUPS (AI-Powered Analysis)")
            print("=" * 140)
            print(f"\nAnalyzed {len(llm_analyzed)} top candidates with GPT-5.2\n")

            for i, r in enumerate(sorted(llm_analyzed, key=lambda x: x.llm_analysis.confidence, reverse=True)[:10], 1):
                llm = r.llm_analysis
                print(f"{i}. {r.ticker} ({r.direction}) - Confidence: {llm.confidence}/10")
                print("   " + "-" * 70)
                print(f"   Entry: ${llm.entry_price:.2f} ({llm.entry_type})")
                print(f"          {llm.entry_reasoning[:80]}...")
                print(f"   Stop:  ${llm.stop_loss:.2f}")
                print(f"          {llm.stop_reasoning[:80]}...")
                print(f"   T1:    ${llm.target_1:.2f} | T2: ${llm.target_2:.2f} | T3: ${llm.target_3:.2f}")
                print(f"          {llm.target_reasoning[:80]}...")
                print(f"\n   Thesis: {llm.thesis}")
                if llm.key_risks:
                    print(f"\n   Risks:")
                    for risk in llm.key_risks[:3]:
                        print(f"   - {risk}")
                print()


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Scan stocks for best deals")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Data directory")
    parser.add_argument("--output", type=Path, default=Path("output/stock_deals.csv"), help="Output CSV path")
    parser.add_argument("--json", type=Path, default=None, help="Also export to JSON")
    parser.add_argument("--top", type=int, default=30, help="Number of top deals to show")
    parser.add_argument("--tweets", type=Path, default=None, help="Path to pre-computed tweets CSV with sentiment_score column")
    parser.add_argument("--lookback-days", type=int, default=30, help="Days of tweets to analyze for sentiment (default: 30)")
    parser.add_argument("--half-life", type=float, default=7.0, help="Days until sentiment weight drops to 50%% (default: 7.0)")
    # LLM Analysis arguments
    parser.add_argument("--llm-analysis", action="store_true", help="Enable LLM analysis for top candidates using GPT-5.2")
    parser.add_argument("--llm-top", type=int, default=20, help="Number of top candidates to analyze with LLM (default: 20)")
    parser.add_argument("--llm-api-key", type=str, default=None, help="NVIDIA API key for LLM analysis (or set NVIDIA_API_KEY env var)")
    args = parser.parse_args()

    # Ensure output directory exists
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Get LLM API key if analysis is enabled
    llm_api_key = None
    if args.llm_analysis:
        llm_api_key = args.llm_api_key or os.environ.get("NVIDIA_API_KEY")
        if not llm_api_key:
            logger.warning("LLM analysis requested but no API key provided. Set --llm-api-key or NVIDIA_API_KEY env var.")

    # Run scanner
    scanner = StockDealScanner(
        args.data_dir,
        tweets_path=args.tweets,
        lookback_days=args.lookback_days,
        half_life=args.half_life,
        llm_api_key=llm_api_key,
        llm_top_n=args.llm_top,
    )
    scanner.scan_all()

    # Export results
    scanner.export_to_csv(args.output)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        scanner.export_to_json(args.json)

    # Print summary
    scanner.print_summary(top_n=args.top)


if __name__ == "__main__":
    main()

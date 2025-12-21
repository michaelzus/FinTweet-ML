#!/usr/bin/env python3
"""Deal Finder - Analyzes a stock ticker and finds optimal entry points.

Usage:
    python scripts/deal_finder.py NVDA
    python scripts/deal_finder.py AAPL --html
    python scripts/deal_finder.py TSLA --html --output report.html
"""

import argparse
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pyarrow.feather as feather


@dataclass
class TechnicalIndicators:
    """Technical analysis indicators for a stock."""

    rsi: float
    bb_upper: float
    bb_middle: float
    bb_lower: float
    bb_position_pct: float
    ma20: Optional[float]
    ma50: Optional[float]
    ma100: Optional[float]
    ma200: Optional[float]
    atr: float
    atr_pct: float


@dataclass
class EntryZone:
    """A recommended entry price zone."""

    zone_low: float
    zone_high: float
    discount_pct: float
    deal_score: int
    rating: str
    reason: str
    stop_loss: float
    target: float
    risk_reward: str


@dataclass
class DealAnalysis:
    """Complete deal analysis for a stock."""

    ticker: str
    current_price: float
    ath: float
    atl: float
    ath_discount_pct: float
    atl_premium_pct: float
    price_percentile: float
    trading_days: int
    technicals: TechnicalIndicators
    deal_score: int
    deal_rating: str
    score_breakdown: list = field(default_factory=list)
    entry_zones: list = field(default_factory=list)
    key_supports: list = field(default_factory=list)


class DealFinder:
    """Finds optimal entry points for a stock."""

    DATA_DIR = Path(__file__).parent.parent / "data"

    def __init__(self, ticker: str):
        """
        Initialize deal finder for a ticker.

        Args:
            ticker: Stock ticker symbol
        """
        self.ticker = ticker.upper()
        self.df_intra: Optional[pd.DataFrame] = None
        self.df_daily: Optional[pd.DataFrame] = None
        self.daily_agg: Optional[pd.DataFrame] = None

    def load_data(self) -> bool:
        """
        Load intraday and daily data for the ticker.

        Returns:
            True if data loaded successfully, False otherwise
        """
        intraday_path = self.DATA_DIR / "intraday" / f"{self.ticker}.feather"
        daily_path = self.DATA_DIR / "daily" / f"{self.ticker}.feather"

        if not intraday_path.exists():
            print(f"❌ No intraday data found for {self.ticker}")
            return False

        self.df_intra = feather.read_feather(intraday_path)

        if daily_path.exists():
            self.df_daily = feather.read_feather(daily_path)

        # Aggregate intraday to daily for technical analysis
        self._aggregate_to_daily()
        return True

    def _aggregate_to_daily(self) -> None:
        """Aggregate intraday data to daily OHLCV."""
        assert self.df_intra is not None
        df = self.df_intra.copy()
        df["day"] = df["date"].dt.date
        self.daily_agg = (
            df.groupby("day").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).reset_index()
        )

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate RSI indicator.

        Args:
            prices: Series of closing prices
            period: RSI period (default 14)

        Returns:
            Series of RSI values
        """
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _calculate_technicals(self) -> TechnicalIndicators:
        """
        Calculate all technical indicators.

        Returns:
            TechnicalIndicators with RSI, BB, MAs, ATR
        """
        assert self.daily_agg is not None
        assert self.df_intra is not None
        df = self.daily_agg.copy()
        current_price = self.df_intra["close"].iloc[-1]

        # Moving Averages
        df["MA20"] = df["close"].rolling(20).mean()
        df["MA50"] = df["close"].rolling(50).mean()
        df["MA100"] = df["close"].rolling(100).mean()
        df["MA200"] = df["close"].rolling(200).mean()

        # RSI
        df["RSI"] = self._calculate_rsi(df["close"])

        # Bollinger Bands
        df["BB_middle"] = df["close"].rolling(20).mean()
        bb_std = df["close"].rolling(20).std()
        df["BB_upper"] = df["BB_middle"] + (2 * bb_std)
        df["BB_lower"] = df["BB_middle"] - (2 * bb_std)

        # ATR
        df["TR"] = np.maximum(
            df["high"] - df["low"], np.maximum(abs(df["high"] - df["close"].shift(1)), abs(df["low"] - df["close"].shift(1)))
        )
        df["ATR"] = df["TR"].rolling(14).mean()

        latest = df.iloc[-1]

        # BB position
        bb_range = latest["BB_upper"] - latest["BB_lower"]
        bb_position = ((current_price - latest["BB_lower"]) / bb_range * 100) if bb_range > 0 else 50

        return TechnicalIndicators(
            rsi=latest["RSI"],
            bb_upper=latest["BB_upper"],
            bb_middle=latest["BB_middle"],
            bb_lower=latest["BB_lower"],
            bb_position_pct=bb_position,
            ma20=latest["MA20"] if pd.notna(latest["MA20"]) else None,
            ma50=latest["MA50"] if pd.notna(latest["MA50"]) else None,
            ma100=latest["MA100"] if pd.notna(latest["MA100"]) else None,
            ma200=latest["MA200"] if pd.notna(latest["MA200"]) else None,
            atr=latest["ATR"],
            atr_pct=latest["ATR"] / current_price * 100 if current_price > 0 else 0,
        )

    def _find_support_levels(self) -> list[float]:
        """
        Find key support levels from price history.

        Returns:
            List of support price levels sorted by proximity
        """
        assert self.daily_agg is not None
        assert self.df_intra is not None
        df = self.daily_agg.copy()
        current_price = self.df_intra["close"].iloc[-1]

        # Find swing lows (local minima)
        df["swing_low"] = (
            (df["low"] < df["low"].shift(1))
            & (df["low"] < df["low"].shift(-1))
            & (df["low"] < df["low"].shift(2))
            & (df["low"] < df["low"].shift(-2))
        )

        swing_lows = df[df["swing_low"]]["low"].tolist()

        # Add MA levels as potential support
        ma200 = df["close"].rolling(200).mean().iloc[-1]
        ma100 = df["close"].rolling(100).mean().iloc[-1]
        ma50 = df["close"].rolling(50).mean().iloc[-1]

        for ma in [ma200, ma100, ma50]:
            if pd.notna(ma) and ma < current_price:
                swing_lows.append(ma)

        # Filter to levels below current price and cluster nearby levels
        supports = [s for s in swing_lows if s < current_price * 0.99]
        supports = sorted(set(round(s, 2) for s in supports), reverse=True)

        # Cluster nearby supports (within 2%)
        clustered: list[float] = []
        for s in supports:
            if not clustered or abs(s - clustered[-1]) / clustered[-1] > 0.02:
                clustered.append(s)

        return clustered[:5]  # Top 5 support levels

    def _calculate_deal_score(self, technicals: TechnicalIndicators, ath_discount: float) -> tuple[int, list[str]]:
        """
        Calculate deal score based on multiple factors.

        Args:
            technicals: Technical indicators for the stock
            ath_discount: Percentage discount from all-time high

        Returns:
            Tuple of (score, list of scoring reasons)
        """
        assert self.df_intra is not None
        score = 0
        reasons: list[str] = []
        current_price = self.df_intra["close"].iloc[-1]

        # RSI score (max 25 points)
        if technicals.rsi < 30:
            score += 25
            reasons.append(f"RSI oversold ({technicals.rsi:.0f}): +25")
        elif technicals.rsi < 40:
            score += 15
            reasons.append(f"RSI low ({technicals.rsi:.0f}): +15")
        elif technicals.rsi < 50:
            score += 5
            reasons.append(f"RSI below 50 ({technicals.rsi:.0f}): +5")
        else:
            reasons.append(f"RSI neutral/high ({technicals.rsi:.0f}): +0")

        # Distance from ATH (max 20 points)
        if ath_discount > 30:
            score += 20
            reasons.append(f"ATH discount >{30}%: +20")
        elif ath_discount > 20:
            score += 15
            reasons.append(f"ATH discount {ath_discount:.0f}%: +15")
        elif ath_discount > 10:
            score += 10
            reasons.append(f"ATH discount {ath_discount:.0f}%: +10")
        elif ath_discount > 5:
            score += 5
            reasons.append(f"ATH discount {ath_discount:.0f}%: +5")
        else:
            reasons.append(f"Near ATH ({ath_discount:.0f}% off): +0")

        # MA position (max 20 points)
        ma_score = 0
        if technicals.ma200 and current_price > technicals.ma200:
            ma_score += 10
            reasons.append("Above 200MA (uptrend): +10")
        if technicals.ma50 and current_price < technicals.ma50:
            ma_score += 5
            reasons.append("Below 50MA (pullback): +5")
        if technicals.ma20 and current_price < technicals.ma20:
            ma_score += 5
            reasons.append("Below 20MA (short-term dip): +5")
        score += ma_score

        # Bollinger position (max 15 points)
        bb_pos = technicals.bb_position_pct
        if bb_pos < 20:
            score += 15
            reasons.append(f"Near lower BB ({bb_pos:.0f}%): +15")
        elif bb_pos < 40:
            score += 10
            reasons.append(f"Lower half of BB ({bb_pos:.0f}%): +10")
        elif bb_pos < 50:
            score += 5
            reasons.append(f"Below BB middle ({bb_pos:.0f}%): +5")
        else:
            reasons.append(f"Upper BB zone ({bb_pos:.0f}%): +0")

        # Support proximity (max 20 points)
        supports = self._find_support_levels()
        if supports:
            nearest = supports[0]
            support_distance = (current_price - nearest) / current_price * 100
            if support_distance < 3:
                score += 20
                reasons.append(f"Very close to support ${nearest:.2f} ({support_distance:.1f}%): +20")
            elif support_distance < 5:
                score += 15
                reasons.append(f"Near support ${nearest:.2f} ({support_distance:.1f}%): +15")
            elif support_distance < 10:
                score += 10
                reasons.append(f"Support nearby ${nearest:.2f} ({support_distance:.1f}%): +10")
            else:
                reasons.append(f"Support distant ${nearest:.2f} ({support_distance:.1f}%): +0")
        else:
            reasons.append("No clear supports identified: +0")

        return score, reasons

    def _generate_entry_zones(self, technicals: TechnicalIndicators, ath: float) -> list[EntryZone]:
        """
        Generate recommended entry zones based on support levels.

        Args:
            technicals: Technical indicators for the stock
            ath: All-time high price

        Returns:
            List of EntryZone objects with entry recommendations
        """
        assert self.df_intra is not None
        current_price = self.df_intra["close"].iloc[-1]
        supports = self._find_support_levels()
        zones = []

        # Add MA-based zones
        ma_levels = []
        if technicals.ma50 and technicals.ma50 < current_price * 0.98:
            ma_levels.append(("50MA", technicals.ma50))
        if technicals.ma100 and technicals.ma100 < current_price * 0.98:
            ma_levels.append(("100MA", technicals.ma100))
        if technicals.ma200 and technicals.ma200 < current_price * 0.98:
            ma_levels.append(("200MA", technicals.ma200))

        # Combine supports with MAs for zone generation
        all_levels: list[tuple[float, Optional[str]]] = [(s, None) for s in supports[:4]]
        for name, level in ma_levels:
            # Check if MA is not too close to existing support
            if not any(abs(level - s[0]) / s[0] < 0.03 for s in all_levels):
                all_levels.append((level, name))

        all_levels = sorted(all_levels, key=lambda x: x[0], reverse=True)

        for i, (level, ma_name) in enumerate(all_levels[:4]):
            zone_high = level * 1.02
            zone_low = level * 0.98
            discount = (ath - level) / ath * 100

            # Calculate stop and target
            stop = level * 0.94  # 6% below zone
            target = current_price * 1.10  # 10% above current

            # Risk/Reward
            risk = level - stop
            reward = target - level
            rr = reward / risk if risk > 0 else 0

            # Score and rating based on discount
            if discount > 30:
                zone_score = 90
                rating = "💎 GIFT"
                reason = "Extreme discount + major support"
            elif discount > 25:
                zone_score = 80
                rating = "🟢 STRONG BUY"
                reason = f"{'200MA test + ' if ma_name == '200MA' else ''}Institutional level"
            elif discount > 15:
                zone_score = 70
                rating = "🟢 EXCELLENT"
                reason = f"Major support confluence{' + ' + ma_name if ma_name else ''}"
            elif discount > 8:
                zone_score = 55
                rating = "🟡 GOOD"
                reason = f"Support retest{' + ' + ma_name if ma_name else ''}"
            else:
                zone_score = 40
                rating = "⚪ FAIR"
                reason = "Minor pullback zone"

            zones.append(
                EntryZone(
                    zone_low=zone_low,
                    zone_high=zone_high,
                    discount_pct=discount,
                    deal_score=zone_score,
                    rating=rating,
                    reason=reason,
                    stop_loss=stop,
                    target=target,
                    risk_reward=f"1:{rr:.1f}" if rr > 0 else "N/A",
                )
            )

        return zones

    def analyze(self) -> Optional[DealAnalysis]:
        """
        Run complete deal analysis.

        Returns:
            DealAnalysis object or None if data not available
        """
        if not self.load_data():
            return None

        assert self.df_intra is not None
        assert self.daily_agg is not None

        current_price = self.df_intra["close"].iloc[-1]
        ath = self.df_intra["high"].max()
        atl = self.df_intra["low"].min()
        ath_discount = (ath - current_price) / ath * 100
        atl_premium = (current_price - atl) / atl * 100

        # Price percentile
        percentile = (self.daily_agg["close"] < current_price).sum() / len(self.daily_agg) * 100

        # Technical indicators
        technicals = self._calculate_technicals()

        # Deal score
        score, breakdown = self._calculate_deal_score(technicals, ath_discount)

        # Rating
        if score >= 70:
            rating = "🟢 EXCELLENT DEAL - Strong buy opportunity"
        elif score >= 55:
            rating = "🟡 GOOD DEAL - Favorable entry"
        elif score >= 40:
            rating = "⚪ FAIR - Wait for better entry or scale in"
        else:
            rating = "🔴 NOT A DEAL - Too expensive, wait for pullback"

        # Entry zones
        entry_zones = self._generate_entry_zones(technicals, ath)

        # Key supports
        supports = self._find_support_levels()

        return DealAnalysis(
            ticker=self.ticker,
            current_price=current_price,
            ath=ath,
            atl=atl,
            ath_discount_pct=ath_discount,
            atl_premium_pct=atl_premium,
            price_percentile=percentile,
            trading_days=len(self.daily_agg),
            technicals=technicals,
            deal_score=score,
            deal_rating=rating,
            score_breakdown=breakdown,
            entry_zones=entry_zones,
            key_supports=supports,
        )


class DealReporter:
    """Formats and prints deal analysis reports."""

    def __init__(self, analysis: DealAnalysis):
        """
        Initialize reporter with analysis data.

        Args:
            analysis: DealAnalysis object to report on
        """
        self.a = analysis

    def print_full_report(self) -> None:
        """Print the complete deal analysis report."""
        self._print_header()
        self._print_current_state()
        self._print_technicals()
        self._print_percentile()
        self._print_deal_score()
        self._print_entry_zones()
        self._print_strategies()
        self._print_probability()
        self._print_bottom_line()

    def _print_header(self) -> None:
        """Print report header."""
        print("=" * 70)
        print(f"🔍 {self.a.ticker} DEAL FINDER - Entry Point Analysis")
        print("=" * 70)

    def _print_current_state(self) -> None:
        """Print current price state."""
        print("\n📍 CURRENT STATE")
        print(f"   Price: ${self.a.current_price:.2f}")
        print(f"   ATH: ${self.a.ath:.2f} ({-self.a.ath_discount_pct:+.1f}% from high)")
        print(f"   ATL: ${self.a.atl:.2f} ({self.a.atl_premium_pct:+.1f}% from low)")

    def _print_technicals(self) -> None:
        """Print technical indicators."""
        t = self.a.technicals

        print("\n" + "=" * 70)
        print("📊 TECHNICAL INDICATORS")
        print("=" * 70)

        # RSI
        print(f"\n   RSI (14): {t.rsi:.1f}", end="")
        if t.rsi < 30:
            print(" 🟢 OVERSOLD")
        elif t.rsi > 70:
            print(" 🔴 OVERBOUGHT")
        elif t.rsi < 40:
            print(" 🟡 Approaching oversold")
        else:
            print(" ⚪ Neutral")

        # Bollinger Bands
        print("\n   Bollinger Bands:")
        print(f"      Upper: ${t.bb_upper:.2f}")
        print(f"      Middle: ${t.bb_middle:.2f}")
        print(f"      Lower: ${t.bb_lower:.2f}")
        print(f"      Current: ${self.a.current_price:.2f} ({t.bb_position_pct:.0f}% of band)")

        # Moving Averages
        print("\n   Moving Averages:")
        for ma_val, name in [(t.ma20, "20-day"), (t.ma50, "50-day"), (t.ma100, "100-day"), (t.ma200, "200-day")]:
            if ma_val:
                pct = (self.a.current_price - ma_val) / ma_val * 100
                status = "🟢 Above" if pct > 0 else "🔴 Below"
                print(f"      {name}: ${ma_val:.2f} ({pct:+.1f}%) {status}")

        # Volatility
        print(f"\n   Volatility (ATR-14): ${t.atr:.2f} ({t.atr_pct:.1f}% of price)")

    def _print_percentile(self) -> None:
        """Print historical percentile analysis."""
        print("\n" + "=" * 70)
        print("📉 HISTORICAL PRICE PERCENTILE")
        print("=" * 70)
        print(f"\n   Current price ${self.a.current_price:.2f} is higher than {self.a.price_percentile:.0f}% of historical closes")
        print(f"   Data: {self.a.trading_days} trading days")

    def _print_deal_score(self) -> None:
        """Print deal score breakdown."""
        print("\n" + "=" * 70)
        print("🎯 DEAL SCORE ANALYSIS")
        print("=" * 70)

        print("\n   Scoring Breakdown:")
        for reason in self.a.score_breakdown:
            print(f"      • {reason}")

        print("\n   ═══════════════════════════════════")
        print(f"   TOTAL DEAL SCORE: {self.a.deal_score}/100")
        print("   ═══════════════════════════════════")
        print(f"   {self.a.deal_rating}")

    def _print_entry_zones(self) -> None:
        """Print recommended entry zones."""
        if not self.a.entry_zones:
            return

        print("\n" + "=" * 70)
        print("🎯 RECOMMENDED ENTRY ZONES (Wait for these levels)")
        print("=" * 70)

        for zone in self.a.entry_zones:
            print("\n   ┌─────────────────────────────────────────────────────────")
            zone_str = f"${zone.zone_low:.2f} - ${zone.zone_high:.2f}"
            print(f"   │ Entry Zone: {zone_str}  ({zone.discount_pct:.0f}% off ATH)")
            print(f"   │ Deal Score: {zone.deal_score}/100  {zone.rating}")
            print(f"   │ Why: {zone.reason}")
            stop_target = f"${zone.stop_loss:.2f} | Target: ${zone.target:.2f}"
            print(f"   │ Stop Loss: {stop_target} | R:R {zone.risk_reward}")
            print("   └─────────────────────────────────────────────────────────")

    def _print_strategies(self) -> None:
        """Print actionable strategies."""
        if self.a.deal_score < 50:
            print("\n" + "=" * 70)
            print("📋 ACTIONABLE STRATEGIES")
            print("=" * 70)

            # Build alert levels from entry zones
            alert_levels = [f"${z.zone_high:.0f}" for z in self.a.entry_zones[:3]]

            print(
                f"""
   STRATEGY 1: WAIT FOR PULLBACK
   ──────────────────────────────
   • Set price alerts at {', '.join(alert_levels)}
   • Scale in: 25% at first zone, add on further weakness
   • Stop: 6% below lowest entry zone

   STRATEGY 2: SCALE IN NOW (Higher Risk)
   ──────────────────────────────────────
   • If you MUST buy now, use small position (10-20%)
   • Add only on pullbacks to support
   • Current R:R may be poor

   STRATEGY 3: OPTIONS PLAY
   ──────────────────────────
   • Sell cash-secured puts at support strikes
   • Collect premium while waiting for pullback
   • If assigned, you get shares at discount + premium
"""
            )

    def _print_probability(self) -> None:
        """Print price probability analysis."""
        print("\n" + "=" * 70)
        print("📊 PRICE PROBABILITY ANALYSIS")
        print("=" * 70)

        t = self.a.technicals
        print(f"\n   Based on ATR of ${t.atr:.2f}/day:")

        for zone in self.a.entry_zones[:3]:
            pts_away = abs(self.a.current_price - zone.zone_high)
            days = pts_away / t.atr if t.atr > 0 else 0
            print(f"      • ${zone.zone_high:.0f} (~{pts_away:.0f} pts away): ~{days:.0f}-{days+1:.0f} trading days")

        print("      (assumes directional move)")

    def _print_bottom_line(self) -> None:
        """Print summary recommendation."""
        print("\n" + "=" * 70)
        print("🎲 BOTTOM LINE")
        print("=" * 70)

        if self.a.deal_score >= 55:
            recommendation = "CONSIDER BUYING"
        else:
            recommendation = "WAIT"

        zones_str = ""
        for zone in self.a.entry_zones[:3]:
            rating_word = zone.rating.split()[1]
            zones_str += f"\n   • ${zone.zone_low:.0f}-${zone.zone_high:.0f} "
            zones_str += f"for {rating_word} entry ({zone.discount_pct:.0f}% pullback)"

        print(
            f"""
   Current Price: ${self.a.current_price:.2f}
   Deal Score: {self.a.deal_score}/100 {'🟢' if self.a.deal_score >= 55 else '🟡' if self.a.deal_score >= 40 else '🔴'}

   RECOMMENDATION: {recommendation}

   Best Entry Zones:{zones_str}
"""
        )


class HtmlReporter:
    """Generates beautiful HTML reports for deal analysis."""

    def __init__(self, analysis: DealAnalysis):
        """
        Initialize HTML reporter.

        Args:
            analysis: DealAnalysis object to report on
        """
        self.a = analysis

    def _get_score_color(self, score: int) -> str:
        """
        Get color based on score.

        Args:
            score: Deal score (0-100)

        Returns:
            Hex color string
        """
        if score >= 70:
            return "#10b981"  # Green
        elif score >= 55:
            return "#f59e0b"  # Amber
        elif score >= 40:
            return "#6b7280"  # Gray
        return "#ef4444"  # Red

    def _get_rsi_status(self) -> tuple[str, str]:
        """
        Get RSI status text and color.

        Returns:
            Tuple of (status text, hex color)
        """
        rsi = self.a.technicals.rsi
        if rsi < 30:
            return "OVERSOLD", "#10b981"
        elif rsi > 70:
            return "OVERBOUGHT", "#ef4444"
        elif rsi < 40:
            return "Approaching oversold", "#f59e0b"
        return "Neutral", "#6b7280"

    def generate(self) -> str:
        """
        Generate complete HTML report.

        Returns:
            HTML string
        """
        t = self.a.technicals
        score_color = self._get_score_color(self.a.deal_score)
        rsi_status, rsi_color = self._get_rsi_status()

        # Build MA rows
        ma_rows = ""
        for ma_val, name in [(t.ma20, "20-day"), (t.ma50, "50-day"), (t.ma100, "100-day"), (t.ma200, "200-day")]:
            if ma_val:
                pct = (self.a.current_price - ma_val) / ma_val * 100
                status_color = "#10b981" if pct > 0 else "#ef4444"
                status = "Above" if pct > 0 else "Below"
                ma_rows += f"""
                <tr>
                    <td>{name}</td>
                    <td>${ma_val:.2f}</td>
                    <td style="color: {status_color}">{pct:+.1f}% ({status})</td>
                </tr>"""

        # Build entry zones
        entry_zones_html = ""
        for zone in self.a.entry_zones:
            zone_color = self._get_score_color(zone.deal_score)
            entry_zones_html += f"""
            <div class="entry-zone">
                <div class="zone-header">
                    <span class="zone-price">${zone.zone_low:.2f} - ${zone.zone_high:.2f}</span>
                    <span class="zone-discount">{zone.discount_pct:.0f}% off ATH</span>
                </div>
                <div class="zone-score" style="background: {zone_color}20; border-left: 4px solid {zone_color}">
                    <span class="score-value">{zone.deal_score}/100</span>
                    <span class="score-rating">{zone.rating}</span>
                </div>
                <div class="zone-details">
                    <p><strong>Why:</strong> {zone.reason}</p>
                    <p><strong>Stop:</strong> ${zone.stop_loss:.2f} | <strong>Target:</strong> ${zone.target:.2f} |
                       <strong>R:R:</strong> {zone.risk_reward}</p>
                </div>
            </div>"""

        # Build score breakdown
        score_breakdown_html = ""
        for reason in self.a.score_breakdown:
            points = reason.split(": ")[-1]
            is_positive = "+0" not in points
            color = "#10b981" if is_positive else "#6b7280"
            score_breakdown_html += f'<li style="color: {color}">{reason}</li>'

        # Probability rows
        prob_rows = ""
        for zone in self.a.entry_zones[:3]:
            pts_away = abs(self.a.current_price - zone.zone_high)
            days = pts_away / t.atr if t.atr > 0 else 0
            prob_rows += f"""
            <tr>
                <td>${zone.zone_high:.0f}</td>
                <td>{pts_away:.0f} pts</td>
                <td>~{days:.0f}-{days+1:.0f} days</td>
            </tr>"""

        recommendation = "CONSIDER BUYING" if self.a.deal_score >= 55 else "WAIT"
        rec_color = "#10b981" if self.a.deal_score >= 55 else "#f59e0b"

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.a.ticker} Deal Analysis</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Outfit:wght@400;500;600;700&display=swap"
          rel="stylesheet">
    <style>
        :root {{
            --bg-primary: #0f0f0f;
            --bg-secondary: #1a1a1a;
            --bg-card: #242424;
            --text-primary: #ffffff;
            --text-secondary: #a1a1aa;
            --border-color: #333;
            --accent-green: #10b981;
            --accent-red: #ef4444;
            --accent-amber: #f59e0b;
            --accent-blue: #3b82f6;
        }}

        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: 'Outfit', sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
            min-height: 100vh;
        }}

        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem;
        }}

        header {{
            text-align: center;
            padding: 3rem 0;
            background: linear-gradient(135deg, var(--bg-secondary) 0%, var(--bg-primary) 100%);
            border-bottom: 1px solid var(--border-color);
            margin-bottom: 2rem;
        }}

        .ticker-badge {{
            display: inline-block;
            background: linear-gradient(135deg, {score_color}40, {score_color}20);
            border: 2px solid {score_color};
            padding: 0.5rem 2rem;
            border-radius: 50px;
            font-size: 2.5rem;
            font-weight: 700;
            letter-spacing: 0.1em;
            margin-bottom: 1rem;
        }}

        header h2 {{
            color: var(--text-secondary);
            font-weight: 400;
            font-size: 1.1rem;
        }}

        .timestamp {{
            color: var(--text-secondary);
            font-size: 0.875rem;
            margin-top: 0.5rem;
            font-family: 'JetBrains Mono', monospace;
        }}

        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2rem;
        }}

        .card {{
            background: var(--bg-card);
            border-radius: 16px;
            padding: 1.5rem;
            border: 1px solid var(--border-color);
        }}

        .card-header {{
            display: flex;
            align-items: center;
            gap: 0.75rem;
            margin-bottom: 1.25rem;
            padding-bottom: 1rem;
            border-bottom: 1px solid var(--border-color);
        }}

        .card-icon {{
            font-size: 1.5rem;
        }}

        .card-title {{
            font-size: 1.1rem;
            font-weight: 600;
        }}

        .hero-card {{
            grid-column: 1 / -1;
            background: linear-gradient(135deg, var(--bg-card) 0%, var(--bg-secondary) 100%);
        }}

        .score-display {{
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 3rem;
            flex-wrap: wrap;
        }}

        .score-circle {{
            width: 180px;
            height: 180px;
            border-radius: 50%;
            background: conic-gradient({score_color} {self.a.deal_score * 3.6}deg, var(--bg-secondary) 0deg);
            display: flex;
            align-items: center;
            justify-content: center;
            position: relative;
        }}

        .score-circle::before {{
            content: '';
            width: 150px;
            height: 150px;
            background: var(--bg-card);
            border-radius: 50%;
            position: absolute;
        }}

        .score-value {{
            position: relative;
            z-index: 1;
            font-size: 3rem;
            font-weight: 700;
            font-family: 'JetBrains Mono', monospace;
        }}

        .score-label {{
            font-size: 0.875rem;
            color: var(--text-secondary);
        }}

        .score-info {{
            text-align: left;
        }}

        .score-rating {{
            font-size: 1.5rem;
            font-weight: 600;
            margin-bottom: 0.5rem;
            color: {score_color};
        }}

        .recommendation {{
            display: inline-block;
            padding: 0.5rem 1.5rem;
            background: {rec_color}20;
            border: 1px solid {rec_color};
            border-radius: 8px;
            color: {rec_color};
            font-weight: 600;
            font-size: 1.1rem;
        }}

        .price-grid {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 1rem;
        }}

        .price-item {{
            text-align: center;
            padding: 1rem;
            background: var(--bg-secondary);
            border-radius: 12px;
        }}

        .price-item .value {{
            font-size: 1.5rem;
            font-weight: 600;
            font-family: 'JetBrains Mono', monospace;
        }}

        .price-item .label {{
            font-size: 0.75rem;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-top: 0.25rem;
        }}

        .price-item .change {{
            font-size: 0.875rem;
            margin-top: 0.25rem;
        }}

        .positive {{ color: var(--accent-green); }}
        .negative {{ color: var(--accent-red); }}

        .indicator-row {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.75rem 0;
            border-bottom: 1px solid var(--border-color);
        }}

        .indicator-row:last-child {{
            border-bottom: none;
        }}

        .indicator-label {{
            color: var(--text-secondary);
        }}

        .indicator-value {{
            font-family: 'JetBrains Mono', monospace;
            font-weight: 600;
        }}

        .status-badge {{
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 600;
            margin-left: 0.5rem;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
        }}

        th, td {{
            padding: 0.75rem;
            text-align: left;
            border-bottom: 1px solid var(--border-color);
        }}

        th {{
            color: var(--text-secondary);
            font-weight: 500;
            font-size: 0.875rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}

        td {{
            font-family: 'JetBrains Mono', monospace;
        }}

        .entry-zones-container {{
            grid-column: 1 / -1;
        }}

        .entry-zones-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 1rem;
        }}

        .entry-zone {{
            background: var(--bg-secondary);
            border-radius: 12px;
            overflow: hidden;
        }}

        .zone-header {{
            padding: 1rem;
            background: var(--bg-primary);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}

        .zone-price {{
            font-family: 'JetBrains Mono', monospace;
            font-weight: 600;
            font-size: 1.1rem;
        }}

        .zone-discount {{
            color: var(--text-secondary);
            font-size: 0.875rem;
        }}

        .zone-score {{
            padding: 0.75rem 1rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}

        .zone-details {{
            padding: 1rem;
            font-size: 0.875rem;
        }}

        .zone-details p {{
            margin-bottom: 0.5rem;
        }}

        .breakdown-list {{
            list-style: none;
        }}

        .breakdown-list li {{
            padding: 0.5rem 0;
            border-bottom: 1px solid var(--border-color);
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.875rem;
        }}

        .breakdown-list li:last-child {{
            border-bottom: none;
        }}

        .bb-visual {{
            margin: 1rem 0;
            height: 40px;
            background: linear-gradient(90deg, var(--accent-red)20, var(--accent-amber)20, var(--accent-green)20);
            border-radius: 8px;
            position: relative;
        }}

        .bb-marker {{
            position: absolute;
            top: 50%;
            transform: translate(-50%, -50%);
            width: 12px;
            height: 12px;
            background: white;
            border-radius: 50%;
            border: 2px solid var(--bg-primary);
            left: {t.bb_position_pct}%;
        }}

        .percentile-bar {{
            height: 24px;
            background: var(--bg-secondary);
            border-radius: 12px;
            overflow: hidden;
            margin: 1rem 0;
        }}

        .percentile-fill {{
            height: 100%;
            background: linear-gradient(90deg, var(--accent-green), var(--accent-amber), var(--accent-red));
            width: {self.a.price_percentile}%;
            border-radius: 12px;
        }}

        footer {{
            text-align: center;
            padding: 2rem;
            color: var(--text-secondary);
            font-size: 0.875rem;
            border-top: 1px solid var(--border-color);
            margin-top: 2rem;
        }}

        @media (max-width: 768px) {{
            .container {{
                padding: 1rem;
            }}
            .price-grid {{
                grid-template-columns: 1fr;
            }}
            .score-display {{
                flex-direction: column;
                gap: 1.5rem;
            }}
            .score-info {{
                text-align: center;
            }}
        }}
    </style>
</head>
<body>
    <header>
        <div class="ticker-badge">{self.a.ticker}</div>
        <h2>Deal Finder Analysis</h2>
        <p class="timestamp">Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    </header>

    <div class="container">
        <!-- Deal Score Hero -->
        <div class="grid">
            <div class="card hero-card">
                <div class="score-display">
                    <div>
                        <div class="score-circle">
                            <span class="score-value">{self.a.deal_score}</span>
                        </div>
                        <p class="score-label" style="text-align: center; margin-top: 0.5rem;">DEAL SCORE</p>
                    </div>
                    <div class="score-info">
                        <div class="score-rating">{self.a.deal_rating}</div>
                        <div class="recommendation">{recommendation}</div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Price Overview -->
        <div class="grid">
            <div class="card">
                <div class="card-header">
                    <span class="card-icon">📍</span>
                    <span class="card-title">Current State</span>
                </div>
                <div class="price-grid">
                    <div class="price-item">
                        <div class="value">${self.a.current_price:.2f}</div>
                        <div class="label">Current Price</div>
                    </div>
                    <div class="price-item">
                        <div class="value">${self.a.ath:.2f}</div>
                        <div class="label">All-Time High</div>
                        <div class="change negative">{-self.a.ath_discount_pct:.1f}%</div>
                    </div>
                    <div class="price-item">
                        <div class="value">${self.a.atl:.2f}</div>
                        <div class="label">All-Time Low</div>
                        <div class="change positive">+{self.a.atl_premium_pct:.1f}%</div>
                    </div>
                </div>
            </div>

            <div class="card">
                <div class="card-header">
                    <span class="card-icon">📉</span>
                    <span class="card-title">Historical Percentile</span>
                </div>
                <p>Current price is higher than <strong>{self.a.price_percentile:.0f}%</strong> of historical closes</p>
                <div class="percentile-bar">
                    <div class="percentile-fill"></div>
                </div>
                <p style="color: var(--text-secondary); font-size: 0.875rem;">Based on {self.a.trading_days} trading days</p>
            </div>
        </div>

        <!-- Technical Indicators -->
        <div class="grid">
            <div class="card">
                <div class="card-header">
                    <span class="card-icon">📊</span>
                    <span class="card-title">Technical Indicators</span>
                </div>
                <div class="indicator-row">
                    <span class="indicator-label">RSI (14)</span>
                    <span class="indicator-value">
                        {t.rsi:.1f}
                        <span class="status-badge" style="background: {rsi_color}20; color: {rsi_color}">{rsi_status}</span>
                    </span>
                </div>
                <div class="indicator-row">
                    <span class="indicator-label">ATR (14)</span>
                    <span class="indicator-value">${t.atr:.2f} ({t.atr_pct:.1f}%)</span>
                </div>
            </div>

            <div class="card">
                <div class="card-header">
                    <span class="card-icon">📈</span>
                    <span class="card-title">Bollinger Bands</span>
                </div>
                <div class="indicator-row">
                    <span class="indicator-label">Upper</span>
                    <span class="indicator-value">${t.bb_upper:.2f}</span>
                </div>
                <div class="indicator-row">
                    <span class="indicator-label">Middle</span>
                    <span class="indicator-value">${t.bb_middle:.2f}</span>
                </div>
                <div class="indicator-row">
                    <span class="indicator-label">Lower</span>
                    <span class="indicator-value">${t.bb_lower:.2f}</span>
                </div>
                <div class="bb-visual">
                    <div class="bb-marker"></div>
                </div>
                <p style="text-align: center; color: var(--text-secondary); font-size: 0.875rem;">
                    Position: {t.bb_position_pct:.0f}% of band
                </p>
            </div>
        </div>

        <!-- Moving Averages -->
        <div class="grid">
            <div class="card">
                <div class="card-header">
                    <span class="card-icon">📏</span>
                    <span class="card-title">Moving Averages</span>
                </div>
                <table>
                    <thead>
                        <tr>
                            <th>Period</th>
                            <th>Value</th>
                            <th>Status</th>
                        </tr>
                    </thead>
                    <tbody>
                        {ma_rows}
                    </tbody>
                </table>
            </div>

            <div class="card">
                <div class="card-header">
                    <span class="card-icon">🎯</span>
                    <span class="card-title">Score Breakdown</span>
                </div>
                <ul class="breakdown-list">
                    {score_breakdown_html}
                </ul>
            </div>
        </div>

        <!-- Entry Zones -->
        <div class="grid">
            <div class="card entry-zones-container">
                <div class="card-header">
                    <span class="card-icon">🎯</span>
                    <span class="card-title">Recommended Entry Zones</span>
                </div>
                <div class="entry-zones-grid">
                    {entry_zones_html}
                </div>
            </div>
        </div>

        <!-- Probability -->
        <div class="grid">
            <div class="card">
                <div class="card-header">
                    <span class="card-icon">📊</span>
                    <span class="card-title">Price Probability</span>
                </div>
                <p style="margin-bottom: 1rem; color: var(--text-secondary);">
                    Based on ATR of ${t.atr:.2f}/day (assumes directional move)
                </p>
                <table>
                    <thead>
                        <tr>
                            <th>Target</th>
                            <th>Distance</th>
                            <th>Est. Time</th>
                        </tr>
                    </thead>
                    <tbody>
                        {prob_rows}
                    </tbody>
                </table>
            </div>
        </div>
    </div>

    <footer>
        <p>Deal Finder Analysis • {self.a.ticker} • {datetime.now().strftime("%Y-%m-%d")}</p>
    </footer>
</body>
</html>"""
        return html

    def save(self, output_path: Path) -> None:
        """
        Save HTML report to file.

        Args:
            output_path: Path to save HTML file
        """
        html = self.generate()
        output_path.write_text(html, encoding="utf-8")
        print(f"✅ HTML report saved to: {output_path}")


def main() -> int:
    """
    Main entry point.

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    parser = argparse.ArgumentParser(
        description="Analyze a stock for optimal entry points",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/deal_finder.py NVDA
    python scripts/deal_finder.py AAPL --html
    python scripts/deal_finder.py TSLA --html --output report.html
        """,
    )
    parser.add_argument("ticker", type=str, help="Stock ticker symbol (e.g., NVDA, AAPL)")
    parser.add_argument("--html", action="store_true", help="Export to HTML file")
    parser.add_argument("--output", "-o", type=str, help="Output file path (default: {ticker}_deal.html)")

    args = parser.parse_args()

    finder = DealFinder(args.ticker)
    analysis = finder.analyze()

    if analysis is None:
        return 1

    if args.html:
        output_path = Path(args.output) if args.output else Path(f"{args.ticker.upper()}_deal.html")
        html_reporter = HtmlReporter(analysis)
        html_reporter.save(output_path)
    else:
        reporter = DealReporter(analysis)
        reporter.print_full_report()

    return 0


if __name__ == "__main__":
    sys.exit(main())

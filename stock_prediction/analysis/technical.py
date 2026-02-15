"""Technical analysis indicators."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class TechnicalSummary:
    """Aggregated technical analysis results."""

    sma_20: float
    sma_50: float
    ema_12: float
    ema_26: float
    rsi_14: float
    macd: float
    macd_signal: float
    macd_histogram: float
    bb_upper: float
    bb_middle: float
    bb_lower: float
    atr_14: float
    volume_avg_20: float  # 20-day average volume
    volume_ratio: float  # latest volume / 20-day average
    price_vs_sma20: str  # "above" / "below"
    price_vs_sma50: str
    rsi_signal: str  # "overbought" / "oversold" / "neutral"
    macd_signal_str: str  # "bullish" / "bearish"
    bb_position: str  # "above_upper" / "below_lower" / "within"
    trend: str  # "uptrend" / "downtrend" / "sideways"
    volatility: str  # "high" / "medium" / "low"
    volume_signal: str  # "high_volume" / "low_volume" / "normal"


class TechnicalAnalyzer:
    """Compute technical indicators on OHLCV data."""

    def __init__(self, df: pd.DataFrame) -> None:
        if df.empty:
            raise ValueError("DataFrame is empty")
        self.df = df.copy()
        self._compute_all()

    # ------------------------------------------------------------------
    # Moving Averages
    # ------------------------------------------------------------------

    @staticmethod
    def sma(series: pd.Series, window: int) -> pd.Series:
        return series.rolling(window=window, min_periods=1).mean()

    @staticmethod
    def ema(series: pd.Series, span: int) -> pd.Series:
        return series.ewm(span=span, adjust=False).mean()

    # ------------------------------------------------------------------
    # RSI
    # ------------------------------------------------------------------

    @staticmethod
    def rsi(series: pd.Series, window: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1 / window, min_periods=window).mean()
        avg_loss = loss.ewm(alpha=1 / window, min_periods=window).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    # ------------------------------------------------------------------
    # MACD
    # ------------------------------------------------------------------

    @staticmethod
    def macd(
        series: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    # ------------------------------------------------------------------
    # Bollinger Bands
    # ------------------------------------------------------------------

    @staticmethod
    def bollinger_bands(
        series: pd.Series, window: int = 20, num_std: float = 2.0
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        middle = series.rolling(window=window, min_periods=1).mean()
        std = series.rolling(window=window, min_periods=1).std()
        upper = middle + num_std * std
        lower = middle - num_std * std
        return upper, middle, lower

    # ------------------------------------------------------------------
    # ATR (Average True Range)
    # ------------------------------------------------------------------

    @staticmethod
    def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
        high = df["High"]
        low = df["Low"]
        close = df["Close"].shift(1)
        tr = pd.concat(
            [high - low, (high - close).abs(), (low - close).abs()], axis=1
        ).max(axis=1)
        return tr.rolling(window=window, min_periods=1).mean()

    # ------------------------------------------------------------------
    # Internal: compute everything
    # ------------------------------------------------------------------

    def _compute_all(self) -> None:
        close = self.df["Close"]
        self.df["SMA_20"] = self.sma(close, 20)
        self.df["SMA_50"] = self.sma(close, 50)
        self.df["EMA_12"] = self.ema(close, 12)
        self.df["EMA_26"] = self.ema(close, 26)
        self.df["RSI_14"] = self.rsi(close, 14)
        macd_line, signal_line, histogram = self.macd(close)
        self.df["MACD"] = macd_line
        self.df["MACD_Signal"] = signal_line
        self.df["MACD_Hist"] = histogram
        bb_upper, bb_middle, bb_lower = self.bollinger_bands(close)
        self.df["BB_Upper"] = bb_upper
        self.df["BB_Middle"] = bb_middle
        self.df["BB_Lower"] = bb_lower
        self.df["ATR_14"] = self.atr(self.df)
        if "Volume" in self.df.columns:
            self.df["Vol_SMA_20"] = self.sma(self.df["Volume"].astype(float), 20)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summarize(self) -> TechnicalSummary:
        """Return a snapshot of the latest indicator values with signals."""
        last = self.df.iloc[-1]
        price = last["Close"]

        # Price vs moving averages
        price_vs_sma20 = "above" if price > last["SMA_20"] else "below"
        price_vs_sma50 = "above" if price > last["SMA_50"] else "below"

        # RSI signal
        rsi_val = last["RSI_14"]
        if rsi_val >= 70:
            rsi_signal = "overbought"
        elif rsi_val <= 30:
            rsi_signal = "oversold"
        else:
            rsi_signal = "neutral"

        # MACD signal
        macd_signal_str = "bullish" if last["MACD"] > last["MACD_Signal"] else "bearish"

        # Bollinger Bands position
        if price > last["BB_Upper"]:
            bb_position = "above_upper"
        elif price < last["BB_Lower"]:
            bb_position = "below_lower"
        else:
            bb_position = "within"

        # Trend detection (SMA20 vs SMA50 + price position)
        if last["SMA_20"] > last["SMA_50"] and price_vs_sma20 == "above":
            trend = "uptrend"
        elif last["SMA_20"] < last["SMA_50"] and price_vs_sma20 == "below":
            trend = "downtrend"
        else:
            trend = "sideways"

        # Volatility classification based on ATR / price ratio
        atr_ratio = last["ATR_14"] / price if price > 0 else 0
        if atr_ratio > 0.03:
            volatility = "high"
        elif atr_ratio > 0.015:
            volatility = "medium"
        else:
            volatility = "low"

        # Volume analysis
        if "Vol_SMA_20" in self.df.columns and last.get("Vol_SMA_20", 0) > 0:
            vol_avg_20 = last["Vol_SMA_20"]
            vol_ratio = last["Volume"] / vol_avg_20
            if vol_ratio > 1.5:
                volume_signal = "high_volume"
            elif vol_ratio < 0.5:
                volume_signal = "low_volume"
            else:
                volume_signal = "normal"
        else:
            vol_avg_20 = 0.0
            vol_ratio = 0.0
            volume_signal = "N/A"

        return TechnicalSummary(
            sma_20=round(last["SMA_20"], 2),
            sma_50=round(last["SMA_50"], 2),
            ema_12=round(last["EMA_12"], 2),
            ema_26=round(last["EMA_26"], 2),
            rsi_14=round(rsi_val, 2),
            macd=round(last["MACD"], 4),
            macd_signal=round(last["MACD_Signal"], 4),
            macd_histogram=round(last["MACD_Hist"], 4),
            bb_upper=round(last["BB_Upper"], 2),
            bb_middle=round(last["BB_Middle"], 2),
            bb_lower=round(last["BB_Lower"], 2),
            atr_14=round(last["ATR_14"], 2),
            volume_avg_20=round(vol_avg_20, 0),
            volume_ratio=round(vol_ratio, 2),
            price_vs_sma20=price_vs_sma20,
            price_vs_sma50=price_vs_sma50,
            rsi_signal=rsi_signal,
            macd_signal_str=macd_signal_str,
            bb_position=bb_position,
            trend=trend,
            volatility=volatility,
            volume_signal=volume_signal,
        )

    def get_dataframe(self) -> pd.DataFrame:
        """Return the enriched DataFrame with all computed indicators."""
        return self.df

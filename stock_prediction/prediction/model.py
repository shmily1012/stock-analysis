"""Stock price prediction using multiple approaches."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler


@dataclass
class PredictionResult:
    """Holds the output of the prediction engine."""

    current_price: float
    predicted_price_5d: float
    predicted_price_20d: float
    predicted_change_5d_pct: float
    predicted_change_20d_pct: float
    confidence: str  # "high" / "medium" / "low"
    direction: str  # "bullish" / "bearish" / "neutral"
    support_level: float
    resistance_level: float
    method: str
    r2_score_5d: float | None = None
    r2_score_20d: float | None = None


class StockPredictor:
    """Predict future stock prices using technical features + linear regression.

    This is a lightweight model meant for educational / indicative use.
    It combines:
      - Linear regression on engineered features (trend extrapolation)
      - Support / resistance estimation from recent price action
    """

    # Maximum allowed predicted return to prevent wild extrapolation
    MAX_RETURN_5D = 0.15   # ±15%
    MAX_RETURN_20D = 0.30  # ±30%

    FEATURE_COLS = [
        "SMA_20",
        "SMA_50",
        "EMA_12",
        "EMA_26",
        "RSI_14",
        "MACD",
        "MACD_Signal",
        "MACD_Hist",
        "BB_Upper",
        "BB_Middle",
        "BB_Lower",
        "ATR_14",
    ]

    def __init__(self, df: pd.DataFrame) -> None:
        """Accepts a DataFrame already enriched with technical indicators."""
        self.df = df.dropna().copy()
        if len(self.df) < 10:
            raise ValueError(
                "Not enough data points for prediction "
                f"(got {len(self.df)}, need >= 10)"
            )
        self._scaler = StandardScaler()
        self._model = LinearRegression()

    def predict(self) -> PredictionResult:
        """Run prediction and return results."""
        df = self.df.copy()
        current_price = df["Close"].iloc[-1]

        # --- Feature engineering ---------------------------------------------------
        df["Return_1d"] = df["Close"].pct_change()
        df["Return_5d"] = df["Close"].pct_change(5)
        df["Momentum_10"] = df["Close"] - df["Close"].shift(10)
        df["Volatility_10"] = df["Return_1d"].rolling(10).std()

        feature_cols = self.FEATURE_COLS + [
            "Return_1d",
            "Return_5d",
            "Momentum_10",
            "Volatility_10",
        ]
        df = df.dropna()

        if len(df) < 10:
            raise ValueError("Not enough data after feature engineering")

        # --- Targets: future returns -----------------------------------------------
        df["Target_5d"] = df["Close"].shift(-5) / df["Close"] - 1
        df["Target_20d"] = df["Close"].shift(-20) / df["Close"] - 1

        # Training set: rows where target is known
        train_5d = df.dropna(subset=["Target_5d"])
        train_20d = df.dropna(subset=["Target_20d"])

        X_latest = df[feature_cols].iloc[[-1]]

        # --- 5-day prediction ------------------------------------------------------
        pred_5d_ret, r2_5d = self._fit_predict(
            train_5d, feature_cols, "Target_5d", X_latest
        )
        pred_5d_ret = np.clip(pred_5d_ret, -self.MAX_RETURN_5D, self.MAX_RETURN_5D)
        pred_5d_price = current_price * (1 + pred_5d_ret)

        # --- 20-day prediction -----------------------------------------------------
        pred_20d_ret, r2_20d = self._fit_predict(
            train_20d, feature_cols, "Target_20d", X_latest
        )
        pred_20d_ret = np.clip(pred_20d_ret, -self.MAX_RETURN_20D, self.MAX_RETURN_20D)
        pred_20d_price = current_price * (1 + pred_20d_ret)

        # --- Support / Resistance --------------------------------------------------
        recent = self.df.tail(60)
        support = recent["Low"].min()
        resistance = recent["High"].max()

        # --- Confidence & direction ------------------------------------------------
        signals = self._aggregate_signals(df)
        confidence = self._estimate_confidence(signals, df)
        direction = self._determine_direction(pred_5d_ret, pred_20d_ret, signals)

        return PredictionResult(
            current_price=round(current_price, 2),
            predicted_price_5d=round(pred_5d_price, 2),
            predicted_price_20d=round(pred_20d_price, 2),
            predicted_change_5d_pct=round(pred_5d_ret * 100, 2),
            predicted_change_20d_pct=round(pred_20d_ret * 100, 2),
            confidence=confidence,
            direction=direction,
            support_level=round(support, 2),
            resistance_level=round(resistance, 2),
            method="Linear Regression + Technical Features",
            r2_score_5d=r2_5d,
            r2_score_20d=r2_20d,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fit_predict(
        self,
        train_df: pd.DataFrame,
        feature_cols: list[str],
        target_col: str,
        X_latest: pd.DataFrame,
    ) -> tuple[float, float | None]:
        """Fit model and return (predicted_return, r2_score)."""
        if len(train_df) < 5:
            # Fallback: simple momentum extrapolation (no R² available)
            return float(self.df["Close"].pct_change(5).iloc[-1] or 0), None

        X = train_df[feature_cols].values
        y = train_df[target_col].values

        X_scaled = self._scaler.fit_transform(X)
        self._model.fit(X_scaled, y)

        y_pred_train = self._model.predict(X_scaled)
        r2 = round(float(r2_score(y, y_pred_train)), 4)

        X_new = self._scaler.transform(X_latest.values)
        return float(self._model.predict(X_new)[0]), r2

    @staticmethod
    def _aggregate_signals(df: pd.DataFrame) -> dict[str, int]:
        """Count bullish (+1) / bearish (-1) signals from indicators."""
        last = df.iloc[-1]
        signals: dict[str, int] = {}

        # SMA cross
        signals["sma_cross"] = 1 if last["SMA_20"] > last["SMA_50"] else -1

        # Price vs SMA20
        signals["price_sma20"] = 1 if last["Close"] > last["SMA_20"] else -1

        # RSI
        rsi = last["RSI_14"]
        if rsi < 30:
            signals["rsi"] = 1  # oversold -> potential bounce
        elif rsi > 70:
            signals["rsi"] = -1  # overbought -> potential drop
        else:
            signals["rsi"] = 0

        # MACD
        signals["macd"] = 1 if last["MACD"] > last["MACD_Signal"] else -1

        # Bollinger Bands
        if last["Close"] < last["BB_Lower"]:
            signals["bb"] = 1  # below lower band -> bounce
        elif last["Close"] > last["BB_Upper"]:
            signals["bb"] = -1  # above upper band -> pullback
        else:
            signals["bb"] = 0

        # Volume confirmation
        if "Vol_SMA_20" in df.columns and last.get("Vol_SMA_20", 0) > 0:
            vol_ratio = last["Volume"] / last["Vol_SMA_20"]
            # High volume confirms the prevailing price direction
            if vol_ratio > 1.5:
                price_chg = last["Close"] - df["Close"].iloc[-2] if len(df) > 1 else 0
                signals["volume"] = 1 if price_chg > 0 else -1
            else:
                signals["volume"] = 0

        return signals

    @staticmethod
    def _estimate_confidence(signals: dict[str, int], df: pd.DataFrame) -> str:
        values = list(signals.values())
        total = sum(values)
        agreement = abs(total) / max(len(values), 1)

        # Also factor in data quantity
        data_factor = min(len(df) / 200, 1.0)
        score = agreement * 0.6 + data_factor * 0.4

        if score > 0.65:
            return "high"
        elif score > 0.4:
            return "medium"
        return "low"

    @staticmethod
    def _determine_direction(
        pred_5d: float, pred_20d: float, signals: dict[str, int]
    ) -> str:
        signal_sum = sum(signals.values())
        model_score = (1 if pred_5d > 0.005 else (-1 if pred_5d < -0.005 else 0)) + (
            1 if pred_20d > 0.01 else (-1 if pred_20d < -0.01 else 0)
        )
        combined = signal_sum + model_score

        if combined >= 2:
            return "bullish"
        elif combined <= -2:
            return "bearish"
        return "neutral"

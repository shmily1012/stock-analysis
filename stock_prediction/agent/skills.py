"""Skills for the Stock Prediction Agent.

Each skill is a self-contained capability that the agent can invoke.
Skills:
    - fetch   : Retrieve historical stock data
    - analyze : Run technical indicator analysis
    - predict : Generate price predictions
    - report  : Full pipeline → formatted report
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import asdict

import pandas as pd
from tabulate import tabulate

from stock_prediction.analysis import TechnicalAnalyzer
from stock_prediction.fetcher import StockDataFetcher
from stock_prediction.fetcher.data_fetcher import Period
from stock_prediction.prediction import StockPredictor


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------


class Skill(ABC):
    """Abstract base for all agent skills."""

    name: str
    description: str

    @abstractmethod
    def execute(self, symbol: str, period_str: str, **kwargs: object) -> str:
        """Run the skill and return a human-readable result string."""


# ---------------------------------------------------------------------------
# Skill: Fetch
# ---------------------------------------------------------------------------


class FetchSkill(Skill):
    """Fetch historical OHLCV data for a stock."""

    name = "fetch"
    description = "Retrieve historical price data (Open/High/Low/Close/Volume)"

    def execute(self, symbol: str, period_str: str, **kwargs: object) -> str:
        fetcher = StockDataFetcher(symbol)
        period = Period.from_str(period_str)

        info = fetcher.get_info()
        history = fetcher.get_history(period)

        header = (
            f"Stock: {info.name} ({info.symbol})\n"
            f"Sector: {info.sector} | Industry: {info.industry}\n"
            f"Period: {period.value} | Data points: {len(history)}\n"
        )

        # Show first and last 5 rows
        summary_parts = [header, "--- First 5 rows ---"]
        summary_parts.append(
            tabulate(
                history.head().reset_index(),
                headers="keys",
                tablefmt="simple",
                floatfmt=".2f",
            )
        )
        summary_parts.append("\n--- Last 5 rows ---")
        summary_parts.append(
            tabulate(
                history.tail().reset_index(),
                headers="keys",
                tablefmt="simple",
                floatfmt=".2f",
            )
        )

        # Basic stats
        summary_parts.append("\n--- Summary Statistics ---")
        stats = history["Close"].describe()
        stats_table = [[k, f"{v:.2f}"] for k, v in stats.items()]
        summary_parts.append(tabulate(stats_table, tablefmt="simple"))

        return "\n".join(summary_parts)


# ---------------------------------------------------------------------------
# Skill: Analyze
# ---------------------------------------------------------------------------


class AnalyzeSkill(Skill):
    """Run technical analysis on a stock."""

    name = "analyze"
    description = "Compute technical indicators (MA, RSI, MACD, BB, ATR) and signals"

    def execute(self, symbol: str, period_str: str, **kwargs: object) -> str:
        fetcher = StockDataFetcher(symbol)
        period = Period.from_str(period_str)
        history = fetcher.get_history(period)
        info = fetcher.get_info()

        analyzer = TechnicalAnalyzer(history)
        summary = analyzer.summarize()

        header = (
            f"Technical Analysis: {info.name} ({info.symbol})\n"
            f"Period: {period.value}\n"
        )

        # Indicators
        ind_table = [
            ["SMA (20)", summary.sma_20],
            ["SMA (50)", summary.sma_50],
            ["EMA (12)", summary.ema_12],
            ["EMA (26)", summary.ema_26],
            ["RSI (14)", summary.rsi_14],
            ["MACD", summary.macd],
            ["MACD Signal", summary.macd_signal],
            ["MACD Histogram", summary.macd_histogram],
            ["Bollinger Upper", summary.bb_upper],
            ["Bollinger Middle", summary.bb_middle],
            ["Bollinger Lower", summary.bb_lower],
            ["ATR (14)", summary.atr_14],
        ]

        sig_table = [
            ["Price vs SMA20", summary.price_vs_sma20],
            ["Price vs SMA50", summary.price_vs_sma50],
            ["RSI Signal", summary.rsi_signal],
            ["MACD Signal", summary.macd_signal_str],
            ["Bollinger Position", summary.bb_position],
            ["Overall Trend", summary.trend],
            ["Volatility", summary.volatility],
        ]

        parts = [
            header,
            "--- Indicators ---",
            tabulate(ind_table, headers=["Indicator", "Value"], tablefmt="simple"),
            "\n--- Signals ---",
            tabulate(sig_table, headers=["Signal", "Status"], tablefmt="simple"),
        ]
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Skill: Predict
# ---------------------------------------------------------------------------


class PredictSkill(Skill):
    """Predict future stock price movement."""

    name = "predict"
    description = "Generate 5-day and 20-day price predictions with confidence level"

    def execute(self, symbol: str, period_str: str, **kwargs: object) -> str:
        fetcher = StockDataFetcher(symbol)
        period = Period.from_str(period_str)
        history = fetcher.get_history(period)
        info = fetcher.get_info()

        analyzer = TechnicalAnalyzer(history)
        enriched_df = analyzer.get_dataframe()

        predictor = StockPredictor(enriched_df)
        pred = predictor.predict()

        direction_icon = {
            "bullish": "[UP]",
            "bearish": "[DOWN]",
            "neutral": "[FLAT]",
        }.get(pred.direction, "")

        pred_table = [
            ["Current Price", f"{pred.current_price}"],
            [
                "5-Day Prediction",
                f"{pred.predicted_price_5d}  ({pred.predicted_change_5d_pct:+.2f}%)",
            ],
            [
                "20-Day Prediction",
                f"{pred.predicted_price_20d}  ({pred.predicted_change_20d_pct:+.2f}%)",
            ],
            ["Direction", f"{pred.direction.upper()} {direction_icon}"],
            ["Confidence", pred.confidence.upper()],
            ["Support Level", f"{pred.support_level}"],
            ["Resistance Level", f"{pred.resistance_level}"],
            ["Method", pred.method],
        ]

        parts = [
            f"Price Prediction: {info.name} ({info.symbol})",
            f"Based on {period.value} of historical data\n",
            tabulate(pred_table, headers=["Metric", "Value"], tablefmt="simple"),
            "\nDisclaimer: This is NOT financial advice. "
            "Predictions are for educational purposes only.",
        ]
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Skill: Report (full pipeline)
# ---------------------------------------------------------------------------


class ReportSkill(Skill):
    """Generate a comprehensive analysis report."""

    name = "report"
    description = "Full pipeline: fetch + analyze + predict → comprehensive report"

    def execute(self, symbol: str, period_str: str, **kwargs: object) -> str:
        from stock_prediction.agent.stock_agent import StockPredictionAgent

        agent = StockPredictionAgent(symbol)
        analysis = agent.analyze(period_str)

        output_format = kwargs.get("format", "text")
        if output_format == "json":
            return json.dumps(agent.format_report_json(analysis), indent=2)
        return agent.format_report(analysis)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

SKILL_REGISTRY: dict[str, Skill] = {
    "fetch": FetchSkill(),
    "analyze": AnalyzeSkill(),
    "predict": PredictSkill(),
    "report": ReportSkill(),
}


def get_skill(name: str) -> Skill:
    """Look up a skill by name."""
    if name not in SKILL_REGISTRY:
        available = ", ".join(SKILL_REGISTRY.keys())
        raise ValueError(f"Unknown skill '{name}'. Available skills: {available}")
    return SKILL_REGISTRY[name]


def list_skills() -> list[dict[str, str]]:
    """Return metadata for all registered skills."""
    return [
        {"name": s.name, "description": s.description}
        for s in SKILL_REGISTRY.values()
    ]

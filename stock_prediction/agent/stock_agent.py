"""Stock Prediction Agent — orchestrates fetching, analysis, and prediction."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd
from tabulate import tabulate

from stock_prediction.analysis import TechnicalAnalyzer
from stock_prediction.analysis.technical import TechnicalSummary
from stock_prediction.fetcher import StockDataFetcher
from stock_prediction.fetcher.data_fetcher import Period, StockInfo
from stock_prediction.prediction import StockPredictor
from stock_prediction.prediction.model import PredictionResult


@dataclass
class AnalysisReport:
    """Complete analysis report for a stock."""

    stock_info: StockInfo
    period: str
    data_points: int
    technical_summary: TechnicalSummary
    prediction: PredictionResult


class StockPredictionAgent:
    """High-level agent that coordinates the full analysis pipeline.

    Usage:
        agent = StockPredictionAgent("AAPL")
        report = agent.analyze("6mo")
        print(agent.format_report(report))
    """

    def __init__(self, symbol: str) -> None:
        self.symbol = symbol.upper()
        self.fetcher = StockDataFetcher(self.symbol)

    def analyze(self, period_str: str = "6mo") -> AnalysisReport:
        """Run the full pipeline: fetch -> analyze -> predict."""
        period = Period.from_str(period_str)

        # 1. Fetch
        stock_info = self.fetcher.get_info()
        history = self.fetcher.get_history(period)

        # 2. Technical analysis
        analyzer = TechnicalAnalyzer(history)
        technical_summary = analyzer.summarize()
        enriched_df = analyzer.get_dataframe()

        # 3. Prediction
        predictor = StockPredictor(enriched_df)
        prediction = predictor.predict()

        return AnalysisReport(
            stock_info=stock_info,
            period=period.value,
            data_points=len(history),
            technical_summary=technical_summary,
            prediction=prediction,
        )

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    @staticmethod
    def format_report(report: AnalysisReport) -> str:
        """Return a human-readable text report."""
        info = report.stock_info
        ts = report.technical_summary
        pred = report.prediction

        sections: list[str] = []

        # Header
        sections.append(
            f"{'=' * 64}\n"
            f"  Stock Analysis Report: {info.name} ({info.symbol})\n"
            f"{'=' * 64}"
        )

        # Basic info
        info_table = [
            ["Sector", info.sector],
            ["Industry", info.industry],
            ["Currency", info.currency],
            ["Market Cap", _fmt_market_cap(info.market_cap)],
            ["Analysis Period", report.period],
            ["Data Points", report.data_points],
        ]
        sections.append(
            "## Basic Information\n" + tabulate(info_table, tablefmt="simple")
        )

        # Technical indicators
        tech_table = [
            ["SMA (20)", ts.sma_20],
            ["SMA (50)", ts.sma_50],
            ["EMA (12)", ts.ema_12],
            ["EMA (26)", ts.ema_26],
            ["RSI (14)", ts.rsi_14],
            ["MACD", ts.macd],
            ["MACD Signal", ts.macd_signal],
            ["MACD Histogram", ts.macd_histogram],
            ["Bollinger Upper", ts.bb_upper],
            ["Bollinger Middle", ts.bb_middle],
            ["Bollinger Lower", ts.bb_lower],
            ["ATR (14)", ts.atr_14],
        ]
        sections.append(
            "## Technical Indicators\n"
            + tabulate(tech_table, headers=["Indicator", "Value"], tablefmt="simple")
        )

        # Signals
        signal_table = [
            ["Price vs SMA20", ts.price_vs_sma20],
            ["Price vs SMA50", ts.price_vs_sma50],
            ["RSI Signal", ts.rsi_signal],
            ["MACD Signal", ts.macd_signal_str],
            ["BB Position", ts.bb_position],
            ["Overall Trend", ts.trend],
            ["Volatility", ts.volatility],
        ]
        sections.append(
            "## Technical Signals\n"
            + tabulate(signal_table, headers=["Signal", "Status"], tablefmt="simple")
        )

        # Prediction
        direction_icon = {
            "bullish": "[UP]",
            "bearish": "[DOWN]",
            "neutral": "[FLAT]",
        }.get(pred.direction, "")

        pred_table = [
            ["Current Price", f"{pred.current_price}"],
            [
                "Predicted (5-day)",
                f"{pred.predicted_price_5d}  ({pred.predicted_change_5d_pct:+.2f}%)",
            ],
            [
                "Predicted (20-day)",
                f"{pred.predicted_price_20d}  ({pred.predicted_change_20d_pct:+.2f}%)",
            ],
            ["Direction", f"{pred.direction.upper()} {direction_icon}"],
            ["Confidence", pred.confidence.upper()],
            ["Support Level", pred.support_level],
            ["Resistance Level", pred.resistance_level],
            ["Model", pred.method],
        ]
        sections.append(
            "## Prediction\n"
            + tabulate(pred_table, headers=["Metric", "Value"], tablefmt="simple")
        )

        # Disclaimer
        sections.append(
            "## Disclaimer\n"
            "This analysis is for educational and informational purposes only.\n"
            "It does NOT constitute financial advice. Past performance does not\n"
            "guarantee future results. Always do your own research before making\n"
            "investment decisions."
        )

        return "\n\n".join(sections) + "\n"

    @staticmethod
    def format_report_json(report: AnalysisReport) -> dict:
        """Return the report as a JSON-serializable dict."""
        return {
            "stock_info": asdict(report.stock_info),
            "period": report.period,
            "data_points": report.data_points,
            "technical_summary": asdict(report.technical_summary),
            "prediction": asdict(report.prediction),
        }


def _fmt_market_cap(cap: float | None) -> str:
    if cap is None:
        return "N/A"
    if cap >= 1e12:
        return f"{cap / 1e12:.2f}T"
    if cap >= 1e9:
        return f"{cap / 1e9:.2f}B"
    if cap >= 1e6:
        return f"{cap / 1e6:.2f}M"
    return str(cap)

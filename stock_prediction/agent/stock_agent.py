"""Stock Prediction Agent — orchestrates fetching, analysis, and prediction."""

from __future__ import annotations

import os
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
    history: pd.DataFrame | None = None


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
            history=history,
        )

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    @staticmethod
    def format_report(report: AnalysisReport, lang: str = "cn") -> str:
        """Return a human-readable text report.

        Args:
            report: The analysis report data.
            lang: 'cn' for Chinese (default), 'en' for English.
        """
        if lang == "cn":
            return StockPredictionAgent._format_report_cn(report)
        return StockPredictionAgent._format_report_en(report)

    @staticmethod
    def _format_report_en(report: AnalysisReport) -> str:
        """English-language report."""
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

        # Price history
        if report.history is not None and not report.history.empty:
            sections.append(_format_price_history_en(report.history))

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
            ["Volume Avg (20)", f"{ts.volume_avg_20:,.0f}"],
            ["Volume Ratio", ts.volume_ratio],
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
            ["Volume Signal", ts.volume_signal],
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
            ["Model R² (5d)", _fmt_r2(pred.r2_score_5d)],
            ["Model R² (20d)", _fmt_r2(pred.r2_score_20d)],
            ["Model", pred.method],
        ]
        sections.append(
            "## Prediction\n"
            + tabulate(pred_table, headers=["Metric", "Value"], tablefmt="simple")
        )

        # Chart
        chart_path = _generate_chart(report)
        if chart_path:
            sections.append(f"## Price Chart\nSaved to: {chart_path}")

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
    def _format_report_cn(report: AnalysisReport) -> str:
        """Chinese-language report."""
        info = report.stock_info
        ts = report.technical_summary
        pred = report.prediction

        sections: list[str] = []

        # Header
        sections.append(
            f"{'=' * 64}\n"
            f"  股票分析报告: {info.name} ({info.symbol})\n"
            f"{'=' * 64}"
        )

        # Basic info
        info_table = [
            ["行业板块", info.sector],
            ["细分行业", info.industry],
            ["货币", info.currency],
            ["市值", _fmt_market_cap(info.market_cap)],
            ["分析周期", report.period],
            ["数据点数", report.data_points],
        ]
        sections.append(
            "## 基本信息\n" + tabulate(info_table, tablefmt="simple")
        )

        # Price history
        if report.history is not None and not report.history.empty:
            sections.append(_format_price_history_cn(report.history))

        # Technical indicators
        tech_table = [
            ["SMA (20)", ts.sma_20],
            ["SMA (50)", ts.sma_50],
            ["EMA (12)", ts.ema_12],
            ["EMA (26)", ts.ema_26],
            ["RSI (14)", ts.rsi_14],
            ["MACD", ts.macd],
            ["MACD 信号线", ts.macd_signal],
            ["MACD 柱状图", ts.macd_histogram],
            ["布林带上轨", ts.bb_upper],
            ["布林带中轨", ts.bb_middle],
            ["布林带下轨", ts.bb_lower],
            ["ATR (14)", ts.atr_14],
            ["20日均量", f"{ts.volume_avg_20:,.0f}"],
            ["量比", ts.volume_ratio],
        ]
        sections.append(
            "## 技术指标\n"
            + tabulate(tech_table, headers=["指标", "数值"], tablefmt="simple")
        )

        # Signals
        _trend_cn = {"uptrend": "上升趋势", "downtrend": "下降趋势", "sideways": "横盘整理"}
        _vol_cn = {"high": "高", "medium": "中", "low": "低"}
        _rsi_cn = {"overbought": "超买", "oversold": "超卖", "neutral": "中性"}
        _macd_cn = {"bullish": "看涨", "bearish": "看跌"}
        _bb_cn = {"above_upper": "突破上轨", "below_lower": "跌破下轨", "within": "通道内"}
        _vol_sig_cn = {"high_volume": "放量", "low_volume": "缩量", "normal": "正常", "N/A": "N/A"}
        _pos_cn = {"above": "上方", "below": "下方"}

        signal_table = [
            ["价格 vs SMA20", _pos_cn.get(ts.price_vs_sma20, ts.price_vs_sma20)],
            ["价格 vs SMA50", _pos_cn.get(ts.price_vs_sma50, ts.price_vs_sma50)],
            ["RSI 信号", _rsi_cn.get(ts.rsi_signal, ts.rsi_signal)],
            ["MACD 信号", _macd_cn.get(ts.macd_signal_str, ts.macd_signal_str)],
            ["布林带位置", _bb_cn.get(ts.bb_position, ts.bb_position)],
            ["成交量信号", _vol_sig_cn.get(ts.volume_signal, ts.volume_signal)],
            ["总体趋势", _trend_cn.get(ts.trend, ts.trend)],
            ["波动率", _vol_cn.get(ts.volatility, ts.volatility)],
        ]
        sections.append(
            "## 技术信号\n"
            + tabulate(signal_table, headers=["信号", "状态"], tablefmt="simple")
        )

        # Prediction
        _dir_cn = {"bullish": "看涨 [UP]", "bearish": "看跌 [DOWN]", "neutral": "中性 [FLAT]"}
        _conf_cn = {"high": "高", "medium": "中", "low": "低"}

        pred_table = [
            ["当前价格", f"{pred.current_price}"],
            [
                "5日预测价",
                f"{pred.predicted_price_5d}  ({pred.predicted_change_5d_pct:+.2f}%)",
            ],
            [
                "20日预测价",
                f"{pred.predicted_price_20d}  ({pred.predicted_change_20d_pct:+.2f}%)",
            ],
            ["方向", _dir_cn.get(pred.direction, pred.direction)],
            ["置信度", _conf_cn.get(pred.confidence, pred.confidence)],
            ["支撑位", pred.support_level],
            ["阻力位", pred.resistance_level],
            ["模型 R² (5日)", _fmt_r2(pred.r2_score_5d)],
            ["模型 R² (20日)", _fmt_r2(pred.r2_score_20d)],
            ["模型", pred.method],
        ]
        sections.append(
            "## 价格预测\n"
            + tabulate(pred_table, headers=["指标", "数值"], tablefmt="simple")
        )

        # Chart
        chart_path = _generate_chart(report)
        if chart_path:
            sections.append(f"## 价格走势图\n已保存至: {chart_path}")

        # Disclaimer
        sections.append(
            "## 免责声明\n"
            "本分析仅供教育和参考用途，不构成任何投资建议。\n"
            "过往表现不代表未来收益。投资有风险，入市需谨慎。\n"
            "请在做出投资决策前进行独立研究。"
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


def _fmt_r2(r2: float | None) -> str:
    if r2 is None:
        return "N/A (insufficient data)"
    return f"{r2:.4f}"


def _format_price_history_en(history: pd.DataFrame) -> str:
    """Format first/last 5 rows of OHLCV data for the report."""
    parts = ["## Price History (Recent)"]
    parts.append("--- First 5 trading days ---")
    parts.append(
        tabulate(
            history.head().reset_index(),
            headers="keys",
            tablefmt="simple",
            floatfmt=".2f",
        )
    )
    parts.append("\n--- Last 5 trading days ---")
    parts.append(
        tabulate(
            history.tail().reset_index(),
            headers="keys",
            tablefmt="simple",
            floatfmt=".2f",
        )
    )
    # Period return
    first_close = history["Close"].iloc[0]
    last_close = history["Close"].iloc[-1]
    period_ret = (last_close / first_close - 1) * 100
    parts.append(f"\nPeriod Return: {period_ret:+.2f}%")
    return "\n".join(parts)


def _format_price_history_cn(history: pd.DataFrame) -> str:
    """Format first/last 5 rows of OHLCV data (Chinese labels)."""
    display = history.copy()
    display.index.name = "日期"
    display.columns = ["开盘", "最高", "最低", "收盘", "成交量"]
    parts = ["## 价格历史（近期）"]
    parts.append("--- 前5个交易日 ---")
    parts.append(
        tabulate(
            display.head().reset_index(),
            headers="keys",
            tablefmt="simple",
            floatfmt=".2f",
        )
    )
    parts.append("\n--- 后5个交易日 ---")
    parts.append(
        tabulate(
            display.tail().reset_index(),
            headers="keys",
            tablefmt="simple",
            floatfmt=".2f",
        )
    )
    first_close = history["Close"].iloc[0]
    last_close = history["Close"].iloc[-1]
    period_ret = (last_close / first_close - 1) * 100
    parts.append(f"\n区间收益率: {period_ret:+.2f}%")
    return "\n".join(parts)


def _generate_chart(report: AnalysisReport) -> str | None:
    """Generate a price chart with technical overlays, saved as PNG.

    Returns the file path, or None if chart generation fails.
    """
    if report.history is None or report.history.empty:
        return None

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError:
        return None

    history = report.history
    info = report.stock_info

    fig, axes = plt.subplots(
        3, 1, figsize=(14, 10), height_ratios=[3, 1, 1],
        gridspec_kw={"hspace": 0.3},
    )

    dates = history.index

    # ── Panel 1: Price + Moving Averages + Bollinger Bands ──────────
    ax1 = axes[0]
    ax1.plot(dates, history["Close"], label="Close", color="#1f77b4", linewidth=1.5)

    from stock_prediction.analysis import TechnicalAnalyzer
    analyzer = TechnicalAnalyzer(history)
    df = analyzer.get_dataframe()

    ax1.plot(dates, df["SMA_20"], label="SMA 20", color="#ff7f0e", linewidth=1, alpha=0.8)
    ax1.plot(dates, df["SMA_50"], label="SMA 50", color="#2ca02c", linewidth=1, alpha=0.8)
    ax1.fill_between(
        dates, df["BB_Upper"], df["BB_Lower"],
        alpha=0.1, color="#9467bd", label="Bollinger Bands",
    )
    ax1.plot(dates, df["BB_Upper"], color="#9467bd", linewidth=0.5, alpha=0.5)
    ax1.plot(dates, df["BB_Lower"], color="#9467bd", linewidth=0.5, alpha=0.5)

    ax1.set_title(f"{info.name} ({info.symbol}) — {report.period}", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Price")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))

    # ── Panel 2: Volume ─────────────────────────────────────────────
    ax2 = axes[1]
    colors = ["#2ca02c" if c >= o else "#d62728"
              for c, o in zip(history["Close"], history["Open"])]
    ax2.bar(dates, history["Volume"], color=colors, alpha=0.7, width=0.8)
    if "Vol_SMA_20" in df.columns:
        ax2.plot(dates, df["Vol_SMA_20"], color="#ff7f0e", linewidth=1, label="Vol SMA 20")
        ax2.legend(loc="upper left", fontsize=8)
    ax2.set_ylabel("Volume")
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))

    # ── Panel 3: RSI ────────────────────────────────────────────────
    ax3 = axes[2]
    ax3.plot(dates, df["RSI_14"], color="#1f77b4", linewidth=1)
    ax3.axhline(y=70, color="#d62728", linestyle="--", linewidth=0.8, alpha=0.7, label="Overbought (70)")
    ax3.axhline(y=30, color="#2ca02c", linestyle="--", linewidth=0.8, alpha=0.7, label="Oversold (30)")
    ax3.fill_between(dates, 30, 70, alpha=0.05, color="gray")
    ax3.set_ylabel("RSI (14)")
    ax3.set_ylim(0, 100)
    ax3.legend(loc="upper left", fontsize=8)
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))

    fig.autofmt_xdate()

    # Save
    chart_dir = os.path.join(os.getcwd(), "charts")
    os.makedirs(chart_dir, exist_ok=True)
    filename = f"{info.symbol}_{report.period}_analysis.png"
    chart_path = os.path.join(chart_dir, filename)
    fig.savefig(chart_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    return chart_path

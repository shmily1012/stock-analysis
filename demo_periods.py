"""Demo: multi-stock reports + comparison for 3-month and 6-month periods."""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from stock_prediction.analysis import TechnicalAnalyzer
from stock_prediction.prediction import StockPredictor
from stock_prediction.agent.stock_agent import StockPredictionAgent, AnalysisReport
from stock_prediction.fetcher.data_fetcher import StockInfo

# ── Stock configurations ──────────────────────────────────────────────────
STOCKS = [
    StockInfo("MSFT", "Microsoft Corporation", "Technology", "Software—Infrastructure", 3_100_000_000_000, "USD"),
    StockInfo("GOOGL", "Alphabet Inc.", "Communication Services", "Internet Content & Information", 2_200_000_000_000, "USD"),
    StockInfo("META", "Meta Platforms, Inc.", "Communication Services", "Internet Content & Information", 1_700_000_000_000, "USD"),
    StockInfo("SNDK", "SanDisk Corporation (WDC)", "Technology", "Data Storage", 23_000_000_000, "USD"),
    StockInfo("MU", "Micron Technology, Inc.", "Technology", "Semiconductors", 110_000_000_000, "USD"),
]

STOCK_PARAMS = {
    "MSFT":  {"base": 420.0, "vol": 0.015, "drift": 0.0005,  "avg_volume": 25_000_000},
    "GOOGL": {"base": 185.0, "vol": 0.018, "drift": 0.0003,  "avg_volume": 30_000_000},
    "META":  {"base": 620.0, "vol": 0.022, "drift": 0.0008,  "avg_volume": 18_000_000},
    "SNDK":  {"base": 52.0,  "vol": 0.028, "drift": -0.0002, "avg_volume": 8_000_000},
    "MU":    {"base": 98.0,  "vol": 0.030, "drift": 0.0004,  "avg_volume": 22_000_000},
}

PERIODS = {
    "3mo": {"start": "2025-11-15", "end": "2026-02-14", "label": "3 Months"},
    "6mo": {"start": "2025-08-15", "end": "2026-02-14", "label": "6 Months"},
}


def generate_stock_data(symbol: str, seed: int, start: str, end: str) -> pd.DataFrame:
    """Generate realistic OHLCV data for a stock over a given date range."""
    np.random.seed(seed)
    params = STOCK_PARAMS[symbol]
    dates = pd.bdate_range(start=start, end=end)
    n = len(dates)

    returns = np.random.normal(params["drift"], params["vol"], n)
    prices = [params["base"]]
    for r in returns[1:]:
        prices.append(prices[-1] * (1 + r))

    close = np.array(prices)
    high = close * (1 + np.abs(np.random.normal(0, 0.010, n)))
    low = close * (1 - np.abs(np.random.normal(0, 0.010, n)))
    open_ = close * (1 + np.random.normal(0, 0.004, n))
    base_vol = params["avg_volume"]
    volume = np.random.randint(int(base_vol * 0.5), int(base_vol * 1.8), n)

    df = pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )
    df.index.name = "Date"
    return df


def run_reports_for_period(period_key: str):
    """Run full reports for all stocks for a given period."""
    pcfg = PERIODS[period_key]
    print(f"\n{'='*70}")
    print(f"  PERIOD: {pcfg['label']} ({period_key})")
    print(f"  Date range: {pcfg['start']} → {pcfg['end']}")
    print(f"{'='*70}\n")

    all_data = {}
    all_technicals = {}

    for i, stock in enumerate(STOCKS):
        # Use different seeds per period to get different but reproducible data
        seed = 200 + i if period_key == "3mo" else 300 + i
        df = generate_stock_data(stock.symbol, seed, pcfg["start"], pcfg["end"])

        analyzer = TechnicalAnalyzer(df)
        technical_summary = analyzer.summarize()
        enriched_df = analyzer.get_dataframe()

        predictor = StockPredictor(enriched_df)
        prediction = predictor.predict()

        report = AnalysisReport(
            stock_info=stock,
            period=period_key,
            data_points=len(df),
            technical_summary=technical_summary,
            prediction=prediction,
            history=df,
        )

        print(StockPredictionAgent.format_report(report))
        print()

        all_data[stock.symbol] = df
        all_technicals[stock.symbol] = technical_summary

    return all_data, all_technicals


def build_comparison_chart(all_data, all_technicals, period_key):
    """Build a 4-panel comparison dashboard for a given period."""
    pcfg = PERIODS[period_key]
    symbols = [s.symbol for s in STOCKS]
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336"]

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(f"Multi-Stock Comparison — {pcfg['label']} ({period_key})",
                 fontsize=18, fontweight="bold", y=0.98)
    gs = GridSpec(2, 2, hspace=0.35, wspace=0.3, top=0.93, bottom=0.06)

    # 1. Normalized performance
    ax1 = fig.add_subplot(gs[0, 0])
    for sym, color in zip(symbols, colors):
        df = all_data[sym]
        norm = (df["Close"] / df["Close"].iloc[0] - 1) * 100
        ax1.plot(norm.index, norm.values, label=sym, color=color, linewidth=1.8)
    ax1.set_title("Normalized Price Performance (%)", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Return (%)")
    ax1.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(alpha=0.3)
    ax1.tick_params(axis="x", rotation=30)

    # 2. RSI comparison
    ax2 = fig.add_subplot(gs[0, 1])
    rsi_vals = [all_technicals[s].rsi_14 for s in symbols]
    bars = ax2.bar(symbols, rsi_vals, color=colors, width=0.6, edgecolor="white", linewidth=1.2)
    ax2.axhline(70, color="red", linewidth=1, linestyle="--", label="Overbought (70)")
    ax2.axhline(30, color="green", linewidth=1, linestyle="--", label="Oversold (30)")
    ax2.axhline(50, color="gray", linewidth=0.5, linestyle=":")
    ax2.set_title("RSI (14) Comparison", fontsize=13, fontweight="bold")
    ax2.set_ylabel("RSI")
    ax2.set_ylim(0, 100)
    ax2.legend(fontsize=9)
    for bar, val in zip(bars, rsi_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                 f"{val:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax2.grid(axis="y", alpha=0.3)

    # 3. MACD bar chart
    ax3 = fig.add_subplot(gs[1, 0])
    macd_vals = [all_technicals[s].macd for s in symbols]
    macd_colors = ["#4CAF50" if v >= 0 else "#F44336" for v in macd_vals]
    bars3 = ax3.bar(symbols, macd_vals, color=macd_colors, width=0.6, edgecolor="white", linewidth=1.2)
    ax3.axhline(0, color="gray", linewidth=0.8)
    ax3.set_title("MACD Value", fontsize=13, fontweight="bold")
    ax3.set_ylabel("MACD")
    for bar, val in zip(bars3, macd_vals):
        offset = 0.3 if val >= 0 else -0.8
        ax3.text(bar.get_x() + bar.get_width() / 2, val + offset,
                 f"{val:.2f}", ha="center", va="bottom" if val >= 0 else "top",
                 fontsize=10, fontweight="bold")
    ax3.grid(axis="y", alpha=0.3)

    # 4. Risk vs Return
    ax4 = fig.add_subplot(gs[1, 1])
    returns = []
    atrs_pct = []
    for sym in symbols:
        df = all_data[sym]
        ret = (df["Close"].iloc[-1] / df["Close"].iloc[0] - 1) * 100
        atr = all_technicals[sym].atr_14
        atr_pct = (atr / df["Close"].iloc[-1]) * 100
        returns.append(ret)
        atrs_pct.append(atr_pct)

    for sym, ret, atr, color in zip(symbols, returns, atrs_pct, colors):
        ax4.scatter(atr, ret, s=200, c=color, edgecolors="black", linewidth=1, zorder=5)
        ax4.annotate(sym, (atr, ret), textcoords="offset points", xytext=(8, 6),
                     fontsize=11, fontweight="bold", color=color)
    ax4.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax4.set_title("Risk vs Return (ATR% vs Period Return)", fontsize=13, fontweight="bold")
    ax4.set_xlabel("ATR as % of Price (Volatility)")
    ax4.set_ylabel("Period Return (%)")
    ax4.grid(alpha=0.3)

    outpath = f"charts/multi_stock_comparison_{period_key}.png"
    plt.savefig(outpath, dpi=150, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close()
    print(f"Saved: {outpath}")
    return outpath


# ── Run both periods ──────────────────────────────────────────────────────
for period in ["3mo", "6mo"]:
    data, technicals = run_reports_for_period(period)
    build_comparison_chart(data, technicals, period)

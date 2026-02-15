"""
Intuitive Investment Signal Dashboard
Converts complex technical indicators into a single -100 ~ +100 score
with gauge meters, heatmap, and ranking chart.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Wedge
import matplotlib.patheffects as pe

from stock_prediction.analysis import TechnicalAnalyzer
from stock_prediction.prediction import StockPredictor
from stock_prediction.fetcher.data_fetcher import StockInfo

# ── Stock configs ─────────────────────────────────────────────────────────
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


# ── Signal scoring system ─────────────────────────────────────────────────
# Weights:  Trend(SMA) 30% | RSI 20% | MACD 25% | Bollinger 15% | Volume 10%
# Score range: -100 (strong sell) ~ +100 (strong buy)

def compute_signal_score(summary):
    scores = {}

    # 1) Trend score (-30 ~ +30)
    trend_score = 0
    if summary.price_vs_sma20 == "above":
        trend_score += 15
    else:
        trend_score -= 15
    if summary.price_vs_sma50 == "above":
        trend_score += 15
    else:
        trend_score -= 15
    scores["Trend"] = trend_score

    # 2) RSI momentum (-20 ~ +20)
    rsi = summary.rsi_14
    if rsi >= 70:
        rsi_score = -10   # overbought -> pullback risk
    elif rsi >= 60:
        rsi_score = 15    # strong momentum
    elif rsi >= 40:
        rsi_score = 0     # neutral
    elif rsi >= 30:
        rsi_score = -15   # weak
    else:
        rsi_score = 10    # oversold -> bounce opportunity
    scores["RSI"] = rsi_score

    # 3) MACD (-25 ~ +25)
    if summary.macd_signal_str == "bullish":
        macd_score = 15
    else:
        macd_score = -15
    hist = summary.macd_histogram
    if hist > 0:
        macd_score += 10
    else:
        macd_score -= 10
    macd_score = max(-25, min(25, macd_score))
    scores["MACD"] = macd_score

    # 4) Bollinger Bands (-15 ~ +15)
    if summary.bb_position == "above_upper":
        bb_score = -15
    elif summary.bb_position == "below_lower":
        bb_score = 15
    else:
        bb_score = 0
    scores["BB"] = bb_score

    # 5) Volume (-10 ~ +10)
    vol_ratio = summary.volume_ratio
    if vol_ratio > 1.5:
        vol_score = 10 if trend_score > 0 else -10
    elif vol_ratio < 0.5:
        vol_score = -5
    else:
        vol_score = 0
    scores["Volume"] = vol_score

    total = max(-100, min(100, sum(scores.values())))
    return total, scores


def score_to_label(score):
    """Return (English label, Chinese label, color)."""
    if score >= 40:
        return "STRONG BUY", "#1B5E20"
    elif score >= 15:
        return "BUY", "#4CAF50"
    elif score > -15:
        return "HOLD", "#FF9800"
    elif score > -40:
        return "SELL", "#F44336"
    else:
        return "STRONG SELL", "#B71C1C"


def score_to_action_cn(score):
    if score >= 40:
        return "BUY"
    elif score >= 15:
        return "BUY DIP"
    elif score > -15:
        return "HOLD"
    elif score > -40:
        return "REDUCE"
    else:
        return "SELL"


# ── Gauge drawing ─────────────────────────────────────────────────────────
def draw_gauge(ax, score, symbol):
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-0.4, 1.4)
    ax.set_aspect("equal")
    ax.axis("off")

    segments = [
        (-100, -40, "#B71C1C"),
        (-40, -15, "#F44336"),
        (-15, 15, "#FF9800"),
        (15, 40, "#4CAF50"),
        (40, 100, "#1B5E20"),
    ]
    for s_min, s_max, color in segments:
        theta1 = 180 - (s_max + 100) / 200 * 180
        theta2 = 180 - (s_min + 100) / 200 * 180
        wedge = Wedge((0, 0), 1.0, theta1, theta2, width=0.3, facecolor=color, alpha=0.3)
        ax.add_patch(wedge)

    # Needle
    angle_deg = 180 - (score + 100) / 200 * 180
    angle_rad = np.radians(angle_deg)
    nx = 0.85 * np.cos(angle_rad)
    ny = 0.85 * np.sin(angle_rad)
    ax.annotate("", xy=(nx, ny), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=2.5))
    ax.plot(0, 0, "ko", markersize=6, zorder=10)

    signal_label, signal_color = score_to_label(score)
    action = score_to_action_cn(score)
    ax.text(0, 1.3, symbol, fontsize=16, fontweight="bold", ha="center", va="center")
    ax.text(0, -0.05, f"{score:+d}", fontsize=28, fontweight="bold", ha="center", va="center",
            color=signal_color,
            path_effects=[pe.withStroke(linewidth=2, foreground="white")])
    ax.text(0, -0.28, f"{signal_label}", fontsize=12, ha="center", va="center",
            color=signal_color, fontweight="bold")

    ax.text(-1.1, -0.05, "-100", fontsize=7, ha="center", color="gray")
    ax.text(1.1, -0.05, "+100", fontsize=7, ha="center", color="gray")
    ax.text(0, 1.08, "0", fontsize=7, ha="center", color="gray")


# ── Heatmap ───────────────────────────────────────────────────────────────
def draw_heatmap(ax, all_scores, symbols, periods_list):
    components = ["Trend", "RSI", "MACD", "BB", "Volume"]
    n_stocks = len(symbols)

    col_labels = []
    matrix = []
    for period_key in periods_list:
        for sym in symbols:
            col_labels.append(f"{sym}\n{PERIODS[period_key]['label']}")
            row = [all_scores[period_key][sym]["components"][c] for c in components]
            matrix.append(row)

    matrix = np.array(matrix).T

    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=-30, vmax=30)
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=8)
    ax.set_yticks(range(len(components)))
    ax.set_yticklabels(components, fontsize=10)
    ax.set_title("Signal Breakdown Heatmap (Green=Bullish  Red=Bearish)",
                 fontsize=13, fontweight="bold", pad=12)

    for i in range(len(components)):
        for j in range(len(col_labels)):
            val = matrix[i, j]
            color = "white" if abs(val) > 15 else "black"
            ax.text(j, i, f"{val:+.0f}", ha="center", va="center",
                    fontsize=9, fontweight="bold", color=color)

    ax.axvline(n_stocks - 0.5, color="white", linewidth=3)
    return im


# ── Ranking chart ─────────────────────────────────────────────────────────
def draw_ranking(ax, all_scores, symbols, periods_list):
    x = np.arange(len(symbols))
    width = 0.35

    for i, period_key in enumerate(periods_list):
        scores = [all_scores[period_key][sym]["total"] for sym in symbols]
        colors = [score_to_label(s)[1] for s in scores]
        offset = -width / 2 + i * width
        bars = ax.bar(x + offset, scores, width * 0.9, color=colors, edgecolor="white",
                      linewidth=1.2, label=PERIODS[period_key]["label"])
        for bar, val in zip(bars, scores):
            y_pos = val + 2 if val >= 0 else val - 5
            ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                    f"{val:+d}", ha="center", va="bottom" if val >= 0 else "top",
                    fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(symbols, fontsize=12, fontweight="bold")
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.axhline(40, color="green", linewidth=0.5, linestyle=":", alpha=0.5)
    ax.axhline(-40, color="red", linewidth=0.5, linestyle=":", alpha=0.5)
    ax.set_ylim(-80, 80)
    ax.set_ylabel("Signal Score", fontsize=11)
    ax.set_title("Signal Score Ranking (3mo vs 6mo)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    ax.text(len(symbols) - 0.3, 45, "STRONG BUY zone", fontsize=8, color="green", alpha=0.7)
    ax.text(len(symbols) - 0.3, -50, "STRONG SELL zone", fontsize=8, color="red", alpha=0.7)


# ══════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════
symbols = [s.symbol for s in STOCKS]
periods_list = ["3mo", "6mo"]
all_scores = {}

for period_key in periods_list:
    pcfg = PERIODS[period_key]
    all_scores[period_key] = {}
    print(f"\n{'='*60}")
    print(f"  {pcfg['label']} Signal Scores")
    print(f"{'='*60}")

    for i, stock in enumerate(STOCKS):
        seed = 200 + i if period_key == "3mo" else 300 + i
        df = generate_stock_data(stock.symbol, seed, pcfg["start"], pcfg["end"])
        analyzer = TechnicalAnalyzer(df)
        summary = analyzer.summarize()

        total, components = compute_signal_score(summary)
        label, color = score_to_label(total)
        action = score_to_action_cn(total)

        all_scores[period_key][stock.symbol] = {
            "total": total,
            "components": components,
            "label": label,
            "action": action,
        }

        print(f"\n  {stock.symbol} ({stock.name})")
        print(f"    Score: {total:+d}  ->  {label}  ->  Action: {action}")
        for k, v in components.items():
            bar = "+" * max(0, v) + "-" * max(0, -v)
            print(f"      {k:8s}: {v:+4d}  {bar}")

# ── Summary table ─────────────────────────────────────────────────────────
print(f"\n\n{'='*75}")
print("  INVESTMENT SIGNAL SUMMARY")
print(f"{'='*75}")
print(f"{'Stock':<8s} {'3mo Score':>10s} {'3mo Signal':>12s} {'6mo Score':>10s} {'6mo Signal':>12s} {'Action':>10s}")
print("-" * 75)
for sym in symbols:
    s3 = all_scores["3mo"][sym]
    s6 = all_scores["6mo"][sym]
    avg = (s3["total"] + s6["total"]) / 2
    combined_action = score_to_action_cn(avg)
    print(f"{sym:<8s} {s3['total']:>+10d} {s3['label']:>12s} {s6['total']:>+10d} {s6['label']:>12s} {combined_action:>10s}")

# ── Build dashboard ───────────────────────────────────────────────────────
fig = plt.figure(figsize=(22, 20))
fig.suptitle("Investment Signal Dashboard\nScore Range: -100 (Strong Sell) to +100 (Strong Buy)",
             fontsize=18, fontweight="bold", y=0.99)

# Row 1: 3-month gauges
gs_top = GridSpec(1, 5, top=0.92, bottom=0.68, left=0.03, right=0.97, wspace=0.15)
for j, sym in enumerate(symbols):
    ax = fig.add_subplot(gs_top[0, j])
    draw_gauge(ax, all_scores["3mo"][sym]["total"], sym)

fig.text(0.02, 0.935, "3-Month Signals", fontsize=14, fontweight="bold", color="#333")

# Row 2: 6-month gauges
gs_mid = GridSpec(1, 5, top=0.66, bottom=0.42, left=0.03, right=0.97, wspace=0.15)
for j, sym in enumerate(symbols):
    ax = fig.add_subplot(gs_mid[0, j])
    draw_gauge(ax, all_scores["6mo"][sym]["total"], sym)

fig.text(0.02, 0.675, "6-Month Signals", fontsize=14, fontweight="bold", color="#333")

# Row 3: Ranking + Heatmap
gs_bot = GridSpec(1, 2, top=0.38, bottom=0.05, left=0.06, right=0.97, wspace=0.25)
ax_rank = fig.add_subplot(gs_bot[0, 0])
draw_ranking(ax_rank, all_scores, symbols, periods_list)

ax_heat = fig.add_subplot(gs_bot[0, 1])
draw_heatmap(ax_heat, all_scores, symbols, periods_list)

fig.text(0.5, 0.01,
         "DISCLAIMER: For educational purposes only. NOT financial advice. Invest at your own risk.",
         fontsize=10, ha="center", color="gray", style="italic")

outpath = "charts/signal_dashboard.png"
plt.savefig(outpath, dpi=150, bbox_inches="tight", facecolor="white", edgecolor="none")
plt.close()
print(f"\n\nSaved: {outpath}")

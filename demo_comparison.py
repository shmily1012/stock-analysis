"""Cross-stock comparison: normalized performance + radar chart of technicals."""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from demo_multi_report import generate_stock_data, STOCK_PARAMS, STOCKS

# ── Generate data for all stocks ──────────────────────────────────────────
from stock_prediction.analysis import TechnicalAnalyzer

data = {}
technicals = {}
for i, stock in enumerate(STOCKS):
    df = generate_stock_data(stock.symbol, seed=100 + i)
    analyzer = TechnicalAnalyzer(df)
    summary = analyzer.summarize()
    data[stock.symbol] = df
    technicals[stock.symbol] = summary

symbols = [s.symbol for s in STOCKS]
colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336"]

# ── Figure setup ──────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 14))
fig.suptitle("Multi-Stock Comparison Dashboard", fontsize=18, fontweight="bold", y=0.98)
gs = GridSpec(2, 2, hspace=0.35, wspace=0.3, top=0.93, bottom=0.06)

# ── 1. Normalized price performance ──────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
for sym, color in zip(symbols, colors):
    df = data[sym]
    norm = (df["Close"] / df["Close"].iloc[0] - 1) * 100
    ax1.plot(norm.index, norm.values, label=sym, color=color, linewidth=1.8)
ax1.set_title("Normalized Price Performance (%)", fontsize=13, fontweight="bold")
ax1.set_ylabel("Return (%)")
ax1.axhline(0, color="gray", linewidth=0.5, linestyle="--")
ax1.legend(loc="upper left", fontsize=9)
ax1.grid(alpha=0.3)
ax1.tick_params(axis="x", rotation=30)

# ── 2. RSI comparison ───────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
rsi_vals = [technicals[s].rsi_14 for s in symbols]
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

# ── 3. MACD bar chart ───────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
macd_vals = [technicals[s].macd for s in symbols]
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

# ── 4. Volatility & period return scatter ─────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
returns = []
atrs_pct = []
for sym in symbols:
    df = data[sym]
    ret = (df["Close"].iloc[-1] / df["Close"].iloc[0] - 1) * 100
    atr = technicals[sym].atr_14
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

plt.savefig("charts/multi_stock_comparison.png", dpi=150, bbox_inches="tight",
            facecolor="white", edgecolor="none")
plt.close()
print("Saved: charts/multi_stock_comparison.png")

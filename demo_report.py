"""Demo script: run full TSLA report with sample data (no network needed)."""

import numpy as np
import pandas as pd

from stock_prediction.analysis import TechnicalAnalyzer
from stock_prediction.prediction import StockPredictor
from stock_prediction.agent.stock_agent import StockPredictionAgent, AnalysisReport
from stock_prediction.fetcher.data_fetcher import StockInfo

# ── Generate realistic TSLA data with enough lookback for indicators ──────
# We need ~60+ trading days so indicators like SMA_50 are well-defined.
# We'll generate 3 months of data and label the report as "1mo" analysis.
np.random.seed(42)
dates = pd.bdate_range(start="2025-11-15", end="2026-02-14")
n = len(dates)

# TSLA has been trading around $350-$390 range recently
base_price = 355.0
returns = np.random.normal(0.001, 0.025, n)  # daily ~2.5% vol typical for TSLA
prices = [base_price]
for r in returns[1:]:
    prices.append(prices[-1] * (1 + r))

close = np.array(prices)
high = close * (1 + np.abs(np.random.normal(0, 0.012, n)))
low = close * (1 - np.abs(np.random.normal(0, 0.012, n)))
open_ = close * (1 + np.random.normal(0, 0.005, n))
volume = np.random.randint(50_000_000, 120_000_000, n)

df = pd.DataFrame(
    {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
    index=dates,
)
df.index.name = "Date"

# ── Run the analysis pipeline ─────────────────────────────────────────────
stock_info = StockInfo(
    symbol="TSLA",
    name="Tesla, Inc.",
    sector="Consumer Cyclical",
    industry="Auto Manufacturers",
    market_cap=1_120_000_000_000,
    currency="USD",
)

analyzer = TechnicalAnalyzer(df)
technical_summary = analyzer.summarize()
enriched_df = analyzer.get_dataframe()

predictor = StockPredictor(enriched_df)
prediction = predictor.predict()

report = AnalysisReport(
    stock_info=stock_info,
    period="1mo",
    data_points=len(df),
    technical_summary=technical_summary,
    prediction=prediction,
)

print(StockPredictionAgent.format_report(report))

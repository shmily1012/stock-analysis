"""Demo: run full reports for multiple stocks with sample data."""

import numpy as np
import pandas as pd

from stock_prediction.analysis import TechnicalAnalyzer
from stock_prediction.prediction import StockPredictor
from stock_prediction.agent.stock_agent import StockPredictionAgent, AnalysisReport
from stock_prediction.fetcher.data_fetcher import StockInfo

# ── Stock configurations with realistic price ranges ──────────────────────
STOCKS = [
    StockInfo("MSFT", "Microsoft Corporation", "Technology", "Software—Infrastructure", 3_100_000_000_000, "USD"),
    StockInfo("GOOGL", "Alphabet Inc.", "Communication Services", "Internet Content & Information", 2_200_000_000_000, "USD"),
    StockInfo("META", "Meta Platforms, Inc.", "Communication Services", "Internet Content & Information", 1_700_000_000_000, "USD"),
    StockInfo("SNDK", "SanDisk Corporation (WDC)", "Technology", "Data Storage", 23_000_000_000, "USD"),
    StockInfo("MU", "Micron Technology, Inc.", "Technology", "Semiconductors", 110_000_000_000, "USD"),
]

# Approximate recent prices and volatility per stock
STOCK_PARAMS = {
    "MSFT":  {"base": 420.0, "vol": 0.015, "drift": 0.0005,  "avg_volume": 25_000_000},
    "GOOGL": {"base": 185.0, "vol": 0.018, "drift": 0.0003,  "avg_volume": 30_000_000},
    "META":  {"base": 620.0, "vol": 0.022, "drift": 0.0008,  "avg_volume": 18_000_000},
    "SNDK":  {"base": 52.0,  "vol": 0.028, "drift": -0.0002, "avg_volume": 8_000_000},
    "MU":    {"base": 98.0,  "vol": 0.030, "drift": 0.0004,  "avg_volume": 22_000_000},
}


def generate_stock_data(symbol: str, seed: int) -> pd.DataFrame:
    """Generate realistic OHLCV data for a stock."""
    np.random.seed(seed)
    params = STOCK_PARAMS[symbol]
    dates = pd.bdate_range(start="2025-11-15", end="2026-02-14")
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


def run_report(stock_info: StockInfo, seed: int) -> None:
    """Run full analysis pipeline and print report for one stock."""
    df = generate_stock_data(stock_info.symbol, seed)

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
        history=df,
    )

    print(StockPredictionAgent.format_report(report))
    print("\n")


# ── Run all reports ───────────────────────────────────────────────────────
for i, stock in enumerate(STOCKS):
    run_report(stock, seed=100 + i)

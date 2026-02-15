"""MCP Server for the Stock Prediction Agent.

Exposes stock analysis skills as MCP tools that Claude Code can invoke directly.

Tools:
    - stock_fetch    : Fetch historical OHLCV data
    - stock_analyze  : Run technical analysis with signals
    - stock_predict  : Generate price predictions
    - stock_report   : Full pipeline → comprehensive report

Run with:
    python -m stock_prediction.mcp_server
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from stock_prediction.agent.skills import (
    AnalyzeSkill,
    FetchSkill,
    PredictSkill,
    ReportSkill,
)

# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------

mcp = FastMCP(
    "Stock Prediction Agent",
    instructions=(
        "An MCP server that provides stock analysis and prediction tools. "
        "Fetch historical data, run technical analysis, and predict future price trends."
    ),
)

# Instantiate skills
_fetch_skill = FetchSkill()
_analyze_skill = AnalyzeSkill()
_predict_skill = PredictSkill()
_report_skill = ReportSkill()


# ---------------------------------------------------------------------------
# Tool: stock_fetch
# ---------------------------------------------------------------------------


@mcp.tool()
def stock_fetch(symbol: str, period: str = "6mo") -> str:
    """Fetch historical stock price data (Open/High/Low/Close/Volume).

    Args:
        symbol: Stock ticker symbol, e.g. "AAPL", "TSLA", "MSFT", "600519.SS"
        period: Time period - "1mo" (1 month), "6mo" (6 months), "1y" (1 year)

    Returns:
        Formatted table of historical OHLCV data with summary statistics.
    """
    return _fetch_skill.execute(symbol, period)


# ---------------------------------------------------------------------------
# Tool: stock_analyze
# ---------------------------------------------------------------------------


@mcp.tool()
def stock_analyze(symbol: str, period: str = "6mo") -> str:
    """Run technical analysis on a stock.

    Computes indicators: SMA(20/50), EMA(12/26), RSI(14), MACD(12/26/9),
    Bollinger Bands(20,2σ), ATR(14). Also generates trading signals
    (trend direction, overbought/oversold, MACD crossover, etc.).

    Args:
        symbol: Stock ticker symbol, e.g. "AAPL", "TSLA", "MSFT", "600519.SS"
        period: Time period - "1mo" (1 month), "6mo" (6 months), "1y" (1 year)

    Returns:
        Technical indicator values and trading signal interpretations.
    """
    return _analyze_skill.execute(symbol, period)


# ---------------------------------------------------------------------------
# Tool: stock_predict
# ---------------------------------------------------------------------------


@mcp.tool()
def stock_predict(symbol: str, period: str = "6mo") -> str:
    """Predict future stock price movement (5-day and 20-day forecasts).

    Uses Linear Regression trained on technical features to predict future
    returns. Combines model predictions with signal-based analysis for
    direction (bullish/bearish/neutral) and confidence estimation.

    Args:
        symbol: Stock ticker symbol, e.g. "AAPL", "TSLA", "MSFT", "600519.SS"
        period: Historical data period for training - "1mo", "6mo", "1y"

    Returns:
        Price predictions, direction, confidence, support/resistance levels.
        NOTE: For educational purposes only, NOT financial advice.
    """
    return _predict_skill.execute(symbol, period)


# ---------------------------------------------------------------------------
# Tool: stock_report
# ---------------------------------------------------------------------------


@mcp.tool()
def stock_report(symbol: str, period: str = "6mo", output_format: str = "text") -> str:
    """Generate a comprehensive stock analysis report.

    Runs the full pipeline: fetch historical data → compute technical
    indicators → generate predictions → format into a complete report.

    Args:
        symbol: Stock ticker symbol, e.g. "AAPL", "TSLA", "MSFT", "600519.SS"
        period: Time period - "1mo" (1 month), "6mo" (6 months), "1y" (1 year)
        output_format: "text" for human-readable report, "json" for structured data

    Returns:
        Complete analysis report with stock info, technicals, signals, and predictions.
    """
    return _report_skill.execute(symbol, period, format=output_format)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run()

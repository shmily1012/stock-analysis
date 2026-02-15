Generate a comprehensive stock analysis report combining data fetching, technical analysis, and price prediction into one complete report.

Usage: /project:stock-report <SYMBOL> [PERIOD] [--format json]

Arguments:
- SYMBOL: Stock ticker symbol (e.g. AAPL, TSLA, MSFT, 600519.SS)
- PERIOD: Time period — 1mo (1 month), 6mo (6 months), 1y (1 year). Default: 6mo
- --format: Output as "text" (default) or "json"

Execute the following command and display the full output to the user:

```
cd /home/user/stock-analysis && python -m stock_prediction.cli report $ARGUMENTS
```

This is the most comprehensive skill — it runs the full pipeline:
1. Fetches historical data
2. Computes all technical indicators and signals
3. Runs the prediction model
4. Formats everything into a structured report

After displaying the report, offer to:
- Analyze a different time period for comparison
- Look at a related stock in the same sector
- Explain any specific indicator in more detail

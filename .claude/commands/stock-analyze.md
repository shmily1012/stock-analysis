Run technical analysis on a stock, computing indicators (SMA, EMA, RSI, MACD, Bollinger Bands, ATR) and generating trading signals.

Usage: /project:stock-analyze <SYMBOL> [PERIOD]

Arguments:
- SYMBOL: Stock ticker symbol (e.g. AAPL, TSLA, MSFT, 600519.SS)
- PERIOD: Time period — 1mo (1 month), 6mo (6 months), 1y (1 year). Default: 6mo

Execute the following command and display the output to the user:

```
cd /home/user/stock-analysis && python -m stock_prediction.cli analyze $ARGUMENTS
```

After showing the raw output, provide a brief human-readable interpretation of the key signals:
- Is the stock in an uptrend, downtrend, or sideways?
- What does the RSI indicate?
- What does the MACD crossover suggest?
- How is price positioned relative to Bollinger Bands?

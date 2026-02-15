Fetch historical stock data for the given ticker symbol and display OHLCV data with summary statistics.

Usage: /project:stock-fetch <SYMBOL> [PERIOD]

Arguments:
- SYMBOL: Stock ticker symbol (e.g. AAPL, TSLA, MSFT, 600519.SS)
- PERIOD: Time period — 1mo (1 month), 6mo (6 months), 1y (1 year). Default: 6mo

Execute the following command and display the output to the user:

```
cd /home/user/stock-analysis && python -m stock_prediction.cli fetch $ARGUMENTS
```

If dependencies are not installed, first run:
```
pip install -r /home/user/stock-analysis/requirements.txt
```

Generate stock price predictions (5-day and 20-day) using technical analysis features and machine learning.

Usage: /project:stock-predict <SYMBOL> [PERIOD]

Arguments:
- SYMBOL: Stock ticker symbol (e.g. AAPL, TSLA, MSFT, 600519.SS)
- PERIOD: Time period for training data — 1mo (1 month), 6mo (6 months), 1y (1 year). Default: 6mo

Execute the following command and display the output to the user:

```
cd /home/user/stock-analysis && python -m stock_prediction.cli predict $ARGUMENTS
```

After displaying the prediction output, provide additional context:
1. Explain what the prediction direction and confidence mean
2. Mention support/resistance levels and their significance
3. Always include the disclaimer that this is for educational purposes only and not financial advice

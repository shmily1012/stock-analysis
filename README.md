# Stock Prediction Agent (MCP)

A Python-based stock prediction agent built on the **MCP (Model Context Protocol)** model. Provides multiple skills for fetching historical data, performing technical analysis, and predicting future price movements. Designed to work as a Claude Code MCP server and as a standalone CLI tool.

## Architecture

**One agent, multiple skills** — implemented as an MCP server:

```
                    ┌─────────────────────────┐
                    │     MCP Server          │
                    │  (stock_prediction/     │
                    │   mcp_server.py)        │
                    └─────────┬───────────────┘
                              │
              ┌───────────────┼───────────────┐
              │               │               │
        ┌─────┴─────┐  ┌─────┴─────┐  ┌─────┴─────┐
        │  Fetcher   │  │  Analysis │  │ Prediction │
        │ (yfinance) │  │(technicals)│ │   (ML)     │
        └───────────┘  └───────────┘  └───────────┘
```

## MCP Tools

| Tool | Description |
|------|-------------|
| `stock_fetch` | Fetch historical OHLCV data from Yahoo Finance |
| `stock_analyze` | Compute SMA, EMA, RSI, MACD, Bollinger Bands, ATR + signals |
| `stock_predict` | Generate 5-day and 20-day price predictions using ML |
| `stock_report` | Full pipeline → comprehensive formatted report |

### Parameters

- **symbol** (required): Stock ticker — `AAPL`, `TSLA`, `MSFT`, `GOOGL`, `600519.SS`, etc.
- **period** (optional): `1mo` / `6mo` (default) / `1y`

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Use as MCP Server (with Claude Code)

The `.mcp.json` at project root auto-configures the server for Claude Code:

```json
{
  "mcpServers": {
    "stock-prediction": {
      "command": "python",
      "args": ["-m", "stock_prediction.mcp_server"],
      "cwd": "/home/user/stock-analysis"
    }
  }
}
```

Once configured, Claude Code can directly call tools like `stock_fetch("AAPL", "6mo")`.

### 3. Use as CLI

```bash
# List available skills
python -m stock_prediction.cli skills

# Fetch stock data
python -m stock_prediction.cli fetch AAPL --period 6mo

# Run technical analysis
python -m stock_prediction.cli analyze TSLA --period 1y

# Get price prediction
python -m stock_prediction.cli predict MSFT --period 6mo

# Generate full report
python -m stock_prediction.cli report AAPL --period 1y

# JSON output
python -m stock_prediction.cli report AAPL --period 6mo --format json
```

### 4. Use as Python API

```python
from stock_prediction.agent import StockPredictionAgent

agent = StockPredictionAgent("AAPL")
report = agent.analyze("6mo")
print(agent.format_report(report))
```

## Claude Code Slash Commands

Also available in `.claude/commands/`:

| Command | Description |
|---------|-------------|
| `/project:stock-fetch <SYMBOL> [PERIOD]` | Fetch historical data |
| `/project:stock-analyze <SYMBOL> [PERIOD]` | Technical analysis |
| `/project:stock-predict <SYMBOL> [PERIOD]` | Price prediction |
| `/project:stock-report <SYMBOL> [PERIOD]` | Full report |

## Project Structure

```
stock-analysis/
├── .mcp.json                       # MCP server config for Claude Code
├── .claude/commands/               # Claude Code slash commands
│   ├── stock-fetch.md
│   ├── stock-analyze.md
│   ├── stock-predict.md
│   └── stock-report.md
├── CLAUDE.md                       # Project context for Claude Code
├── requirements.txt
├── pyproject.toml
└── stock_prediction/
    ├── __init__.py
    ├── cli.py                      # Standalone CLI
    ├── mcp_server.py               # MCP server entry point
    ├── fetcher/
    │   └── data_fetcher.py         # Yahoo Finance data fetcher
    ├── analysis/
    │   └── technical.py            # Technical indicator calculations
    ├── prediction/
    │   └── model.py                # ML prediction model
    └── agent/
        ├── stock_agent.py          # Agent orchestrator
        └── skills.py               # Skill definitions
```

## Disclaimer

This tool is for **educational and informational purposes only**. It does NOT constitute financial advice. Past performance does not guarantee future results. Always do your own research before making investment decisions.

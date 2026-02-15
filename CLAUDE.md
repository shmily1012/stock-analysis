# Stock Prediction Agent

This project is a stock prediction agent built on the **MCP (Model Context Protocol)** model, designed to be run by Claude Code as an MCP server.

## Architecture

The agent follows the MCP pattern: **one agent, multiple skills (tools)**.

```
stock_prediction/
├── fetcher/          # Data fetching from Yahoo Finance (yfinance)
├── analysis/         # Technical analysis (SMA, EMA, RSI, MACD, BB, ATR)
├── prediction/       # Price prediction (Linear Regression + technical features)
├── agent/
│   ├── stock_agent.py   # Agent orchestrator
│   └── skills.py        # Skill definitions
├── mcp_server.py     # MCP server — exposes skills as MCP tools
└── cli.py            # CLI entry point (standalone use)
```

## MCP Tools (Skills)

The MCP server exposes 4 tools:

| MCP Tool | Description |
|----------|-------------|
| `stock_fetch` | Fetch historical OHLCV data for a stock |
| `stock_analyze` | Run technical analysis with indicator signals |
| `stock_predict` | Generate 5-day and 20-day price predictions |
| `stock_report` | Full pipeline: fetch + analyze + predict → report |

### Tool Parameters

All tools accept:
- **symbol** (required): Stock ticker (e.g. `AAPL`, `TSLA`, `GOOGL`, `600519.SS` for Chinese A-shares)
- **period** (optional): `1mo` (1 month), `6mo` (6 months, default), `1y` (1 year)

`stock_report` also accepts:
- **output_format** (optional): `text` (default) or `json`

## MCP Configuration

The `.mcp.json` at project root configures the MCP server for Claude Code:

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

## Claude Code Slash Commands

Slash commands are also available in `.claude/commands/`:

- `/project:stock-fetch` — Fetch data
- `/project:stock-analyze` — Technical analysis
- `/project:stock-predict` — Price prediction
- `/project:stock-report` — Full report

## Setup

```bash
pip install -r requirements.txt
```

## Standalone CLI Usage

```bash
python -m stock_prediction.cli fetch AAPL --period 6mo
python -m stock_prediction.cli analyze AAPL --period 1y
python -m stock_prediction.cli predict AAPL --period 6mo
python -m stock_prediction.cli report AAPL --period 6mo --format json
python -m stock_prediction.cli skills
```

## Technical Indicators

- **SMA** (20, 50): Simple Moving Average
- **EMA** (12, 26): Exponential Moving Average
- **RSI** (14): Relative Strength Index
- **MACD**: Moving Average Convergence Divergence (12/26/9)
- **Bollinger Bands** (20, 2σ): Upper / Middle / Lower
- **ATR** (14): Average True Range

## Disclaimer

All predictions are for **educational and informational purposes only**. NOT financial advice.

"""CLI entry point for the stock prediction agent.

Usage:
    stock-predict fetch  AAPL --period 6mo
    stock-predict analyze AAPL --period 1y
    stock-predict predict AAPL --period 6mo
    stock-predict report  AAPL --period 6mo [--format json]
    stock-predict skills
"""

from __future__ import annotations

import argparse
import sys

from stock_prediction.agent.skills import get_skill, list_skills


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stock-predict",
        description="Stock Prediction Agent — fetch, analyze, and predict stock trends",
    )
    sub = parser.add_subparsers(dest="command", help="Available skills / commands")

    # -- skills (list available) ------------------------------------------------
    sub.add_parser("skills", help="List all available skills")

    # -- fetch ------------------------------------------------------------------
    p_fetch = sub.add_parser("fetch", help="Fetch historical stock data")
    p_fetch.add_argument("symbol", help="Stock ticker symbol (e.g. AAPL, TSLA)")
    p_fetch.add_argument(
        "--period",
        default="6mo",
        help="Time period: 1mo, 6mo, 1y (default: 6mo)",
    )

    # -- analyze ----------------------------------------------------------------
    p_analyze = sub.add_parser("analyze", help="Run technical analysis")
    p_analyze.add_argument("symbol", help="Stock ticker symbol")
    p_analyze.add_argument("--period", default="6mo")

    # -- predict ----------------------------------------------------------------
    p_predict = sub.add_parser("predict", help="Predict future price movement")
    p_predict.add_argument("symbol", help="Stock ticker symbol")
    p_predict.add_argument("--period", default="6mo")

    # -- report -----------------------------------------------------------------
    p_report = sub.add_parser("report", help="Full analysis report")
    p_report.add_argument("symbol", help="Stock ticker symbol")
    p_report.add_argument("--period", default="6mo")
    p_report.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)",
    )

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "skills":
        print("Available skills:\n")
        for skill in list_skills():
            print(f"  {skill['name']:10s}  {skill['description']}")
        return

    try:
        skill = get_skill(args.command)
        kwargs = {}
        if args.command == "report" and hasattr(args, "format"):
            kwargs["format"] = args.format
        result = skill.execute(args.symbol, args.period, **kwargs)
        print(result)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

"""Stock data fetcher using yfinance."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pandas as pd
import yfinance as yf


class Period(Enum):
    """Supported time periods for historical data."""

    ONE_MONTH = "1mo"
    SIX_MONTHS = "6mo"
    ONE_YEAR = "1y"

    @classmethod
    def from_str(cls, label: str) -> "Period":
        mapping = {
            "1m": cls.ONE_MONTH,
            "1mo": cls.ONE_MONTH,
            "6m": cls.SIX_MONTHS,
            "6mo": cls.SIX_MONTHS,
            "1y": cls.ONE_YEAR,
            "1yr": cls.ONE_YEAR,
        }
        key = label.lower().strip()
        if key not in mapping:
            raise ValueError(
                f"Unsupported period '{label}'. Use: {list(mapping.keys())}"
            )
        return mapping[key]


@dataclass
class StockInfo:
    """Basic information about a stock."""

    symbol: str
    name: str
    sector: str
    industry: str
    market_cap: float | None
    currency: str


class StockDataFetcher:
    """Fetch historical stock data and company info from Yahoo Finance."""

    def __init__(self, symbol: str) -> None:
        self.symbol = symbol.upper()
        self._ticker = yf.Ticker(self.symbol)

    def get_info(self) -> StockInfo:
        """Return basic stock information."""
        info = self._ticker.info
        return StockInfo(
            symbol=self.symbol,
            name=info.get("longName", info.get("shortName", self.symbol)),
            sector=info.get("sector", "N/A"),
            industry=info.get("industry", "N/A"),
            market_cap=info.get("marketCap"),
            currency=info.get("currency", "USD"),
        )

    def get_history(self, period: Period) -> pd.DataFrame:
        """Fetch historical OHLCV data for the given period.

        Returns a DataFrame with columns:
            Open, High, Low, Close, Volume
        indexed by Date.
        """
        df = self._ticker.history(period=period.value)
        if df.empty:
            raise ValueError(
                f"No data returned for {self.symbol} over period {period.value}. "
                "Check that the ticker symbol is valid."
            )
        # Keep only the core columns
        cols = ["Open", "High", "Low", "Close", "Volume"]
        return df[[c for c in cols if c in df.columns]]

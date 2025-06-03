from datetime import date, datetime, timedelta
from typing import Any, Literal, overload

from .c_market_data import MarketData


class BarData(MarketData):
    """
    Candlestick bar data representing price movement over a time period.

    Note: The timestamp represents the END of the candlestick period to prevent
    any possibility of future data contamination in quantitative analysis.

    Unlike other market data class, updating the data of the fields in buffer is possible with the __setitem__ method.
    """

    def __init__(
            self,
            ticker: str,
            timestamp: float,
            high_price: float,
            low_price: float,
            open_price: float,
            close_price: float,
            volume: float = 0.0,
            notional: float = 0.0,
            trade_count: int = 0,
            start_timestamp: float = ...,
            bar_span: timedelta | float | None = None,
            **kwargs: Any
    ) -> None:
        """
        Initialize a candlestick bar.

        Args:
            ticker: The ticker of the candlestick
            timestamp: End time of bar (Unix timestamp)
            high_price: Highest price during period
            low_price: Lowest price during period
            open_price: Opening price
            close_price: Closing price
            volume: Total traded volume
            notional: Total traded notional value
            trade_count: Number of trades
            start_timestamp: Start time of bar (Unix timestamp)
            bar_span: Duration of bar as timedelta or seconds
            **kwargs: Additional bar attributes
        """
        ...

    @overload
    def __getitem__(self, key: Literal["high_price", "low_price", "open_price", "close_price", "volume", "notional"]) -> float: ...

    @overload
    def __getitem__(self, key: Literal["trade_count"]) -> int: ...

    def __getitem__(self, key: str) -> float | int:
        """Access bar properties using dictionary-style syntax."""
        ...

    @overload
    def __setitem__(self, key: Literal["high_price", "low_price", "open_price", "close_price", "volume", "notional"], value: float) -> None: ...

    @overload
    def __setitem__(self, key: Literal["trade_count"], value: int) -> None: ...

    def __setitem__(self, key: str, value: float | int) -> None:
        """Set bar properties using dictionary-style syntax."""
        ...

    @classmethod
    def from_bytes(cls, data: bytes) -> BarData:
        """Reconstruct bar from serialized byte data."""
        ...

    def __copy__(self) -> BarData:
        """Create a deep copy of this bar."""
        ...

    @property
    def high_price(self) -> float:
        """Get the highest price during the bar period."""
        ...

    @property
    def low_price(self) -> float:
        """Get the lowest price during the bar period."""
        ...

    @property
    def open_price(self) -> float:
        """Get the opening price of the bar period."""
        ...

    @property
    def close_price(self) -> float:
        """Get the closing price of the bar period."""
        ...

    @property
    def volume(self) -> float:
        """Get the total traded volume during the bar period."""
        ...

    @property
    def notional(self) -> float:
        """Get the total traded notional value during the bar period."""
        ...

    @property
    def trade_count(self) -> int:
        """Get the number of trades during the bar period."""
        ...

    @property
    def start_timestamp(self) -> float:
        """Get the start time of the bar period as Unix timestamp."""
        ...

    @property
    def bar_span_seconds(self) -> float:
        """Get the bar duration in seconds."""
        ...

    @property
    def bar_span(self) -> timedelta:
        """Get the bar duration as timedelta."""
        ...

    @property
    def vwap(self) -> float:
        """Calculate the Volume Weighted Average Price."""
        ...

    @property
    def bar_type(self) -> Literal['Hourly-Plus', 'Hourly', 'Minute-Plus', 'Minute', 'Sub-Minute']:
        """Get the classification of this bar's time period."""
        ...

    @property
    def bar_end_time(self) -> datetime:
        """Get the end time of the bar as a datetime object."""
        ...

    @property
    def bar_start_time(self) -> datetime:
        """Get the start time of the bar as a datetime object."""
        ...


class DailyBar(BarData):
    """
    Specialized candlestick bar representing a full trading day.

    Note: The timestamp field is repurposed to store market date information.
    This is transparent when using the Python interface but requires conversion
    when accessing the underlying buffer directly via Cython or ctypes.
    """

    def __init__(
            self,
            ticker: str,
            market_date: date,
            high_price: float,
            low_price: float,
            open_price: float,
            close_price: float,
            volume: float = 0.0,
            notional: float = 0.0,
            trade_count: int = 0,
            bar_span: int = 1,
            **kwargs: Any
    ) -> None:
        """
        Initialize a daily bar.

        Args:
            ticker: The ticker of the candlestick
            market_date: Trading date
            high_price: Daily high price
            low_price: Daily low price
            open_price: Daily open price
            close_price: Daily close price
            volume: Daily traded volume
            notional: Daily traded notional value
            trade_count: Number of trades
            bar_span: Number of days represented (default 1)
            **kwargs: Additional bar attributes
        """
        ...

    @property
    def market_date(self) -> date:
        """Get the trading date represented by this bar."""
        ...

    @property
    def market_time(self) -> date:
        """Alias for market_date (for interface compatibility)."""
        ...

    @property
    def bar_start_time(self) -> date:
        """Get the start date of the bar period."""
        ...

    @property
    def bar_end_time(self) -> date:
        """Get the end date of the bar period."""
        ...

    @property
    def bar_span(self) -> timedelta:
        """Get the duration of the bar period as timedelta."""
        ...

    @property
    def bar_type(self) -> Literal['Daily', 'Daily-Plus']:
        """Get the classification of this daily bar."""
        ...

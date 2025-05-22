from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any, Dict, Literal

from .market_data import MarketData


class BarData(MarketData):
    _additional: Dict[str, Any]

    def __init__(
            self,
            ticker: str,
            timestamp: float,
            high_price: float,
            low_price: float,
            open_price: float,
            close_price: float,
            volume: float = ...,
            notional: float = ...,
            trade_count: int = ...,
            start_timestamp: float = ...,
            bar_span: timedelta | float | None = ...,
            **kwargs: Any
    ) -> None: ...

    def __setitem__(self, key: str, value: float): ...

    def __getitem__(self, key: str) -> float | int: ...

    @classmethod
    def from_buffer(cls, buffer: bytes | memoryview) -> BarData: ...

    @classmethod
    def from_bytes(cls, data: bytes) -> BarData: ...

    def __copy__(self) -> BarData: ...

    @property
    def high_price(self) -> float: ...

    @property
    def low_price(self) -> float: ...

    @property
    def open_price(self) -> float: ...

    @property
    def close_price(self) -> float: ...

    @property
    def volume(self) -> float: ...

    @property
    def notional(self) -> float: ...

    @property
    def trade_count(self) -> int: ...

    @property
    def start_timestamp(self) -> float: ...

    @property
    def bar_span_seconds(self) -> float: ...

    @property
    def bar_span(self) -> timedelta: ...

    @property
    def vwap(self) -> float: ...

    @property
    def bar_type(self) -> Literal['Hourly-Plus', 'Hourly', 'Minute-Plus', 'Minute', 'Sub-Minute']: ...

    @property
    def bar_end_time(self) -> datetime: ...

    @property
    def bar_start_time(self) -> datetime: ...


class DailyBar(BarData):
    def __init__(
            self,
            ticker: str,
            market_date: date,
            high_price: float,
            low_price: float,
            open_price: float,
            close_price: float,
            volume: float = ...,
            notional: float = ...,
            trade_count: int = ...,
            bar_span: int = ...,
            **kwargs: Any
    ) -> None: ...

    @property
    def market_date(self) -> date: ...

    @property
    def market_time(self) -> date: ...

    @property
    def bar_start_time(self) -> date: ...

    @property
    def bar_end_time(self) -> date: ...

    @property
    def bar_span(self) -> timedelta: ...

    @property
    def bar_type(self) -> Literal['Daily', 'Daily-Plus']: ...

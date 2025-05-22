import enum
from datetime import datetime
from typing import Any


class DataType(enum.IntEnum):
    DTYPE_UNKNOWN: int
    DTYPE_MARKET_DATA: int
    DTYPE_TRANSACTION: int
    DTYPE_ORDER: int
    DTYPE_TICK_LITE: int
    DTYPE_TICK: int
    DTYPE_BAR: int


class MarketData:
    _additional: dict[str, Any]
    _dtype: int

    def __init__(self, ticker: str, timestamp: float, **kwargs: Any) -> None: ...

    def update(self, name: str, value: Any) -> None:
        ...

    @classmethod
    def buffer_size(cls) -> int: ...

    @classmethod
    def from_buffer(cls, buffer: bytes) -> 'MarketData': ...

    @classmethod
    def from_bytes(cls, data: bytes) -> 'MarketData': ...

    def __copy__(self) -> 'MarketData': ...

    def to_bytes(self) -> bytes: ...

    @property
    def ticker(self) -> str: ...

    @property
    def timestamp(self) -> float: ...

    @property
    def dtype(self) -> int: ...

    @property
    def topic(self) -> str: ...

    @property
    def market_time(self) -> datetime: ...

    @property
    def market_price(self) -> float: ...

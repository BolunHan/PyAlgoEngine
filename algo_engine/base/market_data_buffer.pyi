from __future__ import annotations

from typing import Any

from .candlestick import BarData
from .market_data import MarketData
from .tick import TickData, TickDataLite
from .transaction import TransactionData, OrderData


class MarketDataBuffer:
    def __init__(
            self,
            buffer: Any,
            dtype: int = ...,
            capacity: int = ...
    ) -> None: ...

    @classmethod
    def buffer_size(cls, n_transaction_data: int = 0, n_order_data: int = 0, n_tick_data_lite: int = 0, n_tick_data: int = 0, n_bar_data: int = 0) -> int: ...

    @classmethod
    def from_buffer(cls, buffer: Any) -> MarketDataBuffer: ...

    @classmethod
    def from_bytes(cls, data: bytes, buffer: Any) -> MarketDataBuffer: ...

    def push(self, market_data: MarketData) -> None: ...

    def sort(self, inplace: bool = ...) -> None: ...

    def to_bytes(self) -> bytes: ...

    def update(self, dtype: int, **kwargs: Any) -> None: ...

    def __getitem__(self, idx: int) -> MarketData | TransactionData | OrderData | TickDataLite | TickData | BarData: ...

    def __iter__(self) -> MarketDataBuffer: ...

    def __next__(self) -> MarketData | TransactionData | OrderData | TickDataLite | TickData | BarData: ...

    def __len__(self) -> int: ...


class MarketDataRingBuffer:
    def __init__(
            self,
            buffer: Any,
            dtype: int = ...,
            capacity: int = ...
    ) -> None: ...

    def is_empty(self) -> bool: ...

    def is_full(self) -> bool: ...

    def put(self, market_data: MarketData) -> None: ...

    def get(self, idx: int) -> MarketData | TransactionData | OrderData | TickDataLite | TickData | BarData: ...

    def listen(self) -> MarketData | TransactionData | OrderData | TickDataLite | TickData | BarData: ...

    @property
    def head(self) -> int: ...

    @property
    def tail(self) -> int: ...

    @property
    def count(self) -> int: ...


class MarketDataConcurrentBuffer:
    def __init__(
            self,
            buffer: Any,
            n_workers: int,
            dtype: int = ...,
            capacity: int = ...
    ) -> None: ...

    def get_head(self, worker_id: int) -> int: ...

    def min_head(self) -> int: ...

    def is_empty(self, worker_id: int) -> bool: ...

    def is_empty_all(self) -> bool: ...

    def is_full(self) -> bool: ...

    def put(self, market_data: MarketData) -> None: ...

    def get(self, idx: int) -> MarketData | TransactionData | OrderData | TickDataLite | TickData | BarData: ...

    def listen(self, worker_id: int, block: bool = True, timeout: float = ...) -> MarketData | TransactionData | OrderData | TickDataLite | TickData | BarData: ...

    @property
    def head(self) -> list[int]: ...

    @property
    def tail(self) -> int: ...

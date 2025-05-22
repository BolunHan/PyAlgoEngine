from collections.abc import Sequence
from typing import Any

from .market_data import MarketData


class TickDataLite(MarketData):
    def __init__(
            self,
            ticker: str,
            timestamp: float,
            last_price: float,
            bid_price: float,
            bid_volume: float,
            ask_price: float,
            ask_volume: float,
            total_traded_volume: float = ...,
            total_traded_notional: float = ...,
            total_trade_count: int = ...,
            **kwargs: Any
    ) -> None: ...

    @property
    def last_price(self) -> float: ...

    @property
    def bid_price(self) -> float: ...

    @property
    def bid_volume(self) -> float: ...

    @property
    def ask_price(self) -> float: ...

    @property
    def ask_volume(self) -> float: ...

    @property
    def total_traded_volume(self) -> float: ...

    @property
    def total_traded_notional(self) -> float: ...

    @property
    def total_trade_count(self) -> int: ...

    @property
    def mid_price(self) -> float: ...

    @property
    def spread(self) -> float: ...

    @property
    def market_price(self) -> float: ...


class OrderBook:
    side: int
    sorted: bool

    def __init__(
            self,
            side: int | None = ...,
            price: Sequence[float] | None = ...,
            volume: Sequence[float] | None = ...,
            n_orders: Sequence[int] | None = ...,
            is_sorted: bool = ...
    ) -> None: ...

    def __iter__(self) -> OrderBook: ...

    def __next__(self) -> tuple[float, float, int]: ...

    def at_price(self, price: float) -> tuple[float, float, float]: ...

    def at_level(self, index: int) -> tuple[float, float, float]: ...

    def loc_volume(self, p0: float, p1: float) -> float: ...

    def sort(self) -> None: ...

    def to_bytes(self) -> bytes: ...

    @property
    def price(self) -> list[float]: ...

    @property
    def volume(self) -> list[float]: ...

    @property
    def n_orders(self) -> list[int]: ...


class TickData(TickDataLite):
    def __init__(
            self,
            ticker: str,
            timestamp: float,
            last_price: float,
            total_traded_volume: float = ...,
            total_traded_notional: float = ...,
            total_trade_count: int = ...,
            **kwargs: Any
    ) -> None: ...

    def parse(self, kwargs: dict[str, Any]) -> None: ...

    @property
    def bid(self) -> OrderBook: ...

    @property
    def ask(self) -> OrderBook: ...

    @property
    def best_ask_price(self) -> float: ...

    @property
    def best_bid_price(self) -> float: ...

    @property
    def best_ask_volume(self) -> float: ...

    @property
    def best_bid_volume(self) -> float: ...

    @property
    def lite(self) -> TickDataLite: ...

from __future__ import annotations

import enum
import uuid
from typing import Any

from .market_data import MarketData


class OrderType(enum.IntEnum):
    ORDER_UNKNOWN: int
    ORDER_CANCEL: int
    ORDER_GENERIC: int
    ORDER_LIMIT: int
    ORDER_LIMIT_MAKER: int
    ORDER_MARKET: int
    ORDER_FOK: int
    ORDER_FAK: int
    ORDER_IOC: int


class TransactionDirection(enum.IntEnum): ...


class TransactionOffset(enum.IntEnum): ...


class TransactionSide(enum.IntEnum): ...


class TransactionData(MarketData):
    def __init__(
            self,
            ticker: str,
            timestamp: float,
            price: float,
            volume: float,
            side: int,
            multiplier: float = ...,
            notional: float = ...,
            transaction_id: str | int | bytes | uuid.UUID | None = ...,
            buy_id: str | int | bytes | uuid.UUID | None = ...,
            sell_id: str | int | bytes | uuid.UUID | None = ...,
            **kwargs: Any
    ) -> None: ...

    @classmethod
    def from_buffer(cls, buffer: bytes) -> TransactionData: ...

    @classmethod
    def from_bytes(cls, data: bytes) -> TransactionData: ...

    @classmethod
    def merge(cls, data_list: list[TransactionData]) -> TransactionData: ...

    def __copy__(self) -> TransactionData: ...

    @property
    def price(self) -> float: ...

    @property
    def volume(self) -> float: ...

    @property
    def side_int(self) -> int: ...

    @property
    def side(self) -> TransactionSide: ...

    @property
    def multiplier(self) -> float: ...

    @property
    def notional(self) -> float: ...

    @property
    def transaction_id(self) -> str | int | bytes | uuid.UUID | None: ...

    @property
    def buy_id(self) -> str | int | bytes | uuid.UUID | None: ...

    @property
    def sell_id(self) -> str | int | bytes | uuid.UUID | None: ...

    @property
    def market_price(self) -> float: ...

    @property
    def volume_flow(self) -> float: ...

    @property
    def notional_flow(self) -> float: ...


class OrderData(MarketData):
    def __init__(
            self,
            ticker: str,
            timestamp: float,
            price: float,
            volume: float,
            side: int,
            order_id: str | int | bytes | uuid.UUID | None = ...,
            order_type: int = ...,
            **kwargs: Any
    ) -> None: ...

    @property
    def price(self) -> float: ...

    @property
    def volume(self) -> float: ...

    @property
    def side(self) -> int: ...

    @property
    def order_id(self) -> str | int | bytes | uuid.UUID | None: ...

    @property
    def order_type_int(self) -> int: ...

    @property
    def order_type(self) -> OrderType: ...

    @property
    def market_price(self) -> float: ...

    @property
    def flow(self) -> float: ...


class TradeData(TransactionData):
    def __init__(
            self,
            ticker: str,
            timestamp: float,
            trade_price: float,
            trade_volume: float,
            side: int,
            order_id: str | int | bytes | uuid.UUID | None = ...,
            order_type: int = ...,
            **kwargs: Any
    ) -> None: ...

    @property
    def trade_price(self) -> float: ...

    @property
    def trade_volume(self) -> float: ...

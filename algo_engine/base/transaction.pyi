from __future__ import annotations

import enum
import uuid
from typing import Any, Literal

from typing_extensions import deprecated

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


class TransactionDirection(enum.IntEnum):
    DIRECTION_UNKNOWN: int
    DIRECTION_SHORT: int
    DIRECTION_LONG: int

    def __or__(self, offset: TransactionOffset) -> TransactionSide:
        ...

    @property
    def sign(self) -> Literal[-1, 0, 1]:
        ...


class TransactionOffset(enum.IntEnum):
    OFFSET_CANCEL: int
    OFFSET_ORDER: int
    OFFSET_OPEN: int
    OFFSET_CLOSE: int

    def __or__(self, direction: TransactionDirection) -> TransactionSide:
        ...


class TransactionSide(enum.IntEnum):
    SIDE_LONG_OPEN: int
    SIDE_LONG_CLOSE: int
    SIDE_LONG_CANCEL: int
    SIDE_SHORT_OPEN: int
    SIDE_SHORT_CLOSE: int
    SIDE_SHORT_CANCEL: int
    SIDE_BID: int
    SIDE_ASK: int
    SIDE_CANCEL: int
    SIDE_UNKNOWN: int
    SIDE_LONG: int
    SIDE_SHORT: int
    ShortOrder: deprecated('Use SIDE_ASK instead')(int)
    AskOrder: deprecated('Use SIDE_ASK instead')(int)
    Ask: deprecated('Use SIDE_ASK instead')(int)
    LongOrder: deprecated('Use SIDE_BID instead')(int)
    BidOrder: deprecated('Use SIDE_BID instead')(int)
    Bid: deprecated('Use SIDE_BID instead')(int)
    ShortFilled: deprecated('Use SIDE_SHORT instead')(int)
    Unwind: deprecated('Use SIDE_SHORT_CLOSE instead')(int)
    Sell: deprecated('Use SIDE_SHORT_CLOSE instead')(int)
    LongFilled: deprecated('Use SIDE_LONG_OPEN instead')(int)
    LongOpen: deprecated('Use SIDE_LONG_OPEN instead')(int)
    Buy: deprecated('Use SIDE_LONG_OPEN instead')(int)
    ShortOpen: deprecated('Use SIDE_SHORT_OPEN instead')(int)
    Short: deprecated('Use SIDE_SHORT_OPEN instead')(int)
    Cover: deprecated('Use SIDE_LONG_CLOSE instead')(int)
    UNKNOWN: deprecated('Use SIDE_UNKNOWN instead')(int)
    CANCEL: deprecated('Use SIDE_CANCEL instead')(int)
    FAULTY: int

    @property
    def sign(self) -> Literal[-1, 0, 1]:
        ...

    @property
    def offset(self) -> TransactionOffset:
        ...

    @property
    def direction(self) -> TransactionDirection:
        ...

    @property
    def side_name(self) -> str:
        ...

    @property
    def offset_name(self) -> str:
        ...

    @property
    def direction_name(self) -> str:
        ...


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
    def side_int(self) -> int: ...

    @property
    def side(self) -> TransactionSide: ...

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

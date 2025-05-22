from __future__ import annotations

import enum
import uuid
from datetime import datetime

from typing_extensions import deprecated

from .market_data import MarketData
from .transaction import OrderType, TransactionSide, TransactionData


class OrderState(enum.IntEnum):
    STATE_UNKNOWN: int
    STATE_REJECTED: int
    STATE_INVALID: int
    STATE_PENDING: int
    STATE_SENT: int
    STATE_PLACED: int
    STATE_PARTFILLED: int
    STATE_FILLED: int
    STATE_CANCELING: int
    STATE_CANCELED: int

    UNKNOWN: deprecated('Use STATE_UNKNOWN instead')(int)
    Rejected: deprecated('Use STATE_REJECTED instead')(int)
    Invalid: deprecated('Use STATE_INVALID instead')(int)
    Pending: deprecated('Use STATE_PENDING instead')(int)
    Sent: deprecated('Use STATE_SENT instead')(int)
    Placed: deprecated('Use STATE_PLACED instead')(int)
    PartFilled: deprecated('Use STATE_PARTFILLED instead')(int)
    Filled: deprecated('Use STATE_FILLED instead')(int)
    Canceling: deprecated('Use STATE_CANCELING instead')(int)
    Canceled: deprecated('Use STATE_CANCELED instead')(int)

    def __hash__(self) -> int: ...

    @property
    def is_working(self) -> bool: ...

    @property
    def is_done(self) -> bool: ...

    @property
    def state_name(self) -> str: ...


class TradeReport(MarketData):
    def __init__(
            self,
            ticker: str,
            timestamp: float,
            price: float,
            volume: float,
            side: int,
            notional: float = ...,
            multiplier: float = ...,
            fee: float = ...,
            order_id: str | int | bytes | uuid.UUID | None = ...,
            trade_id: str | int | bytes | uuid.UUID | None = ...,
            **kwargs
    ) -> None: ...

    def __eq__(self, other: TradeReport) -> bool: ...

    def __repr__(self) -> str: ...

    def reset_order_id(self, order_id: str | int | bytes | uuid.UUID | None = ...) -> TradeReport: ...

    def reset_trade_id(self, trade_id: str | int | bytes | uuid.UUID | None = ...) -> TradeReport: ...

    @classmethod
    def from_buffer(cls, buffer: bytes) -> TradeReport: ...

    @classmethod
    def from_bytes(cls, data: bytes) -> TradeReport: ...

    def copy(self) -> TradeReport: ...

    def to_trade(self) -> TransactionData: ...

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
    def fee(self) -> float: ...

    @property
    def trade_id(self) -> str | int | bytes | uuid.UUID | None: ...

    @property
    def order_id(self) -> str | int | bytes | uuid.UUID | None: ...

    @property
    def market_price(self) -> float: ...

    @property
    def volume_flow(self) -> float: ...

    @property
    def notional_flow(self) -> float: ...

    @property
    def trade_time(self) -> datetime: ...


class TradeInstruction(MarketData):
    _dtype: int
    trades: dict[str | int | bytes | uuid.UUID, TradeReport]

    def __init__(
            self,
            ticker: str,
            timestamp: float,
            side: int,
            volume: float,
            order_type: int = ...,
            limit_price: float = ...,
            multiplier: float = ...,
            order_id: str | int | bytes | uuid.UUID | None = ...,
            **kwargs
    ) -> None: ...

    def __eq__(self, other: TradeInstruction) -> bool: ...

    def __repr__(self) -> str: ...

    @classmethod
    def from_buffer(cls, buffer: bytes) -> TradeInstruction: ...

    @classmethod
    def from_bytes(cls, data: bytes) -> TradeInstruction: ...

    def __copy__(self) -> TradeInstruction: ...

    def reset(self) -> TradeInstruction: ...

    def reset_order_id(self, order_id: str | int | bytes | uuid.UUID | None = ...) -> TradeInstruction: ...

    def set_order_state(self, order_state: int, timestamp: float = ...) -> TradeInstruction: ...

    def fill(self, trade_report: TradeReport) -> TradeInstruction: ...

    def add_trade(self, trade_report: TradeReport) -> TradeInstruction: ...

    def cancel_order(self, timestamp: float = ...) -> TradeInstruction: ...

    def canceled(self, timestamp: float = ...) -> TradeInstruction: ...

    @property
    def is_working(self) -> bool: ...

    @property
    def is_done(self) -> bool: ...

    @property
    def limit_price(self) -> float: ...

    @property
    def volume(self) -> float: ...

    @property
    def side_int(self) -> int: ...

    @property
    def side(self) -> TransactionSide: ...

    @property
    def order_type_int(self) -> int: ...

    @property
    def order_type(self) -> OrderType: ...

    @property
    def order_state_int(self) -> int: ...

    @property
    def order_state(self) -> OrderState: ...

    @property
    def multiplier(self) -> float: ...

    @property
    def filled_volume(self) -> float: ...

    @property
    def working_volume(self) -> float: ...

    @property
    def filled_notional(self) -> float: ...

    @property
    def fee(self) -> float: ...

    @property
    def order_id(self) -> str | int | bytes | uuid.UUID | None: ...

    @property
    def average_price(self) -> float: ...

    @property
    def start_time(self) -> datetime: ...

    @property
    def placed_time(self) -> datetime | None: ...

    @property
    def canceled_time(self) -> datetime | None: ...

    @property
    def finished_time(self) -> datetime | None: ...

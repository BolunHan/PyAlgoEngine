import enum
import uuid
from collections.abc import Iterator
from typing import Any

from algo_engine.base import TradeReport, TransactionSide
from algo_engine.engine.c_dma_algo import OrderMapView


class AlgoStatus(enum.IntEnum):
    """The algo lifecycle states; values mirror the C enum."""

    idle: AlgoStatus
    preparing: AlgoStatus
    ready: AlgoStatus
    working: AlgoStatus
    done: AlgoStatus
    closed: AlgoStatus
    stopping: AlgoStatus
    rejected: AlgoStatus
    error: AlgoStatus

    Idle: AlgoStatus
    Preparing: AlgoStatus
    Ready: AlgoStatus
    Working: AlgoStatus
    Done: AlgoStatus
    Closed: AlgoStatus
    Stopping: AlgoStatus
    Rejected: AlgoStatus
    Error: AlgoStatus


class AlgoTemplate:
    """
    The bound view of a `dma_algo` record.

    The record holds the algo identity, the registration id, the instrument and
    side, the lifecycle status and the volume, notional, fee and cash-flow
    aggregates; the engine owns the record and maintains those aggregates from
    the order and fill feeds. The class is not an abstract algo base and is not
    constructed directly: a kind is built by its concrete wrapper (`Passive`,
    `Aggressive`, ...), whose record embeds `dma_algo` at offset 0, and the
    kind's behavior lives in the record's C callbacks.
    """

    algo_id: uuid.UUID
    """The identity of the record; the engine's algo index key."""

    algo_reg_id: int
    """The id of the registration that built the record."""

    ticker: str
    side: TransactionSide
    status: AlgoStatus

    target_volume: float
    working_volume: float
    exposure_volume: float
    filled_volume: float
    filled_notional: float
    fee: float
    cash_flow: float
    multiplier: float

    n_orders: int
    n_working_orders: int

    is_active: bool
    """True while the record is in the working state."""

    average_price: float
    filled_progress: float
    placed_progress: float

    ts_started: float | None
    """The unix time the record started working, or None when it never did."""

    ts_finished: float | None
    """The unix time the record completed from the working state, or None."""

    orders: OrderMapView
    """The order-id view of every order the algo created."""

    working_orders: OrderMapView
    """The order-id view of the live orders."""

    trades: Iterator[TradeReport]
    """Every applied report across the algo's orders."""

    def __init__(self) -> None:
        """Refuse direct construction; a concrete algo wrapper builds the record."""

    def launch(self, **kwargs: Any) -> None:
        """Invoke the record's launch hook.

        Raises:
            ValueError: When the record carries no launch hook.
            MemoryError: When the hook cannot allocate its order.
        """

    def cancel(self, **kwargs: Any) -> None:
        """Invoke the record's stop hook and cancel every live order.

        Raises:
            ValueError: When the record carries no stop hook.
        """

    def to_json(self, fmt: str = 'str') -> str | dict:
        """Export the algo state and its orders.

        Args:
            fmt (str): ``'dict'`` returns the mapping, anything else its JSON text.
        """

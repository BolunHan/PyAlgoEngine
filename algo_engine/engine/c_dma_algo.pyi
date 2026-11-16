import enum
from typing import Any

from algo_engine.base import MarketData, TradeInstruction, TradeReport
from algo_engine.engine.algo import AlgoTemplate


class AlgoLimitMode(enum.IntEnum):
    """How two limit prices are combined; values mirror the C enum."""

    strict: AlgoLimitMode
    loose: AlgoLimitMode


class OrderMapView(dict[Any, TradeInstruction]):
    """The order-id view over an algo or engine order index; entries are borrowed views, so the map is read-only."""


class AlgoMapView(dict[str, AlgoTemplate]):
    """The algo-id view over an engine or position algo index; entries are borrowed views, so the map is read-only."""


class AlgoRegistry:
    """
    The C-backed registry of named algo factories.

    The C registry points each name at its current registration context and
    owns every context, current or replaced, by registration id; the Python
    class table maps the registration id to the wrapper class, so a record
    resolves to the wrapper that built it. A name resolves to the class of the
    registration it currently names.
    """

    passive: str
    passive_timeout: str
    aggressive: str
    aggressive_timeout: str

    names: set[str]
    reversed_registry: dict[str, str]

    def add_algo(self, name: str, handler: type[AlgoTemplate]) -> None:
        """Register a wrapper class under a name.

        Args:
            name: The name the C registry already holds a factory under.
            handler: The ``AlgoTemplate`` subclass that wraps the built record.
        """

    def cast(self, name: str) -> type[AlgoTemplate]:
        """Resolve a name to its wrapper class."""


class AlgoEngine:
    """
    The C-backed algo engine.

    The engine holds the named factories, the order indexes and the working
    algo list per ticker. Each access carries the engine bound to it, so the
    feed entries reach the market data through the access. These entries are
    the only way an event moves an order or an algo: the wrapper classes do
    not re-dispatch them.
    """

    registry: AlgoRegistry

    algos: AlgoMapView
    """The algo-id view of every engine algo."""

    orders: OrderMapView
    """The order-id view of every engine order."""

    working_orders: OrderMapView
    """The order-id view of the live orders."""

    def __init__(self):
        """Allocate the engine."""

    def on_order(self, order: TradeInstruction) -> int:
        """Apply an order-state event to the tracked order and its algo.

        Returns:
            int: 1 when the engine applied the event, 0 when it did not track the order.
        """

    def on_trade(self, report: TradeReport) -> int:
        """Apply a fill or cancel report to the tracked order and its algo.

        Returns:
            int: 1 when the engine applied the report, 0 when it did not track the order.
        """

    def on_market_data(self, market_data: MarketData) -> None:
        """Dispatch an observation to every working algo of the ticker."""


ALGO_ENGINE: AlgoEngine
"""The process access's algo engine, adopted from ``DMA.algo_engine`` when this module initializes."""

ALGO_REGISTRY: AlgoRegistry
"""The process engine's registry of named algo factories and wrapper classes."""

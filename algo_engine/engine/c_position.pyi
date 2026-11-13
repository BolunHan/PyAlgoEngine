import enum
import uuid
from typing import Callable, Protocol

from algo_engine.base import TradeInstruction, TradeReport, TransactionSide
from algo_engine.engine.algo import AlgoTemplate
from algo_engine.engine.c_dma import DirectMarketAccess
from algo_engine.engine.c_dma_algo import AlgoEngine, AlgoMapView


class PositionCallbackEvent(enum.IntEnum):
    """The strategy-facing position events."""

    canceled = 1
    rejected = 2
    listed = 3
    filled = 4

    Canceled = canceled
    Rejected = rejected
    Listed = listed
    Filled = filled


class PositionOrderCallbackProtocol(Protocol):
    """The callback shape of the order events: canceled, rejected and listed."""

    def __call__(self, position: "PositionManagementService", algo: AlgoTemplate, order: TradeInstruction) -> int | None: ...


class PositionTradeCallbackProtocol(Protocol):
    """The callback shape of the filled event."""

    def __call__(self, position: "PositionManagementService", algo: AlgoTemplate, order: TradeInstruction, trade: TradeReport) -> int | None: ...


class PositionManagementService:
    """
    One strategy's C-backed view of the algo engine.

    The position hosts a subset of the engine-owned algo records: ``algos`` is
    the read-only algo-id view of the hosted instances, and the engine routes
    every order and trade event of a hosted record back to the position. The
    position never manages an algo's status; the per-ticker exposure, working
    volume, notional and cash flow are updated in place by those events, so no
    read refreshes them and ``pnl`` never walks the algos. The C block is owned
    by the access and freed with it; the wrapper is a view. The volume and
    order properties read the C state.

    Each event keeps a list of strategy callbacks and one C registration: the
    registration is created when the event's first callback arrives and
    released when its last callback leaves.

    Attributes:
        dma (DirectMarketAccess): The access this position trades through.
        algo_engine (AlgoEngine): The engine that owns the hosted records.
        position_id (uuid.UUID): The position identity used as the JSON map key.
        default_algo (str): The default algo name.
        algos (AlgoMapView): The read-only algo-id view of every hosted algo.
        working_algos (AlgoMapView): The read-only view of the hosted algos in
            the active set; a completed, closed, rejected, or errored algo
            leaves it.
    """

    dma: DirectMarketAccess
    algo_engine: AlgoEngine
    position_id: uuid.UUID
    default_algo: str
    algos: AlgoMapView
    working_algos: AlgoMapView

    def __init__(self, default_algo: str = None):
        """
        Allocate the C position in the process access and generate its identity.

        Args:
            default_algo (str): The default algo name; None selects the position default (passive).
        """
        ...

    def register_callback(self, event: PositionCallbackEvent, callback: Callable[..., int]) -> None:
        """
        Register a strategy callback for one event.

        The callback must accept the event's protocol shape: an order event
        calls ``callback(position, algo, order)``; a fill calls
        ``callback(position, algo, order, trade)``. The signature is bound
        against that shape before registration; a callback that cannot be
        called with it is refused. An exception raised by a callback is
        reported and the remaining callbacks still fire.

        The event keeps one C registration, created when its first callback
        arrives; every later callback of the event joins the same list.

        Args:
            event (PositionCallbackEvent): The event the callback answers.
            callback (Callable): The callback.
        """
        ...

    def unregister_callback(self, event: PositionCallbackEvent, callback: Callable[..., int]) -> None:
        """Remove the callback from the event; the event's C registration is released with its last callback."""
        ...

    def add_algo(self, algo: AlgoTemplate) -> None:
        """Host an engine-owned algo so its events route to this position."""
        ...

    def open(self, ticker: str, target_volume: float, trade_side: TransactionSide, algo: str = None, **kwargs) -> AlgoTemplate:
        """
        Build, host and launch an algo.

        Args:
            ticker (str): The instrument.
            target_volume (float): The volume to trade; zero or less is refused.
            trade_side (TransactionSide): The side of the trade.
            algo (str): The algo name; defaults to ``default_algo``.
            **kwargs: Arguments handed to the algo constructor and launch.

        Returns:
            AlgoTemplate: The hosted algo.
        """
        ...

    def unwind(self, ticker: str) -> AlgoTemplate | None:
        """Stop the ticker's working algos and open the unwinding algo."""
        ...

    def add_exposure(self, ticker: str, volume: float, notional: float, multiplier: float, side: TransactionSide) -> None:
        """Add synthetic exposure to the ticker's cache and store its contract multiplier."""
        ...

    def unwind_all(self) -> None:
        """Unwind every exposed ticker with the configured unwind kind."""
        ...

    def cancel_all(self) -> None:
        """Stop every hosted algo that has not completed; emergency use only."""
        ...

    def to_json(self, fmt: str = 'str') -> str | dict:
        """Export the algos; the map key is the position id as a string."""
        ...

    def pnl(self) -> dict[str, float]:
        """The per-ticker PnL of the hosted algos."""
        ...

    @property
    def notional(self) -> dict[str, float]:
        """The per-ticker filled notional."""
        ...

    @property
    def working_volume(self) -> dict[str, dict[str, float]]:
        """The working volume per ticker, split into ``{'Long': ..., 'Short': ...}``."""
        ...

    @property
    def exposure_volume(self) -> dict[str, float]:
        """The signed net exposure per ticker."""
        ...

    @property
    def working_volume_net(self) -> dict[str, float]:
        """The signed net working volume per ticker."""
        ...

    @property
    def orders(self) -> dict[str, TradeInstruction]:
        """Every order of every hosted algo."""
        ...

    @property
    def working_order(self) -> dict[str, TradeInstruction]:
        """Every working order of every hosted algo."""
        ...

    @property
    def trades(self) -> dict[str, TradeReport]:
        """Every trade of every hosted algo."""
        ...

    @property
    def order_mapping(self) -> dict[str, dict[str, TradeInstruction]]:
        """The orders grouped by algo id."""
        ...

    @property
    def reversed_order_mapping(self) -> dict[str, AlgoTemplate]:
        """The owning algo of every order id."""
        ...

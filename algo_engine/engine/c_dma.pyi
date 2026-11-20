import abc
from datetime import datetime
from typing import Any

from algo_engine.base import TradeInstruction, TradeReport, TransactionSide
from algo_engine.engine.c_balance import Balance
from algo_engine.engine.c_dma_algo import AlgoEngine, AlgoLimitMode
from algo_engine.engine.c_market_engine import MarketDataService
from algo_engine.engine.risk_profile.c_risk_profile import RiskProfile


class DirectMarketAccessProtocol(metaclass=abc.ABCMeta):
    """
    The direct market access interface.

    A concrete access implements the two required handlers; the balance handler
    and the rejected and filled callbacks are optional. The C ``dma`` struct
    stores one attached protocol, and the access binds it only when the
    subclass overrides both required handlers.
    """

    @abc.abstractmethod
    def launch_order_handler(self, order: TradeInstruction) -> None:
        """Send a launch order to the market."""

    @abc.abstractmethod
    def cancel_order_handler(self, order: TradeInstruction) -> None:
        """Send a cancel request to the market."""

    def balance_check_handler(self, balance: Balance) -> bool:
        """
        Fill the access's balance, marking the fields the host knows.

        Returns:
            True when a balance was supplied; False leaves the gate passing.
        """

    def rejected_order_callback(self, order: TradeInstruction) -> None:
        """Called when the access refuses to send an order."""

    def filled_order_callback(self, order: TradeInstruction, report: TradeReport) -> None:
        """Called when a fill is routed through the access."""


class DirectMarketAccess(DirectMarketAccessProtocol):
    """
    Direct market access, backed by the C ``dma`` struct.

    The access creates and owns its risk profile, inventory, balance and
    position map. It dispatches only the orders its algo engine creates; a
    caller cannot submit an instruction. A launch is checked against the risk
    profile and either reaches the attached launch handler or is marked
    rejected and reaches the rejected callback. The build selects direct dispatch
    (the default) or the request-ring queue, whose entries are drained by
    ``process_queue``. A subclass that does not override both required handlers
    is not bound and the access stays inert.

    Attributes:
        mds (MarketDataService): The market data source.
        risk_profile (RiskProfile): The access-owned profile, borrowed by this wrapper.
        algo_engine (AlgoEngine): The access-owned engine the access dispatches through.
        balance (Balance): The access-owned balance, borrowed by this wrapper.
    """

    mds: MarketDataService
    risk_profile: RiskProfile
    algo_engine: AlgoEngine
    balance: Balance

    def __init__(self):
        """
        Allocate the C access and its owned sub-blocks.

        The access binds the process ``MDS`` and owns its risk profile,
        balance, algo engine, inventory, positions and request ring.
        """

    def process_queue(self) -> int:
        """Drain every queued request in FIFO order and return the count."""

    def calculate_limit(
            self,
            ticker: str,
            side: TransactionSide,
            limit_price: float | None = None,
            limit_adjust_factor: float | None = None,
            mode: AlgoLimitMode = AlgoLimitMode.strict
    ) -> float | None:
        """
        Calculate an order limit from the access's market price and the algo limits.

        Args:
            ticker (str): The instrument.
            side (TransactionSide): The trade side.
            limit_price (float): The absolute limit; None leaves it unset.
            limit_adjust_factor (float): The market-price factor; None leaves it unset.
            mode (AlgoLimitMode): The limit combination; defaults to strict.

        Returns:
            The calculated limit, or None when the market price is unavailable.
        """

    @property
    def protocol_bound(self) -> bool:
        """Whether the subclass's required handlers are bound to the access."""

    @property
    def timestamp(self) -> float:
        """The market data service's latest timestamp."""

    @property
    def market_price(self) -> dict[str, float]:
        """The market data service's latest traded prices (NaN when unpriced)."""

    @property
    def market_time(self) -> datetime | None:
        """The market data service's latest market time."""


class EventDMA(DirectMarketAccess):
    """
    The process access: it publishes launch and cancel orders on EVENT_ENGINE.

    A locally refused order is already applied to its algo, so the rejected
    callback does nothing.
    """

    event_engine: Any

    def __init__(self, event_engine: Any = None):
        """Allocate an access that routes orders through an event engine; None uses EVENT_ENGINE."""


DMA: EventDMA
"""The process access every algo trades through."""

RISK_PROFILE: RiskProfile
"""The access-owned process risk profile."""

BALANCE: Balance
"""The access-owned process balance."""

import random
from datetime import datetime

from event_engine import EventEngine

from ..base import (
    BarData,
    MarketData,
    OrderData,
    TickData,
    TickDataLite,
    TradeData,
    TradeInstruction,
    TradeReport,
    TransactionData,
    TransactionDirection,
    TransactionSide,
)
from ..engine import TopicSet


class SimMatchEx:
    """C-layer simulation matcher for backtesting.

    Full C implementation of the matching engine: the working order registry,
    lag / hit-probability / slippage logic and report construction all run in
    the native layer. The wrapper owns the Python-side ``TradeInstruction``
    objects, applies fills to their headers and publishes events.

    Attributes:
        ticker: The ticker this matcher serves.
        event_engine: Event engine used for registration and event publishing.
        topic_set: Topic set used for registration and event publishing.
        working: Mapping of order_id to currently working orders.
        history: Mapping of order_id to done (filled / canceled) orders.
        random: Python random instance kept for interface parity; the native
            matching layer uses its own seeded PRNG.
        timestamp: Latest market data timestamp (seconds since epoch).
        last_price: Latest market price, or None before any market data.
        last_transaction_count: Number of transactions seen so far.
        seed: Seed of the native PRNG (0-derived values are materialized).
        matching_config: Read-only mirror of the native matching config
            (``fee_rate``, ``instant_fill``, ``lag``, ``hit``,
            ``incremental_order_volume``). Configure via constructor kwargs.
        market_time: Session-converted datetime of ``timestamp``.

    Note:
        Compared to the legacy pure-Python ``SimMatch``: filled orders are
        removed from ``working`` immediately (including instant fills), and
        each placement / cancel / fill publishes exactly one ``on_order``
        event plus one ``on_report`` per fill.
    """
    ticker: str
    event_engine: EventEngine
    topic_set: TopicSet
    working: dict[str, TradeInstruction]
    history: dict[str, TradeInstruction]
    random: random.Random

    def __init__(
        self,
        ticker: str,
        event_engine: EventEngine | None = None,
        topic_set: TopicSet | None = None,
        seed: int | None = None,
        **kwargs: object,
    ) -> None:
        """Initialize the matcher.

        Args:
            ticker: Ticker to match.
            event_engine: Event engine; defaults to the global
                ``EVENT_ENGINE``.
            topic_set: Topic set; defaults to the global ``TOPIC``.
            seed: Seed for the native PRNG (hit probability); random when
                None.
            **kwargs: Matching configuration:
                fee_rate: Fee as fraction of notional (default 0).
                instant_fill: Fill immediately at launch when no lag is set
                    (default False).
                lag_ts: Minimum time lag in seconds before a fill (default 0).
                lag_n_transaction: Minimum transaction lag before a fill
                    (default 0).
                hit_prob: Probability of a fill attempt succeeding
                    (default 1.0).
                slippery_rate: Slippage as fraction of price (default 0.0001).
                incremental_order_volume: Treat OrderData volume as
                    incremental fills (SH exchange style) and match against
                    it; when False (default) order data is not matched —
                    SZ-style depth reports carry resting volume, not
                    incremental fills.
        """
        ...

    def __call__(self, **kwargs: object) -> None:
        """Process one order or market data update.

        Args:
            **kwargs: Exactly one of:
                order: TradeInstruction to launch (ORDER_LIMIT) or cancel
                    (ORDER_CANCEL). Other order types raise ValueError.
                market_data: Bar / Tick / TickLite / Order / Transaction data
                    to match against.
        """
        ...

    @staticmethod
    def best_price(*price: float | None, side: TransactionSide | TransactionDirection) -> float:
        """Best price for the side (buy: min, sell: max).

        Args:
            price: Candidate prices; None / NaN / inf values are skipped.
            side: Transaction side or direction.

        Returns:
            The best price.

        Raises:
            ValueError: No valid price provided, or invalid side.
        """
        ...

    @staticmethod
    def worst_price(*price: float | None, side: TransactionSide | TransactionDirection) -> float:
        """Worst price for the side (buy: max, sell: min).

        Args:
            price: Candidate prices; None / NaN / inf values are skipped.
            side: Transaction side or direction.

        Returns:
            The worst price.

        Raises:
            ValueError: No valid price provided, or invalid side.
        """
        ...

    def register(
        self,
        topic_set: TopicSet | None = None,
        event_engine: EventEngine | None = None,
    ) -> None:
        """Bind the matcher natively on the event engine.

        When the engine is the native ``EventEngine``, the C layer stores
        the engine interface (message queue and hook maps), registers its C
        dispatch handler directly on the launch / cancel / realtime hooks
        (``c_evt_hook_register_callback``) and publishes ``on_order`` /
        ``on_report`` payloads through the engine queue — everything runs in
        C after registration. Falls back to Python ``register_handler`` for
        non-native engines.

        Args:
            topic_set: Topic set to bind with; keeps the current one when
                None.
            event_engine: Engine to bind on; keeps the current one when None.
        """
        ...

    def unregister(self) -> None:
        """Remove the native dispatch handlers and hooks from the engine."""
        ...

    def launch_order(self, order: TradeInstruction, **kwargs: object) -> None:
        """Launch an order into the working registry.

        Marks the order PLACED at the current timestamp, snapshots the
        transaction counter for lag computation, then applies the
        instant-fill short circuit when configured.

        Args:
            order: Order to launch.
            **kwargs: Ignored (interface parity).

        Raises:
            ValueError: The order_id already exists in working or history.
        """
        ...

    def cancel_order(
        self,
        order: TradeInstruction | None = None,
        order_id: object | None = None,
        **kwargs: object,
    ) -> None:
        """Cancel a working order.

        Args:
            order: Order to cancel (resolved by its order_id).
            order_id: Order id to cancel (str / int / uuid.UUID as stored in
                the registry); required when order is None.
            **kwargs: Ignored (interface parity).

        Raises:
            ValueError: Neither order nor order_id is provided.
        """
        ...

    def eod(self) -> None:
        """Cancel every working order (end of day)."""
        ...

    def clear(self) -> None:
        """Reset all state: registry, market state, PRNG (keeps config)."""
        ...

    def on_order(self, order: TradeInstruction, **kwargs: object) -> None:
        """Publish an order update on the ``on_order`` topic."""
        ...

    def on_report(self, report: TradeReport, **kwargs: object) -> None:
        """Publish a trade report on the ``on_report`` topic."""
        ...

    @property
    def timestamp(self) -> float:
        """Latest market data timestamp (seconds since epoch)."""
        ...

    @timestamp.setter
    def timestamp(self, value: float) -> None: ...

    @property
    def last_price(self) -> float | None:
        """Latest market price, or None before any market data."""
        ...

    @last_price.setter
    def last_price(self, value: float | None) -> None: ...

    @property
    def last_transaction_count(self) -> int:
        """Number of transactions seen so far."""
        ...

    @last_transaction_count.setter
    def last_transaction_count(self, value: int) -> None: ...

    @property
    def seed(self) -> int:
        """Seed of the native PRNG."""
        ...

    @seed.setter
    def seed(self, value: int | None) -> None: ...

    @property
    def matching_config(self) -> dict[str, object]:
        """Read-only mirror of the native matching configuration."""
        ...

    @property
    def market_time(self) -> datetime:
        """Session-converted datetime of ``timestamp``."""
        ...

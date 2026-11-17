import random
from datetime import datetime

from event_engine import EventEngineEx

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
from ..engine import DMAProtocol, TopicSetEx


class SimMatch:
    """C-layer simulation matcher for backtesting.

    The working-order registry, lag / hit-probability / slippage logic and
    report construction run in the native layer. The matcher copies every
    launched order and owns that copy; the object passed to ``launch_order`` is
    never modified. Fills and state transitions update the matcher's copy,
    which is what ``working``, ``history`` and the published events carry.

    Attributes:
        ticker: The ticker this matcher serves.
        event_engine: Native ``EventEngineEx`` used for registration and event
            publishing.
        topic_set: Native ``TopicSetEx`` used for registration and event
            publishing.
        protocol: The ``DMAProtocol`` bound for realtime market data after
            ``register()``, or None while unbound.
        working: Mapping of order_id to the matcher's working order copies.
        history: Mapping of order_id to the matcher's done (filled / canceled)
            order copies.
        random: Python random instance kept for interface parity; the native
            matching layer uses its own seeded PRNG.
        timestamp: Latest market data timestamp (seconds since epoch).
        last_price: Latest market price, or None before any market data.
        last_transaction_count: Number of transactions seen so far.
        seed: Seed of the native PRNG (read-only).
        matching_config: Read-only mirror of the native matching config
            (``fee_rate``, ``instant_fill``, ``lag``, ``hit``,
            ``incremental_order_volume``). Configure via constructor kwargs.
        market_time: Session-converted datetime of ``timestamp``.

    The attributes above are read-only after construction.
    """
    ticker: str
    event_engine: EventEngineEx
    topic_set: TopicSetEx
    protocol: DMAProtocol | None
    working: dict[str, TradeInstruction]
    history: dict[str, TradeInstruction]
    random: random.Random
    seed: int

    def __init__(
        self,
        ticker: str,
        event_engine: EventEngineEx | None = None,
        topic_set: TopicSetEx | None = None,
        seed: int = 0,
        **kwargs: object,
    ) -> None:
        """Initialize the matcher.

        Args:
            ticker: Ticker to match.
            event_engine: Engine instance; defaults to the global
                ``EVENT_ENGINE``.
            topic_set: Topic-set instance; defaults to the global
                ``TOPIC_SET``.
            seed: Seed for the native PRNG (hit probability and trade ids);
                0 derives a random seed.
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

        Raises:
            MemoryError: The native context or the working registry could not
                be allocated.
            TypeError: The engine or topic set is not an ``EventEngineEx`` /
                ``TopicSetEx``.
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

        Raises:
            ValueError: Invalid order type or market data.
            RuntimeError: The matcher is unbound or the native layer failed.
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
        topic_set: TopicSetEx | None = None,
        event_engine: EventEngineEx | None = None,
    ) -> None:
        """Bind the matcher on the event engine.

        Launch and cancel orders are in-process Python payloads (the DMA and
        strategy layers publish them with ``engine.put``), so they bind as
        Python handlers. Market data is a transport payload: the matcher
        creates a :class:`~algo_engine.engine.c_dma_protocol.DMAProtocol` over
        the engine and topic set and registers one realtime callback, so the C
        matcher receives the typed ``md_variant`` directly. The protocol is
        configured non-blocking so a full queue never deadlocks the engine
        loop.

        Args:
            topic_set: Topic set to bind with; keeps the current one when None.
            event_engine: Engine to bind on; keeps the current one when None.

        Raises:
            RuntimeError: The matcher is unbound or a realtime callback could
                not be unregistered from the previous binding.
            TypeError: The engine or topic set is not one of the accepted
                types.
        """
        ...

    def unregister(self) -> None:
        """Remove the dispatch handlers and the realtime callback."""
        ...

    def launch_order(self, order: TradeInstruction, **kwargs: object) -> None:
        """Launch a copy of an order into the working registry.

        The matcher snapshots the transaction counter, marks its copy PLACED
        and applies the instant-fill short circuit when configured. The passed
        instruction is not modified.

        Args:
            order: Order to copy into the registry.
            **kwargs: Ignored (interface parity).

        Raises:
            ValueError: The order_id already exists in working or history, or
                the native registry rejected the order.
            MemoryError: The working registry could not grow.
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
            RuntimeError: The matcher is unbound or the native cancel failed.
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
    def matching_config(self) -> dict[str, object]:
        """Read-only mirror of the native matching configuration."""
        ...

    @property
    def market_time(self) -> datetime:
        """Session-converted datetime of ``timestamp``."""
        ...

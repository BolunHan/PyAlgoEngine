from typing import Any

from event_engine.capi import Topic, EventEngineEx

from ..base.c_market_data.c_market_data import MarketData


class TopicSet:
    """Legacy helper for building and parsing event topics (Python Topic objects).

    Not the module default: :data:`TOPIC_SET` is the
    :class:`TopicSetEx` singleton. A legacy instance binds its nine topics as non-owning
    views over the module's C topic set, so it never frees a topic the C set
    owns.

    Attributes:
        on_order: Exact topic for order submission events.
        on_report: Exact topic for order/trade report events.
        eod: Exact topic signaling end-of-day start.
        eod_done: Exact topic signaling end-of-day completion.
        bod: Exact topic signaling start-of-day start.
        bod_done: Exact topic signaling start-of-day completion.
        launch_order: Template topic, e.g., 'launch_order.{ticker}'.
        cancel_order: Template topic, e.g., 'cancel_order.{ticker}'.
        realtime: Template for real-time market data, e.g., 'realtime.{ticker}.{dtype}'.
        push_topic_map: Per-ticker cache mapping dtype to resolved `Topic`.
    """
    __dict__: dict[str, Any]

    on_order: Topic
    on_report: Topic
    eod: Topic
    eod_done: Topic
    bod: Topic
    bod_done: Topic
    launch_order: Topic
    cancel_order: Topic
    realtime: Topic
    push_topic_map: dict[str, dict[int, Topic]]

    def __init__(self) -> None: ...

    def push(self, market_data: MarketData) -> Topic:
        """Return the cached real-time topic for the given market data.

        Resolves to 'realtime.{ticker}.{dtype}' from the C market data buffer
        and caches the topic per (ticker, dtype).

        Args:
            market_data: Market data object holding a C buffer.

        Returns:
            A resolved `Topic` for publishing the market data.
        """
        ...

    def parse(self, topic: Topic) -> dict[str, str]:
        """Parse an exact realtime topic back into its placeholder values.

        Args:
            topic: A fully-resolved `Topic` built from the `realtime` template.

        Returns:
            A mapping from placeholder name to its string value.
        """
        ...


class TopicSetEx:
    """C-backed set of the nine built-in event topics plus a dynamic registry.

    The topic keys are the legacy ones, with no dispatch suffix::

        on_order, on_report, eod, eod_done, bod, bod_done,
        launch_order.{ticker}, cancel_order.{ticker},
        realtime.{ticker}.{dtype}

    The underlying ``evt_topic`` objects are allocator-backed and owned by this
    set; the ``Topic`` objects exposed here are non-owning views valid while
    the set lives. :data:`TOPIC_SET` is the owning singleton. A second owning
    set for the same literals shares interned topics, so do not create one.

    Attributes:
        on_order: Exact topic ``on_order``.
        on_report: Exact topic ``on_report``.
        eod: Exact topic ``eod``.
        eod_done: Exact topic ``eod_done``.
        bod: Exact topic ``bod``.
        bod_done: Exact topic ``bod_done``.
        launch_order: Template topic ``launch_order.{ticker}``.
        cancel_order: Template topic ``cancel_order.{ticker}``.
        realtime: Template topic ``realtime.{ticker}.{dtype}``.
        address: Address of the underlying C ``evt_topic_set``.
    """
    on_order: Topic
    on_report: Topic
    eod: Topic
    eod_done: Topic
    bod: Topic
    bod_done: Topic
    launch_order: Topic
    cancel_order: Topic
    realtime: Topic

    __dict__: dict[str, Any]

    def __init__(self) -> None:
        """Allocate a topic set and bind the nine built-in topics.

        Raises:
            MemoryError: If the C topic set or one of its topics cannot be
                allocated.
            KeyError: If a built-in topic is missing after initialization.
        """
        ...

    def push(self, market_data: MarketData) -> Topic:
        """Return the realtime topic for a market data buffer.

        Resolves to ``realtime.{ticker}.{dtype}``. Tick, transaction and order
        data use a dedicated ticker cache; other dtypes resolve through the
        registry. The topic is registered on first use.

        Each set-owned topic has one cached ``Topic`` view, so pushing the same
        ticker and dtype again returns the identical object; the C-level
        ``c_push`` is the path that builds a fresh view per call.

        Args:
            market_data: Market data buffer whose ``meta_info`` is read.

        Returns:
            The cached, non-owning ``Topic`` view over a set-owned topic.

        Raises:
            ValueError: If the ticker is missing or the dtype is unknown.
        """
        ...

    def parse(self, topic: Topic) -> tuple[str, str]:
        """Parse an exact realtime topic into ``(ticker, dtype_name)``.

        Args:
            topic: An exact topic shaped like ``realtime.<ticker>.<dtype>``.

        Returns:
            A ``(ticker, dtype_name)`` tuple, both fresh Python strings.

        Raises:
            TypeError: If ``topic`` is not initialized.
            ValueError: If the topic is not an exact realtime topic or carries
                an unknown dtype name.
        """
        ...

    def parse_map(self, topic: Topic) -> dict[str, str]:
        """Mapping form of :meth:`parse`: ``{'ticker': ..., 'dtype': ...}``."""
        ...

    def get(self, name: str) -> Topic:
        """Get-or-set: the topic registered as ``name``, created on first use.

        Args:
            name: Full topic key.

        Returns:
            A non-owning ``Topic`` view over a set-owned topic.

        Raises:
            MemoryError: If a missing topic cannot be created.
        """
        ...

    def register(self, name: str) -> Topic:
        """Alias of :meth:`get`; the same get-or-set."""
        ...

    def __getitem__(self, name: str) -> Topic:
        """``tset['realtime.RB2510.TickData']`` - the same get-or-set as :meth:`get`."""
        ...

    def __contains__(self, name: object) -> bool:
        """``'on_order' in tset`` - true when the literal is registered.

        Non-string values are never contained.
        """
        ...

    def __getattr__(self, name: str) -> Topic:
        """Fallback attribute access: resolve ``name`` from the registry.

        Only consulted when normal lookup fails, so declared attributes always
        win. Raises ``AttributeError`` when the literal is not registered.
        """
        ...

    def __or__(self, other: TopicSetEx) -> TopicSetEx:
        """Merge two sets: ``left | right`` returns a new merged set.

        Python-only operator. The result contains every literal of both inputs
        and owns none of the topics: it borrows them, so both inputs must
        outlive it. A literal present in both resolves to the left set's topic.
        The merged set is read-only: get, ``[]`` and attribute lookup resolve
        registered literals only and raise KeyError/AttributeError otherwise.
        """
        ...

    @property
    def address(self) -> int:
        """Address of the underlying C ``evt_topic_set``."""
        ...


#: Global event engine singleton used across the algo engine.
EVENT_ENGINE: EventEngineEx

#: Global owning C-backed topic set singleton.
TOPIC_SET: TopicSetEx

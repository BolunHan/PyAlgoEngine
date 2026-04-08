from event_engine.capi import Topic, EventEngineEx

from ..base.c_market_data.c_market_data import MarketData


class TopicSet:
    """Helper for building and parsing event topics used by the algo engine.

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

        Resolves to 'realtime.{ticker}.{dtype}' using values read from the
        low-level market data buffer referenced by `market_data._data_addr`.
        Caches the topic per (ticker, dtype) for reuse.

        Args:
            market_data: Object exposing `_data_addr` pointing to a C buffer.

        Returns:
            A resolved `Topic` for publishing the market data.
        """
        ...

    def parse(self, topic: Topic) -> dict[str, str]:
        """Parse a resolved real-time topic back into its parameters.

        Expects an exact topic that matches the `realtime` template. Extracts
        literal placeholders (e.g., 'ticker', 'dtype') with their values.

        Args:
            topic: A fully-resolved `Topic` built from the `realtime` template.

        Returns:
            A mapping from placeholder name to its string value.
        """
        ...


#: Global event engine singleton used across the algo engine.
EVENT_ENGINE: EventEngineEx

#: Global helper with common topics and utilities.
TOPIC: TopicSet

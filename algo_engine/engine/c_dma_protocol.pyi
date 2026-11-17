from collections.abc import Callable

from event_engine.capi import EventEngineEx, Topic

from ..base.c_market_data.c_market_data import MarketData


class DMAProtocol:
    """Market-data transport over the process event engine.

    The protocol publishes ``md_variant`` payloads on the topic selected by the
    payload dtype: a trade instruction on ``on_order``, an exchange report on
    ``on_report``, and every other dtype (including order data) on
    ``realtime.{ticker}.{dtype}``. Outgoing orders use the C-side
    ``c_dma_protocol_launch_order`` / ``c_dma_protocol_cancel_order`` handlers,
    which publish on ``launch_order.{ticker}`` / ``cancel_order.{ticker}`` and
    are signature-compatible with the access's order handlers.

    Send copies the payload into the protocol's shared-memory region and hands
    the copy to the engine; the source stays the caller's. One static engine
    payload is reused: a second send before the previous message is dispatched
    is refused with ``-8`` (busy).

    Each callback registration is its own engine handler, carrying the callback
    and its topic, so a callback receives only the payloads of the topic it was
    registered on. The callback receives a borrowed :class:`MarketData` view,
    which is not a copy and must not be modified; it is nullified through a
    lightweight allocator callback when the payload is released. A message sent
    through this protocol is delivered to a callback of the same protocol as the
    source wrapper itself.

    The protocol binds the process :data:`EVENT_ENGINE`, :data:`TOPIC_SET`,
    ``MD_SHM_ALLOCATOR`` and ``C_POOL`` by default; all four must outlive it.
    A caller may bind another process engine and topic set (for example a test
    engine) as long as their shared-memory pool is the process pool.
    Unregister every callback and drain the event engine before releasing the
    protocol.

    Raises:
        MemoryError: If the protocol block cannot be allocated.
    """

    def __init__(
        self,
        event_engine: EventEngineEx | None = None,
        topic_set: object | None = None,
    ) -> None:
        """Create a protocol over the process engine/topic set (or the given ones).

        Args:
            event_engine: Engine to publish on and register callbacks with;
                defaults to the process ``EVENT_ENGINE``.
            topic_set: Topic registry used to resolve payload topics; defaults
                to the process ``TOPIC_SET``.
        """
        ...

    def send(self, market_data: MarketData) -> int:
        """Publish market data on its dtype's topic.

        A trade instruction publishes on ``on_order``, an exchange report on
        ``on_report``, and every other dtype (including order data) on
        ``realtime.{ticker}.{dtype}``.

        The transport copies the payload into shared memory; the source stays
        the caller's and is never left unbound.

        Args:
            market_data: The market data to publish.

        Returns:
            ``0`` on success, otherwise a negative ``dma_protocol_ret_code``:
            ``-1`` uninitialized input, ``-3`` the shared-memory copy failed,
            ``-4`` the payload has no exact topic, ``-5`` the engine rejected
            the message and released the copy, ``-8`` the previous static
            payload is still in flight.
        """
        ...

    def register_callback(self, callback: Callable[[MarketData], None], topic: Topic) -> int:
        """Register a callback on one topic directly with the event engine.

        The callback receives a borrowed :class:`MarketData` view per event.
        The view is not a copy and must not be modified; it is nullified when
        the shared-memory payload is released.

        The registered topic must carry only transport payloads: the callback
        is registered as an args/topic/user-data handler and receives the
        payload's ``args`` as the market data, with no payload tag check.

        Args:
            callback: Called as ``callback(market_data)`` for every payload on
                the topic.
            topic: Exact or generic topic to register on.

        Returns:
            ``0`` on success, or ``-9`` when the callback is already registered
            on that topic.

        Raises:
            MemoryError: If the callback context cannot be allocated or the
                event engine runs out of memory.
            ValueError: If the protocol or the event engine is uninitialized.
            RuntimeError: If the event engine returns an unmapped code.
        """
        ...

    def unregister_callback(self, callback: Callable[[MarketData], None], topic: Topic) -> int:
        """Remove a callback registration.

        The registration is removed from the protocol and its engine handler
        with it.

        Args:
            callback: The callback to remove.
            topic: The topic it was registered on.

        Returns:
            ``0`` when the registration was removed, or ``-7`` when no such
            registration exists.

        Raises:
            ValueError: If the protocol is uninitialized.
            RuntimeError: If the event engine returns an unmapped code.
        """
        ...

    def config(self, block: bool, max_spin: int = ..., timeout: float = 0.0) -> None:
        """Configure how :meth:`send` publishes.

        Args:
            block: Block when the engine queue is full.
            max_spin: Spin iterations before falling back to blocking.
            timeout: Blocking timeout in seconds; values <= 0 wait forever.

        Raises:
            ValueError: If the protocol is uninitialized.
        """
        ...

    @property
    def address(self) -> int:
        """Address of the underlying C ``dma_protocol_ctx`` block."""
        ...

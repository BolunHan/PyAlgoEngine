import abc
from datetime import date, datetime
from typing import Any

from algo_engine.base import FilterMode, InternalData, MarketData
from algo_engine.exchange_profile import ExchangeProfile

MDS: MarketDataService


class MarketDataMonitorProtocol(metaclass=abc.ABCMeta):
    """
    Template for a market data monitor.

    A monitor processes incoming market data and generates custom indices.

    When MDS receives an update, it triggers the monitor's `__call__` method. All subscribed tickers' data
    will be passed to the monitor, so proper storage for multiple tickers is assumed.

    To use:
        - Access via: `monitor = MDS[monitor_id]`
        - Access output via: `monitor.value`
        - Mark ready via: `monitor.is_ready = True`
        - Add to engine using: `MDS.add_monitor(monitor)`

    Attributes:
        name (str): Unique name of the monitor (not enforced).
        monitor_id (Any): Unique identifier, auto-generated if not provided.
        enabled (bool): Whether the monitor is active (default: True).
    """
    name: str
    monitor_id: Any
    enabled: bool

    def __init__(self, name: str, monitor_id: Any = None):
        """
        Initialize the monitor.

        Args:
            name (str): Unique name of the monitor.
            monitor_id (Any): Unique identifier, auto-generated if not provided.
        """

    @abc.abstractmethod
    def __call__(self, market_data: MarketData, **kwargs):
        """
        Feed market data into the monitor.

        Args:
            market_data (MarketData): The incoming market data.
            **kwargs: For compatibility only; should not be used.
        """

    @abc.abstractmethod
    def clear(self) -> None:
        """
        Clear the monitor's internal data.

        Called at end-of-day (EoD) before shutdown, for graceful termination.
        """

    @property
    @abc.abstractmethod
    def value(self) -> float | dict[str, float] | None:
        """
        Retrieve the computed value(s) from this monitor.

        Returns:
            Either a single float or a dictionary of string keys to floats.
        """

    @property
    def is_ready(self) -> bool:
        """
        Indicates whether the monitor is ready to produce values.

        Returns:
            True if ready; False otherwise. Defaults to True.
        """



class MarketDataMonitor(MarketDataMonitorProtocol):
    """
    Functional C-backed monitor base, bound to an ``mds_monitor`` block.

    The class is a CCPType-bound cdef class: it owns (or adopts) an
    ``mds_monitor`` header, exposes the generic monitor fields through C, and
    is registered as a virtual subclass of :class:`MarketDataMonitorProtocol`.
    Concrete monitors subclass it and override ``__call__``, ``clear`` and
    ``value``.

    Attributes:
        name (str): Monitor name; its UTF-8 bytes back the C base.
        monitor_id (Any): Python identity; a UUID is written into the C base.
        enabled (bool): Whether the monitor is active (C field).
        is_ready (bool): Whether the monitor has output ready (C field).
        priority (int): Dispatch order hint (C field).
        md_filter (FilterMode): Generic market-data filter flags.
    """

    name: str
    monitor_id: Any
    enabled: bool
    is_ready: bool
    priority: int
    md_filter: FilterMode

    def __init__(
            self,
            name: str,
            filter_mode: FilterMode | int = 0,
            priority: int = 0,
            monitor_id: Any = None,
    ):
        """
        Allocate and bind the monitor's C base.

        Args:
            name (str): Monitor name; copied into the C base.
            filter_mode (FilterMode | int): Generic market-data filter; 0 means no filter.
            priority (int): Dispatch priority.
            monitor_id (Any): Python identity; a UUID is copied into the C id,
                any other object is hashed into it.
        """

    def __call__(self, market_data: MarketData, **kwargs):
        """Default feed hook; subclasses override."""

    def clear(self) -> None:
        """Default clear hook; subclasses override."""

    @property
    def value(self) -> float | dict[str, float] | None:
        """Default output (None); subclasses override."""


class MonitorManager(object):
    """
    Manages market data monitors.

    This is a basic, single-threaded implementation. It is typically used as a component of MDS.

    Attributes:
        monitor (dict[Any, MarketDataMonitor]): Registered monitors, bound to the manager's C registry.
    """
    monitor: dict[Any, MarketDataMonitor]

    def __call__(self, market_data: MarketData):
        """
        Dispatch market data to all registered monitors.

        Args:
            market_data (MarketData): Incoming data.
        """

    def feed_monitor(self, monitor_id: Any, market_data: MarketData) -> None:
        """
        Feed market data to a specific monitor.

        Args:
            monitor_id (Any): The monitor's unique ID.
            market_data (MarketData): Incoming data.
        """

    def on_market_data(self, market_data: MarketData) -> None:
        """
        Dispatch market data to all registered monitors.

        Args:
            market_data (MarketData): Incoming data.
        """

    def add_monitor(self, monitor: MarketDataMonitor) -> None:
        """
        Register a monitor.

        Args:
            monitor (MarketDataMonitor): The monitor to add.
        """

    def pop_monitor(self, monitor_id: Any) -> None:
        """
        Unregister a monitor by ID.

        Args:
            monitor_id (Any): ID of the monitor to remove.
        """

    def clear_monitors(self) -> None:
        """
        Clear all monitors.
        """

    def get_values(self) -> dict:
        """
        Collect current monitor values into a single dictionary.

        Iterates over all monitors and merges their ``value`` dicts.

        Returns:
            A dictionary of aggregated monitor values.
        """

    def __contains__(self, monitor_id: Any) -> bool:
        """
        Check if a monitor is registered.

        Args:
            monitor_id (Any): The monitor's unique ID.

        Returns:
            True if the monitor is registered.
        """

    def start(self):
        """
        Start the manager.

        Placeholder for subclasses that require initialization.
        """

    def stop(self):
        """
        Stop the manager.

        Placeholder for subclasses that require cleanup.
        """

    def clear(self):
        """
        Clear all monitor data for graceful shutdown.
        """

    def __enter__(self):
        """Start the manager as a context manager."""

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop the manager when exiting the context."""

    @property
    def values(self) -> dict[str, Any]:
        """
        Collect current values from all monitors.

        Returns:
            A dictionary mapping monitor names to their values.
        """

    @property
    def n_monitors(self) -> int:
        """
        Number of monitors in the manager's C registry.

        The registry stores the monitor bases (`mds_monitor*`), mirroring the
        Python ``monitor`` dict.
        """


class MarketDataService:
    """
    Singleton service for managing market data.

    Handles distribution, storage, and monitor coordination.
    All market data–related operations should be registered here.

    Attributes:
        profile (ExchangeProfile): The active market profile.
        n_subscribed (int): Number of tickers currently subscribed.
        monitor (dict[Any, MarketDataMonitor]): Registered monitors, bound to the service's C registry.
        monitor_manager (MonitorManager): The monitor manager in use.
        timestamp (float): UNIX timestamp of the latest market data (NaN if uninitialized).
    """
    profile: ExchangeProfile
    n_subscribed: int
    monitor: dict[Any, MarketDataMonitor]
    monitor_manager: MonitorManager
    timestamp: float

    def __len__(self) -> int:
        """
        Get number of tickers processed.

        Returns:
            Number of tickers with available data.
        """
        ...

    def __call__(self, market_data: MarketData):
        """
        Alias for `on_market_data`.

        Args:
            market_data (MarketData): Incoming data.
        """
        ...

    def __getitem__(self, monitor_id: Any) -> MarketDataMonitorProtocol:
        """
        Access monitor by ID.

        Args:
            monitor_id (Any): Monitor's unique ID.

        Returns:
            MarketDataMonitor: The corresponding monitor.
        """
        ...

    def on_internal_data(self, internal_data: InternalData) -> None:
        """
        Feed internal communication protocol data to the service.

        Args:
            internal_data (InternalData): The internal protocol message.
        """
        ...

    def on_market_data(self, market_data: MarketData) -> None:
        """
        Handle market data update and trigger monitors.

        Args:
            market_data (MarketData): Incoming data.
        """
        ...

    def set_manager(self, manager: MonitorManager | None) -> None:
        """
        Replace the monitor manager, re-registering all existing monitors.

        The manager's C base struct is installed on the service before the
        previous manager is released, so the feed path never sees a dangling
        pointer. Passing ``None`` disables manager dispatch.

        Args:
            manager (MonitorManager | None): The new manager to assign.
        """

    def get_market_price(self, ticker: str) -> float:
        """
        Retrieve the latest traded price of a ticker.

        A transaction is the primary source. After
        ``MDS_TICK_PRICE_STALE_SECONDS`` (3 s by default) without a trade, a
        tick may refresh the price from its last price, the bid/ask mid-point,
        or the one present side. Order and bar observations never move it.

        Args:
            ticker (str): The instrument's symbol.

        Returns:
            float: Latest traded price, or NaN when the ticker is unknown,
            unfed, or has no traded/tick price yet.
        """
        ...

    def add_monitor(self, monitor: MarketDataMonitor) -> None:
        """
        Register a monitor with the service.

        Note:
            MDS and its ``MonitorManager`` store monitors independently.

        Args:
            monitor (MarketDataMonitor): The monitor to add.
        """
        ...

    def pop_monitor(
            self,
            monitor: MarketDataMonitorProtocol = None,
            monitor_id: Any = None,
            monitor_name: str = None,
    ) -> None:
        """
        Remove a monitor from the service.

        At least one of ``monitor``, ``monitor_id``, or ``monitor_name`` must be provided.

        Args:
            monitor (MarketDataMonitor): The monitor instance to remove.
            monitor_id (Any): The ID of the monitor to remove.
            monitor_name (str): The name of the monitor to remove.
        """
        ...

    def clear(self):
        """
        Reset the service to its initial state.
        """
        ...

    @property
    def market_price(self) -> dict[str, float]:
        """
        Latest known traded prices.

        Tickers without a priced observation map to NaN; order-only tickers
        are subscribed but carry no market price.

        Returns:
            A dictionary mapping tickers to prices.
        """
        ...

    @property
    def market_time(self) -> datetime | None:
        """
        Latest timestamp of received market data.

        Returns:
            A `datetime` object or None.
        """
        ...

    @property
    def market_date(self) -> date | None:
        """
        Date component of the latest market data.

        Returns:
            A `date` object or None.
        """
        ...

    @property
    def n_monitors(self) -> int:
        """
        Number of monitors in the service's C registry.

        The service mirrors the Python ``monitor`` dict into a C registry of
        monitor bases, so C consumers can find a monitor without the GIL.
        """

    @property
    def subscriptions(self) -> dict[str, int]:
        """
        Current subscription mapping, as a snapshot.

        The service stores each subscription in its C bytemap; this property
        walks that map and returns a plain dict keyed by ticker, in insertion
        order. A mapping obtained earlier does not update when new tickers are
        subscribed -- read the property again for a fresh snapshot.

        Returns:
            A dictionary mapping ticker symbols to their subscription indices.
        """
        ...

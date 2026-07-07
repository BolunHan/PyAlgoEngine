import abc
from datetime import date, datetime
from typing import Any

from algo_engine.base import InternalData, MarketData
from algo_engine.exchange_profile import ExchangeProfile

MDS: MarketDataService


class MarketDataMonitor(metaclass=abc.ABCMeta):
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
    def value(self) -> float | dict[str, float]:
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


class MonitorManager(object):
    """
    Manages market data monitors.

    This is a basic, single-threaded implementation. It is typically used as a component of MDS.

    Attributes:
        monitor (dict[Any, MarketDataMonitor]): Registered monitors.
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
    def values(self) -> dict[str, float]:
        """
        Collect current values from all monitors.

        Returns:
            A dictionary mapping monitor names to their values.
        """


class MarketDataService:
    """
    Singleton service for managing market data.

    Handles distribution, storage, and monitor coordination.
    All market data–related operations should be registered here.

    Attributes:
        profile (ExchangeProfile): The active market profile.
        n_subscribed (int): Number of tickers currently subscribed.
        monitor (dict): Direct reference to monitor storage.
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

    def __getitem__(self, monitor_id: Any) -> MarketDataMonitor:
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

    def set_manager(self, manager: MonitorManager) -> None:
        """
        Replace the monitor manager, re-registering all existing monitors.

        Args:
            manager (MonitorManager): The new manager to assign.
        """

    def get_market_price(self, ticker: str) -> float:
        """
        Retrieve the latest price of a ticker.

        Args:
            ticker (str): The instrument's symbol.

        Returns:
            float: Latest price or NaN if unavailable.
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
            monitor: MarketDataMonitor = None,
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
        Latest known market prices.

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
    def subscriptions(self) -> dict[str, int]:
        """
        Current subscription mapping.

        Returns:
            A dictionary mapping ticker symbols to their subscription indices.
        """
        ...

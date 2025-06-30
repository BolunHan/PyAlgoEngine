import abc
from datetime import datetime, date

from algo_engine.base import MarketData, InternalData
from algo_engine.profile import Profile

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
        monitor_id (str): Unique identifier, auto-generated if not provided.
        enabled (bool): Whether the monitor is active (default: True).
    """
    name: str
    monitor_id: str
    enabled: bool

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
        monitor (dict[str, MarketDataMonitor]): Registered monitors.
    """
    monitor: dict[str, MarketDataMonitor]

    def __call__(self, market_data: MarketData):
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

    def pop_monitor(self, monitor_id: str) -> None:
        """
        Unregister a monitor by ID.

        Args:
            monitor_id (str): ID of the monitor to remove.
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
        profile (Profile): The active market profile.
        max_subscription (int): Maximum number of subscriptions allowed.
    """
    profile: Profile
    max_subscription: int

    def __len__(self) -> int:
        """
        Get number of tickers processed.

        Returns:
            Number of tickers with available data.
        """

    def __call__(self, market_data: MarketData):
        """
        Alias for `on_market_data`.

        Args:
            market_data (MarketData): Incoming data.
        """

    def __getitem__(self, monitor_id: str) -> MarketDataMonitor:
        """
        Access monitor by ID.

        Args:
            monitor_id (str): Monitor's unique ID.

        Returns:
            MarketDataMonitor: The corresponding monitor.
        """

    def on_internal_data(self, internal_data: InternalData) -> None:
        """
        Feed internal communication protocol data to the service.

        Args:
            internal_data (InternalData): The internal protocol message.
        """

    def on_market_data(self, market_data: MarketData) -> None:
        """
        Handle market data update and trigger monitors.

        Args:
            market_data (MarketData): Incoming data.
        """

    def get_market_price(self, ticker: str) -> float:
        """
        Retrieve the latest price of a ticker.

        Args:
            ticker (str): The instrument's symbol.

        Returns:
            float: Latest price or NaN if unavailable.
        """

    def add_monitor(self, monitor: MarketDataMonitor, **kwargs) -> None:
        """
        Register a monitor with the service.

        Note:
            MDS and its `MonitorManager` store monitors independently.

        Args:
            monitor (MarketDataMonitor): The monitor to add.
        """

    def pop_monitor(self, monitor_id: str, **kwargs) -> None:
        """
        Remove a monitor from the service.

        Args:
            monitor_id (str): The ID of the monitor to remove.
        """

    def clear(self):
        """
        Reset the service to its initial state.
        """

    @property
    def market_price(self) -> dict[str, float]:
        """
        Latest known market prices.

        Returns:
            A dictionary mapping tickers to prices.
        """

    @property
    def market_time(self) -> datetime | None:
        """
        Latest timestamp of received market data.

        Returns:
            A `datetime` object or None.
        """

    @property
    def market_date(self) -> date | None:
        """
        Date component of the latest market data.

        Returns:
            A `date` object or None.
        """

    @property
    def timestamp(self) -> float | None:
        """
        UNIX timestamp of the latest market data.

        Returns:
            A float or None.
        """

    @property
    def monitor(self) -> dict[str, MarketDataMonitor]:
        """
        Direct reference to monitor storage.

        Returns:
            The monitor registry. Caution: not a copy.
        """

    @property
    def monitor_manager(self) -> MonitorManager:
        """
        The monitor manager in use.

        Returns:
            The current `MonitorManager` instance.
        """

    @monitor_manager.setter
    def monitor_manager(self, manager: MonitorManager) -> None:
        """
        Set a new monitor manager.

        This will re-register all existing monitors with the new manager.

        Args:
            manager (MonitorManager): The new manager to assign.
        """

import datetime
import unittest
import uuid

from algo_engine.base import FilterMode, TransactionData, TransactionSide
from algo_engine.engine import (
    MarketDataMonitor,
    MarketDataMonitorProtocol,
    MarketDataService,
    MonitorManager,
)
from algo_engine.exchange_profile import PROFILE_CN


class TestMarketDataMonitorHierarchy(unittest.TestCase):
    """The CCP-bound MarketDataMonitor base and its protocol registration."""

    @classmethod
    def setUpClass(cls) -> None:
        PROFILE_CN.activate()

        cls.trade = TransactionData(
            ticker='MONITOR.HIERARCHY',
            price=12.5,
            volume=3,
            timestamp=datetime.datetime(2024, 11, 11, 10, 0).timestamp(),
            side=TransactionSide.LongFilled,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        PROFILE_CN.deactivate()

    def test_00_is_a_virtual_protocol_subclass(self) -> None:
        self.assertTrue(issubclass(MarketDataMonitor, MarketDataMonitorProtocol))
        self.assertTrue(isinstance(MarketDataMonitor('Monitor.Protocol'), MarketDataMonitorProtocol))

    def test_01_base_defaults_and_properties(self) -> None:
        monitor = MarketDataMonitor('Monitor.Defaults')

        self.assertEqual(monitor.name, 'Monitor.Defaults')
        self.assertIsInstance(monitor.monitor_id, uuid.UUID)
        self.assertTrue(monitor.enabled)
        self.assertTrue(monitor.is_ready)
        self.assertEqual(monitor.priority, 0)
        self.assertIsInstance(monitor.md_filter, FilterMode)
        self.assertIsNone(monitor.value)

        monitor.enabled = False
        monitor.is_ready = False
        monitor.priority = 7
        self.assertFalse(monitor.enabled)
        self.assertFalse(monitor.is_ready)
        self.assertEqual(monitor.priority, 7)

    def test_02_python_subclasses_keep_their_attributes(self) -> None:
        class CountingMonitor(MarketDataMonitor):
            def __init__(self, name: str):
                super().__init__(name)
                self.calls = 0
                self.extra = 'kept'

            def __call__(self, market_data):
                self.calls += 1

            def clear(self):
                self.calls = 0

            @property
            def value(self):
                return {'calls': self.calls}

        monitor = CountingMonitor('Monitor.Counting')

        manager = MonitorManager()
        manager.add_monitor(monitor)

        self.assertEqual(manager.n_monitors, 1)
        self.assertEqual(monitor.extra, 'kept')
        self.assertEqual(monitor.value, {'calls': 0})

        manager.pop_monitor(monitor.monitor_id)
        self.assertEqual(manager.n_monitors, 0)

    def test_03_service_registry_and_dispatch(self) -> None:
        class CountingMonitor(MarketDataMonitor):
            def __init__(self, name: str):
                super().__init__(name)
                self.calls = 0

            def __call__(self, market_data):
                self.calls += 1

        monitor = CountingMonitor('Monitor.Dispatch')
        service = MarketDataService()

        service.add_monitor(monitor)
        self.assertEqual(service.n_monitors, 1)
        self.assertEqual(service.monitor_manager.n_monitors, 1)

        service.on_market_data(self.trade)
        self.assertEqual(monitor.calls, 1)

        service.pop_monitor(monitor=monitor)
        self.assertEqual(service.n_monitors, 0)
        self.assertEqual(service.monitor_manager.n_monitors, 0)

        service.clear()

    def test_04_manager_replacement_keeps_registry_in_sync(self) -> None:
        monitor = MarketDataMonitor('Monitor.Replace')
        service = MarketDataService()

        service.add_monitor(monitor)

        replacement = MonitorManager()
        service.set_manager(replacement)

        self.assertIs(service.monitor_manager, replacement)
        self.assertEqual(service.n_monitors, 1)
        self.assertEqual(service.monitor_manager.n_monitors, 1)

        service.clear()
        self.assertEqual(service.n_monitors, 0)
        self.assertEqual(service.monitor_manager.n_monitors, 0)


if __name__ == '__main__':
    unittest.main()

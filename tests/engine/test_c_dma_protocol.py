import time
import unittest

from algo_engine.base import MD_FREELIST, MD_SHARED
from algo_engine.base.c_market_data.c_tick import TickData
from algo_engine.base.c_market_data.c_trade_utils import TradeInstruction, TradeReport
from algo_engine.base.c_market_data.c_transaction import OrderData, TransactionSide
from algo_engine.engine import DMAProtocol, EVENT_ENGINE, TOPIC_SET

SIDE = TransactionSide.SIDE_BID


def _wait_until(predicate, timeout=3.0):
    """Polling wait for the engine loop to deliver a queued event."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


def _tick_snapshot(md):
    """Copy the values a listener needs while the borrowed view is valid."""
    return type(md).__name__, md.ticker, md.last_price


def _order_snapshot(md):
    return type(md).__name__, md.ticker, md.price


def _report_snapshot(md):
    return type(md).__name__, md.ticker, md.price, md.volume


def _instruction_snapshot(md):
    return type(md).__name__, md.ticker, md.volume


class TestDMAProtocol(unittest.TestCase):
    def setUp(self):
        self.engine = EVENT_ENGINE
        self.protocol = DMAProtocol()

        if self.engine.active:
            self.engine.stop()
        self.engine.start()
        self.addCleanup(self._cleanup)

    def _cleanup(self):
        if self.engine.active:
            self.engine.stop()
        self.protocol = None
        self.engine.clear()

    # --- Market data round trips -----------------------------------------

    def test_send_and_receive_from_shm(self):
        received = []
        topic = TOPIC_SET.push(TickData(ticker='DPROTO1', timestamp=0.0, last_price=1.0))
        self.protocol.register_callback(lambda md: received.append(_tick_snapshot(md)), topic)

        md = TickData(ticker='DPROTO1', timestamp=1.0, last_price=321.5)
        self.assertTrue(md.data_addr)
        self.protocol.send(md)

        # The transport copies even an in-shm source; the wrapper stays bound.
        self.assertNotEqual(md.data_addr, 0)

        self.assertTrue(_wait_until(lambda: received))
        time.sleep(0.05)

        self.assertEqual(received, [('TickData', 'DPROTO1', 321.5)])
        self.assertEqual(md.last_price, 321.5)

    def test_send_and_receive_from_heap(self):
        received = []
        topic = TOPIC_SET.push(TickData(ticker='DPROTO2', timestamp=0.0, last_price=1.0))
        self.protocol.register_callback(lambda md: received.append(_tick_snapshot(md)), topic)

        with ~MD_SHARED | ~MD_FREELIST:
            md = TickData(ticker='DPROTO2', timestamp=1.0, last_price=12.25)

        self.protocol.send(md)

        # A non-shm source is copied; the caller keeps it.
        self.assertNotEqual(md.data_addr, 0)

        self.assertTrue(_wait_until(lambda: received))
        time.sleep(0.05)

        self.assertEqual(received, [('TickData', 'DPROTO2', 12.25)])

    def test_received_view_is_nullified_when_the_payload_is_released(self):
        # The listener did not send this payload, so it owns a borrowed view.
        listener = DMAProtocol()
        received = []
        seed = TickData(ticker='DPROTO16', timestamp=1.0, last_price=1.0)
        topic = TOPIC_SET.push(seed)
        listener.register_callback(received.append, topic)

        sender = DMAProtocol()
        md = TickData(ticker='DPROTO16', timestamp=2.0, last_price=2.0)
        sender.send(md)

        self.assertTrue(_wait_until(lambda: received))
        time.sleep(0.05)

        self.assertIsNot(received[0], md)
        self.assertEqual(received[0].data_addr, 0)
        self.assertIsNone(received[0].address)

    def test_generic_realtime_registration_receives_exact_topics(self):
        # A generic registration is its own handler; the engine matches it
        # against the exact payload topics.
        received = []
        self.protocol.register_callback(lambda md: received.append(md.ticker), TOPIC_SET.realtime)

        self.protocol.send(TickData(ticker='DPROTO20', timestamp=1.0, last_price=1.0))
        self.assertTrue(_wait_until(lambda: received))

        # The generic realtime registration must not catch another family.
        self.protocol.send(TradeReport(ticker='DPROTO20', timestamp=1.0, price=1.0, volume=1.0, side=SIDE))
        time.sleep(0.1)

        self.assertEqual(received, ['DPROTO20'])

    def test_send_order_routes_to_realtime(self):
        received = []
        seed = OrderData(ticker='DPROTO3', timestamp=0.0, price=1.0, volume=1.0, side=SIDE)
        topic = TOPIC_SET.push(seed)
        self.protocol.register_callback(lambda md: received.append(_order_snapshot(md)), topic)

        order = OrderData(ticker='DPROTO3', timestamp=1.0, price=5.25, volume=1.0, side=SIDE)
        self.protocol.send(order)

        self.assertTrue(_wait_until(lambda: received))
        time.sleep(0.05)

        self.assertEqual(received, [('OrderData', 'DPROTO3', 5.25)])

    def test_send_report_routes_to_on_report(self):
        received = []
        self.protocol.register_callback(lambda md: received.append(_report_snapshot(md)), TOPIC_SET.on_report)

        report = TradeReport(ticker='DPROTO4', timestamp=1.0, price=10.0, volume=2.0, side=SIDE)
        self.protocol.send(report)

        self.assertTrue(_wait_until(lambda: received))
        time.sleep(0.05)

        self.assertEqual(received, [('TradeReport', 'DPROTO4', 10.0, 2.0)])

    def test_send_instruction_routes_to_on_order(self):
        received = []
        self.protocol.register_callback(lambda md: received.append(_instruction_snapshot(md)), TOPIC_SET.on_order)

        instruction = TradeInstruction(ticker='DPROTO5', timestamp=1.0, side=SIDE, volume=4.5)
        self.protocol.send(instruction)

        self.assertTrue(_wait_until(lambda: received))
        time.sleep(0.05)

        self.assertEqual(received, [('TradeInstruction', 'DPROTO5', 4.5)])

    # --- Topic routing ----------------------------------------------------

    def test_register_callback_filters_by_exact_realtime_topic(self):
        received = []
        seed = TickData(ticker='DPROTO7', timestamp=1.0, last_price=1.0)
        topic = TOPIC_SET.push(seed)
        self.protocol.register_callback(lambda md: received.append(md.ticker), topic)

        md7 = TickData(ticker='DPROTO7', timestamp=2.0, last_price=2.0)
        md8 = TickData(ticker='DPROTO8', timestamp=2.0, last_price=3.0)

        self.protocol.send(md7)
        self.assertTrue(_wait_until(lambda: received))
        self.protocol.send(md8)

        time.sleep(0.1)
        self.assertEqual(received, ['DPROTO7'])

    def test_duplicate_register_callback_returns_duplicated(self):
        received = []
        seed = TickData(ticker='DPROTO9', timestamp=1.0, last_price=1.0)
        topic = TOPIC_SET.push(seed)
        callback = lambda md: received.append(md.ticker)

        self.assertEqual(self.protocol.register_callback(callback, topic), 0)
        self.assertEqual(self.protocol.register_callback(callback, topic), -9)

        md = TickData(ticker='DPROTO9', timestamp=2.0, last_price=2.0)
        self.protocol.send(md)

        self.assertTrue(_wait_until(lambda: received))
        time.sleep(0.05)

        self.assertEqual(received, ['DPROTO9'])

    def test_multiple_callbacks_on_one_topic(self):
        first = []
        second = []
        seed = TickData(ticker='DPROTO10', timestamp=1.0, last_price=1.0)
        topic = TOPIC_SET.push(seed)
        self.protocol.register_callback(lambda md: first.append(md.ticker), topic)
        self.protocol.register_callback(lambda md: second.append(md.ticker), topic)

        md = TickData(ticker='DPROTO10', timestamp=2.0, last_price=2.0)
        self.protocol.send(md)

        self.assertTrue(_wait_until(lambda: first and second))
        time.sleep(0.05)

        self.assertEqual(first, ['DPROTO10'])
        self.assertEqual(second, ['DPROTO10'])

    def test_callbacks_are_isolated_per_topic(self):
        ticks = []
        reports = []
        seed = TickData(ticker='DPROTO17', timestamp=1.0, last_price=1.0)
        topic = TOPIC_SET.push(seed)
        self.protocol.register_callback(lambda md: ticks.append(md.ticker), topic)
        self.protocol.register_callback(lambda md: reports.append(md.ticker), TOPIC_SET.on_report)

        self.protocol.send(TickData(ticker='DPROTO17', timestamp=2.0, last_price=2.0))
        self.assertTrue(_wait_until(lambda: ticks))
        self.protocol.send(TradeReport(ticker='DPROTO18', timestamp=1.0, price=1.0, volume=1.0, side=SIDE))
        self.assertTrue(_wait_until(lambda: reports))
        time.sleep(0.05)

        self.assertEqual(ticks, ['DPROTO17'])
        self.assertEqual(reports, ['DPROTO18'])

    def test_unregister_callback_stops_delivery(self):
        received = []

        def callback(md):
            received.append(md.ticker)

        seed = TickData(ticker='DPROTO11', timestamp=1.0, last_price=1.0)
        topic = TOPIC_SET.push(seed)
        self.protocol.register_callback(callback, topic)

        first = TickData(ticker='DPROTO11', timestamp=2.0, last_price=2.0)
        self.protocol.send(first)
        self.assertTrue(_wait_until(lambda: received))

        self.protocol.unregister_callback(callback, topic)
        second = TickData(ticker='DPROTO11', timestamp=3.0, last_price=3.0)
        self.protocol.send(second)

        time.sleep(0.1)
        self.assertEqual(received, ['DPROTO11'])

    # --- Surface ----------------------------------------------------------

    def test_address(self):
        self.assertGreater(self.protocol.address, 0)

    def test_register_callback_rejects_non_callable(self):
        with self.assertRaises(TypeError):
            self.protocol.register_callback(42, TOPIC_SET.on_report)

    def test_register_callback_rejects_non_topic(self):
        with self.assertRaises(TypeError):
            self.protocol.register_callback(lambda md: None, 'on_report')

    def test_send_rejects_non_market_data(self):
        with self.assertRaises(TypeError):
            self.protocol.send('not a MarketData')

    def test_send_keeps_the_source_reusable(self):
        received = []
        topic = TOPIC_SET.push(TickData(ticker='DPROTO12', timestamp=0.0, last_price=1.0))
        self.protocol.register_callback(lambda md: received.append(md.last_price), topic)

        md = TickData(ticker='DPROTO12', timestamp=1.0, last_price=1.0)

        self.protocol.send(md)
        self.assertTrue(_wait_until(lambda: received))
        time.sleep(0.05)

        # The transport copied; the same source can be sent again.
        self.protocol.send(md)
        self.assertTrue(_wait_until(lambda: len(received) == 2))
        time.sleep(0.05)

        self.assertEqual(received, [1.0, 1.0])
        self.assertNotEqual(md.data_addr, 0)

    def test_config(self):
        self.protocol.config(False, 64, 0.5)
        self.protocol.config(True)

    def test_send_is_busy_while_the_static_payload_is_in_flight(self):
        received = []
        topic = TOPIC_SET.push(TickData(ticker='DPROTO13', timestamp=0.0, last_price=1.0))
        self.protocol.register_callback(lambda md: received.append(md.ticker), topic)

        self.engine.stop()

        first = TickData(ticker='DPROTO13', timestamp=1.0, last_price=1.0)
        second = TickData(ticker='DPROTO13', timestamp=1.0, last_price=2.0)
        self.assertEqual(self.protocol.send(first), 0)
        self.assertEqual(self.protocol.send(second), -8)

        self.engine.start()
        self.assertTrue(_wait_until(lambda: received))
        self.assertEqual(received, ['DPROTO13'])


if __name__ == '__main__':
    unittest.main()

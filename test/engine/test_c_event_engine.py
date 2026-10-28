import time
import unittest

from event_engine.capi import Topic

from algo_engine.base import MD_FREELIST, MD_SHARED
from algo_engine.base.c_market_data.c_candlestick import BarData
from algo_engine.base.c_market_data.c_internal import InternalData
from algo_engine.base.c_market_data.c_tick import TickData, TickDataLite
from algo_engine.base.c_market_data.c_trade_utils import TradeInstruction, TradeReport
from algo_engine.base.c_market_data.c_transaction import OrderData, TransactionData, TransactionSide
from algo_engine.engine import c_event_engine as ee

# (attribute name, full topic literal, is_exact)
BUILTIN_TOPICS = [
    ('on_order', 'on_order', True),
    ('on_report', 'on_report', True),
    ('eod', 'eod', True),
    ('eod_done', 'eod_done', True),
    ('bod', 'bod', True),
    ('bod_done', 'bod_done', True),
    ('launch_order', 'launch_order.{ticker}', False),
    ('cancel_order', 'cancel_order.{ticker}', False),
    ('realtime', 'realtime.{ticker}.{dtype}', False),
]

EXACT_STEMS = ('on_order', 'on_report', 'eod', 'eod_done', 'bod', 'bod_done')

SIDE = TransactionSide.SIDE_BID


def _md_objects(ticker: str = 'RB2510'):
    """One concrete MarketData instance per dtype, in a no-shm context."""
    return [
        InternalData(ticker=ticker, timestamp=1.0, code=1),
        TransactionData(ticker=ticker, timestamp=1.0, price=1.0, volume=1.0, side=SIDE),
        OrderData(ticker=ticker, timestamp=1.0, price=1.0, volume=1.0, side=SIDE),
        TickDataLite(ticker=ticker, timestamp=1.0, last_price=1.0, bid_price=1.0, bid_volume=1.0, ask_price=1.0, ask_volume=1.0),
        TickData(ticker=ticker, timestamp=1.0, last_price=1.0),
        BarData(ticker=ticker, timestamp=1.0, high_price=1.0, low_price=1.0, open_price=1.0, close_price=1.0, bar_span=60.0),
        TradeReport(ticker=ticker, timestamp=1.0, price=1.0, volume=1.0, side=SIDE),
        TradeInstruction(ticker=ticker, timestamp=1.0, side=SIDE, volume=1.0),
    ]


class TestTopicSetExCore(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.topic = ee.TOPIC_SET

    # --- Globals ---------------------------------------------------------

    def test_module_globals(self):
        # TopicSetEx is the default and TOPIC no longer exists.
        self.assertFalse(hasattr(ee, 'TOPIC'))
        self.assertIsInstance(ee.TOPIC_SET, ee.TopicSetEx)
        self.assertGreater(ee.TOPIC_SET.address, 0)

    def test_legacy_set_still_exists_and_shares_topics(self):
        self.assertIsNot(ee.TopicSet, ee.TopicSetEx)

        legacy = ee.TopicSet()
        self.assertIsInstance(legacy, ee.TopicSet)
        # The legacy set's topics are views over the C set's topics.
        self.assertEqual(legacy.on_order.addr, ee.TOPIC_SET.on_order.addr)
        self.assertEqual(legacy.realtime.addr, ee.TOPIC_SET.realtime.addr)

    # --- Built-ins -------------------------------------------------------

    def test_builtin_topics(self):
        for attr, expected, is_exact in BUILTIN_TOPICS:
            with self.subTest(attr=attr):
                topic = getattr(self.topic, attr)

                self.assertIsInstance(topic, Topic)
                self.assertEqual(topic.value, expected)
                self.assertEqual(topic.is_exact, is_exact)
                self.assertEqual(self.topic.get(expected).addr, topic.addr)

    def test_builtin_keys_carry_no_variant_suffix(self):
        for attr, expected, _ in BUILTIN_TOPICS:
            with self.subTest(attr=attr):
                self.assertFalse(expected.endswith('.c'))
                self.assertFalse(expected.endswith('.py'))

    def test_builtin_exact_topics_do_not_match(self):
        exact = [getattr(self.topic, stem) for stem in EXACT_STEMS]

        for i, topic in enumerate(exact):
            with self.subTest(stem=EXACT_STEMS[i]):
                self.assertFalse(bool(topic.match(exact[(i + 1) % len(exact)])))

    # --- Templates -------------------------------------------------------

    def test_template_format(self):
        launch = self.topic.launch_order.format(ticker='RB2510')
        self.assertEqual(launch.value, 'launch_order.RB2510')
        self.assertTrue(launch.is_exact)
        self.assertTrue(bool(self.topic.launch_order.match(launch)))

        realtime = self.topic.realtime.format(ticker='RB2510', dtype='TickData')
        self.assertEqual(realtime.value, 'realtime.RB2510.TickData')
        self.assertEqual(len(realtime), 3)
        self.assertTrue(bool(self.topic.realtime.match(realtime)))

        # The composition is the legacy one: no .c / .py variant is appended.
        self.assertNotIn('.c', launch.value)
        self.assertNotIn('.py', launch.value)
        self.assertEqual(realtime.value, ee.TOPIC_SET.realtime.format(ticker='RB2510', dtype='TickData').value)

    # --- Push / parse ----------------------------------------------------

    def test_push_parse_every_dtype(self):
        with ~MD_SHARED | ~MD_FREELIST:
            market_data = _md_objects()

        for md in market_data:
            dtype_name = type(md).__name__
            with self.subTest(dtype=dtype_name):
                topic = self.topic.push(md)
                expected = f'realtime.{md.ticker}.{dtype_name}'

                self.assertEqual(topic.value, expected)
                self.assertTrue(topic.is_exact)
                self.assertEqual(len(topic), 3)
                self.assertTrue(bool(self.topic.realtime.match(topic)))

                # parse() round-trips ticker and dtype.
                self.assertEqual(self.topic.parse(topic), (md.ticker, dtype_name))
                self.assertEqual(self.topic.parse_map(topic), {'ticker': md.ticker, 'dtype': dtype_name})

                # Repeated pushes return the identical topic.
                self.assertEqual(self.topic.push(md).addr, topic.addr)

    def test_same_ticker_different_dtype(self):
        with ~MD_SHARED | ~MD_FREELIST:
            tick = TickData(ticker='RB2510', timestamp=1.0, last_price=1.0)
            trade = TransactionData(ticker='RB2510', timestamp=1.0, price=1.0, volume=1.0, side=SIDE)
            order = OrderData(ticker='RB2510', timestamp=1.0, price=1.0, volume=1.0, side=SIDE)

        topics = [self.topic.push(md) for md in (tick, trade, order)]

        # Each dtype resolves to its own topic.
        self.assertEqual(len({topic.addr for topic in topics}), len(topics))

        for topic, md in zip(topics, (tick, trade, order)):
            with self.subTest(dtype=type(md).__name__):
                self.assertEqual(self.topic.parse(topic), ('RB2510', type(md).__name__))
                self.assertEqual(self.topic.push(md).addr, topic.addr)

    def test_push_dotted_ticker_is_a_single_part(self):
        with ~MD_SHARED | ~MD_FREELIST:
            md = InternalData(ticker='600010.SH', timestamp=1.0, code=1)

        topic = self.topic.push(md)
        self.assertEqual(topic.value, 'realtime.600010.SH.InternalData')
        self.assertEqual(len(topic), 3)
        self.assertEqual(topic[1].part, '600010.SH')
        self.assertEqual(topic[2].part, 'InternalData')
        self.assertEqual(self.topic.parse(topic), ('600010.SH', 'InternalData'))

    def test_push_long_ticker(self):
        ticker = 'L' * 100
        with ~MD_SHARED | ~MD_FREELIST:
            md = TickData(ticker=ticker, timestamp=1.0, last_price=1.0)

        topic = self.topic.push(md)
        self.assertEqual(topic.value, f'realtime.{ticker}.TickData')
        self.assertEqual(self.topic.parse(topic), (ticker, 'TickData'))

    # --- Dynamic registration -------------------------------------------

    def test_dynamic_registration(self):
        name = 'custom.alpha'
        topic = self.topic.register(name)
        self.assertEqual(topic.addr, self.topic.get(name).addr)
        self.assertEqual(self.topic.register(name).addr, topic.addr)

        # get() is a get-or-set: a missing literal is registered on first use.
        missing = 'custom.not.registered'
        lazy = self.topic.get(missing)
        self.assertEqual(lazy.value, missing)
        self.assertEqual(self.topic.get(missing).addr, lazy.addr)

        template = self.topic.register('custom.order.{ticker}')
        exact = self.topic.register('custom.order.RB2510')
        self.assertFalse(template.is_exact)
        self.assertTrue(exact.is_exact)
        self.assertTrue(bool(template.match(exact)))

    def test_registration_fills_the_ticker_cache(self):
        with ~MD_SHARED | ~MD_FREELIST:
            tick = TickData(ticker='SYNCPY1', timestamp=1.0, last_price=1.0)
            trade = TransactionData(ticker='SYNCPY2', timestamp=1.0, price=1.0, volume=1.0, side=SIDE)
            bar = BarData(ticker='SYNCPY3', timestamp=1.0, high_price=1.0, low_price=1.0, open_price=1.0, close_price=1.0, bar_span=60.0)

        # Registering a realtime literal fills that dtype's ticker cache, so
        # push() serves the registered topic without a registry lookup.
        tick_topic = self.topic.get('realtime.SYNCPY1.TickData')
        self.assertEqual(self.topic.push(tick).addr, tick_topic.addr)

        trade_topic = self.topic.get('realtime.SYNCPY2.TransactionData')
        self.assertEqual(self.topic.push(trade).addr, trade_topic.addr)

        # A dtype without a dedicated cache resolves through the registry.
        bar_topic = self.topic.get('realtime.SYNCPY3.BarData')
        self.assertEqual(self.topic.push(bar).addr, bar_topic.addr)

    # --- Errors ----------------------------------------------------------

    def test_parse_errors(self):
        with self.assertRaises(ValueError):
            self.topic.parse(self.topic.on_order)

        with self.assertRaises(ValueError):
            self.topic.parse(self.topic.realtime)

        unknown = self.topic.register('realtime.RB2510.NoSuchDtype')
        with self.assertRaises(ValueError):
            self.topic.parse(unknown)

        # The old suffixed composition is a four-part topic, not a realtime one.
        suffixed = self.topic.register('realtime.RB2510.TickData.c')
        with self.assertRaises(ValueError):
            self.topic.parse(suffixed)

    # --- Legacy cross-check ---------------------------------------------

    def test_parse_matches_legacy_topic_set(self):
        legacy = ee.TopicSet()

        with ~MD_SHARED | ~MD_FREELIST:
            market_data = _md_objects(ticker='600010.SH')

        for md in market_data:
            with self.subTest(dtype=type(md).__name__):
                legacy_topic = legacy.push(md)
                new_topic = self.topic.push(md)

                # Identical composition, character for character.
                self.assertEqual(new_topic.value, legacy_topic.value)
                self.assertEqual(
                    self.topic.parse_map(new_topic),
                    legacy.parse(legacy_topic),
                )

    def test_legacy_push_caches_per_dtype(self):
        legacy = ee.TopicSet()

        with ~MD_SHARED | ~MD_FREELIST:
            md = TickData(ticker='LEGACY1', timestamp=1.0, last_price=1.0)

        topic = legacy.push(md)
        self.assertEqual(topic.value, 'realtime.LEGACY1.TickData')
        self.assertIs(legacy.push(md), topic)
        self.assertEqual(legacy.parse(topic), {'ticker': 'LEGACY1', 'dtype': 'TickData'})

    # --- Engine routing --------------------------------------------------

    def _start_engine(self):
        engine = ee.EVENT_ENGINE
        self.assertFalse(engine.active)
        engine.start()
        self.addCleanup(self._stop_engine)

    def _stop_engine(self):
        engine = ee.EVENT_ENGINE
        if engine.active:
            engine.stop()
        engine.clear()

    def test_topic_routing(self):
        with ~MD_SHARED | ~MD_FREELIST:
            md = TickData(ticker='RB2510', timestamp=1.0, last_price=1.0)

        received = []

        def handler(value, topic=None):
            received.append((value, topic.value if topic is not None else None))

        self._start_engine()
        engine = ee.EVENT_ENGINE
        engine.register_handler(self.topic.realtime, handler)

        topic = self.topic.push(md)
        engine.publish(topic, ('event',), {})

        deadline = time.monotonic() + 2.0
        while not received and time.monotonic() < deadline:
            time.sleep(0.005)

        self.assertEqual(received, [('event', topic.value)])

        # A second delivery has time to show up, then asserts none arrived.
        time.sleep(0.05)
        self.assertEqual(len(received), 1)


class TestTopicSetExSurface(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.topic = ee.TOPIC_SET

    # --- __getitem__ / __contains__ --------------------------------------

    def test_getitem_is_get_set(self):
        self.assertEqual(self.topic['on_order'].addr, self.topic.on_order.addr)

        created = self.topic['item.one']
        self.assertEqual(created.value, 'item.one')
        self.assertIn('item.one', self.topic)
        self.assertEqual(self.topic['item.one'].addr, created.addr)

        with self.assertRaises(TypeError):
            self.topic[42]

    def test_contains(self):
        self.assertIn('on_order', self.topic)
        self.assertNotIn('never.registered.literal', self.topic)
        self.assertNotIn(42, self.topic)
        self.assertNotIn(None, self.topic)

    # --- __getattr__ / __dict__ ------------------------------------------

    def test_getattr_falls_back_to_the_registry(self):
        # A declared attribute always wins over the fallback.
        self.assertEqual(self.topic.on_order.value, 'on_order')

        self.topic.register('attr_literal')
        self.assertEqual(self.topic.attr_literal.value, 'attr_literal')

        with self.assertRaises(AttributeError):
            self.topic.not_a_topic

    def test_instance_dict(self):
        self.assertIsInstance(self.topic.__dict__, dict)

        self.topic.tag_payload = {'a': 1}
        self.assertEqual(self.topic.tag_payload, {'a': 1})
        self.assertIn('tag_payload', self.topic.__dict__)

        del self.topic.tag_payload
        with self.assertRaises(AttributeError):
            self.topic.tag_payload

    # --- __or__ ----------------------------------------------------------

    def test_or_merges_into_a_set_that_owns_nothing(self):
        registered = self.topic.register('realtime.XMERGE.TickData')

        merged = self.topic | self.topic

        self.assertIs(type(merged), ee.TopicSetEx)
        self.assertIsNot(merged, self.topic)
        self.assertIn('on_order', merged)
        self.assertIn('realtime.XMERGE.TickData', merged)
        self.assertEqual(merged['on_order'].addr, self.topic.on_order.addr)
        self.assertEqual(merged['realtime.XMERGE.TickData'].addr, registered.addr)
        self.assertEqual(merged.on_order.value, 'on_order')

        # Freeing the merged set must not free the source's topics.
        del merged
        self.assertEqual(self.topic.on_order.value, 'on_order')
        self.assertIn('on_order', self.topic)

    def test_merged_set_is_read_only(self):
        merged = self.topic | self.topic

        with self.assertRaises(KeyError):
            merged['never.registered.literal']
        self.assertNotIn('never.registered.literal', merged)

    def test_or_of_a_merged_set(self):
        merged = (self.topic | self.topic) | self.topic
        self.assertIn('on_order', merged)
        self.assertEqual(merged['on_order'].addr, self.topic.on_order.addr)

    def test_or_rejects_foreign_operands(self):
        with self.assertRaises(TypeError):
            self.topic | 3
        with self.assertRaises(TypeError):
            self.topic | 'on_order'


if __name__ == '__main__':
    unittest.main()

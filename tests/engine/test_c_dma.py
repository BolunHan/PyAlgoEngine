import datetime
import json
import time
import unittest

from algo_engine.base import (
    TradeInstruction,
    TradeReport,
    TransactionData,
    TransactionDirection as Direction,
    TransactionOffset as Offset,
    TransactionSide,
)
from algo_engine.engine import (
    MDS,
    Balance,
    DirectMarketAccess,
    DirectMarketAccessProtocol,
    Inventory,
    PositionManagementService,
    RiskProfile,
    Singleton,
)
from algo_engine.exchange_profile import PROFILE_CN
from algo_engine.strategy import BALANCE, DMA, INVENTORY, RISK_PROFILE


TICKER = '600000.SH'
SIDE_LONG = Direction.DIRECTION_LONG | Offset.OFFSET_OPEN
SIDE_SHORT = Direction.DIRECTION_SHORT | Offset.OFFSET_OPEN
TS = datetime.datetime(2024, 11, 11, 10, 0).timestamp()


def setup_profile_rules(risk_profile: RiskProfile, ticker: str = TICKER) -> None:
    risk_profile.set_rule('max_percentile', float('inf'), ticker=ticker)

    for key in ('max_trade_long', 'max_trade_short', 'max_exposure_long', 'max_exposure_short'):
        risk_profile.set_rule(key, 1000., ticker=ticker)

    for key in ('max_notional_long', 'max_notional_short'):
        risk_profile.set_rule(key, 1e9, ticker=ticker)

    for key in ('max_ttl_notional_long', 'max_ttl_notional_short', 'max_net_notional_long', 'max_net_notional_short'):
        risk_profile.set_rule(key, float('inf'))


def feed_price(ticker: str = TICKER, price: float = 10.5) -> None:
    MDS.on_market_data(
        TransactionData(
            ticker=ticker,
            price=price,
            volume=100,
            timestamp=TS,
            side=TransactionSide.SIDE_LONG_OPEN,
        )
    )


def make_order(volume: float = 10., side=SIDE_LONG, ticker: str = TICKER) -> TradeInstruction:
    return TradeInstruction(ticker=ticker, side=side, volume=volume, timestamp=TS)


class TestRiskProfile(unittest.TestCase):
    """The C-backed risk profile: complete rules are mandatory and fail closed."""

    @classmethod
    def setUpClass(cls) -> None:
        PROFILE_CN.activate()
        feed_price()

    @classmethod
    def tearDownClass(cls) -> None:
        PROFILE_CN.deactivate()

    def setUp(self) -> None:
        RISK_PROFILE.clear()

    def test_00_incomplete_profile_refuses_every_order(self) -> None:
        self.assertFalse(RISK_PROFILE.check(make_order()))
        self.assertEqual(RISK_PROFILE.rules['entry'], set())

        order = make_order()
        self.assertFalse(RISK_PROFILE.check(order))
        self.assertEqual(order.order_state.name, 'STATE_PENDING')

    def test_01_missing_global_rule_still_refuses(self) -> None:
        # clear() leaves the global rules defined, so a profile that has never
        # been cleared is the only one missing them; build one on a fresh
        # singleton and drop it again so the module global stays untouched.
        Singleton._instances.pop(RiskProfile, None)
        self.addCleanup(Singleton._instances.pop, RiskProfile, None)

        risk_profile = RiskProfile(mds=MDS, balance=BALANCE)

        self.assertFalse(risk_profile.check(make_order()))

        for key in (
                'max_percentile', 'max_trade_long', 'max_trade_short',
                'max_exposure_long', 'max_exposure_short',
                'max_notional_long', 'max_notional_short',
        ):
            risk_profile.set_rule(key, float('inf') if key == 'max_percentile' else 1000., ticker=TICKER)

        self.assertFalse(risk_profile.check(make_order()))

        risk_profile.set_rule('max_ttl_notional_long', float('inf'))
        risk_profile.set_rule('max_ttl_notional_short', float('inf'))
        risk_profile.set_rule('max_net_notional_long', float('inf'))
        self.assertFalse(risk_profile.check(make_order()))

        risk_profile.set_rule('max_net_notional_short', float('inf'))
        self.assertTrue(risk_profile.check(make_order()))

    def test_02_complete_profile_passes_and_limits_reject(self) -> None:
        setup_profile_rules(RISK_PROFILE)

        self.assertTrue(RISK_PROFILE.check(make_order(10.)))
        self.assertFalse(RISK_PROFILE.check(make_order(5000.)))

        risk = RISK_PROFILE.get(TICKER)
        self.assertEqual(risk['max_trade_long'], 1000.)
        self.assertEqual(risk['max_ttl_notional_long'], float('inf'))

    def test_03_rules_snapshot_and_json_round_trip(self) -> None:
        setup_profile_rules(RISK_PROFILE)

        rules = RISK_PROFILE.rules
        self.assertIn(TICKER, rules['entry'])
        self.assertEqual(rules['max_trade_long'][TICKER], 1000.)

        payload = RISK_PROFILE.to_json(fmt='dict')
        self.assertEqual(payload['entry'], [TICKER])

        RISK_PROFILE.clear()
        self.assertFalse(RISK_PROFILE.check(make_order()))

        RISK_PROFILE.from_json(json.dumps(payload))
        self.assertTrue(RISK_PROFILE.check(make_order()))

    def test_04_clear_keeps_the_global_rules_defined(self) -> None:
        setup_profile_rules(RISK_PROFILE)
        RISK_PROFILE.clear()

        self.assertEqual(RISK_PROFILE.rules['entry'], set())
        self.assertEqual(RISK_PROFILE.rules['max_net_notional_long'], float('inf'))
        self.assertFalse(RISK_PROFILE.check(make_order()))

    def test_05_rule_scope_is_enforced(self) -> None:
        with self.assertLogs('PyAlgoEngine.Engine', level='ERROR'):
            RISK_PROFILE.set_rule('max_trade_long', 1.)

        with self.assertLogs('PyAlgoEngine.Engine', level='ERROR'):
            RISK_PROFILE.set_rule('max_net_notional_long', 1., ticker=TICKER)

        with self.assertLogs('PyAlgoEngine.Engine', level='ERROR'):
            RISK_PROFILE.set_rule('not_a_rule', 1., ticker=TICKER)


class TestInventory(unittest.TestCase):
    """The C-backed inventory: lots, merging, usage and recalls."""

    def setUp(self) -> None:
        INVENTORY.clear()

    def test_00_entry_merge(self) -> None:
        entry_a = Inventory.Entry(TICKER, 100., 10., Inventory.SecurityType.Stock)
        entry_b = Inventory.Entry(TICKER, 100., 12., Inventory.SecurityType.Stock)

        merged = entry_a + entry_b

        self.assertEqual(merged.volume, 200.)
        self.assertEqual(merged.price, 11.)
        self.assertEqual(merged.notional, 2200.)
        self.assertEqual(merged.ticker, TICKER)
        self.assertEqual(merged.security_type, Inventory.SecurityType.Stock)
        self.assertEqual(merged.direction, Direction.DIRECTION_LONG)

        entry_a.merge(entry_b, inplace=True)
        self.assertEqual(entry_a.volume, 200.)
        self.assertEqual(entry_a.price, 11.)

    def test_01_inventory_lifecycle(self) -> None:
        INVENTORY.add_inv(Inventory.Entry(TICKER, 100., 10., Inventory.SecurityType.Stock))
        INVENTORY.add_inv(Inventory.Entry(TICKER, 100., 12., Inventory.SecurityType.Stock))

        self.assertEqual(INVENTORY.tickers, {TICKER})
        self.assertEqual(INVENTORY.available_volume(TICKER), 200.)
        self.assertEqual(INVENTORY(TICKER), {'Long': 200., 'Short': 0.})

        merged = INVENTORY.get_inv(TICKER)
        self.assertEqual(merged.volume, 200.)
        self.assertEqual(merged.price, 11.)
        self.assertEqual(merged.fee, 0.)
        self.assertIsNone(INVENTORY.get_inv(TICKER, Direction.DIRECTION_SHORT))

        INVENTORY.use_inv(TICKER, 50.)
        self.assertEqual(INVENTORY.available_volume(TICKER), 150.)

        INVENTORY.recall(TICKER, 120.)
        self.assertEqual(INVENTORY.get_inv(TICKER).volume, 100.)
        self.assertEqual(INVENTORY.get_inv(TICKER).recalled, 20.)
        self.assertEqual(INVENTORY.available_volume(TICKER), 30.)

        INVENTORY.clear()
        self.assertEqual(INVENTORY.tickers, set())
        self.assertEqual(INVENTORY.available_volume(TICKER), 0.)

    def test_02_json_round_trip(self) -> None:
        INVENTORY.add_inv(Inventory.Entry(TICKER, 100., 10., Inventory.SecurityType.Stock, fee=1., recalled=5.))

        payload = INVENTORY.to_json(fmt='dict')
        key = f'{TICKER}.DIRECTION_LONG'

        self.assertIn(key, payload)
        self.assertEqual(payload[key]['inv'][0]['fee'], 1.)
        self.assertEqual(payload[key]['inv'][0]['recalled'], 5.)

        INVENTORY.clear()
        INVENTORY.from_json(json.dumps(payload))

        entry = INVENTORY.get_inv(TICKER)
        self.assertEqual(entry.volume, 100.)
        self.assertEqual(entry.fee, 1.)
        self.assertEqual(entry.recalled, 5.)
        self.assertEqual(entry.available, 95.)

    def test_03_inventory_is_a_singleton(self) -> None:
        self.assertIs(Inventory(), INVENTORY)


class TestDirectMarketAccess(unittest.TestCase):
    """The C-backed access: interface dispatch, enabled/risk gating, buffering."""

    @classmethod
    def setUpClass(cls) -> None:
        PROFILE_CN.activate()
        feed_price()

    @classmethod
    def tearDownClass(cls) -> None:
        PROFILE_CN.deactivate()

    class _CountingAccess(DirectMarketAccess):
        def __init__(self, *args, **kwargs):
            self.calls = []
            super().__init__(*args, **kwargs)

        def _launch_order_handler(self, order, **kwargs):
            self.calls.append(('launch', order.order_id, kwargs))

        def _cancel_order_handler(self, order, **kwargs):
            self.calls.append(('cancel', order.order_id, kwargs))

        def _reject_order_handler(self, order, **kwargs):
            self.calls.append(('reject', order.order_id, kwargs))

    def setUp(self) -> None:
        RISK_PROFILE.clear()
        setup_profile_rules(RISK_PROFILE)

    def test_00_is_a_virtual_protocol_subclass(self) -> None:
        self.assertTrue(issubclass(DirectMarketAccess, DirectMarketAccessProtocol))
        self.assertTrue(isinstance(self._CountingAccess(MDS, RISK_PROFILE), DirectMarketAccessProtocol))

    def test_01_disabled_access_rejects(self) -> None:
        access = self._CountingAccess(MDS, RISK_PROFILE)
        order = make_order()

        access.launch_order(order)

        self.assertEqual(order.order_state.name, 'STATE_REJECTED')
        self.assertEqual(access.calls[0][0], 'reject')
        self.assertFalse(access.enabled)

    def test_02_enabled_access_dispatches_the_interface(self) -> None:
        access = self._CountingAccess(MDS, RISK_PROFILE)
        access.start()

        order = make_order()
        access.launch_order(order, tag='payload')

        self.assertTrue(access.enabled)
        self.assertEqual(order.order_state.name, 'STATE_SENT')
        self.assertEqual(access.calls, [('launch', order.order_id, {'tag': 'payload'})])

        cancel = make_order()
        access.cancel_order(cancel)
        self.assertEqual(cancel.order_state.name, 'STATE_CANCELING')
        self.assertEqual(access.calls[-1][0], 'cancel')

        access.shut_down()
        self.assertFalse(access.enabled)

    def test_03_risk_rejection_blocks_the_transaction(self) -> None:
        access = self._CountingAccess(MDS, RISK_PROFILE)
        access.start()

        order = make_order(5000.)
        access.launch_order(order)

        self.assertEqual(order.order_state.name, 'STATE_REJECTED')
        self.assertEqual(access.calls[0][0], 'reject')

    def test_04_no_risk_profile_blocks_the_transaction(self) -> None:
        access = self._CountingAccess(MDS, None)
        access.start()

        order = make_order()
        access.launch_order(order)

        self.assertEqual(order.order_state.name, 'STATE_REJECTED')
        self.assertEqual(access.calls[0][0], 'reject')

    def test_05_buffered_path_keeps_the_payload(self) -> None:
        access = self._CountingAccess(MDS, RISK_PROFILE, cool_down=0.02)
        access.start()

        order = make_order()
        access.launch_order(order, tag='buffered')

        for _ in range(200):
            if access.calls:
                break
            time.sleep(0.01)

        self.assertEqual(access.calls, [('launch', order.order_id, {'tag': 'buffered'})])
        self.assertEqual(order.order_state.name, 'STATE_SENT')

        access.shut_down()


class TestPositionManagementService(unittest.TestCase):
    """The C-backed position service: algo registries, exposure and fills."""

    @classmethod
    def setUpClass(cls) -> None:
        PROFILE_CN.activate()
        feed_price()

    @classmethod
    def tearDownClass(cls) -> None:
        PROFILE_CN.deactivate()

    def setUp(self) -> None:
        RISK_PROFILE.clear()
        setup_profile_rules(RISK_PROFILE)

    def test_00_dummy_exposure_and_clear(self) -> None:
        access = DirectMarketAccess(MDS, RISK_PROFILE)
        position = PositionManagementService(dma=access)

        self.assertIsNotNone(position.position_id)
        self.assertEqual(position.default_algo, 'passive')
        self.assertFalse(position.no_cache)

        position.add_exposure(TICKER, 100., 1050., TransactionSide.SIDE_LONG_OPEN, TS)

        self.assertEqual(position.exposure_volume, {TICKER: 100.})
        self.assertEqual(position.notional, {TICKER: 1050.})
        self.assertEqual(position.pnl(), {TICKER: 0.})
        self.assertEqual(len(position.orders), 1)
        self.assertIn(position.position_id, position.to_json(fmt='dict'))

        position.clear()
        self.assertEqual(len(position.algos), 0)
        self.assertEqual(position.exposure_volume, {})

    def test_01_open_order_and_fill(self) -> None:
        access = DirectMarketAccess(MDS, RISK_PROFILE)
        access.start()
        position = PositionManagementService(dma=access)

        algo = position.open(TICKER, 100., SIDE_LONG)
        self.assertIsNotNone(algo)
        self.assertEqual(len(position.working_algos), 1)
        self.assertEqual(position.exposure_volume, {})

        order = list(position.working_order.values())[0]
        self.assertEqual(order.order_state.name, 'STATE_SENT')

        report = TradeReport(
            ticker=TICKER,
            volume=100.,
            price=10.5,
            side=TransactionSide.SIDE_LONG_OPEN,
            timestamp=TS,
            order_id=order.order_id,
        )

        position.on_filled(report)

        self.assertEqual(position.exposure_volume, {TICKER: 100.})
        self.assertEqual(len(position.working_algos), 0)
        self.assertEqual(len(position.trades), 1)

        access.shut_down()

    def test_02_balance_routes_reports(self) -> None:
        BALANCE.reset()

        access = DirectMarketAccess(MDS, RISK_PROFILE)
        access.start()
        position = PositionManagementService(dma=access)

        position.open(TICKER, 100., SIDE_LONG)
        order = list(position.working_order.values())[0]

        BALANCE.add(map_id='test-position', position_tracker=position)
        self.assertIs(BALANCE.get(map_id='test-position'), position)
        self.assertEqual(BALANCE.tracker_mapping['test-position'], position.position_id)

        report = TradeReport(
            ticker=TICKER,
            volume=100.,
            price=10.5,
            side=TransactionSide.SIDE_LONG_OPEN,
            timestamp=TS,
            order_id=order.order_id,
        )

        self.assertEqual(BALANCE.on_report(report), 1)
        self.assertEqual(len(BALANCE.trade_logs), 1)
        self.assertEqual(position.exposure_volume, {TICKER: 100.})

        BALANCE.pop('test-position')
        self.assertIsNone(BALANCE.get(map_id='test-position'))

        access.shut_down()


if __name__ == '__main__':
    unittest.main()

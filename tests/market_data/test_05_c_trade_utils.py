import math
import unittest
import uuid
from datetime import datetime, timezone

from algo_engine.base.c_market_data.c_trade_utils import (
    OrderState,
    TradeInstruction,
    TradeReport,
)
from algo_engine.base.c_market_data.c_transaction import TransactionSide, OrderType


class TestOrderState(unittest.TestCase):

    def test_state_helpers(self):
        self.assertTrue(OrderState.STATE_PLACED.is_working)
        self.assertTrue(OrderState.STATE_PLACED.is_placed)
        self.assertFalse(OrderState.STATE_PLACED.is_done)
        self.assertTrue(OrderState.STATE_FILLED.is_done)


class TestTradeReport(unittest.TestCase):

    def _make_report(self, **overrides):
        params = dict(
            ticker='RPT',
            timestamp=100.0,
            price=2.5,
            volume=3.0,
            side=TransactionSide.SIDE_LONG_OPEN,
            multiplier=4.0,
            order_id='ORDER-1',
            trade_id='TRADE-1',
        )
        params.update(overrides)
        return TradeReport(**params)

    def test_notional_and_flows(self):
        report = self._make_report()
        expected_notional = 2.5 * 3.0 * 4.0
        self.assertEqual(report.notional, expected_notional)
        self.assertEqual(report.volume_flow, 3.0)
        self.assertEqual(report.notional_flow, expected_notional)
        dict_repr = report.to_json(fmt='dict')
        restored = TradeReport.from_json(dict_repr)
        self.assertEqual(restored.ticker, report.ticker)
        self.assertEqual(restored.trade_id, report.trade_id)
        td2 = TradeReport(
            ticker='RPT',
            timestamp=100.0,
            price=2.5,
            volume=3.0,
            side=TransactionSide.SIDE_SHORT_OPEN,
            multiplier=4.0,
            order_id='ORDER-1',
            trade_id='TRADE-1',
        )
        self.assertEqual(td2.notional, expected_notional)
        self.assertEqual(td2.volume_flow, -3.0)
        self.assertEqual(td2.notional_flow, -expected_notional)


    def test_reset_ids_and_to_trade(self):
        report = self._make_report()
        old_order = report.order_id
        report.reset_order_id()
        self.assertNotEqual(report.order_id, old_order)
        new_trade_id = uuid.uuid4()
        report.reset_trade_id(new_trade_id)
        self.assertEqual(report.trade_id, new_trade_id)
        trade = report.to_trade()
        self.assertEqual(trade.ticker, report.ticker)
        self.assertEqual(trade.transaction_id, new_trade_id)


class TestTradeInstruction(unittest.TestCase):

    def _make_instruction(self, **overrides):
        params = dict(
            ticker='INS',
            timestamp=200.0,
            side=TransactionSide.SIDE_LONG_OPEN,
            volume=10.0,
            order_type=OrderType.ORDER_LIMIT,
            limit_price=5.0,
            multiplier=2.0,
        )
        params.update(overrides)
        return TradeInstruction(**params)

    def _make_report(self, instruction, **overrides):
        params = dict(
            ticker=instruction.ticker,
            timestamp=instruction.timestamp + 1,
            price=instruction.limit_price or 1.0,
            volume=3.0,
            side=instruction.side_int,
            multiplier=instruction.multiplier,
            order_id=instruction.order_id,
            trade_id=uuid.uuid4(),
        )
        params.update(overrides)
        return TradeReport(**params)

    def test_fill_to_completion(self):
        instruction = self._make_instruction(volume=4.0)
        report = self._make_report(instruction, volume=4.0)
        instruction.fill(report)
        self.assertEqual(instruction.filled_volume, 4.0)
        self.assertEqual(instruction.order_state, OrderState.STATE_FILLED)
        self.assertEqual(instruction.working_volume, 0.0)
        self.assertEqual(instruction.finished_ts, report.timestamp)
        self.assertIn(report.trade_id, instruction.trades)

    def test_partial_fill_and_duplicate_guard(self):
        instruction = self._make_instruction(volume=8.0)
        report = self._make_report(instruction, volume=3.0)
        instruction.fill(report)
        self.assertEqual(instruction.order_state, OrderState.STATE_PARTFILLED)
        instruction.fill(report)
        self.assertEqual(instruction.filled_volume, 3.0)
        self.assertEqual(len(instruction.trades), 1)

    def test_fill_volume_overflow_raises(self):
        instruction = self._make_instruction(volume=2.0)
        report = self._make_report(instruction, volume=3.0)
        with self.assertRaises(ValueError):
            instruction.fill(report)

    def test_reset_and_reset_order_id_updates_trades(self):
        instruction = self._make_instruction()
        report = self._make_report(instruction)
        instruction.fill(report)
        new_order_id = 'ORDER-RENAMED'
        instruction.reset_order_id(new_order_id)
        self.assertEqual(instruction.order_id, new_order_id)
        self.assertEqual(instruction.trades[report.trade_id].order_id, new_order_id)
        instruction.reset()
        self.assertEqual(instruction.order_state, OrderState.STATE_PENDING)
        self.assertEqual(instruction.filled_volume, 0.0)
        self.assertFalse(instruction.trades)

    def test_add_trade_sets_state(self):
        instruction = self._make_instruction(volume=5.0)
        report = self._make_report(instruction, volume=5.0)
        instruction.add_trade(report)
        self.assertEqual(instruction.order_state, OrderState.STATE_FILLED)
        self.assertEqual(instruction.finished_ts, report.timestamp)
        self.assertTrue(math.isclose(instruction.filled_notional, report.notional))


class TestTradeReportPropertyContract(unittest.TestCase):
    """Contract: every TradeReport property returns its documented value with
    a stable Python type.

    Expected behavior:
        - Numeric fields (price/volume/multiplier/notional/fee) are float.
        - side is a TransactionSide enum; side_int/side_sign are int.
        - volume_flow/notional_flow carry the side sign.
        - trade_time is a tz-aware datetime equal to the profile conversion of
          the header timestamp (microsecond precision).

    Oracle: expected values computed from constructor inputs; time properties
    cross-checked against market_time and fixed UTC literals.
    """

    def _make_report(self, **overrides):
        params = dict(
            ticker='RPT',
            timestamp=100.123456,
            price=2.5,
            volume=3.0,
            side=TransactionSide.SIDE_LONG_OPEN,
            multiplier=4.0,
            fee=0.25,
            order_id='ORDER-1',
            trade_id='TRADE-1',
        )
        params.update(overrides)
        return TradeReport(**params)

    def test_field_properties_values_and_types(self):
        report = self._make_report()
        self.assertIsInstance(report.ticker, str)
        self.assertEqual(report.ticker, 'RPT')
        self.assertIsInstance(report.timestamp, float)
        self.assertEqual(report.timestamp, 100.123456)
        for value in (report.price, report.volume, report.multiplier, report.notional, report.fee):
            self.assertIsInstance(value, float)
        self.assertEqual(report.price, 2.5)
        self.assertEqual(report.volume, 3.0)
        self.assertEqual(report.multiplier, 4.0)
        self.assertEqual(report.notional, 2.5 * 3.0 * 4.0)
        self.assertEqual(report.fee, 0.25)

    def test_side_properties_values_and_types(self):
        report = self._make_report(side=TransactionSide.SIDE_SHORT_OPEN)
        self.assertIs(report.side, TransactionSide.SIDE_SHORT_OPEN)
        self.assertEqual(report.side_int, TransactionSide.SIDE_SHORT_OPEN.value)
        self.assertIsInstance(report.side_int, int)
        self.assertEqual(report.side_sign, -1)
        self.assertIsInstance(report.side_sign, int)

    def test_flow_properties_carry_side_sign(self):
        buy = self._make_report(side=TransactionSide.SIDE_LONG_OPEN)
        self.assertEqual(buy.volume_flow, 3.0)
        self.assertEqual(buy.notional_flow, buy.notional)
        sell = self._make_report(side=TransactionSide.SIDE_SHORT_OPEN)
        self.assertEqual(sell.volume_flow, -3.0)
        self.assertEqual(sell.notional_flow, -sell.notional)

    def test_trade_time_is_tz_aware_datetime(self):
        report = self._make_report()
        self.assertIsInstance(report.trade_time, datetime)
        self.assertIsNotNone(report.trade_time.tzinfo)
        # Default profile is UTC: ts 100.123456 -> 1970-01-01 00:01:40.123456.
        expected = datetime(1970, 1, 1, 0, 1, 40, 123456, tzinfo=timezone.utc)
        self.assertEqual(report.trade_time, expected)
        # Consistent with the header-derived market_time on the same instance.
        self.assertEqual(report.trade_time, report.market_time)

    def test_identifier_types_preserved(self):
        report = self._make_report()
        self.assertIsInstance(report.order_id, str)
        self.assertEqual(report.order_id, 'ORDER-1')
        self.assertIsInstance(report.trade_id, str)
        self.assertEqual(report.trade_id, 'TRADE-1')
        # Default trade_id (NO_DEFAULT) is auto-generated as a UUID.
        auto = TradeReport(
            ticker='RPT', timestamp=100.0, price=1.0, volume=1.0,
            side=TransactionSide.SIDE_LONG_OPEN, order_id='ORD-AUTO',
        )
        self.assertIsInstance(auto.trade_id, uuid.UUID)
        auto.reset_trade_id()
        self.assertIsInstance(auto.trade_id, uuid.UUID)
        auto.reset_order_id()
        self.assertIsInstance(auto.order_id, uuid.UUID)


class TestTradeInstructionPropertyContract(unittest.TestCase):
    """Contract: every TradeInstruction property returns its documented value
    with a stable Python type across the order lifecycle.

    Expected behavior:
        - limit_price is float: finite for limit orders, NAN for market
          orders (never None).
        - State timestamps start at 0.0 (float); placed/canceled/finished_time
          are None until the matching state is entered, then tz-aware
          datetime.
        - average_price is the per-unit fill price = filled_notional /
          filled_volume / multiplier; NAN before any fill.
        - fee accumulates over fills; working_volume = volume - filled_volume.

    Oracle: expected values computed from constructor inputs and fill reports.
    """

    def _make_instruction(self, **overrides):
        params = dict(
            ticker='INS',
            timestamp=200.123456,
            side=TransactionSide.SIDE_LONG_OPEN,
            volume=10.0,
            order_type=OrderType.ORDER_LIMIT,
            limit_price=5.0,
            multiplier=2.0,
        )
        params.update(overrides)
        if params['limit_price'] is None:
            # Market orders carry no limit price (typed double defaults to NAN).
            del params['limit_price']
        return TradeInstruction(**params)

    def _make_fill_report(self, instruction, price, volume, fee=0.0):
        return TradeReport(
            ticker=instruction.ticker,
            timestamp=instruction.timestamp + 1,
            price=price,
            volume=volume,
            side=instruction.side_int,
            multiplier=instruction.multiplier,
            fee=fee,
            order_id=instruction.order_id,
            trade_id=uuid.uuid4(),
        )

    def test_initial_state_values_and_types(self):
        instruction = self._make_instruction()
        self.assertIsInstance(instruction.ticker, str)
        self.assertIsInstance(instruction.timestamp, float)
        self.assertIs(instruction.side, TransactionSide.SIDE_LONG_OPEN)
        self.assertIsInstance(instruction.side_int, int)
        self.assertEqual(instruction.side_sign, 1)
        self.assertIs(instruction.order_type, OrderType.ORDER_LIMIT)
        self.assertIsInstance(instruction.order_type_int, int)
        self.assertIs(instruction.order_state, OrderState.STATE_PENDING)
        self.assertIsInstance(instruction.order_state_int, int)
        self.assertEqual(instruction.limit_price, 5.0)
        self.assertIsInstance(instruction.limit_price, float)
        self.assertEqual(instruction.volume, 10.0)
        self.assertEqual(instruction.multiplier, 2.0)
        self.assertEqual(instruction.filled_volume, 0.0)
        self.assertEqual(instruction.working_volume, 10.0)
        self.assertEqual(instruction.filled_notional, 0.0)
        self.assertEqual(instruction.fee, 0.0)
        self.assertIsInstance(instruction.order_id, uuid.UUID)
        self.assertEqual(instruction.trades, {})

    def test_market_order_limit_price_is_nan(self):
        instruction = self._make_instruction(order_type=OrderType.ORDER_MARKET, limit_price=None)
        self.assertIs(instruction.order_type, OrderType.ORDER_MARKET)
        self.assertTrue(math.isnan(instruction.limit_price))

    def test_average_price_nan_before_any_fill(self):
        instruction = self._make_instruction()
        self.assertTrue(math.isnan(instruction.average_price))

    def test_time_properties_before_and_after_state_entries(self):
        instruction = self._make_instruction()
        for value in (instruction.placed_ts, instruction.canceled_ts, instruction.finished_ts):
            self.assertEqual(value, 0.0)
            self.assertIsInstance(value, float)
        for value in (instruction.placed_time, instruction.canceled_time, instruction.finished_time):
            self.assertIsNone(value)

        # start_time maps the instruction timestamp through the profile.
        self.assertIsInstance(instruction.start_time, datetime)
        self.assertIsNotNone(instruction.start_time.tzinfo)
        expected = datetime(1970, 1, 1, 0, 3, 20, 123456, tzinfo=timezone.utc)
        self.assertEqual(instruction.start_time, expected)
        self.assertEqual(instruction.start_time, instruction.market_time)

        instruction.set_order_state(OrderState.STATE_PLACED, 200.123456 + 60)
        self.assertAlmostEqual(instruction.placed_ts, 260.123456)
        self.assertIsInstance(instruction.placed_ts, float)
        self.assertEqual(instruction.placed_time, datetime(1970, 1, 1, 0, 4, 20, 123456, tzinfo=timezone.utc))
        self.assertIsNone(instruction.canceled_time)
        self.assertIsNone(instruction.finished_time)

        instruction.canceled(200.123456 + 120)
        self.assertAlmostEqual(instruction.canceled_ts, 320.123456)
        self.assertEqual(instruction.canceled_time, datetime(1970, 1, 1, 0, 5, 20, 123456, tzinfo=timezone.utc))

    def test_average_price_is_filled_volume_weighted(self):
        """Partial fills price the average by the filled volume, not the order
        volume: 3 @ 4.0 + 2 @ 5.0 -> (3*4.0 + 2*5.0) / 5 = 4.4."""
        instruction = self._make_instruction(volume=10.0, multiplier=1.0)
        instruction.fill(self._make_fill_report(instruction, price=4.0, volume=3.0))
        instruction.fill(self._make_fill_report(instruction, price=5.0, volume=2.0))
        self.assertEqual(instruction.filled_volume, 5.0)
        self.assertEqual(instruction.filled_notional, 3.0 * 4.0 + 2.0 * 5.0)
        self.assertEqual(instruction.average_price, 4.4)
        self.assertIs(instruction.order_state, OrderState.STATE_PARTFILLED)
        self.assertEqual(instruction.working_volume, 5.0)

    def test_fill_to_completion_sets_finished_state_and_time(self):
        instruction = self._make_instruction(volume=5.0, multiplier=1.0)
        report = self._make_fill_report(instruction, price=4.0, volume=3.0, fee=0.1)
        instruction.fill(report)
        self.assertEqual(instruction.fee, 0.1)
        self.assertEqual(len(instruction.trades), 1)
        self.assertIn(report.trade_id, instruction.trades)
        self.assertIs(instruction.order_state, OrderState.STATE_PARTFILLED)

        rest = self._make_fill_report(instruction, price=5.0, volume=2.0, fee=0.2)
        instruction.fill(rest)
        self.assertEqual(instruction.filled_volume, 5.0)
        self.assertEqual(instruction.working_volume, 0.0)
        self.assertAlmostEqual(instruction.fee, 0.3)
        self.assertIs(instruction.order_state, OrderState.STATE_FILLED)
        self.assertAlmostEqual(instruction.finished_ts, rest.timestamp)
        # finished_time maps the report timestamp (201.123456) through the profile.
        self.assertEqual(instruction.finished_time, datetime(1970, 1, 1, 0, 3, 21, 123456, tzinfo=timezone.utc))
        self.assertIsInstance(instruction.finished_time, datetime)
        self.assertEqual(instruction.average_price, (3.0 * 4.0 + 2.0 * 5.0) / 5.0)

    def test_reset_restores_initial_state(self):
        instruction = self._make_instruction(volume=5.0, multiplier=1.0)
        instruction.set_order_state(OrderState.STATE_PLACED, 250.0)
        instruction.fill(self._make_fill_report(instruction, price=4.0, volume=5.0))
        self.assertIs(instruction.order_state, OrderState.STATE_FILLED)
        instruction.reset()
        self.assertIs(instruction.order_state, OrderState.STATE_PENDING)
        self.assertEqual(instruction.filled_volume, 0.0)
        self.assertEqual(instruction.working_volume, 5.0)
        self.assertEqual(instruction.filled_notional, 0.0)
        self.assertEqual(instruction.fee, 0.0)
        self.assertEqual(instruction.trades, {})
        for value in (instruction.placed_ts, instruction.canceled_ts, instruction.finished_ts):
            self.assertEqual(value, 0.0)
        for value in (instruction.placed_time, instruction.canceled_time, instruction.finished_time):
            self.assertIsNone(value)
        self.assertTrue(math.isnan(instruction.average_price))


if __name__ == '__main__':
    unittest.main()


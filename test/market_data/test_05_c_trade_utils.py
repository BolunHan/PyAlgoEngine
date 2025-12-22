import math
import unittest
import uuid

from algo_engine.base.c_market_data_ng.c_trade_utils import (
    OrderState,
    TradeInstruction,
    TradeReport,
)
from algo_engine.base.c_market_data_ng.c_transaction import TransactionSide, OrderType


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


if __name__ == '__main__':
    unittest.main()


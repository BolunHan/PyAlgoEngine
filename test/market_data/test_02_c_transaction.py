import unittest
import uuid

from algo_engine.base.c_market_data_ng.c_transaction import (
    OrderData,
    OrderType,
    TradeData,
    TransactionData,
    TransactionDirection,
    TransactionOffset,
    TransactionSide,
)


class TestTransactionEnums(unittest.TestCase):

    def test_direction_offset_bitwise_combination(self):
        long_open_side = TransactionDirection.DIRECTION_LONG | TransactionOffset.OFFSET_OPEN
        self.assertEqual(long_open_side, TransactionSide.SIDE_LONG_OPEN)

        short_close_side = TransactionOffset.OFFSET_CLOSE | TransactionDirection.DIRECTION_SHORT
        self.assertEqual(short_close_side, TransactionSide.SIDE_SHORT_CLOSE)

    def test_side_properties(self):
        side = TransactionSide.SIDE_SHORT_CLOSE
        self.assertEqual(side.sign, -1)
        self.assertEqual(side.offset, TransactionOffset.OFFSET_CLOSE)
        self.assertEqual(side.direction, TransactionDirection.DIRECTION_SHORT)
        self.assertEqual(side.side_name, 'sell')
        self.assertEqual(side.offset_name, 'close')
        self.assertEqual(side.direction_name, 'short')

        side = TransactionSide.SIDE_LONG_CLOSE
        self.assertEqual(side.sign, 1)
        self.assertEqual(side.offset, TransactionOffset.OFFSET_CLOSE)
        self.assertEqual(side.direction, TransactionDirection.DIRECTION_LONG)
        self.assertEqual(side.side_name, 'cover')
        self.assertEqual(side.offset_name, 'close')
        self.assertEqual(side.direction_name, 'long')

        side = TransactionSide.SIDE_LONG_OPEN
        self.assertEqual(side.sign, 1)
        self.assertEqual(side.offset, TransactionOffset.OFFSET_OPEN)
        self.assertEqual(side.direction, TransactionDirection.DIRECTION_LONG)
        self.assertEqual(side.side_name, 'buy')
        self.assertEqual(side.offset_name, 'open')
        self.assertEqual(side.direction_name, 'long')

        side = TransactionSide.SIDE_SHORT_OPEN
        self.assertEqual(side.sign, -1)
        self.assertEqual(side.offset, TransactionOffset.OFFSET_OPEN)
        self.assertEqual(side.direction, TransactionDirection.DIRECTION_SHORT)
        self.assertEqual(side.side_name, 'short')
        self.assertEqual(side.offset_name, 'open')
        self.assertEqual(side.direction_name, 'short')


class TestTransactionData(unittest.TestCase):
    def test_init_sets_notional_and_ids(self):
        tx_id = uuid.uuid4()
        buy_id = 'buy-1'
        sell_id = tx_id.int

        data = TransactionData(
            ticker='600010.SH',
            timestamp=123.456,
            price=10.0,
            volume=2.0,
            side=TransactionSide.SIDE_LONG_OPEN,
            multiplier=2.0,
            transaction_id=tx_id,
            buy_id=buy_id,
            sell_id=sell_id,
        )

        self.assertEqual(data.price, 10.0)
        self.assertEqual(data.volume, 2.0)
        self.assertEqual(data.multiplier, 2.0)
        self.assertEqual(data.notional, 40.0)
        self.assertEqual(data.side, TransactionSide.SIDE_LONG_OPEN)
        self.assertEqual(data.side_sign, 1)
        self.assertEqual(data.volume_flow, 2.0)
        self.assertEqual(data.notional_flow, 40.0)
        self.assertEqual(data.transaction_id, tx_id)
        self.assertEqual(data.buy_id, buy_id)
        self.assertEqual(data.sell_id, sell_id)

        data = TransactionData(
            ticker='600010.SH',
            timestamp=123.456,
            price=10.0,
            volume=2.0,
            notional=15,
            side=TransactionSide.SIDE_SHORT_CLOSE,
            multiplier=2.0,
            transaction_id=tx_id,
            buy_id=buy_id,
            sell_id=sell_id,
        )

        self.assertEqual(data.price, 10.0)
        self.assertEqual(data.volume, 2.0)
        self.assertEqual(data.multiplier, 2.0)
        self.assertEqual(data.notional, 15.0)
        self.assertEqual(data.side, TransactionSide.SIDE_SHORT_CLOSE)
        self.assertEqual(data.side_sign, -1)
        self.assertEqual(data.volume_flow, -2.0)
        self.assertEqual(data.notional_flow, -15.0)
        self.assertEqual(data.transaction_id, tx_id)
        self.assertEqual(data.buy_id, buy_id)
        self.assertEqual(data.sell_id, sell_id)

    def test_init_respects_explicit_notional(self):
        data = TransactionData(
            ticker='TEST-2',
            timestamp=10.0,
            price=1.23,
            volume=4.0,
            side=TransactionSide.SIDE_SHORT_CLOSE,
            multiplier=3.0,
            notional=99.9,
        )

        self.assertEqual(data.notional, 99.9)
        self.assertEqual(data.side_sign, -1)
        self.assertEqual(data.volume_flow, -4.0)
        self.assertEqual(data.notional_flow, -99.9)

    def test_merge_combines_transactions(self):
        first = TransactionData(
            ticker='MERGE',
            timestamp=10.0,
            price=10.0,
            volume=2.0,
            side=TransactionSide.SIDE_LONG_OPEN,
        )
        second = TransactionData(
            ticker='MERGE',
            timestamp=11.5,
            price=12.0,
            volume=1.0,
            side=TransactionSide.SIDE_SHORT_CLOSE,
        )

        merged = TransactionData.merge([first, second])

        self.assertEqual(merged.ticker, 'MERGE')
        self.assertEqual(merged.timestamp, 11.5)
        self.assertEqual(merged.volume, 1.0)
        self.assertEqual(merged.notional, 8.0)
        self.assertEqual(merged.price, 8.0)
        self.assertEqual(merged.side, TransactionSide.SIDE_LONG_OPEN)
        self.assertEqual(merged.multiplier, 1.0)

    def test_merge_mismatched_ticker_raises(self):
        first = TransactionData(
            ticker='A',
            timestamp=1.0,
            price=1.0,
            volume=1.0,
            side=TransactionSide.SIDE_LONG_OPEN,
        )
        second = TransactionData(
            ticker='B',
            timestamp=2.0,
            price=1.0,
            volume=1.0,
            side=TransactionSide.SIDE_LONG_OPEN,
        )

        with self.assertRaises(AssertionError):
            TransactionData.merge([first, second])


class TestOrderData(unittest.TestCase):

    def test_order_properties(self):
        oid = uuid.uuid4().int
        order = OrderData(
            ticker='ORDER',
            timestamp=42.0,
            price=5.0,
            volume=3.0,
            side=TransactionSide.SIDE_ASK,
            order_id=oid,
            order_type=OrderType.ORDER_LIMIT,
        )

        self.assertEqual(order.price, 5.0)
        self.assertEqual(order.volume, 3.0)
        self.assertEqual(order.side, TransactionSide.SIDE_ASK)
        self.assertEqual(order.side_sign, -1)
        self.assertEqual(order.flow, -3.0)
        self.assertEqual(order.order_id, oid)
        self.assertEqual(order.order_type, OrderType.ORDER_LIMIT)
        self.assertEqual(order.order_type_int, OrderType.ORDER_LIMIT.value)


class TestTradeData(unittest.TestCase):

    def test_trade_alias_properties(self):
        trade = TradeData(
            ticker='TRADE',
            timestamp=100.0,
            trade_price=20.0,
            trade_volume=4.0,
            trade_side=TransactionSide.SIDE_SHORT_CLOSE,
            multiplier=2.0,
        )

        expected_notional = 20.0 * 4.0 * 2.0
        self.assertEqual(trade.trade_price, 20.0)
        self.assertEqual(trade.trade_volume, 4.0)
        self.assertEqual(trade.trade_side, TransactionSide.SIDE_SHORT_CLOSE)
        self.assertEqual(trade.notional, expected_notional)
        self.assertEqual(trade.volume_flow, -4.0)
        self.assertEqual(trade.notional_flow, -expected_notional)


if __name__ == '__main__':
    unittest.main()

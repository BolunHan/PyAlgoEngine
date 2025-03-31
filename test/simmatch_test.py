import time
import unittest
from datetime import datetime
from unittest.mock import Mock

from algo_engine.backtest import SimMatch
from algo_engine.base import TickData, TickDataLite, BarData, TransactionData, TransactionSide, OrderType, TradeInstruction, OrderState


class TestSimMatch(unittest.TestCase):
    def run_test(self):
        self.setUp()
        self.test_tick_no_fill()
        self.test_tick_full_fill()
        self.test_tick_partial_fill()

        self.test_transaction_no_fill()
        self.test_transaction_full_fill()
        self.test_transaction_partial_fill()

        self.sim.matching_config['instant_fill'] = True
        self.test_instant_fill_buy_order()
        self.test_instant_fill_sell_order()
        self.test_instant_fill_market_order()
        self.test_instant_fill_without_lag()
        self.test_instant_fill_with_hit_probability()
        self.test_no_instant_fill_when_disabled()

    def setUp(self):
        self.event_engine = Mock()
        self.topic_set = Mock()
        self.sim = SimMatch(
            ticker='TEST',
            event_engine=self.event_engine,
            topic_set=self.topic_set,
            fee_rate=0.001,
            hit_prob=1.0,
            instant_fill=False
        )
        self.base_order = {
            'ticker': 'TEST',
            'volume': 100,
            'multiplier': 1.0,
            'order_type': OrderType.ORDER_LIMIT
        }

    def create_order(self, side=TransactionSide.SIDE_BID, limit_price=50.0):
        return TradeInstruction(side=side, limit_price=limit_price, timestamp=time.time(), **self.base_order)

    def test_tick_full_fill(self):
        """Test order fully filled by tick data"""
        # 1. Create and launch buy order
        order = self.create_order(side=TransactionSide.SIDE_BID, limit_price=50.0)
        self.sim.launch_order(order=order)

        # 2. Send tick that should fully fill the order
        tick = TickData(
            ticker='TEST',
            timestamp=datetime.now().timestamp(),
            last_price=49.5,
            bid_price_1=49.5,
            ask_price_1=49.9,
            bid_volume_1=100,
            ask_volume_1=100,
        )
        self.sim(market_data=tick)

        # 3. Verify order is fully filled
        self.assertEqual(order.order_state, OrderState.STATE_FILLED)
        self.assertEqual(order.filled_volume, 100)
        self.assertAlmostEqual(order.average_price, 49.9 * 1.0001)  # With slippage

    def test_tick_no_fill(self):
        """Test order not filled by tick data"""
        # 1. Create and launch buy order
        order = self.create_order(side=TransactionSide.SIDE_BID, limit_price=50.0)
        self.sim.launch_order(order=order)

        # 2. Send tick that shouldn't fill the order
        tick = TickData(
            ticker='TEST',
            timestamp=datetime.now().timestamp(),
            last_price=50.1,
            bid_price_1=50.0,
            ask_price_1=50.1,
            bid_volume_1=100,
            ask_volume_1=100,
        )
        self.sim(market_data=tick)

        # 3. Verify order is not filled
        self.assertEqual(order.order_state, OrderState.STATE_PLACED)
        self.assertEqual(order.filled_volume, 0)

    def test_tick_partial_fill(self):
        """Test order partially filled by tick lite data"""
        # 1. Create and launch larger order
        order = self.create_order(side=TransactionSide.SIDE_BID, limit_price=50.0)
        self.sim.launch_order(order=order)

        # 2. Send tick lite with partial volume
        tick = TickData(
            ticker='TEST',
            timestamp=datetime.now().timestamp(),
            last_price=49.9,
            bid_price_1=49.8,
            bid_volume_1=50,
            ask_price_1=49.9,
            ask_volume_1=10  # Only 100 available
        )
        self.sim(market_data=tick)

        # 3. Verify partial fill
        self.assertEqual(order.order_state, OrderState.STATE_PARTFILLED)
        self.assertEqual(order.filled_volume, 10)
        self.assertAlmostEqual(order.average_price, 49.9 * 1.0001)  # With slippage

        # 3. Send tick lite with partial volume
        tick = TickData(
            ticker='TEST',
            timestamp=datetime.now().timestamp(),
            last_price=49.9,
            bid_price_1=49.8,
            bid_volume_1=50,
            ask_price_1=49.7,
            ask_price_2=49.8,
            ask_volume_1=10,
            ask_volume_2=5
        )
        self.sim(market_data=tick)

        # 3. Verify partial fill
        self.assertEqual(order.order_state, OrderState.STATE_PARTFILLED)
        self.assertEqual(order.filled_volume, 25)
        self.assertAlmostEqual(order.average_price, 49.8 * 1.0001)  # With slippage

    def test_transaction_full_fill(self):
        """Test order fully filled by single transaction"""
        # 1. Create and launch buy order
        order = self.create_order(side=TransactionSide.SIDE_BID, limit_price=50.0)
        self.sim.launch_order(order=order)

        # 2. Send matching transaction (opposite side)
        trans = TransactionData(
            ticker='TEST',
            timestamp=datetime.now().timestamp(),
            price=49.9,
            volume=100,
            side=TransactionSide.SIDE_ASK  # Opposite side
        )
        self.sim(market_data=trans)

        # 3. Verify full fill
        self.assertEqual(order.order_state, OrderState.STATE_FILLED)
        self.assertEqual(order.filled_volume, 100)
        self.assertAlmostEqual(order.average_price, 49.9 * 1.0001)  # With slippage

    def test_transaction_partial_fill(self):
        """Test order partially filled by multiple transactions"""
        # 1. Create and launch larger order
        order = self.create_order(side=TransactionSide.SIDE_BID, limit_price=50.0)
        self.sim.launch_order(order=order)

        # 2. First transaction - partial fill
        trans1 = TransactionData(
            ticker='TEST',
            timestamp=datetime.now().timestamp(),
            price=49.9,
            volume=10,
            side=TransactionSide.SIDE_ASK
        )
        self.sim(market_data=trans1)

        # 3. Verify partial fill
        self.assertEqual(order.order_state, OrderState.STATE_PARTFILLED)
        self.assertEqual(order.filled_volume, 10)
        self.assertAlmostEqual(order.average_price, 49.9 * 1.0001)

        # 4. Second transaction - another partial fill
        trans2 = TransactionData(
            ticker='TEST',
            timestamp=datetime.now().timestamp() + 1,
            price=49.8,
            volume=20,
            side=TransactionSide.SIDE_ASK
        )
        self.sim(market_data=trans2)

        # 5. Verify updated fill (total 120/150)
        self.assertEqual(order.order_state, OrderState.STATE_PARTFILLED)
        self.assertEqual(order.filled_volume, 30)
        # Verify VWAP calculation
        expected_avg = ((10 * 49.9) + (20 * 49.8)) / 30 * 1.0001
        self.assertAlmostEqual(order.average_price, expected_avg)

        # 4. Second transaction - another partial fill
        trans2 = TransactionData(
            ticker='TEST',
            timestamp=datetime.now().timestamp() + 1,
            price=49.5,
            volume=100,
            side=TransactionSide.SIDE_ASK
        )
        self.sim(market_data=trans2)

        # 5. Verify updated fill (total 120/150)
        self.assertEqual(order.order_state, OrderState.STATE_FILLED)
        self.assertEqual(order.filled_volume, 100)
        # Verify VWAP calculation
        expected_avg = ((10 * 49.9) + (20 * 49.8) + (70 * 49.5)) / 100 * 1.0001
        self.assertAlmostEqual(order.average_price, expected_avg)

    def test_transaction_no_fill(self):
        """Test order not filled by transactions"""
        # 1. Create and launch buy order
        order = self.create_order(side=TransactionSide.SIDE_BID, limit_price=50.0)
        self.sim.launch_order(order=order)

        # 2. Send transaction that shouldn't match (same side)
        trans1 = TransactionData(
            ticker='TEST',
            timestamp=datetime.now().timestamp(),
            price=50.10,
            volume=100,
            side=TransactionSide.SIDE_BID  # Same side - no match
        )
        self.sim(market_data=trans1)

        # 3. Verify no fill
        self.assertEqual(order.order_state, OrderState.STATE_PLACED)
        self.assertEqual(order.filled_volume, 0)

        # 4. Send transaction that shouldn't match (price above limit)
        trans2 = TransactionData(
            ticker='TEST',
            timestamp=datetime.now().timestamp() + 1,
            price=50.1,
            volume=100,
            side=TransactionSide.SIDE_ASK  # Opposite side but price > limit
        )
        self.sim(market_data=trans2)

        # 5. Verify still no fill
        self.assertEqual(order.order_state, OrderState.STATE_PLACED)
        self.assertEqual(order.filled_volume, 0)

    def test_transaction_fill_with_lag(self):
        """Test fill considering lag configuration"""
        # Configure lag
        self.sim.matching_config['lag'] = {'ts': 1.0, 'n_transaction': 1}

        # 1. Create and launch order
        order = self.create_order(side=TransactionSide.SIDE_BID, limit_price=50.0)
        order_time = time.time()
        self.sim.launch_order(order=order)

        # 2. First transaction - too early (should not fill)
        trans1 = TransactionData(
            ticker='TEST',
            timestamp=order_time + 0.5,  # 0.5s after order
            price=49.9,
            volume=100,
            side=TransactionSide.SIDE_ASK
        )
        self.sim(market_data=trans1)
        self.assertEqual(order.order_state, OrderState.STATE_PLACED)

        # 3. Second transaction - after time lag (should fill)
        trans2 = TransactionData(
            ticker='TEST',
            timestamp=order_time + 1.1,  # 1.1s after order
            price=49.9,
            volume=100,
            side=TransactionSide.SIDE_ASK
        )
        self.sim(market_data=trans2)
        self.assertEqual(order.order_state, OrderState.STATE_FILLED)

    def test_instant_fill_buy_order(self):
        """Test buy order is instantly filled when instant_fill=True"""
        # 1. Create buy order
        order = self.create_order(side=TransactionSide.SIDE_BID, limit_price=50.0)

        # 2. Launch order (should fill immediately)
        self.sim.launch_order(order=order)

        # 3. Verify instant fill
        self.assertEqual(order.order_state, OrderState.STATE_FILLED)
        self.assertEqual(order.filled_volume, 100)
        # Should execute at limit price (50.0) + slippage
        self.assertAlmostEqual(order.average_price, 50.0)

    def test_instant_fill_sell_order(self):
        """Test sell order is instantly filled when instant_fill=True"""
        # 1. Create sell order
        order = self.create_order(side=TransactionSide.SIDE_ASK, limit_price=50.0)

        # 2. Launch order (should fill immediately)
        self.sim.launch_order(order=order)

        # 3. Verify instant fill
        self.assertEqual(order.order_state, OrderState.STATE_FILLED)
        self.assertEqual(order.filled_volume, 100)
        # Should execute at limit price (50.0) - slippage
        self.assertAlmostEqual(order.average_price, 50.0 * 0.9999)

    def test_instant_fill_market_order(self):
        """Test market order is instantly filled when instant_fill=True"""
        # 1. Create market order (no limit price)
        order = self.create_order(side=TransactionSide.SIDE_BID, limit_price=None)

        # Set last price for market order reference
        self.sim.last_price = 49.5

        # 2. Launch order (should fill immediately)
        self.sim.launch_order(order=order)

        # 3. Verify instant fill
        self.assertEqual(order.order_state, OrderState.STATE_FILLED)
        self.assertEqual(order.filled_volume, 100)
        # Should execute at last price + slippage
        self.assertAlmostEqual(order.average_price, 49.5 * 1.0001)

    def test_instant_fill_with_hit_probability(self):
        """Test instant fill respects hit probability"""
        # Configure hit probability
        self.sim.matching_config['hit_prob'] = 0.5
        self.sim.set_seed(123)  # Fixed seed for deterministic test

        # 1. Create order
        order = self.create_order()

        # 2. Launch order
        self.sim.launch_order(order=order)

        # 3. Verify fill based on probability
        # With seed=123 and prob=0.5, should fill
        self.assertEqual(order.order_state, OrderState.STATE_FILLED)

    def test_instant_fill_without_lag(self):
        """Test instant fill works when no lag is configured"""
        # Verify lag settings
        self.assertEqual(self.sim.matching_config['lag']['ts'], 0)
        self.assertEqual(self.sim.matching_config['lag']['n_transaction'], 0)

        # 1. Create order
        order = self.create_order()

        # 2. Launch order
        self.sim.launch_order(order=order)

        # 3. Verify instant fill
        self.assertEqual(order.order_state, OrderState.STATE_FILLED)


if __name__ == '__main__':
    t0 = TestSimMatch()
    t0.run_test()

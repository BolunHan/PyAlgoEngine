import unittest

from algo_engine.base import TransactionDirection as Direction, TransactionOffset as Offset
from algo_engine.engine import Balance, BalanceFlag


SIDE_LONG = Direction.DIRECTION_LONG | Offset.OFFSET_OPEN
SIDE_SHORT = Direction.DIRECTION_SHORT | Offset.OFFSET_OPEN


class TestBalance(unittest.TestCase):
    """The C-backed balance: the supplied fields and the affordability check."""

    def test_00_lifecycle_and_flags(self) -> None:
        balance = Balance()

        self.assertIn(Balance.__name__, repr(balance))
        self.assertEqual(balance.flags, 0)
        self.assertEqual(balance.available_cash, 0.)

        balance.available_cash = 100.
        self.assertEqual(balance.available_cash, 100.)
        self.assertEqual(balance.flags, BalanceFlag.available_cash)

        balance.equity = 250.
        self.assertEqual(balance.flags, BalanceFlag.available_cash | BalanceFlag.equity)

        balance.clear()
        self.assertEqual(balance.flags, 0)
        self.assertEqual(balance.available_cash, 0.)

    def test_01_every_field_marks_its_flag(self) -> None:
        balance = Balance()

        fields = (
            ('balance', BalanceFlag.balance),
            ('equity', BalanceFlag.equity),
            ('net_asset_value', BalanceFlag.net_asset_value),
            ('total_asset', BalanceFlag.total_asset),
            ('total_liability', BalanceFlag.total_liability),
            ('unrealized_pnl', BalanceFlag.unrealized_pnl),
            ('realized_pnl', BalanceFlag.realized_pnl),
            ('cash_balance', BalanceFlag.cash_balance),
            ('available_cash', BalanceFlag.available_cash),
            ('frozen_cash', BalanceFlag.frozen_cash),
            ('deposit', BalanceFlag.deposit),
            ('withdrawal', BalanceFlag.withdrawal),
            ('cash_in', BalanceFlag.cash_in),
            ('cash_out', BalanceFlag.cash_out),
            ('margin', BalanceFlag.margin),
            ('initial_margin', BalanceFlag.initial_margin),
            ('maintenance_margin', BalanceFlag.maintenance_margin),
            ('available_margin', BalanceFlag.available_margin),
            ('frozen_margin', BalanceFlag.frozen_margin),
            ('collateral', BalanceFlag.collateral),
            ('available_collateral', BalanceFlag.available_collateral),
            ('borrowed', BalanceFlag.borrowed),
            ('interest', BalanceFlag.interest),
            ('buying_power', BalanceFlag.buying_power),
            ('leverage', BalanceFlag.leverage),
        )

        for name, flag in fields:
            balance.clear()
            setattr(balance, name, 7.)

            self.assertEqual(getattr(balance, name), 7.)
            self.assertEqual(balance.flags, flag, name)

    def test_02_an_unfilled_balance_passes(self) -> None:
        balance = Balance()

        self.assertTrue(balance.check(SIDE_LONG, 1000., 10.))

        # Another flagged field stands in for nothing; the check reads
        # available cash only.
        balance.cash_balance = 250.
        balance.buying_power = 400.
        balance.balance = 750.
        self.assertTrue(balance.check(SIDE_LONG, 1000., 10.))

    def test_03_the_buy_is_limited_by_the_available_cash(self) -> None:
        balance = Balance()
        balance.available_cash = 100.

        self.assertTrue(balance.check(SIDE_LONG, 10., 10.))
        self.assertFalse(balance.check(SIDE_LONG, 11., 10.))

        # A sell raises no cash requirement.
        self.assertTrue(balance.check(SIDE_SHORT, 11., 10.))

        # The multiplier scales the requirement.
        self.assertTrue(balance.check(SIDE_LONG, 3., 10., 3.))
        self.assertFalse(balance.check(SIDE_LONG, 4., 10., 3.))

    def test_04_the_margin_rate_selects_the_available_margin(self) -> None:
        balance = Balance()
        balance.available_cash = 1000.
        balance.available_margin = 100.

        # 200 notional needs 100 at 0.5; the cash field is not consulted.
        self.assertTrue(balance.check(SIDE_LONG, 10., 10., 1., 0.5))
        self.assertFalse(balance.check(SIDE_LONG, 21., 10., 1., 0.5))

        # A rate of 1 is a margin trade too: 100 notional against 100 margin.
        self.assertTrue(balance.check(SIDE_LONG, 10., 10., 1., 1.))
        self.assertFalse(balance.check(SIDE_LONG, 11., 10., 1., 1.))

        # The margin branch constrains the short side too.
        self.assertTrue(balance.check(SIDE_SHORT, 10., 10., 1., 0.5))
        self.assertFalse(balance.check(SIDE_SHORT, 21., 10., 1., 0.5))

        # Without the margin field supplied the margin branch passes.
        balance.clear()
        self.assertTrue(balance.check(SIDE_LONG, 10., 10., 1., 0.5))

        # The default is NaN: the cash field is checked.
        balance.available_cash = 100.
        self.assertTrue(balance.check(SIDE_LONG, 10., 10.))
        self.assertFalse(balance.check(SIDE_LONG, 11., 10.))

    def test_05_unbound_wrapper_is_a_husk(self) -> None:
        balance = Balance.__new__(Balance)

        self.assertEqual(repr(balance), '<Balance: unbound>')

        for name in ('flags', 'available_cash', 'leverage'):
            with self.assertRaises(RuntimeError):
                getattr(balance, name)

        with self.assertRaises(RuntimeError):
            balance.clear()
        with self.assertRaises(RuntimeError):
            balance.check(SIDE_LONG, 1., 1.)


if __name__ == '__main__':
    unittest.main()

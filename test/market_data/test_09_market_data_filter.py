import unittest

from algo_engine.base.c_market_data_ng import c_market_data as md

FilterMode = md.FilterMode


class TestFilterMode(unittest.TestCase):
    def test_bitwise_operations_and_contains(self):
        no_order = FilterMode.NO_ORDER
        no_trade = FilterMode.NO_TRADE
        combined = no_order | no_trade

        self.assertEqual(combined.value, no_order.value | no_trade.value)
        self.assertIn(no_order, combined)
        self.assertIn(no_trade, combined)
        self.assertEqual((combined & no_order).value, no_order.value)
        self.assertEqual((combined & no_trade).value, no_trade.value)

    def test_all_includes_every_flag(self):
        flags = [
            FilterMode.NO_INTERNAL,
            FilterMode.NO_CANCEL,
            FilterMode.NO_AUCTION,
            FilterMode.NO_ORDER,
            FilterMode.NO_TRADE,
            FilterMode.NO_TICK,
        ]
        all_mode = FilterMode.all()
        for flag in flags:
            self.assertIn(flag, all_mode)

    def test_invert_toggles_only_known_flags(self):
        no_order = FilterMode.NO_ORDER
        inverted = ~no_order

        self.assertNotIn(no_order, inverted)
        self.assertIn(FilterMode.NO_TRADE, inverted)

    def test_repr_lists_active_flags(self):
        combined = FilterMode.NO_ORDER | FilterMode.NO_TRADE
        text = repr(combined)

        self.assertIn("NO_ORDER", text)
        self.assertIn("NO_TRADE", text)
        self.assertIn(hex(combined.value), text)

    def test_all_matches_manual_union(self):
        manual = (
                FilterMode.NO_INTERNAL
                | FilterMode.NO_CANCEL
                | FilterMode.NO_AUCTION
                | FilterMode.NO_ORDER
                | FilterMode.NO_TRADE
                | FilterMode.NO_TICK
        )
        self.assertEqual(FilterMode.all().value, manual.value)


if __name__ == "__main__":
    unittest.main()

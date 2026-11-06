import unittest

from algo_engine.base.c_market_data import c_market_data as md

FilterMode = md.FilterMode


class TestFilterMode(unittest.TestCase):
    # === Existing coverage (preserved & augmented) ===

    def test_bitwise_operations_and_contains(self):
        no_order = FilterMode.NO_ORDER
        no_trade = FilterMode.NO_TRADE
        combined = no_order | no_trade

        self.assertEqual(combined.value, no_order.value | no_trade.value)
        self.assertIn(no_order, combined)
        self.assertIn(no_trade, combined)
        self.assertEqual((combined & no_order).value, no_order.value)
        self.assertEqual((combined & no_trade).value, no_trade.value)

    def test_all_includes_every_non_auto_flag(self):
        flags = [
            FilterMode.NO_INTERNAL,
            FilterMode.NO_CANCEL,
            FilterMode.NO_AUCTION,
            FilterMode.NO_BREAK,
            FilterMode.NO_ORDER,
            FilterMode.NO_TRADE,
            FilterMode.NO_TICK,
        ]
        all_mode = FilterMode.all()
        for flag in flags:
            with self.subTest(flag=flag.name):
                self.assertIn(flag, all_mode)

    def test_auto_flag_exists_and_excluded_from_all(self):
        self.assertEqual(FilterMode.AUTO.value, 1)
        self.assertNotIn(FilterMode.AUTO, FilterMode.all())

    def test_no_break_in_all(self):
        self.assertIn(FilterMode.NO_BREAK, FilterMode.all())

    def test_invert_preserves_auto_bit(self):
        """AUTO is a meta-flag: inversion must not toggle it."""
        # Source has AUTO → inverted should keep AUTO
        source_with_auto = FilterMode.AUTO | FilterMode.NO_ORDER
        inv_with = ~source_with_auto
        self.assertIn(FilterMode.AUTO, inv_with)
        self.assertNotIn(FilterMode.NO_ORDER, inv_with)
        self.assertIn(FilterMode.NO_TRADE, inv_with)

        # Source without AUTO → inverted should NOT gain AUTO
        source_without_auto = FilterMode.NO_ORDER
        inv_without = ~source_without_auto
        self.assertNotIn(FilterMode.AUTO, inv_without)
        self.assertNotIn(FilterMode.NO_ORDER, inv_without)
        self.assertIn(FilterMode.NO_TRADE, inv_without)

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
                | FilterMode.NO_BREAK
                | FilterMode.NO_ORDER
                | FilterMode.NO_TRADE
                | FilterMode.NO_TICK
        )
        self.assertEqual(FilterMode.all().value, manual.value)

    # === Frozen semantics ===

    def test_class_constants_are_frozen(self):
        for const in [
            FilterMode.AUTO,
            FilterMode.NO_INTERNAL,
            FilterMode.NO_CANCEL,
            FilterMode.NO_AUCTION,
            FilterMode.NO_BREAK,
            FilterMode.NO_ORDER,
            FilterMode.NO_TRADE,
            FilterMode.NO_TICK,
        ]:
            with self.subTest(const=const.name):
                self.assertTrue(const.frozen)

    def test_frozen_prevents_inplace_ops(self):
        frozen = FilterMode.NO_ORDER
        with self.assertRaises(ValueError):
            frozen |= FilterMode.NO_TRADE
        with self.assertRaises(ValueError):
            frozen &= FilterMode.NO_TRADE
        with self.assertRaises(ValueError):
            frozen ^= FilterMode.NO_TRADE

    def test_frozen_prevents_enable_disable(self):
        with self.assertRaises(ValueError):
            FilterMode.NO_ORDER.enable_flags(FilterMode.NO_TRADE.value)
        with self.assertRaises(ValueError):
            FilterMode.NO_TRADE.disable_flag(FilterMode.NO_TRADE.value)

    # === Freeze / Unfreeze ===

    def test_freeze_unfreeze_toggle(self):
        fm: FilterMode = FilterMode[FilterMode.AUTO.value | FilterMode.NO_ORDER.value]
        self.assertTrue(fm.frozen, "fresh instance should be frozen by default")

        fm.unfreeze()
        self.assertFalse(fm.frozen)

        fm.freeze()
        self.assertTrue(fm.frozen)

    def test_unfrozen_allows_enable_disable(self):
        fm = FilterMode[FilterMode.AUTO.value | FilterMode.NO_ORDER.value]
        fm.unfreeze()

        fm.enable_flags(FilterMode.NO_TRADE)
        self.assertIn(FilterMode.NO_TRADE, fm)

        fm.disable_flag(FilterMode.NO_ORDER)
        self.assertNotIn(FilterMode.NO_ORDER, fm)
        self.assertIn(FilterMode.NO_TRADE, fm)  # still there

    def test_unfrozen_allows_inplace_ops(self):
        fm = FilterMode[FilterMode.AUTO.value | FilterMode.NO_ORDER.value]
        fm.unfreeze()

        fm |= FilterMode.NO_TRADE
        self.assertIn(FilterMode.NO_TRADE, fm)

        fm &= ~FilterMode.NO_ORDER.value & 0xFE
        self.assertNotIn(FilterMode.NO_ORDER, fm)

    def test_enable_disable_accept_raw_int(self):
        fm = FilterMode[FilterMode.AUTO.value | FilterMode.NO_ORDER.value]
        fm.unfreeze()

        fm.enable_flags(FilterMode.NO_TRADE.value)
        self.assertIn(FilterMode.NO_TRADE, fm)

        fm.disable_flag(FilterMode.NO_ORDER.value)
        self.assertNotIn(FilterMode.NO_ORDER, fm)

    def test_refreeze_prevents_mutation(self):
        fm = FilterMode[FilterMode.AUTO.value | FilterMode.NO_ORDER.value]
        fm.unfreeze()
        fm.freeze()

        with self.assertRaises(ValueError):
            fm.enable_flags(FilterMode.NO_TRADE)
        with self.assertRaises(ValueError):
            fm |= FilterMode.NO_TRADE

    # === New coverage: __xor__ operator ===

    def test_xor_operator(self):
        a = FilterMode.NO_ORDER | FilterMode.NO_TRADE  # 0x60
        b = FilterMode.NO_TRADE | FilterMode.NO_TICK   # 0xC0
        xor_result = a ^ b                               # ~0x60 & 0xC0 | 0x60 & ~0xC0

        self.assertEqual(xor_result.value, a.value ^ b.value)
        # NO_ORDER (only in a) should be in xor_result
        self.assertIn(FilterMode.NO_ORDER, xor_result)
        # NO_TICK (only in b) should be in xor_result
        self.assertIn(FilterMode.NO_TICK, xor_result)
        # NO_TRADE (in both) should NOT be in xor_result
        self.assertNotIn(FilterMode.NO_TRADE, xor_result)

    # === New coverage: widened type acceptance ===

    def test_accepts_raw_int_flag(self):
        fm = FilterMode.NO_ORDER | FilterMode.NO_TRADE
        raw = FilterMode.NO_ORDER.value

        self.assertTrue(fm == (fm.value))
        self.assertTrue((fm & raw).value == raw)
        self.assertTrue((fm | raw).value == fm.value)
        self.assertIn(raw, fm)

        # contains with raw int that is a subset
        self.assertIn(FilterMode.NO_TRADE.value, fm)

    # === Property coverage ===

    def test_name_property(self):
        combined = FilterMode.NO_ORDER | FilterMode.NO_TRADE
        name = combined.name
        self.assertIn("NO_ORDER", name)
        self.assertIn("NO_TRADE", name)

    def test_flags_property_returns_filtermode_instances(self):
        combined = FilterMode.NO_ORDER | FilterMode.NO_AUCTION
        flags = combined.flags
        self.assertIsInstance(flags, list)
        for f in flags:
            self.assertIsInstance(f, FilterMode)
        flag_values = {f.value for f in flags}
        self.assertIn(FilterMode.NO_ORDER.value, flag_values)
        self.assertIn(FilterMode.NO_AUCTION.value, flag_values)


if __name__ == "__main__":
    unittest.main()

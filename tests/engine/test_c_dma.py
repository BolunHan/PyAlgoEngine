import datetime
import gc
import json
import unittest
import uuid

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
    ALGO_ENGINE,
    AlgoStatus,
    INVENTORY as ENGINE_INVENTORY,
    Inventory,
    RefusalWindowAfterRule,
    RefusalWindowBeforeRule,
    RefusalWindowBetweenRule,
    PositionCallbackEvent,
    PositionManagementService,
    RiskLimits,
    RiskLimitsMapView,
    RiskProfile,
    RiskRule,
    RiskRuleTemplate,
    RiskType,
    RiskViolation,
)
from algo_engine.exchange_profile import PROFILE_CN
from algo_engine.strategy import INVENTORY, RISK_PROFILE


TICKER = '600000.SH'
RISK_TICKER = 'RISK.SH'
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
        risk_profile.set_rule(key, 1e9)


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
        feed_price(RISK_TICKER)

    @classmethod
    def tearDownClass(cls) -> None:
        PROFILE_CN.deactivate()

    def setUp(self) -> None:
        RISK_PROFILE.clear()

    def test_00_incomplete_profile_refuses_every_order(self) -> None:
        # A fresh profile defines nothing and refuses every transaction; the
        # process profile only reaches this state before clear() defines its
        # globals, so the case uses a local profile.
        fresh = RiskProfile()

        with self.assertLogs('PyAlgoEngine.Engine', level='ERROR') as logs:
            with self.assertRaises(RiskViolation):
                fresh.check(make_order(ticker=RISK_TICKER))
        self.assertIn('no rule constrains', logs.output[0])
        self.assertEqual(fresh.rules['entry'], set())

        order = make_order(ticker=RISK_TICKER)
        with self.assertRaises(RiskViolation):
            fresh.check(order)
        self.assertEqual(order.order_state.name, 'STATE_IDLE')

    def test_01_defined_rules_constrain_their_side_only(self) -> None:
        # A local profile: only the rules that are defined constrain, and only
        # the side they apply to.
        feed_price('GLOBAL.SH')

        risk_profile = RiskProfile()

        with self.assertLogs('PyAlgoEngine.Engine', level='ERROR') as logs:
            with self.assertRaises(RiskViolation):
                risk_profile.check(make_order(ticker=RISK_TICKER))
        self.assertIn('no rule constrains a LONG order', logs.output[0])

        # One long rule constrains the long side only.
        risk_profile.set_rule('max_trade_long', 100., ticker=RISK_TICKER)

        self.assertTrue(risk_profile.check(make_order(10., ticker=RISK_TICKER)))
        with self.assertRaises(RiskViolation):
            risk_profile.check(make_order(500., ticker=RISK_TICKER))

        with self.assertLogs('PyAlgoEngine.Engine', level='ERROR') as logs:
            with self.assertRaises(RiskViolation):
                risk_profile.check(make_order(10., side=SIDE_SHORT, ticker=RISK_TICKER))
        self.assertIn('no rule constrains a SHORT order', logs.output[0])

        # A global long rule constrains the long side of any ticker.
        risk_profile.set_rule('max_ttl_notional_long', 1e9)

        self.assertTrue(risk_profile.check(make_order(10., ticker='GLOBAL.SH')))

        with self.assertLogs('PyAlgoEngine.Engine', level='ERROR'):
            with self.assertRaises(RiskViolation):
                risk_profile.check(make_order(10., side=SIDE_SHORT, ticker='GLOBAL.SH'))

    def test_02_complete_profile_passes_and_limits_reject(self) -> None:
        setup_profile_rules(RISK_PROFILE, RISK_TICKER)

        self.assertTrue(RISK_PROFILE.check(make_order(10., ticker=RISK_TICKER)))
        with self.assertRaises(RiskViolation):
            RISK_PROFILE.check(make_order(5000., ticker=RISK_TICKER))

        risk = RISK_PROFILE.get(RISK_TICKER)
        self.assertEqual(risk['max_trade_long'], 1000.)
        self.assertEqual(risk['max_ttl_notional_long'], 1e9)

    def test_03_rules_snapshot_and_json_round_trip(self) -> None:
        setup_profile_rules(RISK_PROFILE, RISK_TICKER)

        rules = RISK_PROFILE.rules
        self.assertIn(RISK_TICKER, rules['entry'])
        self.assertEqual(rules['max_trade_long'][RISK_TICKER], 1000.)

        payload = RISK_PROFILE.to_json(fmt='dict')
        self.assertEqual(payload['entry'], [RISK_TICKER])

        RISK_PROFILE.clear()
        self.assertEqual(RISK_PROFILE.rules['max_trade_long'], {})

        RISK_PROFILE.from_json(json.dumps(payload))
        self.assertEqual(RISK_PROFILE.rules['max_trade_long'][RISK_TICKER], 1000.)
        self.assertTrue(RISK_PROFILE.check(make_order(ticker=RISK_TICKER)))

    def test_04_clear_disables_every_rule(self) -> None:
        setup_profile_rules(RISK_PROFILE, RISK_TICKER)
        RISK_PROFILE.clear()

        self.assertEqual(RISK_PROFILE.rules['entry'], set())
        self.assertIsNone(RISK_PROFILE.rules['max_net_notional_long'])

        # A cleared profile enables nothing: the next check is incomplete.
        with self.assertLogs('PyAlgoEngine.Engine', level='ERROR'):
            with self.assertRaises(RiskViolation):
                RISK_PROFILE.check(make_order(10., ticker=RISK_TICKER))

    def test_05_rule_scope_is_enforced(self) -> None:
        with self.assertLogs('PyAlgoEngine.Engine', level='ERROR'):
            RISK_PROFILE.set_rule('max_trade_long', 1.)

        with self.assertLogs('PyAlgoEngine.Engine', level='ERROR'):
            RISK_PROFILE.set_rule('max_net_notional_long', 1., ticker=RISK_TICKER)

        with self.assertLogs('PyAlgoEngine.Engine', level='ERROR'):
            RISK_PROFILE.set_rule('not_a_rule', 1., ticker=RISK_TICKER)

    def _setup_generous_rules(self, ticker: str = RISK_TICKER) -> None:
        setup_profile_rules(RISK_PROFILE, ticker)

        for key in ('max_trade_long', 'max_trade_short', 'max_exposure_long', 'max_exposure_short'):
            RISK_PROFILE.set_rule(key, 1e15, ticker=ticker)

        for key in ('max_notional_long', 'max_notional_short'):
            RISK_PROFILE.set_rule(key, 1e15, ticker=ticker)

        for key in ('max_ttl_notional_long', 'max_ttl_notional_short', 'max_net_notional_long', 'max_net_notional_short'):
            RISK_PROFILE.set_rule(key, 1e15)

        RISK_PROFILE.set_rule('max_percentile', 1.0, ticker=ticker)

    def test_06_every_rule_rejects_and_accepts(self) -> None:
        # Each rule is tightened alone; the case asserts the violation it
        # names, then the same order passes once the rule admits it.
        cases = (
            ('max_trade_long', SIDE_LONG, 10., 5., 15., 'RiskProfile.TradeLimit.Long'),
            ('max_trade_short', SIDE_SHORT, 10., 5., 15., 'RiskProfile.TradeLimit.Short'),
            ('max_exposure_long', SIDE_LONG, 10., 5., 15., 'RiskProfile.ExposureLimit.Long'),
            ('max_exposure_short', SIDE_SHORT, 10., 5., 15., 'RiskProfile.ExposureLimit.Short'),
            ('max_notional_long', SIDE_LONG, 10., 50., 200., 'RiskProfile.NotionalLimit.Long'),
            ('max_notional_short', SIDE_SHORT, 10., 50., 200., 'RiskProfile.NotionalLimit.Short'),
            ('max_percentile', SIDE_LONG, 1e9, 0.5, 1.0, 'RiskProfile.PercentileLimit.Long'),
            ('max_percentile', SIDE_SHORT, 1e9, 0.5, 1.0, 'RiskProfile.PercentileLimit.Short'),
            ('max_ttl_notional_long', SIDE_LONG, 1e9, 50., 1e12, 'RiskProfile.NotionalLimit.PortfolioTotal.Long'),
            ('max_ttl_notional_short', SIDE_SHORT, 1e9, 50., 1e12, 'RiskProfile.NotionalLimit.PortfolioTotal.Short'),
            ('max_net_notional_long', SIDE_LONG, 1e9, 50., 1e12, 'RiskProfile.NotionalLimit.PortfolioNet.Long'),
            ('max_net_notional_short', SIDE_SHORT, 1e9, 50., 1e12, 'RiskProfile.NotionalLimit.PortfolioNet.Short'),
        )

        for key, side, volume, rejected, accepted, violation in cases:
            is_global = key.startswith('max_ttl') or key.startswith('max_net')

            with self.subTest(rule=key, side=int(side), outcome='rejected'):
                RISK_PROFILE.clear()
                self._setup_generous_rules()

                if is_global:
                    RISK_PROFILE.set_rule(key, rejected)
                else:
                    RISK_PROFILE.set_rule(key, rejected, ticker=RISK_TICKER)

                with self.assertLogs('PyAlgoEngine.Engine', level='ERROR') as logs:
                    with self.assertRaises(RiskViolation):
                        RISK_PROFILE.check(make_order(volume, side=side, ticker=RISK_TICKER))

                self.assertIn(violation, logs.output[0])

            with self.subTest(rule=key, side=int(side), outcome='accepted'):
                RISK_PROFILE.clear()
                self._setup_generous_rules()

                if is_global:
                    RISK_PROFILE.set_rule(key, accepted)
                else:
                    RISK_PROFILE.set_rule(key, accepted, ticker=RISK_TICKER)

                # A percentile of 1 is inactive; every other accepted value
                # admits the same order.
                self.assertTrue(RISK_PROFILE.check(make_order(volume, side=side, ticker=RISK_TICKER)))

    def test_07_missing_price_is_refused(self) -> None:
        setup_profile_rules(RISK_PROFILE, 'NOPRICE.SH')

        with self.assertLogs('PyAlgoEngine.Engine', level='ERROR') as logs:
            with self.assertRaises(RiskViolation):
                RISK_PROFILE.check(make_order(1., ticker='NOPRICE.SH'))

        self.assertIn('RiskProfile.Internal.Price', logs.output[0])
        self.assertIn('no valid market price', logs.output[0])

        # Price validity is unconditional: a volume-only rule needs the price too.
        trade_only = RiskProfile()
        trade_only.set_rule('max_trade_long', 1e9, ticker='NOPRICE.SH')

        with self.assertLogs('PyAlgoEngine.Engine', level='ERROR') as logs:
            with self.assertRaises(RiskViolation):
                trade_only.check(make_order(1., ticker='NOPRICE.SH'))
        self.assertIn('RiskProfile.Internal.Price', logs.output[0])

    def test_08_whitelist_and_blacklist(self) -> None:
        setup_profile_rules(RISK_PROFILE, RISK_TICKER)

        self.assertTrue(RISK_PROFILE.check(make_order(10., ticker=RISK_TICKER)))

        # An absent whitelist gates nothing; a present one refuses a ticker it
        # does not list.
        RISK_PROFILE.set_whitelist('OTHER.SH')
        with self.assertLogs('PyAlgoEngine.Engine', level='ERROR') as logs:
            with self.assertRaises(RiskViolation):
                RISK_PROFILE.check(make_order(10., ticker=RISK_TICKER))
        self.assertIn('RiskProfile.TickerLimit.Unlisted', logs.output[0])

        RISK_PROFILE.set_whitelist(RISK_TICKER)
        self.assertTrue(RISK_PROFILE.check(make_order(10., ticker=RISK_TICKER)))

        # A ticker cannot be on both lists: the blacklist add is refused.
        with self.assertLogs('PyAlgoEngine.Engine', level='ERROR') as logs:
            RISK_PROFILE.set_blacklist(RISK_TICKER)
        self.assertIn('is whitelisted', logs.output[0])
        self.assertTrue(RISK_PROFILE.check(make_order(10., ticker=RISK_TICKER)))

        # The reverse conflict is refused too.
        RISK_PROFILE.set_blacklist('BAN.SH')
        with self.assertLogs('PyAlgoEngine.Engine', level='ERROR') as logs:
            RISK_PROFILE.set_whitelist('BAN.SH')
        self.assertIn('is blacklisted', logs.output[0])
        RISK_PROFILE.set_blacklist('BAN.SH', enabled=False)

        # Dropping it from the whitelist and blacklisting it refuses it.
        RISK_PROFILE.set_whitelist(RISK_TICKER, enabled=False)
        RISK_PROFILE.set_blacklist(RISK_TICKER)
        with self.assertLogs('PyAlgoEngine.Engine', level='ERROR') as logs:
            with self.assertRaises(RiskViolation):
                RISK_PROFILE.check(make_order(10., ticker=RISK_TICKER))
        self.assertIn('RiskProfile.TickerLimit.Blocked', logs.output[0])

        RISK_PROFILE.set_blacklist(RISK_TICKER, enabled=False)
        RISK_PROFILE.set_whitelist('OTHER.SH', enabled=False)
        self.assertTrue(RISK_PROFILE.check(make_order(10., ticker=RISK_TICKER)))
        self.assertEqual(RISK_PROFILE.rules['whitelist'], set())
        self.assertEqual(RISK_PROFILE.rules['blacklist'], set())

    def test_10_refusal_window_before_rule(self) -> None:
        setup_profile_rules(RISK_PROFILE, RISK_TICKER)

        # The MDS clock is 10:00: a 09:45 ban is over.
        rule = RefusalWindowBeforeRule(datetime.time(9, 45))
        self.assertEqual(rule.end, datetime.time(9, 45))

        rule.register(RISK_PROFILE, RISK_TICKER)
        self.assertEqual(rule.ticker, RISK_TICKER)
        self.assertEqual(RISK_PROFILE.rules['custom'][RISK_TICKER], [6001])
        self.assertTrue(RISK_PROFILE.check(make_order(10., ticker=RISK_TICKER)))

        # A 10:15 ban is in force at 10:00.
        later = RefusalWindowBeforeRule(datetime.time(10, 15))
        later.register(RISK_PROFILE, RISK_TICKER)
        self.assertEqual(RISK_PROFILE.rules['custom'][RISK_TICKER], [6001, 6001])

        with self.assertLogs('PyAlgoEngine.Engine', level='ERROR') as logs:
            with self.assertRaises(RiskViolation):
                RISK_PROFILE.check(make_order(10., ticker=RISK_TICKER))
        self.assertIn('RiskProfile.Custom.6001', logs.output[0])
        self.assertIn('refusal_window', logs.output[0])
        self.assertIn('10:15:00', logs.output[0])

        later.unregister()
        self.assertTrue(RISK_PROFILE.check(make_order(10., ticker=RISK_TICKER)))

        # A second registration of the same rule is refused.
        with self.assertRaises(RuntimeError):
            rule.register(RISK_PROFILE, RISK_TICKER)

        rule.unregister()
        self.assertEqual(RISK_PROFILE.rules['custom'], {})

    def test_11_custom_rule_needs_an_absolute_constraint(self) -> None:
        # The percentile limit is relative and a custom rule never completes a
        # profile: only an absolute risk value constrains the side.
        risk_profile = RiskProfile()
        risk_profile.set_rule('max_percentile', 1e-9, ticker=RISK_TICKER)

        with self.assertLogs('PyAlgoEngine.Engine', level='ERROR') as logs:
            with self.assertRaises(RiskViolation):
                risk_profile.check(make_order(10., ticker=RISK_TICKER))
        self.assertIn('no rule constrains', logs.output[0])

        rule = RefusalWindowBeforeRule(datetime.time(10, 30))
        rule.register(risk_profile, RISK_TICKER)

        with self.assertLogs('PyAlgoEngine.Engine', level='ERROR') as logs:
            with self.assertRaises(RiskViolation):
                risk_profile.check(make_order(10., ticker=RISK_TICKER))
        self.assertIn('no rule constrains', logs.output[0])

        # One absolute rule per side constrains it; the custom rule then runs.
        risk_profile.set_rule('max_percentile', 1.0, ticker=RISK_TICKER)
        risk_profile.set_rule('max_trade_long', 1e15, ticker=RISK_TICKER)
        risk_profile.set_rule('max_trade_short', 1e15, ticker=RISK_TICKER)

        with self.assertLogs('PyAlgoEngine.Engine', level='ERROR') as logs:
            with self.assertRaises(RiskViolation):
                risk_profile.check(make_order(10., ticker=RISK_TICKER))
        self.assertIn('refusal_window', logs.output[0])

        rule.unregister()

    def test_12_rule_lifetime_follows_the_profile(self) -> None:
        rule = RefusalWindowBeforeRule(datetime.time(10, 30))
        rule.register(RISK_PROFILE, RISK_TICKER)
        self.assertEqual(RISK_PROFILE.rules['custom'][RISK_TICKER], [6001])

        # The profile owns the node: dropping the wrapper leaves the rule.
        del rule
        gc.collect()
        self.assertEqual(RISK_PROFILE.rules['custom'][RISK_TICKER], [6001])

        # Only an explicit unregister (or the profile's teardown) removes it.
        # A global registration carries no ticker and bans every ticker.
        local = RiskProfile()
        setup_profile_rules(local, RISK_TICKER)

        rule = RefusalWindowBeforeRule(datetime.time(10, 30))
        rule.register_global(local)
        self.assertIsNone(rule.ticker)
        self.assertEqual(local.rules['custom_global'], [6001])

        with self.assertLogs('PyAlgoEngine.Engine', level='ERROR') as logs:
            with self.assertRaises(RiskViolation):
                local.check(make_order(10., ticker=RISK_TICKER))
        self.assertIn('refusal_window', logs.output[0])

        rule.unregister()
        self.assertTrue(local.check(make_order(10., ticker=RISK_TICKER)))
        self.assertEqual(local.rules['custom_global'], [])

    def test_13_custom_rule_is_bound_to_its_ticker(self) -> None:
        setup_profile_rules(RISK_PROFILE, RISK_TICKER)

        rule = RefusalWindowBeforeRule(datetime.time(10, 30))
        rule.register(RISK_PROFILE, 'OTHER.SH')
        self.assertEqual(RISK_PROFILE.rules['custom']['OTHER.SH'], [6001])

        # The RISK ticker is unaffected by another ticker's rule.
        self.assertTrue(RISK_PROFILE.check(make_order(10., ticker=RISK_TICKER)))

        rule.unregister()

    def test_14_limits_view_and_rule_template(self) -> None:
        self.assertTrue(issubclass(RiskViolation, Exception))
        self.assertFalse(hasattr(RiskProfile, 'Risk'))

        # A limit set is hosted by pointer: the view reads the defined rules
        # and reports an undefined one as None. A view bound before the first
        # write sees the block on its insert event, not a null slot.
        local = RiskProfile()

        limits_view = local.limits
        local.set_rule('max_trade_long', 100., ticker=RISK_TICKER)

        self.assertIsInstance(limits_view, RiskLimitsMapView)
        self.assertIsInstance(limits_view[RISK_TICKER], RiskLimits)
        self.assertIs(local.limits, limits_view)
        self.assertIn(RISK_TICKER, limits_view)

        limits = limits_view[RISK_TICKER]
        self.assertIsInstance(limits, RiskLimits)
        self.assertEqual(limits.max_trade_long, 100.)
        self.assertIsNone(limits.max_trade_short)
        self.assertIsNone(limits.max_notional_long)
        self.assertEqual(list(limits.rules), [])

        # Writing through the view is refused.
        with self.assertRaises(PermissionError):
            limits_view['OTHER.SH'] = limits

        rule = RefusalWindowBeforeRule(datetime.time(10, 30))
        self.assertIsInstance(rule, RiskRuleTemplate)
        self.assertFalse(rule.is_registered)

        rule.register(local, RISK_TICKER)
        self.assertTrue(rule.is_registered)
        self.assertEqual(rule.risk_rule_code, 6001)
        self.assertEqual(rule.name, 'refusal_window')
        self.assertIsInstance(rule.rule_id, int)

        # The limit set lists the rule as a bare template view of the node.
        templates = list(limits.rules)
        self.assertEqual(len(templates), 1)
        self.assertIsInstance(templates[0], RiskRuleTemplate)
        self.assertEqual(templates[0].rule_id, rule.rule_id)
        self.assertFalse(templates[0].is_registered)

        rule.unregister()
        self.assertFalse(rule.is_registered)
        self.assertEqual(list(limits.rules), [])

        # The ban time is the wrapper's own configuration: it survives the node.
        self.assertEqual(rule.end, datetime.time(10, 30))
        with self.assertRaises(RuntimeError):
            _ = rule.ticker

        # Clearing the profile frees the node with its adopted state; the
        # node-backed facts are husks from then on.
        late = RefusalWindowBeforeRule(datetime.time(10, 30))
        late.register(local, RISK_TICKER)
        local.clear()

        self.assertEqual(dict(local.limits), {})
        self.assertFalse(late.is_registered)
        self.assertEqual(late.end, datetime.time(10, 30))
        with self.assertRaises(RuntimeError):
            _ = late.ticker

        del late

    def test_15_rule_enable_disable(self) -> None:
        setup_profile_rules(RISK_PROFILE, RISK_TICKER)

        rule = RefusalWindowBeforeRule(datetime.time(10, 30))
        rule.register(RISK_PROFILE, RISK_TICKER)
        self.assertTrue(rule.enabled)

        # The MDS clock is 10:00: the 10:30 ban rejects.
        with self.assertLogs('PyAlgoEngine.Engine', level='ERROR') as logs:
            with self.assertRaises(RiskViolation):
                RISK_PROFILE.check(make_order(10., ticker=RISK_TICKER))
        self.assertIn('refusal_window', logs.output[0])

        # A disabled rule stays linked but the check skips it.
        rule.disable()
        self.assertFalse(rule.enabled)
        self.assertTrue(RISK_PROFILE.check(make_order(10., ticker=RISK_TICKER)))
        self.assertEqual(RISK_PROFILE.rules['custom'][RISK_TICKER], [6001])

        rule.enable()
        self.assertTrue(rule.enabled)
        with self.assertLogs('PyAlgoEngine.Engine', level='ERROR'):
            with self.assertRaises(RiskViolation):
                RISK_PROFILE.check(make_order(10., ticker=RISK_TICKER))

        rule.unregister()

    def test_16_rule_and_type_enums_mirror_the_c_enum(self) -> None:
        # The rule key is the member name; the global split is the C predicate.
        self.assertEqual(RiskRule['max_trade_long'], RiskRule.max_trade_long)
        self.assertEqual(RiskRule.max_trade_long, 1)
        self.assertFalse(RiskRule.max_trade_long.is_global)
        self.assertTrue(RiskRule.max_ttl_notional_long.is_global)

        # set_rule takes the member or its name.
        RISK_PROFILE.set_rule(RiskRule.max_trade_long, 100., ticker=RISK_TICKER)
        RISK_PROFILE.set_rule('max_trade_short', 200., ticker=RISK_TICKER)

        risk = RISK_PROFILE.get(RISK_TICKER)
        self.assertEqual(risk['max_trade_long'], 100.)
        self.assertEqual(risk['max_trade_short'], 200.)

        # The violation types its type from the C record it wraps.
        violation = RiskViolation(RiskType.trade_long, 1001, 'rejected')
        self.assertIs(violation.type, RiskType.trade_long)
        self.assertEqual(violation.code, 1001)
        self.assertEqual(violation.msg, 'rejected')

        with self.assertRaises(ValueError):
            RiskViolation(-2, 0, 'invalid type')

    def test_17_rule_values_are_validated(self) -> None:
        # NaN and negative values are refused.
        with self.assertLogs('PyAlgoEngine.Engine', level='ERROR'):
            RISK_PROFILE.set_rule('max_trade_long', float('nan'), ticker=RISK_TICKER)
        with self.assertLogs('PyAlgoEngine.Engine', level='ERROR'):
            RISK_PROFILE.set_rule('max_trade_long', -1., ticker=RISK_TICKER)

        self.assertNotIn('max_trade_long', RISK_PROFILE.get(RISK_TICKER))

        # An infinite value disables the rule and resets it to 0.
        RISK_PROFILE.set_rule('max_trade_long', 100., ticker=RISK_TICKER)
        self.assertEqual(RISK_PROFILE.get(RISK_TICKER)['max_trade_long'], 100.)

        RISK_PROFILE.set_rule('max_trade_long', float('inf'), ticker=RISK_TICKER)
        self.assertNotIn('max_trade_long', RISK_PROFILE.get(RISK_TICKER))
        self.assertEqual(RISK_PROFILE.rules['max_trade_long'], {})
        self.assertIsNone(RISK_PROFILE.limits[RISK_TICKER].max_trade_long)

        # A percentile of one or more disables it too.
        RISK_PROFILE.set_rule('max_percentile', 1., ticker=RISK_TICKER)
        self.assertNotIn('max_percentile', RISK_PROFILE.get(RISK_TICKER))

    def test_09_gates_json_round_trip(self) -> None:
        setup_profile_rules(RISK_PROFILE, RISK_TICKER)
        RISK_PROFILE.set_whitelist(RISK_TICKER)
        RISK_PROFILE.set_blacklist('BAN.SH')

        payload = RISK_PROFILE.to_json(fmt='dict')
        self.assertEqual(payload['whitelist'], [RISK_TICKER])
        self.assertEqual(payload['blacklist'], ['BAN.SH'])

        RISK_PROFILE.clear()
        self.assertEqual(RISK_PROFILE.rules['whitelist'], set())

        RISK_PROFILE.from_json(payload)
        self.assertEqual(RISK_PROFILE.rules['whitelist'], {RISK_TICKER})
        self.assertEqual(RISK_PROFILE.rules['blacklist'], {'BAN.SH'})
        self.assertTrue(RISK_PROFILE.check(make_order(10., ticker=RISK_TICKER)))

    def test_18_refusal_window_variants(self) -> None:
        setup_profile_rules(RISK_PROFILE, RISK_TICKER)

        def clock_at(hour: int, minute: int) -> None:
            MDS.on_market_data(TransactionData(
                ticker=RISK_TICKER,
                price=10.5,
                volume=100,
                timestamp=datetime.datetime(2024, 11, 11, hour, minute).timestamp(),
                side=TransactionSide.SIDE_LONG_OPEN,
            ))

        try:
            # The clock is 10:00: a rule refusing from 10:00 rejects now.
            after = RefusalWindowAfterRule(datetime.time(10, 0))
            after.register(RISK_PROFILE, RISK_TICKER)

            with self.assertLogs('PyAlgoEngine.Engine', level='ERROR') as logs:
                with self.assertRaises(RiskViolation):
                    RISK_PROFILE.check(make_order(10., ticker=RISK_TICKER))
            self.assertIn('bans trading after 10:00:00', logs.output[0])

            after.unregister()

            # The period [10:00, 10:30) refuses at 10:00 and releases at 10:30.
            between = RefusalWindowBetweenRule(datetime.time(10, 0), datetime.time(10, 30))
            between.register(RISK_PROFILE, RISK_TICKER)

            with self.assertLogs('PyAlgoEngine.Engine', level='ERROR') as logs:
                with self.assertRaises(RiskViolation):
                    RISK_PROFILE.check(make_order(10., ticker=RISK_TICKER))
            self.assertIn('bans trading between 10:00:00 and 10:30:00', logs.output[0])

            clock_at(10, 30)
            self.assertTrue(RISK_PROFILE.check(make_order(10., ticker=RISK_TICKER)))

            between.unregister()

            # An end-only window still refuses before its end.
            before = RefusalWindowBeforeRule(datetime.time(10, 31))
            before.register(RISK_PROFILE, RISK_TICKER)

            with self.assertLogs('PyAlgoEngine.Engine', level='ERROR') as logs:
                with self.assertRaises(RiskViolation):
                    RISK_PROFILE.check(make_order(10., ticker=RISK_TICKER))
            self.assertIn('bans trading before 10:31:00', logs.output[0])

            before.unregister()
        finally:
            clock_at(10, 0)

    def test_19_margin_rate_designation(self) -> None:
        # A finite rate designates the ticker, NaN removes the designation.
        self.assertIsNone(RISK_PROFILE.get_margin_rate(RISK_TICKER))

        RISK_PROFILE.set_margin_rate(RISK_TICKER, 0.5)
        self.assertEqual(RISK_PROFILE.get_margin_rate(RISK_TICKER), 0.5)

        RISK_PROFILE.set_margin_rate(RISK_TICKER, 1.)
        self.assertEqual(RISK_PROFILE.get_margin_rate(RISK_TICKER), 1.)

        # A negative or infinite rate is refused and the designation stands.
        with self.assertLogs('PyAlgoEngine.Engine', level='ERROR'):
            RISK_PROFILE.set_margin_rate(RISK_TICKER, -1.)
        self.assertEqual(RISK_PROFILE.get_margin_rate(RISK_TICKER), 1.)

        with self.assertLogs('PyAlgoEngine.Engine', level='ERROR'):
            RISK_PROFILE.set_margin_rate(RISK_TICKER, float('inf'))
        self.assertEqual(RISK_PROFILE.get_margin_rate(RISK_TICKER), 1.)

        RISK_PROFILE.set_margin_rate(RISK_TICKER, float('nan'))
        self.assertIsNone(RISK_PROFILE.get_margin_rate(RISK_TICKER))

        # clear() frees the designation map.
        RISK_PROFILE.set_margin_rate(RISK_TICKER, 0.5)
        RISK_PROFILE.clear()
        self.assertIsNone(RISK_PROFILE.get_margin_rate(RISK_TICKER))


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

        first = INVENTORY.get_inv(TICKER)
        self.assertEqual(first.volume, 100.)
        self.assertEqual(first.price, 10.)
        self.assertEqual(first.fee, 0.)
        self.assertIsNone(INVENTORY.get_inv(TICKER, Direction.DIRECTION_SHORT))

        INVENTORY.use_inv(TICKER, 50.)
        self.assertEqual(INVENTORY.available_volume(TICKER), 150.)

        # The getter is a live view: recalling the first lot releases it.
        INVENTORY.recall(TICKER, 120.)

        with self.assertRaises(RuntimeError):
            _ = first.volume

        remaining = INVENTORY.get_inv(TICKER)
        self.assertEqual(remaining.volume, 100.)
        self.assertEqual(remaining.recalled, 20.)
        self.assertEqual(INVENTORY.available_volume(TICKER), 30.)

        INVENTORY.clear()
        self.assertEqual(INVENTORY.tickers, set())
        self.assertEqual(INVENTORY.available_volume(TICKER), 0.)

    def test_02_json_round_trip(self) -> None:
        # Recalled volume is runtime state: it is not stored when a lot is added.
        INVENTORY.add_inv(Inventory.Entry(TICKER, 100., 10., Inventory.SecurityType.Stock, fee=1., recalled=5.))

        payload = INVENTORY.to_json(fmt='dict')
        key = f'{TICKER}.DIRECTION_LONG'

        self.assertIn(key, payload)
        self.assertEqual(payload[key]['inv'][0]['fee'], 1.)
        self.assertEqual(payload[key]['inv'][0]['recalled'], 0.)

        INVENTORY.clear()
        INVENTORY.from_json(json.dumps(payload))

        entry = INVENTORY.get_inv(TICKER)
        self.assertEqual(entry.volume, 100.)
        self.assertEqual(entry.fee, 1.)
        self.assertEqual(entry.recalled, 0.)
        self.assertEqual(entry.available, 100.)

    def test_03_inventory_is_the_process_singleton(self) -> None:
        self.assertIs(ENGINE_INVENTORY, INVENTORY)

    def test_04_inventory_corporate_actions(self) -> None:
        record_date = datetime.date(2024, 11, 1)
        ex_date = datetime.date(2024, 11, 11)

        INVENTORY.add_inv(Inventory.Entry(TICKER, 100., 10., Inventory.SecurityType.Stock))
        INVENTORY.add_inv(Inventory.Entry(TICKER, 100., 10., Inventory.SecurityType.Stock, direction=Direction.DIRECTION_SHORT))

        INVENTORY.cash_dividend(record_date, ex_date, 0.5)
        merged = INVENTORY.get_inv(TICKER)
        self.assertEqual(merged.volume, 100.)
        self.assertEqual(merged.price, 9.5)
        self.assertEqual(merged.notional, 950.)

        INVENTORY.stock_split(record_date, ex_date, 2.)
        merged = INVENTORY.get_inv(TICKER)
        self.assertEqual(merged.volume, 200.)
        self.assertAlmostEqual(merged.price, 9.5 / 2.)

        INVENTORY.stock_dividend(record_date, ex_date, 0.2)
        merged = INVENTORY.get_inv(TICKER)
        self.assertAlmostEqual(merged.volume, 240.)
        self.assertAlmostEqual(merged.price, 9.5 / 2.4)

        INVENTORY.stock_conversion(record_date, ex_date, TICKER, 'NEW.TICKER', 1.5)
        merged = INVENTORY.get_inv(TICKER)
        self.assertAlmostEqual(merged.volume, 360.)
        self.assertAlmostEqual(merged.price, 9.5 / 3.6)

        short_inv = INVENTORY.get_inv(TICKER, Direction.DIRECTION_SHORT)
        self.assertAlmostEqual(short_inv.volume, 360.)
        self.assertAlmostEqual(short_inv.price, 9.5 / 3.6)

    def test_05_event_wrappers_are_ccp_backed(self) -> None:
        record_date = datetime.date(2024, 11, 1)
        ex_date = datetime.date(2024, 11, 11)

        cash = Inventory.CashDividend(record_date, ex_date, 0.5)
        self.assertEqual(cash.record_date, record_date)
        self.assertEqual(cash.ex_date, ex_date)
        self.assertEqual(cash.dividend_per_share, 0.5)
        self.assertFalse(cash.applied)

        stock = Inventory.StockDividend(record_date, ex_date, 0.2)
        self.assertEqual(stock.dividend_per_share, 0.2)

        split = Inventory.StockSplit(record_date, ex_date, 2.)
        self.assertEqual(split.multiplier, 2.)

        conversion = Inventory.StockConversion(record_date, ex_date, TICKER, 'NEW.TICKER', 1.5)
        self.assertEqual(conversion.convert_from, TICKER)
        self.assertEqual(conversion.convert_to, 'NEW.TICKER')
        self.assertEqual(conversion.multiplier, 1.5)

    def test_06_invalid_direction_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            Inventory.Entry(TICKER, 100., 10., Inventory.SecurityType.Stock, direction=Direction.DIRECTION_NEUTRAL)

        with self.assertRaises(ValueError):
            INVENTORY.get_inv(TICKER, direction=Direction.DIRECTION_UNKNOWN)

        with self.assertRaises(ValueError):
            INVENTORY.available_volume(TICKER, direction=Direction.DIRECTION_NEUTRAL)

    def test_07_entry_corporate_actions(self) -> None:
        record_date = datetime.date(2024, 11, 1)
        ex_date = datetime.date(2024, 11, 11)

        entry = Inventory.Entry(TICKER, 100., 10., Inventory.SecurityType.Stock)

        entry.apply_cash_dividend(Inventory.CashDividend(record_date, ex_date, 0.5))
        self.assertEqual(entry.volume, 100.)
        self.assertEqual(entry.price, 9.5)
        self.assertEqual(entry.notional, 950.)

        entry = Inventory.Entry(TICKER, 100., 10., Inventory.SecurityType.Stock)

        entry.apply_stock_dividend(Inventory.StockDividend(record_date, ex_date, 0.2))
        self.assertAlmostEqual(entry.volume, 120.)
        self.assertAlmostEqual(entry.price, 10. / 1.2)
        self.assertEqual(entry.notional, 1000.)

        entry.apply_split(Inventory.StockSplit(record_date, ex_date, 2.))
        self.assertAlmostEqual(entry.volume, 240.)
        self.assertAlmostEqual(entry.price, 10. / 2.4)
        self.assertEqual(entry.notional, 1000.)

        entry.apply_conversion(Inventory.StockConversion(record_date, ex_date, TICKER, 'NEW.TICKER', 1.5))
        self.assertEqual(entry.ticker, 'NEW.TICKER')
        self.assertAlmostEqual(entry.volume, 360.)
        self.assertAlmostEqual(entry.price, 10. / 3.6)
        self.assertEqual(entry.notional, 1000.)


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

        # The process engine keeps every order it ever created, and the risk
        # query counts those fills: the suites that ran before this one would
        # otherwise spend the trade limit. This class tests the position
        # aggregates, not the gate, so its limits are unreachable.
        for key in ('max_trade_long', 'max_trade_short', 'max_exposure_long', 'max_exposure_short'):
            RISK_PROFILE.set_rule(key, 1e15, ticker=TICKER)

        for key in ('max_notional_long', 'max_notional_short'):
            RISK_PROFILE.set_rule(key, 1e15, ticker=TICKER)

        for key in ('max_ttl_notional_long', 'max_ttl_notional_short', 'max_net_notional_long', 'max_net_notional_short'):
            RISK_PROFILE.set_rule(key, 1e15)

    def test_00_dummy_exposure(self) -> None:
        position = PositionManagementService()

        self.assertIsInstance(position.position_id, uuid.UUID)
        self.assertEqual(position.default_algo, 'passive')

        position.add_exposure(TICKER, 100., 1050., 1., TransactionSide.SIDE_LONG_OPEN)

        self.assertEqual(position.exposure_volume, {TICKER: 100.})
        self.assertEqual(position.working_volume, {'Long': {}, 'Short': {}})
        self.assertEqual(position.notional, {TICKER: 1050.})
        self.assertEqual(position.pnl(), {TICKER: 0.})
        self.assertEqual(position.orders, {})
        self.assertIn(str(position.position_id), position.to_json(fmt='dict'))

    def test_01_open_order_and_fill(self) -> None:
        position = PositionManagementService()

        algo = position.open(TICKER, 100., SIDE_LONG)
        self.assertIsNotNone(algo)
        self.assertEqual(len(position.working_algos), 1)
        self.assertEqual(position.exposure_volume, {})
        self.assertEqual(position.working_volume, {'Long': {TICKER: 100.}, 'Short': {}})
        self.assertEqual(position.working_volume_net, {TICKER: 100.})

        record = algo
        self.assertEqual(record.ticker, TICKER)
        self.assertEqual(record.side, SIDE_LONG)
        self.assertEqual(record.target_volume, 100.)
        self.assertEqual(record.working_volume, 100.)
        self.assertEqual(record.status, AlgoStatus.Working)

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

        ALGO_ENGINE.on_trade(report)

        self.assertEqual(position.exposure_volume, {TICKER: 100.})
        self.assertEqual(len(position.working_algos), 0)
        self.assertEqual(len(position.trades), 1)
        self.assertEqual(record.status, AlgoStatus.Done)
        self.assertEqual(record.exposure_volume, 100.)
        self.assertEqual(record.filled_notional, 1050.)

    def test_03_host_routing_and_callbacks(self) -> None:
        position = PositionManagementService()
        events = []

        callback = lambda *args: events.append(args)
        position.register_callback(PositionCallbackEvent.filled, callback)

        algo = position.open(TICKER, 100., SIDE_LONG)
        order = next(iter(position.working_order.values()))

        report = TradeReport(
            ticker=TICKER,
            volume=100.,
            price=10.5,
            side=TransactionSide.SIDE_LONG_OPEN,
            timestamp=TS,
            order_id=order.order_id,
        )

        self.assertEqual(ALGO_ENGINE.on_trade(report), 1)
        self.assertEqual(position.exposure_volume, {TICKER: 100.})
        self.assertEqual(len(events), 1)

        event_position, event_algo, event_order, event_trade = events[0]
        self.assertIs(event_position, position)
        self.assertEqual(event_algo.algo_id, algo.algo_id)
        self.assertEqual(event_trade.trade_id, report.trade_id)

        position.unregister_callback(PositionCallbackEvent.filled, callback)

        # The unregistered callback no longer fires; the aggregates still track.
        short = position.open(TICKER, 50., SIDE_SHORT)
        short_order = next(iter(short.working_orders.values()))
        short_report = TradeReport(
            ticker=TICKER,
            volume=50.,
            price=10.5,
            side=TransactionSide.SIDE_SHORT_OPEN,
            timestamp=TS,
            order_id=short_order.order_id,
        )

        self.assertEqual(ALGO_ENGINE.on_trade(short_report), 1)
        self.assertEqual(len(events), 1)
        self.assertEqual(position.exposure_volume, {TICKER: 50.})

    def test_04_positions_do_not_interfere(self) -> None:
        first = PositionManagementService()
        second = PositionManagementService()

        first_events: list = []
        second_events: list = []

        first.register_callback(PositionCallbackEvent.filled, lambda *args: first_events.append(args))
        second.register_callback(PositionCallbackEvent.filled, lambda *args: second_events.append(args))

        first.open(TICKER, 100., SIDE_LONG)
        second.open(TICKER, 50., SIDE_LONG)

        first_order = next(iter(first.working_order.values()))
        second_order = next(iter(second.working_order.values()))

        first_report = TradeReport(
            ticker=TICKER,
            volume=100.,
            price=10.5,
            side=TransactionSide.SIDE_LONG_OPEN,
            timestamp=TS,
            order_id=first_order.order_id,
        )

        # The engine routes the fill to the position that hosts the order's algo.
        self.assertEqual(ALGO_ENGINE.on_trade(first_report), 1)

        self.assertEqual(first.exposure_volume, {TICKER: 100.})
        self.assertEqual(second.exposure_volume, {})
        self.assertEqual(len(first_events), 1)
        self.assertEqual(len(second_events), 0)
        self.assertEqual(second.working_volume, {'Long': {TICKER: 50.}, 'Short': {}})

        second_report = TradeReport(
            ticker=TICKER,
            volume=50.,
            price=10.5,
            side=TransactionSide.SIDE_LONG_OPEN,
            timestamp=TS,
            order_id=second_order.order_id,
        )

        self.assertEqual(ALGO_ENGINE.on_trade(second_report), 1)

        self.assertEqual(first.exposure_volume, {TICKER: 100.})
        self.assertEqual(second.exposure_volume, {TICKER: 50.})
        self.assertEqual(len(first_events), 1)
        self.assertEqual(len(second_events), 1)

    def test_05_callback_registration_and_algo_views(self) -> None:
        position = PositionManagementService()

        # A callback that cannot take the event's protocol shape is refused.
        with self.assertRaises(TypeError):
            position.register_callback(PositionCallbackEvent.filled, lambda only_one: None)

        with self.assertRaises(TypeError):
            position.register_callback(PositionCallbackEvent.canceled, lambda position, algo, order, trade: None)

        first_events: list = []
        second_events: list = []

        first_callback = lambda position, algo, order, trade: first_events.append(trade.trade_id) or 0
        second_callback = lambda position, algo, order, trade: second_events.append(trade.trade_id) or 0

        position.register_callback(PositionCallbackEvent.filled, first_callback)
        position.register_callback(PositionCallbackEvent.filled, second_callback)

        # The same callback registers once.
        duplicate = lambda *args: None
        position.register_callback(PositionCallbackEvent.canceled, duplicate)

        with self.assertRaises(ValueError):
            position.register_callback(PositionCallbackEvent.canceled, duplicate)

        # The event pins the list the callback was registered on.
        with self.assertRaises(KeyError):
            position.unregister_callback(PositionCallbackEvent.canceled, first_callback)

        algo = position.open(TICKER, 100., SIDE_LONG)
        order = next(iter(position.working_order.values()))

        # The algo views are live, read-only and keyed by the algo id; the
        # working view holds an active algo and drops it once it completes.
        self.assertEqual(position.algos[algo.algo_id.hex].algo_id, algo.algo_id)
        self.assertEqual(position.working_algos[algo.algo_id.hex].algo_id, algo.algo_id)
        self.assertIn(algo.algo_id.hex, ALGO_ENGINE.algos)

        with self.assertRaises(PermissionError):
            position.algos[algo.algo_id.hex] = algo

        report = TradeReport(
            ticker=TICKER,
            volume=100.,
            price=10.5,
            side=TransactionSide.SIDE_LONG_OPEN,
            timestamp=TS,
            order_id=order.order_id,
        )

        self.assertEqual(ALGO_ENGINE.on_trade(report), 1)

        # Both callbacks of the event fire from the one C registration.
        self.assertEqual(first_events, [report.trade_id])
        self.assertEqual(second_events, [report.trade_id])
        self.assertNotIn(algo.algo_id.hex, position.working_algos)

        # Removing one callback keeps the event's registration and the other callback.
        position.unregister_callback(PositionCallbackEvent.filled, first_callback)

        short = position.open(TICKER, 50., SIDE_SHORT)
        short_order = next(iter(short.working_orders.values()))
        short_report = TradeReport(
            ticker=TICKER,
            volume=50.,
            price=10.5,
            side=TransactionSide.SIDE_SHORT_OPEN,
            timestamp=TS,
            order_id=short_order.order_id,
        )

        self.assertEqual(ALGO_ENGINE.on_trade(short_report), 1)
        self.assertEqual(len(first_events), 1)
        self.assertEqual(len(second_events), 2)

        # Removing the last callback releases the event's registration.
        position.unregister_callback(PositionCallbackEvent.filled, second_callback)

        with self.assertRaises(KeyError):
            position.unregister_callback(PositionCallbackEvent.filled, second_callback)

        third = position.open(TICKER, 25., SIDE_LONG)
        third_order = next(iter(third.working_orders.values()))

        third_report = TradeReport(
            ticker=TICKER,
            volume=25.,
            price=10.5,
            side=TransactionSide.SIDE_LONG_OPEN,
            timestamp=TS,
            order_id=third_order.order_id,
        )

        self.assertEqual(ALGO_ENGINE.on_trade(third_report), 1)
        self.assertEqual(len(first_events), 1)
        self.assertEqual(len(second_events), 2)


if __name__ == '__main__':
    unittest.main()

import enum
import pathlib
from collections.abc import Iterator
from typing import Any

import pandas as pd

from algo_engine.base import TradeInstruction, TransactionSide


class RiskRule(enum.IntEnum):
    """
    The built-in rules a profile carries.

    The member name is the rule key: a payload, ``set_rule`` and the rules
    snapshot all name a rule by it. A rule is defined once it is set.
    """

    max_percentile: RiskRule
    max_trade_long: RiskRule
    max_trade_short: RiskRule
    max_exposure_long: RiskRule
    max_exposure_short: RiskRule
    max_notional_long: RiskRule
    max_notional_short: RiskRule
    max_ttl_notional_long: RiskRule
    max_ttl_notional_short: RiskRule
    max_net_notional_long: RiskRule
    max_net_notional_short: RiskRule

    @property
    def is_global(self) -> bool:
        """True for the four portfolio-wide rules."""


class RiskType(enum.IntEnum):
    """The check a rejection attributes; values mirror the C ``dma_risk_type``."""

    invalid: RiskType
    passed: RiskType
    incomplete: RiskType
    price: RiskType
    trade_invalid: RiskType
    trade_long: RiskType
    trade_short: RiskType
    exposure_long: RiskType
    exposure_short: RiskType
    percentile_long: RiskType
    percentile_short: RiskType
    notional_long: RiskType
    notional_short: RiskType
    portfolio_net_long: RiskType
    portfolio_net_short: RiskType
    portfolio_total_long: RiskType
    portfolio_total_short: RiskType
    ticker_blocked: RiskType
    ticker_unlisted: RiskType
    custom: RiskType


class RiskViolation(Exception):
    """
    The violation a rejected check reports.

    The C checker fills the same information into the violation it reports;
    ``c_from_header`` wraps that record, and the wrapper owns its own block
    when it is constructed from Python.

    Attributes:
        code (int): Numeric code of the violated rule.
        type (RiskType): The check that rejected.
        msg (str): Human-readable rejection reason.
    """

    code: int
    type: RiskType
    msg: str

    def __init__(self, risk_type: RiskType, code: int, msg: str, *args, **kwargs):
        """
        Build a risk violation.

        Args:
            risk_type (RiskType): The check that rejected.
            code (int): Numeric code of the violated rule.
            msg (str): Human-readable rejection reason.
        """


class RiskLimits:
    """
    A view over one ticker's limit set in a risk profile.

    The view is borrowed: the profile owns the limit set and frees it on
    ``clear()`` and on teardown, after which the view is a husk and raises.

    A rule reports None while it is disabled; setting an infinite limit (or a
    percentile of one or more) disables it.

    Attributes:
        max_percentile (float | None): Maximum share of the portfolio.
        max_trade_long (float | None): Maximum traded + working long volume.
        max_trade_short (float | None): Maximum traded + working short volume.
        max_exposure_long (float | None): Maximum net long exposure volume.
        max_exposure_short (float | None): Maximum net short exposure volume.
        max_notional_long (float | None): Maximum net long exposure notional.
        max_notional_short (float | None): Maximum net short exposure notional.
        enabled (int): The enable mask, one bit per ``RiskRule``.
        rules (Iterator[RiskRuleTemplate]): The ticker's custom rules, in
            registration order.
    """

    max_percentile: float | None
    max_trade_long: float | None
    max_trade_short: float | None
    max_exposure_long: float | None
    max_exposure_short: float | None
    max_notional_long: float | None
    max_notional_short: float | None
    enabled: int
    rules: Iterator[RiskRuleTemplate]


class RiskLimitsMapView(dict[str, RiskLimits]):
    """
    A read-only, live view of a profile's per-ticker limit sets.

    The view is cached on the profile and stays bound to the C map: a limit
    set appears when its first rule is written and disappears on ``clear()``.
    Writing to the view raises.
    """


class RiskRuleTemplate:
    """
    The common base of the custom rules, wrapping one ``dma_risk_rule_ctx``.

    A rule is registered on a profile, ticker-scoped or global: the profile
    composes the rule node, adopts the rule's state, and owns it, so
    unregistering or clearing the profile frees both. Dropping this wrapper
    leaves the rule registered.

    Attributes:
        ticker (str | None): The registration scope; None for a global rule or
            a bare view yielded by ``RiskLimits.rules``.
        risk_rule_code (int): The code a rejection reports.
        name (str | None): The rule name reported in a rejection message.
        rule_id (int): The node address that identifies the rule.
        is_registered (bool): True while this wrapper holds a live node.
        enabled (bool): False while a registered rule is skipped by the check.
    """

    ticker: str | None
    risk_rule_code: int
    name: str | None
    rule_id: int
    is_registered: bool
    enabled: bool

    def register(self, risk_profile: RiskProfile, ticker: str = None) -> None:
        """
        Register the rule on the profile for one ticker.

        The profile composes the rule node, owns it, and keeps this wrapper
        alive while the rule is registered. A rule already holding a node
        refuses a second registration. A subclass overrides the registration
        the same way: ``NoTradeWindowRule`` builds its C state from the ticker
        and its window.

        Args:
            risk_profile (RiskProfile): The profile to register on.
            ticker (str): The ticker scope; None defers to the rule's own state.
        """

    def register_global(self, risk_profile: RiskProfile) -> None:
        """Register the rule globally; it applies to every ticker."""

    def unregister(self) -> None:
        """Unlink the rule from its profile and free its node.

        Raises:
            RuntimeError: When the rule is not registered.
        """

    def enable(self) -> None:
        """Let the check run the rule again."""

    def disable(self) -> None:
        """Keep the rule linked but skip it in the check."""


class RiskProfile:
    """
    The C-backed risk profile.

    The profile stores one limit set per ticker plus the four portfolio-wide
    limits as structured C rules. A rule is enabled once it is set; the checks
    run only the rules enabled for the order's side, so a profile needs the
    rules that matter to it, not all of them. A side is constrained only by an
    enabled absolute risk value: a trade, exposure or notional limit on the
    ticker, or a portfolio total or net limit. The percentile limit is relative
    to the portfolio and a custom rule never completes a profile, so neither
    keeps a side out of the incomplete refusal. A side with no enabled absolute
    rule is refused as incomplete and the message lists the disabled rules that
    apply; a fresh profile enables nothing and refuses every transaction.
    ``clear()`` drops the ticker rules and disables the four portfolio limits,
    so a cleared profile refuses every transaction until a rule is set.

    Two ticker gates override the rules: a blacklisted ticker is always
    refused, and a whitelist, when present, allows only the tickers it lists.
    Either gate may be absent; an absent gate is skipped. A ticker cannot be on
    both lists: the add refuses the conflict and logs it.

    A ticker can also carry custom rules (see ``NoTradeWindowRule``); they run
    after the standard rules pass, ticker rules before global ones, and a
    disabled rule is skipped.

    The checks and the volume report run against the process access: it
    supplies the hosted positions, the engine's orders and the market price, so
    no portfolio state is rebuilt here.

    Attributes:
        limits (RiskLimitsMapView): The live per-ticker limit sets.
    """

    def __init__(self, **kwargs):
        """
        Allocate the profile and apply any initial rule payload.

        Args:
            **kwargs: Initial rules; a dict value sets a rule per ticker, a
                scalar sets a portfolio-wide rule, and a ``whitelist`` or
                ``blacklist`` list fills that gate.
        """

    def __call__(self, *order: TradeInstruction) -> bool:
        """
        Check one order, or a basket of orders individually.

        Args:
            *order (TradeInstruction): The order(s) to check.

        Returns:
            True when every checked order passes.

        Raises:
            RiskViolation: When an order is refused; a basket stops at the
                first refusal.
        """

    def set_rule(self, key: RiskRule | str, value: float, ticker: str = None) -> None:
        """
        Set one rule, globally or for a ticker.

        The key names a rule by its ``RiskRule`` member or by the member name.
        A per-ticker rule requires a ticker; a portfolio-wide rule refuses one.
        Unknown keys, scope errors, NaN and negative values are logged and
        refused. An infinite value, or a percentile of one or more, disables
        the rule.

        Args:
            key (RiskRule | str): The rule, or its member name as a string.
            value (float): The limit value.
            ticker (str): The ticker for a per-ticker rule, or None for a global rule.
        """

    def set_rules(self, rules: dict) -> None:
        """
        Set the rules and gates from a mapping.

        The mapping has the shape ``to_json(fmt='dict')`` returns: a rule key
        sets that rule per ticker (a dict value) or portfolio-wide (a scalar),
        and ``whitelist``/``blacklist`` lists fill the gates. ``entry``,
        ``custom`` and ``custom_global`` are a snapshot, not configuration, and
        are ignored.

        Args:
            rules (dict): The rule and gate mapping.
        """

    def set_whitelist(self, ticker: str, enabled: bool = True) -> None:
        """
        Add the ticker to the whitelist, or drop it.

        An absent whitelist gates nothing; a present one allows only the
        tickers it lists. A ticker on the blacklist is refused and the conflict
        is logged; drop it from the blacklist first.

        Args:
            ticker (str): The instrument.
            enabled (bool): True to list the ticker, False to drop it.
        """

    def set_blacklist(self, ticker: str, enabled: bool = True) -> None:
        """
        Add the ticker to the blacklist, or drop it.

        A blacklisted ticker is refused even when its rules are complete. A
        ticker on the whitelist is refused and the conflict is logged; drop it
        from the whitelist first. Dropping the last ticker frees that gate.

        Args:
            ticker (str): The instrument.
            enabled (bool): True to list the ticker, False to drop it.
        """

    def get(self, ticker: str) -> dict:
        """
        Collect the current limits and volumes of a ticker.

        Returns:
            A dict with ``market_price``, ``working``, ``traded``,
            ``exposure`` and every defined limit of the ticker.
        """

    def check(self, order: TradeInstruction) -> bool:
        """
        Check one order against the profile through the process access.

        The access supplies the portfolio state and the price; the decision is
        made by the C checker. A side with no enabled absolute risk value, a
        ticker without a market price, and every other refusal are logged as
        errors and raise ``RiskViolation``.

        Args:
            order (TradeInstruction): The order to check.

        Returns:
            True when the order may be sent.

        Raises:
            RiskViolation: When the order is refused.
        """

    def check_order(self, ticker: str, volume: float, side: TransactionSide) -> bool:
        """
        Check a hypothetical order at the ticker's market price.

        Args:
            ticker (str): The instrument.
            volume (float): The order volume.
            side (TransactionSide): The composed transaction side.

        Returns:
            True when the order may be sent.

        Raises:
            RiskViolation: When the order is refused.
        """

    def check_basket(self, *order: TradeInstruction) -> None:
        """
        Check a basket, one order at a time.

        Basket-level risk control is not implemented; each order is checked
        individually and a warning is logged.

        Args:
            *order (TradeInstruction): The orders to check.
        """

    def clear(self) -> None:
        """
        Drop the per-ticker rules, the global custom rules and both gates,
        and disable the portfolio limits.
        """

    def to_json(self, fmt: str = 'str') -> str | dict:
        """
        Export the rules.

        Args:
            fmt (str): ``'dict'`` for a dict, anything else for a JSON string.

        Returns:
            The rule payload.
        """

    def from_json(self, json_str: str | bytes | dict) -> RiskProfile:
        """
        Set the rules and gates from a payload.

        Args:
            json_str (str | bytes | dict): The payload to apply.

        Returns:
            self.
        """

    def dump(self, file_path: str | pathlib.Path) -> None:
        """
        Write the rules to a JSON file.

        Args:
            file_path (str | pathlib.Path): The output path.
        """

    def load(self, file_path: str | pathlib.Path) -> None:
        """
        Apply the rules stored in a JSON file.

        Args:
            file_path (str | pathlib.Path): The input path.
        """

    @property
    def rules(self) -> dict:
        """
        A snapshot of the rule set.

        The snapshot mirrors the C profile: ``entry`` is the set of tickers
        with at least one rule or custom rule, ``whitelist`` and ``blacklist``
        are the gate sets, ``custom`` maps a ticker to its custom rule codes,
        ``custom_global`` lists the global custom rule codes, the per-ticker
        rules are dicts keyed by ticker, and an undefined portfolio limit is
        None. Custom rules are code, not configuration: a payload reports
        their codes but does not recreate them.
        """

    @property
    def limits(self) -> RiskLimitsMapView:
        """The live per-ticker limit sets; the view is cached on the profile."""

    @property
    def market_time(self) -> Any:
        """The market data service's latest market time."""

    @property
    def info(self) -> pd.DataFrame:
        """The rule set as a DataFrame, one row per ticker plus ``global``."""

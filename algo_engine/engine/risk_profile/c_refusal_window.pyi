import datetime

from .c_risk_profile import RiskProfile, RiskRuleTemplate


class RefusalWindowRule(RiskRuleTemplate):
    """
    A custom rule that refuses trading inside a session-time window.

    Trading is refused in ``[start, end)``. A bound that is None is not part
    of the window, so an end-only rule refuses trading before it, a start-only
    rule refuses trading at or after it, and a rule with both refuses the
    period. The wrapper stores only the window; registration converts it into
    the C rule state (a bound bit mask plus the two session times) and hands
    it to the rule node, which owns it. A ticker registration refuses that
    ticker; a global registration carries no ticker and refuses every ticker.
    The rule reports ``RiskProfile.Custom`` with code 6001 and the ``refusal_window``
    rule name when it rejects. It runs after the profile's standard rules
    pass; the clock is the access's latest session datetime, and without one
    the rule fails closed.

    Attributes:
        start (datetime.time | None): The window opens here; None leaves it open.
        end (datetime.time | None): The window closes here, exclusive; None
            leaves it open.
    """

    start: datetime.time | None
    end: datetime.time | None

    def __init__(self, start: datetime.time = None, end: datetime.time = None):
        """
        Create the window.

        Args:
            start (datetime.time): The session time the window opens.
            end (datetime.time): The session time the window closes, exclusive.

        Raises:
            ValueError: When both bounds are None.
        """

    def register(self, risk_profile: RiskProfile, ticker: str = None) -> None:
        """
        Register the rule for one ticker.

        The ticker is the rule's scope and the single source of the ticker:
        the C rule state is built from it at registration.

        Args:
            risk_profile (RiskProfile): The profile to register on.
            ticker (str): The instrument.

        Raises:
            TypeError: When no ticker is given.
            RuntimeError: When the rule is already registered.
        """

    def register_global(self, risk_profile: RiskProfile) -> None:
        """
        Register the rule globally: it refuses every ticker.

        The global rule carries no ticker; every ticker is refused while the
        session clock is inside the window.

        Args:
            risk_profile (RiskProfile): The profile to register on.

        Raises:
            RuntimeError: When the rule is already registered.
        """

    def unregister(self) -> None:
        """Unlink the rule; the node and the rule state are freed.

        Raises:
            RuntimeError: When the rule is not registered.
        """


class RefusalWindowBeforeRule(RefusalWindowRule):
    """Refuse trading strictly before a session time."""

    def __init__(self, until: datetime.time):
        """
        Create the rule.

        Args:
            until (datetime.time): Trading is refused strictly before this time.
        """


class RefusalWindowAfterRule(RefusalWindowRule):
    """Refuse trading at or after a session time."""

    def __init__(self, since: datetime.time):
        """
        Create the rule.

        Args:
            since (datetime.time): Trading is refused from this time onward.
        """


class RefusalWindowBetweenRule(RefusalWindowRule):
    """Refuse trading in a session-time period."""

    def __init__(self, start: datetime.time, end: datetime.time):
        """
        Create the rule.

        Args:
            start (datetime.time): Trading is refused from this time.
            end (datetime.time): Trading is refused until this time, exclusive.
        """

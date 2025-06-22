from datetime import datetime, date, time, tzinfo
from typing import Literal

PROFILE: ProfileDispatcher
PROFILE_DEFAULT: Profile


class ProfileDispatcher:
    """
    Global profile dispatcher, designed to be a singleton instance managing the market session profile.

    Attributes:
        profile_id (str): ID of the current profile.
        session_start (time | None): Start time of the market session, if applicable.
        session_end (time | None): End time of the market session, if applicable.
        time_zone (tzinfo | None): Default timezone of this market session. Recommended to set as None to use the system timezone.
        session_break (list[tuple[time, time]]): List of market session breaks as (break_start, break_end) tuples.
        session_length (float): Duration of a typical trading session in seconds.
    """
    profile_id: str
    session_start: time | None
    session_end: time | None
    time_zone: tzinfo | None
    session_break: list[tuple[time, time]]
    session_length: float

    def time_to_seconds(self, t: time, break_adjusted: bool = True) -> float:
        """
        Convert a time object to the number of elapsed seconds since the start of the trading day.

        Args:
            t (time): The time to convert.
            break_adjusted (bool, optional): Whether to exclude non-trading hours from the elapsed time. Defaults to True.

        Returns:
            float: Elapsed seconds since the start of the trading session.
        """

    def timestamp_to_seconds(self, t: float, break_adjusted: bool = True) -> float:
        """
        Convert a UNIX timestamp to the number of elapsed seconds since the start of the trading day.

        Args:
            t (float): The UNIX timestamp in seconds.
            break_adjusted (bool, optional): Whether to exclude non-trading hours from the elapsed time. Defaults to True.

        Returns:
            float: Elapsed seconds since the start of the trading session.
        """

    def break_adjusted(self, elapsed_seconds: float) -> float:
        """
        Adjust elapsed seconds to exclude non-trading hours.

        Args:
            elapsed_seconds (float): Elapsed seconds since the start of the day (e.g., 36000.0 for 10:00 AM).

        Returns:
            float: Adjusted elapsed seconds excluding non-trading hours.
        """

    def trading_time_between(self, start_time: datetime | float | int, end_time: datetime | float | int) -> float:
        """
        Calculate the total trading time between two timestamps, in seconds.

        Args:
            start_time (datetime | float | int): Start timestamp (datetime object or UNIX timestamp in seconds).
            end_time (datetime | float | int): End timestamp (must be greater than start_time).

        Returns:
            float: Total trading time in seconds.

        Notes:
            Time zone awareness is not recommended for input.
        """

    def is_market_session(self, timestamp: float | int | datetime | time) -> bool:
        """
        Check whether the given timestamp is within the current market session.

        Args:
            timestamp (float | int | datetime | time): The timestamp to check.

        Returns:
            bool: True if the timestamp is within the market session, False otherwise.

        Notes:
            - UNIX numeric timestamps always assume the date is a trading day to conserve computational resources.
            - After all the UNIX numeric timestamp is most likely, if not always, comes from a MarketData. Which existence proofs this day is a trading day.
            - This behavior is consistent with the corresponding Cython interface `c_timestamp_in_market_session(t)`.
        """

    def trade_calendar(self, start_date: date, end_date: date) -> list[date]:
        """
        Generate a list of trading days within the specified range, inclusive.

        Args:
            start_date (date): Start date (inclusive).
            end_date (date): End date (inclusive, must be >= start_date).

        Returns:
            list[date]: List of trading days within the range.
        """

    def trading_days_before(self, market_date: date, days: int) -> date:
        """
        Get the trading date a specified number of trading days before the given date.

        Args:
            market_date (date): Reference trading date.
            days (int): Number of trading days to go back.

        Returns:
            date: The resulting trading date.
        """

    def trading_days_after(self, market_date: date, days: int) -> date:
        """
        Get the trading date a specified number of trading days after the given date.

        Args:
            market_date (date): Reference trading date.
            days (int): Number of trading days to advance.

        Returns:
            date: The resulting trading date.
        """

    def trading_days_between(self, start_date: date, end_date: date) -> int:
        """
        Calculate the number of trading days between two dates.

        Args:
            start_date (date): Start trading date.
            end_date (date): End trading date (must be >= start_date).

        Returns:
            int: Number of trading days from the pre-open of start_date to the pre-open of end_date.

        Example:
            For dates 2024-11-11 to 2024-11-12, returns 1 (from 2024-11-11 09:30:00 to 2024-11-12 09:30:00).
        """

    def nearest_trading_date(self, market_date: date, method: Literal['previous', 'next'] = 'previous') -> date:
        """
        Find the nearest trading date relative to a given date.

        Args:
            market_date (date): Reference date.
            method (Literal['previous', 'next'], optional): Direction to search. Defaults to 'previous'.

        Returns:
            date: Nearest trading date according to the specified method.

        Raises:
            ValueError: If `method` is not 'previous' or 'next'.
        """

    @property
    def range_break(self) -> list[dict]:
        """
        Generate a list of range breaks for use with Plotly X-axis configuration.

        Returns:
            list[dict]: Range break specifications for Plotly.
        """


class Profile:
    """
    Individual profile object that can override the global profile dispatcher.

    This Profile class is designed to be a base class, also a non-stop trading profile, like cryptocurrency.
    """

    def override_profile(self) -> ProfileDispatcher:
        """
        Override the global ProfileDispatcher with this profile.

        Returns:
            ProfileDispatcher: The overridden ProfileDispatcher instance.
        """

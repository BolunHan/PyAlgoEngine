import datetime
import enum
import pathlib

import pandas as pd

from algo_engine.base import TransactionDirection
from algo_engine.exchange_profile import SessionDate

INVENTORY: Inventory


class Inventory:
    """
    The C-backed security lending inventory, a process singleton.

    Each ticker has one C node holding a long lot list and a short lot list; a
    getter returns the first stored lot as a live view, and the consumed
    volume of that side is subtracted from the available volume. The long and
    short sides are the only directions the inventory carries.
    """

    class SecurityType(enum.IntEnum):
        """The security classes an inventory lot can carry; values mirror the C enum."""

        Commodity: Inventory.SecurityType
        CurrencySwap: Inventory.SecurityType
        Crypto: Inventory.SecurityType
        IndexFuture: Inventory.SecurityType
        Stock: Inventory.SecurityType

    class CashDividend:
        """A cash dividend applied to a lot."""

        record_date: SessionDate
        ex_date: SessionDate
        dividend_per_share: float

        def __init__(self, record_date: datetime.date, ex_date: datetime.date, dividend_per_share: float):
            """
            Args:
                record_date (datetime.date): The record date.
                ex_date (datetime.date): The ex-date.
                dividend_per_share (float): Cash per share.
            """

    class StockDividend:
        """A stock dividend applied to a lot."""

        record_date: SessionDate
        ex_date: SessionDate
        dividend_per_share: float

        def __init__(self, record_date: datetime.date, ex_date: datetime.date, dividend_per_share: float):
            """
            Args:
                record_date (datetime.date): The record date.
                ex_date (datetime.date): The ex-date.
                dividend_per_share (float): Shares per held share.
            """

    class StockSplit:
        """A stock split applied to a lot."""

        record_date: SessionDate
        ex_date: SessionDate
        multiplier: float

        def __init__(self, record_date: datetime.date, ex_date: datetime.date, multiplier: float):
            """
            Args:
                record_date (datetime.date): The record date.
                ex_date (datetime.date): The ex-date.
                multiplier (float): The split multiplier.
            """

    class StockConversion:
        """A stock conversion applied to a lot."""

        record_date: SessionDate
        ex_date: SessionDate
        convert_from: str
        convert_to: str
        multiplier: float

        def __init__(self, record_date: datetime.date, ex_date: datetime.date, convert_from: str, convert_to: str, multiplier: float):
            """
            Args:
                record_date (datetime.date): The record date.
                ex_date (datetime.date): The ex-date.
                convert_from (str): The source ticker.
                convert_to (str): The target ticker.
                multiplier (float): The conversion multiplier.
            """

    class Entry:
        """
        One C-backed inventory lot.

        An entry built directly owns its C block; an entry returned by
        ``Inventory.get_inv`` is a borrowed view over a stored lot.

        Attributes:
            ticker (str): The instrument.
            volume (float): The lot volume.
            price (float): The lot price.
            notional (float): The lot notional; defaults to ``volume * price``.
            fee (float): The allocated fee.
            recalled (float): The recalled volume.
            direction (TransactionDirection): The lot direction.
            security_type (Inventory.SecurityType): The security class.
            available (float): ``max(volume - recalled, 0)``.
        """

        ticker: str
        volume: float
        price: float
        notional: float
        fee: float
        recalled: float
        direction: TransactionDirection
        security_type: Inventory.SecurityType

        def __init__(
                self,
                ticker: str,
                volume: float,
                price: float,
                security_type: Inventory.SecurityType,
                direction: TransactionDirection = TransactionDirection.DIRECTION_LONG,
                **kwargs
        ):
            """
            Build a lot.

            Args:
                ticker (str): The instrument.
                volume (float): The lot volume.
                price (float): The lot price.
                security_type (Inventory.SecurityType): The security class.
                direction (TransactionDirection): The lot direction; only
                    ``DIRECTION_LONG`` and ``DIRECTION_SHORT`` are supported.
                **kwargs: ``notional``, ``fee`` and ``recalled`` overrides.
            """

        def __add__(self, other: Inventory.Entry) -> Inventory.Entry:
            """Merge two lots into a new one."""

        def __bool__(self) -> bool:
            """True when the lot has volume."""

        def merge(self, entry: Inventory.Entry, inplace: bool = False, **kwargs) -> Inventory.Entry:
            """
            Merge another lot into this one.

            The lots must share ticker, direction sign and security type. The
            merged price is the volume-weighted average.

            Args:
                entry (Inventory.Entry): The lot to merge.
                inplace (bool): True updates this lot, False builds a new one.
                **kwargs: Explicit ``volume``, ``notional``, ``price``, ``fee``
                    or ``recalled`` overrides.

            Returns:
                The merged lot.
            """

        def to_json(self, fmt: str = 'str') -> str | dict:
            """Export the lot."""

        @classmethod
        def from_json(cls, json_str: str | bytes | dict) -> Inventory.Entry:
            """Build a lot from an exported payload."""

        def apply_cash_dividend(self, dividend: Inventory.CashDividend):
            """
            Mark this lot's price down by the dividend per share.

            A single-lot convenience; the inventory-level apply records the
            event on every stored lot.
            """

        def apply_stock_dividend(self, dividend: Inventory.StockDividend):
            """
            Increase this lot's volume by ``dividend_per_share`` per held share.

            A single-lot convenience; the inventory-level apply records the
            event on every stored lot.
            """

        def apply_conversion(self, stock_conversion: Inventory.StockConversion):
            """
            Convert this lot to ``convert_to`` at the conversion multiplier.

            A single-lot convenience; the inventory-level apply records the
            event on every stored lot.
            """

        def apply_split(self, stock_split: Inventory.StockSplit):
            """
            Apply the split multiplier to this lot.

            A single-lot convenience; the inventory-level apply records the
            event on every stored lot.
            """

        @property
        def available(self) -> float:
            """The un-recalled volume."""

    def __init__(self):
        """Allocate the C inventory."""

    def __call__(self, ticker: str) -> dict[str, float]:
        """
        The available volume of a ticker on both directions.

        Args:
            ticker (str): The instrument.

        Returns:
            ``{'Long': ..., 'Short': ...}``.
        """

    def recall(self, ticker: str, volume: float, direction: TransactionDirection = TransactionDirection.DIRECTION_LONG) -> None:
        """
        Recall volume from the earliest lots.

        Args:
            ticker (str): The instrument.
            volume (float): The volume to recall.
            direction (TransactionDirection): The lot direction.
        """

    def add_inv(self, entry: Inventory.Entry) -> None:
        """
        Store one lot.

        Args:
            entry (Inventory.Entry): The lot to store; the inventory keeps its own copy.
        """

    def get_inv(self, ticker: str, direction: TransactionDirection = TransactionDirection.DIRECTION_LONG) -> Inventory.Entry | None:
        """
        The first stored lot of a ticker and direction.

        The entry is a live view over the stored lot: mutating it mutates the
        inventory, and it becomes a husk when the lot is recalled or the
        inventory is cleared.

        Args:
            ticker (str): The instrument.
            direction (TransactionDirection): The lot direction.

        Returns:
            The first stored lot, or None when the bucket is empty.
        """

    def use_inv(self, ticker: str, volume: float, direction: TransactionDirection = TransactionDirection.DIRECTION_LONG) -> None:
        """
        Record consumed volume.

        Args:
            ticker (str): The instrument.
            volume (float): The consumed volume.
            direction (TransactionDirection): The direction consumed.
        """

    def available_volume(self, ticker: str, direction: TransactionDirection = TransactionDirection.DIRECTION_LONG) -> float:
        """
        The available volume after recalls and usage.

        Args:
            ticker (str): The instrument.
            direction (TransactionDirection): The direction.

        Returns:
            ``max(held - recalled - used, 0)``.
        """

    def clear(self) -> None:
        """Drop every lot and every consumed-volume record."""

    def cash_dividend(self, record_date: datetime.date, ex_date: datetime.date, dividend_per_share: float) -> None:
        """
        Apply and record a cash dividend across the inventory.

        Every lot of every ticker is marked down by the dividend per share and
        stores the event on its own list.

        Args:
            record_date (datetime.date): The record date.
            ex_date (datetime.date): The ex-date.
            dividend_per_share (float): Cash per held share.
        """

    def stock_dividend(self, record_date: datetime.date, ex_date: datetime.date, dividend_per_share: float) -> None:
        """
        Apply and record a stock dividend across the inventory.

        Args:
            record_date (datetime.date): The record date.
            ex_date (datetime.date): The ex-date.
            dividend_per_share (float): Shares received per held share.
        """

    def stock_split(self, record_date: datetime.date, ex_date: datetime.date, multiplier: float) -> None:
        """
        Apply and record a split across the inventory.

        Args:
            record_date (datetime.date): The record date.
            ex_date (datetime.date): The ex-date.
            multiplier (float): The split multiplier.
        """

    def stock_conversion(self, record_date: datetime.date, ex_date: datetime.date, convert_from: str, convert_to: str, multiplier: float) -> None:
        """
        Apply and record a stock conversion across the inventory.

        Args:
            record_date (datetime.date): The record date.
            ex_date (datetime.date): The ex-date.
            convert_from (str): The source ticker.
            convert_to (str): The target ticker.
            multiplier (float): The conversion multiplier.
        """

    def to_json(self, fmt: str = 'str') -> str | dict:
        """Export every bucket."""

    def from_json(self, json_str: str | bytes | dict, with_used: bool = False) -> Inventory:
        """Apply an exported payload."""

    def dump(self, file_path: str | pathlib.Path) -> None:
        """Write the inventory to a JSON file."""

    def load(self, file_path: str | pathlib.Path, with_used: bool = False) -> None:
        """Replace the inventory with the contents of a JSON file."""

    def to_csv(self, file_path: str | pathlib.Path) -> None:
        """Write the per-ticker long/short volumes to a CSV file."""

    @property
    def tickers(self) -> set[str]:
        """The tickers carrying at least one lot or usage record."""

    @property
    def info(self) -> pd.DataFrame:
        """The long/short volumes as a DataFrame."""

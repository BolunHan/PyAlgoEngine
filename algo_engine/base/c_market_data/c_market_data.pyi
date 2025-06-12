import abc
import enum
from datetime import datetime
from typing import Any, Self


class OrderType(enum.IntEnum):
    """Enum representing different order types."""
    ORDER_UNKNOWN: int = ...
    ORDER_CANCEL: int = ...
    ORDER_GENERIC: int = ...
    ORDER_LIMIT: int = ...
    ORDER_LIMIT_MAKER: int = ...
    ORDER_MARKET: int = ...
    ORDER_FOK: int = ...
    ORDER_FAK: int = ...
    ORDER_IOC: int = ...


class DataType(enum.IntEnum):
    """Enum representing different market data types."""
    DTYPE_UNKNOWN: int = ...
    DTYPE_INTERNAL: int = ...
    DTYPE_MARKET_DATA: int = ...
    DTYPE_TRANSACTION: int = ...
    DTYPE_ORDER: int = ...
    DTYPE_TICK_LITE: int = ...
    DTYPE_TICK: int = ...
    DTYPE_BAR: int = ...
    DTYPE_REPORT: int = ...
    DTYPE_INSTRUCTION: int = ...


class MarketData(object, metaclass=abc.ABCMeta):
    """
    Abstract base class for all market data types.

    This includes InternalData, TransactionData, OrderData, BarData, TickData, and TickDataLite.
    Note this is a virtual parent class - implementations don't need to inherit from it directly.

    Any field in the underlying buffer cannot be set directly to avoid accidental reversed contamination.
    """

    __dict__: dict[str, Any]
    _data_addr: int

    def __init__(self, ticker: str, timestamp: float, **kwargs: Any) -> None:
        """
        Initialize market data.

        Note any field not specified in the constructor is not stored in the underlying buffer. And these fields will not be distributed by the MarketDataBuffer.

        Args:
            ticker: The financial instrument identifier
            timestamp: Unix timestamp of the data
            **kwargs: Additional data-specific attributes
        """
        ...

    @classmethod
    def buffer_size(cls) -> int:
        """
        Get the fixed buffer size for this market data type.

        All instances of the same market data class have the same size, which is the length of the bytes returned by to_bytes method.

        Returns:
            Size in bytes needed to store serialized data
        """
        ...

    def to_bytes(self) -> bytes:
        """
        Serialize the market data instance to bytes.

        Returns:
            Byte representation of the data
        """
        ...

    @classmethod
    def from_bytes(cls, data: bytes) -> Self:
        """
        Reconstruct a market data instance from bytes.

        Args:
            data: Serialized byte data

        Returns:
            Reconstructed market data instance
        """
        ...

    def __copy__(self) -> Self:
        """
        Create a deep copy of the market data instance.

        The underlying buffer is also copied.

        Returns:
            New instance with copied underlying buffer
        """
        ...

    def __reduce__(self) -> tuple[Any, ...]:
        """Support for pickle protocol."""
        ...

    def __setstate__(self, state: Any) -> None:
        """Support for pickle protocol."""
        ...

    @property
    def ticker(self) -> str:
        """Get the financial instrument identifier."""
        ...

    @property
    def timestamp(self) -> float:
        """
        Get the Unix timestamp of this market data.

        Note: Represents the "last time" (not start time) to prevent future data issues.
        """
        ...

    @property
    def dtype(self) -> int:
        """Get the DataType int value for this market data class."""
        ...

    @property
    def topic(self) -> str:
        """
        Get the topic string identifying this market data.

        Example: "000016.SH.TickData"
        """
        ...

    @property
    def market_time(self) -> datetime:
        """
        Get the timestamp as a Python datetime object.

        The market_time is subjected to the timezone info. If necessary, set the tz in PROFILE.

        """
        ...

    @property
    def market_price(self) -> float:
        """
        Get the representative market price.

        For all MarketData except InternalData, this represents either:
        - last trade price or
        - mid-price (depending on data type)
        """
        ...


class InternalData(MarketData):
    """
    Special market data class for internal communications.

    Used for heartbeats, triggers, and callback protocols.
    """

    def __init__(
            self,
            ticker: str,
            timestamp: float,
            code: int,
            **kwargs: Any
    ) -> None:
        """
        Initialize internal data message.

        Args:
            ticker: Message identifier
            timestamp: Unix timestamp
            code: Protocol code to trigger
            **kwargs: Additional protocol-specific data
        """
        ...

    @property
    def code(self) -> int:
        """Get the protocol code this message triggers."""
        ...


class FilterMode:
    """A bitmask class for filtering different types of market data."""

    value: int
    """The underlying integer value of the bitmask"""

    # Class-level constants
    NO_INTERNAL: FilterMode = ...
    NO_CANCEL: FilterMode = ...
    NO_AUCTION: FilterMode = ...
    NO_ORDER: FilterMode = ...
    NO_TRADE: FilterMode = ...
    NO_TICK: FilterMode = ...

    def __init__(self, value: int = 0) -> None:
        """Initialize the filter with a bitmask value.

        Args:
            value: Initial bitmask value (defaults to 0)
        """

    @classmethod
    def all(cls) -> FilterMode:
        """Create a FilterMode with all filter flags enabled.

        Returns:
            FilterMode: A filter that blocks all message types
        """

    def __or__(self, other: FilterMode) -> FilterMode:
        """Combine filters using bitwise OR.

        Args:
            other: Another FilterMode value

        Returns:
            FilterMode: New combined filter
        """

    def __and__(self, other: FilterMode) -> FilterMode:
        """Intersect filters using bitwise AND.

        Args:
            other: Another FilterMode value

        Returns:
            FilterMode: New intersected filter
        """

    def __contains__(self, other: FilterMode) -> bool:
        """Check if this filter contains all flags of another filter.

        Args:
            other: FilterMode to test against

        Returns:
            bool: True if all flags in other are set in this filter
        """

    def __repr__(self) -> str:
        """Return a string representation of the filter.

        Returns:
            str: String showing hex value and active flags
        """

    def mask_data(self, market_data: MarketData) -> bool:
        """Check if market data passes through this filter.

        Args:
            market_data: Market data object to check

        Returns:
            bool: True if data should pass through filter
        """

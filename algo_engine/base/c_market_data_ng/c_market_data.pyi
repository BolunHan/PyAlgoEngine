import ctypes
import enum
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Self, Annotated
from warnings import deprecated


@dataclass
class ValueRange:
    lo: int
    hi: int


UINTPTR_MAX = ctypes.c_void_p(-1).value
SIZE_MAX = ctypes.c_size_t(-1).value
uintptr_t = Annotated[int, ValueRange(0, UINTPTR_MAX), ctypes.c_void_p]
size_t = Annotated[int, ValueRange(0, SIZE_MAX), ctypes.c_size_t]


class DataType(enum.IntEnum):
    """Enum representing different market data types."""
    DTYPE_UNKNOWN: DataType
    DTYPE_INTERNAL: DataType
    DTYPE_MARKET_DATA: DataType
    DTYPE_TRANSACTION: DataType
    DTYPE_ORDER: DataType
    DTYPE_TICK_LITE: DataType
    DTYPE_TICK: DataType
    DTYPE_BAR: DataType
    DTYPE_REPORT: DataType
    DTYPE_INSTRUCTION: DataType


class EnvConfigContext:
    """Context manager for temporary environment configuration changes."""

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize the context with configuration changes.

        Args:
            **kwargs: Configuration key-value pairs to set temporarily
        """
        ...

    def __enter__(self) -> EnvConfigContext:
        """Enter the context, applying configuration changes."""
        ...

    def __exit__(self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: Any) -> None:
        """Exit the context, reverting configuration changes."""
        ...

    def __call__(self, func: Callable[[...], Any]) -> Callable[[...], Any]:
        """Decorator to apply the context to a function."""
        ...

    def __or__(self, other: EnvConfigContext) -> EnvConfigContext:
        """
        Combine two EnvConfigContext instances.

        Args:
            other: Another EnvConfigContext instance

        Returns:
            A new EnvConfigContext with combined configurations
        """
        ...

    def __invert__(self) -> EnvConfigContext:
        """
        Invert the EnvConfigContext.

        Returns:
            A new EnvConfigContext that reverts the configurations set in the original.
        """
        ...


MD_SHARED: EnvConfigContext
"""
EnvConfigContext instance to set flag for MarketData to use SHM allocator.
"""

MD_LOCKED: EnvConfigContext
"""
EnvConfigContext instance to set flag for MarketData to use thread safe mode.
"""

MD_FREELIST: EnvConfigContext
"""
EnvConfigContext instance to set flag for MarketData to use freelist. Have no effect when in MD_SHARED mode, which enforces its own free list.
"""


class MarketData(object):
    """
    Base class for all market data types.

    This includes ``InternalData``, ``TransactionData``, ``OrderData``, ``BarData``, ``TickData``, and ``TickDataLite``.
    Also includes the ``ReportData`` and ``InstructionData`` for trade reporting and instructions.

    Any field in the underlying buffer cannot be set directly to avoid accidental reversed contamination.

    This is a python wrapper of the underlying C struct buffer.

    By default, the ticker field is interned.

    The initializing of a MarketData instance creates a new underlying data buffer, in a way defined by environment configuration.
    - If MD_SHARED is set, the buffer is allocated from shared memory. Defaults to True
    - If MD_LOCKED is set, the buffer initialization is thread-safe. Defaults to False
    - If MD_FREELIST is set, the uses a freelist when deallocating buffer. Note that MD_SHARED enforces its own freelist. Defaults to True.

    To config the environment for a block of code, use the EnvConfigContext as a context manager or decorator.
    e.g.
    >>> from algo_engine.base import InternalData
    >>> with MD_SHARED | MD_LOCKED:
    ...     data = InternalData(...)

    Or as a decorator:

    >>> @(MD_SHARED | MD_LOCKED)
    ... def create_data(...) -> InternalData:
    ...     return InternalData(...)

    Attributes:
        owner (bool): Whether this instance owns the underlying data buffer.
        data_addr (uintptr_t): Internal address of the underlying data buffer.
    """

    owner: bool
    data_addr: uintptr_t

    def __init__(self) -> None:
        """
        Initialize an uninitialized empty MarketData instance.

        Should not be called directly.

        Use subclass constructors instead.
        """
        ...

    def __reduce__(self):
        """Support for pickle protocol."""
        ...

    def __setstate__(self, state):
        """Support for pickle protocol, for attributes, which not in the underlying buffer, assigned with python interface."""
        ...

    def __copy__(self):
        """
        Create a deep copy of the market data instance.

        The underlying buffer is also copied, but local-process only.
        Use ENV_SHARED context to initialize a market_data in shared memory.

        Returns:
            New instance with copied underlying buffer
        """
        ...

    @staticmethod
    def buffer_size(dtype: DataType) -> size_t:
        """
        Get the size of the underlying buffer for a given DataType.

        Args:
            dtype: DataType enum value

        Returns:
            Size of the buffer in bytes
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

    @staticmethod
    def from_ptr(ptr: uintptr_t) -> MarketData:  # undocumented
        """
        Reconstruct a MarketData instance from a raw pointer.

        Args:
            ptr: Pointer address to the underlying data buffer

        Returns:
            MarketData instance wrapping the buffer at the given pointer
        """
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

    @property
    def price(self) -> float:
        """alias of market_price"""

    @property
    def address(self) -> str | None:
        """Get hex address of the underlying data buffer. None if not initialized."""
        ...


class FilterMode:
    """
    A pseudo-IntEnum bitmask class for filtering different types of market data.

    Each filter flag corresponds to a specific type of market data:
    - NO_INTERNAL: Filter out InternalData messages
    - NO_CANCEL: Filter out OrderData messages with cancel actions
    - NO_AUCTION: Filter out Auction-related messages
    - NO_ORDER: Filter out all OrderData messages
    - NO_TRADE: Filter out all TransactionData messages
    - NO_TICK: Filter out all TickData messages

    Attributes:
        value (int): The underlying integer value of the bitmask
    """
    value: int

    # Class-level constants
    NO_INTERNAL: FilterMode
    NO_CANCEL: FilterMode
    NO_AUCTION: FilterMode
    NO_ORDER: FilterMode
    NO_TRADE: FilterMode
    NO_TICK: FilterMode

    def __init__(self, value: int) -> None:
        """Initialize the filter with a bitmask value.

        Args:
            value: Initial bitmask value.
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

    def __invert__(self):
        """
        Bitwise NOT operator (~)

        Returns:
            FilterMode: a new inverted filter.
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


class ConfigViewer(object):
    """
    Viewer for compile-time and runtime configuration constants.

    Provides read-only access to configuration parameters defined at compile time.
    """

    def __init__(self) -> None:
        """Initialize the ConfigViewer instance."""
        ...

    @property
    def DEBUG(self):
        ...

    @property
    def TICKER_SIZE(self):
        ...

    @property
    def BOOK_SIZE(self):
        ...

    @property
    def ID_SIZE(self):
        ...

    @property
    def LONG_ID_SIZE(self):
        ...

    @deprecated
    @property
    def MAX_WORKERS(self):
        ...

    @property
    def MD_CFG_LOCKED(self):
        ...

    @property
    def MD_CFG_SHARED(self):
        ...

    @property
    def MD_CFG_FREELIST(self):
        ...

    @property
    def MD_CFG_BOOK_SIZE(self):
        ...


CONFIG: ConfigViewer

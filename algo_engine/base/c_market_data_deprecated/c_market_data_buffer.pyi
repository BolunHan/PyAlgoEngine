from typing import Any, Iterator, Union

from .c_candlestick import BarData
from .c_market_data import MarketData, InternalData
from .c_tick import TickData, TickDataLite
from .c_transaction import TransactionData, OrderData

MarketDataType = Union[
    InternalData,
    TransactionData,
    OrderData,
    TickData,
    TickDataLite,
    BarData
]


class MarketDataBuffer:
    """
    A memory-efficient buffer for storing and replaying market data.

    Uses a raw memory buffer divided into three parts:
    1. Header section for metadata
    2. Pointer array storing offsets to actual data
    3. Data array storing the market data instances

    The buffer can be sorted for temporal iteration with minimal overhead
    by only rearranging the pointer array.
    """

    def __init__(
            self,
            buffer: Any,  # Accepts bytes, bytearray or shared memory
            skip_initialize: bool = False,
            capacity: int = 0
    ) -> None:
        """
        Initialize the market data buffer.

        Args:
            buffer: Raw memory buffer to use for storage
            skip_initialize: If True, won't clear existing buffer contents
            capacity: Maximum number of items to store (0 = auto-calculate)
        """
        ...

    def __iter__(self) -> Iterator[MarketDataType]:
        """
        Returns an iterator that yields market data in temporal order.

        Resets the iteration index and sorts the pointer array by timestamp.
        """
        ...

    def __getitem__(self, idx: int) -> MarketDataType:
        """
        Get market data by index position.

        Args:
            idx: Index in the pointer array (0-based)

        Returns:
            The market data instance at the given index
        """
        ...

    def __len__(self) -> int:
        """Returns the number of market data items currently stored."""
        ...

    def __next__(self) -> MarketDataType:
        """Get the next market data item during iteration."""
        ...

    @classmethod
    def buffer_size(
            cls,
            n_internal_data: int = 0,
            n_transaction_data: int = 0,
            n_order_data: int = 0,
            n_tick_data_lite: int = 0,
            n_tick_data: int = 0,
            n_bar_data: int = 0
    ) -> int:
        """
        Calculate the required buffer size for given data counts.

        Args:
            n_*_data: Number of each market data type to accommodate

        Returns:
            Total required buffer size in bytes
        """
        ...

    def put(self, market_data: MarketData) -> None:
        """
        Store a market data instance in the buffer.

        Makes a deep copy of the data into the buffer.
        Any subsequent modifications to the original won't affect the stored copy.
        """
        ...

    def update(self, dtype: int, **kwargs: Any) -> None:
        """
        Directly update buffer with market data attributes.

        Args:
            dtype: Market data type (DataType enum value)
            **kwargs: Constructor arguments for the market data type
        """
        ...

    def sort(self) -> None:
        """Sort market data by timestamp."""
        ...

    def to_bytes(self) -> bytes:
        """Serialize the buffer contents to bytes."""
        ...

    @classmethod
    def from_bytes(cls, data: bytes, buffer: Any = None) -> MarketDataBuffer:
        """
        Create buffer from serialized data.

        Args:
            data: Serialized buffer data
            buffer: Optional target buffer to copy into

        Returns:
            New MarketDataBuffer instance
        """
        ...

    @classmethod
    def from_buffer(cls, buffer: Any) -> MarketDataBuffer:
        """
        Create buffer wrapper around existing memory.

        Args:
            buffer: Existing buffer (bytes, bytearray or shared memory)

        Returns:
            New MarketDataBuffer instance sharing the same memory
        """
        ...

    @property
    def ptr_capacity(self) -> int:
        """Maximum number of pointers/items that can be stored."""
        ...

    @property
    def ptr_tail(self) -> int:
        """Next write position in the pointer array."""
        ...

    @property
    def data_capacity(self) -> int:
        """Maximum data storage size in bytes."""
        ...

    @property
    def data_tail(self) -> int:
        """Next write position in the data array."""
        ...


class MarketDataRingBuffer:
    """
    FIFO ring buffer for streaming market data.

    Optimized for low-latency producer/consumer scenarios with
    blocking iteration support.
    """

    def __init__(self, buffer: Any, capacity: int = 0) -> None:
        """
        Initialize the ring buffer.

        Args:
            buffer: Raw memory buffer to use
            capacity: Maximum items to store (0 = auto-calculate)
        """
        ...

    def __len__(self) -> int:
        """Current number of items in the buffer."""
        ...

    def __call__(self, timeout: float = -1.0) -> Iterator[MarketDataType]:
        """
        Get a blocking iterator for the buffer.

        Args:
            timeout: Max seconds to wait for new data (-1 = infinite)

        Yields:
            Market data items as they arrive

        Raises:
            TimeoutError: If timeout reached before data available
        """
        ...

    def is_full(self) -> bool:
        """Check if the buffer has reached capacity."""
        ...

    def is_empty(self) -> bool:
        """Check if buffer contains no data."""
        ...

    def read(self, idx: int) -> bytes:
        """
        Get raw bytes for item at index.

        Args:
            idx: The index position of the pointer array

        Returns:
            Raw byte data
        """
        ...

    def put(self, market_data: MarketData) -> None:
        """Add market data to the buffer."""
        ...

    def get(self, idx: int) -> MarketDataType:
        """
        Get market data by index.

        Args:
            idx: The index position of the pointer array

        Returns:
            Market data instance
        """
        ...

    def listen(
            self,
            block: bool = True,
            timeout: float = -1.0
    ) -> MarketDataType:
        """
        Wait for and return the next market data item.

        Args:
            block: Whether to wait for data if empty
            timeout: Max seconds to wait (-1 = infinite)

        Returns:
            Next available market data

        Raises:
            TimeoutError: If timeout reached before data available
        """
        ...

    def collect_info(self) -> dict[str, Any]:
        """
        Gather buffer statistics and diagnostics.

        Returns:
            Dictionary of buffer metrics and state information
        """
        ...


class MarketDataConcurrentBuffer:
    """
    Multiprocessing safe ring buffer for producer/consumer workflows.

    Supports multiple worker processes consuming the same data stream,
    with each item only being freed after all workers have consumed it.
    Designed for use with shared memory (SharedMemory/RawArray).
    """

    def __init__(
            self,
            buffer: Any,
            n_workers: int,
            dtype: int = 0,
            capacity: int = 0
    ) -> None:
        """
        Initialize concurrent buffer.

        Args:
            buffer: Shared memory buffer
            n_workers: Number of worker processes
            dtype: Specific market data type to store (0 = any)
            capacity: Maximum items to store (0 = auto-calculate)
        """
        ...

    def __call__(
            self,
            worker_id: int,
            timeout: float = -1.0
    ) -> Iterator[MarketDataType]:
        """
        Get blocking iterator for a specific worker.

        Args:
            worker_id: Unique worker identifier (0 <= id < n_workers)
            timeout: Max seconds to wait for new data (-1 = infinite)

        Yields:
            Market data items for this worker
        """
        ...

    def ptr_head(self, worker_id: int) -> int:
        """
        Get current read head position for a worker.

        Args:
            worker_id: Worker identifier

        Returns:
            Current pointer array position
        """
        ...

    def data_head(self, worker_id: int) -> int:
        """
        Get current data head position for a worker.

        Args:
            worker_id: Worker identifier

        Returns:
            Current data array position
        """
        ...

    def is_full(self) -> bool:
        """Check if the buffer has reached capacity."""
        ...

    def is_empty(self) -> bool:
        """
        Check if all workers have consumed all available data.
        """
        ...

    def read(self, idx: int) -> bytes:
        """
        Get raw bytes for item at index.

        Args:
            idx: The index position of the pointer array

        Returns:
            Raw byte data
        """
        ...

    def put(self, market_data: MarketData) -> None:
        """Add market data to the buffer."""
        ...

    def get(self, idx: int) -> MarketDataType:
        """
        Get market data by index.

        Args:
            idx: The index position of the pointer array

        Returns:
            Market data instance
        """
        ...

    def listen(
            self,
            worker_id: int,
            block: bool = True,
            timeout: float = -1.0
    ) -> MarketDataType:
        """
        Wait for next data item for specific worker.

        Args:
            worker_id: Worker identifier
            block: Whether to wait for data
            timeout: Max seconds to wait (-1 = infinite)

        Returns:
            Next market data item for this worker
        """
        ...

    def collect_header_info(self) -> dict[str, Any]:
        """
        Gather buffer header information.

        Returns:
            Dictionary of header metrics and state
        """
        ...

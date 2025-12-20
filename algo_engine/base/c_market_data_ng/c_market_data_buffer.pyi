from collections.abc import Iterator
from typing import Optional

from .c_market_data import MarketData


class MarketDataBufferCache(object):
    """Mutable staging area for collecting `MarketData` objects before flushing them into a shared buffer."""
    parent: MarketDataBuffer
    capacity: int
    size: int

    def __init__(self, capacity: int, parent: object) -> None:
        """Allocate internal arrays large enough for `capacity` entries and remember the originating buffer."""
        ...

    def __iter__(self) -> Iterator[MarketData]:
        """Yield the cached `MarketData` entries in insertion order."""
        ...

    def __len__(self) -> int:
        """Return the number of cached entries currently stored."""
        ...

    def __getitem__(self, idx: int) -> MarketData:
        """Return the `MarketData` at `idx`, supporting negative indices like a list."""
        ...

    def __enter__(self) -> MarketDataBufferCache:
        """Reset the cache so it can be used inside a context manager block."""
        ...

    def __exit__(self, exc_type, exc_value, traceback) -> Optional[bool]:
        """Flush cached entries back to the parent `MarketDataBuffer` before clearing the cache."""
        ...

    def put(self, market_data: MarketData) -> None:
        """Append a `MarketData` object to the cache, resizing the backing arrays if required."""
        ...

    def get(self, idx: int) -> MarketData:
        """Retrieve the cached `MarketData` at `idx` without removing it."""
        ...

    def clear(self) -> None:
        """Drop all cached references and reset the cache size back to zero."""
        ...


class MarketDataBuffer:
    """
    Resizable block buffer that stores serialized `MarketData` and exposes Python-friendly accessors.

    Supports a staging area for bucketed writes via `MarketDataBufferCache` to minimize
    the number of reallocations needed when inserting many entries.

    e.g.

    >>> buffer = MarketDataBuffer(ptr_cap=1024, data_cap=65536)
    >>> market_data_list = [MarketData(...), MarketData(...), ...]  # some list of MarketData
    >>> with buffer.cache() as cache:
    ...     for md in market_data_list:
    ...         cache.put(md)
    >>> buffer.sort()

    """

    header: object
    owner: bool
    iter_idx: int

    def __init__(self, ptr_cap: int, data_cap: int) -> None:
        """Allocate an new empty block buffer sized for `ptr_cap` pointers and `data_cap` bytes."""
        ...

    def __iter__(self) -> MarketDataBuffer:
        """Return the buffer itself after sorting so it can be iterated in chronological order."""
        ...

    def __getitem__(self, idx: int) -> MarketData:
        """Return the `MarketData` stored at `idx`, supporting negative indexing."""
        ...

    def __len__(self) -> int:
        """Return how many `MarketData` entries are currently stored."""
        ...

    def __next__(self) -> MarketData:
        """Produce the next `MarketData` in the sorted buffer, raising `StopIteration` at the end."""
        ...

    def cache(self) -> MarketDataBufferCache:
        """Create a `MarketDataBufferCache` bound to this buffer for batch writes."""
        ...

    def put(self, market_data: MarketData) -> None:
        """Serialize `market_data` into the buffer, expanding capacity as needed."""
        ...

    def get(self, idx: int) -> MarketData:
        """Fetch the `MarketData` at `idx` without mutating the buffer."""
        ...

    def sort(self) -> None:
        """Sort buffered entries according to their timestamp sequence."""
        ...

    def to_bytes(self) -> bytes:
        """Serialize the buffer into a bytes object that can be copied or persisted."""
        ...

    @classmethod
    def from_bytes(cls, data: bytes) -> MarketDataBuffer:
        """Create a new `MarketDataBuffer` by copying the serialized state contained in `data`."""
        ...

    @property
    def ptr_capacity(self) -> int:
        """Number of pointer slots currently allocated for the buffer."""
        ...

    @property
    def ptr_tail(self) -> int:
        """Current pointer tail offset, equivalent to the number of stored entries."""
        ...

    @property
    def data_capacity(self) -> int:
        """Number of data bytes currently reserved in the buffer."""
        ...

    @property
    def data_tail(self) -> int:
        """Current data tail offset, representing how many bytes hold serialized payloads."""
        ...


class Empty(Exception):
    """Raised when attempting to read from an empty `MarketDataRingBuffer`."""
    ...


class Full(Exception):
    """Raised when a `MarketDataRingBuffer` lacks enough pointer or data capacity."""
    ...


class MarketDataRingBuffer:
    """
    Fixed-capacity ring buffer that stores serialized `MarketData` and supports
    low-latency producer/consumer workflows with optional blocking reads.
    """
    iter_idx: int

    def __init__(self, ptr_cap: int, data_cap: int) -> None:
        """Allocate a ring buffer sized for `ptr_cap` pointers and `data_cap` bytes of payload storage."""
        ...

    def __iter__(self) -> MarketDataRingBuffer:
        """Return the buffer itself so it can be iterated over the currently stored entries."""
        ...

    def __getitem__(self, idx: int) -> MarketData:
        """Return the `MarketData` at `idx`, supporting Python-style negative indexing."""
        ...

    def __len__(self) -> int:
        """Return the number of `MarketData` entries that can be iterated without blocking."""
        ...

    def __next__(self) -> MarketData:
        """Produce the next `MarketData` in the ring, raising `StopIteration` when exhausted."""
        ...

    def put(self, market_data: MarketData) -> None:
        """Insert serialized market data into the ring buffer.

        Args:
            market_data: The `MarketData` instance to serialize and append.

        Raises:
            Full: If either the pointer array or backing byte buffer cannot
                accommodate the serialized payload.
        """
        ...

    def get(self, idx: int) -> MarketData:
        """Return the item at `idx` without mutating the buffer.

        Args:
            idx: Zero-based index of the item to fetch. Negative indices work
                like Python lists and count from the end.

        Returns:
            MarketData: The `MarketData` entry stored at the requested index.

        Raises:
            IndexError: If `idx` falls outside the readable range.
        """
        ...

    def listen(self, block: bool = True, timeout: float = 0) -> MarketData:
        """Wait for the next entry to become available.

        Args:
            block: When True, block until data arrives or the timeout elapses.
            timeout: Maximum number of seconds to wait when blocking. A value of
                0 means wait indefinitely.

        Returns:
            MarketData: The next available entry in chronological order.

        Raises:
            Empty: If no data is available immediately (non-blocking) or before
                the timeout expires.
            TimeoutError: If blocking is enabled and the timeout elapses.
        """
        ...

    @property
    def ptr_capacity(self) -> int:
        """Total pointer slots provisioned for the ring buffer."""
        ...

    @property
    def ptr_head(self) -> int:
        """Head pointer index indicating the next read position."""
        ...

    @property
    def ptr_tail(self) -> int:
        """Tail pointer index that advances with each successful write."""
        ...

    @property
    def data_capacity(self) -> int:
        """Total serialized byte capacity reserved for payload storage."""
        ...

    @property
    def data_tail(self) -> int:
        """Offset of the next free byte within the backing data buffer."""
        ...

    @property
    def is_empty(self) -> bool:
        """Return `True` when the ring buffer has no readable entries."""
        ...


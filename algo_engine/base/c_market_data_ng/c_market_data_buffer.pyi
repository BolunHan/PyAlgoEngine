from collections.abc import Iterator
from typing import Optional

from .c_market_data import MarketData


class InvalidBufferError(Exception):
    pass


class NotInSharedMemoryError(Exception):
    pass


class BufferFull(Exception):
    pass


class BufferEmpty(Exception):
    pass


class PipeTimeoutError(Exception):
    pass


class BufferCorruptedError(Exception):
    pass


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


class MarketDataBuffer(object):
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
        """Serialize `market_data` into the buffer, expanding capacity as needed.

        Raises:
            ValueError: If arguments are invalid.
            BufferFull: If the buffer lacks capacity for the serialized payload.
        """
        ...

    def get(self, idx: int) -> MarketData:
        """Fetch the `MarketData` at `idx` without mutating the buffer."""
        ...

    def sort(self) -> None:
        """Sort buffered entries according to their timestamp sequence.

        Raises:
            InvalidBufferError: If the underlying buffer header is invalid.
        """
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

    def put(self, market_data: MarketData, block: bool = True, timeout: float = 0.0) -> None:
        """Insert serialized market data into the ring buffer.

        Args:
            market_data: The `MarketData` instance to serialize and append.
            block: When True, block until space is available or the timeout elapses.
            timeout: Maximum number of seconds to wait when blocking. A value of 0 means wait indefinitely.

        Raises:
            ValueError: If arguments are invalid.
            BufferFull: If either the pointer array or backing byte buffer cannot
                accommodate the serialized payload.
            PipeTimeoutError: If blocking is enabled and the timeout elapses.
        """
        ...

    def get(self, idx: int) -> MarketData:
        """Return the item at `idx` without mutating the buffer.

        Args:
            idx: Zero-based index of the item to fetch. Negative indices are supported.

        Returns:
            MarketData: The `MarketData` entry stored at the requested index.

        Raises:
            IndexError: If `idx` falls outside the readable range.
        """
        ...

    def listen(self, block: bool = True, timeout: float = 0.0) -> MarketData:
        """Wait for the next entry to become available.

        Args:
            block: When True, block until data arrives or the timeout elapses.
            timeout: Maximum number of seconds to wait when blocking. A value of 0 means wait indefinitely.

        Returns:
            MarketData: The next available entry in chronological order.

        Raises:
            ValueError: If arguments are invalid.
            BufferEmpty: If no data is available immediately (non-blocking) or before the timeout expires.
            BufferCorruptedError: If the buffer reports a corrupted offset.
            PipeTimeoutError: If blocking is enabled and the timeout elapses.
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


class MarketDataConcurrentBuffer:
    """Concurrent multi-consumer buffer backed by shared memory.

    Provides a simple API for multiple workers to consume `MarketData` entries
    produced by a single writer. Writes can optionally block until space is
    available, and reads can optionally block until data arrives.

    Example:

    >>> buf = MarketDataConcurrentBuffer(n_workers=2, capacity=1024)
    >>> buf.put(MarketData(...))
    >>> md = buf.listen(worker_id=0, block=True, timeout=1.0)
    >>> md
    """

    def __init__(self, n_workers: int, capacity: int) -> None:
        """Create a concurrent buffer for `n_workers` with ring capacity `capacity`."""
        ...

    def put(self, market_data: MarketData, block: bool = True, timeout: float = 0.0) -> None:
        """Append a `MarketData` to the buffer.

        Args:
            market_data: The `MarketData` instance to append. If not already in shared memory, it will be COPIED into SHM.
            block: When True, block until space becomes available or the timeout elapses.
            timeout: Maximum number of seconds to wait when blocking. A value of 0 means wait indefinitely.

        Raises:
            ValueError: If arguments are invalid.
            NotInSharedMemoryError: If the data is not SHM-backed. Python / Cython interface will not raise this error but c-API just might.
            BufferFull: If the buffer is full and non-blocking mode is used.
            PipeTimeoutError: If blocking is enabled and the timeout elapses.
        """
        ...

    def listen(self, worker_id: int, block: bool = True, timeout: float = 0.0) -> MarketData:
        """Fetch the next item for `worker_id`.

        Args:
            worker_id: The consumer slot to read from.
            block: When True, block until data arrives or the timeout elapses.
            timeout: Maximum number of seconds to wait when blocking. A value of 0 means wait indefinitely.

        Returns:
            MarketData: The next available entry for the given worker.

        Raises:
            ValueError: If arguments are invalid or the worker is disabled.
            IndexError: If `worker_id` is out of range.
            BufferEmpty: If no data is available immediately (non-blocking).
            PipeTimeoutError: If blocking is enabled and the timeout elapses.
        """
        ...

    def is_worker_empty(self, worker_id: int) -> bool:
        """Return `True` when there is no data available for `worker_id`."""
        ...

    def is_empty(self) -> bool:
        """Return `True` when there is no data available for any worker."""
        ...

    def is_full(self) -> bool:
        """Return `True` when the buffer cannot accept new entries."""
        ...

    def disable_worker(self, worker_id: int) -> None:
        """Disable `worker_id`, preventing it from receiving new entries."""
        ...

    def enable_worker(self, worker_id: int) -> None:
        """
        Enable `worker_id`, allowing it to receive new entries.

        This also resets the worker's read pointer to the current write position.
        Re-enabling a worker that was already enabled will also reset its read pointer.
        """
        ...

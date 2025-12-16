"""POSIX shared-memory backed allocator

This module provides a Python wrapper around a C-backed shared memory allocator.

The allocator:
- Is backed by POSIX shared memory (shm). It creates named shm objects and maps them into a reserved virtual address region so multiple processes can agree on fixed addresses for mapped pages.
- Requests a fixed virtual address region so pages can be mapped at stable addresses across processes (this enables multi-process access to the same in-region pointers).
- Is a thin wrapper around the underlying C implementation (`c_shm_allocator`).
- By default reserves a ``128 GiB`` virtual address space for page mappings but the default can be changed at compile-time using the `SHM_ALLOCATOR_DEFAULT_REGION_SIZE` constant (documented below).
- Is POSIX-only (depends on shm_open/mmap semantics available on POSIX systems such as Linux).

Compile-time constants (names and defaults)

These values are provided by the C header and may be changed at
compile time when building the extension. They are documented here for
consumers; the runtime values come from the compiled extension.

- DEFAULT_AUTOPAGE_CAPACITY: 64 * 1024              # 64 KiB — first/auto page size
- MAX_AUTOPAGE_CAPACITY: 16 * 1024 * 1024           # 16 MiB — maximum auto-grown page
- DEFAULT_AUTOPAGE_ALIGNMENT: 4 * 1024              # 4 KiB — page alignment
- SHM_ALLOCATOR_PREFIX: "/c_shm_allocator"          # prefix used for allocator SHM names
- SHM_PAGE_PREFIX: "/c_shm_page"                    # prefix used for page SHM names
- SHM_NAME_LEN: 256                                 # maximum SHM object name length used by the implementation
- SHM_ALLOCATOR_DEFAULT_REGION_SIZE: 128 << 30      # 128 GiB default reserved region
"""
import ctypes
from collections.abc import Generator
from dataclasses import dataclass
from typing import Annotated


@dataclass
class ValueRange:
    lo: int
    hi: int


UINTPTR_MAX = ctypes.c_void_p(-1).value
uintptr_t = Annotated[int, ValueRange(0, UINTPTR_MAX), ctypes.c_void_p]


class SharedMemoryBlock(object):
    """Represents an allocated block inside a shared memory page.

    Instances are thin Python wrappers around an internal C block header.
    Consumers can inspect size/capacity and traverse linked lists of
    free/allocated blocks. The raw underlying memory buffer is not
    exposed directly from the stub — use the C implementation at runtime
    if you need to read/write data.

    Attributes:
        owner (bool): True when this wrapper owns the block and is responsible for freeing it on destruction (runtime detail).
    """

    owner: bool

    def __init__(self, block: uintptr_t = 0, owner: bool = False) -> None:
        """Create a block wrapper.
        This constructor is primarily a typing helper for callers; the
        runtime C extension returns real wrappers. Fields are informational.

        Notes:
            The block address must point to the start of the block header.
            If you wish to reconstruct a block from its buffer address, use the ``from_buffer`` classmethod.

        Args:
            block (int): The virtual address of the block header.
            owner (bool): Whether this wrapper owns the block.
        """

    def __repr__(self) -> str: ...

    @classmethod
    def from_buffer(cls, buffer_addr: uintptr_t) -> SharedMemoryPage:
        """Create a block wrapper from a buffer address.

        Notes:
            The buffer address must point to the start of the user-visible buffer area of a block.
            The method computes the block header address by subtracting the header size.
            If you wish to reconstruct a block from its header address, use the constructor directly.

        Args:
            buffer_addr (int): The virtual address of the block buffer.
        Returns:
            SharedMemoryBlock: The block wrapper.
        """

    @property
    def size(self) -> int:
        """Size (in bytes) of the user-visible allocation.

        Returns:
            int: The number of bytes requested when the block was allocated. -1 if uninitialized. 0 if freed.
        """

    @property
    def capacity(self) -> int:
        """Capacity (in bytes) available in the block.

        Returns:
            int: The effective capacity that can be used (rounded/aligned). -1 if uninitialized.
        """

    @property
    def next_free(self) -> SharedMemoryBlock | None:
        """Next block in the allocator free-list, if any.

        Returns:
            SharedMemoryBlock | None: The next free block or None.
        """

    @property
    def next_allocated(self) -> SharedMemoryBlock | None:
        """Next block in the page's allocation list, if any.

        Returns:
            SharedMemoryBlock | None: The next allocated block on the page or None.
        """

    @property
    def buffer(self) -> memoryview:
        """The raw memory buffer of the block.

        Returns:
            memoryview: A memoryview of the block buffer. None if uninitialized or zero-sized.
        """

    @property
    def address(self) -> str | None:
        """The virtual address of the block buffer (as hex string).

        Returns:
            str | None: The block buffer address in hex or None if uninitialized.
        """


class SharedMemoryPage(object):
    """Represents a mapped shared-memory page.

    A page contains allocation metadata and is mapped into the allocator's reserved virtual region.
    You can inspect name, capacity and currently occupied bytes and iterate allocated blocks on the page.
    """

    def __init__(self, page_addr: uintptr_t) -> None:
        """Create a page wrapper.

        This constructor is primarily a typing helper for callers; the
        runtime C extension returns real wrappers. Fields are informational.

        Args:
            page_addr (int): The base virtual address of the mapped page.
        """

    def __repr__(self) -> str: ...

    def allocated(self) -> Generator[SharedMemoryBlock]:
        """Iterate allocated blocks on this page, latest-first (LIFO).

        Notes:
            The buffers reused by allocator request method, will not update their position.

        Yields:
            Generator[SharedMemoryBlock]: Allocated blocks (block headers wrapped).
        """

    def reclaim(self) -> None:
        """Best-effort reclaim freed blocks on this page back to unallocated state.

        This is the only way to reduce the occupied byte count on a page, and returning the buffer from free_list to the memory page.
        This method does NOT acquire mutex lock, the caller must ensure thread-safety if needed.
        """

    @property
    def name(self) -> str | None:
        """The OS shared-memory object name for this page.

        Returns:
            str: The page SHM name (with leading '/') or None if uninitialized.
        """

    @property
    def capacity(self) -> int:
        """Total capacity (in bytes) of the page.

        Returns:
            int: The full page size. 0 if uninitialized.
        """

    @property
    def occupied(self) -> int:
        """Number of bytes currently occupied/allocated on this page.

        Returns:
            int: Occupied bytes. 0 if uninitialized.
        """


class SharedMemoryAllocator(object):
    """Top-level allocator managing a large virtual region and multiple pages.

    The allocator reserves a large virtual region and maps page-sized shared-memory objects into that region. Typical usage:

    >>> alloc = SharedMemoryAllocator(1 << 32)  # region_size optional
    >>> page = alloc.extend(1 << 16)
    >>> block = alloc.calloc(1024)
    >>> alloc.free(block)
    >>> alloc.reclaim()

    This class is a singleton: the extension provides a single global `ALLOCATOR` instance
    and attempts to create a second Python-level instance will raise or be prevented by the runtime.
    Only c/cython code that directly manipulates the extension internals can create more allocator instances.

    Attributes:
        owner (bool): True when this object represents the owner of the runtime allocator and is responsible for freeing/shutting down global resources.
    """

    owner: bool

    def __init__(self, region_size: int = 0) -> None:
        """Create and initialize an allocator context.

        The constructor reserves a large virtual region (defaults to 128GiB) and creates the allocator metadata backed by a named shared memory object.

        Args:
            region_size (int): Virtual region size to reserve. Use 0 will skip the initializing of the underlying shm, leaving the python wrapper uninitialized.

        Raises:
            OSError: If the underlying C allocator object cannot be created.
        """

    def __repr__(self) -> str: ...

    @classmethod
    def get_pid(cls, shm_name: str) -> int:
        """Get the creator PID of a given allocator or page SHM object.

        Args:
            shm_name (str): The SHM object name (with leading '/').

        Returns:
            int: The creator PID.
        """

    def extend(self, capacity: int = 0, with_lock: bool = True) -> SharedMemoryPage:
        """Extend the allocator with a new page and return a page wrapper.

        Extension sizing Policy:
        - Explict required capacity: Request a named page with aligned total capacity including the meta header.
        - Auto-sizing (capacity=0): The allocator will pick a page size based on the following policy:
            1. If no pages are mapped yet, use DEFAULT_AUTOPAGE_CAPACITY (64 KiB).
            2. If pages are already mapped, double the last mapped page capacity up to MAX_AUTOPAGE_CAPACITY (16 MiB).

        Args:
            capacity (int): Payload capacity requested for the new page. If 0, the allocator will auto-size following the configured policy.
            with_lock (bool): Whether to use the allocator's internal mutex while extending. Defaults to True.

        Returns:
            SharedMemoryPage: A wrapper for the newly mapped page.

        Raises:
            OSError: On mapping/allocation failure.
        """

    def calloc(self, size: int, with_lock: bool = True) -> SharedMemoryBlock:
        """Request and allocate a **new**, **zero-initialized** memory from the allocator active page.

        If the active page does not have enough space, a new page will be auto-extended to fit the need.

        Args:
            size (int): Number of user bytes requested.
            with_lock (bool): Whether to use the allocator's internal mutex.

        Returns:
            SharedMemoryBlock: Wrapper for the allocated block.

        Raises:
            OSError: On allocation failure.
        """

    def request(self, size: int, scan_all_pages: bool = True, with_lock: bool = True) -> SharedMemoryBlock:
        """Request a free **zero-initialized** memory from the allocator.

        This method will first attempt to reuse blocks from the allocator free-list.
        And if no suitable free block is found, it traverses **ALL** the memory pages to find an available buffer space.
        And if still no space is found, a new page will be auto-extended to fit the need.

        Args:
            size (int): Number of user bytes requested.
            scan_all_pages (bool): Whether to scan all mapped pages for available space before extending a new page.
            with_lock (bool): Whether to use the allocator's internal mutex.

        Returns:
            SharedMemoryBlock: Wrapper for the allocated block.

        Raises:
            OSError: On allocation failure.
        """

    def free(self, buffer: SharedMemoryBlock, with_lock: bool = True) -> None:
        """Return a previously allocated block back to the allocator free list.

        Args:
            buffer (SharedMemoryBlock): The block to free.
            with_lock (bool): Whether to use the allocator's internal mutex.
        """

    def reclaim(self, with_lock: bool = True) -> None:
        """Best-effort reclaim of freed blocks across all mapped pages.

        See ``SharedMemoryPage.reclaim`` for details.

        Args:
            with_lock (bool): Whether to use the allocator's internal mutex.
        """

    def dangling(self) -> list[str]:
        """Return all allocator SHM names that appear dangling.

        The method enumerates shared-memory objects in /dev/shm, parses the
        embedded PID using the same parsing rules as the C helper, and
        returns names whose owner PID no longer exists.

        Returns:
            list[str]: A list of allocator SHM names (each with a leading '/').
        """

    def dangling_pages(self) -> list[str]:
        """Return all page SHM names that appear dangling.

        Returns:
            list[str]: A list of page SHM names (each with a leading '/').
        """

    def cleanup_dangling(self) -> None:
        """Unlink dangling allocator and page SHM objects.

        This is a best-effort cleanup that mirrors the behavior of the C
        implementation's `c_shm_clear_dangling` helper.
        """

    def pages(self) -> Generator[SharedMemoryPage]:
        """Iterate mapped pages from newest to oldest.

        Yields:
            Generator[SharedMemoryPage]: Page wrappers in newest-first order.
        """

    def allocated(self) -> Generator[SharedMemoryBlock]:
        """Iterate all allocated blocks across all pages.

        Yields:
            Generator[SharedMemoryBlock]: Allocated blocks across all pages.
        """

    def free_list(self) -> Generator[SharedMemoryBlock]:
        """Iterate the allocator-level free list (recyclable blocks).

        Yields:
            Generator[SharedMemoryBlock]: Free blocks available for reuse.
        """

    # Metadata
    @property
    def name(self) -> str | None:
        """The allocator SHM object name (with leading '/') or None.

        Returns:
            str | None: Allocator SHM name or None if uninitialized.
        """

    @property
    def pid(self) -> int:
        """PID of the process that created the allocator SHM object.

        Returns:
            int: Creator PID or -1 when unknown.
        """

    @property
    def region(self) -> int:
        """Base virtual address of the reserved region (as integer).

        Returns:
            int: Region base address or -1 when uninitialized.
        """

    @property
    def region_addr(self) -> str:
        """Base virtual address of the reserved region (as hex string).

        Returns:
            str: Region base address in hex or '0x-1' when uninitialized.
        """

    @property
    def region_size(self) -> int:
        """Virtual region size reserved by the allocator.

        Returns:
            int: Region size in bytes.
        """

    @property
    def mapped_size(self) -> int:
        """Total mapped bytes within the reserved region.

        Returns:
            int: Number of bytes mapped into the region.
        """

    @property
    def mapped_pages(self) -> int:
        """Number of pages currently mapped into the region.

        Returns:
            int: Count of mapped pages.
        """

    @property
    def active_page(self) -> SharedMemoryPage | None:
        """Wrapper for the allocator's current active page, or None.

        Returns:
            SharedMemoryPage | None: Active page wrapper or None.
        """


# Module-level convenience
ALLOCATOR: SharedMemoryAllocator | None  # Global singleton allocator instance


def cleanup() -> None:
    """Module-level cleanup performed at interpreter exit.

    This function attempts to release any global resources or dangling
    shared-memory objects that the extension may have left behind.
    """

"""Local heap-backed allocator

This module exposes a Python veneer over the in-process heap allocator
implemented in ``c_heap_allocator``. It mirrors the shared-memory
allocator API but stores all state in the caller's address space.

Compile-time constants (documented defaults):
    - DEFAULT_AUTOPAGE_CAPACITY:   64 * 1024        # 64 KiB
    - MAX_AUTOPAGE_CAPACITY:       16 * 1024 * 1024 # 16 MiB
    - DEFAULT_AUTOPAGE_ALIGNMENT:  4 * 1024         # 4 KiB
"""
from __future__ import annotations

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


class HeapMemoryBlock:
    """Typed wrapper around an allocator block header."""

    owner: bool

    def __init__(self, block: uintptr_t = 0, owner: bool = False) -> None: ...

    def __repr__(self) -> str: ...

    @property
    def size(self) -> int:
        """Requested payload size in bytes (-1 if uninitialized)."""

    @property
    def capacity(self) -> int:
        """Aligned capacity of the block (-1 if uninitialized)."""

    @property
    def next_free(self) -> HeapMemoryBlock | None:
        """Next entry in the allocator free list, if any."""

    @property
    def next_allocated(self) -> HeapMemoryBlock | None:
        """Next entry in the owning page's allocation list, if any."""

    @property
    def parent_page(self) -> HeapMemoryPage | None:
        """Owning page wrapper, or None when detached."""

    @property
    def buffer(self) -> memoryview:
        """Memoryview of the user buffer."""

    @property
    def address(self) -> str | None:
        """Hex string of the buffer address."""


class HeapMemoryPage:
    """Represents a single heap-managed page."""

    def __init__(self, page_addr: uintptr_t = 0) -> None: ...

    def __repr__(self) -> str: ...

    @classmethod
    def from_buffer(cls, buffer_addr: uintptr_t) -> HeapMemoryPage:
        """Reconstruct a page wrapper from one of its block buffers."""

    def allocated(self) -> Generator[HeapMemoryBlock]:
        """Iterate allocated blocks newest-first."""

    def reclaim(self) -> None:
        """Best-effort reclaim of freed blocks back to unused space."""

    @property
    def capacity(self) -> int:
        """Total payload capacity of the page in bytes."""

    @property
    def occupied(self) -> int:
        """Currently occupied bytes on the page."""

    @property
    def allocator(self) -> HeapAllocator | None:
        """Owning allocator wrapper, or None."""


class HeapAllocator:
    """Top-level heap allocator (singleton in practice)."""

    owner: bool

    def __init__(self, owner: bool = True) -> None: ...

    def __repr__(self) -> str: ...

    def extend(self, capacity: int = 0, with_lock: bool = True) -> HeapMemoryPage:
        """Add a page (auto-sized when capacity is 0)."""

    def calloc(self, size: int, with_lock: bool = True) -> HeapMemoryBlock:
        """Allocate zeroed memory from the active page (auto-extending)."""

    def request(
        self,
        size: int,
        with_lock: bool = True,
        scan_all_pages: bool = True,
    ) -> HeapMemoryBlock:
        """Request a zeroed block, optionally scanning older pages."""

    def free(self, buffer: HeapMemoryBlock, with_lock: bool = True) -> None:
        """Return a block to the allocator free list."""

    def reclaim(self, with_lock: bool = True) -> None:
        """Reclaim freed space across all pages."""

    def pages(self) -> Generator[HeapMemoryPage]:
        """Iterate mapped pages newest-first."""

    def allocated(self) -> Generator[HeapMemoryBlock]:
        """Iterate allocated blocks across all pages."""

    def free_list(self) -> Generator[HeapMemoryBlock]:
        """Iterate the allocator-wide free list."""

    @property
    def mapped_pages(self) -> int:
        """Count of pages currently mapped by the allocator."""

    @property
    def active_page(self) -> HeapMemoryPage | None:
        """Most recently mapped page, if any."""


ALLOCATOR: HeapAllocator
DEFAULT_AUTOPAGE_CAPACITY: int
MAX_AUTOPAGE_CAPACITY: int
DEFAULT_AUTOPAGE_ALIGNMENT: int

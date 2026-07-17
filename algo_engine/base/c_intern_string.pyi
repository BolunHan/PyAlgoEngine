from cbase.intern_string.c_intern_string import InternString, InternStringPool

POOL: InternStringPool
"""The global SHM-backed ``InternStringPool`` for cross-process sharing.

Backed by ``MD_SHM_ALLOCATOR`` — child processes inherit the shared
memory mapping and see all entries regardless of when they were interned.
"""

INTRA_POOL: InternStringPool
"""The global heap-backed ``InternStringPool`` for intra-process use.

Backed by ``MD_HEAP_ALLOCATOR``.  After ``fork()`` the child sees a
COW copy of pre-fork entries but post-fork insertions are isolated to
each process.
"""

C_POOL: int
"""Raw pointer (``uintptr_t``) to the C ``istr_map`` backing ``POOL``."""

C_INTRA_POOL: int
"""Raw pointer (``uintptr_t``) to the C ``istr_map`` backing ``INTRA_POOL``."""

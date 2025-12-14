from collections.abc import Generator
from typing import Never, Any


class InternStringPool(object):
    """A pool that stores interned (deduplicated) strings.

    The real implementation is provided by the compiled c extension;
    This class is merely a python wrapper.

    This class is singleton; use the module-level `POOL` instance.
    Trying to init new instance from python interface will raise runtime error.
    """

    def __init__(self, *args, **kwargs) -> Never:
        """Create (not intended for direct Python use).

        The actual pool is created by the compiled extension and exposed as
        the module-level ``POOL``. Calling this constructor from Python will
        raise at runtime in typical builds.
        """

    def __len__(self) -> int:
        """Return the number of unique interned strings in the pool.

        The returned value matches the pool's ``size`` property.
        """

    def __getitem__(self, key: str) -> InternString:
        """Return an ``InternString`` view for the given key.

        This provides mapping-like access: ``pool['s']`` returns an
        InternString if ``s`` is present, otherwise behavior is
        implementation-defined (may raise or return an uninitialized view).
        """

    def istr(self, string: str, with_lock: bool = False) -> InternString:
        """Intern (or lookup) ``string`` in the pool and return a view.

        Parameters
        - string: the Python str to intern
        - with_lock: whether to acquire the pool lock during insertion
        """

    def internalized(self) -> Generator[InternString]:
        """Yield all current interned entries as ``InternString`` views.

        The generator yields zero or more ``InternString`` objects. The
        order is implementation-dependent.
        """

    @property
    def size(self) -> int:
        """The current number of unique interned strings (same as len(pool))."""

    @property
    def address(self) -> str:
        """Hex string address of the underlying C header or storage.

        The exact format is implementation-defined but is typically a
        string like ``'0x7fabc...'``.
        """


class InternString(object):
    """A python wrapper to view interned string from an InternStringPool.
    """

    def __gt__(self, other: object) -> bool:
        """Return ``True`` if this interned string is greater than ``other``.

        The comparison supports comparing with another ``InternString`` or
        a plain Python ``str``.
        """

    def __eq__(self, other: object) -> bool:
        """Equality comparison with another interned string or Python str."""

    def __hash__(self) -> int:
        """Return the cached hash value for the interned string.

        The value is suitable for use in dictionaries and sets when the
        object is fully initialized.
        """

    def __repr__(self) -> str:
        """Return a short, informative representation (safe to call)."""

    @property
    def intern_pool(self) -> InternStringPool:
        """Return the ``InternStringPool`` that owns this interned entry."""

    @property
    def string(self) -> str:
        """Return the Python ``str`` content of this interned entry.

        Accessing this property may raise if the view is uninitialized.
        """

    @property
    def hash_value(self) -> int:
        """Return the raw integer hash stored for this interned string."""

    @property
    def address(self) -> str:
        """Return the hex address of the underlying C string buffer."""


POOL: InternStringPool
"""The global singleton multi-process InternStringPool instance for general use."""

INTRA_POOL: InternStringPool
"""The global singleton local thread InternStringPool instance for general use."""

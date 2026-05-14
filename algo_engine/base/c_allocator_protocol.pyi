from collections.abc import Callable
from typing import Any


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
EnvConfigContext instance to set flag for algo_engine.c_market_data to use SHM allocator.
"""

MD_LOCKED: EnvConfigContext
"""
EnvConfigContext instance to set flag for algo_engine.c_market_data to use thread safe mode.
"""

MD_FREELIST: EnvConfigContext
"""
EnvConfigContext instance to set flag for algo_engine.c_market_data to use freelist. Have no effect when in MD_SHARED mode, which enforces its own free list.
"""


class AllocatorProtocol(object):
    """Protocol for memory allocation with environment-based configuration.

    Manages an underlying `allocator_protocol` C structure, handling memory
    allocation and deallocation based on global environment settings.

    Attributes:
        protocol: **cython internal** Pointer to the underlying `allocator_protocol` C structure.
        owner: **cython internal** Boolean indicating if this instance owns the protocol and is
            responsible for its deallocation.
    """

    def __init__(self, size: int) -> None:
        """Initialize the AllocatorProtocol with a specified buffer size.

        Args:
            size: The size of the buffer to allocate in bytes.
        """
        ...

    @property
    def buf(self) -> memoryview:
        """Memory buffer managed by this allocator protocol.

        Returns:
            A memoryview representing the allocated buffer.
        """
        ...

    @property
    def size(self) -> int:
        """Size of the allocated buffer.

        Returns:
            The size of the buffer in bytes.
        """
        ...

    @property
    def with_shm(self) -> bool:
        """Indicates if the allocator uses shared memory.

        Returns:
            True if shared memory is used, False otherwise.
        """
        ...

    @property
    def with_lock(self) -> bool:
        """Indicates if the allocator uses locking for thread safety.

        Returns:
            True if locking is enabled, False otherwise.
        """
        ...

    @property
    def addr(self) -> int:
        """Memory address of the underlying allocator protocol structure.

        Returns:
            The memory address as an integer.
        """
        ...

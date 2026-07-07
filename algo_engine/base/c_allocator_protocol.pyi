from typing import Any

from cbase.allocator_protocol import AllocatorConfigContext


class MDConfigContext(AllocatorConfigContext):
    """Market-data specific configuration context.

    Extends ``AllocatorConfigContext`` to additionally update module-level
    ``MD_CFG_*`` globals, and binds to ``MD_DEFAULT_ALLOCATOR`` by default.

    Accepted keyword arguments (in addition to those inherited):

    * ``locked: bool`` — also updates ``MD_CFG_LOCKED``
    * ``shared: bool`` — also updates ``MD_CFG_SHARED``
    * ``freelist: bool`` — also updates ``MD_CFG_FREELIST``
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize the context with configuration changes.

        Args:
            **kwargs: Configuration key-value pairs to set temporarily
        """
        ...

    def __or__(self, other: MDConfigContext) -> MDConfigContext:
        """
        Combine two MDConfigContext instances.

        Args:
            other: Another MDConfigContext instance

        Returns:
            A new MDConfigContext with combined configurations
        """
        ...

    def __invert__(self) -> MDConfigContext:
        """
        Invert the MDConfigContext.

        Returns:
            A new MDConfigContext that reverts the configurations.
        """
        ...


MD_SHARED: MDConfigContext
"""
MDConfigContext instance to set flag for PyAlgoEngine to use SHM allocator.
"""

MD_LOCKED: MDConfigContext
"""
MDConfigContext instance to set flag for PyAlgoEngine to use thread safe mode.
"""

MD_LOCKFREE: MDConfigContext
"""
MDConfigContext instance to set flag for PyAlgoEngine to disable thread safe mode.
"""

MD_FREELIST: MDConfigContext
"""
MDConfigContext instance to set flag for PyAlgoEngine to use freelist.
"""

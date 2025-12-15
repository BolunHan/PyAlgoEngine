import ctypes
from dataclasses import dataclass
from typing import Annotated

from .c_market_data import MarketData


@dataclass
class ValueRange:
    lo: int
    hi: int


UINT32_MAX = ctypes.c_uint32(-1).value
uint32_t = Annotated[int, ValueRange(0, UINT32_MAX), ctypes.c_uint32]


class InternalData(MarketData):
    """
    Special market data class for internal communications.

    Used for heartbeats, triggers, and callback protocols.
    """

    def __init__(
            self,
            *,
            ticker: str,
            timestamp: float,
            code: uint32_t,
            **kwargs
    ) -> None:
        """
        Initialize internal data message.

        Args:
            ticker: Message identifier
            timestamp: Unix timestamp
            code: Protocol code to trigger
            **kwargs: Additional fields to set on the instance.__dict__
        """
        ...

    @property
    def code(self) -> uint32_t:
        """Get the protocol code this message triggers."""
        ...

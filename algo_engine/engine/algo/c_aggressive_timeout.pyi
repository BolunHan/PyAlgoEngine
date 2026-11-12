from typing import Any

from algo_engine.base import TransactionSide
from algo_engine.engine.c_dma_algo import AlgoLimitMode
from .c_passive import Passive


class AggressiveTimeout(Passive):
    """The aggressive algo that also cancels its order after a timeout."""

    def __init__(
            self,
            ticker: str,
            side: TransactionSide,
            target_volume: float,
            timeout: float = 0.0,
            limit_price: float = ...,
            limit_adjust_factor: float = ...,
            limit_mode: AlgoLimitMode = AlgoLimitMode.strict,
            **kwargs: Any
    ):
        """
        Allocate an aggressive algo with a timeout; it trades through the process DMA singleton.

        Args:
            ticker (str): The instrument.
            side (TransactionSide): The trade side.
            target_volume (float): The volume to trade.
            timeout (float): Seconds before cancelling; 0 disables.
            limit_price (float): The absolute limit price; NaN when unset.
            limit_adjust_factor (float): The market-price factor; NaN when unset.
            limit_mode (AlgoLimitMode): The limit combination; defaults to strict.
        """

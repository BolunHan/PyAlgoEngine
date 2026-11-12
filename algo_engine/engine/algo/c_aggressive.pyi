from typing import Any

from algo_engine.base import TransactionSide
from algo_engine.engine.c_dma_algo import AlgoLimitMode
from .c_passive import Passive


class Aggressive(Passive):
    """The passive algo that relaunches after a fill or cancel until the target is met."""

    def __init__(
            self,
            ticker: str,
            side: TransactionSide,
            target_volume: float,
            limit_price: float = ...,
            limit_adjust_factor: float = ...,
            limit_mode: AlgoLimitMode = AlgoLimitMode.strict,
            **kwargs: Any
    ):
        """
        Allocate an aggressive algo; it trades through the process DMA singleton.

        Args:
            ticker (str): The instrument.
            side (TransactionSide): The trade side.
            target_volume (float): The volume to trade.
            limit_price (float): The absolute limit price; NaN when unset.
            limit_adjust_factor (float): The market-price factor; NaN when unset.
            limit_mode (AlgoLimitMode): The limit combination; defaults to strict.
        """

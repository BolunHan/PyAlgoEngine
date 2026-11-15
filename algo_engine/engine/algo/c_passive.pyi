from typing import Any

from algo_engine.base import TransactionSide
from algo_engine.engine.c_dma_algo import AlgoLimitMode
from .c_algo_template import AlgoTemplate


class Passive(AlgoTemplate):
    """
    The passive algo: one limit order for the whole target volume.

    The order limit is the combination of the absolute limit, the market-price
    factor and the market price, picked by ``limit_mode``.
    """

    @property
    def limit_price(self) -> float | None:
        """The absolute limit; None when unset."""
        ...

    @limit_price.setter
    def limit_price(self, value: float) -> None: ...

    @property
    def limit_adjust_factor(self) -> float | None:
        """The market-price factor; None when unset."""
        ...

    @limit_adjust_factor.setter
    def limit_adjust_factor(self, value: float) -> None: ...

    limit_mode: AlgoLimitMode
    timeout: float
    relaunch: bool

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
        Allocate a passive algo; it trades through the process DMA singleton.

        Args:
            ticker (str): The instrument.
            side (TransactionSide): The trade side.
            target_volume (float): The volume to trade.
            limit_price (float): The absolute limit price; NaN when unset.
            limit_adjust_factor (float): The market-price factor; NaN when unset.
            limit_mode (AlgoLimitMode): The limit combination; defaults to strict.
        """

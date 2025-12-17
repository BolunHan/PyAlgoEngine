from collections.abc import Sequence, Iterator
from math import nan
from typing import Any, Union

import numpy as np

from .c_market_data import MarketData
from .c_transaction import TransactionDirection, TransactionSide

D_ARRAY = Union[np.ndarray, np.ndarray[tuple[int], np.dtype[np.double]], memoryview]
L_ARRAY = Union[np.ndarray, np.ndarray[tuple[int], np.dtype[np.uint64]], memoryview]
order_book_entry_t = tuple[float, float, int]


class TickDataLite(MarketData):
    """
    Lightweight market snapshot (Level-1 data) without full order book information.

    Contains basic market information including
    - Last traded price
    - Best bid/ask prices and volumes
    - Aggregated trade statistics
    """

    def __init__(
            self,
            *,
            ticker: str,
            timestamp: float,
            last_price: float,
            bid_price: float,
            bid_volume: float,
            ask_price: float,
            ask_volume: float,
            open_price: float = nan,
            prev_close: float = nan,
            total_traded_volume: float = 0.0,
            total_traded_notional: float = 0.0,
            total_trade_count: int = 0,
            **kwargs: Any
    ) -> None:
        """
        Initialize TickDataLite instance.

        Args:
            ticker: The ticker of the TickDataLite
            timestamp: Unix timestamp
            last_price: Last traded price
            bid_price: Best bid price
            bid_volume: Volume at best bid
            ask_price: Best ask price
            ask_volume: Volume at best ask
            open_price: The open price of this market session
            prev_close: The close price of last market session
            total_traded_volume: Cumulative traded volume
            total_traded_notional: Cumulative traded notional value
            total_trade_count: Cumulative number of trades
            **kwargs: Additional market data attributes
        """
        ...

    @property
    def last_price(self) -> float:
        """Get the last traded price."""
        ...

    @property
    def bid_price(self) -> float:
        """Get the current best bid price."""
        ...

    @property
    def bid_volume(self) -> float:
        """Get the volume available at the best bid."""
        ...

    @property
    def ask_price(self) -> float:
        """Get the current best ask price."""
        ...

    @property
    def ask_volume(self) -> float:
        """Get the volume available at the best ask."""
        ...

    @property
    def open_price(self) -> float:
        """Get the open price of this market session."""
        ...

    @property
    def prev_close(self) -> float:
        """Get the previous close price (close price of last market session)."""
        ...

    @property
    def total_traded_volume(self) -> float:
        """Get the cumulative traded volume."""
        ...

    @property
    def total_traded_notional(self) -> float:
        """Get the cumulative traded notional value."""
        ...

    @property
    def total_trade_count(self) -> int:
        """Get the total number of trades."""
        ...

    @property
    def mid_price(self) -> float:
        """Calculate the mid-price ((bid + ask) / 2)."""
        ...

    @property
    def spread(self) -> float:
        """Calculate the bid-ask spread (ask - bid)."""
        ...

    @property
    def market_price(self) -> float:
        """
        Get the representative market price.

        Defaults to last traded price if available, otherwise mid-price.
        """
        ...


class OrderBook:
    """
    Represents one side (bid or ask) of an order book.

    Note: Some exchanges may not provide order count information (n_orders).
    """

    def __init__(
            self,
            *,
            direction: TransactionDirection = TransactionDirection.DIRECTION_UNKNOWN,
            price: Sequence[float] = None,
            volume: Sequence[float] = None,
            n_orders: Sequence[int] = None,
            is_sorted: bool = False
    ) -> None:
        """
        Initialize OrderBook instance.

        Args:
            direction: Book side, DIRECTION_LONG for bid and DIRECTION_SHORT for ask. Default to DIRECTION_UNKNOWN, which will not initialize the underlying buffer.
            price: Sequence of price levels
            volume: Sequence of volumes at each price level
            n_orders: Sequence of order counts at each price level (optional)
            is_sorted: Whether the price levels are pre-sorted
        """
        ...

    def __iter__(self) -> Iterator[order_book_entry_t]:
        """Iterate through price levels as (price, volume, n_orders) tuples."""
        ...

    def __len__(self):
        """Get the number of initialized entries / levels in is order book."""
        ...

    def __next__(self) -> order_book_entry_t:
        ...

    def __getbuffer__(self):
        """Support for python buffer protocol."""
        ...

    def at_price(self, price: float) -> order_book_entry_t:
        """
        Get order book entry at a specific price level.

        Returns:
            Tuple of (price, volume, n_orders) if found

        Raises:
            IndexError: If the price level is not found
        """
        ...

    def at_level(self, index: int) -> order_book_entry_t:
        """
        Get order book entry at a specific depth level.

        Args:
            index: Depth level (0 = best price)

        Returns:
            Tuple of (price, volume, n_orders)

        Raises:
            IndexError: If the index is out of range
        """
        ...

    def loc_volume(self, p0: float, p1: float) -> float:
        """
        Calculate total volume between two price levels.

        Args:
            p0: Lower price bound, inclusive
            p1: Upper price bound, exclusive

        Returns:
            Total volume in price range
        """
        ...

    def sort(self) -> None:
        """Sort the order book by price (ascending for bids, descending for asks)."""
        ...

    def to_bytes(self) -> bytes:
        """Serialize the order book to bytes."""
        ...

    def to_numpy(self) -> np.ndarray:
        ...

    @property
    def price(self) -> D_ARRAY:
        """Get the actual underlying buffer of the price, backed by numpy array."""
        ...

    @property
    def volume(self) -> D_ARRAY:
        """Get the actual underlying buffer of the volume, backed by numpy array."""
        ...

    @property
    def n_orders(self) -> L_ARRAY:
        """Get the actual underlying buffer of the n_orders, backed by numpy array."""
        ...

    @property
    def sorted(self) -> bool:
        """Check if the order book is sorted."""
        ...

    @property
    def side(self) -> TransactionSide:
        """Get the side of the order book (Bid or Ask)."""
        ...

    @property
    def direction(self) -> TransactionDirection:
        """Get the direction of the order book (Long or Short)."""
        ...

    @property
    def size(self) -> int:
        """Get the number of initialized entries / levels in is order book."""
        ...

    @property
    def capacity(self) -> int:
        """Get the maximum capacity of the order book."""
        ...


class TickData(MarketData):
    """
    Complete market snapshot with order books (Level-2 data).

    Note: The order book depth is fixed at compile time (default 10 levels).
          Changing this requires recompilation and makes previously saved
          binary data (the pickle dumps and binary dumps of MarketDataBuffer) incompatible.
    """

    def __init__(
            self,
            *,
            ticker: str,
            timestamp: float,
            last_price: float,
            open_price: float = nan,
            prev_close: float = nan,
            total_traded_volume: float = 0.0,
            total_traded_notional: float = 0.0,
            total_trade_count: int = 0,
            total_bid_volume: float = 0.0,
            total_ask_volume: float = 0.0,
            weighted_bid_price: float = nan,
            weighted_ask_price: float = nan,
            **kwargs: Any
    ) -> None:
        """
        Initialize TickData instance.

        Args:
            ticker: The ticker of the TickData
            timestamp: Unix timestamp
            last_price: Last traded price
            total_traded_volume: Cumulative traded volume
            total_traded_notional: Cumulative traded notional value
            total_trade_count: Total number of trades
            total_bid_volume: Total queued volume of bid side
            total_ask_volume: Total queued volume of ask side
            weighted_bid_price: Weighted average bid price
            weighted_ask_price: Weighted average ask price
            **kwargs: Additional market data including order book details

        keyword args:
            bid_price_1, bid_volume_1, bid_n_orders_1, ...
            ask_price_1, ask_volume_1, ask_n_orders_1, ...
        """
        ...

    def parse(self, kwargs: dict[str, Any]) -> None:
        """Parse the order book info and additional market data from keyword arguments."""
        ...

    def lite(self) -> TickDataLite:
        """Get the lightweight TickDataLite representation."""
        ...

    @property
    def last_price(self) -> float:
        """Get the last traded price."""
        ...

    @property
    def bid_price(self) -> float:
        """Get the current best bid price."""
        ...

    @property
    def bid_volume(self) -> float:
        """Get the volume available at the best bid."""
        ...

    @property
    def ask_price(self) -> float:
        """Get the current best ask price."""
        ...

    @property
    def ask_volume(self) -> float:
        """Get the volume available at the best ask."""
        ...

    @property
    def open_price(self) -> float:
        """Get the open price of this market session."""
        ...

    @property
    def prev_close(self) -> float:
        """Get the previous close price (close price of last market session)."""
        ...

    @property
    def total_traded_volume(self) -> float:
        """Get the cumulative traded volume."""
        ...

    @property
    def total_traded_notional(self) -> float:
        """Get the cumulative traded notional value."""
        ...

    @property
    def total_trade_count(self) -> int:
        """Get the total number of trades."""
        ...

    @property
    def total_bid_volume(self) -> float:
        """
        Get the total queued volume of bid side.

        Note that this might differ from the sum of self.bid.volume.
        As the bid OrderBook is limited.
        """

    @property
    def total_ask_volume(self) -> float:
        """
        Get the total queued volume of ask side.

        Note that this might differ from the sum of self.ask.volume.
        As the bid OrderBook is limited.
        """

    @property
    def weighted_bid_price(self) -> float:
        """Get the weighted average bid price."""

    @property
    def weighted_ask_price(self) -> float:
        """Get the weighted average ask price."""

    @property
    def mid_price(self) -> float:
        """Calculate the mid-price ((bid + ask) / 2)."""
        ...

    @property
    def spread(self) -> float:
        """Calculate the bid-ask spread (ask - bid)."""
        ...

    @property
    def market_price(self) -> float:
        """
        Get the representative market price.

        Defaults to last traded price if available, otherwise mid-price.
        """
        ...

    @property
    def bid(self) -> OrderBook:
        """Get the bid-side order book."""
        ...

    @property
    def ask(self) -> OrderBook:
        """Get the ask-side order book."""
        ...

    @property
    def best_ask_price(self) -> float:
        """Get the current best ask price."""
        ...

    @property
    def best_bid_price(self) -> float:
        """Get the current best bid price."""
        ...

    @property
    def best_ask_volume(self) -> float:
        """Get the volume available at best ask."""
        ...

    @property
    def best_bid_volume(self) -> float:
        """Get the volume available at best bid."""
        ...

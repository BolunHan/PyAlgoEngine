from collections.abc import Sequence, Iterator
from typing import Any, Optional

from .c_market_data import MarketData


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
            ticker: str,
            timestamp: float,
            last_price: float,
            bid_price: float,
            bid_volume: float,
            ask_price: float,
            ask_volume: float,
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
    def prev_close(self) -> float:
        """Get the previous close price."""
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

    side: int
    sorted: bool

    def __init__(
            self,
            side: Optional[int] = None,
            price: Optional[Sequence[float]] = None,
            volume: Optional[Sequence[float]] = None,
            n_orders: Optional[Sequence[int]] = None,
            is_sorted: bool = False
    ) -> None:
        """
        Initialize OrderBook instance.

        Args:
            side: Book side (bid/ask)
            price: Sequence of price levels
            volume: Sequence of volumes at each price level
            n_orders: Sequence of order counts at each price level (optional)
            is_sorted: Whether the price levels are pre-sorted
        """
        ...

    def __iter__(self) -> Iterator[tuple[float, float, Optional[int]]]:
        """Iterate through price levels as (price, volume, n_orders) tuples."""
        ...

    def at_price(self, price: float) -> tuple[float, float, Optional[int]]:
        """
        Get order book entry at a specific price level.

        Returns:
            Tuple of (price, volume, n_orders) if found
        """
        ...

    def at_level(self, index: int) -> tuple[float, float, Optional[int]]:
        """
        Get order book entry at a specific depth level.

        Args:
            index: Depth level (0 = best price)

        Returns:
            Tuple of (price, volume, n_orders)
        """
        ...

    def loc_volume(self, p0: float, p1: float) -> float:
        """
        Calculate total volume between two price levels.

        Args:
            p0: Lower price bound
            p1: Upper price bound

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

    @property
    def price(self) -> list[float]:
        """Get a list of sorted price levels, based on the given side of the order book."""
        ...

    @property
    def volume(self) -> list[float]:
        """Get a list of volumes at each sorted price level."""
        ...

    @property
    def n_orders(self) -> list[Optional[int]]:
        """Get a list of order counts at each sorted price level (may contain None)."""
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
            ticker: str,
            timestamp: float,
            last_price: float,
            total_traded_volume: float = 0.0,
            total_traded_notional: float = 0.0,
            total_trade_count: int = 0,
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
            **kwargs: Additional market data including order book details
        """
        ...

    def parse(self, kwargs: dict[str, Any]) -> None:
        """Parse the order book info and additional market data from keyword arguments."""
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
    def prev_close(self) -> float:
        """Get the previous close price."""
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

    @property
    def lite(self) -> TickDataLite:
        """Convert to lightweight TickDataLite representation."""
        ...
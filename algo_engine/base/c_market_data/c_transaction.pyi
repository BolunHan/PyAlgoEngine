import enum
import uuid
from typing import Any, Literal

from typing_extensions import deprecated

from .c_market_data import MarketData


class OrderType(enum.IntEnum):
    """Enum representing different order types."""
    ORDER_UNKNOWN: OrderType
    ORDER_CANCEL: OrderType
    ORDER_GENERIC: OrderType
    ORDER_LIMIT: OrderType
    ORDER_LIMIT_MAKER: OrderType
    ORDER_MARKET: OrderType
    ORDER_FOK: OrderType
    ORDER_FAK: OrderType
    ORDER_IOC: OrderType


class TransactionDirection(enum.IntEnum):
    """
    Enum representing the direction of a financial transaction.

    Attributes:
        DIRECTION_UNKNOWN: Unknown transaction direction
        DIRECTION_SHORT: Sell party initiated transaction (selling)
        DIRECTION_LONG: Buy party initiated transaction (buying)
        DIRECTION_NEUTRAL: Neither party initiated transaction, commonly in auction session (neutral)
    """
    DIRECTION_UNKNOWN: TransactionDirection
    DIRECTION_SHORT: TransactionDirection
    DIRECTION_LONG: TransactionDirection
    DIRECTION_NEUTRAL: TransactionDirection

    def __or__(self, offset: TransactionOffset) -> TransactionSide:
        """
        Combine direction and offset to create a TransactionSide using | operator.

        Example:
            >>> TransactionDirection.DIRECTION_LONG | TransactionOffset.OFFSET_OPEN
            >>> # <TransactionSide.SIDE_LONG_OPEN: 10>
        """
        ...

    @property
    def sign(self) -> Literal[-1, 0, 1]:
        """
        Get the numerical sign representing the transaction direction.

        Returns:
            - 1 for long (buy)
            - -1 for short (sell)
            - 0 for cancel/unknown/neutral
        """
        ...


class TransactionOffset(enum.IntEnum):
    """
    Enum representing the offset type of a transaction.

    Attributes:
        OFFSET_CANCEL: Order cancellation
        OFFSET_ORDER: Regular order placement
        OFFSET_OPEN: Opening a new position
        OFFSET_CLOSE: Closing an existing position
    """
    OFFSET_CANCEL: TransactionOffset
    OFFSET_ORDER: TransactionOffset
    OFFSET_OPEN: TransactionOffset
    OFFSET_CLOSE: TransactionOffset

    def __or__(self, direction: TransactionDirection) -> TransactionSide:
        """
        Combine offset and direction to create a TransactionSide using | operator.

        Example:
            >>> TransactionOffset.OFFSET_OPEN | TransactionDirection.DIRECTION_LONG
            >>> # <TransactionSide.SIDE_LONG_OPEN: 10>
        """
        ...


class TransactionSide(enum.IntEnum):
    """
    Comprehensive enum representing all possible transaction sides.

    Combines direction and offset to describe complete transaction types.
    Includes deprecated legacy names for backward compatibility.
    """
    SIDE_LONG_OPEN: TransactionSide
    SIDE_LONG_CLOSE: TransactionSide
    SIDE_LONG_CANCEL: TransactionSide
    SIDE_SHORT_OPEN: TransactionSide
    SIDE_SHORT_CLOSE: TransactionSide
    SIDE_NEUTRAL_OPEN: TransactionSide
    SIDE_NEUTRAL_CLOSE: TransactionSide
    SIDE_SHORT_CANCEL: TransactionSide
    SIDE_BID: TransactionSide
    SIDE_ASK: TransactionSide
    SIDE_CANCEL: TransactionSide
    SIDE_UNKNOWN: TransactionSide
    SIDE_LONG: TransactionSide
    SIDE_SHORT: TransactionSide
    SIDE_FAULTY: TransactionSide

    # Deprecated aliases
    ShortOrder: deprecated('Use SIDE_ASK instead')(TransactionSide)
    AskOrder: deprecated('Use SIDE_ASK instead')(TransactionSide)
    Ask: deprecated('Use SIDE_ASK instead')(TransactionSide)
    LongOrder: deprecated('Use SIDE_BID instead')(TransactionSide)
    BidOrder: deprecated('Use SIDE_BID instead')(TransactionSide)
    Bid: deprecated('Use SIDE_BID instead')(TransactionSide)
    ShortFilled: deprecated('Use SIDE_SHORT instead')(TransactionSide)
    Unwind: deprecated('Use SIDE_SHORT_CLOSE instead')(TransactionSide)
    Sell: deprecated('Use SIDE_SHORT_CLOSE instead')(TransactionSide)
    LongFilled: deprecated('Use SIDE_LONG_OPEN instead')(TransactionSide)
    LongOpen: deprecated('Use SIDE_LONG_OPEN instead')(TransactionSide)
    Buy: deprecated('Use SIDE_LONG_OPEN instead')(TransactionSide)
    ShortOpen: deprecated('Use SIDE_SHORT_OPEN instead')(TransactionSide)
    Short: deprecated('Use SIDE_SHORT_OPEN instead')(TransactionSide)
    Cover: deprecated('Use SIDE_LONG_CLOSE instead')(TransactionSide)
    UNKNOWN: deprecated('Use SIDE_UNKNOWN instead')(TransactionSide)
    CANCEL: deprecated('Use SIDE_CANCEL instead')(TransactionSide)

    @property
    def sign(self) -> Literal[-1, 0, 1]:
        """
        Get the numerical sign representing the transaction direction.

        Returns:
            - 1 for long (buy)
            - -1 for short (sell)
            - 0 for cancel/unknown
        """
        ...

    @property
    def offset(self) -> TransactionOffset:
        """Get the transaction offset component of this side."""
        ...

    @property
    def direction(self) -> TransactionDirection:
        """Get the transaction direction component of this side."""
        ...

    @property
    def side_name(self) -> str:
        """Get the human-readable name of this transaction side."""
        ...

    @property
    def offset_name(self) -> str:
        """Get the human-readable name of the offset component."""
        ...

    @property
    def direction_name(self) -> str:
        """Get the human-readable name of the direction component."""
        ...


class TransactionData(MarketData):
    """
    Market data representing a completed transaction or order cancellation.

    Represents any action that reduces listing volume from the order book,
    including trades and cancellations. The side indicates which party initiated
    the action. The cancel order has no side information (as it only has 1 party).

    Multiplier should be provided for instruments like future contracts where
    the contract multiplier is not 1. This affects notional and flow calculations.
    """

    def __init__(
            self,
            ticker: str,
            timestamp: float,
            price: float,
            volume: float,
            side: int,
            multiplier: float = 1.0,
            notional: float = ...,
            transaction_id: str | int | bytes | uuid.UUID | None = None,
            buy_id: str | int | bytes | uuid.UUID | None = None,
            sell_id: str | int | bytes | uuid.UUID | None = None,
            **kwargs: Any
    ) -> None:
        """
        Initialize a TransactionData instance.

        Args:
            ticker: The Ticker of the transaction
            timestamp: Unix timestamp of transaction
            price: Execution price
            volume: Transaction volume
            side: Transaction side (see TransactionSide)
            multiplier: Contract multiplier (default 1.0)
            notional: Optional pre-calculated notional value
            transaction_id: Unique transaction identifier
            buy_id: Buyer's order identifier
            sell_id: Seller's order identifier
            **kwargs: Additional transaction attributes
        """
        ...

    @classmethod
    def from_bytes(cls, data: bytes) -> TransactionData:
        """Reconstruct from serialized byte data."""
        ...

    @classmethod
    def merge(cls, data_list: list[TransactionData]) -> TransactionData:
        """
        Merge multiple transactions into a single aggregate transaction.

        Args:
            data_list: List of transactions to merge

        Returns:
            New combined TransactionData instance
        """
        ...

    def __copy__(self) -> TransactionData:
        """Create a deep copy of this transaction."""
        ...

    @property
    def price(self) -> float:
        """Get the execution price of this transaction."""
        ...

    @property
    def volume(self) -> float:
        """Get the volume/size of this transaction."""
        ...

    @property
    def side_int(self) -> int:
        """Get the raw int enum value of the transaction side."""
        ...

    @property
    def side_sign(self) -> Literal[-1, 0, 1]:
        """Get the directional sign of this transaction (-1, 0, 1). See TransactionDirection.side for details."""
        ...

    @property
    def side(self) -> TransactionSide:
        """Get the TransactionSide enum value for this transaction."""
        ...

    @property
    def multiplier(self) -> float:
        """Get the contract multiplier for this transaction."""
        ...

    @property
    def notional(self) -> float:
        """Get the notional value (price * volume * multiplier)."""
        ...

    @property
    def transaction_id(self) -> str | int | bytes | uuid.UUID | None:
        """Get the unique identifier for this transaction."""
        ...

    @property
    def buy_id(self) -> str | int | bytes | uuid.UUID | None:
        """Get the buyer's order identifier."""
        ...

    @property
    def sell_id(self) -> str | int | bytes | uuid.UUID | None:
        """Get the seller's order identifier."""
        ...

    @property
    def volume_flow(self) -> float:
        """Get the signed volume flow (volume * side_sign)."""
        ...

    @property
    def notional_flow(self) -> float:
        """Get the signed notional flow (notional * side_sign)."""
        ...


class OrderData(MarketData):
    """
    Market data representing an order (bid/ask) in the order book.

    Represents any action that increases listing volume to the order book.
    The side indicates the side of the order book it affected.
    """

    def __init__(
            self,
            ticker: str,
            timestamp: float,
            price: float,
            volume: float,
            side: int,
            order_id: str | int | bytes | uuid.UUID | None = None,
            order_type: int = ...,
            **kwargs: Any
    ) -> None:
        """
        Initialize an OrderData instance.

        Args:
            ticker: The Ticker of the order
            timestamp: Unix timestamp
            price: Order price
            volume: Order volume
            side: Order side (see TransactionSide)
            order_id: Unique order identifier
            order_type: Order type (see OrderType from c_market_data)
            **kwargs: Additional order attributes
        """
        ...

    @property
    def price(self) -> float:
        """Get the order price."""
        ...

    @property
    def volume(self) -> float:
        """Get the order volume."""
        ...

    @property
    def side_int(self) -> int:
        """Get the raw integer value of the order side."""
        ...

    @property
    def side_sign(self) -> Literal[-1, 0, 1]:
        """Get the directional sign of this order (-1, 0, 1)."""
        ...

    @property
    def side(self) -> TransactionSide:
        """Get the TransactionSide enum value for this order."""
        ...

    @property
    def order_id(self) -> str | int | bytes | uuid.UUID | None:
        """Get the unique order identifier."""
        ...

    @property
    def order_type_int(self) -> int:
        """Get the raw integer value of the order type."""
        ...

    @property
    def order_type(self) -> OrderType:
        """Get the OrderType enum value for this order."""
        ...

    @property
    def market_price(self) -> float:
        """Get the current market price relevant to this order."""
        ...

    @property
    def flow(self) -> float:
        """Get the signed order flow (volume * side_sign)."""
        ...


class TradeData(TransactionData):
    """
    An alias for TransactionData for backwards compatibility.

    Inherits all TransactionData functionality while providing
    trade-specific property aliases.
    """

    def __init__(
            self,
            ticker: str,
            timestamp: float,
            trade_price: float,
            trade_volume: float,
            side: int,
            order_id: str | int | bytes | uuid.UUID | None = None,
            order_type: int = ...,
            **kwargs: Any
    ) -> None:
        """
        Initialize a TradeData instance.

        Args:
            ticker: Instrument identifier
            timestamp: Unix timestamp of trade
            trade_price: Execution price
            trade_volume: Trade volume
            side: Trade side (see TransactionSide)
            order_id: Related order identifier
            order_type: Order type (see OrderType from c_market_data for details)
            **kwargs: Additional trade attributes
        """
        ...

    @property
    def trade_price(self) -> float:
        """Alias for price property (trade execution price)."""
        ...

    @property
    def trade_volume(self) -> float:
        """Alias for volume property (trade volume)."""
        ...

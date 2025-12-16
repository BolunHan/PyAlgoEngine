import enum
import uuid
from math import nan
from typing import Literal
from warnings import deprecated

from .c_market_data import MarketData

mid_t = str | int | bytes | uuid.UUID | None
sign_t = Literal[-1, 0, 1]


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
            <TransactionSide.SIDE_LONG_OPEN: 10>

        """
        ...

    @property
    def sign(self) -> sign_t:
        """
        Get the numerical sign representing the transaction direction.

        Returns:
            - 1 for long (buy)
            - -1 for short (sell)
            - 0 for cancel / unknown / neutral
        """
        ...


class TransactionOffset(enum.IntEnum):
    """
    Enum representing the offset type of transaction.

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
            <TransactionSide.SIDE_LONG_OPEN: 10>

        """
        ...


@deprecated('Use entries from side_t instead')
class TransactionSideDeprecated(TransactionSide):
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
    ShortOrder: TransactionSideDeprecated
    AskOrder: TransactionSideDeprecated
    Ask: TransactionSideDeprecated
    LongOrder: TransactionSideDeprecated
    BidOrder: TransactionSideDeprecated
    Bid: TransactionSideDeprecated
    ShortFilled: TransactionSideDeprecated
    Unwind: TransactionSideDeprecated
    Sell: TransactionSideDeprecated
    LongFilled: TransactionSideDeprecated
    LongOpen: TransactionSideDeprecated
    Buy: TransactionSideDeprecated
    ShortOpen: TransactionSideDeprecated
    Short: TransactionSideDeprecated
    Cover: TransactionSideDeprecated
    UNKNOWN: TransactionSideDeprecated
    CANCEL: TransactionSideDeprecated

    @property
    def sign(self) -> sign_t:
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
            *
            ticker: str,
            timestamp: float,
            price: float,
            volume: float,
            side: int,
            multiplier: float = 1.0,
            notional: float = nan,
            transaction_id: mid_t = None,
            buy_id: mid_t = None,
            sell_id: mid_t = None,
            **kwargs
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
            notional: Optional pre-calculated notional value, if nan, will be computed as price * volume * multiplier
            transaction_id: Unique transaction identifier
            buy_id: Buyer's order identifier
            sell_id: Seller's order identifier
            **kwargs: Additional transaction attributes
        """
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

    @property
    def price(self) -> float:
        """Get the execution price of this transaction."""
        ...

    @property
    def volume(self) -> float:
        """Get the volume/size of this transaction."""
        ...

    @property
    def side(self) -> TransactionSide:
        """Get the TransactionSide enum value for this transaction."""
        ...

    @property
    def side_int(self) -> int:
        """Get the raw int enum value of the transaction side."""
        ...

    @property
    def side_sign(self) -> sign_t:
        """Get the directional sign of this transaction (-1, 0, 1). See TransactionDirection.side for details."""
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
    def transaction_id(self) -> mid_t:
        """Get the unique identifier for this transaction."""
        ...

    @property
    def buy_id(self) -> mid_t:
        """Get the buyer's order identifier."""
        ...

    @property
    def sell_id(self) -> mid_t:
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
    Market data representing an order (bid / ask) in the order book.

    Represents any action that increases listing volume to the order book.
    The side indicates the side of the order book it affected.
    """

    def __init__(
            self,
            *
            ticker: str,
            timestamp: float,
            price: float,
            volume: float,
            side: int,
            order_id: mid_t = None,
            order_type: int = OrderType.ORDER_GENERIC,
            **kwargs
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
    def side(self) -> TransactionSide:
        """Get the TransactionSide enum value for this order."""
        ...

    @property
    def side_int(self) -> int:
        """Get the raw integer value of the order side."""
        ...

    @property
    def side_sign(self) -> sign_t:
        """Get the directional sign of this order (-1, 0, 1)."""
        ...

    @property
    def order_id(self) -> mid_t:
        """Get the unique order identifier."""
        ...

    @property
    def order_type(self) -> OrderType:
        """Get the OrderType enum value for this order."""
        ...

    @property
    def order_type_int(self) -> int:
        """Get the raw integer value of the order type."""
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
            *,
            ticker: str,
            timestamp: float,
            trade_price: float,
            trade_volume: float,
            trade_side: int,
            multiplier: float = 1.0,
            notional: float = nan,
            transaction_id: mid_t = None,
            buy_id: mid_t = None,
            sell_id: mid_t = None,
            **kwargs
    ) -> None:
        """
        Initialize a TradeData instance.

        Args:
            ticker: Instrument identifier
            timestamp: Unix timestamp of trade
            trade_price: Execution price
            trade_volume: Trade volume
            trade_side: Trade side (see TransactionSide)
            multiplier: Contract multiplier (default 1.0)
            notional: Optional pre-calculated notional value, if nan, will be computed as price * volume * multiplier
            transaction_id: Unique transaction identifier
            buy_id: Buyer's order identifier
            sell_id: Seller's order identifier
            **kwargs: Additional transaction attributes
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

    @property
    def trade_side(self) -> TransactionSide:
        """Alias for side property (trade side)."""
        ...

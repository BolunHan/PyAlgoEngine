import enum
import uuid
from datetime import datetime
from typing import Any, Literal, Optional, Union

from typing_extensions import deprecated

from .c_market_data import OrderType
from .c_transaction import TransactionSide, TransactionData


class OrderState(enum.IntEnum):
    """
    Enum representing the various states an order can be in during its lifecycle.
    """
    STATE_UNKNOWN: int = ...
    STATE_REJECTED: int = ...
    STATE_INVALID: int = ...
    STATE_PENDING: int = ...
    STATE_SENT: int = ...
    STATE_PLACED: int = ...
    STATE_PARTFILLED: int = ...
    STATE_FILLED: int = ...
    STATE_CANCELING: int = ...
    STATE_CANCELED: int = ...

    # Deprecated aliases
    UNKNOWN: deprecated('Use STATE_UNKNOWN instead')(int) = ...
    Rejected: deprecated('Use STATE_REJECTED instead')(int) = ...
    Invalid: deprecated('Use STATE_INVALID instead')(int) = ...
    Pending: deprecated('Use STATE_PENDING instead')(int) = ...
    Sent: deprecated('Use STATE_SENT instead')(int) = ...
    Placed: deprecated('Use STATE_PLACED instead')(int) = ...
    PartFilled: deprecated('Use STATE_PARTFILLED instead')(int) = ...
    Filled: deprecated('Use STATE_FILLED instead')(int) = ...
    Canceling: deprecated('Use STATE_CANCELING instead')(int) = ...
    Canceled: deprecated('Use STATE_CANCELED instead')(int) = ...

    def __hash__(self) -> int:
        """Enable usage as a dictionary key."""
        ...

    @property
    def is_working(self) -> bool:
        """
        Check if order is in a working state.

        Returns:
            True for SENT, PLACED, PARTFILLED, or CANCELING states
        """
        ...

    @property
    def is_done(self) -> bool:
        """
        Check if order is in a terminal state.

        Returns:
            True for FILLED, CANCELED, REJECTED, or INVALID states
        """
        ...

    @property
    def state_name(self) -> str:
        """Get human-readable name of the order state."""
        ...


class TradeReport:
    """
    Represents an execution report from the exchange for a filled trade.

    Contains details about price, volume, fees, and identifiers for both
    the trade and the original order.
    """

    def __init__(
            self,
            ticker: str,
            timestamp: float,
            price: float,
            volume: float,
            side: int,
            notional: float = 0.0,
            multiplier: float = 1.0,
            fee: float = 0.0,
            order_id: Union[str, int, bytes, uuid.UUID, None] = None,
            trade_id: Union[str, int, bytes, uuid.UUID, None] = None,
            **kwargs: Any
    ) -> None:
        """
        Initialize a trade report.

        Args:
            ticker: The ticker of the trade report
            timestamp: Unix timestamp of trade
            price: Execution price
            volume: Traded volume
            side: Trade side (see TransactionSide)
            notional: Optional pre-calculated notional value
            multiplier: Contract multiplier (default 1.0)
            fee: Transaction fee/cost
            order_id: Original order identifier
            trade_id: Unique trade identifier
            **kwargs: Additional trade attributes
        """
        ...

    def __eq__(self, other: TradeReport) -> bool:
        """Compare trade reports for equality. By trade_id and order_id."""
        ...

    def __repr__(self) -> str:
        """Get string representation of the trade report."""
        ...

    def __reduce__(self) -> tuple[Any, ...]:
        """Support for pickle protocol."""
        ...

    def __setstate__(self, state: Any) -> None:
        """Support for pickle protocol."""
        ...

    def __copy__(self) -> TradeReport:
        """Create a deep copy of the trade report."""
        ...

    def reset_order_id(
            self,
            order_id: Union[str, int, bytes, uuid.UUID, None] = None
    ) -> TradeReport:
        """
        Update the order ID reference and return self.

        Args:
            order_id: New order identifier

        Returns:
            self (for method chaining)
        """
        ...

    def reset_trade_id(
            self,
            trade_id: Union[str, int, bytes, uuid.UUID, None] = None
    ) -> TradeReport:
        """
        Update the trade ID and return self.

        Args:
            trade_id: New trade identifier

        Returns:
            self (for method chaining)
        """
        ...

    @classmethod
    def from_bytes(cls, data: bytes) -> TradeReport:
        """Deserialize from byte data."""
        ...

    def copy(self) -> TradeReport:
        """Create a deep copy of the trade report."""
        ...

    def to_trade(self) -> TransactionData:
        """Convert to TransactionData format."""
        ...

    @property
    def price(self) -> float:
        """Get execution price."""
        ...

    @property
    def volume(self) -> float:
        """Get traded volume."""
        ...

    @property
    def side_int(self) -> int:
        """Get raw integer value of trade side."""
        ...

    @property
    def side_sign(self) -> Literal[-1, 0, 1]:
        """Get directional sign of trade (-1, 0, 1)."""
        ...

    @property
    def side(self) -> TransactionSide:
        """Get TransactionSide enum value."""
        ...

    @property
    def multiplier(self) -> float:
        """Get contract multiplier."""
        ...

    @property
    def notional(self) -> float:
        """Get notional value (price * volume * multiplier)."""
        ...

    @property
    def fee(self) -> float:
        """Get transaction fee/cost."""
        ...

    @property
    def trade_id(self) -> Union[str, int, bytes, uuid.UUID, None]:
        """Get unique trade identifier."""
        ...

    @property
    def order_id(self) -> Union[str, int, bytes, uuid.UUID, None]:
        """Get the original order identifier."""
        ...

    @property
    def market_price(self) -> float:
        """Same as price."""
        ...

    @property
    def volume_flow(self) -> float:
        """Get signed volume flow (volume * side_sign)."""
        ...

    @property
    def notional_flow(self) -> float:
        """Get signed notional flow (notional * side_sign)."""
        ...

    @property
    def trade_time(self) -> datetime:
        """Get timestamp as datetime object."""
        ...


class TradeInstruction:
    """
    Represents a trading instruction/order and its execution state.

    Manages order lifecycle automatically through fill, cancel_order,
    and canceled methods. Tracks all executions in the "trades" dictionary.
    """

    trades: dict[Union[str, int, bytes, uuid.UUID], TradeReport]

    def __init__(
            self,
            ticker: str,
            timestamp: float,
            side: int,
            volume: float,
            order_type: int = ...,
            limit_price: float = ...,
            multiplier: float = 1.0,
            order_id: Union[str, int, bytes, uuid.UUID, None] = None,
            **kwargs: Any
    ) -> None:
        """
        Initialize a trade instruction.

        Args:
            ticker: The ticker of the trade instruction
            timestamp: Creation timestamp
            side: Order side (see TransactionSide)
            volume: Total order volume
            order_type: Order type (see OrderType)
            limit_price: Limit price (for limit orders)
            multiplier: Contract multiplier (default 1.0)
            order_id: Optional order identifier
            **kwargs: Additional order attributes
        """
        ...

    def __eq__(self, other: TradeInstruction) -> bool:
        """Compare trade instructions for equality. By order_id."""
        ...

    def __repr__(self) -> str:
        """Get string representation of the instruction."""
        ...

    @classmethod
    def from_bytes(cls, data: bytes) -> TradeInstruction:
        """Deserialize from byte data."""
        ...

    def __copy__(self) -> TradeInstruction:
        """Create a shallow copy of the instruction."""
        ...

    def reset(self) -> TradeInstruction:
        """Reset instruction to initial state and return self."""
        ...

    def reset_order_id(
            self,
            order_id: Union[str, int, bytes, uuid.UUID, None] = None
    ) -> TradeInstruction:
        """
        Update the order ID and return self.

        Args:
            order_id: New order identifier

        Returns:
            self (for method chaining)
        """
        ...

    def set_order_state(
            self,
            order_state: int,
            timestamp: float = ...
    ) -> TradeInstruction:
        """Update order state and return self."""
        ...

    def fill(self, trade_report: TradeReport) -> TradeInstruction:
        """Add a trade report to instruction and return self."""
        ...

    def add_trade(self, trade_report: TradeReport) -> TradeInstruction:
        """Add a trade report and return self. This method skips all safety checks."""
        ...

    def cancel_order(self, timestamp: float = ...) -> TradeInstruction:
        """
        Transition to CANCELING state and return self.

        Args:
            timestamp: Cancellation timestamp (defaults to now)
        """
        ...

    def canceled(self, timestamp: float = ...) -> TradeInstruction:
        """
        Transition to CANCELED state and return self.

        Args:
            timestamp: Cancellation timestamp (defaults to now)
        """
        ...

    @property
    def is_working(self) -> bool:
        """
        Check if order is active/working.

        Returns:
            True for SENT, PLACED, PARTFILLED or CANCELING states
        """
        ...

    @property
    def is_done(self) -> bool:
        """
        Check if order is in terminal state.

        Returns:
            True for FILLED, CANCELED, REJECTED or INVALID states
        """
        ...

    @property
    def limit_price(self) -> float:
        """Get limit price (for limit orders)."""
        ...

    @property
    def volume(self) -> float:
        """Get total order volume."""
        ...

    @property
    def side_int(self) -> int:
        """Get raw integer value of order side."""
        ...

    @property
    def side_sign(self) -> Literal[-1, 0, 1]:
        """Get directional sign of order (-1, 0, 1)."""
        ...

    @property
    def side(self) -> TransactionSide:
        """Get TransactionSide enum value."""
        ...

    @property
    def order_type_int(self) -> int:
        """Get raw integer value of the order type."""
        ...

    @property
    def order_type(self) -> OrderType:
        """Get OrderType enum value."""
        ...

    @property
    def order_state_int(self) -> int:
        """Get raw integer value of order state."""
        ...

    @property
    def order_state(self) -> OrderState:
        """Get OrderState enum value."""
        ...

    @property
    def multiplier(self) -> float:
        """Get contract multiplier."""
        ...

    @property
    def filled_volume(self) -> float:
        """Get total filled volume across all executions."""
        ...

    @property
    def working_volume(self) -> float:
        """Get the remaining working volume (volume - filled)."""
        ...

    @property
    def filled_notional(self) -> float:
        """Get total notional value of fills."""
        ...

    @property
    def fee(self) -> float:
        """Get cumulative fees across all executions."""
        ...

    @property
    def order_id(self) -> Union[str, int, bytes, uuid.UUID, None]:
        """Get order identifier."""
        ...

    @property
    def average_price(self) -> float:
        """Calculate volume-weighted average fill price."""
        ...

    @property
    def start_time(self) -> datetime:
        """Get creation time as datetime."""
        ...

    @property
    def placed_time(self) -> Optional[datetime]:
        """Get order placement time (if placed)."""
        ...

    @property
    def canceled_time(self) -> Optional[datetime]:
        """Get cancellation time (if canceled)."""
        ...

    @property
    def finished_time(self) -> Optional[datetime]:
        """Get completion time (if done)."""
        ...
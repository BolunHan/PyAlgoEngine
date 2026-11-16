import enum

from algo_engine.base import TransactionSide


class BalanceFlag(enum.IntFlag):
    """The supplied balance fields; the values mirror the C flag bits."""

    balance: BalanceFlag
    equity: BalanceFlag
    net_asset_value: BalanceFlag
    total_asset: BalanceFlag
    total_liability: BalanceFlag
    unrealized_pnl: BalanceFlag
    realized_pnl: BalanceFlag
    cash_balance: BalanceFlag
    available_cash: BalanceFlag
    frozen_cash: BalanceFlag
    deposit: BalanceFlag
    withdrawal: BalanceFlag
    cash_in: BalanceFlag
    cash_out: BalanceFlag
    margin: BalanceFlag
    initial_margin: BalanceFlag
    maintenance_margin: BalanceFlag
    available_margin: BalanceFlag
    frozen_margin: BalanceFlag
    collateral: BalanceFlag
    available_collateral: BalanceFlag
    borrowed: BalanceFlag
    interest: BalanceFlag
    buying_power: BalanceFlag
    leverage: BalanceFlag


class Balance:
    """
    The account balance a host supplies to the access.

    Setting a field marks it supplied; the access's gate checks only the
    available field the trade selects, cash or margin. A balance the host never
    fills (or never supplies at all) leaves the gate passing.
    """

    flags: BalanceFlag
    """The supplied fields."""

    balance: float
    equity: float
    net_asset_value: float
    total_asset: float
    total_liability: float
    unrealized_pnl: float
    realized_pnl: float
    cash_balance: float
    available_cash: float
    frozen_cash: float
    deposit: float
    withdrawal: float
    cash_in: float
    cash_out: float
    margin: float
    initial_margin: float
    maintenance_margin: float
    available_margin: float
    frozen_margin: float
    collateral: float
    available_collateral: float
    borrowed: float
    interest: float
    buying_power: float
    leverage: float

    def __init__(self):
        """Allocate an empty balance."""

    def clear(self) -> None:
        """Drop every supplied field."""

    def check(self, side: TransactionSide, volume: float, price: float, multiplier: float = 1., margin_rate: float = float('nan')) -> bool:
        """
        Tell whether an order is affordable at the supplied balance.

        Args:
            side: The trade side; a cash sale raises no requirement.
            volume: The order volume.
            price: The per-unit price.
            multiplier: The contract multiplier.
            margin_rate: The designated margin rate; NaN checks the buy's
                notional against available cash, any finite value checks the
                notional times the rate against available margin in either
                direction.

        Returns:
            False only when the order exceeds the selected available field; an
            unsupplied field or a cash sale passes.
        """

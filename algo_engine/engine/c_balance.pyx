import enum

from algo_engine.base.c_allocator_protocol cimport MD_DEFAULT_ALLOCATOR


class BalanceFlag(enum.IntFlag):
    balance = dma_balance_flag.DMA_BALANCE_F_BALANCE
    equity = dma_balance_flag.DMA_BALANCE_F_EQUITY
    net_asset_value = dma_balance_flag.DMA_BALANCE_F_NET_ASSET_VALUE
    total_asset = dma_balance_flag.DMA_BALANCE_F_TOTAL_ASSET
    total_liability = dma_balance_flag.DMA_BALANCE_F_TOTAL_LIABILITY
    unrealized_pnl = dma_balance_flag.DMA_BALANCE_F_UNREALIZED_PNL
    realized_pnl = dma_balance_flag.DMA_BALANCE_F_REALIZED_PNL
    cash_balance = dma_balance_flag.DMA_BALANCE_F_CASH_BALANCE
    available_cash = dma_balance_flag.DMA_BALANCE_F_AVAILABLE_CASH
    frozen_cash = dma_balance_flag.DMA_BALANCE_F_FROZEN_CASH
    deposit = dma_balance_flag.DMA_BALANCE_F_DEPOSIT
    withdrawal = dma_balance_flag.DMA_BALANCE_F_WITHDRAWAL
    cash_in = dma_balance_flag.DMA_BALANCE_F_CASH_IN
    cash_out = dma_balance_flag.DMA_BALANCE_F_CASH_OUT
    margin = dma_balance_flag.DMA_BALANCE_F_MARGIN
    initial_margin = dma_balance_flag.DMA_BALANCE_F_INITIAL_MARGIN
    maintenance_margin = dma_balance_flag.DMA_BALANCE_F_MAINTENANCE_MARGIN
    available_margin = dma_balance_flag.DMA_BALANCE_F_AVAILABLE_MARGIN
    frozen_margin = dma_balance_flag.DMA_BALANCE_F_FROZEN_MARGIN
    collateral = dma_balance_flag.DMA_BALANCE_F_COLLATERAL
    available_collateral = dma_balance_flag.DMA_BALANCE_F_AVAILABLE_COLLATERAL
    borrowed = dma_balance_flag.DMA_BALANCE_F_BORROWED
    interest = dma_balance_flag.DMA_BALANCE_F_INTEREST
    buying_power = dma_balance_flag.DMA_BALANCE_F_BUYING_POWER
    leverage = dma_balance_flag.DMA_BALANCE_F_LEVERAGE


cdef class Balance(CCPType):
    def __init__(self):
        self.header = c_balance_new(MD_DEFAULT_ALLOCATOR)
        if not self.header:
            raise MemoryError(f'Failed to allocate memory for <{self.__class__.__name__}>.')

        self.owner = True
        self.ccp_bind(&self.header)

    def __dealloc__(self):
        if not self.owner:
            return

        if self.header:
            c_balance_free(self.header)
            self.header = NULL

    @staticmethod
    cdef Balance c_from_header(dma_balance* header, bint owner=False):
        cdef Balance instance = Balance.__new__(Balance)

        if not header:
            raise MemoryError('Failed to adopt an empty <Balance>.')

        instance.header = header
        instance.owner = owner
        instance.ccp_bind(&instance.header)
        return instance

    def __repr__(self):
        if not self.header:
            return f'<{self.__class__.__name__}: unbound>'
        return f'<{self.__class__.__name__}>(id={id(self)})'

    # === Python Interfaces ===

    cpdef void clear(self):
        if not self.header:
            raise RuntimeError(f'Uninitialized <{self.__class__.__name__}>')
        c_balance_clear(self.header)

    cpdef bint check(self, md_side side, double volume, double price, double multiplier=1., double margin_rate=1.):
        if not self.header:
            raise RuntimeError(f'Uninitialized <{self.__class__.__name__}>')
        return c_balance_check(self.header, side, volume, price, multiplier, margin_rate)

    # === Python Properties ===

    property flags:
        def __get__(self):
            if not self.header:
                raise RuntimeError(f'Uninitialized <{self.__class__.__name__}>')
            return self.header.flags

    property balance:
        def __get__(self):
            if not self.header:
                raise RuntimeError(f'Uninitialized <{self.__class__.__name__}>')
            return self.header.balance

        def __set__(self, double value):
            if not self.header:
                raise RuntimeError(f'Uninitialized <{self.__class__.__name__}>')
            self.header.balance = value
            self.header.flags |= dma_balance_flag.DMA_BALANCE_F_BALANCE

    property equity:
        def __get__(self):
            if not self.header:
                raise RuntimeError(f'Uninitialized <{self.__class__.__name__}>')
            return self.header.equity

        def __set__(self, double value):
            if not self.header:
                raise RuntimeError(f'Uninitialized <{self.__class__.__name__}>')
            self.header.equity = value
            self.header.flags |= dma_balance_flag.DMA_BALANCE_F_EQUITY

    property net_asset_value:
        def __get__(self):
            if not self.header:
                raise RuntimeError(f'Uninitialized <{self.__class__.__name__}>')
            return self.header.net_asset_value

        def __set__(self, double value):
            if not self.header:
                raise RuntimeError(f'Uninitialized <{self.__class__.__name__}>')
            self.header.net_asset_value = value
            self.header.flags |= dma_balance_flag.DMA_BALANCE_F_NET_ASSET_VALUE

    property total_asset:
        def __get__(self):
            if not self.header:
                raise RuntimeError(f'Uninitialized <{self.__class__.__name__}>')
            return self.header.total_asset

        def __set__(self, double value):
            if not self.header:
                raise RuntimeError(f'Uninitialized <{self.__class__.__name__}>')
            self.header.total_asset = value
            self.header.flags |= dma_balance_flag.DMA_BALANCE_F_TOTAL_ASSET

    property total_liability:
        def __get__(self):
            if not self.header:
                raise RuntimeError(f'Uninitialized <{self.__class__.__name__}>')
            return self.header.total_liability

        def __set__(self, double value):
            if not self.header:
                raise RuntimeError(f'Uninitialized <{self.__class__.__name__}>')
            self.header.total_liability = value
            self.header.flags |= dma_balance_flag.DMA_BALANCE_F_TOTAL_LIABILITY

    property unrealized_pnl:
        def __get__(self):
            if not self.header:
                raise RuntimeError(f'Uninitialized <{self.__class__.__name__}>')
            return self.header.unrealized_pnl

        def __set__(self, double value):
            if not self.header:
                raise RuntimeError(f'Uninitialized <{self.__class__.__name__}>')
            self.header.unrealized_pnl = value
            self.header.flags |= dma_balance_flag.DMA_BALANCE_F_UNREALIZED_PNL

    property realized_pnl:
        def __get__(self):
            if not self.header:
                raise RuntimeError(f'Uninitialized <{self.__class__.__name__}>')
            return self.header.realized_pnl

        def __set__(self, double value):
            if not self.header:
                raise RuntimeError(f'Uninitialized <{self.__class__.__name__}>')
            self.header.realized_pnl = value
            self.header.flags |= dma_balance_flag.DMA_BALANCE_F_REALIZED_PNL

    property cash_balance:
        def __get__(self):
            if not self.header:
                raise RuntimeError(f'Uninitialized <{self.__class__.__name__}>')
            return self.header.cash_balance

        def __set__(self, double value):
            if not self.header:
                raise RuntimeError(f'Uninitialized <{self.__class__.__name__}>')
            self.header.cash_balance = value
            self.header.flags |= dma_balance_flag.DMA_BALANCE_F_CASH_BALANCE

    property available_cash:
        def __get__(self):
            if not self.header:
                raise RuntimeError(f'Uninitialized <{self.__class__.__name__}>')
            return self.header.available_cash

        def __set__(self, double value):
            if not self.header:
                raise RuntimeError(f'Uninitialized <{self.__class__.__name__}>')
            self.header.available_cash = value
            self.header.flags |= dma_balance_flag.DMA_BALANCE_F_AVAILABLE_CASH

    property frozen_cash:
        def __get__(self):
            if not self.header:
                raise RuntimeError(f'Uninitialized <{self.__class__.__name__}>')
            return self.header.frozen_cash

        def __set__(self, double value):
            if not self.header:
                raise RuntimeError(f'Uninitialized <{self.__class__.__name__}>')
            self.header.frozen_cash = value
            self.header.flags |= dma_balance_flag.DMA_BALANCE_F_FROZEN_CASH

    property deposit:
        def __get__(self):
            if not self.header:
                raise RuntimeError(f'Uninitialized <{self.__class__.__name__}>')
            return self.header.deposit

        def __set__(self, double value):
            if not self.header:
                raise RuntimeError(f'Uninitialized <{self.__class__.__name__}>')
            self.header.deposit = value
            self.header.flags |= dma_balance_flag.DMA_BALANCE_F_DEPOSIT

    property withdrawal:
        def __get__(self):
            if not self.header:
                raise RuntimeError(f'Uninitialized <{self.__class__.__name__}>')
            return self.header.withdrawal

        def __set__(self, double value):
            if not self.header:
                raise RuntimeError(f'Uninitialized <{self.__class__.__name__}>')
            self.header.withdrawal = value
            self.header.flags |= dma_balance_flag.DMA_BALANCE_F_WITHDRAWAL

    property cash_in:
        def __get__(self):
            if not self.header:
                raise RuntimeError(f'Uninitialized <{self.__class__.__name__}>')
            return self.header.cash_in

        def __set__(self, double value):
            if not self.header:
                raise RuntimeError(f'Uninitialized <{self.__class__.__name__}>')
            self.header.cash_in = value
            self.header.flags |= dma_balance_flag.DMA_BALANCE_F_CASH_IN

    property cash_out:
        def __get__(self):
            if not self.header:
                raise RuntimeError(f'Uninitialized <{self.__class__.__name__}>')
            return self.header.cash_out

        def __set__(self, double value):
            if not self.header:
                raise RuntimeError(f'Uninitialized <{self.__class__.__name__}>')
            self.header.cash_out = value
            self.header.flags |= dma_balance_flag.DMA_BALANCE_F_CASH_OUT

    property margin:
        def __get__(self):
            if not self.header:
                raise RuntimeError(f'Uninitialized <{self.__class__.__name__}>')
            return self.header.margin

        def __set__(self, double value):
            if not self.header:
                raise RuntimeError(f'Uninitialized <{self.__class__.__name__}>')
            self.header.margin = value
            self.header.flags |= dma_balance_flag.DMA_BALANCE_F_MARGIN

    property initial_margin:
        def __get__(self):
            if not self.header:
                raise RuntimeError(f'Uninitialized <{self.__class__.__name__}>')
            return self.header.initial_margin

        def __set__(self, double value):
            if not self.header:
                raise RuntimeError(f'Uninitialized <{self.__class__.__name__}>')
            self.header.initial_margin = value
            self.header.flags |= dma_balance_flag.DMA_BALANCE_F_INITIAL_MARGIN

    property maintenance_margin:
        def __get__(self):
            if not self.header:
                raise RuntimeError(f'Uninitialized <{self.__class__.__name__}>')
            return self.header.maintenance_margin

        def __set__(self, double value):
            if not self.header:
                raise RuntimeError(f'Uninitialized <{self.__class__.__name__}>')
            self.header.maintenance_margin = value
            self.header.flags |= dma_balance_flag.DMA_BALANCE_F_MAINTENANCE_MARGIN

    property available_margin:
        def __get__(self):
            if not self.header:
                raise RuntimeError(f'Uninitialized <{self.__class__.__name__}>')
            return self.header.available_margin

        def __set__(self, double value):
            if not self.header:
                raise RuntimeError(f'Uninitialized <{self.__class__.__name__}>')
            self.header.available_margin = value
            self.header.flags |= dma_balance_flag.DMA_BALANCE_F_AVAILABLE_MARGIN

    property frozen_margin:
        def __get__(self):
            if not self.header:
                raise RuntimeError(f'Uninitialized <{self.__class__.__name__}>')
            return self.header.frozen_margin

        def __set__(self, double value):
            if not self.header:
                raise RuntimeError(f'Uninitialized <{self.__class__.__name__}>')
            self.header.frozen_margin = value
            self.header.flags |= dma_balance_flag.DMA_BALANCE_F_FROZEN_MARGIN

    property collateral:
        def __get__(self):
            if not self.header:
                raise RuntimeError(f'Uninitialized <{self.__class__.__name__}>')
            return self.header.collateral

        def __set__(self, double value):
            if not self.header:
                raise RuntimeError(f'Uninitialized <{self.__class__.__name__}>')
            self.header.collateral = value
            self.header.flags |= dma_balance_flag.DMA_BALANCE_F_COLLATERAL

    property available_collateral:
        def __get__(self):
            if not self.header:
                raise RuntimeError(f'Uninitialized <{self.__class__.__name__}>')
            return self.header.available_collateral

        def __set__(self, double value):
            if not self.header:
                raise RuntimeError(f'Uninitialized <{self.__class__.__name__}>')
            self.header.available_collateral = value
            self.header.flags |= dma_balance_flag.DMA_BALANCE_F_AVAILABLE_COLLATERAL

    property borrowed:
        def __get__(self):
            if not self.header:
                raise RuntimeError(f'Uninitialized <{self.__class__.__name__}>')
            return self.header.borrowed

        def __set__(self, double value):
            if not self.header:
                raise RuntimeError(f'Uninitialized <{self.__class__.__name__}>')
            self.header.borrowed = value
            self.header.flags |= dma_balance_flag.DMA_BALANCE_F_BORROWED

    property interest:
        def __get__(self):
            if not self.header:
                raise RuntimeError(f'Uninitialized <{self.__class__.__name__}>')
            return self.header.interest

        def __set__(self, double value):
            if not self.header:
                raise RuntimeError(f'Uninitialized <{self.__class__.__name__}>')
            self.header.interest = value
            self.header.flags |= dma_balance_flag.DMA_BALANCE_F_INTEREST

    property buying_power:
        def __get__(self):
            if not self.header:
                raise RuntimeError(f'Uninitialized <{self.__class__.__name__}>')
            return self.header.buying_power

        def __set__(self, double value):
            if not self.header:
                raise RuntimeError(f'Uninitialized <{self.__class__.__name__}>')
            self.header.buying_power = value
            self.header.flags |= dma_balance_flag.DMA_BALANCE_F_BUYING_POWER

    property leverage:
        def __get__(self):
            if not self.header:
                raise RuntimeError(f'Uninitialized <{self.__class__.__name__}>')
            return self.header.leverage

        def __set__(self, double value):
            if not self.header:
                raise RuntimeError(f'Uninitialized <{self.__class__.__name__}>')
            self.header.leverage = value
            self.header.flags |= dma_balance_flag.DMA_BALANCE_F_LEVERAGE

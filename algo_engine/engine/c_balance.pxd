from libc.stdint cimport uint64_t
from libcpp cimport bool as c_bool

from cbase.allocator_protocol cimport CCPType, allocator_protocol

from algo_engine.base.c_market_data.c_market_data cimport md_side


cdef extern from "algo_engine/engine/c_balance.h":
    ctypedef enum dma_balance_flag:
        DMA_BALANCE_F_BALANCE
        DMA_BALANCE_F_EQUITY
        DMA_BALANCE_F_NET_ASSET_VALUE
        DMA_BALANCE_F_TOTAL_ASSET
        DMA_BALANCE_F_TOTAL_LIABILITY
        DMA_BALANCE_F_UNREALIZED_PNL
        DMA_BALANCE_F_REALIZED_PNL
        DMA_BALANCE_F_CASH_BALANCE
        DMA_BALANCE_F_AVAILABLE_CASH
        DMA_BALANCE_F_FROZEN_CASH
        DMA_BALANCE_F_DEPOSIT
        DMA_BALANCE_F_WITHDRAWAL
        DMA_BALANCE_F_CASH_IN
        DMA_BALANCE_F_CASH_OUT
        DMA_BALANCE_F_MARGIN
        DMA_BALANCE_F_INITIAL_MARGIN
        DMA_BALANCE_F_MAINTENANCE_MARGIN
        DMA_BALANCE_F_AVAILABLE_MARGIN
        DMA_BALANCE_F_FROZEN_MARGIN
        DMA_BALANCE_F_COLLATERAL
        DMA_BALANCE_F_AVAILABLE_COLLATERAL
        DMA_BALANCE_F_BORROWED
        DMA_BALANCE_F_INTEREST
        DMA_BALANCE_F_BUYING_POWER
        DMA_BALANCE_F_LEVERAGE

    ctypedef enum dma_balance_ret_code:
        BALANCE_OK
        BALANCE_ERR_INVALID_INPUT
        BALANCE_ERR_OOM

    ctypedef struct dma_balance:
        double   balance
        double   equity
        double   net_asset_value
        double   total_asset
        double   total_liability
        double   unrealized_pnl
        double   realized_pnl
        double   cash_balance
        double   available_cash
        double   frozen_cash
        double   deposit
        double   withdrawal
        double   cash_in
        double   cash_out
        double   margin
        double   initial_margin
        double   maintenance_margin
        double   available_margin
        double   frozen_margin
        double   collateral
        double   available_collateral
        double   borrowed
        double   interest
        double   buying_power
        double   leverage
        uint64_t flags

    dma_balance* c_balance_new(allocator_protocol* allocator) noexcept nogil
    int          c_balance_init(dma_balance* balance) noexcept nogil
    void         c_balance_dealloc(dma_balance* balance) noexcept nogil
    void         c_balance_free(dma_balance* balance) noexcept nogil
    int          c_balance_clear(dma_balance* balance) noexcept nogil

    c_bool       c_balance_check(const dma_balance* balance, md_side side, double volume, double price, double multiplier, double margin_rate) noexcept nogil


cdef class Balance(CCPType):
    cdef dma_balance* header
    cdef bint owner

    cpdef void clear(self)

    cpdef bint check(self, md_side side, double volume, double price, double multiplier=?, double margin_rate=?)

    @staticmethod
    cdef Balance c_from_header(dma_balance* header, bint owner=?)

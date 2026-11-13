from libcpp cimport bool as c_bool

from cpython.datetime cimport date as py_date

from cbase.allocator_protocol cimport CCPType, allocator_protocol
from cbase.bytemap cimport bytemap

from algo_engine.base.c_market_data.c_market_data cimport md_direction
from algo_engine.exchange_profile.c_exchange_profile cimport session_date_t


cdef extern from "algo_engine/engine/c_inventory.h":
    ctypedef enum inventory_ret_code:
        INVENTORY_OK
        INVENTORY_ERR_INVALID_INPUT
        INVENTORY_ERR_OOM
        INVENTORY_ERR_NOT_FOUND

    ctypedef enum dma_security_type:
        DMA_SECURITY_COMMODITY
        DMA_SECURITY_CURRENCY_SWAP
        DMA_SECURITY_CRYPTO
        DMA_SECURITY_INDEX_FUTURE
        DMA_SECURITY_STOCK

    ctypedef struct dma_inventory_cash_dividend:
        session_date_t record_date
        session_date_t ex_date
        double         dividend_per_share
        c_bool         applied
        dma_inventory_cash_dividend* next

    ctypedef struct dma_inventory_stock_dividend:
        session_date_t record_date
        session_date_t ex_date
        double         dividend_per_share
        c_bool         applied
        dma_inventory_stock_dividend* next

    ctypedef struct dma_inventory_stock_split:
        session_date_t record_date
        session_date_t ex_date
        double         multiplier
        c_bool         applied
        dma_inventory_stock_split* next

    ctypedef struct dma_inventory_stock_conversion:
        session_date_t record_date
        session_date_t ex_date
        char*          convert_from
        char*          convert_to
        double         multiplier
        c_bool         applied
        dma_inventory_stock_conversion* next

    ctypedef struct dma_inventory_entry:
        double               volume
        double               price
        double               notional
        double               fee
        double               recalled
        dma_inventory_cash_dividend*    cash_dividends
        dma_inventory_stock_dividend*   stock_dividends
        dma_inventory_stock_split*      stock_splits
        dma_inventory_stock_conversion* stock_conversions
        dma_inventory_entry* next
        dma_security_type    security_type

    ctypedef struct dma_inventory_node:
        const char*          ticker
        dma_inventory_entry* long_inv
        dma_inventory_entry* short_inv
        double               long_used
        double               short_used

    ctypedef struct dma_inventory:
        bytemap* nodes

    dma_inventory* c_dma_inventory_new(allocator_protocol* allocator) noexcept nogil
    int c_dma_inventory_init(dma_inventory* inventory, allocator_protocol* allocator) noexcept nogil
    void c_dma_inventory_dealloc(dma_inventory* inventory) noexcept nogil
    void c_dma_inventory_free(dma_inventory* inventory) noexcept nogil
    int c_dma_inventory_clear(dma_inventory* inventory) noexcept nogil

    double c_dma_inventory_entry_available(const dma_inventory_entry* entry) noexcept nogil
    void c_dma_inventory_entry_clear(dma_inventory_entry* entry) noexcept nogil

    dma_inventory_entry* c_dma_inventory_entry_new(allocator_protocol* allocator) noexcept nogil
    void c_dma_inventory_entry_free(dma_inventory_entry* entry) noexcept nogil

    dma_inventory_cash_dividend* c_dma_inventory_cash_dividend_new(allocator_protocol* allocator) noexcept nogil
    void c_dma_inventory_cash_dividend_free(dma_inventory_cash_dividend* event) noexcept nogil

    dma_inventory_stock_dividend* c_dma_inventory_stock_dividend_new(allocator_protocol* allocator) noexcept nogil
    void c_dma_inventory_stock_dividend_free(dma_inventory_stock_dividend* event) noexcept nogil

    dma_inventory_stock_split* c_dma_inventory_stock_split_new(allocator_protocol* allocator) noexcept nogil
    void c_dma_inventory_stock_split_free(dma_inventory_stock_split* event) noexcept nogil

    dma_inventory_stock_conversion* c_dma_inventory_stock_conversion_new(allocator_protocol* allocator) noexcept nogil
    int c_dma_inventory_stock_conversion_set_tickers(dma_inventory_stock_conversion* event, const char* convert_from, const char* convert_to) noexcept nogil
    void c_dma_inventory_stock_conversion_free(dma_inventory_stock_conversion* event) noexcept nogil

    int c_dma_inventory_cash_dividend(dma_inventory* inventory, session_date_t record_date, session_date_t ex_date, double dividend_per_share) noexcept nogil
    int c_dma_inventory_stock_dividend(dma_inventory* inventory, session_date_t record_date, session_date_t ex_date, double dividend_per_share) noexcept nogil
    int c_dma_inventory_stock_split(dma_inventory* inventory, session_date_t record_date, session_date_t ex_date, double multiplier) noexcept nogil
    int c_dma_inventory_stock_conversion(dma_inventory* inventory, session_date_t record_date, session_date_t ex_date, const char* convert_from, const char* convert_to, double multiplier) noexcept nogil
    int c_dma_inventory_add_inv_long(dma_inventory* inventory, const char* ticker, double price, double volume, double notional, double fee, dma_security_type security_type) noexcept nogil
    int c_dma_inventory_add_inv_short(dma_inventory* inventory, const char* ticker, double price, double volume, double notional, double fee, dma_security_type security_type) noexcept nogil
    int c_dma_inventory_get_inv_long(const dma_inventory* inventory, const char* ticker, const dma_inventory_entry** out) noexcept nogil
    int c_dma_inventory_get_inv_short(const dma_inventory* inventory, const char* ticker, const dma_inventory_entry** out) noexcept nogil
    int c_dma_inventory_use_inv_long(dma_inventory* inventory, const char* ticker, double volume) noexcept nogil
    int c_dma_inventory_use_inv_short(dma_inventory* inventory, const char* ticker, double volume) noexcept nogil
    int c_dma_inventory_recall_inv_long(dma_inventory* inventory, const char* ticker, double volume) noexcept nogil
    int c_dma_inventory_recall_inv_short(dma_inventory* inventory, const char* ticker, double volume) noexcept nogil
    int c_dma_inventory_available(const dma_inventory* inventory, const char* ticker, double* out_long, double* out_short) noexcept nogil


cdef class InventoryCashDividend(CCPType):
    cdef dma_inventory_cash_dividend* header
    cdef bint owner

    @staticmethod
    cdef InventoryCashDividend c_from_header(dma_inventory_cash_dividend* header, bint owner)


cdef class InventoryStockDividend(CCPType):
    cdef dma_inventory_stock_dividend* header
    cdef bint owner

    @staticmethod
    cdef InventoryStockDividend c_from_header(dma_inventory_stock_dividend* header, bint owner)


cdef class InventoryStockSplit(CCPType):
    cdef dma_inventory_stock_split* header
    cdef bint owner

    @staticmethod
    cdef InventoryStockSplit c_from_header(dma_inventory_stock_split* header, bint owner)


cdef class InventoryStockConversion(CCPType):
    cdef dma_inventory_stock_conversion* header
    cdef bint owner

    @staticmethod
    cdef InventoryStockConversion c_from_header(dma_inventory_stock_conversion* header, bint owner)


cdef class InventoryEntry(CCPType):
    cdef dma_inventory_entry* header
    cdef bint owner

    cdef readonly str ticker
    cdef md_direction c_direction

    @staticmethod
    cdef InventoryEntry c_from_header(dma_inventory_entry* header, str ticker, md_direction direction, bint owner)


cdef class Inventory(CCPType):
    cdef dma_inventory* header
    cdef bint owner

    cdef double c_side_volume(self, str ticker, bint is_long, bint* found)

    cpdef void recall(self, str ticker, double volume, md_direction direction=?)

    cpdef void add_inv(self, InventoryEntry entry)

    cpdef InventoryEntry get_inv(self, str ticker, md_direction direction=?)

    cpdef void use_inv(self, str ticker, double volume, md_direction direction=?)

    cpdef double available_volume(self, str ticker, md_direction direction=?)

    cpdef void cash_dividend(self, py_date record_date, py_date ex_date, double dividend_per_share)

    cpdef void stock_dividend(self, py_date record_date, py_date ex_date, double dividend_per_share)

    cpdef void stock_split(self, py_date record_date, py_date ex_date, double multiplier)

    cpdef void stock_conversion(self, py_date record_date, py_date ex_date, str convert_from, str convert_to, double multiplier)

    cpdef void clear(self)

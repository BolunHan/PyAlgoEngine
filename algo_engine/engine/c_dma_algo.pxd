from libc.stdint cimport uint8_t, uint32_t

from cbase.allocator_protocol cimport CCPType, allocator_protocol
from cbase.bytemap cimport BoundByteMap, bytemap

from algo_engine.base.c_market_data.c_market_data cimport long_md_id, md_side, md_trade_instruction, md_trade_report, md_variant

from .algo.c_algo_template cimport AlgoTemplate, dma_algo


cdef extern from "algo_engine/engine/c_dma_algo.h":
    ctypedef struct dma:
        pass

    ctypedef enum dma_algo_ret_code:
        DMA_ALGO_OK
        DMA_ALGO_ERR_INVALID_INPUT
        DMA_ALGO_ERR_OOM
        DMA_ALGO_ERR_NOT_FOUND
        DMA_ALGO_ERR_WORKING_ORDERS

    ctypedef enum dma_algo_limit_mode:
        DMA_ALGO_LIMIT_STRICT
        DMA_ALGO_LIMIT_LOOSE

    ctypedef enum dma_order_flag:
        DMA_ORDER_LISTED
        DMA_ORDER_CANCELED
        DMA_ORDER_REJECTED
        DMA_ORDER_ERROR

    ctypedef dma_algo* (*dma_algo_new)(dma* DMA, const char* ticker, md_side side, double target_volume, const void* params, allocator_protocol* allocator) noexcept

    ctypedef struct dma_algo_registry:
        bytemap* entries
        bytemap* id_map

    ctypedef struct dma_algo_engine:
        dma_algo_registry* registry
        bytemap* orders
        bytemap* working_orders_by_id
        bytemap* working_orders_by_ticker
        bytemap* algo
        bytemap* working_algo
        dma_algo* error_algo
        bytemap* position

    ctypedef struct dma_algo_ctx:
        dma_algo_new algo_new_fn
        bytemap* orders
        bytemap* algos
        uint32_t reg_id

    ctypedef struct dma_order_ctx:
        md_trade_instruction order
        bytemap* reports
        dma_algo* algo
        dma_order_ctx* next
        uint8_t flags

    int c_dma_order_ctx_init(dma_order_ctx* ctx, allocator_protocol* allocator) noexcept nogil
    int c_dma_order_ctx_fill(dma_order_ctx* ctx, const md_trade_report* report) noexcept nogil
    void c_dma_order_ctx_dealloc(dma_order_ctx* ctx) noexcept nogil

    dma_algo_registry* c_dma_algo_registry_new(allocator_protocol* allocator) noexcept nogil
    int c_dma_algo_registry_init(dma_algo_registry* registry, allocator_protocol* allocator) noexcept nogil
    void c_dma_algo_registry_dealloc(dma_algo_registry* registry) noexcept nogil
    void c_dma_algo_registry_free(dma_algo_registry* registry) noexcept nogil
    dma_algo_ctx* c_dma_algo_registry_reg(dma_algo_registry* registry, const char* name, dma_algo_new algo_new_fn, uint32_t reg_id) noexcept nogil
    dma_algo_ctx* c_dma_algo_registry_get(const dma_algo_registry* registry, uint32_t reg_id) noexcept nogil

    dma_algo_engine* c_dma_algo_engine_new(allocator_protocol* allocator) noexcept nogil
    int c_dma_algo_engine_init(dma_algo_engine* engine, allocator_protocol* allocator) noexcept nogil
    int c_dma_algo_engine_clear(dma_algo_engine* engine) noexcept nogil
    void c_dma_algo_engine_dealloc(dma_algo_engine* engine) noexcept nogil
    void c_dma_algo_engine_free(dma_algo_engine* engine) noexcept nogil

    void c_dma_algo_engine_pop_working_order(dma_algo_engine* engine, const long_md_id* order_id) noexcept nogil
    md_trade_instruction* c_dma_algo_engine_new_order(dma_algo_engine* engine, dma_algo* algo, double limit_price, double volume) noexcept nogil
    void c_dma_algo_engine_free_orders(dma_algo_engine* engine, dma_algo* algo) noexcept nogil
    int c_dma_algo_engine_on_order(dma_algo_engine* engine, const md_trade_instruction* order) noexcept nogil
    int c_dma_algo_engine_on_trade(dma_algo_engine* engine, const md_trade_report* trade) noexcept nogil
    void c_dma_algo_engine_on_md(dma_algo_engine* engine, const md_variant* market_data) noexcept nogil

    double c_dma_algo_compare_limit(md_side side, double limit_price, double original_limit, dma_algo_limit_mode mode) noexcept nogil


cdef class OrderMapView(BoundByteMap):
    @staticmethod
    cdef OrderMapView c_from_header(bytemap* header, bint owner=?)


cdef class AlgoMapView(BoundByteMap):
    @staticmethod
    cdef AlgoMapView c_from_header(bytemap* header, bint owner=?)



cdef class AlgoRegistry(CCPType):
    cdef dma_algo_registry* header
    cdef bint owner
    cdef dict classes

    cpdef void add_algo(self, str name, type handler)

    cpdef object cast(self, str name)

    cdef AlgoTemplate c_from_algo(self, dma_algo* algo)

    @staticmethod
    cdef AlgoRegistry c_from_header(dma_algo_registry* header, bint owner)


cdef class AlgoEngine(CCPType):
    cdef dma_algo_engine* header
    cdef bint owner

    cdef readonly AlgoRegistry registry
    cdef OrderMapView orders_map
    cdef OrderMapView working_orders_map
    cdef AlgoMapView algos_map

    cdef int c_on_order(self, const md_trade_instruction* order)

    cdef int c_on_trade(self, const md_trade_report* trade)

    cdef void c_on_md(self, const md_variant* market_data)


cdef AlgoRegistry ALGO_REGISTRY
cdef AlgoEngine ALGO_ENGINE

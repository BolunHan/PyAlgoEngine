from libc.stdint cimport uint32_t
from libcpp cimport bool as c_bool

from cbase.allocator_protocol cimport CCPType, allocator_protocol
from cbase.bytemap cimport bytemap

from algo_engine.base.c_market_data.c_market_data cimport md_side, md_trade_instruction, md_trade_report

from .c_balance cimport Balance, dma_balance
from .c_dma_algo cimport AlgoEngine, dma_algo, dma_algo_engine, dma_algo_limit_mode
from .c_inventory cimport dma_inventory
from .c_market_engine cimport MarketDataService, mds
from .risk_profile.c_risk_profile cimport RiskProfile, dma_risk_profile, dma_risk_violation


cdef extern from "algo_engine/engine/c_dma.h":
    ctypedef enum dma_ret_code:
        DMA_OK_NOOP
        DMA_OK_BUFFERED
        DMA_OK
        DMA_ERR_INVALID_INPUT
        DMA_ERR_OOM
        DMA_ERR_NOT_FOUND
        DMA_ERR_EMPTY
        DMA_ERR_NO_RISK
        DMA_ERR_RISK
        DMA_ERR_FULL

    ctypedef enum dma_order_action:
        DMA_ACTION_LAUNCH
        DMA_ACTION_CANCEL

    ctypedef void (*dma_order_launch_func)(md_trade_instruction* order, uint32_t seq_id, void* user_data) noexcept
    ctypedef void (*dma_order_cancel_func)(const md_trade_instruction* order, uint32_t seq_id, void* user_data) noexcept
    ctypedef void (*dma_order_rejected_func)(const md_trade_instruction* order, void* user_data) noexcept
    ctypedef void (*dma_order_filled_func)(const md_trade_instruction* order, md_trade_report* report, void* user_data) noexcept
    ctypedef int (*dma_balance_query)(const void* DMA, dma_balance* out, void* user_data) noexcept

    ctypedef struct dma_order_handler_ctx:
        dma_order_launch_func order_launch_fn
        dma_order_cancel_func order_cancel_fn
        dma_balance_query     balance_check_fn

    ctypedef struct dma_order_callbacks_ctx:
        dma_order_rejected_func order_rejected_fn
        dma_order_filled_func order_filled_fn

    ctypedef struct dma_protocol:
        dma_order_handler_ctx   handlers
        dma_order_callbacks_ctx callbacks
        void*                   user_data

    ctypedef struct dma_order_request:
        md_trade_instruction* order
        dma_order_action      action

    ctypedef struct dma_order_queue:
        dma_order_request* entries
        size_t             capacity
        size_t             head
        size_t             size

    ctypedef struct dma:
        mds*                  MDS
        dma_algo_engine*      ALGO_ENGINE
        dma_risk_profile*     RISK_PROFILE
        dma_inventory*        INVENTORY
        dma_balance*          BALANCE
        bytemap*              positions
        dma_protocol          protocol
        dma_order_queue       requests
        uint32_t              seq_id
        c_bool                balance_valid

    dma*          c_dma_new(mds* MDS, allocator_protocol* allocator) noexcept nogil
    int           c_dma_init(dma* DMA, mds* MDS, allocator_protocol* allocator) noexcept nogil
    void          c_dma_dealloc(dma* DMA) noexcept nogil
    void          c_dma_free(dma* DMA) noexcept nogil
    int           c_dma_clear(dma* DMA) noexcept nogil

    dma_algo*     c_dma_new_algo(dma* DMA, uint32_t algo_reg_id, const char* ticker, md_side side, double target_volume, const void* params, allocator_protocol* allocator) noexcept nogil
    double        c_dma_calculate_limit(dma* DMA, const char* ticker, md_side side, double limit_price, double limit_adjust_factor, dma_algo_limit_mode mode) noexcept nogil

    int           c_dma_attach_protocol_handler(dma* DMA, dma_order_launch_func order_launch_fn, dma_order_cancel_func order_cancel_fn, dma_balance_query balance_check_fn, void* user_data) noexcept nogil
    int           c_dma_attach_protocol_callback(dma* DMA, dma_order_rejected_func order_rejected_fn, dma_order_filled_func order_filled_fn) noexcept nogil
    int           c_dma_detach_protocol_handler(dma* DMA) noexcept nogil
    int           c_dma_detach_protocol_callback(dma* DMA) noexcept nogil
    int           c_dma_query_balance(dma* DMA) noexcept nogil

    void          c_dma_refuse(const dma* DMA, const md_trade_instruction* order, const char* op) noexcept nogil
    int           c_dma_check_order(dma* DMA, const md_trade_instruction* order, dma_risk_violation* out_violation) noexcept nogil
    uint32_t      c_dma_next_seq_id(dma* DMA) noexcept nogil
    int           c_dma_launch_order(dma* DMA, md_trade_instruction* order) noexcept nogil
    int           c_dma_cancel_order(dma* DMA, md_trade_instruction* order) noexcept nogil
    int           c_dma_launch_order_unchecked(dma* DMA, md_trade_instruction* order) noexcept nogil
    int           c_dma_process_queue(dma* DMA) noexcept nogil

    int           c_dma_order_queue_init(dma_order_queue* queue, allocator_protocol* allocator) noexcept nogil
    void          c_dma_order_queue_dealloc(dma_order_queue* queue) noexcept nogil
    int           c_dma_order_queue_grow(dma_order_queue* queue) noexcept nogil
    int           c_dma_order_queue_push(dma_order_queue* queue, dma_order_action action, md_trade_instruction* order) noexcept nogil
    int           c_dma_order_queue_pop(dma_order_queue* queue, dma_order_action* out_action, md_trade_instruction** out_order) noexcept nogil
    void          c_dma_order_queue_clear(dma_order_queue* queue) noexcept nogil


cdef class DirectMarketAccess(CCPType):
    cdef dma* header
    cdef bint owner

    cdef readonly AlgoEngine algo_engine
    cdef readonly MarketDataService mds
    cdef readonly RiskProfile risk_profile
    cdef readonly Balance balance

    cpdef object calculate_limit(self, str ticker, md_side side, object limit_price=?, object limit_adjust_factor=?, dma_algo_limit_mode mode=?)

    cpdef int process_queue(self)

    cdef int c_bind_protocol(self)

    @staticmethod
    cdef inline void c_launch_order_adaptor(md_trade_instruction* order, uint32_t seq_id, void* user_data) noexcept with gil

    @staticmethod
    cdef inline void c_cancel_order_adaptor(const md_trade_instruction* order, uint32_t seq_id, void* user_data) noexcept with gil

    @staticmethod
    cdef inline int c_balance_check_adaptor(const void* DMA, dma_balance* out, void* user_data) noexcept with gil

    @staticmethod
    cdef inline void c_rejected_order_adaptor(const md_trade_instruction* order, void* user_data) noexcept with gil

    @staticmethod
    cdef inline void c_filled_order_adaptor(const md_trade_instruction* order, md_trade_report* report, void* user_data) noexcept with gil

    cdef void c_launch_order_handler(self, md_trade_instruction* order, uint32_t seq_id)

    cdef void c_cancel_order_handler(self, const md_trade_instruction* order, uint32_t seq_id)

    cdef int c_balance_check_handler(self, const void* DMA, dma_balance* out)

    cdef void c_rejected_order_callback(self, const md_trade_instruction* order)

    cdef void c_filled_order_callback(self, const md_trade_instruction* order, md_trade_report* report)


cdef DirectMarketAccess DMA
cdef RiskProfile RISK_PROFILE
cdef Balance BALANCE

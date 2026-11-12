from libc.stdint cimport uint32_t

from cbase.allocator_protocol cimport CCPType, allocator_protocol
from cbase.bytemap cimport bytemap

from algo_engine.base.c_market_data.c_market_data cimport long_md_id, md_side, md_trade_instruction, md_trade_report, md_variant
from algo_engine.engine.c_dma_algo cimport OrderMapView, dma
from algo_engine.exchange_profile.c_exchange_profile cimport session_datetime_t


cdef extern from "algo_engine/engine/algo/c_algo_template.h":
    ctypedef enum dma_algo_status:
        DMA_ALGO_STATUS_IDLE
        DMA_ALGO_STATUS_PREPARING
        DMA_ALGO_STATUS_READY
        DMA_ALGO_STATUS_WORKING
        DMA_ALGO_STATUS_DONE
        DMA_ALGO_STATUS_CLOSED
        DMA_ALGO_STATUS_STOPPING
        DMA_ALGO_STATUS_REJECTED
        DMA_ALGO_STATUS_ERROR

    ctypedef unsigned char uuid_t[16]

    ctypedef void (*dma_algo_order_func)(dma* DMA, dma_algo* algo, const md_trade_instruction* order, void* user_data) noexcept
    ctypedef void (*dma_algo_trade_func)(dma* DMA, dma_algo* algo, const md_trade_report* trade, void* user_data) noexcept
    ctypedef void (*dma_algo_md_func)(dma* DMA, dma_algo* algo, const md_variant* market_data, void* user_data) noexcept

    ctypedef struct dma_algo_callback_ctx:
        dma_algo_order_func on_canceled
        dma_algo_order_func on_rejected
        dma_algo_order_func on_listed
        dma_algo_trade_func on_filled
        dma_algo_md_func    on_md
        void*               user_data

    ctypedef int (*dma_algo_launch_func)(dma* DMA, dma_algo* algo, void* user_data) noexcept
    ctypedef int (*dma_algo_stop_func)(dma* DMA, dma_algo* algo, void* user_data) noexcept

    ctypedef struct dma_algo_handler_ctx:
        dma_algo_launch_func launch
        dma_algo_stop_func   stop
        void*                user_data

    ctypedef struct dma_algo:
        dma*                  DMA
        uuid_t                algo_id
        uint32_t              algo_reg_id
        const char*           ticker
        md_side               side
        dma_algo_status       status
        double                target_volume
        double                working_volume
        double                exposure_volume
        double                filled_volume
        double                filled_notional
        double                fee
        double                cash_flow
        double                multiplier
        dma_algo_callback_ctx callbacks
        dma_algo_handler_ctx  handlers
        bytemap*              orders
        bytemap*              working_orders
        session_datetime_t    ts_started
        session_datetime_t    ts_finished
        dma_algo*             prev
        dma_algo*             next

    int                         c_dma_algo_init(dma_algo* algo, dma* DMA, const char* ticker, md_side side, double target_volume, double multiplier, allocator_protocol* allocator) noexcept nogil
    void                        c_dma_algo_gen_uid(dma_algo* algo) noexcept nogil
    int                         c_dma_algo_set_status(dma_algo* algo, dma_algo_status status) noexcept nogil
    void                        c_dma_algo_dealloc(dma_algo* algo) noexcept nogil
    void                        c_dma_algo_free(dma_algo* algo) noexcept nogil

    const md_trade_instruction* c_dma_algo_get_order(const dma_algo* algo, const long_md_id* order_id) noexcept nogil
    const md_trade_instruction* c_dma_algo_get_working_order(const dma_algo* algo, const long_md_id* order_id) noexcept nogil

    int  c_dma_algo_launch(dma_algo* algo) noexcept nogil
    int  c_dma_algo_stop(dma_algo* algo) noexcept nogil


cdef class AlgoTemplate(CCPType):
    cdef dma_algo* header

    cdef OrderMapView orders_map
    cdef OrderMapView working_orders_map

    cdef int c_launch(self)

    cdef int c_stop(self)

    @staticmethod
    cdef AlgoTemplate c_from_header(dma_algo* header)

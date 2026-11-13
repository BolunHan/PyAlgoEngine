from cpython.object cimport PyObject
from libc.stdint cimport uintptr_t
from libcpp cimport bool as c_bool

from cbase.allocator_protocol cimport CCPType, allocator_protocol
from cbase.bytemap cimport bytemap

from algo_engine.base.c_market_data.c_market_data cimport md_side, md_trade_instruction, md_trade_report

from .algo.c_algo_template cimport AlgoTemplate, dma_algo_status, uuid_t
from .c_dma cimport DirectMarketAccess, dma
from .c_dma_algo cimport AlgoEngine, AlgoMapView, dma_algo, dma_algo_engine


cdef extern from "Python.h":
    PyObject* PyObject_CallFunctionObjArgs(PyObject* callable, ...)
    PyObject* PyErr_GetRaisedException()
    void PyErr_DisplayException(PyObject* exc)


cdef extern from "algo_engine/engine/c_position.h":
    const char* DMA_POSITION_DEFAULT_ALGO

    ctypedef enum position_ret_code:
        POSITION_OK
        POSITION_ERR_INVALID_INPUT
        POSITION_ERR_OOM
        POSITION_ERR_NOT_FOUND

    ctypedef enum dma_callback_event:
        on_canceled
        on_rejected
        on_listed
        on_filled

    ctypedef int (*dma_position_order_func)(dma_position* position, const dma_algo* algo, const md_trade_instruction* order, void* user_data) noexcept
    ctypedef int (*dma_position_trade_func)(dma_position* position, const dma_algo* algo, const md_trade_instruction* order, const md_trade_report* trade, void* user_data) noexcept

    ctypedef struct dma_position_callback_ctx:
        dma_position_order_func on_canceled
        dma_position_order_func on_rejected
        dma_position_order_func on_listed
        dma_position_trade_func on_filled
        void* user_data
        uintptr_t id
        dma_position_callback_ctx* next

    ctypedef struct dma_position_ticker:
        double exposure_volume
        double working_long
        double working_short
        double filled_notional
        double cash_flow
        double multiplier

    ctypedef struct dma_position:
        dma* dma
        dma_algo_engine* algo_engine
        uuid_t position_id
        char* default_algo
        bytemap* algos
        bytemap* working_algos
        bytemap* tickers
        dma_position_callback_ctx* callbacks

    # Lifecycle
    dma_position* c_dma_position_new(dma* access, dma_algo_engine* engine, const char* default_algo, allocator_protocol* allocator) noexcept nogil
    int c_dma_position_init(dma_position* position, dma* access, dma_algo_engine* engine, const char* default_algo, allocator_protocol* allocator) noexcept nogil
    void c_dma_position_dealloc(dma_position* position) noexcept nogil
    void c_dma_position_free(dma_position* position) noexcept nogil
    int c_dma_position_clear(dma_position* position) noexcept nogil

    # Identity
    void c_dma_position_gen_uid(dma_position* position) noexcept nogil

    # Status
    c_bool c_dma_position_status_is_active(dma_algo_status status) noexcept nogil

    # Algo registry
    int c_dma_position_add_algo(dma_position* position, dma_algo* algo) noexcept nogil
    int c_dma_position_add_algo_by_id(dma_position* position, const unsigned char* algo_id) noexcept nogil
    int c_dma_position_pop_algo(dma_position* position, const unsigned char* algo_id) noexcept nogil
    dma_algo* c_dma_position_get_algo(const dma_position* position, const unsigned char* algo_id) noexcept nogil

    # Queries
    size_t c_dma_position_n_algos(const dma_position* position) noexcept nogil
    size_t c_dma_position_n_working_algos(const dma_position* position) noexcept nogil
    double c_dma_position_get_exposure(const dma_position* position, const char* ticker) noexcept nogil
    int c_dma_position_get_working(const dma_position* position, const char* ticker, double* out_long, double* out_short) noexcept nogil
    double c_dma_position_working_net(const dma_position* position, const char* ticker) noexcept nogil
    double c_dma_position_notional(const dma_position* position, const char* ticker) noexcept nogil
    double c_dma_position_pnl(const dma_position* position, const char* ticker) noexcept nogil

    # Exposure
    int c_dma_position_add_exposure(dma_position* position, const char* ticker, double volume, double notional, double multiplier, md_side side) noexcept nogil

    # Callbacks
    int c_dma_position_register_callback(dma_position* position, dma_callback_event event, void* callback, void* user_data, uintptr_t* out_id) noexcept nogil
    int c_dma_position_unregister_callback(dma_position* position, uintptr_t callback_id) noexcept nogil

    # Trading
    int c_dma_position_fully_stop(dma_position* position) noexcept nogil
    int c_dma_position_fully_unwind(dma_position* position) noexcept nogil

    # Hot path
    int c_dma_position_invoke_callbacks(dma_position* position, dma_callback_event event, const dma_algo* algo, const md_trade_instruction* order, const md_trade_report* trade) noexcept nogil


cdef class PositionManagementService(CCPType):
    cdef dma_position* header

    cdef uintptr_t on_canceled_callback_id
    cdef uintptr_t on_rejected_callback_id
    cdef uintptr_t on_listed_callback_id
    cdef uintptr_t on_filled_callback_id

    cdef AlgoMapView algos_map
    cdef AlgoMapView working_algos_map
    cdef list on_canceled_callback
    cdef list on_rejected_callback
    cdef list on_listed_callback
    cdef list on_filled_callback

    cdef readonly AlgoEngine algo_engine
    cdef readonly DirectMarketAccess dma
    cdef readonly dict callbacks

    cdef object c_register_callback(self, dma_callback_event event, object callback)

    cpdef void add_algo(self, AlgoTemplate algo)

    cpdef void add_exposure(self, str ticker, double volume, double notional, double multiplier, md_side side)

    cpdef void unwind_all(self)

    cpdef void cancel_all(self)

    cpdef AlgoTemplate unwind(self, str ticker)

    cpdef dict pnl(self)

    @staticmethod
    cdef int c_canceled_callback_adaptor(dma_position* position, const dma_algo* algo, const md_trade_instruction* order, void* user_data) noexcept with gil

    @staticmethod
    cdef int c_rejected_callback_adaptor(dma_position* position, const dma_algo* algo, const md_trade_instruction* order, void* user_data) noexcept with gil

    @staticmethod
    cdef int c_listed_callback_adaptor(dma_position* position, const dma_algo* algo, const md_trade_instruction* order, void* user_data) noexcept with gil

    @staticmethod
    cdef int c_filled_callback_adaptor(dma_position* position, const dma_algo* algo, const md_trade_instruction* order, const md_trade_report* trade, void* user_data) noexcept with gil

from cpython.object cimport PyObject
from libc.stdint cimport int8_t, uint64_t, uintptr_t

from cbase.allocator_protocol.c_allocator_protocol cimport allocator_protocol
from cbase.bytemap.c_bytemap cimport bytemap
from event_engine.capi.c_engine cimport message_queue
from event_engine.capi.c_event cimport evt_hook
from event_engine.capi.c_topic cimport evt_topic

from algo_engine.base.c_market_data.c_market_data cimport long_md_id, md_trade_report, md_variant


cdef extern from "algo_engine/backtest/c_simmatch_ex.h":
    ctypedef enum smm_match_event:
        SMM_EVENT_MATCH
        SMM_EVENT_CANCEL
        SMM_EVENT_PLACED

    ctypedef enum smm_ret_code:
        SMM_OK
        SMM_ERR_INVALID_ARG
        SMM_ERR_DUPLICATE
        SMM_ERR_NOT_FOUND
        SMM_ERR_NO_VALID_PRICE
        SMM_ERR_INVALID_SIDE
        SMM_ERR_OOM

    ctypedef struct smm_order_entry
    ctypedef struct smm_match_ctx
    ctypedef struct smm_listener
    ctypedef struct smm_engine_binding

    ctypedef void (*smm_match_callback)(smm_match_ctx* ctx, smm_match_event event,
                                        const smm_order_entry* entry,
                                        const md_trade_report* report,
                                        void* user_data) noexcept

    ctypedef struct smm_order_entry:
        long_md_id order_id
        md_variant* header
        PyObject* py_order
        uint64_t transaction_count_at_placement
        smm_order_entry* next

    ctypedef struct smm_listener:
        smm_match_callback callback
        void* user_data
        uintptr_t id
        smm_listener* next

    ctypedef struct smm_engine_binding:
        message_queue* mq
        bytemap* exact_topic_hooks
        bytemap* generic_topic_hooks
        evt_topic* launch_topic
        evt_hook* launch_hook
        PyObject* launch_hook_obj
        evt_topic* cancel_topic
        evt_hook* cancel_hook
        PyObject* cancel_hook_obj
        evt_topic* realtime_topic
        evt_hook* realtime_hook
        PyObject* realtime_hook_obj
        evt_topic* on_order_topic
        PyObject* on_order_topic_obj
        evt_topic* on_report_topic
        PyObject* on_report_topic_obj
        PyObject* market_data_class

    ctypedef struct smm_match_ctx:
        allocator_protocol* allocator
        double timestamp
        double last_price
        uint64_t last_transaction_count
        uint64_t seed
        uint64_t rng_state
        double fee_rate
        bint instant_fill
        double lag_ts
        uint64_t lag_n_transaction
        double hit_prob
        double slippery_rate
        bint incremental_order_volume
        smm_order_entry* working
        size_t n_working
        size_t n_history
        smm_listener* listeners
        smm_engine_binding* engine
        md_variant ws_report

    smm_match_ctx* c_smm_match_new(double fee_rate, bint instant_fill, double lag_ts,
                                   uint64_t lag_n_transaction, double hit_prob,
                                   double slippery_rate, bint incremental_order_volume,
                                   allocator_protocol* allocator)
    void c_smm_match_free(smm_match_ctx* ctx)
    void c_smm_match_init(smm_match_ctx* ctx)
    void c_smm_match_dealloc(smm_match_ctx* ctx)
    void c_smm_match_clear(smm_match_ctx* ctx)
    int c_smm_match_set_config(smm_match_ctx* ctx, double fee_rate, bint instant_fill,
                               double lag_ts, uint64_t lag_n_transaction,
                               double hit_prob, double slippery_rate,
                               bint incremental_order_volume)
    int c_smm_match_register_listener(smm_match_ctx* ctx, smm_match_callback callback,
                                      void* user_data, uintptr_t* out_id)
    int c_smm_match_deregister_listener(smm_match_ctx* ctx, uintptr_t listener_id)
    int c_smm_match_set_seed(smm_match_ctx* ctx, uint64_t seed)
    int c_smm_match_register(smm_match_ctx* ctx, const smm_engine_binding* binding)
    int c_smm_match_unregister(smm_match_ctx* ctx)
    int c_smm_match_launch(smm_match_ctx* ctx, md_variant* order_header, PyObject* py_order)
    int c_smm_match_cancel(smm_match_ctx* ctx, const long_md_id* order_id)
    void c_smm_match_eod(smm_match_ctx* ctx)
    int c_smm_match_process(smm_match_ctx* ctx, const md_variant* market_data)
    int c_smm_match_best_price(const double* prices, size_t n, int8_t sign, double* out)
    int c_smm_match_worst_price(const double* prices, size_t n, int8_t sign, double* out)


cdef class SimMatchEx:
    cdef smm_match_ctx* ctx
    cdef bint owner
    cdef uintptr_t _callback_id
    cdef public str ticker
    cdef public object event_engine
    cdef public object topic_set
    cdef public dict working
    cdef public dict history
    cdef public object random
    cdef uint64_t _seed

    cdef inline int c_process(self, const md_variant* market_data)

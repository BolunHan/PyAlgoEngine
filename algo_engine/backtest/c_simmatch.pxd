from libc.stdint cimport int8_t, uint64_t, uintptr_t
from libcpp cimport bool as c_bool

from cbase.allocator_protocol cimport CCPType, allocator_protocol
from cbase.bytemap cimport bytemap
from event_engine.capi cimport EventEngineEx

from algo_engine.base.c_market_data.c_market_data cimport long_md_id, md_candlestick, md_order_data, md_tick_data, md_tick_data_lite, md_trade_instruction, md_trade_report, md_transaction_data, md_variant
from algo_engine.engine.c_dma_protocol cimport DMAProtocol
from algo_engine.engine.c_event_engine cimport TopicSetEx


cdef extern from "algo_engine/backtest/c_simmatch.h":
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
    ctypedef struct smm_order_ring
    ctypedef struct smm_order_ring_cursor
    ctypedef struct smm_match_ctx
    ctypedef struct smm_listener

    ctypedef void (*smm_match_callback)(smm_match_ctx* ctx, smm_match_event event, const smm_order_entry* entry, const md_trade_report* report, void* user_data) noexcept

    ctypedef struct smm_order_entry:
        md_trade_instruction* order
        uint64_t              transaction_count_at_placement
        long_md_id            order_id
        c_bool                occupied

    ctypedef struct smm_order_ring:
        smm_order_entry* slots
        bytemap*         index
        size_t           capacity
        size_t           head
        size_t           span
        size_t           n_working
        uint64_t         generation

    ctypedef struct smm_order_ring_cursor:
        smm_order_ring* ring
        size_t          slot
        size_t          remaining
        uint64_t        generation

    ctypedef struct smm_listener:
        smm_match_callback callback
        void*              user_data
        uintptr_t          id
        smm_listener*      next

    ctypedef struct smm_match_ctx:
        double         timestamp
        double         last_price
        uint64_t       last_transaction_count
        smm_order_ring orders
        double         fee_rate
        double         lag_ts
        uint64_t       lag_n_transaction
        double         hit_prob
        double         slippery_rate
        uint64_t       seed
        uint64_t       rng_state
        smm_listener*  listeners
        md_variant*    ws_report
        c_bool         instant_fill
        c_bool         incremental_order_volume

    # --- Lifecycle ---
    smm_match_ctx* c_smm_match_new(double fee_rate, c_bool instant_fill, double lag_ts, uint64_t lag_n_transaction, double hit_prob, double slippery_rate, c_bool incremental_order_volume, allocator_protocol* allocator) noexcept nogil
    int            c_smm_match_init(smm_match_ctx* ctx) noexcept nogil
    void           c_smm_match_dealloc(smm_match_ctx* ctx) noexcept nogil
    void           c_smm_match_free(smm_match_ctx* ctx) noexcept nogil
    void           c_smm_match_clear(smm_match_ctx* ctx) noexcept nogil

    # --- Configuration ---
    int            c_smm_match_set_config(smm_match_ctx* ctx, double fee_rate, c_bool instant_fill, double lag_ts, uint64_t lag_n_transaction, double hit_prob, double slippery_rate, c_bool incremental_order_volume) noexcept nogil
    int            c_smm_match_set_seed(smm_match_ctx* ctx, uint64_t seed) noexcept nogil

    # --- Listeners ---
    int            c_smm_match_register_listener(smm_match_ctx* ctx, smm_match_callback callback, void* user_data, uintptr_t* out_id) noexcept nogil
    int            c_smm_match_deregister_listener(smm_match_ctx* ctx, uintptr_t listener_id) noexcept nogil

    # --- Queries ---
    int            c_smm_match_best_price(const double* prices, size_t n, int8_t sign, double* out) noexcept nogil
    int            c_smm_match_worst_price(const double* prices, size_t n, int8_t sign, double* out) noexcept nogil

    # --- Trading ---
    int            c_smm_match_launch(smm_match_ctx* ctx, const md_trade_instruction* order) noexcept nogil
    int            c_smm_match_cancel(smm_match_ctx* ctx, const long_md_id* order_id) noexcept nogil
    void           c_smm_match_eod(smm_match_ctx* ctx) noexcept nogil

    # --- Hot path ---
    int            c_smm_match_process(smm_match_ctx* ctx, const md_variant* market_data) noexcept nogil

    # --- Ring ---
    void                  c_smm_id_key(const long_md_id* id, char* out) noexcept nogil
    size_t                c_smm_ring_mask(const smm_order_ring* ring) noexcept nogil
    int                   c_smm_ring_init(smm_order_ring* ring, allocator_protocol* allocator) noexcept nogil
    void                  c_smm_ring_dealloc(smm_order_ring* ring) noexcept nogil
    int                   c_smm_ring_rebuild(smm_order_ring* ring, allocator_protocol* allocator, size_t new_capacity) noexcept nogil
    void                  c_smm_ring_clear(smm_order_ring* ring) noexcept nogil
    smm_order_entry*      c_smm_ring_find(smm_order_ring* ring, const long_md_id* order_id) noexcept nogil
    smm_order_entry*      c_smm_ring_push(smm_order_ring* ring, const md_trade_instruction* order) noexcept nogil
    void                  c_smm_ring_remove(smm_order_ring* ring, smm_order_entry* entry) noexcept nogil
    smm_order_ring_cursor c_smm_ring_cursor_newest(smm_order_ring* ring) noexcept nogil
    smm_order_entry*      c_smm_ring_next_newest(smm_order_ring_cursor* cursor) noexcept nogil
    smm_order_entry*      c_smm_ring_newest(smm_order_ring* ring) noexcept nogil

    # --- Matching internals ---
    void c_smm_check_bar(smm_match_ctx* ctx, const md_candlestick* bar) noexcept nogil
    void c_smm_check_tick(smm_match_ctx* ctx, const md_tick_data* tick) noexcept nogil
    void c_smm_check_tick_lite(smm_match_ctx* ctx, const md_tick_data_lite* lite) noexcept nogil
    void c_smm_check_order(smm_match_ctx* ctx, const md_order_data* order_data) noexcept nogil
    void c_smm_check_trade(smm_match_ctx* ctx, const md_transaction_data* transaction) noexcept nogil


cdef class SimMatch(CCPType):
    cdef smm_match_ctx* header
    cdef bint owner

    cdef uintptr_t callback_id
    cdef bint registered
    cdef readonly str ticker

    cdef readonly EventEngineEx event_engine
    cdef readonly TopicSetEx topic_set
    cdef readonly DMAProtocol protocol
    cdef readonly dict working
    cdef readonly dict history
    cdef readonly object random
    cdef readonly uint64_t seed

    @staticmethod
    cdef inline void c_simmatch_callback(smm_match_ctx* ctx, smm_match_event event, const smm_order_entry* entry, const md_trade_report* report, void* user_data) noexcept

    @staticmethod
    cdef inline double c_price_extreme(tuple prices, int8_t sign, bint select_best, object side) except *

    cdef inline int c_process(self, const md_variant* market_data) except -1

    cdef void c_unregister_protocol(self, DMAProtocol protocol) except *

from cpython.datetime cimport time as py_time
from libc.stdint cimport uint8_t

from cbase.allocator_protocol cimport allocator_protocol

from algo_engine.base.c_market_data.c_market_data cimport md_side
from algo_engine.exchange_profile.c_exchange_profile cimport session_time_t

from .c_risk_profile cimport RiskRuleTemplate, dma, dma_risk_profile, dma_risk_rule_ctx, dma_risk_type, dma_risk_violation


cdef extern from "algo_engine/engine/risk_profile/c_refusal_window.h":
    # === Constants and enums ===
    const int REFUSAL_WINDOW_RULE_CODE
    const char* REFUSAL_WINDOW_RULE_NAME

    ctypedef enum refusal_window_bound:
        REFUSAL_WINDOW_BOUND_START
        REFUSAL_WINDOW_BOUND_END

    ctypedef enum refusal_window_ret_code:
        REFUSAL_WINDOW_OK
        REFUSAL_WINDOW_ERR_INVALID_INPUT
        REFUSAL_WINDOW_ERR_OOM

    # === Structs ===
    ctypedef struct refusal_window:
        session_time_t start
        session_time_t end
        uint8_t        enabled

    # === Lifecycle ===
    refusal_window*            c_refusal_window_new(session_time_t* start, session_time_t* end, allocator_protocol* allocator) noexcept nogil
    int                c_refusal_window_init(refusal_window* window, session_time_t* start, session_time_t* end) noexcept nogil
    void               c_refusal_window_dealloc(refusal_window* window) noexcept nogil
    void               c_refusal_window_free(refusal_window* window) noexcept nogil

    # === Registration ===
    dma_risk_rule_ctx* c_refusal_window_register(dma_risk_profile* risk_profile, refusal_window* window, const char* ticker) noexcept nogil
    dma_risk_rule_ctx* c_refusal_window_register_global(dma_risk_profile* risk_profile, refusal_window* window) noexcept nogil

    # === Check ===
    dma_risk_type      c_refusal_window_check(const dma_risk_rule_ctx* rule, const dma* DMA, const dma_risk_profile* risk_profile, const char* ticker, md_side side, double volume, double price, void* user_data, dma_risk_violation* out) noexcept nogil


cdef class RefusalWindowRule(RiskRuleTemplate):
    cdef readonly py_time start
    cdef readonly py_time end


cdef class RefusalWindowBeforeRule(RefusalWindowRule):
    pass


cdef class RefusalWindowAfterRule(RefusalWindowRule):
    pass


cdef class RefusalWindowBetweenRule(RefusalWindowRule):
    pass

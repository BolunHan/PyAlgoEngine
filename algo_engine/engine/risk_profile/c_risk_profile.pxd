from libc.stdint cimport uint32_t, uintptr_t
from libcpp cimport bool as c_bool

from cbase.allocator_protocol cimport CCPType, allocator_protocol, ccp_ctx
from cbase.bytemap cimport BoundByteMap, bytemap

from algo_engine.base.c_market_data.c_market_data cimport md_side, md_trade_instruction
from algo_engine.base.c_market_data.c_trade_utils cimport TradeInstruction
from algo_engine.engine.algo.c_algo_template cimport dma_algo
from algo_engine.engine.c_dma cimport dma


cdef extern from "algo_engine/engine/risk_profile/c_risk_profile.h":
    # === Enums and constants ===
    const size_t DMA_RISK_MESSAGE_SIZE

    ctypedef enum dma_risk_ret_code:
        DMA_RISK_OK_RET
        DMA_RISK_ERR_INVALID_INPUT
        DMA_RISK_ERR_OOM
        DMA_RISK_ERR_UNKNOWN_RULE
        DMA_RISK_ERR_RULE_SCOPE
        DMA_RISK_ERR_NOT_FOUND
        DMA_RISK_ERR_CONFLICT

    ctypedef enum dma_risk_rule:
        DMA_RULE_MAX_PERCENTILE
        DMA_RULE_MAX_TRADE_LONG
        DMA_RULE_MAX_TRADE_SHORT
        DMA_RULE_MAX_EXPOSURE_LONG
        DMA_RULE_MAX_EXPOSURE_SHORT
        DMA_RULE_MAX_NOTIONAL_LONG
        DMA_RULE_MAX_NOTIONAL_SHORT
        DMA_RULE_MAX_TTL_NOTIONAL_LONG
        DMA_RULE_MAX_TTL_NOTIONAL_SHORT
        DMA_RULE_MAX_NET_NOTIONAL_LONG
        DMA_RULE_MAX_NET_NOTIONAL_SHORT

    ctypedef enum dma_risk_type:
        DMA_RISK_INVALID
        DMA_RISK_PASS
        DMA_RISK_INCOMPLETE
        DMA_RISK_PRICE
        DMA_RISK_TRADE_INVALID
        DMA_RISK_TRADE_LONG
        DMA_RISK_TRADE_SHORT
        DMA_RISK_EXPOSURE_LONG
        DMA_RISK_EXPOSURE_SHORT
        DMA_RISK_PERCENTILE_LONG
        DMA_RISK_PERCENTILE_SHORT
        DMA_RISK_NOTIONAL_LONG
        DMA_RISK_NOTIONAL_SHORT
        DMA_RISK_PORTFOLIO_NET_LONG
        DMA_RISK_PORTFOLIO_NET_SHORT
        DMA_RISK_PORTFOLIO_TOTAL_LONG
        DMA_RISK_PORTFOLIO_TOTAL_SHORT
        DMA_RISK_TICKER_BLOCKED
        DMA_RISK_TICKER_UNLISTED
        DMA_RISK_CUSTOM

    # === Structs ===

    ctypedef struct dma_risk_limits_globals:
        double max_ttl_notional_long
        double max_ttl_notional_short
        double max_net_notional_long
        double max_net_notional_short
        uint32_t enabled
        dma_risk_rule_ctx* rules

    ctypedef struct dma_risk_profile:
        bytemap*                entries
        bytemap*                whitelist
        bytemap*                blacklist
        dma_risk_limits_globals globals

    ctypedef struct dma_risk_violation:
        dma_risk_type type
        int           code
        char          msg[DMA_RISK_MESSAGE_SIZE]

    ctypedef dma_risk_type (*dma_risk_rule_check_func)(const dma_risk_rule_ctx* rule, const dma* DMA, const dma_risk_profile* risk_profile, const char* ticker, md_side side, double volume, double price, void* user_data, dma_risk_violation* out) noexcept

    ctypedef struct dma_risk_limits:
        double             max_percentile
        double             max_trade_long
        double             max_trade_short
        double             max_exposure_long
        double             max_exposure_short
        double             max_notional_long
        double             max_notional_short
        uint32_t           enabled
        dma_risk_rule_ctx* rules

    ctypedef struct dma_risk_rule_ctx:
        dma_risk_rule_check_func  check_fn
        void*                     user_data
        dma_risk_rule_ctx*        next
        const char*               ticker
        const char*               name
        uintptr_t                 rule_id
        int                       risk_rule_code
        c_bool                    enabled

    ctypedef struct dma_risk_check_state:
        const char* ticker
        double      volume
        double      market_price
        double      working_long
        double      working_short
        double      traded_long
        double      traded_short
        double      exposure_long
        double      exposure_short
        double      portfolio_exposure_notional
        double      portfolio_working_notional
        double      portfolio_exposure_notional_abs
        double      ttl_working_notional_long
        double      ttl_working_notional_short
        double      ttl_exposure_notional_long
        double      ttl_exposure_notional_short
        int         side_sign

    # === Rule tables ===
    const char*           c_dma_risk_rule_name(dma_risk_rule rule) noexcept nogil
    c_bool                c_dma_risk_rule_is_global(dma_risk_rule rule) noexcept nogil
    uint32_t              c_dma_risk_rule_bit(dma_risk_rule rule) noexcept nogil
    const char*           c_dma_risk_type_name(dma_risk_type type) noexcept nogil
    int                   c_dma_risk_violation_code(dma_risk_type type) noexcept nogil

    # === Lifecycle ===
    dma_risk_profile*             c_dma_risk_profile_new(allocator_protocol* allocator) noexcept nogil
    int                   c_dma_risk_profile_init(dma_risk_profile* risk_profile, allocator_protocol* allocator) noexcept nogil
    void                  c_dma_risk_profile_dealloc(dma_risk_profile* risk) noexcept nogil
    void                  c_dma_risk_profile_free(dma_risk_profile* risk) noexcept nogil
    int                   c_dma_risk_profile_clear(dma_risk_profile* risk) noexcept nogil

    # === Ticker gates ===
    int                   c_dma_risk_profile_whitelist_add(dma_risk_profile* risk_profile, const char* ticker) noexcept nogil
    int                   c_dma_risk_profile_whitelist_remove(dma_risk_profile* risk_profile, const char* ticker) noexcept nogil
    int                   c_dma_risk_profile_blacklist_add(dma_risk_profile* risk_profile, const char* ticker) noexcept nogil
    int                   c_dma_risk_profile_blacklist_remove(dma_risk_profile* risk_profile, const char* ticker) noexcept nogil
    c_bool                c_dma_risk_profile_is_ticker_allowed(const dma_risk_profile* risk_profile, const char* ticker) noexcept nogil

    # === Limits and queries ===
    int                   c_dma_risk_profile_set_limit(dma_risk_profile* risk_profile, const char* ticker, dma_risk_rule rule, double value) noexcept nogil
    c_bool                c_dma_risk_profile_get_limit(const dma_risk_profile* risk_profile, const char* ticker, dma_risk_rule rule, double* out) noexcept nogil
    dma_risk_limits*      c_dma_risk_profile_get_limits(const dma_risk_profile* risk_profile, const char* ticker) noexcept nogil

    # === Custom rules ===
    dma_risk_rule_ctx*    c_dma_risk_profile_register_rule(dma_risk_profile* risk_profile, const char* ticker, dma_risk_rule_check_func check_fn, int risk_rule_code, const char* name, void* user_data) noexcept nogil
    dma_risk_rule_ctx*    c_dma_risk_profile_register_global_rule(dma_risk_profile* risk_profile, dma_risk_rule_check_func check_fn, int risk_rule_code, const char* name, void* user_data) noexcept nogil
    int                   c_dma_risk_profile_unregister_rule(dma_risk_profile* risk_profile, const char* ticker, uintptr_t rule_id) noexcept nogil
    int                   c_dma_risk_profile_unregister_global_rule(dma_risk_profile* risk_profile, uintptr_t rule_id) noexcept nogil
    int                   c_dma_risk_profile_enable_rule(dma_risk_profile* risk_profile, const char* ticker, uintptr_t rule_id) noexcept nogil
    int                   c_dma_risk_profile_disable_rule(dma_risk_profile* risk_profile, const char* ticker, uintptr_t rule_id) noexcept nogil
    int                   c_dma_risk_profile_enable_global_rule(dma_risk_profile* risk_profile, uintptr_t rule_id) noexcept nogil
    int                   c_dma_risk_profile_disable_global_rule(dma_risk_profile* risk_profile, uintptr_t rule_id) noexcept nogil

    # === Checks ===
    dma_risk_type         c_dma_risk_profile_check(const dma_risk_profile* risk_profile, const dma_risk_check_state* query, dma_risk_violation* out) noexcept nogil
    dma_risk_violation    c_dma_risk_profile_check_algo(const dma* DMA, const dma_risk_profile* risk_profile, const dma_algo* algo) noexcept nogil
    dma_risk_violation    c_dma_risk_profile_check_order(const dma* DMA, const dma_risk_profile* risk_profile, const md_trade_instruction* order) noexcept nogil
    dma_risk_violation    c_dma_risk_profile_check_volume(const dma* DMA, const dma_risk_profile* risk_profile, const char* ticker, md_side side, double volume) noexcept nogil
    void                  c_dma_risk_profile_check_state_init(const dma* DMA, const char* ticker, double price, dma_risk_check_state* check_state) noexcept nogil
    dma_risk_violation    c_dma_risk_profile_check_limit(const dma* DMA, const dma_risk_profile* risk_profile, const char* ticker, md_side side, double volume, double price) noexcept nogil


cdef class RiskViolation(Exception):
    cdef ccp_ctx ccp_ctx
    cdef const dma_risk_violation* header
    cdef bint owner

    cdef dict __dict__

    cdef inline void ccp_attach(self)

    cdef inline void ccp_detach(self)

    @staticmethod
    cdef RiskViolation c_from_header(const dma_risk_violation* header, bint owner=?)

    @staticmethod
    cdef RiskViolation c_from_value(dma_risk_violation value)


cdef class RiskLimits(CCPType):
    cdef dma_risk_limits* header

    @staticmethod
    cdef RiskLimits c_from_header(dma_risk_limits* header)


cdef class RiskLimitsMapView(BoundByteMap):
    @staticmethod
    cdef RiskLimitsMapView c_from_header(bytemap* header, bint owner=?)


cdef class RiskProfile(CCPType):
    cdef dma_risk_profile* header
    cdef bint owner

    cdef dict registered_rules

    cdef RiskLimitsMapView limits_map

    cpdef void set_rules(self, dict rules)

    cdef void c_set_limit(self, dma_risk_rule rule, str ticker, double value)

    cdef dict c_snapshot_rules(self)

    cdef bint c_report_violation(self, const dma_risk_violation* violation)

    cpdef void set_rule(self, object key, double value, str ticker=?)

    cpdef void set_whitelist(self, str ticker, bint enabled=?)

    cpdef void set_blacklist(self, str ticker, bint enabled=?)

    cpdef dict get(self, str ticker)

    cpdef void clear(self)

    cpdef bint check(self, TradeInstruction order)

    cpdef bint check_order(self, str ticker, double volume, md_side side)


cdef class RiskRuleTemplate(CCPType):
    cdef dma_risk_rule_ctx* risk_rule_ctx
    cdef RiskProfile profile
    cdef readonly str name
    cdef readonly int code

    @staticmethod
    cdef dma_risk_type c_risk_rule_check_adaptor(const dma_risk_rule_ctx* rule, const dma* DMA, const dma_risk_profile* risk_profile, const char* ticker, md_side side, double volume, double price, void* user_data, dma_risk_violation* out) noexcept

    @staticmethod
    cdef RiskRuleTemplate c_from_header(dma_risk_rule_ctx* risk_rule_ctx, RiskProfile parent)

    cdef dma_risk_type c_check(self, const dma_risk_rule_ctx* rule, const dma* DMA, const dma_risk_profile* risk_profile, const char* ticker, md_side side, double volume, double price, dma_risk_violation* out)

    cdef void c_register(self, RiskProfile risk_profile, str ticker)

    cdef void c_register_global(self, RiskProfile risk_profile)

    cdef void c_unregister(self)

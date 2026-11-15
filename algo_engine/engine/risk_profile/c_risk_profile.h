#ifndef C_RISK_PROFILE_H
#define C_RISK_PROFILE_H

#include <math.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <algo_engine/base/c_market_data/c_market_data.h>
#include <cbase/allocator_protocol/c_allocator_protocol.h>
#include <cbase/bytemap/c_bytemap.h>

/** @brief Capacity of the violation message buffer, including the terminator. */
#ifndef DMA_RISK_MESSAGE_SIZE
#define DMA_RISK_MESSAGE_SIZE 512
#endif

// The access and algo records are defined by headers that include this one;
// the access-aware checks only hold pointers to them.
typedef struct dma               dma;
typedef struct dma_algo          dma_algo;

// ========== Enums ==========

// clang-format off

/** @enum dma_risk_ret_code
 *  @brief 0 is success, negatives are errors. */
typedef enum dma_risk_ret_code {
    DMA_RISK_OK_RET                   =  0,
    DMA_RISK_ERR_INVALID_INPUT        = -1,
    DMA_RISK_ERR_OOM                  = -2,
    DMA_RISK_ERR_UNKNOWN_RULE         = -3,
    DMA_RISK_ERR_RULE_SCOPE           = -4,  // A ticker rule was set globally, or a global rule per ticker.
    DMA_RISK_ERR_NOT_FOUND            = -5,  // No custom rule is registered under the id.
    DMA_RISK_ERR_CONFLICT             = -6,  // The ticker is on the other gate list; the lists stay disjoint.
} dma_risk_ret_code;

/**
 * @enum dma_risk_rule
 * @brief The risk rules a profile carries.
 *
 * Rules 0-6 are per ticker; rules 7-10 are portfolio-wide. A rule is defined
 * when its value is not NaN; a side is checked only against the rules defined
 * for it.
 */
typedef enum dma_risk_rule {
    DMA_RULE_MAX_PERCENTILE           =  0,
    DMA_RULE_MAX_TRADE_LONG           =  1,
    DMA_RULE_MAX_TRADE_SHORT          =  2,
    DMA_RULE_MAX_EXPOSURE_LONG        =  3,
    DMA_RULE_MAX_EXPOSURE_SHORT       =  4,
    DMA_RULE_MAX_NOTIONAL_LONG        =  5,
    DMA_RULE_MAX_NOTIONAL_SHORT       =  6,
    DMA_RULE_MAX_TTL_NOTIONAL_LONG    =  7,
    DMA_RULE_MAX_TTL_NOTIONAL_SHORT   =  8,
    DMA_RULE_MAX_NET_NOTIONAL_LONG    =  9,
    DMA_RULE_MAX_NET_NOTIONAL_SHORT   = 10,
} dma_risk_rule;

/**
 * @enum dma_risk_type
 * @brief The check that rejected an order; `c_dma_risk_type_name` and
 *        `c_dma_risk_violation_code` carry its name/code pair.
 */
typedef enum dma_risk_type {
    DMA_RISK_INVALID                  = -1,
    DMA_RISK_PASS                     =  0,
    DMA_RISK_INCOMPLETE               =  1,
    DMA_RISK_PRICE                    =  2,
    DMA_RISK_TRADE_INVALID            =  3,
    DMA_RISK_TRADE_LONG               =  4,
    DMA_RISK_TRADE_SHORT              =  5,
    DMA_RISK_EXPOSURE_LONG            =  6,
    DMA_RISK_EXPOSURE_SHORT           =  7,
    DMA_RISK_PERCENTILE_LONG          =  8,
    DMA_RISK_PERCENTILE_SHORT         =  9,
    DMA_RISK_NOTIONAL_LONG            = 10,
    DMA_RISK_NOTIONAL_SHORT           = 11,
    DMA_RISK_PORTFOLIO_NET_LONG       = 12,
    DMA_RISK_PORTFOLIO_NET_SHORT      = 13,
    DMA_RISK_PORTFOLIO_TOTAL_LONG     = 14,
    DMA_RISK_PORTFOLIO_TOTAL_SHORT    = 15,
    DMA_RISK_TICKER_BLOCKED           = 16,
    DMA_RISK_TICKER_UNLISTED          = 17,
    DMA_RISK_CUSTOM                   = 18,  // A custom rule rejected; the violation code names the rule.
} dma_risk_type;

// ========== Tables ==========

/** @brief The rule keys, indexed by `dma_risk_rule`. */
static const char* const DMA_RISK_RULE_NAMES[] = {
    [DMA_RULE_MAX_PERCENTILE]         = "max_percentile",
    [DMA_RULE_MAX_TRADE_LONG]         = "max_trade_long",
    [DMA_RULE_MAX_TRADE_SHORT]        = "max_trade_short",
    [DMA_RULE_MAX_EXPOSURE_LONG]      = "max_exposure_long",
    [DMA_RULE_MAX_EXPOSURE_SHORT]     = "max_exposure_short",
    [DMA_RULE_MAX_NOTIONAL_LONG]      = "max_notional_long",
    [DMA_RULE_MAX_NOTIONAL_SHORT]     = "max_notional_short",
    [DMA_RULE_MAX_TTL_NOTIONAL_LONG]  = "max_ttl_notional_long",
    [DMA_RULE_MAX_TTL_NOTIONAL_SHORT] = "max_ttl_notional_short",
    [DMA_RULE_MAX_NET_NOTIONAL_LONG]  = "max_net_notional_long",
    [DMA_RULE_MAX_NET_NOTIONAL_SHORT] = "max_net_notional_short",
};

/** @brief The dotted name of each violation, indexed by `dma_risk_type`. */
static const char* const DMA_RISK_TYPE_NAMES[] = {
    [DMA_RISK_PASS]                  = "RiskProfile.Pass",
    [DMA_RISK_INCOMPLETE]            = "RiskProfile.Incomplete",
    [DMA_RISK_PRICE]                 = "RiskProfile.Internal.Price",
    [DMA_RISK_TRADE_INVALID]         = "RiskProfile.TradeLimit.Invalid",
    [DMA_RISK_TRADE_LONG]            = "RiskProfile.TradeLimit.Long",
    [DMA_RISK_TRADE_SHORT]           = "RiskProfile.TradeLimit.Short",
    [DMA_RISK_EXPOSURE_LONG]         = "RiskProfile.ExposureLimit.Long",
    [DMA_RISK_EXPOSURE_SHORT]        = "RiskProfile.ExposureLimit.Short",
    [DMA_RISK_PERCENTILE_LONG]       = "RiskProfile.PercentileLimit.Long",
    [DMA_RISK_PERCENTILE_SHORT]      = "RiskProfile.PercentileLimit.Short",
    [DMA_RISK_NOTIONAL_LONG]         = "RiskProfile.NotionalLimit.Long",
    [DMA_RISK_NOTIONAL_SHORT]        = "RiskProfile.NotionalLimit.Short",
    [DMA_RISK_PORTFOLIO_NET_LONG]    = "RiskProfile.NotionalLimit.PortfolioNet.Long",
    [DMA_RISK_PORTFOLIO_NET_SHORT]   = "RiskProfile.NotionalLimit.PortfolioNet.Short",
    [DMA_RISK_PORTFOLIO_TOTAL_LONG]  = "RiskProfile.NotionalLimit.PortfolioTotal.Long",
    [DMA_RISK_PORTFOLIO_TOTAL_SHORT] = "RiskProfile.NotionalLimit.PortfolioTotal.Short",
    [DMA_RISK_TICKER_BLOCKED]        = "RiskProfile.TickerLimit.Blocked",
    [DMA_RISK_TICKER_UNLISTED]       = "RiskProfile.TickerLimit.Unlisted",
    [DMA_RISK_CUSTOM]                = "RiskProfile.Custom",
};

/** @brief The numeric code of each violation, indexed by `dma_risk_type`. */
static const int DMA_RISK_VIOLATION_CODES[] = {
    [DMA_RISK_PASS]                  = 0,
    [DMA_RISK_INCOMPLETE]            = 1000,
    [DMA_RISK_PRICE]                 = 100,
    [DMA_RISK_TRADE_INVALID]         = 1003,
    [DMA_RISK_TRADE_LONG]            = 1001,
    [DMA_RISK_TRADE_SHORT]           = 1002,
    [DMA_RISK_EXPOSURE_LONG]         = 2001,
    [DMA_RISK_EXPOSURE_SHORT]        = 2002,
    [DMA_RISK_PERCENTILE_LONG]       = 3001,
    [DMA_RISK_PERCENTILE_SHORT]      = 3002,
    [DMA_RISK_NOTIONAL_LONG]         = 4001,
    [DMA_RISK_NOTIONAL_SHORT]        = 4002,
    [DMA_RISK_PORTFOLIO_NET_LONG]    = 5001,
    [DMA_RISK_PORTFOLIO_NET_SHORT]   = 5002,
    [DMA_RISK_PORTFOLIO_TOTAL_LONG]  = 5003,
    [DMA_RISK_PORTFOLIO_TOTAL_SHORT] = 5004,
    [DMA_RISK_TICKER_BLOCKED]        = 1004,
    [DMA_RISK_TICKER_UNLISTED]       = 1005,
    [DMA_RISK_CUSTOM]                = 0,  // The rule reports its own code.
};

// clang-format on

// ========== Structs ==========

typedef struct dma_risk_rule_ctx dma_risk_rule_ctx;
typedef struct dma_risk_profile  dma_risk_profile;

/**
 * @struct dma_risk_limits_globals
 * @brief The portfolio-wide rule values and the global custom rule chain.
 *
 * A rule is enabled when its bit in `enabled` is set; a disabled rule stores
 * 0 and the check skips it. The allocator returns the block zeroed, so a fresh
 * profile enables nothing. A global custom rule applies to every ticker and
 * runs after the ticker's own custom rules.
 */
typedef struct dma_risk_limits_globals {
    double             max_ttl_notional_long;   // Total notional limit, long side; 0 when disabled.
    double             max_ttl_notional_short;  // Total notional limit, short side; 0 when disabled.
    double             max_net_notional_long;   // Net notional limit, long side; 0 when disabled.
    double             max_net_notional_short;  // Net notional limit, short side; 0 when disabled.
    uint32_t           enabled;                 // One bit per dma_risk_rule; a clear bit disables the rule.
    dma_risk_rule_ctx* rules;                   // The global custom rule chain; the profile owns the nodes.
} dma_risk_limits_globals;

/**
 * @struct dma_risk_profile
 * @brief The risk profile rules: one limit set per ticker, the global limits
 *        and the two ticker gates.
 *
 * A check runs the rules enabled for the ticker and the side; a side with no
 * enabled absolute risk value for it is refused as incomplete. The profile is
 * a 72-byte block, so the gates, the entries map and the globals it reads span
 * at most two cache lines.
 *
 * Either gate list may be absent. A ticker on the `blacklist` is refused; when
 * the `whitelist` exists, only a ticker it lists may trade. A ticker cannot be
 * on both lists: the add refuses the conflict.
 */
typedef struct dma_risk_profile {
    bytemap*                entries;    // OWNED - ticker -> dma_risk_limits*; each limit set is its own block.
    bytemap*                whitelist;  // OWNED when present - ticker -> uint8_t; NULL when no whitelist gate.
    bytemap*                blacklist;  // OWNED when present - ticker -> uint8_t; NULL when no blacklist gate.
    dma_risk_limits_globals globals;    // Portfolio-wide limits and the global custom rule chain.
} dma_risk_profile;

/**
 * @struct dma_risk_violation
 * @brief The rejection a failed check reports.
 *
 * `msg` carries the human-readable reason, including the disabled rules when
 * the profile cannot constrain the side. A passing check leaves the type PASS,
 * the code 0 and the message empty: the type/code header leads the block and
 * the 512-byte message is written only on a rejection.
 */
typedef struct dma_risk_violation {
    dma_risk_type type;
    int           code;
    char          msg[DMA_RISK_MESSAGE_SIZE];
} dma_risk_violation;

/**
 * @brief Invoked for one custom rule, after the standard rules pass.
 *
 * The arguments between `rule` and `user_data` are the ones
 * `c_dma_risk_profile_check_limit` receives. `user_data` is the rule's state,
 * set at registration. On a rejection the rule fills `out` and returns the
 * violation type; a passing rule returns DMA_RISK_PASS and leaves `out`
 * untouched.
 */
typedef dma_risk_type (*dma_risk_rule_check_func)(const dma_risk_rule_ctx* rule, const dma* DMA, const dma_risk_profile* risk_profile, const char* ticker, md_side side, double volume, double price, void* user_data, dma_risk_violation* out);

/**
 * @struct dma_risk_rule_ctx
 * @brief One custom rule of a ticker; owned by the ticker's limit set.
 *
 * The fields follow the chain walk: the check, its state and the next link
 * lead, the ticker scope, the name and the identity follow, and the flags word
 * closes the block. The node is 56 bytes, so a 16-byte-aligned node spans two
 * cache lines only when it starts 56 bytes into a line.
 *
 * The profile composes the node in `c_dma_risk_profile_register_rule` under
 * the ticker's limit set and frees it on unregister, clear and dealloc. The
 * ticker and the name are copies owned by the node; the ticker is NULL for a
 * global rule. `user_data` is the rule's state: a rule that parents its state
 * under the node (`c_ap_acquire_ownership`) gives the node its lifetime too. A
 * disabled rule stays linked but the check skips it.
 */
typedef struct dma_risk_rule_ctx {
    dma_risk_rule_check_func  check_fn;        // NOT owned - the rule; invoked while the rule is enabled.
    void*                     user_data;       // The rule's state; adoption under the node frees it with the rule.
    struct dma_risk_rule_ctx* next;            // The next rule in the chain.
    const char*               ticker;          // The ticker scope; a copy owned by the node, NULL for a global rule.
    const char*               name;            // The rule name reported in a message; a copy owned by the node.
    uintptr_t                 rule_id;         // The node address.
    int                       risk_rule_code;  // The code a rejection reports.
    bool                      enabled;         // A disabled rule stays linked and is skipped.
} dma_risk_rule_ctx;

_Static_assert(sizeof(dma_risk_rule_ctx) <= 64, "dma_risk_rule_ctx must fit a cache line");

/**
 * @struct dma_risk_limits
 * @brief One ticker's rule values and its custom rule chain.
 *
 * A rule is enabled when its bit in `enabled` is set: a disabled rule stores 0
 * and the check skips it. A side with no enabled absolute risk value is
 * refused as incomplete.
 *
 * The profile hosts each limit set by pointer in its entries map. The block is
 * its own allocation, so a `dma_risk_limits*` stays valid while the profile
 * lives, even when the map grows or rehashes. The set is 72 bytes, so it spans
 * at most two cache lines, and a check reads it front to back.
 */
typedef struct dma_risk_limits {
    double             max_percentile;      // Maximum share of the portfolio, in [0, 1); 0 when disabled.
    double             max_trade_long;      // Maximum traded + working long volume; 0 when disabled.
    double             max_trade_short;     // Maximum traded + working short volume; 0 when disabled.
    double             max_exposure_long;   // Maximum net long exposure volume; 0 when disabled.
    double             max_exposure_short;  // Maximum net short exposure volume; 0 when disabled.
    double             max_notional_long;   // Maximum net long exposure notional; 0 when disabled.
    double             max_notional_short;  // Maximum net short exposure notional; 0 when disabled.
    uint32_t           enabled;             // One bit per dma_risk_rule; a clear bit disables the rule.
    dma_risk_rule_ctx* rules;               // The ticker's custom rule chain; the nodes are children of this block.
} dma_risk_limits;

/**
 * @struct dma_risk_check_state
 * @brief The market and portfolio state a risk check runs against.
 *
 * `exposure_long - exposure_short` is the ticker's signed net exposure volume;
 * the checks read their difference, never one side alone. The record is built
 * on the stack once per check and read front to back; the int closes it after
 * the doubles, so the block stays hole-free.
 */
typedef struct dma_risk_check_state {
    const char* ticker;        // The order's ticker (borrowed).
    double      volume;        // The order volume; the sign comes from side_sign.
    double      market_price;  // NAN when unavailable.

    double      working_long;    // Working long volume of the ticker.
    double      working_short;   // Working short volume of the ticker.
    double      traded_long;     // Traded long volume of the ticker today.
    double      traded_short;    // Traded short volume of the ticker today.
    double      exposure_long;   // Positive part of the ticker's net exposure.
    double      exposure_short;  // Absolute negative part of the ticker's net exposure.

    double      portfolio_exposure_notional;      // Signed sum of exposure notional.
    double      portfolio_working_notional;       // Sum of working notional.
    double      portfolio_exposure_notional_abs;  // Absolute sum of exposure notional.

    double      ttl_working_notional_long;    // Total working notional, long side.
    double      ttl_working_notional_short;   // Total working notional, short side.
    double      ttl_exposure_notional_long;   // Total exposure notional, long side.
    double      ttl_exposure_notional_short;  // Total exposure notional, short side.
    int         side_sign;                    // -1 short, 0 neutral, 1 long.
} dma_risk_check_state;

// ========== Forward Declarations ==========

// Rule tables
static inline const char*        c_dma_risk_rule_name(dma_risk_rule rule);
static inline bool               c_dma_risk_rule_is_global(dma_risk_rule rule);
static inline uint32_t           c_dma_risk_rule_bit(dma_risk_rule rule);
static inline const char*        c_dma_risk_type_name(dma_risk_type type);
static inline int                c_dma_risk_violation_code(dma_risk_type type);

// Lifecycle
static inline dma_risk_profile*  c_dma_risk_profile_new(allocator_protocol* allocator);
static inline int                c_dma_risk_profile_init(dma_risk_profile* risk_profile, allocator_protocol* allocator);
static inline void               c_dma_risk_profile_dealloc(dma_risk_profile* risk_profile);
static inline void               c_dma_risk_profile_free(dma_risk_profile* risk_profile);
static inline int                c_dma_risk_profile_clear(dma_risk_profile* risk_profile);

// Ticker gates
static inline int                c_dma_risk_profile_whitelist_add(dma_risk_profile* risk_profile, const char* ticker);
static inline int                c_dma_risk_profile_whitelist_remove(dma_risk_profile* risk_profile, const char* ticker);
static inline int                c_dma_risk_profile_blacklist_add(dma_risk_profile* risk_profile, const char* ticker);
static inline int                c_dma_risk_profile_blacklist_remove(dma_risk_profile* risk_profile, const char* ticker);
static inline bool               c_dma_risk_profile_is_ticker_allowed(const dma_risk_profile* risk_profile, const char* ticker);

// Limits and queries
static inline int                c_dma_risk_profile_set_limit(dma_risk_profile* risk_profile, const char* ticker, dma_risk_rule rule, double value);
static inline bool               c_dma_risk_profile_get_limit(const dma_risk_profile* risk_profile, const char* ticker, dma_risk_rule rule, double* out);
static inline dma_risk_limits*   c_dma_risk_profile_get_limits(const dma_risk_profile* risk_profile, const char* ticker);

// Custom rules
static inline dma_risk_rule_ctx* c_dma_risk_profile_register_rule(dma_risk_profile* risk_profile, const char* ticker, dma_risk_rule_check_func check_fn, int risk_rule_code, const char* name, void* user_data);
static inline dma_risk_rule_ctx* c_dma_risk_profile_register_global_rule(dma_risk_profile* risk_profile, dma_risk_rule_check_func check_fn, int risk_rule_code, const char* name, void* user_data);
static inline int                c_dma_risk_profile_unregister_rule(dma_risk_profile* risk_profile, const char* ticker, uintptr_t rule_id);
static inline int                c_dma_risk_profile_unregister_global_rule(dma_risk_profile* risk_profile, uintptr_t rule_id);
static inline int                c_dma_risk_profile_enable_rule(dma_risk_profile* risk_profile, const char* ticker, uintptr_t rule_id);
static inline int                c_dma_risk_profile_disable_rule(dma_risk_profile* risk_profile, const char* ticker, uintptr_t rule_id);
static inline int                c_dma_risk_profile_enable_global_rule(dma_risk_profile* risk_profile, uintptr_t rule_id);
static inline int                c_dma_risk_profile_disable_global_rule(dma_risk_profile* risk_profile, uintptr_t rule_id);

// Check
static inline dma_risk_type      c_dma_risk_profile_check(const dma_risk_profile* risk_profile, const dma_risk_check_state* check_state, dma_risk_violation* out);
static inline dma_risk_violation c_dma_risk_profile_check_algo(const dma* DMA, const dma_risk_profile* risk_profile, const dma_algo* algo);
static inline dma_risk_violation c_dma_risk_profile_check_order(const dma* DMA, const dma_risk_profile* risk_profile, const md_trade_instruction* order);
static inline dma_risk_violation c_dma_risk_profile_check_volume(const dma* DMA, const dma_risk_profile* risk_profile, const char* ticker, md_side side, double volume);
static inline void               c_dma_risk_profile_check_state_init(const dma* DMA, const char* ticker, double price, dma_risk_check_state* check_state);
static inline dma_risk_violation c_dma_risk_profile_check_limit(const dma* DMA, const dma_risk_profile* risk_profile, const char* ticker, md_side side, double volume, double price);

#include <algo_engine/engine/c_dma.h>

// ========== Utility Functions ==========

/**
 * @brief Map a rule to its snake_case key.
 * @param rule The rule.
 * @return The constant key, or NULL for an unknown rule.
 */
static inline const char* c_dma_risk_rule_name(dma_risk_rule rule) {
    if ((int) rule < 0 || rule > DMA_RULE_MAX_NET_NOTIONAL_SHORT) return NULL;

    return DMA_RISK_RULE_NAMES[rule];
}

/**
 * @brief Tell whether a rule is portfolio-wide.
 * @param rule The rule.
 * @return true for the four global rules, false otherwise.
 */
static inline bool c_dma_risk_rule_is_global(dma_risk_rule rule) {
    return rule >= DMA_RULE_MAX_TTL_NOTIONAL_LONG && rule <= DMA_RULE_MAX_NET_NOTIONAL_SHORT;
}

/**
 * @brief The enable bit of a rule in a limit mask.
 * @param rule The rule.
 * @return 1 shifted by the rule index.
 */
static inline uint32_t c_dma_risk_rule_bit(dma_risk_rule rule) {
    return (uint32_t) 1u << (uint32_t) rule;
}

/**
 * @brief Tell whether a rule constrains a side.
 * @param rule The rule.
 * @param side_sign -1 short, 0 neutral (every rule applies), 1 long.
 * @return true when the rule applies to the side.
 */
static inline bool c_dma_risk_rule_applies_to(dma_risk_rule rule, int side_sign) {
    if (side_sign == 0) return true;
    if (rule == DMA_RULE_MAX_PERCENTILE) return true;

    bool is_long_rule = rule == DMA_RULE_MAX_TRADE_LONG
                        || rule == DMA_RULE_MAX_EXPOSURE_LONG
                        || rule == DMA_RULE_MAX_NOTIONAL_LONG
                        || rule == DMA_RULE_MAX_TTL_NOTIONAL_LONG
                        || rule == DMA_RULE_MAX_NET_NOTIONAL_LONG;

    return side_sign > 0 ? is_long_rule : !is_long_rule;
}

/**
 * @brief The dotted name of a violation type.
 * @param type The violation type.
 * @return The constant name; "RiskProfile.Invalid" for an unknown type.
 */
static inline const char* c_dma_risk_type_name(dma_risk_type type) {
    if ((int) type < DMA_RISK_PASS || type > DMA_RISK_CUSTOM) return "RiskProfile.Invalid";

    return DMA_RISK_TYPE_NAMES[type];
}

/**
 * @brief The numeric code of a violation type.
 * @param type The violation type.
 * @return The code; 0 for an unknown type or a custom rule (which reports its own code).
 */
static inline int c_dma_risk_violation_code(dma_risk_type type) {
    if ((int) type < DMA_RISK_PASS || type > DMA_RISK_CUSTOM) return 0;

    return DMA_RISK_VIOLATION_CODES[type];
}

/**
 * @brief Get a ticker's limit set, creating it on first use.
 *
 * The map stores the limit set by pointer, so the pointer is stable across map
 * growth. A missing slot is inserted with the composed block already in it, so
 * a bound view sees the block on the insert event rather than a null slot. An
 * existing slot is returned as stored and the unused block is freed.
 *
 * @param risk_profile The profile (NULL-safe).
 * @param ticker The ticker (non-empty).
 * @return The ticker's limit set, or NULL on invalid input or allocation failure.
 */
static inline dma_risk_limits* c_dma_risk_profile_get_or_insert_limits(dma_risk_profile* risk_profile, const char* ticker) {
    if (!risk_profile || !risk_profile->entries || !ticker || !ticker[0]) return NULL;

    allocator_protocol* allocator = c_ap_protocol_from_ptr(risk_profile->entries);

    dma_risk_limits*    new_limits = (dma_risk_limits*) c_ap_alloc(sizeof(dma_risk_limits), allocator);
    if (!new_limits) return NULL;

    char* stored = NULL;

    int   ret_code = c_bytemap_ex_set_default(
        risk_profile->entries,
        ticker,
        strlen(ticker),
        (const char*) &new_limits,
        sizeof(new_limits),
        0,
        &stored,
        NULL
    );

    if (ret_code != BYTEMAP_OK) {
        c_ap_free_owned(new_limits);
        return NULL;
    }

    dma_risk_limits** slot = (dma_risk_limits**) stored;
    if (!slot) {
        c_ap_free_owned(new_limits);
        return NULL;
    }

    if (*slot != new_limits) c_ap_free_owned(new_limits);

    return *slot;
}

/**
 * @brief Append a rule node to a chain.
 * @param head The chain head (NULL-safe).
 * @param rule The node to append (NULL-safe).
 */
static inline void c_dma_risk_profile_link_rule(dma_risk_rule_ctx** head, dma_risk_rule_ctx* rule) {
    if (!head || !rule) return;

    if (!*head) {
        *head = rule;
        return;
    }

    dma_risk_rule_ctx* tail = *head;
    while (tail->next) tail = tail->next;
    tail->next = rule;
}

/**
 * @brief Unlink the node with the given id from a chain.
 * @param head The chain head (NULL-safe).
 * @param rule_id The node address (0 never matches).
 * @return The unlinked node, or NULL when the id is not in the chain.
 */
static inline dma_risk_rule_ctx* c_dma_risk_profile_unlink_rule(dma_risk_rule_ctx** head, uintptr_t rule_id) {
    if (!head || !rule_id) return NULL;

    dma_risk_rule_ctx** link = head;

    while (*link) {
        dma_risk_rule_ctx* node = *link;

        if (node->rule_id == rule_id) {
            *link = node->next;
            return node;
        }

        link = &node->next;
    }

    return NULL;
}

/**
 * @brief Enable or disable a linked rule.
 * @param risk_profile The profile (NULL-safe).
 * @param ticker The ticker whose chain holds the rule, or NULL for the global chain.
 * @param rule_id The node address.
 * @param enabled true to run the rule, false to skip it.
 * @return DMA_RISK_OK_RET, DMA_RISK_ERR_INVALID_INPUT, or DMA_RISK_ERR_NOT_FOUND.
 */
static inline int c_dma_risk_profile_set_rule_enabled(dma_risk_profile* risk_profile, const char* ticker, uintptr_t rule_id, bool enabled) {
    if (!risk_profile || !rule_id) return DMA_RISK_ERR_INVALID_INPUT;

    dma_risk_rule_ctx* rule;

    if (ticker) {
        dma_risk_limits* limits = c_dma_risk_profile_get_limits(risk_profile, ticker);
        if (!limits) return DMA_RISK_ERR_NOT_FOUND;
        rule = limits->rules;
    }
    else {
        rule = risk_profile->globals.rules;
    }

    while (rule) {
        if (rule->rule_id == rule_id) {
            rule->enabled = enabled;
            return DMA_RISK_OK_RET;
        }

        rule = rule->next;
    }

    return DMA_RISK_ERR_NOT_FOUND;
}

/**
 * @brief Free the global custom rule chain.
 * @param risk_profile The profile (NULL-safe).
 */
static inline void c_dma_risk_profile_global_rules_dealloc(dma_risk_profile* risk_profile) {
    if (!risk_profile) return;

    dma_risk_rule_ctx* rule = risk_profile->globals.rules;

    while (rule) {
        dma_risk_rule_ctx* next = rule->next;
        c_ap_free_owned(rule);
        rule = next;
    }

    risk_profile->globals.rules = NULL;
}

/**
 * @brief Tell whether the profile constrains a side of a ticker.
 *
 * A side is constrained by an enabled absolute risk value that applies to it:
 * a trade, exposure or notional limit on the ticker, or a portfolio total or
 * net limit. The percentile limit is relative to the portfolio, and a custom
 * rule never completes a profile, so neither constrains a side.
 *
 * @param risk_profile The profile (NULL-safe).
 * @param limits The ticker's limit set (NULL when the ticker has no rule).
 * @param side_sign -1 short, 0 neutral (either side), 1 long.
 * @return true when at least one absolute rule applies.
 */
static inline bool c_dma_risk_profile_side_is_constrained(const dma_risk_profile* risk_profile, const dma_risk_limits* limits, int side_sign) {
    if (!risk_profile) return false;

    if (side_sign >= 0) {
        if (limits) {
            if (limits->enabled & c_dma_risk_rule_bit(DMA_RULE_MAX_TRADE_LONG)) return true;
            if (limits->enabled & c_dma_risk_rule_bit(DMA_RULE_MAX_EXPOSURE_LONG)) return true;
            if (limits->enabled & c_dma_risk_rule_bit(DMA_RULE_MAX_NOTIONAL_LONG)) return true;
        }

        if (risk_profile->globals.enabled & c_dma_risk_rule_bit(DMA_RULE_MAX_TTL_NOTIONAL_LONG)) return true;
        if (risk_profile->globals.enabled & c_dma_risk_rule_bit(DMA_RULE_MAX_NET_NOTIONAL_LONG)) return true;
    }

    if (side_sign <= 0) {
        if (limits) {
            if (limits->enabled & c_dma_risk_rule_bit(DMA_RULE_MAX_TRADE_SHORT)) return true;
            if (limits->enabled & c_dma_risk_rule_bit(DMA_RULE_MAX_EXPOSURE_SHORT)) return true;
            if (limits->enabled & c_dma_risk_rule_bit(DMA_RULE_MAX_NOTIONAL_SHORT)) return true;
        }

        if (risk_profile->globals.enabled & c_dma_risk_rule_bit(DMA_RULE_MAX_TTL_NOTIONAL_SHORT)) return true;
        if (risk_profile->globals.enabled & c_dma_risk_rule_bit(DMA_RULE_MAX_NET_NOTIONAL_SHORT)) return true;
    }

    return false;
}

/**
 * @brief Fill a violation with a formatted reason and its type/code pair.
 * @param out The violation being filled (NULL-safe).
 * @param type The violation type.
 * @param format The printf-style format.
 * @param ... The format arguments.
 */
static inline void c_dma_risk_violation_format(dma_risk_violation* out, dma_risk_type type, const char* format, ...) {
    if (!out) return;

    out->type = type;
    out->code = c_dma_risk_violation_code(type);

    va_list args;
    va_start(args, format);
    vsnprintf(out->msg, DMA_RISK_MESSAGE_SIZE, format, args);
    va_end(args);
}

/**
 * @brief Fill a violation with the incomplete-profile message.
 *
 * Lists the rules that apply to the side and are disabled.
 *
 * @param out The violation being filled (NULL-safe).
 * @param risk_profile The profile.
 * @param ticker The ticker the order targets.
 * @param side_sign -1 short, 0 neutral (every rule), 1 long.
 */
static inline void c_dma_risk_violation_incomplete(dma_risk_violation* out, const dma_risk_profile* risk_profile, const char* ticker, int side_sign) {
    if (!out) return;

    const char* side_name = side_sign > 0 ? "LONG" : (side_sign < 0 ? "SHORT" : "any");

    int         n     = snprintf(out->msg, DMA_RISK_MESSAGE_SIZE, "RiskProfile incomplete: no rule constrains a %s order for <%s>; disabled rules: ", side_name, ticker ? ticker : "(null)");
    size_t      used  = (n < 0) ? 0 : (size_t) n;
    bool        first = true;

    if (used >= DMA_RISK_MESSAGE_SIZE) used = DMA_RISK_MESSAGE_SIZE - 1;

    for (int rule = DMA_RULE_MAX_PERCENTILE; rule <= DMA_RULE_MAX_NET_NOTIONAL_SHORT; rule++) {
        if (!c_dma_risk_rule_applies_to((dma_risk_rule) rule, side_sign)) continue;
        if (c_dma_risk_profile_get_limit(risk_profile, ticker, (dma_risk_rule) rule, NULL)) continue;

        const char* name = c_dma_risk_rule_name((dma_risk_rule) rule);
        if (!name) continue;

        n = snprintf(out->msg + used, DMA_RISK_MESSAGE_SIZE - used, "%s%s", first ? "" : ", ", name);
        if (n < 0) break;
        if ((size_t) n >= DMA_RISK_MESSAGE_SIZE - used) {
            used = DMA_RISK_MESSAGE_SIZE - 1;
            break;
        }

        used += (size_t) n;
        first = false;
    }

    out->type = DMA_RISK_INCOMPLETE;
    out->code = c_dma_risk_violation_code(DMA_RISK_INCOMPLETE);
}

/**
 * @brief Run the max-percentile check.
 * @return true when the order passes.
 */
static inline bool c_dma_risk_limits_check_percentile(const dma_risk_limits* limits, const dma_risk_check_state* check_state, dma_risk_violation* out) {
    if (check_state->side_sign == 0) return true;
    if (!isfinite(check_state->portfolio_exposure_notional_abs)) return true;

    double max_notional = check_state->portfolio_exposure_notional_abs / (1 - limits->max_percentile) * limits->max_percentile;
    double max_position = max_notional / check_state->market_price;
    double action       = fabs(check_state->volume);
    double ttl_exposure = check_state->exposure_long - check_state->exposure_short;
    double working      = check_state->side_sign > 0 ? check_state->working_long : check_state->working_short;

    double v0 = ttl_exposure + working;
    double v1 = ttl_exposure + action * (double) check_state->side_sign;
    double v2 = ttl_exposure + action * (double) check_state->side_sign + working;

    if (fabs(v0) <= max_position && fabs(v1) <= max_position && fabs(v2) <= max_position) return true;

    if (check_state->side_sign > 0) {
        c_dma_risk_violation_format(
            out, DMA_RISK_PERCENTILE_LONG,
            "%s %g rejected! lmt_pct=%g, lmt_exp=%g, exp=%g, working=%g, action=%g",
            check_state->ticker, check_state->side_sign * action, limits->max_percentile, max_position, ttl_exposure, working, action
        );
    }
    else if (check_state->side_sign < 0) {
        c_dma_risk_violation_format(
            out, DMA_RISK_PERCENTILE_SHORT,
            "%s %g rejected! lmt_pct=%g, lmt_exp=%g, exp=%g, working=%g, action=%g",
            check_state->ticker, check_state->side_sign * action, limits->max_percentile, max_position, ttl_exposure, working, -action
        );
    }

    return false;
}

/**
 * @brief Run the max-notional check.
 * @return true when the order passes.
 */
static inline bool c_dma_risk_limits_check_notional(const dma_risk_limits* limits, const dma_risk_check_state* check_state, dma_risk_violation* out) {
    double action       = fabs(check_state->volume);
    double ttl_exposure = check_state->exposure_long - check_state->exposure_short;

    if (check_state->side_sign > 0) {
        double max_notional = limits->max_notional_long;
        double max_position = max_notional / check_state->market_price;
        double working      = check_state->working_long;

        double v0 = ttl_exposure + working;
        double v1 = ttl_exposure + action;
        double v2 = ttl_exposure + action + working;

        if (v0 <= max_position && v1 <= max_position && v2 <= max_position) return true;

        c_dma_risk_violation_format(
            out, DMA_RISK_NOTIONAL_LONG,
            "%s %g rejected! lmt_ntl=%g, lmt_exp=%g, exp=%g, working=%g, action=%g",
            check_state->ticker, check_state->side_sign * action, max_notional, max_position, ttl_exposure, working, action
        );
        return false;
    }

    if (check_state->side_sign < 0) {
        double max_notional = limits->max_notional_short;
        double max_position = max_notional / check_state->market_price;
        double working      = check_state->working_short;

        double v0 = ttl_exposure + working;
        double v1 = ttl_exposure - action;
        double v2 = ttl_exposure - action + working;

        if (v0 >= -max_position && v1 >= -max_position && v2 >= -max_position) return true;

        c_dma_risk_violation_format(
            out, DMA_RISK_NOTIONAL_SHORT,
            "%s %g rejected! lmt_ntl=%g, lmt_exp=%g, exp=%g, working=%g, action=%g",
            check_state->ticker, check_state->side_sign * action, max_notional, max_position, ttl_exposure, working, -action
        );
        return false;
    }

    return true;
}

/**
 * @brief Run the portfolio net-notional check.
 * @return true when the order passes.
 */
static inline bool c_dma_risk_limits_globals_check_net(const dma_risk_limits_globals* globals, const dma_risk_check_state* check_state, dma_risk_violation* out) {
    double action = fabs(check_state->volume);
    double price  = check_state->market_price;

    double net_exposure = check_state->portfolio_exposure_notional;
    double net_working  = check_state->portfolio_working_notional;

    double v0 = net_exposure + net_working;
    double v1 = net_exposure + action * (double) check_state->side_sign * price;
    double v2 = net_exposure + action * (double) check_state->side_sign * price + net_working;

    if (check_state->side_sign > 0) {
        double max_net_notional = globals->max_net_notional_long;

        if (v0 <= max_net_notional && v1 <= max_net_notional && v2 <= max_net_notional) return true;

        c_dma_risk_violation_format(
            out, DMA_RISK_PORTFOLIO_NET_LONG,
            "%s %g rejected! lmt_ntl=%g, net_exp=%g, net_working=%g, action=%g",
            check_state->ticker, check_state->side_sign * action, max_net_notional, net_exposure, net_working, action
        );
        return false;
    }

    if (check_state->side_sign < 0) {
        double max_net_notional = globals->max_net_notional_short;

        if (-max_net_notional <= v0 && -max_net_notional <= v1 && -max_net_notional <= v2) return true;

        c_dma_risk_violation_format(
            out, DMA_RISK_PORTFOLIO_NET_SHORT,
            "%s %g rejected! lmt_ntl=%g, net_exp=%g, net_working=%g, action=%g",
            check_state->ticker, check_state->side_sign * action, max_net_notional, net_exposure, net_working, action
        );
        return false;
    }

    return true;
}

/**
 * @brief Run the portfolio total-notional check.
 * @return true when the order passes.
 */
static inline bool c_dma_risk_limits_globals_check_ttl(const dma_risk_limits_globals* globals, const dma_risk_check_state* check_state, dma_risk_violation* out) {
    double action = fabs(check_state->volume);
    double price  = check_state->market_price;

    double ttl_exposure = check_state->side_sign > 0 ? check_state->ttl_exposure_notional_long : check_state->ttl_exposure_notional_short;
    double ttl_working  = check_state->side_sign > 0 ? check_state->ttl_working_notional_long : check_state->ttl_working_notional_short;

    if (check_state->side_sign > 0) {
        double max_notional = globals->max_ttl_notional_long;

        double v0 = ttl_exposure + ttl_working;
        double v1 = ttl_exposure + action * price;
        double v2 = ttl_exposure + action * price + ttl_working;

        if (v0 <= max_notional && v1 <= max_notional && v2 <= max_notional) return true;

        c_dma_risk_violation_format(
            out, DMA_RISK_PORTFOLIO_TOTAL_LONG,
            "%s %g rejected! lmt_ntl=%g, ttl_exp=%g, ttl_working=%g, action=%g",
            check_state->ticker, check_state->side_sign * action, max_notional, ttl_exposure, ttl_working, action
        );
        return false;
    }

    if (check_state->side_sign < 0) {
        double max_notional = globals->max_ttl_notional_short;

        double v0 = ttl_exposure + ttl_working;
        double v1 = ttl_exposure + action * price;
        double v2 = ttl_exposure + action * price + ttl_working;

        if (v0 <= max_notional && v1 <= max_notional && v2 <= max_notional) return true;

        c_dma_risk_violation_format(
            out, DMA_RISK_PORTFOLIO_TOTAL_SHORT,
            "%s %g rejected! lmt_ntl=%g, ttl_exp=%g, ttl_working=%g, action=%g",
            check_state->ticker, check_state->side_sign * action, max_notional, ttl_exposure, ttl_working, action
        );
        return false;
    }

    return true;
}

// ========== Public APIs ==========

/**
 * @brief Allocate and initialize a risk profile.
 * @param allocator Allocator for the profile and its maps.
 * @return The new profile, or NULL on allocation failure.
 */
static inline dma_risk_profile* c_dma_risk_profile_new(allocator_protocol* allocator) {
    dma_risk_profile* risk_profile = (dma_risk_profile*) c_ap_alloc(sizeof(dma_risk_profile), allocator);
    if (!risk_profile) return NULL;

    int ret_code = c_dma_risk_profile_init(risk_profile, allocator);
    if (ret_code != DMA_RISK_OK_RET) {
        c_ap_free(risk_profile);
        return NULL;
    }

    return risk_profile;
}

/**
 * @brief Initialize a caller-allocated, zeroed risk profile.
 *
 * A fresh profile defines no rule and gates no ticker: every transaction is
 * refused until the caller defines the rules that matter.
 *
 * @param risk_profile The zeroed profile to initialize.
 * @param allocator Allocator for the maps; derived from risk_profile when NULL.
 * @return DMA_RISK_OK_RET, DMA_RISK_ERR_INVALID_INPUT, or DMA_RISK_ERR_OOM.
 */
static inline int c_dma_risk_profile_init(dma_risk_profile* risk_profile, allocator_protocol* allocator) {
    if (!risk_profile) return DMA_RISK_ERR_INVALID_INPUT;
    if (!allocator) allocator = c_ap_protocol_from_ptr(risk_profile);

    risk_profile->entries = c_bytemap_ex_new(0, sizeof(dma_risk_limits*), allocator);
    if (!risk_profile->entries) return DMA_RISK_ERR_OOM;

    risk_profile->whitelist = NULL;
    risk_profile->blacklist = NULL;

    return DMA_RISK_OK_RET;
}

/**
 * @brief Tear down a caller-allocated risk profile, leaving it zeroed.
 *
 * Every ticker's limit set is freed with its custom rule tree, the global
 * custom rules are freed, and both gate maps are released.
 *
 * @param risk_profile The profile (NULL-safe).
 */
static inline void c_dma_risk_profile_dealloc(dma_risk_profile* risk_profile) {
    if (!risk_profile) return;

    c_dma_risk_profile_global_rules_dealloc(risk_profile);

    if (risk_profile->entries) {
        bytemap_entry* entry = risk_profile->entries->first;

        while (entry) {
            dma_risk_limits* limits = *(dma_risk_limits**) entry->value;
            if (limits) c_ap_free_owned(limits);
            entry = entry->next;
        }

        c_bytemap_ex_free(risk_profile->entries);
        risk_profile->entries = NULL;
    }

    if (risk_profile->whitelist) {
        c_bytemap_ex_free(risk_profile->whitelist);
        risk_profile->whitelist = NULL;
    }

    if (risk_profile->blacklist) {
        c_bytemap_ex_free(risk_profile->blacklist);
        risk_profile->blacklist = NULL;
    }

    memset(risk_profile, 0, sizeof(dma_risk_profile));
}

/**
 * @brief Free a risk profile allocated by `c_dma_risk_profile_new`.
 * @param risk_profile The profile (NULL-safe).
 */
static inline void c_dma_risk_profile_free(dma_risk_profile* risk_profile) {
    if (!risk_profile) return;

    c_dma_risk_profile_dealloc(risk_profile);
    c_ap_free(risk_profile);
}

/**
 * @brief Remove every ticker rule, every custom rule and both ticker gates.
 *
 * Each ticker's limit set is freed with its custom rule tree, the global
 * custom rules are freed, both gate maps are released, and the four portfolio
 * limits reset to disabled. A cleared profile constrains nothing, so the next
 * check is refused as incomplete until a rule is set.
 *
 * @param risk_profile The profile (NULL-safe).
 * @return DMA_RISK_OK_RET, or DMA_RISK_ERR_INVALID_INPUT.
 */
static inline int c_dma_risk_profile_clear(dma_risk_profile* risk_profile) {
    if (!risk_profile || !risk_profile->entries) return DMA_RISK_ERR_INVALID_INPUT;

    bytemap_entry* entry = risk_profile->entries->first;

    while (entry) {
        dma_risk_limits* limits = *(dma_risk_limits**) entry->value;
        if (limits) c_ap_free_owned(limits);
        entry = entry->next;
    }

    c_bytemap_ex_clear(risk_profile->entries);

    c_dma_risk_profile_global_rules_dealloc(risk_profile);

    if (risk_profile->whitelist) {
        c_bytemap_ex_free(risk_profile->whitelist);
        risk_profile->whitelist = NULL;
    }

    if (risk_profile->blacklist) {
        c_bytemap_ex_free(risk_profile->blacklist);
        risk_profile->blacklist = NULL;
    }

    memset(&risk_profile->globals, 0, sizeof(risk_profile->globals));

    return DMA_RISK_OK_RET;
}

/**
 * @brief List a ticker on the whitelist.
 *
 * The whitelist map is created on first use. A ticker on the blacklist is
 * refused: a ticker cannot be on both lists.
 *
 * @param risk_profile The profile (NULL-safe).
 * @param ticker The ticker (non-empty).
 * @return DMA_RISK_OK_RET, DMA_RISK_ERR_INVALID_INPUT, DMA_RISK_ERR_CONFLICT,
 *         or DMA_RISK_ERR_OOM.
 */
static inline int c_dma_risk_profile_whitelist_add(dma_risk_profile* risk_profile, const char* ticker) {
    if (!risk_profile || !risk_profile->entries || !ticker || !ticker[0]) return DMA_RISK_ERR_INVALID_INPUT;

    if (risk_profile->blacklist && c_bytemap_ex_contains(risk_profile->blacklist, ticker, strlen(ticker)) == BYTEMAP_OK) return DMA_RISK_ERR_CONFLICT;

    if (!risk_profile->whitelist) {
        allocator_protocol* allocator = c_ap_protocol_from_ptr(risk_profile->entries);
        risk_profile->whitelist       = c_bytemap_ex_new(0, sizeof(uint8_t), allocator);
        if (!risk_profile->whitelist) return DMA_RISK_ERR_OOM;
    }

    uint8_t marker   = 1;
    int     ret_code = c_bytemap_ex_set(risk_profile->whitelist, ticker, strlen(ticker), (const char*) &marker, sizeof(marker), 0, NULL);

    if (ret_code == BYTEMAP_OK) return DMA_RISK_OK_RET;

    return (ret_code == BYTEMAP_ERR_OOM || ret_code == BYTEMAP_ERR_FULL) ? DMA_RISK_ERR_OOM : DMA_RISK_ERR_INVALID_INPUT;
}

/**
 * @brief Drop a ticker from the whitelist; a missing ticker is a no-op.
 *
 * Removing the last ticker frees the map, so an absent whitelist gates
 * nothing.
 *
 * @param risk_profile The profile (NULL-safe).
 * @param ticker The ticker (non-empty).
 * @return DMA_RISK_OK_RET, or DMA_RISK_ERR_INVALID_INPUT.
 */
static inline int c_dma_risk_profile_whitelist_remove(dma_risk_profile* risk_profile, const char* ticker) {
    if (!risk_profile || !ticker || !ticker[0]) return DMA_RISK_ERR_INVALID_INPUT;
    if (!risk_profile->whitelist) return DMA_RISK_OK_RET;

    int ret_code = c_bytemap_ex_pop(risk_profile->whitelist, ticker, strlen(ticker), 0, NULL, NULL);

    if (ret_code != BYTEMAP_OK && ret_code != BYTEMAP_ERR_NOT_FOUND) return DMA_RISK_ERR_INVALID_INPUT;

    if (c_bytemap_ex_len(risk_profile->whitelist) == 0) {
        c_bytemap_ex_free(risk_profile->whitelist);
        risk_profile->whitelist = NULL;
    }

    return DMA_RISK_OK_RET;
}

/**
 * @brief List a ticker on the blacklist; the ticker is refused.
 *
 * The blacklist map is created on first use. A ticker on the whitelist is
 * refused: a ticker cannot be on both lists.
 *
 * @param risk_profile The profile (NULL-safe).
 * @param ticker The ticker (non-empty).
 * @return DMA_RISK_OK_RET, DMA_RISK_ERR_INVALID_INPUT, DMA_RISK_ERR_CONFLICT,
 *         or DMA_RISK_ERR_OOM.
 */
static inline int c_dma_risk_profile_blacklist_add(dma_risk_profile* risk_profile, const char* ticker) {
    if (!risk_profile || !risk_profile->entries || !ticker || !ticker[0]) return DMA_RISK_ERR_INVALID_INPUT;

    if (risk_profile->whitelist && c_bytemap_ex_contains(risk_profile->whitelist, ticker, strlen(ticker)) == BYTEMAP_OK) return DMA_RISK_ERR_CONFLICT;

    if (!risk_profile->blacklist) {
        allocator_protocol* allocator = c_ap_protocol_from_ptr(risk_profile->entries);
        risk_profile->blacklist       = c_bytemap_ex_new(0, sizeof(uint8_t), allocator);
        if (!risk_profile->blacklist) return DMA_RISK_ERR_OOM;
    }

    uint8_t marker   = 1;
    int     ret_code = c_bytemap_ex_set(risk_profile->blacklist, ticker, strlen(ticker), (const char*) &marker, sizeof(marker), 0, NULL);

    if (ret_code == BYTEMAP_OK) return DMA_RISK_OK_RET;

    return (ret_code == BYTEMAP_ERR_OOM || ret_code == BYTEMAP_ERR_FULL) ? DMA_RISK_ERR_OOM : DMA_RISK_ERR_INVALID_INPUT;
}

/**
 * @brief Drop a ticker from the blacklist; a missing ticker is a no-op.
 *
 * Removing the last ticker frees the map, so an absent blacklist gates
 * nothing.
 *
 * @param risk_profile The profile (NULL-safe).
 * @param ticker The ticker (non-empty).
 * @return DMA_RISK_OK_RET, or DMA_RISK_ERR_INVALID_INPUT.
 */
static inline int c_dma_risk_profile_blacklist_remove(dma_risk_profile* risk_profile, const char* ticker) {
    if (!risk_profile || !ticker || !ticker[0]) return DMA_RISK_ERR_INVALID_INPUT;
    if (!risk_profile->blacklist) return DMA_RISK_OK_RET;

    int ret_code = c_bytemap_ex_pop(risk_profile->blacklist, ticker, strlen(ticker), 0, NULL, NULL);

    if (ret_code != BYTEMAP_OK && ret_code != BYTEMAP_ERR_NOT_FOUND) return DMA_RISK_ERR_INVALID_INPUT;

    if (c_bytemap_ex_len(risk_profile->blacklist) == 0) {
        c_bytemap_ex_free(risk_profile->blacklist);
        risk_profile->blacklist = NULL;
    }

    return DMA_RISK_OK_RET;
}

/**
 * @brief Tell whether the ticker may trade under the two gates.
 *
 * Each gate is skipped when its map is absent: a profile with no lists allows
 * every ticker, a blacklist alone refuses its listed tickers, and a whitelist
 * alone allows only its listed tickers.
 *
 * @param risk_profile The profile (NULL-safe).
 * @param ticker The ticker (NULL-safe).
 * @return false when the ticker is blacklisted or not whitelisted; true
 *         otherwise.
 */
static inline bool c_dma_risk_profile_is_ticker_allowed(const dma_risk_profile* risk_profile, const char* ticker) {
    if (!risk_profile || !ticker) return false;

    size_t ticker_len = strlen(ticker);

    if (risk_profile->blacklist && c_bytemap_ex_contains(risk_profile->blacklist, ticker, ticker_len) == BYTEMAP_OK) return false;
    if (risk_profile->whitelist && c_bytemap_ex_contains(risk_profile->whitelist, ticker, ticker_len) != BYTEMAP_OK) return false;

    return true;
}

/**
 * @brief Write one rule value and its enable bit.
 *
 * A percentile of one or more, and an infinite value of any other rule, does
 * not constrain: the field resets to 0 and the bit clears. The caller refuses
 * NaN and negative values.
 *
 * @param field The rule's value.
 * @param enabled The limit set's enable mask.
 * @param rule The rule being written.
 * @param value The value (not NaN, not negative).
 */
static inline void c_dma_risk_field_set(double* field, uint32_t* enabled, dma_risk_rule rule, double value) {
    uint32_t bit = c_dma_risk_rule_bit(rule);

    if (rule == DMA_RULE_MAX_PERCENTILE ? value >= 1. : isinf(value)) {
        *field = 0.;
        *enabled &= ~bit;
        return;
    }

    *field = value;
    *enabled |= bit;
}

/**
 * @brief Set one rule, globally or for a ticker.
 *
 * The ticker's limit set is created on first write (get-or-insert in one
 * probe) with every rule disabled. NaN and negative values are refused. A
 * percentile of one or more, and an infinite value of any other rule, disables
 * the rule: the field resets to 0 and its enable bit clears.
 *
 * @param risk_profile The profile (NULL-safe).
 * @param ticker The ticker, or NULL for a portfolio-wide rule.
 * @param rule The rule to set.
 * @param value The limit value (not NaN, not negative).
 * @return DMA_RISK_OK_RET, DMA_RISK_ERR_INVALID_INPUT, DMA_RISK_ERR_UNKNOWN_RULE,
 *         DMA_RISK_ERR_RULE_SCOPE, or DMA_RISK_ERR_OOM.
 */
static inline int c_dma_risk_profile_set_limit(dma_risk_profile* risk_profile, const char* ticker, dma_risk_rule rule, double value) {
    if (!risk_profile || !risk_profile->entries) return DMA_RISK_ERR_INVALID_INPUT;
    if (!c_dma_risk_rule_name(rule)) return DMA_RISK_ERR_UNKNOWN_RULE;
    if (isnan(value) || value < 0.) return DMA_RISK_ERR_INVALID_INPUT;

    if (!ticker) {
        if (!c_dma_risk_rule_is_global(rule)) return DMA_RISK_ERR_RULE_SCOPE;

        switch (rule) {
            case DMA_RULE_MAX_TTL_NOTIONAL_LONG:
                c_dma_risk_field_set(&risk_profile->globals.max_ttl_notional_long, &risk_profile->globals.enabled, rule, value);
                break;
            case DMA_RULE_MAX_TTL_NOTIONAL_SHORT:
                c_dma_risk_field_set(&risk_profile->globals.max_ttl_notional_short, &risk_profile->globals.enabled, rule, value);
                break;
            case DMA_RULE_MAX_NET_NOTIONAL_LONG:
                c_dma_risk_field_set(&risk_profile->globals.max_net_notional_long, &risk_profile->globals.enabled, rule, value);
                break;
            case DMA_RULE_MAX_NET_NOTIONAL_SHORT:
                c_dma_risk_field_set(&risk_profile->globals.max_net_notional_short, &risk_profile->globals.enabled, rule, value);
                break;
            default:
                return DMA_RISK_ERR_RULE_SCOPE;
        }

        return DMA_RISK_OK_RET;
    }

    if (c_dma_risk_rule_is_global(rule)) return DMA_RISK_ERR_RULE_SCOPE;

    dma_risk_limits* limits = c_dma_risk_profile_get_or_insert_limits(risk_profile, ticker);
    if (!limits) return DMA_RISK_ERR_OOM;

    switch (rule) {
        case DMA_RULE_MAX_PERCENTILE:
            c_dma_risk_field_set(&limits->max_percentile, &limits->enabled, rule, value);
            break;
        case DMA_RULE_MAX_TRADE_LONG:
            c_dma_risk_field_set(&limits->max_trade_long, &limits->enabled, rule, value);
            break;
        case DMA_RULE_MAX_TRADE_SHORT:
            c_dma_risk_field_set(&limits->max_trade_short, &limits->enabled, rule, value);
            break;
        case DMA_RULE_MAX_EXPOSURE_LONG:
            c_dma_risk_field_set(&limits->max_exposure_long, &limits->enabled, rule, value);
            break;
        case DMA_RULE_MAX_EXPOSURE_SHORT:
            c_dma_risk_field_set(&limits->max_exposure_short, &limits->enabled, rule, value);
            break;
        case DMA_RULE_MAX_NOTIONAL_LONG:
            c_dma_risk_field_set(&limits->max_notional_long, &limits->enabled, rule, value);
            break;
        case DMA_RULE_MAX_NOTIONAL_SHORT:
            c_dma_risk_field_set(&limits->max_notional_short, &limits->enabled, rule, value);
            break;
        default:
            return DMA_RISK_ERR_RULE_SCOPE;
    }

    return DMA_RISK_OK_RET;
}

/**
 * @brief Read one rule's value.
 *
 * A rule is readable only while it is enabled: a disabled rule reports false.
 * A global rule ignores the ticker argument; a ticker rule requires it.
 *
 * @param risk_profile The profile (NULL-safe).
 * @param ticker The ticker, or NULL for a portfolio-wide rule.
 * @param rule The rule to read.
 * @param out Optional: receives the value when the rule is enabled.
 * @return true when the rule is enabled and the value was written.
 */
static inline bool c_dma_risk_profile_get_limit(const dma_risk_profile* risk_profile, const char* ticker, dma_risk_rule rule, double* out) {
    if (!risk_profile || !c_dma_risk_rule_name(rule)) return false;

    if (c_dma_risk_rule_is_global(rule)) {
        if (!(risk_profile->globals.enabled & c_dma_risk_rule_bit(rule))) return false;

        double value;

        switch (rule) {
            case DMA_RULE_MAX_TTL_NOTIONAL_LONG:
                value = risk_profile->globals.max_ttl_notional_long;
                break;
            case DMA_RULE_MAX_TTL_NOTIONAL_SHORT:
                value = risk_profile->globals.max_ttl_notional_short;
                break;
            case DMA_RULE_MAX_NET_NOTIONAL_LONG:
                value = risk_profile->globals.max_net_notional_long;
                break;
            case DMA_RULE_MAX_NET_NOTIONAL_SHORT:
                value = risk_profile->globals.max_net_notional_short;
                break;
            default:
                return false;
        }

        if (out) *out = value;
        return true;
    }

    if (!ticker) return false;

    const dma_risk_limits* limits = c_dma_risk_profile_get_limits(risk_profile, ticker);
    if (!limits || !(limits->enabled & c_dma_risk_rule_bit(rule))) return false;

    double value;

    switch (rule) {
        case DMA_RULE_MAX_PERCENTILE:
            value = limits->max_percentile;
            break;
        case DMA_RULE_MAX_TRADE_LONG:
            value = limits->max_trade_long;
            break;
        case DMA_RULE_MAX_TRADE_SHORT:
            value = limits->max_trade_short;
            break;
        case DMA_RULE_MAX_EXPOSURE_LONG:
            value = limits->max_exposure_long;
            break;
        case DMA_RULE_MAX_EXPOSURE_SHORT:
            value = limits->max_exposure_short;
            break;
        case DMA_RULE_MAX_NOTIONAL_LONG:
            value = limits->max_notional_long;
            break;
        case DMA_RULE_MAX_NOTIONAL_SHORT:
            value = limits->max_notional_short;
            break;
        default:
            return false;
    }

    if (out) *out = value;
    return true;
}

/**
 * @brief Look up a ticker's limit set.
 *
 * The map stores the limit set by pointer, so the returned block is stable
 * while the profile lives.
 *
 * @param risk_profile The profile (NULL-safe).
 * @param ticker The ticker (NULL-safe).
 * @return The borrowed limit set, or NULL when the ticker has no rule.
 */
static inline dma_risk_limits* c_dma_risk_profile_get_limits(const dma_risk_profile* risk_profile, const char* ticker) {
    if (!risk_profile || !risk_profile->entries || !ticker) return NULL;

    char* stored = NULL;

    if (c_bytemap_ex_get_ptr(risk_profile->entries, ticker, strlen(ticker), &stored, NULL) != BYTEMAP_OK) return NULL;

    if (!stored) return NULL;

    return *(dma_risk_limits**) stored;
}

/**
 * @brief Copy a string under a rule node.
 * @param source The string to copy (NULL returns NULL).
 * @param rule The node that owns the copy.
 * @return The copy, or NULL when source is NULL or the allocation fails.
 */
static inline const char* c_dma_risk_rule_strdup(const char* source, dma_risk_rule_ctx* rule) {
    if (!source) return NULL;

    size_t len  = strlen(source) + 1;
    char*  copy = (char*) c_ap_alloc_child(len, NULL, rule);
    if (!copy) return NULL;

    memcpy(copy, source, len);
    return copy;
}

/**
 * @brief Compose and append a custom rule to a ticker's chain.
 *
 * The ticker's limit set is created on first registration with every rule
 * disabled. The profile allocates the node under the ticker's limit set, sets
 * its id to the node address and owns it: unregister, clear and dealloc free
 * it. `ticker` and `name` are copied under the node. `user_data` is the
 * rule's state; a rule that adopts it under the node gives the node its
 * lifetime.
 *
 * @param risk_profile The profile (NULL-safe).
 * @param ticker The ticker (non-empty).
 * @param check_fn The rule's check (NULL is refused).
 * @param risk_rule_code The code a rejection reports.
 * @param name The rule name reported in a message (NULL allowed).
 * @param user_data The rule's state (NULL allowed).
 * @return The composed node, or NULL on invalid input or allocation failure.
 */
static inline dma_risk_rule_ctx* c_dma_risk_profile_register_rule(dma_risk_profile* risk_profile, const char* ticker, dma_risk_rule_check_func check_fn, int risk_rule_code, const char* name, void* user_data) {
    if (!risk_profile || !risk_profile->entries || !ticker || !ticker[0] || !check_fn) return NULL;

    dma_risk_limits* limits = c_dma_risk_profile_get_or_insert_limits(risk_profile, ticker);
    if (!limits) return NULL;

    dma_risk_rule_ctx* rule = (dma_risk_rule_ctx*) c_ap_alloc_child(sizeof(dma_risk_rule_ctx), NULL, limits);
    if (!rule) return NULL;

    rule->risk_rule_code = risk_rule_code;
    rule->enabled        = true;
    rule->check_fn       = check_fn;
    rule->user_data      = user_data;
    rule->rule_id        = (uintptr_t) rule;
    rule->next           = NULL;

    rule->ticker = c_dma_risk_rule_strdup(ticker, rule);
    if (!rule->ticker) {
        c_ap_free_owned(rule);
        return NULL;
    }

    if (name) {
        rule->name = c_dma_risk_rule_strdup(name, rule);
        if (!rule->name) {
            c_ap_free_owned(rule);
            return NULL;
        }
    }

    c_dma_risk_profile_link_rule(&limits->rules, rule);

    return rule;
}

/**
 * @brief Compose and append a global custom rule.
 *
 * A global rule applies to every ticker and runs after the ticker's own custom
 * rules. The node is its own allocator block; unregister, clear and dealloc
 * free it. `name` is copied under the node, and `user_data` follows the same
 * adoption rule as a ticker rule.
 *
 * @param risk_profile The profile (NULL-safe).
 * @param check_fn The rule's check (NULL is refused).
 * @param risk_rule_code The code a rejection reports.
 * @param name The rule name reported in a message (NULL allowed).
 * @param user_data The rule's state (NULL allowed).
 * @return The composed node, or NULL on invalid input or allocation failure.
 */
static inline dma_risk_rule_ctx* c_dma_risk_profile_register_global_rule(dma_risk_profile* risk_profile, dma_risk_rule_check_func check_fn, int risk_rule_code, const char* name, void* user_data) {
    if (!risk_profile || !risk_profile->entries || !check_fn) return NULL;

    allocator_protocol* allocator = c_ap_protocol_from_ptr(risk_profile->entries);

    dma_risk_rule_ctx*  rule = (dma_risk_rule_ctx*) c_ap_alloc(sizeof(dma_risk_rule_ctx), allocator);
    if (!rule) return NULL;

    rule->risk_rule_code = risk_rule_code;
    rule->enabled        = true;
    rule->check_fn       = check_fn;
    rule->user_data      = user_data;
    rule->rule_id        = (uintptr_t) rule;
    rule->next           = NULL;
    rule->ticker         = NULL;

    if (name) {
        rule->name = c_dma_risk_rule_strdup(name, rule);
        if (!rule->name) {
            c_ap_free_owned(rule);
            return NULL;
        }
    }

    c_dma_risk_profile_link_rule(&risk_profile->globals.rules, rule);

    return rule;
}

/**
 * @brief Unlink a ticker's custom rule and free it.
 *
 * The node is freed with its owned tree, so a rule state adopted under the
 * node is released here.
 *
 * @param risk_profile The profile (NULL-safe).
 * @param ticker The ticker (non-empty).
 * @param rule_id The node address returned by the registration.
 * @return DMA_RISK_OK_RET, DMA_RISK_ERR_INVALID_INPUT, or DMA_RISK_ERR_NOT_FOUND.
 */
static inline int c_dma_risk_profile_unregister_rule(dma_risk_profile* risk_profile, const char* ticker, uintptr_t rule_id) {
    if (!risk_profile || !risk_profile->entries || !ticker || !ticker[0] || !rule_id) return DMA_RISK_ERR_INVALID_INPUT;

    dma_risk_limits* limits = c_dma_risk_profile_get_limits(risk_profile, ticker);
    if (!limits) return DMA_RISK_ERR_NOT_FOUND;

    dma_risk_rule_ctx* rule = c_dma_risk_profile_unlink_rule(&limits->rules, rule_id);
    if (!rule) return DMA_RISK_ERR_NOT_FOUND;

    c_ap_free_owned(rule);

    return DMA_RISK_OK_RET;
}

/**
 * @brief Unlink a global custom rule and free it.
 *
 * @param risk_profile The profile (NULL-safe).
 * @param rule_id The node address returned by the registration.
 * @return DMA_RISK_OK_RET, DMA_RISK_ERR_INVALID_INPUT, or DMA_RISK_ERR_NOT_FOUND.
 */
static inline int c_dma_risk_profile_unregister_global_rule(dma_risk_profile* risk_profile, uintptr_t rule_id) {
    if (!risk_profile || !rule_id) return DMA_RISK_ERR_INVALID_INPUT;

    dma_risk_rule_ctx* rule = c_dma_risk_profile_unlink_rule(&risk_profile->globals.rules, rule_id);
    if (!rule) return DMA_RISK_ERR_NOT_FOUND;

    c_ap_free_owned(rule);

    return DMA_RISK_OK_RET;
}

/**
 * @brief Enable a ticker's custom rule.
 * @param risk_profile The profile (NULL-safe).
 * @param ticker The ticker (non-empty).
 * @param rule_id The node address.
 * @return DMA_RISK_OK_RET, DMA_RISK_ERR_INVALID_INPUT, or DMA_RISK_ERR_NOT_FOUND.
 */
static inline int c_dma_risk_profile_enable_rule(dma_risk_profile* risk_profile, const char* ticker, uintptr_t rule_id) {
    if (!ticker || !ticker[0]) return DMA_RISK_ERR_INVALID_INPUT;

    return c_dma_risk_profile_set_rule_enabled(risk_profile, ticker, rule_id, true);
}

/**
 * @brief Disable a ticker's custom rule; the check skips it but it stays linked.
 * @param risk_profile The profile (NULL-safe).
 * @param ticker The ticker (non-empty).
 * @param rule_id The node address.
 * @return DMA_RISK_OK_RET, DMA_RISK_ERR_INVALID_INPUT, or DMA_RISK_ERR_NOT_FOUND.
 */
static inline int c_dma_risk_profile_disable_rule(dma_risk_profile* risk_profile, const char* ticker, uintptr_t rule_id) {
    if (!ticker || !ticker[0]) return DMA_RISK_ERR_INVALID_INPUT;

    return c_dma_risk_profile_set_rule_enabled(risk_profile, ticker, rule_id, false);
}

/**
 * @brief Enable a global custom rule.
 * @param risk_profile The profile (NULL-safe).
 * @param rule_id The node address.
 * @return DMA_RISK_OK_RET, DMA_RISK_ERR_INVALID_INPUT, or DMA_RISK_ERR_NOT_FOUND.
 */
static inline int c_dma_risk_profile_enable_global_rule(dma_risk_profile* risk_profile, uintptr_t rule_id) {
    return c_dma_risk_profile_set_rule_enabled(risk_profile, NULL, rule_id, true);
}

/**
 * @brief Disable a global custom rule; the check skips it but it stays linked.
 * @param risk_profile The profile (NULL-safe).
 * @param rule_id The node address.
 * @return DMA_RISK_OK_RET, DMA_RISK_ERR_INVALID_INPUT, or DMA_RISK_ERR_NOT_FOUND.
 */
static inline int c_dma_risk_profile_disable_global_rule(dma_risk_profile* risk_profile, uintptr_t rule_id) {
    return c_dma_risk_profile_set_rule_enabled(risk_profile, NULL, rule_id, false);
}

/**
 * @brief Check one order against the profile.
 *
 * The gate runs first: a blacklisted ticker, or a ticker missing from a
 * non-empty whitelist, is refused. The check then requires an enabled absolute
 * risk value for the side; with none, it reports DMA_RISK_INCOMPLETE and
 * lists the disabled rules that apply. It then requires the ticker's market
 * price; without one, it reports DMA_RISK_PRICE. Only the enabled rules run,
 * in the upstream order: traded volume, exposure, percentile, notional,
 * portfolio net, portfolio total. The first failure fills `out` and returns
 * its type.
 *
 * The pure check_state check cannot run the custom rules (they need the access);
 * `c_dma_risk_profile_check_limit` runs them after this check passes.
 *
 * @param risk_profile The profile (NULL-safe).
 * @param check_state The order and portfolio state (NULL-safe).
 * @param out Optional: receives the violation on failure.
 * @return DMA_RISK_PASS when the order may be sent, DMA_RISK_INCOMPLETE when
 *         no absolute rule constrains the side, the failing check's type, or
 *         DMA_RISK_INVALID for invalid input.
 */
static inline dma_risk_type c_dma_risk_profile_check(const dma_risk_profile* risk_profile, const dma_risk_check_state* check_state, dma_risk_violation* out) {
    if (out) {
        out->type   = DMA_RISK_PASS;
        out->code   = 0;
        out->msg[0] = '\0';
    }

    if (!risk_profile || !check_state || !check_state->ticker) {
        c_dma_risk_violation_format(out, DMA_RISK_INVALID, "invalid risk_profile check input");
        return DMA_RISK_INVALID;
    }

    // Step 0: the ticker gates.
    if (!c_dma_risk_profile_is_ticker_allowed(risk_profile, check_state->ticker)) {
        if (risk_profile->blacklist && c_bytemap_ex_contains(risk_profile->blacklist, check_state->ticker, strlen(check_state->ticker)) == BYTEMAP_OK) {
            c_dma_risk_violation_format(out, DMA_RISK_TICKER_BLOCKED, "%s rejected! the ticker is blacklisted", check_state->ticker);
            return DMA_RISK_TICKER_BLOCKED;
        }

        c_dma_risk_violation_format(out, DMA_RISK_TICKER_UNLISTED, "%s rejected! the ticker is not whitelisted", check_state->ticker);
        return DMA_RISK_TICKER_UNLISTED;
    }

    const dma_risk_limits* limits    = c_dma_risk_profile_get_limits(risk_profile, check_state->ticker);
    int                    side_sign = check_state->side_sign;
    bool                   is_long   = side_sign > 0;

    // Step 1: the profile must constrain the side; an unconstrained side has
    // no absolute risk value to check and is refused as incomplete.
    if (!c_dma_risk_profile_side_is_constrained(risk_profile, limits, side_sign)) {
        c_dma_risk_violation_incomplete(out, risk_profile, check_state->ticker, side_sign);
        return DMA_RISK_INCOMPLETE;
    }

    // Step 2: validity -- a trade needs its ticker's market price.
    if (isnan(check_state->market_price)) {
        c_dma_risk_violation_format(out, DMA_RISK_PRICE, "no valid market price for ticker %s", check_state->ticker);
        return DMA_RISK_PRICE;
    }

    double action = fabs(check_state->volume);

    // Step 3: traded volume.
    if (side_sign != 0 && limits) {
        dma_risk_rule rule = is_long ? DMA_RULE_MAX_TRADE_LONG : DMA_RULE_MAX_TRADE_SHORT;

        if (limits->enabled & c_dma_risk_rule_bit(rule)) {
            double limit = is_long ? limits->max_trade_long : limits->max_trade_short;

            double trade_count = is_long ? check_state->working_long + check_state->traded_long : check_state->working_short + check_state->traded_short;

            if (trade_count + action > limit) {
                c_dma_risk_violation_format(
                    out, is_long ? DMA_RISK_TRADE_LONG : DMA_RISK_TRADE_SHORT,
                    "%s %g rejected! lmt=%g, ttl=%g, inv=%g, action=%g",
                    check_state->ticker, side_sign * action, limit, trade_count, limit - trade_count, is_long ? action : -action
                );
                return is_long ? DMA_RISK_TRADE_LONG : DMA_RISK_TRADE_SHORT;
            }
        }
    }

    // Step 4: exposure.
    if (side_sign != 0 && limits) {
        dma_risk_rule rule = is_long ? DMA_RULE_MAX_EXPOSURE_LONG : DMA_RULE_MAX_EXPOSURE_SHORT;

        if (limits->enabled & c_dma_risk_rule_bit(rule)) {
            double limit = is_long ? limits->max_exposure_long : limits->max_exposure_short;

            double ttl_exposure = check_state->exposure_long - check_state->exposure_short;
            double working      = is_long ? check_state->working_long : check_state->working_short;

            double v0 = ttl_exposure + working;
            double v1 = ttl_exposure + action * (double) side_sign;
            double v2 = ttl_exposure + action * (double) side_sign + working;

            bool   passed = is_long ? (v0 <= limit && v1 <= limit && v2 <= limit)
                                    : (v0 >= -limit && v1 >= -limit && v2 >= -limit);

            if (!passed) {
                c_dma_risk_violation_format(
                    out, is_long ? DMA_RISK_EXPOSURE_LONG : DMA_RISK_EXPOSURE_SHORT,
                    "%s %g rejected! lmt_exp=%g, exp=%g, working=%g, action=%g",
                    check_state->ticker, side_sign * action, limit, ttl_exposure, working, is_long ? action : -action
                );
                return is_long ? DMA_RISK_EXPOSURE_LONG : DMA_RISK_EXPOSURE_SHORT;
            }
        }
    }

    // Step 5: percentile.
    if (limits && (limits->enabled & c_dma_risk_rule_bit(DMA_RULE_MAX_PERCENTILE))) {
        if (!c_dma_risk_limits_check_percentile(limits, check_state, out)) {
            return out ? out->type : DMA_RISK_INVALID;
        }
    }

    // Step 6: notional.
    if (limits && (limits->enabled & c_dma_risk_rule_bit(is_long ? DMA_RULE_MAX_NOTIONAL_LONG : DMA_RULE_MAX_NOTIONAL_SHORT))) {
        if (!c_dma_risk_limits_check_notional(limits, check_state, out)) {
            return out ? out->type : DMA_RISK_INVALID;
        }
    }

    // Step 7: portfolio net.
    if (risk_profile->globals.enabled & c_dma_risk_rule_bit(is_long ? DMA_RULE_MAX_NET_NOTIONAL_LONG : DMA_RULE_MAX_NET_NOTIONAL_SHORT)) {
        if (!c_dma_risk_limits_globals_check_net(&risk_profile->globals, check_state, out)) {
            return out ? out->type : DMA_RISK_INVALID;
        }
    }

    // Step 8: portfolio total.
    if (risk_profile->globals.enabled & c_dma_risk_rule_bit(is_long ? DMA_RULE_MAX_TTL_NOTIONAL_LONG : DMA_RULE_MAX_TTL_NOTIONAL_SHORT)) {
        if (!c_dma_risk_limits_globals_check_ttl(&risk_profile->globals, check_state, out)) {
            return out ? out->type : DMA_RISK_INVALID;
        }
    }

    return DMA_RISK_PASS;
}

// ========== Public APIs (Access Checks) ==========

// HOT PATH: c_dma_risk_profile_check_limit runs once per order launch and is
// reached through c_dma_check_order. It allocates nothing, logs nothing, and
// writes the violation only on a rejection.

/**
 * @brief Fill the access-derived volumes of a check state for one ticker.
 *
 * The hosted positions supply the ticker's working and net exposure volumes
 * and the portfolio notional totals, each ticker priced at its own market
 * price; the engine's orders supply the ticker's filled volume for the
 * access's session date. The record is zeroed first, so the caller sets only
 * the order's own volume and side.
 *
 * @param DMA The access (NULL-safe).
 * @param ticker The instrument (borrowed).
 * @param price The price the ticker is priced at (NaN when unavailable).
 * @param check_state The state to fill (NULL-safe).
 */
static inline void c_dma_risk_profile_check_state_init(const dma* DMA, const char* ticker, double price, dma_risk_check_state* check_state) {
    if (!check_state) return;

    memset(check_state, 0, sizeof(*check_state));

    check_state->ticker       = ticker;
    check_state->market_price = price;

    if (!DMA || !ticker) return;

    // === Step 0 ===
    // The hosted positions supply the ticker's working and net exposure and
    // the portfolio totals, each ticker priced at its own market price.
    if (DMA->positions) {
        bytemap_entry* entry = DMA->positions->first;

        while (entry) {
            const dma_position* position = *(const dma_position**) entry->value;

            if (position && position->tickers) {
                bytemap_entry* ticker_entry = position->tickers->first;

                while (ticker_entry) {
                    const dma_position_ticker* cache        = (const dma_position_ticker*) ticker_entry->value;
                    const char*                name         = ticker_entry->key;
                    bool                       own          = name && strcmp(name, ticker) == 0;
                    double                     ticker_price = own ? price : c_mds_get_market_price(DMA->mds, name);

                    if (!isfinite(ticker_price)) ticker_price = 0.;

                    if (own) {
                        check_state->working_long += cache->working_long;
                        check_state->working_short += cache->working_short;

                        if (cache->exposure_volume > 0) check_state->exposure_long += cache->exposure_volume;
                        else check_state->exposure_short -= cache->exposure_volume;
                    }

                    double net_notional     = cache->exposure_volume * ticker_price;
                    double working_notional = (cache->working_long + cache->working_short) * ticker_price;

                    check_state->portfolio_exposure_notional += net_notional;
                    check_state->portfolio_exposure_notional_abs += fabs(net_notional);
                    check_state->portfolio_working_notional += working_notional;

                    if (net_notional > 0) check_state->ttl_exposure_notional_long += net_notional;
                    else check_state->ttl_exposure_notional_short -= net_notional;

                    check_state->ttl_working_notional_long += cache->working_long * ticker_price;
                    check_state->ttl_working_notional_short += cache->working_short * ticker_price;

                    ticker_entry = ticker_entry->next;
                }
            }

            entry = entry->next;
        }
    }

    // === Step 1 ===
    // The engine's orders supply the ticker's filled volume for the access's
    // session date.
    if (DMA->algo_engine && DMA->algo_engine->orders && DMA->mds) {
        const session_date_t today = DMA->mds->ts.date;
        bytemap_entry*       entry = DMA->algo_engine->orders->first;

        while (entry) {
            const dma_order_ctx* ctx = *(const dma_order_ctx**) entry->value;

            if (ctx && ctx->order.meta_info.ticker && ctx->order.filled_volume > 0. && strcmp(ctx->order.meta_info.ticker, ticker) == 0) {
                const session_date_t placed = ctx->order.meta_info.dt.date;
                int                  sign   = c_md_side_sign(ctx->order.side);

                if (placed.year == today.year && placed.month == today.month && placed.day == today.day) {
                    if (sign > 0) check_state->traded_long += ctx->order.filled_volume;
                    else if (sign < 0) check_state->traded_short += ctx->order.filled_volume;
                }
            }

            entry = entry->next;
        }
    }
}

/**
 * @brief Check a volume on a side at an explicit price against the profile.
 *
 * Runs once per algo open and per order launch. The check_state is filled from
 * the access by `c_dma_risk_profile_check_state_init`; a portfolio ticker
 * without a market price counts as zero. The ticker's custom rules run after
 * the standard rules pass; the first rejection is returned.
 *
 * @param DMA The access (NULL-safe).
 * @param risk_profile The profile (NULL-safe).
 * @param ticker The instrument (non-empty).
 * @param side The trade side.
 * @param volume The volume to check.
 * @param price The price the rules are checked at (NaN when unavailable).
 * @return The check's violation; a passing check reports DMA_RISK_PASS.
 */
static inline dma_risk_violation c_dma_risk_profile_check_limit(const dma* DMA, const dma_risk_profile* risk_profile, const char* ticker, md_side side, double volume, double price) {
    dma_risk_check_state check_state;
    dma_risk_violation   violation;

    if (!DMA || !risk_profile || !ticker) {
        c_dma_risk_violation_format(&violation, DMA_RISK_INVALID, "invalid risk_profile check input");
        return violation;
    }

    c_dma_risk_profile_check_state_init(DMA, ticker, price, &check_state);

    check_state.volume    = volume;
    check_state.side_sign = c_md_side_sign(side);

    // === Step 2 ===
    // The standard rules run first; a rejection returns immediately.
    if (c_dma_risk_profile_check(risk_profile, &check_state, &violation) != DMA_RISK_PASS) return violation;

    // === Step 3 ===
    // The ticker's custom rules run first, then the global rules, each in
    // registration order; a disabled rule is skipped and the first rejection
    // is returned.
    const dma_risk_limits* limits = c_dma_risk_profile_get_limits(risk_profile, ticker);

    for (const dma_risk_rule_ctx* rule = limits ? limits->rules : NULL; rule; rule = rule->next) {
        if (!rule->enabled) continue;
        if (rule->check_fn(rule, DMA, risk_profile, ticker, side, volume, price, rule->user_data, &violation) != DMA_RISK_PASS) return violation;
    }

    for (const dma_risk_rule_ctx* rule = risk_profile->globals.rules; rule; rule = rule->next) {
        if (!rule->enabled) continue;
        if (rule->check_fn(rule, DMA, risk_profile, ticker, side, volume, price, rule->user_data, &violation) != DMA_RISK_PASS) return violation;
    }

    return violation;
}

/**
 * @brief Check a volume on a side at the ticker's market price.
 * @param DMA The access (NULL-safe).
 * @param risk_profile The profile (NULL-safe).
 * @param ticker The instrument (non-empty).
 * @param side The trade side.
 * @param volume The volume to check.
 * @return The check's violation; a passing check reports DMA_RISK_PASS.
 */
static inline dma_risk_violation c_dma_risk_profile_check_volume(const dma* DMA, const dma_risk_profile* risk_profile, const char* ticker, md_side side, double volume) {
    double price = (DMA && DMA->mds) ? c_mds_get_market_price(DMA->mds, ticker) : NAN;

    return c_dma_risk_profile_check_limit(DMA, risk_profile, ticker, side, volume, price);
}

/**
 * @brief Check an algo's remaining target volume against the profile.
 * @param DMA The access (NULL-safe).
 * @param risk_profile The profile (NULL-safe).
 * @param algo The algo (NULL-safe).
 * @return The check's violation; a passing check reports DMA_RISK_PASS.
 */
static inline dma_risk_violation c_dma_risk_profile_check_algo(const dma* DMA, const dma_risk_profile* risk_profile, const dma_algo* algo) {
    dma_risk_violation violation;

    if (!DMA || !risk_profile || !algo || !algo->ticker) {
        c_dma_risk_violation_format(&violation, DMA_RISK_INVALID, "invalid risk_profile check input");
        return violation;
    }

    double remaining = algo->target_volume - algo->filled_volume;

    return c_dma_risk_profile_check_volume(DMA, risk_profile, algo->ticker, algo->side, remaining);
}

/**
 * @brief Check one order against the profile.
 *
 * The order's limit price is the checked price; a market order uses the
 * ticker's market price.
 *
 * @param DMA The access (NULL-safe).
 * @param risk_profile The profile (NULL-safe).
 * @param order The order (NULL-safe).
 * @return The check's violation; a passing check reports DMA_RISK_PASS.
 */
static inline dma_risk_violation c_dma_risk_profile_check_order(const dma* DMA, const dma_risk_profile* risk_profile, const md_trade_instruction* order) {
    dma_risk_violation violation;

    if (!DMA || !risk_profile || !order || !order->meta_info.ticker) {
        c_dma_risk_violation_format(&violation, DMA_RISK_INVALID, "invalid risk_profile check input");
        return violation;
    }

    double price = order->limit_price;
    if (!isfinite(price) || price <= 0.) price = DMA->mds ? c_mds_get_market_price(DMA->mds, order->meta_info.ticker) : NAN;

    return c_dma_risk_profile_check_limit(DMA, risk_profile, order->meta_info.ticker, order->side, order->volume, price);
}

#endif /* C_RISK_PROFILE_H */

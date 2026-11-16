#ifndef C_REFUSAL_WINDOW_H
#define C_REFUSAL_WINDOW_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include <algo_engine/engine/risk_profile/c_risk_profile.h>
#include <algo_engine/exchange_profile/c_ex_profile_base.h>
#include <cbase/allocator_protocol/c_allocator_protocol.h>

/** @brief The violation code a refusal-window rejection reports. */
static const int         REFUSAL_WINDOW_RULE_CODE = 6001;

/** @brief The rule name a refusal-window rejection reports. */
static const char* const REFUSAL_WINDOW_RULE_NAME = "refusal_window";

// ========== Enums ==========

/** @enum refusal_window_bound
 *  @brief The window bounds, one bit each in `refusal_window.enabled`. */
typedef enum refusal_window_bound {
    REFUSAL_WINDOW_BOUND_START = 1 << 0,
    REFUSAL_WINDOW_BOUND_END   = 1 << 1,
} refusal_window_bound;

/** @enum refusal_window_ret_code
 *  @brief 0 is success, negatives are errors. */
typedef enum refusal_window_ret_code {
    REFUSAL_WINDOW_OK                = 0,
    REFUSAL_WINDOW_ERR_INVALID_INPUT = -1,
    REFUSAL_WINDOW_ERR_OOM           = -2,
} refusal_window_ret_code;

// ========== Structs ==========

/**
 * @struct refusal_window
 * @brief The state of a custom rule that refuses trading in a session-time window.
 *
 * Trading is refused inside [start, end). The enabled mask says which bounds
 * are set: an end-only window refuses trading before it, a start-only window
 * refuses trading at or after it, and a window with both refuses the period.
 * Registering transfers the state to the rule node, which adopts it as an
 * owned child; this state is the node's `user_data` and dies with the node.
 * A state that was never registered is freed with `c_refusal_window_free`.
 */
typedef struct refusal_window {
    session_time_t start;    // Used when REFUSAL_WINDOW_BOUND_START is set.
    session_time_t end;      // Used when REFUSAL_WINDOW_BOUND_END is set; the end is exclusive.
    uint8_t        enabled;  // The bound mask: REFUSAL_WINDOW_BOUND_START / REFUSAL_WINDOW_BOUND_END.
} refusal_window;

// ========== Forward Declarations ==========

// Lifecycle
static inline refusal_window*    c_refusal_window_new(session_time_t* start, session_time_t* end, allocator_protocol* allocator);
static inline int                c_refusal_window_init(refusal_window* window, session_time_t* start, session_time_t* end);
static inline void               c_refusal_window_dealloc(refusal_window* window);
static inline void               c_refusal_window_free(refusal_window* window);

// Registration
static inline dma_risk_rule_ctx* c_refusal_window_register(dma_risk_profile* risk_profile, refusal_window* window, const char* ticker);
static inline dma_risk_rule_ctx* c_refusal_window_register_global(dma_risk_profile* risk_profile, refusal_window* window);

// Check
static inline dma_risk_type      c_refusal_window_check(const dma_risk_rule_ctx* rule, const dma* DMA, const dma_risk_profile* risk_profile, const char* ticker, md_side side, double volume, double price, void* user_data, dma_risk_violation* out);

// ========== Public APIs ==========

/**
 * @brief Allocate and initialize a refusal_window window.
 * @param start The session time the window opens, or NULL for an open start.
 * @param end The session time the window closes (exclusive), or NULL for an open end.
 * @param allocator Allocator for the window block.
 * @return The new window, or NULL on allocation failure or when both bounds are open.
 */
static inline refusal_window* c_refusal_window_new(session_time_t* start, session_time_t* end, allocator_protocol* allocator) {
    refusal_window* window = (refusal_window*) c_ap_alloc(sizeof(refusal_window), allocator);
    if (!window) return NULL;

    if (c_refusal_window_init(window, start, end) != REFUSAL_WINDOW_OK) {
        c_refusal_window_dealloc(window);
        c_ap_free(window);
        return NULL;
    }

    return window;
}

/**
 * @brief Initialize a caller-allocated, zeroed refusal_window window.
 *
 * A NULL bound leaves that side of the window open and clears its mask bit.
 *
 * @param window The zeroed window.
 * @param start The session time the window opens, or NULL for an open start.
 * @param end The session time the window closes (exclusive), or NULL for an open end.
 * @return REFUSAL_WINDOW_OK or REFUSAL_WINDOW_ERR_INVALID_INPUT.
 */
static inline int c_refusal_window_init(refusal_window* window, session_time_t* start, session_time_t* end) {
    if (!window || (!start && !end)) return REFUSAL_WINDOW_ERR_INVALID_INPUT;

    if (start) {
        window->start = *start;
        window->enabled |= REFUSAL_WINDOW_BOUND_START;
    }

    if (end) {
        window->end = *end;
        window->enabled |= REFUSAL_WINDOW_BOUND_END;
    }

    return REFUSAL_WINDOW_OK;
}

/**
 * @brief Tear down a caller-allocated refusal_window window, leaving it zeroed.
 * @param window The window (NULL-safe).
 */
static inline void c_refusal_window_dealloc(refusal_window* window) {
    if (!window) return;

    memset(window, 0, sizeof(refusal_window));
}

/**
 * @brief Free a refusal_window window that was never registered.
 *
 * A registered state belongs to its rule node and is released by the node's
 * teardown; freeing it directly would double-free.
 *
 * @param window The window (NULL-safe).
 */
static inline void c_refusal_window_free(refusal_window* window) {
    if (!window) return;

    c_refusal_window_dealloc(window);
    c_ap_free(window);
}

/**
 * @brief Register the window on a ticker and hand the state to the node.
 *
 * The profile composes the node and adopts the state under it, so the node's
 * teardown frees the state. The node owns the ticker copy.
 *
 * @param risk_profile The profile (NULL-safe).
 * @param window The window state (NULL-safe; transferred on success).
 * @param ticker The instrument (the node copies it).
 * @return The composed rule node, or NULL.
 */
static inline dma_risk_rule_ctx* c_refusal_window_register(dma_risk_profile* risk_profile, refusal_window* window, const char* ticker) {
    if (!risk_profile || !window || !ticker || !ticker[0]) return NULL;

    dma_risk_rule_ctx* node = c_dma_risk_profile_register_rule(risk_profile, ticker, c_refusal_window_check, REFUSAL_WINDOW_RULE_CODE, REFUSAL_WINDOW_RULE_NAME, window);
    if (!node) return NULL;

    c_ap_acquire_ownership(window, node);

    return node;
}

/**
 * @brief Register the window as a global rule on every ticker.
 *
 * A state without a ticker imposes the window on every ticker; one with a
 * ticker only on that ticker, even when it carries no limits. The profile
 * composes the node and adopts the state under it, so the node's teardown
 * frees the state.
 *
 * @param risk_profile The profile (NULL-safe).
 * @param window The window state (NULL-safe; transferred on success).
 * @return The composed rule node, or NULL.
 */
static inline dma_risk_rule_ctx* c_refusal_window_register_global(dma_risk_profile* risk_profile, refusal_window* window) {
    if (!risk_profile || !window) return NULL;

    dma_risk_rule_ctx* node = c_dma_risk_profile_register_global_rule(risk_profile, c_refusal_window_check, REFUSAL_WINDOW_RULE_CODE, REFUSAL_WINDOW_RULE_NAME, window);
    if (!node) return NULL;

    c_ap_acquire_ownership(window, node);

    return node;
}

/**
 * @brief The custom rule: refuse the ticker while the session clock is in the window.
 *
 * The check needs the access's session clock; without one it fails closed.
 *
 * @param rule The rule node (borrowed).
 * @param DMA The access (borrowed).
 * @param risk_profile The profile (borrowed; unused by this rule).
 * @param ticker The instrument being checked.
 * @param side The trade side (unused by this rule).
 * @param volume The order volume (unused by this rule).
 * @param price The checked price (unused by this rule).
 * @param user_data The rule state (a `refusal_window`, owned by the node).
 * @param out The violation filled on a rejection.
 * @return DMA_RISK_PASS, or DMA_RISK_CUSTOM with `out` filled.
 */
static inline dma_risk_type c_refusal_window_check(const dma_risk_rule_ctx* rule, const dma* DMA, const dma_risk_profile* risk_profile, const char* ticker, md_side side, double volume, double price, void* user_data, dma_risk_violation* out) {
    (void) risk_profile;
    (void) side;
    (void) volume;
    (void) price;

    const refusal_window* window = (const refusal_window*) user_data;

    if (!window || !ticker || !rule) {
        c_dma_risk_violation_format(out, DMA_RISK_CUSTOM, "refusal_window cannot judge <%s>: no rule state", ticker ? ticker : "(null)");
        if (out) out->code = rule ? rule->risk_rule_code : 0;
        return DMA_RISK_CUSTOM;
    }

    const char* name = rule->name ? rule->name : "custom";

    if (!DMA || !DMA->MDS) {
        c_dma_risk_violation_format(out, DMA_RISK_CUSTOM, "%s rejected! the %s rule needs a session clock", ticker, name);
        if (out) out->code = rule->risk_rule_code;
        return DMA_RISK_CUSTOM;
    }

    const session_time_t* now       = &DMA->MDS->ts.time;
    bool                  has_start = window->enabled & REFUSAL_WINDOW_BOUND_START;
    bool                  has_end   = window->enabled & REFUSAL_WINDOW_BOUND_END;

    if ((!has_start || now->elapsed_seconds >= window->start.elapsed_seconds)
        && (!has_end || now->elapsed_seconds < window->end.elapsed_seconds)) {
        if (has_start && has_end) {
            c_dma_risk_violation_format(
                out, DMA_RISK_CUSTOM,
                "%s rejected! the %s rule bans trading between %02u:%02u:%02u and %02u:%02u:%02u",
                ticker, name,
                (unsigned) window->start.hour, (unsigned) window->start.minute, (unsigned) window->start.second,
                (unsigned) window->end.hour, (unsigned) window->end.minute, (unsigned) window->end.second
            );
        }
        else if (has_start) {
            c_dma_risk_violation_format(
                out, DMA_RISK_CUSTOM,
                "%s rejected! the %s rule bans trading after %02u:%02u:%02u",
                ticker, name,
                (unsigned) window->start.hour, (unsigned) window->start.minute, (unsigned) window->start.second
            );
        }
        else if (has_end) {
            c_dma_risk_violation_format(
                out, DMA_RISK_CUSTOM,
                "%s rejected! the %s rule bans trading before %02u:%02u:%02u",
                ticker, name,
                (unsigned) window->end.hour, (unsigned) window->end.minute, (unsigned) window->end.second
            );
        }
        else {
            c_dma_risk_violation_format(out, DMA_RISK_CUSTOM, "%s rejected! the %s rule bans trading all day", ticker, name);
        }

        if (out) out->code = rule->risk_rule_code;
        return DMA_RISK_CUSTOM;
    }

    return DMA_RISK_PASS;
}

#endif /* C_REFUSAL_WINDOW_H */

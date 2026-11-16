#ifndef C_DMA_H
#define C_DMA_H

#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algo_engine/base/c_market_data/c_market_data.h>
#include <algo_engine/engine/c_balance.h>
#include <algo_engine/engine/c_inventory.h>
#include <algo_engine/engine/c_market_engine.h>
#include <cbase/allocator_protocol/c_allocator_protocol.h>
#include <cbase/bytemap/c_bytemap.h>

// ========== Constants ==========

#ifndef DMA_ORDER_QUEUE_INITIAL_CAPACITY
#define DMA_ORDER_QUEUE_INITIAL_CAPACITY 8U
#endif

#ifndef DMA_ORDER_QUEUE_MAX_CAPACITY
#define DMA_ORDER_QUEUE_MAX_CAPACITY 0xFFFFFFFFU
#endif

#if DMA_ORDER_QUEUE_INITIAL_CAPACITY < 1U
#error "DMA_ORDER_QUEUE_INITIAL_CAPACITY must be at least 1"
#endif

#if DMA_ORDER_QUEUE_MAX_CAPACITY < DMA_ORDER_QUEUE_INITIAL_CAPACITY
#error "DMA_ORDER_QUEUE_MAX_CAPACITY must be at least the initial capacity"
#endif

#ifndef DMA_NO_QUEUE_MODE
#define DMA_NO_QUEUE_MODE 1U
#endif

#if DMA_NO_QUEUE_MODE != 0U && DMA_NO_QUEUE_MODE != 1U
#error "DMA_NO_QUEUE_MODE must be 0 or 1"
#endif

#ifndef DMA_VIGILANT
#define DMA_VIGILANT 0
#endif

#if DMA_VIGILANT != 0 && DMA_VIGILANT != 1
#error "DMA_VIGILANT must be 0 or 1"
#endif

// ========== Enums ==========

// clang-format off

/** @enum dma_ret_code
 *  @brief 0 is success, positives are success-with-status, negatives are errors. */
typedef enum dma_ret_code {
    DMA_OK_NOOP           =  2,  // The pre-state is tolerated; no operation ran.
    DMA_OK_BUFFERED       =  1,  // The request was queued for the buffered path.
    DMA_OK                =  0,
    DMA_ERR_INVALID_INPUT = -1,
    DMA_ERR_OOM           = -2,
    DMA_ERR_NOT_FOUND     = -3,
    DMA_ERR_EMPTY         = -4,  // The request queue is empty.
    DMA_ERR_NO_RISK       = -5,  // No risk profile is bound; no transaction may leave.
    DMA_ERR_RISK          = -6,  // The risk profile refused the order.
    DMA_ERR_FULL          = -7,  // The request ring reached its configured maximum.
} dma_ret_code;

/** @enum dma_order_action
 *  @brief The buffered order actions. */
typedef enum dma_order_action {
    DMA_ACTION_LAUNCH     = 0,
    DMA_ACTION_CANCEL     = 1,
} dma_order_action;

// clang-format on

// ========== Structs ==========

typedef enum dma_algo_limit_mode  dma_algo_limit_mode;

typedef struct dma                dma;
typedef struct dma_algo           dma_algo;
typedef struct dma_algo_engine    dma_algo_engine;
typedef struct dma_position       dma_position;
typedef struct dma_risk_profile   dma_risk_profile;
typedef struct dma_risk_violation dma_risk_violation;

/**
 * @brief The required order handlers and the optional balance query.
 *
 * The launch and cancel handlers receive the access's per-send `seq_id`; the
 * balance query fills the access's balance and reports 0 when it supplied one.
 * `user_data` is always the last parameter.
 */
typedef void (*dma_order_launch_func)(md_trade_instruction* order, uint32_t seq_id, void* user_data);
typedef void (*dma_order_cancel_func)(const md_trade_instruction* order, uint32_t seq_id, void* user_data);
typedef void (*dma_order_filled_func)(const md_trade_instruction* order, md_trade_report* report, void* user_data);
typedef void (*dma_order_rejected_func)(const md_trade_instruction* order, void* user_data);
typedef int (*dma_balance_query)(const void* DMA, dma_balance* out, void* user_data);

/**
 * @struct dma_order_handler_ctx
 * @brief The handlers that post an order to the market.
 *
 * Launch and cancel are required; the balance query is optional and its absence
 * leaves the balance gate passing.
 */
typedef struct dma_order_handler_ctx {
    dma_order_launch_func order_launch_fn;
    dma_order_cancel_func order_cancel_fn;
    dma_balance_query     balance_check_fn;
} dma_order_handler_ctx;

/**
 * @struct dma_order_callbacks_ctx
 * @brief The optional event callbacks; they never dispatch an order.
 */
typedef struct dma_order_callbacks_ctx {
    dma_order_rejected_func order_rejected_fn;
    dma_order_filled_func   order_filled_fn;
} dma_order_callbacks_ctx;

/**
 * @struct dma_protocol
 * @brief The access's single attached protocol: required handlers, optional callbacks.
 */
typedef struct dma_protocol {
    dma_order_handler_ctx   handlers;
    dma_order_callbacks_ctx callbacks;
    void*                   user_data;
} dma_protocol;

/**
 * @struct dma_order_request
 * @brief One buffered launch/cancel request; the order is borrowed.
 */
typedef struct dma_order_request {
    md_trade_instruction* order;  // NOT owned - the order being routed.
    dma_order_action      action;
} dma_order_request;

/**
 * @struct dma_order_queue
 * @brief The contiguous FIFO ring of buffered requests.
 *
 * The buffer doubles when full, up to DMA_ORDER_QUEUE_MAX_CAPACITY; `head` is
 * the oldest entry and `size` is the live count. The buffer is its own owned
 * block, not a child of the access.
 */
typedef struct dma_order_queue {
    dma_order_request* entries;  // OWNED - contiguous ring buffer.
    size_t             capacity;
    size_t             head;
    size_t             size;
} dma_order_queue;

_Static_assert(DMA_ORDER_QUEUE_MAX_CAPACITY <= SIZE_MAX / sizeof(dma_order_request), "DMA_ORDER_QUEUE_MAX_CAPACITY overflows the ring byte size");

/**
 * @struct dma
 * @brief The direct market access state.
 *
 * The access owns the profile, inventory, balance, position map, protocol and
 * request ring it creates; the market data source and the algo engine are
 * borrowed and must outlive it.
 */
typedef struct dma {
    mds*              MDS;            // NOT owned - the market data source.
    dma_algo_engine*  ALGO_ENGINE;    // OWNED - created by c_dma_init.
    dma_risk_profile* RISK_PROFILE;   // OWNED - created by c_dma_init.
    dma_inventory*    INVENTORY;      // OWNED - created by c_dma_init.
    dma_balance*      BALANCE;        // OWNED - the balance the query handler fills (created by c_dma_init).
    bytemap*          positions;      // OWNED - position id -> owned dma_position* (created by c_dma_init).
    dma_protocol      protocol;       // OWNED - embedded handler and callback set.
    dma_order_queue   requests;       // OWNED - buffered launch/cancel ring.
    uint32_t          seq_id;         // Per-send sequence; incremented on launch and cancel.
    bool              balance_valid;  // The last query supplied a balance.
} dma;

// ========== Forward Declarations ==========

// Lifecycle
static inline dma*      c_dma_new(mds* MDS, allocator_protocol* allocator);
static inline int       c_dma_init(dma* DMA, mds* MDS, allocator_protocol* allocator);
static inline int       c_dma_clear(dma* DMA);
static inline void      c_dma_dealloc(dma* DMA);
static inline void      c_dma_free(dma* DMA);

// Positions (defined in c_position.h)
static inline void      c_dma_position_free(dma_position* position);

// Algo engine entries (the record and limit-mode types are defined by the includes above)
static inline dma_algo* c_dma_new_algo(dma* DMA, uint32_t algo_reg_id, const char* ticker, md_side side, double target_volume, const void* params, allocator_protocol* allocator);
static inline double    c_dma_calculate_limit(dma* DMA, const char* ticker, md_side side, double limit_price, double limit_adjust_factor, dma_algo_limit_mode mode);

// Protocol
static inline int       c_dma_attach_protocol_handler(dma* DMA, dma_order_launch_func order_launch_fn, dma_order_cancel_func order_cancel_fn, dma_balance_query balance_check_fn, void* user_data);
static inline int       c_dma_attach_protocol_callback(dma* DMA, dma_order_rejected_func order_rejected_fn, dma_order_filled_func order_filled_fn);
static inline int       c_dma_detach_protocol_handler(dma* DMA);
static inline int       c_dma_detach_protocol_callback(dma* DMA);
static inline int       c_dma_query_balance(dma* DMA);

// Order entry (hot path: one check and one dispatch per order request)
static inline void      c_dma_refuse(const dma* DMA, const md_trade_instruction* order, const char* op);
static inline int       c_dma_check_order(dma* DMA, const md_trade_instruction* order, dma_risk_violation* out_violation);
static inline uint32_t  c_dma_next_seq_id(dma* DMA);
static inline int       c_dma_launch_order(dma* DMA, md_trade_instruction* order);
static inline int       c_dma_cancel_order(dma* DMA, md_trade_instruction* order);
static inline int       c_dma_launch_order_unchecked(dma* DMA, md_trade_instruction* order);
static inline int       c_dma_process_queue(dma* DMA);

// Request queue
static inline int       c_dma_order_queue_init(dma_order_queue* queue, allocator_protocol* allocator);
static inline void      c_dma_order_queue_dealloc(dma_order_queue* queue);
static inline int       c_dma_order_queue_grow(dma_order_queue* queue);
static inline int       c_dma_order_queue_push(dma_order_queue* queue, dma_order_action action, md_trade_instruction* order);
static inline int       c_dma_order_queue_pop(dma_order_queue* queue, dma_order_action* out_action, md_trade_instruction** out_order);
static inline void      c_dma_order_queue_clear(dma_order_queue* queue);

#include <algo_engine/engine/algo/c_algo_template.h>
#include <algo_engine/engine/c_dma_algo.h>
#include <algo_engine/engine/c_position.h>
#include <algo_engine/engine/risk_profile/c_risk_profile.h>

// ========== Public APIs (Direct Market Access) ==========

/**
 * @brief Allocate and initialize a direct market access.
 *
 * Allocates the block and delegates to `c_dma_init`, which creates the owned
 * algo engine, risk profile, inventory, balance, position map and request
 * ring; the protocol starts detached.
 *
 * @param MDS The market data source (borrowed).
 * @param allocator Allocator for the block and its children.
 * @return The new access, or NULL on allocation failure.
 */
static inline dma* c_dma_new(mds* MDS, allocator_protocol* allocator) {
    dma* DMA = (dma*) c_ap_alloc(sizeof(dma), allocator);
    if (!DMA) return NULL;

    if (c_dma_init(DMA, MDS, allocator) != DMA_OK) {
        c_ap_free(DMA);
        return NULL;
    }

    return DMA;
}

/**
 * @brief Initialize a zeroed direct market access and its owned sub-blocks.
 *
 * Creates the algo engine, risk profile, inventory, balance, position map and
 * request ring. A failure releases everything through `c_dma_dealloc`, leaving
 * the block zeroed.
 *
 * @param DMA The zeroed access to initialize.
 * @param MDS The market data source (borrowed).
 * @param allocator Allocator for the children; NULL derives it from the block
 *        (the block must be an allocator block start).
 * @return DMA_OK, DMA_ERR_INVALID_INPUT, or DMA_ERR_OOM.
 */
static inline int c_dma_init(dma* DMA, mds* MDS, allocator_protocol* allocator) {
    if (!DMA) return DMA_ERR_INVALID_INPUT;

    DMA->MDS = MDS;

    allocator_protocol* protocol = allocator ? allocator : c_ap_protocol_from_ptr(DMA);

    DMA->ALGO_ENGINE = c_dma_algo_engine_new(protocol);
    if (!DMA->ALGO_ENGINE) goto fail;

    DMA->RISK_PROFILE = c_dma_risk_profile_new(protocol);
    if (!DMA->RISK_PROFILE) goto fail;

    DMA->INVENTORY = c_dma_inventory_new(protocol);
    if (!DMA->INVENTORY) goto fail;

    DMA->BALANCE = c_balance_new(protocol);
    if (!DMA->BALANCE) goto fail;

    DMA->positions = c_bytemap_ex_new(8, sizeof(dma_position*), protocol);
    if (!DMA->positions) goto fail;

    if (c_dma_order_queue_init(&DMA->requests, protocol) != DMA_OK) goto fail;

    return DMA_OK;

fail:
    c_dma_dealloc(DMA);
    return DMA_ERR_OOM;
}

/**
 * @brief Remove every algo record, order and stored trade the bound algo engine tracks.
 *
 * The engine belongs to the access, so the sweep covers every record in it.
 * It frees every order context (with the trade reports it stored), every algo
 * record, and every registration index entry; the hosted positions drop the
 * references to those records in the same pass. The engine's maps, the
 * registrations and the access's own configuration (handlers, inventory,
 * positions, request ring) survive, so the access can trade again after the
 * sweep.
 *
 * @param DMA The access (NULL-safe).
 * @return DMA_OK, or DMA_ERR_INVALID_INPUT.
 */
static inline int c_dma_clear(dma* DMA) {
    if (!DMA) return DMA_ERR_INVALID_INPUT;

    c_dma_order_queue_clear(&DMA->requests);

    if (DMA->ALGO_ENGINE) c_dma_algo_engine_clear(DMA->ALGO_ENGINE);

    return DMA_OK;
}

/**
 * @brief Tear down a caller-allocated access, leaving it zeroed.
 *
 * The trading records are removed first (see `c_dma_clear`); every owned
 * position, the request ring, the balance, the inventory, the risk profile and
 * the algo engine are released next. The borrowed market data block stays with
 * its owner; the algo engine is released after the positions because the sweep
 * and the position teardown read it.
 *
 * @param DMA The access (NULL-safe).
 */
static inline void c_dma_dealloc(dma* DMA) {
    if (!DMA) return;

    c_dma_clear(DMA);

    c_dma_order_queue_dealloc(&DMA->requests);

    if (DMA->positions) {
        while (c_bytemap_ex_len(DMA->positions)) {
            bytemap_entry* entry    = DMA->positions->first;
            dma_position*  position = *(dma_position**) entry->value;
            uuid_t         position_id;

            memcpy(position_id, entry->key, sizeof(uuid_t));
            c_bytemap_ex_pop(DMA->positions, (const char*) position_id, sizeof(uuid_t), 0, NULL, NULL);

            if (position) c_dma_position_free(position);
        }

        c_bytemap_ex_free(DMA->positions);
        DMA->positions = NULL;
    }

    if (DMA->ALGO_ENGINE) {
        c_dma_algo_engine_free(DMA->ALGO_ENGINE);
        DMA->ALGO_ENGINE = NULL;
    }

    if (DMA->BALANCE) {
        c_balance_free(DMA->BALANCE);
        DMA->BALANCE = NULL;
    }

    if (DMA->INVENTORY) {
        c_dma_inventory_free(DMA->INVENTORY);
        DMA->INVENTORY = NULL;
    }

    if (DMA->RISK_PROFILE) {
        c_dma_risk_profile_free(DMA->RISK_PROFILE);
        DMA->RISK_PROFILE = NULL;
    }

    memset(DMA, 0, sizeof(dma));
}

/**
 * @brief Free an access allocated by `c_dma_new`.
 *
 * The sweep and the subordinate release run first, then the access block is
 * freed.
 *
 * @param DMA The access (NULL-safe).
 */
static inline void c_dma_free(dma* DMA) {
    if (!DMA) return;

    c_dma_dealloc(DMA);
    c_ap_free(DMA);
}

// ========== Public APIs (Algo Engine) ==========

/**
 * @brief Build an algo of the registered kind through this access's engine.
 *
 * The registration id resolves the kind; the factory builds the record and the
 * engine records it under that registration in the same pass.
 *
 * @param DMA The access (NULL-safe; its algo engine must be bound).
 * @param algo_reg_id The registration id of the kind.
 * @param ticker The instrument.
 * @param side The trade side.
 * @param target_volume The volume to trade.
 * @param params Algo-kind parameters (borrowed).
 * @param allocator Allocator for the record.
 * @return The new record, or NULL.
 */
static inline dma_algo* c_dma_new_algo(dma* DMA, uint32_t algo_reg_id, const char* ticker, md_side side, double target_volume, const void* params, allocator_protocol* allocator) {
    if (!DMA || !DMA->ALGO_ENGINE || !ticker) return NULL;

    dma_algo_ctx* registration = c_dma_algo_registry_get(DMA->ALGO_ENGINE->registry, algo_reg_id);

    if (!registration || !registration->algo_new_fn || !registration->algos) return NULL;

    dma_algo* algo = registration->algo_new_fn(DMA, ticker, side, target_volume, params, allocator);
    if (!algo) return NULL;

    algo->algo_reg_id = registration->reg_id;

    int ret_code = c_bytemap_ex_set_default(registration->algos, (const char*) algo->algo_id, sizeof(uuid_t), (const char*) &algo, sizeof(dma_algo*), 0, NULL, NULL);

    if (ret_code != BYTEMAP_OK) {
        c_dma_algo_free(algo);
        return NULL;
    }

    return algo;
}

/**
 * @brief Calculate the order limit from the access's market price and the algo limits.
 *
 * The absolute limit, the market factor and the market price are combined;
 * a missing market price yields NaN.
 *
 * @param DMA The access (NULL-safe; its market data source must be bound).
 * @param ticker The instrument (non-empty).
 * @param side The trade side.
 * @param limit_price The absolute limit (NaN when unset).
 * @param limit_adjust_factor The market-price factor (NaN when unset).
 * @param mode Strict or loose.
 * @return The calculated limit, or NaN.
 */
static inline double c_dma_calculate_limit(dma* DMA, const char* ticker, md_side side, double limit_price, double limit_adjust_factor, dma_algo_limit_mode mode) {
    if (!DMA || !DMA->MDS || !ticker) return NAN;

    double market_price = c_mds_get_market_price(DMA->MDS, ticker);
    if (isnan(market_price)) return NAN;

    double calculated = c_dma_algo_compare_limit(side, limit_price, NAN, mode);

    if (!isnan(limit_adjust_factor)) {
        double adjusted = market_price * (1. + limit_adjust_factor * c_md_side_sign(side));
        calculated      = c_dma_algo_compare_limit(side, adjusted, calculated, mode);
    }

    calculated = c_dma_algo_compare_limit(side, market_price, calculated, mode);
    return calculated;
}

/**
 * @brief Attach the access's required handlers, replacing any attached.
 *
 * An exchange connection in production and a simulated matcher in backtest
 * both attach a handler set; the access dispatches to exactly one. The balance
 * query is optional: without it the balance gate passes.
 *
 * @param DMA The access (NULL-safe).
 * @param order_launch_fn The launch handler (NULL is refused).
 * @param order_cancel_fn The cancel handler (NULL is refused).
 * @param balance_check_fn The balance query (optional; NULL passes the gate).
 * @param user_data Opaque value handed to every handler.
 * @return DMA_OK, or DMA_ERR_INVALID_INPUT.
 */
static inline int c_dma_attach_protocol_handler(dma* DMA, dma_order_launch_func order_launch_fn, dma_order_cancel_func order_cancel_fn, dma_balance_query balance_check_fn, void* user_data) {
    if (!DMA || !order_launch_fn || !order_cancel_fn) return DMA_ERR_INVALID_INPUT;

    DMA->protocol.handlers.order_launch_fn  = order_launch_fn;
    DMA->protocol.handlers.order_cancel_fn  = order_cancel_fn;
    DMA->protocol.handlers.balance_check_fn = balance_check_fn;
    DMA->protocol.user_data                 = user_data;
    return DMA_OK;
}

/**
 * @brief Attach the access's optional event callbacks; either may be NULL.
 * @param DMA The access (NULL-safe).
 * @param order_rejected_fn The rejected callback (optional).
 * @param order_filled_fn The filled callback (optional).
 * @return DMA_OK, or DMA_ERR_INVALID_INPUT.
 */
static inline int c_dma_attach_protocol_callback(dma* DMA, dma_order_rejected_func order_rejected_fn, dma_order_filled_func order_filled_fn) {
    if (!DMA) return DMA_ERR_INVALID_INPUT;

    DMA->protocol.callbacks.order_rejected_fn = order_rejected_fn;
    DMA->protocol.callbacks.order_filled_fn   = order_filled_fn;
    return DMA_OK;
}

/**
 * @brief Detach the access's required handlers; idempotent.
 * @param DMA The access (NULL-safe).
 * @return DMA_OK, or DMA_ERR_INVALID_INPUT.
 */
static inline int c_dma_detach_protocol_handler(dma* DMA) {
    if (!DMA) return DMA_ERR_INVALID_INPUT;

    memset(&DMA->protocol.handlers, 0, sizeof(DMA->protocol.handlers));
    DMA->protocol.user_data = NULL;
    DMA->balance_valid      = false;
    return DMA_OK;
}

/**
 * @brief Detach the access's optional callbacks; idempotent.
 * @param DMA The access (NULL-safe).
 * @return DMA_OK, or DMA_ERR_INVALID_INPUT.
 */
static inline int c_dma_detach_protocol_callback(dma* DMA) {
    if (!DMA) return DMA_ERR_INVALID_INPUT;

    memset(&DMA->protocol.callbacks, 0, sizeof(DMA->protocol.callbacks));
    return DMA_OK;
}

/**
 * @brief Refresh the access's balance through the attached query, if any.
 *
 * The handler fills the access's balance and reports 0 when it supplied one;
 * a missing handler or a failed query clears the valid flag, so the gate passes.
 *
 * @param DMA The access (NULL-safe).
 * @return The query's code, DMA_OK_NOOP without a query, or DMA_ERR_INVALID_INPUT.
 */
static inline int c_dma_query_balance(dma* DMA) {
    if (!DMA || !DMA->BALANCE) return DMA_ERR_INVALID_INPUT;

    DMA->balance_valid = false;

    if (!DMA->protocol.handlers.balance_check_fn) return DMA_OK_NOOP;

    memset(DMA->BALANCE, 0, sizeof(dma_balance));

    int ret_code       = DMA->protocol.handlers.balance_check_fn(DMA, DMA->BALANCE, DMA->protocol.user_data);
    DMA->balance_valid = (ret_code == DMA_OK);
    return ret_code;
}

/**
 * @brief Refuse a forbidden order transition: report it, and stop under vigil.
 *
 * The order is left unchanged. The default prints the cause to stderr; with
 * DMA_VIGILANT set the access prints the order's telemetry and aborts.
 *
 * @param DMA The access (may be NULL).
 * @param order The order (may be NULL).
 * @param op The operation name.
 */
static inline void c_dma_refuse(const dma* DMA, const md_trade_instruction* order, const char* op) {
    const char* ticker = (order && order->meta_info.ticker) ? order->meta_info.ticker : "<unknown>";
    const char* state  = order ? c_md_state_name(order->order_state) : "<null>";

    fprintf(stderr, "[DMA] ERROR: %s refused; order <%s> state <%s> is banned\n", op, ticker, state);

#if DMA_VIGILANT
    fprintf(
        stderr,
        "[DMA] VIGILANT: %s refused; ticker <%s> state <%s> side <%d> volume <%.6f> filled <%.6f> notional <%.6f> fee <%.6f> seq <%u> order <%p>\n",
        op,
        ticker,
        state,
        order ? (int) order->side : -1,
        order ? order->volume : 0.,
        order ? order->filled_volume : 0.,
        order ? order->filled_notional : 0.,
        order ? order->fee : 0.,
        DMA ? DMA->seq_id : 0U,
        (const void*) order
    );
    abort();
#else
    (void) DMA;
#endif
}

/**
 * @brief Dry-run the validity and risk checks for an order.
 *
 * HOT PATH: runs once per order request. It allocates nothing and logs
 * nothing; the optional balance query is the one host callback.
 *
 * A missing risk profile and any risk violation refuse the order. The check
 * does not dispatch and does not change the order.
 *
 * @param DMA The access (non-NULL for the balance query; NULL-safe).
 * @param order The order (NULL-safe).
 * @param out_violation Optional: receives the risk violation on refusal.
 * @return DMA_OK when the order may be sent, DMA_ERR_NO_RISK, DMA_ERR_RISK,
 *         or DMA_ERR_INVALID_INPUT.
 */
static inline int c_dma_check_order(dma* DMA, const md_trade_instruction* order, dma_risk_violation* out_violation) {
    if (out_violation) {
        out_violation->type   = DMA_RISK_PASS;
        out_violation->code   = 0;
        out_violation->msg[0] = '\0';
    }

    if (!DMA || !order) return DMA_ERR_INVALID_INPUT;
    if (!DMA->RISK_PROFILE) return DMA_ERR_NO_RISK;

    c_dma_query_balance(DMA);

    dma_risk_violation violation = c_dma_risk_profile_check_order(DMA, DMA->RISK_PROFILE, order);

    if (out_violation) *out_violation = violation;

    if (violation.type != DMA_RISK_PASS) return DMA_ERR_RISK;

    return DMA_OK;
}

// ========== Public APIs (Request Queue) ==========

/**
 * @brief Initialize the request ring with its initial contiguous buffer.
 * @param queue The zeroed queue.
 * @param allocator Allocator for the buffer.
 * @return DMA_OK, DMA_ERR_INVALID_INPUT, or DMA_ERR_OOM.
 */
static inline int c_dma_order_queue_init(dma_order_queue* queue, allocator_protocol* allocator) {
    if (!queue) return DMA_ERR_INVALID_INPUT;

    queue->entries = (dma_order_request*) c_ap_alloc(sizeof(dma_order_request) * DMA_ORDER_QUEUE_INITIAL_CAPACITY, allocator);
    if (!queue->entries) return DMA_ERR_OOM;

    queue->capacity = DMA_ORDER_QUEUE_INITIAL_CAPACITY;
    queue->head     = 0;
    queue->size     = 0;
    return DMA_OK;
}

/**
 * @brief Release the request ring's buffer, leaving the queue zeroed.
 * @param queue The queue (NULL-safe).
 */
static inline void c_dma_order_queue_dealloc(dma_order_queue* queue) {
    if (!queue) return;

    if (queue->entries) c_ap_free(queue->entries);

    memset(queue, 0, sizeof(dma_order_queue));
}

/**
 * @brief Double the ring's contiguous buffer, linearizing the live entries.
 *
 * The buffer is one allocator block; growth allocates a larger block, copies
 * the live entries in FIFO order, then frees the old block. The allocator is
 * derived from the current buffer.
 *
 * @param queue The queue (NULL-safe; initialized).
 * @return DMA_OK, DMA_ERR_INVALID_INPUT, DMA_ERR_FULL, or DMA_ERR_OOM.
 */
static inline int c_dma_order_queue_grow(dma_order_queue* queue) {
    if (!queue || !queue->entries) return DMA_ERR_INVALID_INPUT;
    if (queue->capacity >= DMA_ORDER_QUEUE_MAX_CAPACITY) return DMA_ERR_FULL;

    size_t              next_capacity = queue->capacity < (size_t) DMA_ORDER_QUEUE_MAX_CAPACITY / 2
                                            ? queue->capacity * 2
                                            : (size_t) DMA_ORDER_QUEUE_MAX_CAPACITY;

    allocator_protocol* allocator = c_ap_protocol_from_ptr(queue->entries);
    dma_order_request*  entries   = (dma_order_request*) c_ap_alloc(sizeof(dma_order_request) * next_capacity, allocator);
    if (!entries) return DMA_ERR_OOM;

    for (size_t index = 0; index < queue->size; index++) {
        entries[index] = queue->entries[(queue->head + index) % queue->capacity];
    }

    c_ap_free(queue->entries);
    queue->entries  = entries;
    queue->capacity = next_capacity;
    queue->head     = 0;
    return DMA_OK;
}

/**
 * @brief Append one request, doubling the ring when full.
 * @param queue The queue (NULL-safe; initialized).
 * @param action The action to buffer.
 * @param order The order to route (NULL is refused).
 * @return DMA_OK, DMA_ERR_INVALID_INPUT, DMA_ERR_FULL, or DMA_ERR_OOM.
 */
static inline int c_dma_order_queue_push(dma_order_queue* queue, dma_order_action action, md_trade_instruction* order) {
    if (!queue || !order) return DMA_ERR_INVALID_INPUT;

    if (queue->size == queue->capacity) {
        int ret_code = c_dma_order_queue_grow(queue);
        if (ret_code != DMA_OK) return ret_code;
    }

    size_t tail = (queue->head + queue->size) % queue->capacity;

    queue->entries[tail].order  = order;
    queue->entries[tail].action = action;
    queue->size++;
    return DMA_OK;
}

/**
 * @brief Pop the oldest request.
 * @param queue The queue (NULL-safe).
 * @param out_action Optional: receives the action.
 * @param out_order Optional: receives the order (still borrowed).
 * @return DMA_OK, DMA_ERR_INVALID_INPUT, or DMA_ERR_EMPTY.
 */
static inline int c_dma_order_queue_pop(dma_order_queue* queue, dma_order_action* out_action, md_trade_instruction** out_order) {
    if (!queue) return DMA_ERR_INVALID_INPUT;
    if (!queue->size) return DMA_ERR_EMPTY;

    dma_order_request* entry = &queue->entries[queue->head];

    if (out_action) *out_action = entry->action;
    if (out_order) *out_order = entry->order;

    queue->head = (queue->head + 1) % queue->capacity;
    queue->size--;
    return DMA_OK;
}

/**
 * @brief Drop every buffered request without releasing the buffer.
 * @param queue The queue (NULL-safe).
 */
static inline void c_dma_order_queue_clear(dma_order_queue* queue) {
    if (!queue) return;

    queue->head = 0;
    queue->size = 0;
}

/**
 * @brief The engine's order context for an instruction, or NULL.
 *
 * Only an order built by `c_dma_algo_engine_new_order` is a `dma_order_ctx`;
 * a bare instruction is not, so the access must not treat it as one.
 *
 * @param engine The access's engine (borrowed; NULL-safe).
 * @param order The instruction (NULL-safe).
 * @return The stored context, or NULL when the engine does not own the order.
 */
static inline dma_order_ctx* c_dma_order_ctx_lookup(dma_algo_engine* engine, const md_trade_instruction* order) {
    if (!engine || !order) return NULL;

    char* stored = NULL;
    if (c_bytemap_ex_get_ptr(engine->orders, (const char*) &order->order_id, sizeof(long_md_id), &stored, NULL) != BYTEMAP_OK) return NULL;

    dma_order_ctx* ctx = *(dma_order_ctx**) stored;
    return ctx == (const dma_order_ctx*) order ? ctx : NULL;
}

// ========== Public APIs (Order Entry: Hot Path) ==========

// Order states: creation is STATE_IDLE; a direct launch writes STATE_SENT (or
// STATE_REJECTED on a failed check) and a queued launch STATE_PENDING; a cancel
// is local for IDLE/PENDING and writes STATE_CANCELING otherwise. Every other
// pre-state is banned: no operation, an error code and stderr. Each order is
// built by c_dma_algo_engine_new_order, so the instruction starts its context.

/**
 * @brief Advance and return the access's per-send sequence.
 * @param DMA The access (non-NULL).
 * @return The next sequence; 0 is skipped.
 */
static inline uint32_t c_dma_next_seq_id(dma* DMA) {
    DMA->seq_id++;
    if (!DMA->seq_id) DMA->seq_id = 1;

    return DMA->seq_id;
}

/**
 * @brief Launch an order: direct dispatch in no-queue mode, otherwise enqueue.
 *
 * STATE_IDLE is the explicit start. A queued launch moves to STATE_PENDING
 * until c_dma_process_queue checks and sends it. In queue mode a repeated
 * launch on STATE_PENDING is tolerated: it prints a warning and returns
 * DMA_OK_NOOP. Every other pre-state is banned. Build with
 * DMA_NO_QUEUE_MODE=0 for the queueing variant.
 *
 * @param DMA The access (NULL-safe).
 * @param order The order (NULL-safe; must be STATE_IDLE; an engine order).
 * @return DMA_OK when sent, DMA_OK_BUFFERED when queued, DMA_OK_NOOP for a
 *         tolerated repeat, DMA_ERR_INVALID_INPUT for a banned state, or a
 *         check error code.
 */
static inline int c_dma_launch_order(dma* DMA, md_trade_instruction* order) {
    if (!DMA || !order) return DMA_ERR_INVALID_INPUT;

    md_order_state state = order->order_state;

    if (state == STATE_IDLE) {
#if DMA_NO_QUEUE_MODE
        dma_risk_violation violation;
        int                ret_code = c_dma_check_order(DMA, order, &violation);

        if (ret_code != DMA_OK) {
            order->order_state = STATE_REJECTED;
            c_dma_algo_engine_on_order_rejected_inplace(DMA->ALGO_ENGINE, c_dma_order_ctx_lookup(DMA->ALGO_ENGINE, order));
            if (DMA->protocol.callbacks.order_rejected_fn) DMA->protocol.callbacks.order_rejected_fn(order, DMA->protocol.user_data);
            return ret_code;
        }
        else {
            uint32_t seq_id    = c_dma_next_seq_id(DMA);
            order->order_state = STATE_SENT;
            c_dma_algo_engine_on_order_launched_inplace(DMA->ALGO_ENGINE, c_dma_order_ctx_lookup(DMA->ALGO_ENGINE, order));
            if (DMA->protocol.handlers.order_launch_fn) DMA->protocol.handlers.order_launch_fn(order, seq_id, DMA->protocol.user_data);
            return DMA_OK;
        }
#else
        order->order_state = STATE_PENDING;
        int ret_code       = c_dma_order_queue_push(&DMA->requests, DMA_ACTION_LAUNCH, order);
        if (ret_code != DMA_OK) {
            order->order_state = STATE_IDLE;
            return ret_code;
        }
        c_dma_algo_engine_on_order_pending_inplace(DMA->ALGO_ENGINE, c_dma_order_ctx_lookup(DMA->ALGO_ENGINE, order));
        return DMA_OK_BUFFERED;
#endif
    }

#if !DMA_NO_QUEUE_MODE
    if (state == STATE_PENDING) {
        const char* ticker = order->meta_info.ticker ? order->meta_info.ticker : "<unknown>";
        fprintf(stderr, "[DMA] WARNING: launch ignored; order <%s> is already queued\n", ticker);
        return DMA_OK_NOOP;
    }
#endif

    c_dma_refuse(DMA, order, "launch");
    return DMA_ERR_INVALID_INPUT;
}

/**
 * @brief Cancel an order: local for an unsubmitted order, otherwise dispatch.
 *
 * STATE_IDLE and STATE_PENDING have not left the access: they are marked
 * STATE_CANCELED locally with no payload and no handler. STATE_SENT, STATE_PLACED
 * and STATE_PARTFILLED dispatch a cancel; a queued cancel records
 * STATE_CANCELING at enqueue. STATE_CANCELING is tolerated and returns
 * DMA_OK_NOOP. Every other pre-state is banned.
 *
 * @param DMA The access (NULL-safe).
 * @param order The order (NULL-safe; an engine order).
 * @return DMA_OK when canceled locally or sent, DMA_OK_BUFFERED when queued,
 *         DMA_OK_NOOP for an already-canceling order, or
 *         DMA_ERR_INVALID_INPUT for a banned state.
 */
static inline int c_dma_cancel_order(dma* DMA, md_trade_instruction* order) {
    if (!DMA || !order) return DMA_ERR_INVALID_INPUT;

    md_order_state state = order->order_state;

    if (state == STATE_IDLE || state == STATE_PENDING) {
        order->order_state = STATE_CANCELED;
        c_dma_algo_engine_on_order_canceled_inplace(DMA->ALGO_ENGINE, c_dma_order_ctx_lookup(DMA->ALGO_ENGINE, order));
        return DMA_OK;
    }

    if (c_md_state_cancellable(state)) {
#if DMA_NO_QUEUE_MODE
        uint32_t seq_id    = c_dma_next_seq_id(DMA);
        order->order_state = STATE_CANCELING;
        c_dma_algo_engine_on_order_canceling_inplace(DMA->ALGO_ENGINE, c_dma_order_ctx_lookup(DMA->ALGO_ENGINE, order));
        if (DMA->protocol.handlers.order_cancel_fn) DMA->protocol.handlers.order_cancel_fn(order, seq_id, DMA->protocol.user_data);
        return DMA_OK;
#else
        order->order_state = STATE_CANCELING;
        int ret_code       = c_dma_order_queue_push(&DMA->requests, DMA_ACTION_CANCEL, order);
        if (ret_code != DMA_OK) {
            order->order_state = state;
            return ret_code;
        }
        c_dma_algo_engine_on_order_canceling_inplace(DMA->ALGO_ENGINE, c_dma_order_ctx_lookup(DMA->ALGO_ENGINE, order));
        return DMA_OK_BUFFERED;
#endif
    }

    if (state == STATE_CANCELING) return DMA_OK_NOOP;

    c_dma_refuse(DMA, order, "cancel");
    return DMA_ERR_INVALID_INPUT;
}

/**
 * @brief Launch an order without the validity or risk check.
 *
 * STATE_IDLE is the only accepted pre-state; it moves to STATE_SENT and
 * reaches the launch handler. A warning is printed to stderr because no risk
 * gate ran. Every other state is banned: the unchecked path bypasses the
 * queue, so a queued order must not be sent twice.
 *
 * @param DMA The access (NULL-safe).
 * @param order The order (NULL-safe; must be STATE_IDLE; an engine order).
 * @return DMA_OK or DMA_ERR_INVALID_INPUT.
 */
static inline int c_dma_launch_order_unchecked(dma* DMA, md_trade_instruction* order) {
    if (!DMA || !order) return DMA_ERR_INVALID_INPUT;

    if (order->order_state != STATE_IDLE) {
        c_dma_refuse(DMA, order, "unchecked launch");
        return DMA_ERR_INVALID_INPUT;
    }

    const char* ticker = order->meta_info.ticker ? order->meta_info.ticker : "<unknown>";
    fprintf(stderr, "[DMA] WARNING: unchecked launch of <%s>; no validity or risk check ran\n", ticker);
    uint32_t seq_id    = c_dma_next_seq_id(DMA);
    order->order_state = STATE_SENT;
    c_dma_algo_engine_on_order_launched_inplace(DMA->ALGO_ENGINE, c_dma_order_ctx_lookup(DMA->ALGO_ENGINE, order));
    if (DMA->protocol.handlers.order_launch_fn) DMA->protocol.handlers.order_launch_fn(order, seq_id, DMA->protocol.user_data);
    return DMA_OK;
}

/**
 * @brief Process every queued request in FIFO order.
 *
 * A queued launch is checked at this point: STATE_PENDING reaches the launch
 * handler when it passes and the rejected callback otherwise; STATE_CANCELED and
 * STATE_PLACED are stale entries and are dropped. A queued cancel sends for
 * STATE_CANCELING; STATE_FILLED and STATE_CANCELED are resolved and dropped.
 * Any other state is banned and stops the drain with an error. The handler
 * itself owns any pacing or cool-down.
 *
 * @param DMA The access (NULL-safe).
 * @return The number of queued requests processed, or DMA_ERR_INVALID_INPUT
 *         when a banned state stops the drain.
 */
static inline int c_dma_process_queue(dma* DMA) {
    if (!DMA) return DMA_ERR_INVALID_INPUT;

    int                   processed = 0;
    dma_order_action      action;
    md_trade_instruction* order = NULL;

    while (c_dma_order_queue_pop(&DMA->requests, &action, &order) == DMA_OK) {
        md_order_state state = order->order_state;

        if (action == DMA_ACTION_LAUNCH) {
            if (state == STATE_PENDING) {
                dma_risk_violation violation;
                int                ret_code = c_dma_check_order(DMA, order, &violation);

                if (ret_code != DMA_OK) {
                    order->order_state = STATE_REJECTED;
                    c_dma_algo_engine_on_order_rejected_inplace(DMA->ALGO_ENGINE, c_dma_order_ctx_lookup(DMA->ALGO_ENGINE, order));
                    if (DMA->protocol.callbacks.order_rejected_fn) DMA->protocol.callbacks.order_rejected_fn(order, DMA->protocol.user_data);
                }
                else {
                    uint32_t seq_id    = c_dma_next_seq_id(DMA);
                    order->order_state = STATE_SENT;
                    c_dma_algo_engine_on_order_launched_inplace(DMA->ALGO_ENGINE, c_dma_order_ctx_lookup(DMA->ALGO_ENGINE, order));
                    if (DMA->protocol.handlers.order_launch_fn) DMA->protocol.handlers.order_launch_fn(order, seq_id, DMA->protocol.user_data);
                }
            }
            else if (state != STATE_CANCELED && state != STATE_PLACED) {
                c_dma_refuse(DMA, order, "queue launch");
                return DMA_ERR_INVALID_INPUT;
            }
        }
        else if (action == DMA_ACTION_CANCEL) {
            if (state == STATE_CANCELING) {
                uint32_t seq_id    = c_dma_next_seq_id(DMA);
                order->order_state = STATE_CANCELING;
                c_dma_algo_engine_on_order_canceling_inplace(DMA->ALGO_ENGINE, c_dma_order_ctx_lookup(DMA->ALGO_ENGINE, order));
                if (DMA->protocol.handlers.order_cancel_fn) DMA->protocol.handlers.order_cancel_fn(order, seq_id, DMA->protocol.user_data);
            }
            else if (state != STATE_FILLED && state != STATE_CANCELED) {
                c_dma_refuse(DMA, order, "queue cancel");
                return DMA_ERR_INVALID_INPUT;
            }
        }

        processed++;
    }

    return processed;
}

#endif /* C_DMA_H */

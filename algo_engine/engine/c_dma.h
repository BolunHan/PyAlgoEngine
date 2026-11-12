#ifndef C_DMA_H
#define C_DMA_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include <algo_engine/base/c_market_data/c_market_data.h>
#include <algo_engine/engine/c_inventory.h>
#include <algo_engine/engine/c_market_engine.h>
#include <algo_engine/engine/c_risk_profile.h>
#include <cbase/allocator_protocol/c_allocator_protocol.h>
#include <cbase/bytemap/c_bytemap.h>

// ========== Enums ==========

/** @enum dma_ret_code
 *  @brief 0 is success, positives are success-with-status, negatives are errors. */
typedef enum dma_ret_code {
    DMA_OK_BUFFERED       = 1,  // The request was queued for the buffered path.
    DMA_OK                = 0,
    DMA_ERR_INVALID_INPUT = -1,
    DMA_ERR_OOM           = -2,
    DMA_ERR_NOT_FOUND     = -3,
    DMA_ERR_EMPTY         = -4,  // The request queue is empty.
    DMA_ERR_DISABLED      = -5,  // The access is not enabled; no transaction may leave.
    DMA_ERR_NO_RISK       = -6,  // No risk profile is bound; no transaction may leave.
    DMA_ERR_RISK          = -7,  // The risk profile refused the order.
} dma_ret_code;

/** @enum dma_order_action
 *  @brief The buffered order actions. */
typedef enum dma_order_action {
    DMA_ACTION_LAUNCH = 0,
    DMA_ACTION_CANCEL = 1,
} dma_order_action;

// ========== Structs ==========

typedef struct dma             dma;
typedef struct dma_position    dma_position;
typedef struct dma_algo_engine dma_algo_engine;

/**
 * @brief The four per-action trading interface handlers.
 *
 * Only the launch handler carries the registration's `seq_id`; `user_data` is
 * always the last parameter.
 */
typedef void (*dma_order_launch_func)(md_trade_instruction* order, uint32_t seq_id, void* user_data);
typedef void (*dma_order_cancel_func)(const md_trade_instruction* order, void* user_data);
typedef void (*dma_order_reject_func)(const md_trade_instruction* order, void* user_data);
typedef void (*dma_order_filled_callback)(const md_trade_instruction* order, md_trade_report* report, void* user_data);

/**
 * @struct dma_order_handler_ctx
 * @brief One registered handler set; a calloc'ed linked-list node.
 */
typedef struct dma_order_handler_ctx {
    dma_order_launch_func         order_launch_fn;
    dma_order_cancel_func         order_cancel_fn;
    dma_order_reject_func         order_reject_fn;
    dma_order_filled_callback     order_filled_fn;
    void*                         user_data;
    struct dma_order_handler_ctx* next;
    uintptr_t                     reg_id;
    uint32_t                      seq_id;
} dma_order_handler_ctx;

/**
 * @struct dma_order_request
 * @brief One buffered launch/cancel request; the order is borrowed.
 */
typedef struct dma_order_request {
    md_trade_instruction*     order;  // NOT owned - the order being routed.
    struct dma_order_request* next;
    dma_order_action          action;
} dma_order_request;

/**
 * @struct dma
 * @brief The direct market access state.
 *
 * `mds`, `algo_engine`, `risk` and `position` are borrowed; `inventory`,
 * `handlers` and the request queue are owned.
 */
typedef struct dma {
    mds*                   mds;          // NOT owned - the market data source.
    dma_algo_engine*       algo_engine;  // NOT owned - the algo engine bound to this access.
    dma_risk*              risk;         // NOT owned - the risk profile.
    dma_position*          position;     // NOT owned - the position bound to this access.
    dma_inventory          inventory;    // OWNED - embedded inventory, initialized by c_dma_new.
    dma_order_handler_ctx* handlers;     // OWNED - calloc'ed registration nodes.
    double                 cool_down;    // Order launch cool-down in seconds; 0 disables buffering.
    bool                   enabled;      // false rejects every launch.
    dma_order_request*     head;         // OWNED - FIFO head.
    dma_order_request*     tail;         // OWNED - FIFO tail.
    size_t                 n_requests;
} dma;

#include <algo_engine/engine/algo/c_algo_template.h>
#include <algo_engine/engine/c_dma_algo.h>

// ========== Forward Declarations ==========

// Lifecycle
static inline dma*          c_dma_new(mds* market_data, dma_risk* risk, double cool_down, allocator_protocol* allocator);
static inline int           c_dma_init(dma* self, mds* market_data, dma_risk* risk, double cool_down);
static inline void          c_dma_dealloc(dma* self);
static inline void          c_dma_free(dma* self);
static inline int           c_dma_clear(dma* self);

// Position
static inline void          c_dma_set_position(dma* self, dma_position* position);
static inline dma_position* c_dma_get_position(const dma* self);

// Algo engine
static inline dma_algo*     c_dma_new_algo(dma* DMA, uint32_t algo_reg_id, const char* ticker, md_side side, double target_volume, const void* params, allocator_protocol* allocator);
static inline double        c_dma_calculate_limit(dma* DMA, const char* ticker, md_side side, double limit_price, double limit_adjust_factor, dma_algo_limit_mode mode);

// Order handlers
static inline int           c_dma_register_order_handler(dma* self, dma_order_launch_func order_launch_fn, dma_order_cancel_func order_cancel_fn, dma_order_reject_func order_reject_fn, dma_order_filled_callback order_filled_fn, void* user_data, uint32_t seq_id, uintptr_t* out_id);
static inline int           c_dma_unregister_order_handler(dma* self, uintptr_t handler_id);

// Order dispatch
static inline void          c_dma_invoke_launch_order(dma* self, md_trade_instruction* order);
static inline void          c_dma_invoke_cancel_order(dma* self, const md_trade_instruction* order);
static inline void          c_dma_invoke_reject_order(dma* self, const md_trade_instruction* order);
static inline void          c_dma_invoke_filled_order(dma* self, const md_trade_instruction* order, md_trade_report* report);

// Gating
static inline void          c_dma_set_enabled(dma* self, bool enabled);
static inline bool          c_dma_is_enabled(const dma* self);
static inline int           c_dma_check_order(const dma* self, const dma_risk_query* query, dma_risk_violation* out_violation);

// Request queue
static inline int           c_dma_request_order(dma* self, dma_order_action action, md_trade_instruction* order);
static inline int           c_dma_pop_request(dma* self, dma_order_action* out_action, md_trade_instruction** out_order);
static inline void          c_dma_clear_requests(dma* self);

// ========== Public APIs (Direct Market Access) ==========

/**
 * @brief Allocate and initialize a direct market access.
 *
 * The embedded inventory is a child of the access block and is initialized
 * here.
 *
 * @param allocator Allocator for the block and its children.
 * @return The new access, or NULL on allocation failure.
 */
static inline dma*          c_dma_new(mds* market_data, dma_risk* risk, double cool_down, allocator_protocol* allocator) {
    dma* self = (dma*) c_ap_alloc(sizeof(dma), allocator);
    if (!self) return NULL;

    c_dma_init(self, market_data, risk, cool_down);

    // The inventory is embedded, so its block start is the access: the access
    // protocol is derived when the caller did not supply an allocator.
    allocator_protocol* protocol = allocator ? allocator : c_ap_protocol_from_ptr(self);

    if (c_dma_inventory_init(&self->inventory, protocol) != INVENTORY_OK) {
        c_ap_free(self);
        return NULL;
    }

    return self;
}

/**
 * @brief Initialize a caller-allocated, zeroed direct market access.
 *
 * The embedded inventory is left zeroed; initialize it with
 * `c_dma_inventory_init(&self->inventory, allocator)`.
 *
 * @param self The zeroed access to initialize.
 * @return DMA_OK, or DMA_ERR_INVALID_INPUT.
 */
static inline int c_dma_init(dma* self, mds* market_data, dma_risk* risk, double cool_down) {
    if (!self) return DMA_ERR_INVALID_INPUT;

    self->mds       = market_data;
    self->risk      = risk;
    self->cool_down = cool_down > 0 ? cool_down : 0.;
    self->enabled   = false;
    return DMA_OK;
}

/**
 * @brief Tear down a caller-allocated access, releasing the embedded
 *        inventory, the buffered requests and the registered interface nodes.
 * @param self The access (NULL-safe).
 */
static inline void c_dma_dealloc(dma* self) {
    if (!self) return;

    c_dma_clear_requests(self);
    c_dma_inventory_dealloc(&self->inventory);

    dma_order_handler_ctx* handler = self->handlers;
    while (handler) {
        dma_order_handler_ctx* next = handler->next;
        free(handler);
        handler = next;
    }

    memset(self, 0, sizeof(dma));
}

/**
 * @brief Free an access allocated by `c_dma_new`.
 * @param self The access (NULL-safe).
 */
static inline void c_dma_free(dma* self) {
    if (!self) return;

    c_dma_dealloc(self);
    c_ap_free(self);
}

/**
 * @brief Drop every buffered request and reset the access state.
 * @param self The access (NULL-safe).
 * @return DMA_OK, or DMA_ERR_INVALID_INPUT.
 */
static inline int c_dma_clear(dma* self) {
    if (!self) return DMA_ERR_INVALID_INPUT;

    c_dma_clear_requests(self);
    self->enabled = false;
    return DMA_OK;
}

/**
 * @brief Bind the position this access trades through.
 * @param self The access (NULL-safe).
 * @param position The borrowed position (may be NULL to clear).
 */
static inline void c_dma_set_position(dma* self, dma_position* position) {
    if (!self) return;

    self->position = position;
}

/**
 * @brief Read the position bound to this access.
 * @param self The access (NULL-safe).
 * @return The borrowed position, or NULL.
 */
static inline dma_position* c_dma_get_position(const dma* self) {
    return self ? self->position : NULL;
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
    if (!DMA || !DMA->algo_engine || !ticker) return NULL;

    dma_algo_ctx* registration = c_dma_algo_registry_get(DMA->algo_engine->registry, algo_reg_id);

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
    if (!DMA || !DMA->mds || !ticker) return NAN;

    double market_price = c_mds_get_market_price(DMA->mds, ticker);
    if (isnan(market_price)) return NAN;

    double calculated = NAN;

    calculated = c_dma_algo_compare_limit(side, limit_price, calculated, mode);

    if (!isnan(limit_adjust_factor)) {
        double adjusted = market_price * (1. + limit_adjust_factor * c_md_side_sign(side));
        calculated      = c_dma_algo_compare_limit(side, adjusted, calculated, mode);
    }

    calculated = c_dma_algo_compare_limit(side, market_price, calculated, mode);
    return calculated;
}

/**
 * @brief Register one order handler set.
 * @param self The access (NULL-safe).
 * @param order_launch_fn The launch handler (NULL is refused).
 * @param order_cancel_fn The cancel handler (NULL is refused).
 * @param order_reject_fn The reject handler (NULL is refused).
 * @param order_filled_fn The filled handler (NULL is refused).
 * @param user_data Opaque value handed to every handler.
 * @param seq_id The id handed to the launch handler for self-suppression.
 * @param out_id Optional: receives the id used to unregister.
 * @return DMA_OK, DMA_ERR_INVALID_INPUT, or DMA_ERR_OOM.
 */
static inline int c_dma_register_order_handler(dma* self, dma_order_launch_func order_launch_fn, dma_order_cancel_func order_cancel_fn, dma_order_reject_func order_reject_fn, dma_order_filled_callback order_filled_fn, void* user_data, uint32_t seq_id, uintptr_t* out_id) {
    if (out_id) *out_id = 0;
    if (!self || !order_launch_fn || !order_cancel_fn || !order_reject_fn || !order_filled_fn) return DMA_ERR_INVALID_INPUT;

    dma_order_handler_ctx* node = (dma_order_handler_ctx*) calloc(1, sizeof(dma_order_handler_ctx));
    if (!node) return DMA_ERR_OOM;

    node->order_launch_fn = order_launch_fn;
    node->order_cancel_fn = order_cancel_fn;
    node->order_reject_fn = order_reject_fn;
    node->order_filled_fn = order_filled_fn;
    node->user_data       = user_data;
    node->seq_id          = seq_id;
    node->reg_id          = (uintptr_t) node;

    if (!self->handlers) {
        self->handlers = node;
    }
    else {
        dma_order_handler_ctx* tail = self->handlers;
        while (tail->next) tail = tail->next;
        tail->next = node;
    }

    if (out_id) *out_id = node->reg_id;
    return DMA_OK;
}

/**
 * @brief Unregister an order handler set by the id registration returned.
 * @param self The access (NULL-safe).
 * @param handler_id The id from registration (0 is refused).
 * @return DMA_OK, DMA_ERR_INVALID_INPUT, or DMA_ERR_NOT_FOUND.
 */
static inline int c_dma_unregister_order_handler(dma* self, uintptr_t handler_id) {
    if (!self || !handler_id) return DMA_ERR_INVALID_INPUT;

    dma_order_handler_ctx* previous = NULL;
    dma_order_handler_ctx* current  = self->handlers;

    while (current) {
        if (current->reg_id == handler_id) {
            if (previous) previous->next = current->next;
            else self->handlers = current->next;
            free(current);
            return DMA_OK;
        }

        previous = current;
        current  = current->next;
    }

    return DMA_ERR_NOT_FOUND;
}

/**
 * @brief Hand a launch to every registered order handler.
 *
 * The walk snapshots `next` before each call, so a handler may unregister
 * itself without corrupting the traversal. Each handler receives the seq id it
 * registered with.
 *
 * @param self The access (NULL-safe).
 * @param order The launched order (NULL-safe).
 */
static inline void c_dma_invoke_launch_order(dma* self, md_trade_instruction* order) {
    if (!self || !order) return;

    dma_order_handler_ctx* handler = self->handlers;
    while (handler) {
        dma_order_handler_ctx* next = handler->next;
        if (handler->order_launch_fn) handler->order_launch_fn(order, handler->seq_id, handler->user_data);
        handler = next;
    }
}

/**
 * @brief Hand a cancel to every registered order handler.
 * @param self The access (NULL-safe).
 * @param order The cancelled order (NULL-safe).
 */
static inline void c_dma_invoke_cancel_order(dma* self, const md_trade_instruction* order) {
    if (!self || !order) return;

    dma_order_handler_ctx* handler = self->handlers;
    while (handler) {
        dma_order_handler_ctx* next = handler->next;
        if (handler->order_cancel_fn) handler->order_cancel_fn(order, handler->user_data);
        handler = next;
    }
}

/**
 * @brief Hand a reject to every registered order handler.
 * @param self The access (NULL-safe).
 * @param order The rejected order (NULL-safe).
 */
static inline void c_dma_invoke_reject_order(dma* self, const md_trade_instruction* order) {
    if (!self || !order) return;

    dma_order_handler_ctx* handler = self->handlers;
    while (handler) {
        dma_order_handler_ctx* next = handler->next;
        if (handler->order_reject_fn) handler->order_reject_fn(order, handler->user_data);
        handler = next;
    }
}

/**
 * @brief Hand a fill to every registered order handler.
 * @param self The access (NULL-safe).
 * @param order The filled order (NULL-safe).
 * @param report The fill report (NULL-safe).
 */
static inline void c_dma_invoke_filled_order(dma* self, const md_trade_instruction* order, md_trade_report* report) {
    if (!self || !order || !report) return;

    dma_order_handler_ctx* handler = self->handlers;
    while (handler) {
        dma_order_handler_ctx* next = handler->next;
        if (handler->order_filled_fn) handler->order_filled_fn(order, report, handler->user_data);
        handler = next;
    }
}

/**
 * @brief Set the access's enabled flag.
 * @param self The access (NULL-safe).
 * @param enabled The new flag.
 */
static inline void c_dma_set_enabled(dma* self, bool enabled) {
    if (!self) return;

    self->enabled = enabled;
}

/**
 * @brief Read the access's enabled flag.
 * @param self The access (NULL-safe).
 * @return The flag, or false.
 */
static inline bool c_dma_is_enabled(const dma* self) {
    return self ? self->enabled : false;
}

/**
 * @brief Decide whether an order may leave the access.
 *
 * A disabled access, a missing risk profile and any risk violation all refuse
 * the order; there is no path that sends an order without a passing risk
 * check. The violation carries the reason for the refusal.
 *
 * @param self The access (NULL-safe).
 * @param query The order and portfolio state (NULL-safe).
 * @param out_violation Optional: receives the risk violation on refusal.
 * @return DMA_OK when the order may be sent, DMA_ERR_DISABLED,
 *         DMA_ERR_NO_RISK, DMA_ERR_RISK, or DMA_ERR_INVALID_INPUT.
 */
static inline int c_dma_check_order(const dma* self, const dma_risk_query* query, dma_risk_violation* out_violation) {
    if (out_violation) {
        out_violation->type   = DMA_RISK_PASS;
        out_violation->code   = 0;
        out_violation->msg[0] = '\0';
    }

    if (!self || !query) return DMA_ERR_INVALID_INPUT;
    if (!self->enabled) return DMA_ERR_DISABLED;
    if (!self->risk) return DMA_ERR_NO_RISK;

    if (c_dma_risk_check(self->risk, query, out_violation) != DMA_RISK_PASS) return DMA_ERR_RISK;

    return DMA_OK;
}

/**
 * @brief Append one buffered request to the access queue.
 *
 * The node is a child of the access block, so a failed access teardown frees
 * any request still queued. The order is borrowed.
 *
 * @param self The access (NULL-safe).
 * @param action The action to buffer.
 * @param order The order to route (NULL-safe).
 * @return DMA_OK, DMA_ERR_INVALID_INPUT, or DMA_ERR_OOM.
 */
static inline int c_dma_request_order(dma* self, dma_order_action action, md_trade_instruction* order) {
    if (!self || !order) return DMA_ERR_INVALID_INPUT;

    allocator_protocol* allocator = c_ap_protocol_from_ptr(self);

    dma_order_request*  node = (dma_order_request*) c_ap_alloc_child(sizeof(dma_order_request), allocator, self);
    if (!node) return DMA_ERR_OOM;

    node->action = action;
    node->order  = order;
    node->next   = NULL;

    if (self->tail) self->tail->next = node;
    else self->head = node;
    self->tail = node;
    self->n_requests++;

    return DMA_OK;
}

/**
 * @brief Pop the oldest buffered request.
 * @param self The access (NULL-safe).
 * @param out_action Optional: receives the action.
 * @param out_order Optional: receives the order (still borrowed).
 * @return DMA_OK, DMA_ERR_INVALID_INPUT, or DMA_ERR_EMPTY.
 */
static inline int c_dma_pop_request(dma* self, dma_order_action* out_action, md_trade_instruction** out_order) {
    if (!self) return DMA_ERR_INVALID_INPUT;
    if (!self->head) return DMA_ERR_EMPTY;

    dma_order_request* node = self->head;
    self->head              = node->next;
    if (!self->head) self->tail = NULL;
    self->n_requests--;

    if (out_action) *out_action = node->action;
    if (out_order) *out_order = node->order;

    c_ap_free(node);
    return DMA_OK;
}

/**
 * @brief Drop every buffered request.
 * @param self The access (NULL-safe).
 */
static inline void c_dma_clear_requests(dma* self) {
    if (!self) return;

    dma_order_request* node = self->head;
    while (node) {
        dma_order_request* next = node->next;
        c_ap_free(node);
        node = next;
    }

    self->head       = NULL;
    self->tail       = NULL;
    self->n_requests = 0;
}

#endif /* C_DMA_H */

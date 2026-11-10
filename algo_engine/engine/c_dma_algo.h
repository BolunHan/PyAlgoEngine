#ifndef C_DMA_ALGO_H
#define C_DMA_ALGO_H

#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include <algo_engine/base/c_market_data/c_market_data.h>
#include <algo_engine/engine/c_market_engine.h>
#include <cbase/allocator_protocol/c_allocator_protocol.h>
#include <cbase/bytemap/c_bytemap.h>

typedef struct dma      dma;
typedef struct dma_algo dma_algo;

// ========== Enums ==========

/** @enum dma_algo_ret_code
 *  @brief 0 is success, negatives are errors. */
typedef enum dma_algo_ret_code {
    DMA_ALGO_OK                = 0,
    DMA_ALGO_ERR_INVALID_INPUT = -1,
    DMA_ALGO_ERR_OOM           = -2,
    DMA_ALGO_ERR_NOT_FOUND     = -3,
} dma_algo_ret_code;

/** @enum dma_algo_limit_mode
 *  @brief How two limit prices are combined. */
typedef enum dma_algo_limit_mode {
    DMA_ALGO_LIMIT_STRICT = 0,
    DMA_ALGO_LIMIT_LOOSE  = 1,
} dma_algo_limit_mode;

// ========== Structs ==========

typedef struct dma_algo_ctx      dma_algo_ctx;
typedef struct dma_algo_engine   dma_algo_engine;
typedef struct dma_algo_registry dma_algo_registry;
typedef struct dma_order_ctx     dma_order_ctx;

/**
 * @brief Build one algo of the registered kind.
 * @param engine The engine (borrowed).
 * @param DMA The process-global access (borrowed).
 * @param ticker The instrument.
 * @param side The full direction and offset specification.
 * @param target_volume The volume to trade.
 * @param params Algo-kind parameters (borrowed).
 * @param registration The registration that builds the algo (borrowed).
 * @param allocator Allocator for the record.
 * @return The new record, or NULL.
 */
typedef dma_algo* (*dma_algo_new)(dma* DMA, const char* ticker, md_side side, double target_volume, const void* params, allocator_protocol* allocator);

/**
 * @struct dma_algo_ctx
 * @brief One registration of an algo kind.
 *
 * The context owns the maps of every algo built under this registration and
 * every order those algos created. The registry owns the context block; the
 * name entry points at the current registration and the id map keeps every
 * context, so replacing a name preserves the older contexts and their maps.
 */
typedef struct dma_algo_ctx {
    dma_algo_new algo_new_fn;  // NOT owned - the factory.
    bytemap*     orders;       // OWNED - order id -> borrowed dma_order_ctx*.
    bytemap*     algos;        // OWNED - algo id -> borrowed dma_algo*.
    uint32_t     reg_id;
} dma_algo_ctx;

/**
 * @struct dma_order_ctx
 * @brief One engine-owned order.
 *
 * `order` is the instruction the algo launched; `next` links the orders of one
 * ticker so market data can be routed without a per-order probe.
 */
typedef struct dma_order_ctx {
    md_trade_instruction  order;
    bytemap*              reports;
    dma_algo*             algo;  // NOT owned - the algo that launched the order.
    struct dma_order_ctx* next;  // Ticker-list link.
} dma_order_ctx;

/**
 * @struct dma_algo_registry
 * @brief Named algo registrations; re-registering a name overrides it and raises the id.
 */
typedef struct dma_algo_registry {
    bytemap* entries;  // OWNED - name -> borrowed dma_algo_ctx*.
    bytemap* id_map;   // OWNED - reg_id -> owned full dma_algo_ctx*.
} dma_algo_registry;

/**
 * @struct dma_algo_engine
 * @brief The process-global algo engine state.
 */
typedef struct dma_algo_engine {
    dma_algo_registry* registry;                  // OWNED - named factories.
    bytemap*           orders;                    // OWNED - order id -> owned dma_order_ctx*.
    bytemap*           working_orders_by_id;      // OWNED - live order id -> owned dma_order_ctx* (alias of orders).
    bytemap*           working_orders_by_ticker;  // OWNED - ticker -> dma_order_ctx* list head.
    double             timestamp;                 // Latest observation timestamp.
} dma_algo_engine;

// ========== Forward Declarations ==========

// Order context
static inline int                c_dma_order_ctx_init(dma_order_ctx* ctx, allocator_protocol* allocator);
static inline int                c_dma_order_ctx_fill(dma_order_ctx* ctx, const md_trade_report* report);
static inline void               c_dma_order_ctx_dealloc(dma_order_ctx* ctx);

// Registration context
static inline int                c_dma_algo_ctx_init(dma_algo_ctx* ctx, dma_algo_new algo_new_fn, uint32_t reg_id, allocator_protocol* allocator);
static inline void               c_dma_algo_ctx_dealloc(dma_algo_ctx* ctx);

// Algo registry
static inline dma_algo_registry* c_dma_algo_registry_new(allocator_protocol* allocator);
static inline int                c_dma_algo_registry_init(dma_algo_registry* registry, allocator_protocol* allocator);
static inline void               c_dma_algo_registry_dealloc(dma_algo_registry* registry);
static inline void               c_dma_algo_registry_free(dma_algo_registry* registry);
static inline dma_algo_ctx*      c_dma_algo_registry_reg(dma_algo_registry* registry, const char* name, dma_algo_new algo_new_fn, uint32_t reg_id);
static inline dma_algo_ctx*      c_dma_algo_registry_get(const dma_algo_registry* registry, uint32_t reg_id);

// Algo engine
static inline dma_algo_engine*   c_dma_algo_engine_new(allocator_protocol* allocator);
static inline int                c_dma_algo_engine_init(dma_algo_engine* engine, allocator_protocol* allocator);
static inline void               c_dma_algo_engine_dealloc(dma_algo_engine* engine);
static inline void               c_dma_algo_engine_free(dma_algo_engine* engine);

// Limit helpers
static inline double             c_dma_algo_compare_limit(md_side side, double limit_price, double original_limit, dma_algo_limit_mode mode);

// ========== Public APIs (Order Context) ==========

/**
 * @brief Initialize a zeroed order context's report index.
 * @param ctx The zeroed order context.
 * @param allocator Allocator for the report index.
 * @return DMA_ALGO_OK, DMA_ALGO_ERR_INVALID_INPUT, or DMA_ALGO_ERR_OOM.
 */
static inline int c_dma_order_ctx_init(dma_order_ctx* ctx, allocator_protocol* allocator) {
    if (!ctx) return DMA_ALGO_ERR_INVALID_INPUT;

    ctx->reports = c_bytemap_ex_new(8, sizeof(md_trade_report*), allocator);
    if (!ctx->reports) return DMA_ALGO_ERR_OOM;

    return DMA_ALGO_OK;
}

/**
 * @brief Apply a fill or cancel report to the order and store it.
 *
 * A report arriving while the order is canceling is the cancel confirmation;
 * its volume, when non-zero, is the cancelled amount and is not added to the
 * filled volume. Otherwise the report is a fill.
 *
 * @param ctx The order context (NULL-safe).
 * @param report The report (NULL-safe).
 * @return DMA_ALGO_OK, DMA_ALGO_ERR_INVALID_INPUT, or DMA_ALGO_ERR_OOM.
 */
static inline int c_dma_order_ctx_fill(dma_order_ctx* ctx, const md_trade_report* report) {
    if (!ctx || !report) return DMA_ALGO_ERR_INVALID_INPUT;
    if (!c_md_long_id_equal(&ctx->order.order_id, &report->order_id)) return DMA_ALGO_ERR_INVALID_INPUT;

    size_t n_before = c_bytemap_ex_len(ctx->reports);

    int    ret_code = c_bytemap_ex_set_default(
        ctx->reports,
        (const char*) &report->trade_id,
        sizeof(long_md_id),
        (const char*) &report,
        sizeof(md_trade_report*),
        0,
        NULL,
        NULL
    );

    if (ret_code != BYTEMAP_OK) return DMA_ALGO_ERR_OOM;
    if (c_bytemap_ex_len(ctx->reports) == n_before) return DMA_ALGO_OK;

    md_trade_instruction* order     = &ctx->order;
    double                timestamp = report->meta_info.timestamp;
    double                volume    = report->volume;
    bool                  canceled  = order->order_state == STATE_CANCELING || order->order_state == STATE_CANCELED;
    bool                  valid     = true;

    if (!canceled && volume > 0.) {
        if (order->filled_volume + volume > order->volume) valid = false;
        else if (isnan(order->multiplier)) order->multiplier = report->multiplier;
        else if (order->multiplier != report->multiplier) valid = false;
    }

    if (!valid) {
        c_bytemap_ex_pop(ctx->reports, (const char*) &report->trade_id, sizeof(long_md_id), 0, NULL, NULL);
        return DMA_ALGO_ERR_INVALID_INPUT;
    }

    if (canceled) {
        order->order_state = STATE_CANCELED;
        order->ts_canceled = timestamp;
        return DMA_ALGO_OK;
    }

    if (volume <= 0.) return DMA_ALGO_OK;

    order->filled_volume += volume;
    order->filled_notional += fabs(report->notional);
    order->fee += report->fee;

    if (order->filled_volume >= order->volume) {
        order->order_state = STATE_FILLED;
        order->ts_finished = timestamp;
    }
    else {
        order->order_state = STATE_PARTFILLED;
    }

    return DMA_ALGO_OK;
}

/**
 * @brief Release the report index, leaving the context zeroed.
 * @param ctx The order context (NULL-safe).
 */
static inline void c_dma_order_ctx_dealloc(dma_order_ctx* ctx) {
    if (!ctx) return;

    if (ctx->reports) {
        c_bytemap_ex_free(ctx->reports);
        ctx->reports = NULL;
    }

    memset(ctx, 0, sizeof(dma_order_ctx));
}

// ========== Public APIs (Algo Registry) ==========

/**
 * @brief Initialize a registration context and its maps.
 * @param ctx The context to initialize.
 * @param algo_new_fn The factory (NULL is refused).
 * @param reg_id The registration id.
 * @param allocator Allocator for the maps.
 * @return DMA_ALGO_OK, DMA_ALGO_ERR_INVALID_INPUT, or DMA_ALGO_ERR_OOM.
 */
static inline int c_dma_algo_ctx_init(dma_algo_ctx* ctx, dma_algo_new algo_new_fn, uint32_t reg_id, allocator_protocol* allocator) {
    if (!ctx || !algo_new_fn) return DMA_ALGO_ERR_INVALID_INPUT;

    ctx->algo_new_fn = algo_new_fn;
    ctx->reg_id      = reg_id;

    ctx->orders = c_bytemap_ex_new(0, sizeof(dma_order_ctx*), allocator);
    if (!ctx->orders) return DMA_ALGO_ERR_OOM;

    ctx->algos = c_bytemap_ex_new(0, sizeof(dma_algo*), allocator);
    if (!ctx->algos) {
        c_bytemap_ex_free(ctx->orders);
        ctx->orders = NULL;
        return DMA_ALGO_ERR_OOM;
    }

    return DMA_ALGO_OK;
}

/**
 * @brief Release a registration context and its maps, leaving it zeroed.
 * @param ctx The context (NULL-safe).
 */
static inline void c_dma_algo_ctx_dealloc(dma_algo_ctx* ctx) {
    if (!ctx) return;

    if (ctx->orders) {
        c_bytemap_ex_free(ctx->orders);
        ctx->orders = NULL;
    }

    if (ctx->algos) {
        c_bytemap_ex_free(ctx->algos);
        ctx->algos = NULL;
    }

    memset(ctx, 0, sizeof(dma_algo_ctx));
}

/**
 * @brief Allocate a registry and its maps.
 * @param allocator Allocator for the block and its maps.
 * @return The new registry, or NULL.
 */
static inline dma_algo_registry* c_dma_algo_registry_new(allocator_protocol* allocator) {
    dma_algo_registry* registry = (dma_algo_registry*) c_ap_alloc(sizeof(dma_algo_registry), allocator);
    if (!registry) return NULL;

    if (c_dma_algo_registry_init(registry, allocator) != DMA_ALGO_OK) {
        c_ap_free(registry);
        return NULL;
    }

    return registry;
}

/**
 * @brief Initialize a caller-allocated, zeroed registry.
 * @param registry The zeroed registry.
 * @param allocator Allocator for the maps; derived from registry when NULL.
 * @return DMA_ALGO_OK, DMA_ALGO_ERR_INVALID_INPUT, or DMA_ALGO_ERR_OOM.
 */
static inline int c_dma_algo_registry_init(dma_algo_registry* registry, allocator_protocol* allocator) {
    if (!registry) return DMA_ALGO_ERR_INVALID_INPUT;
    if (!allocator) allocator = c_ap_protocol_from_ptr(registry);

    registry->entries = c_bytemap_ex_new(0, sizeof(dma_algo_ctx*), allocator);
    if (!registry->entries) return DMA_ALGO_ERR_OOM;

    registry->id_map = c_bytemap_ex_new(0, sizeof(dma_algo_ctx*), allocator);
    if (!registry->id_map) {
        c_bytemap_ex_free(registry->entries);
        registry->entries = NULL;
        return DMA_ALGO_ERR_OOM;
    }

    return DMA_ALGO_OK;
}

/**
 * @brief Tear down a caller-allocated registry, releasing every context.
 *
 * The id map owns every context block, current or replaced; the name entries
 * are borrowed aliases and never free them.
 *
 * @param registry The registry (NULL-safe).
 */
static inline void c_dma_algo_registry_dealloc(dma_algo_registry* registry) {
    if (!registry) return;

    if (registry->id_map) {
        bytemap_entry* entry = registry->id_map->first;

        while (entry) {
            bytemap_entry* next = entry->next;
            dma_algo_ctx*  ctx  = *(dma_algo_ctx**) entry->value;

            if (ctx) {
                c_dma_algo_ctx_dealloc(ctx);
                c_ap_free(ctx);
            }

            entry = next;
        }

        c_bytemap_ex_free(registry->id_map);
        registry->id_map = NULL;
    }

    if (registry->entries) {
        c_bytemap_ex_free(registry->entries);
        registry->entries = NULL;
    }

    memset(registry, 0, sizeof(dma_algo_registry));
}

/**
 * @brief Free a registry allocated by `c_dma_algo_registry_new`.
 * @param registry The registry (NULL-safe).
 */
static inline void c_dma_algo_registry_free(dma_algo_registry* registry) {
    if (!registry) return;

    c_dma_algo_registry_dealloc(registry);
    c_ap_free(registry);
}

/**
 * @brief Register or override a named factory.
 *
 * A zero id takes the first free id from `id_map size + 1` upward; a caller
 * supplied id is refused when it is already registered. Re-registering a name
 * repoints the name at the new context; the replaced context stays owned by the
 * id map.
 *
 * @param registry The registry (NULL-safe).
 * @param name The algo name (non-empty).
 * @param algo_new_fn The factory (NULL is refused).
 * @param reg_id The requested registration id; 0 auto-generates one.
 * @return The new registration context, or NULL.
 */
static inline dma_algo_ctx* c_dma_algo_registry_reg(dma_algo_registry* registry, const char* name, dma_algo_new algo_new_fn, uint32_t reg_id) {
    if (!registry || !registry->entries || !registry->id_map || !name || !algo_new_fn) return NULL;

    allocator_protocol* allocator = c_ap_protocol_from_ptr(registry->id_map);

    char*               stored = NULL;

    if (reg_id == 0) {
        reg_id = (uint32_t) (c_bytemap_ex_len(registry->id_map) + 1);

        while (c_bytemap_ex_get_ptr(registry->id_map, (const char*) &reg_id, sizeof(uint32_t), &stored, NULL) == BYTEMAP_OK) {
            reg_id += 1;
        }
    }
    else if (c_bytemap_ex_get_ptr(registry->id_map, (const char*) &reg_id, sizeof(uint32_t), &stored, NULL) == BYTEMAP_OK) {
        return NULL;
    }

    dma_algo_ctx* ctx = (dma_algo_ctx*) c_ap_alloc(sizeof(dma_algo_ctx), allocator);
    if (!ctx) return NULL;

    if (c_dma_algo_ctx_init(ctx, algo_new_fn, reg_id, allocator) != DMA_ALGO_OK) {
        c_ap_free(ctx);
        return NULL;
    }

    // The id map owns the full context block.
    if (c_bytemap_ex_set(registry->id_map, (const char*) &reg_id, sizeof(uint32_t), (const char*) &ctx, sizeof(dma_algo_ctx*), 0, NULL) != BYTEMAP_OK) {
        c_dma_algo_ctx_dealloc(ctx);
        c_ap_free(ctx);
        return NULL;
    }

    // The name points at the current registration; a replaced context stays in
    // the id map.
    if (c_bytemap_ex_set(registry->entries, name, strlen(name), (const char*) &ctx, sizeof(dma_algo_ctx*), 0, NULL) != BYTEMAP_OK) {
        c_bytemap_ex_pop(registry->id_map, (const char*) &reg_id, sizeof(uint32_t), 0, NULL, NULL);
        c_dma_algo_ctx_dealloc(ctx);
        c_ap_free(ctx);
        return NULL;
    }

    return ctx;
}

/**
 * @brief Find a registration context by its id, current or replaced.
 * @param registry The registry (NULL-safe).
 * @param reg_id The registration id.
 * @return The borrowed context, or NULL.
 */
static inline dma_algo_ctx* c_dma_algo_registry_get(const dma_algo_registry* registry, uint32_t reg_id) {
    if (!registry || !registry->id_map) return NULL;

    char* stored = NULL;

    if (c_bytemap_ex_get_ptr(registry->id_map, (const char*) &reg_id, sizeof(uint32_t), &stored, NULL) != BYTEMAP_OK) return NULL;

    return *(dma_algo_ctx**) stored;
}

// ========== Public APIs (Algo Engine) ==========

/**
 * @brief Allocate an engine with its registry and order indexes.
 * @param allocator Allocator for the block and its children.
 * @return The new engine, or NULL.
 */
static inline dma_algo_engine* c_dma_algo_engine_new(allocator_protocol* allocator) {
    dma_algo_engine* engine = (dma_algo_engine*) c_ap_alloc(sizeof(dma_algo_engine), allocator);
    if (!engine) return NULL;

    if (c_dma_algo_engine_init(engine, allocator) != DMA_ALGO_OK) {
        c_dma_algo_engine_dealloc(engine);
        c_ap_free(engine);
        return NULL;
    }

    return engine;
}

/**
 * @brief Initialize a caller-allocated, zeroed engine.
 * @param engine The zeroed engine.
 * @param allocator Allocator for the children; derived from engine when NULL.
 * @return DMA_ALGO_OK, DMA_ALGO_ERR_INVALID_INPUT, or DMA_ALGO_ERR_OOM.
 */
static inline int c_dma_algo_engine_init(dma_algo_engine* engine, allocator_protocol* allocator) {
    if (!engine) return DMA_ALGO_ERR_INVALID_INPUT;
    if (!allocator) allocator = c_ap_protocol_from_ptr(engine);

    engine->registry = c_dma_algo_registry_new(allocator);
    if (!engine->registry) return DMA_ALGO_ERR_OOM;

    engine->orders = c_bytemap_ex_new(0, sizeof(dma_order_ctx*), allocator);
    if (!engine->orders) return DMA_ALGO_ERR_OOM;

    engine->working_orders_by_id = c_bytemap_ex_new(0, sizeof(dma_order_ctx*), allocator);
    if (!engine->working_orders_by_id) return DMA_ALGO_ERR_OOM;

    engine->working_orders_by_ticker = c_bytemap_ex_new(0, sizeof(dma_order_ctx*), allocator);
    if (!engine->working_orders_by_ticker) return DMA_ALGO_ERR_OOM;

    return DMA_ALGO_OK;
}

/**
 * @brief Tear down a caller-allocated engine, releasing every order context.
 * @param engine The engine (NULL-safe).
 */
static inline void c_dma_algo_engine_dealloc(dma_algo_engine* engine) {
    if (!engine) return;

    if (engine->orders) {
        bytemap_entry* entry = engine->orders->first;

        while (entry) {
            bytemap_entry* next = entry->next;
            dma_order_ctx* ctx  = *(dma_order_ctx**) entry->value;

            if (ctx) {
                c_dma_order_ctx_dealloc(ctx);
                c_ap_free(ctx);
            }

            entry = next;
        }

        c_bytemap_ex_free(engine->orders);
        engine->orders = NULL;
    }

    if (engine->working_orders_by_id) {
        c_bytemap_ex_free(engine->working_orders_by_id);
        engine->working_orders_by_id = NULL;
    }

    if (engine->working_orders_by_ticker) {
        c_bytemap_ex_free(engine->working_orders_by_ticker);
        engine->working_orders_by_ticker = NULL;
    }

    if (engine->registry) {
        c_dma_algo_registry_free(engine->registry);
        engine->registry = NULL;
    }

    memset(engine, 0, sizeof(dma_algo_engine));
}

/**
 * @brief Free an engine allocated by `c_dma_algo_engine_new`.
 * @param engine The engine (NULL-safe).
 */
static inline void c_dma_algo_engine_free(dma_algo_engine* engine) {
    if (!engine) return;

    c_dma_algo_engine_dealloc(engine);
    c_ap_free(engine);
}

/**
 * @brief Combine two limit prices by the side and mode.
 *
 * A NaN candidate keeps the original; a NaN original takes the candidate.
 *
 * @param side The trade side.
 * @param limit_price The candidate limit.
 * @param original_limit The accumulated limit (NaN when none).
 * @param mode Strict takes the tighter price; loose takes the wider one.
 * @return The combined limit, or NaN when neither input is valid.
 */
static inline double c_dma_algo_compare_limit(md_side side, double limit_price, double original_limit, dma_algo_limit_mode mode) {
    if (isnan(limit_price)) return original_limit;
    if (isnan(original_limit)) return limit_price;

    int sign = c_md_side_sign(side);

    if (mode == DMA_ALGO_LIMIT_LOOSE) {
        if (sign > 0) return original_limit > limit_price ? original_limit : limit_price;
        return original_limit < limit_price ? original_limit : limit_price;
    }

    if (sign > 0) return original_limit < limit_price ? original_limit : limit_price;
    return original_limit > limit_price ? original_limit : limit_price;
}

#endif /* C_DMA_ALGO_H */

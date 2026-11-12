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
    DMA_ALGO_OK                 = 0,
    DMA_ALGO_ERR_INVALID_INPUT  = -1,
    DMA_ALGO_ERR_OOM            = -2,
    DMA_ALGO_ERR_NOT_FOUND      = -3,
    DMA_ALGO_ERR_WORKING_ORDERS = -4,  // A live order refuses the working-list exit.
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
 * `order` is the instruction the algo launched and sits at offset 0, so the
 * instruction address is also the context address; `reports` owns a copy of
 * every applied report. `next` links the orders of one ticker so market data
 * can be routed without a per-order probe.
 */
typedef struct dma_order_ctx {
    md_trade_instruction  order;
    bytemap*              reports;  // OWNED - trade id -> owned md_trade_report copy.
    dma_algo*             algo;     // NOT owned - the algo that launched the order.
    struct dma_order_ctx* next;     // Ticker-list link.
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
    bytemap*           algo;                      // OWNED - algo id -> owned dma_algo* (the engine owns the records).
    bytemap*           working_algo;              // OWNED - ticker -> dma_algo* working-list head (borrowed; joined at creation, left on completion or error).
    dma_algo*          error_algo;                // Algo in error state which needs manual intervene
    double             timestamp;                 // Latest observation timestamp.
} dma_algo_engine;

// ========== Forward Declarations ==========

// Order context
static inline int                   c_dma_order_ctx_init(dma_order_ctx* ctx, allocator_protocol* allocator);
static inline int                   c_dma_order_ctx_fill(dma_order_ctx* ctx, const md_trade_report* report);
static inline void                  c_dma_order_ctx_dealloc(dma_order_ctx* ctx);

// Registration context
static inline int                   c_dma_algo_ctx_init(dma_algo_ctx* ctx, dma_algo_new algo_new_fn, uint32_t reg_id, allocator_protocol* allocator);
static inline void                  c_dma_algo_ctx_dealloc(dma_algo_ctx* ctx);

// Algo registry
static inline dma_algo_registry*    c_dma_algo_registry_new(allocator_protocol* allocator);
static inline int                   c_dma_algo_registry_init(dma_algo_registry* registry, allocator_protocol* allocator);
static inline void                  c_dma_algo_registry_dealloc(dma_algo_registry* registry);
static inline void                  c_dma_algo_registry_free(dma_algo_registry* registry);
static inline dma_algo_ctx*         c_dma_algo_registry_reg(dma_algo_registry* registry, const char* name, dma_algo_new algo_new_fn, uint32_t reg_id);
static inline dma_algo_ctx*         c_dma_algo_registry_get(const dma_algo_registry* registry, uint32_t reg_id);

// Algo engine
static inline dma_algo_engine*      c_dma_algo_engine_new(allocator_protocol* allocator);
static inline int                   c_dma_algo_engine_init(dma_algo_engine* engine, allocator_protocol* allocator);
static inline void                  c_dma_algo_engine_dealloc(dma_algo_engine* engine);
static inline void                  c_dma_algo_engine_free(dma_algo_engine* engine);

// Algo record (defined in algo/c_algo_template.h)
static inline void                  c_dma_algo_free(dma_algo* algo);

// Engine entries
static inline void                  c_dma_algo_engine_pop_working_order(dma_algo_engine* engine, const long_md_id* order_id);
static inline md_trade_instruction* c_dma_algo_engine_new_order(dma_algo_engine* engine, dma_algo* algo, double limit_price, double volume, double timestamp);
static inline void                  c_dma_algo_engine_free_orders(dma_algo_engine* engine, dma_algo* algo);
static inline int                   c_dma_algo_engine_on_order(dma_algo_engine* engine, const md_trade_instruction* order);
static inline int                   c_dma_algo_engine_on_trade(dma_algo_engine* engine, const md_trade_report* trade);
static inline void                  c_dma_algo_engine_on_md(dma_algo_engine* engine, const md_variant* market_data);

// Limit helpers
static inline double                c_dma_algo_compare_limit(md_side side, double limit_price, double original_limit, dma_algo_limit_mode mode);

#include <algo_engine/engine/algo/c_algo_template.h>

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

    allocator_protocol* allocator = c_ap_protocol_from_ptr(ctx->reports);
    md_trade_report*    owned     = (md_trade_report*) c_ap_alloc(sizeof(md_trade_report), allocator);

    if (!owned) return DMA_ALGO_ERR_OOM;

    memcpy(owned, report, sizeof(md_trade_report));

    size_t n_before = c_bytemap_ex_len(ctx->reports);

    int    ret_code = c_bytemap_ex_set_default(
        ctx->reports,
        (const char*) &report->trade_id,
        sizeof(long_md_id),
        (const char*) &owned,
        sizeof(md_trade_report*),
        0,
        NULL,
        NULL
    );

    if (ret_code != BYTEMAP_OK) {
        c_ap_free(owned);
        return DMA_ALGO_ERR_OOM;
    }

    if (c_bytemap_ex_len(ctx->reports) == n_before) {
        c_ap_free(owned);
        return DMA_ALGO_OK;
    }

    md_trade_instruction* order     = &ctx->order;
    double                timestamp = owned->meta_info.timestamp;
    double                volume    = owned->volume;
    bool                  canceled  = order->order_state == STATE_CANCELING || order->order_state == STATE_CANCELED;
    bool                  valid     = true;

    if (!canceled && volume > 0.) {
        if (order->filled_volume + volume > order->volume) valid = false;
        else if (isnan(order->multiplier)) order->multiplier = owned->multiplier;
        else if (order->multiplier != owned->multiplier) valid = false;
    }

    if (!valid) {
        c_bytemap_ex_pop(ctx->reports, (const char*) &report->trade_id, sizeof(long_md_id), 0, NULL, NULL);
        c_ap_free(owned);
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
        bytemap_entry* entry = ctx->reports->first;

        while (entry) {
            bytemap_entry*   next   = entry->next;
            md_trade_report* report = *(md_trade_report**) entry->value;

            if (report) c_ap_free(report);

            entry = next;
        }

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

    engine->algo = c_bytemap_ex_new(0, sizeof(dma_algo*), allocator);
    if (!engine->algo) return DMA_ALGO_ERR_OOM;

    engine->working_algo = c_bytemap_ex_new(0, sizeof(dma_algo*), allocator);
    if (!engine->working_algo) return DMA_ALGO_ERR_OOM;

    return DMA_ALGO_OK;
}

/**
 * @brief Tear down a caller-allocated engine, releasing its order contexts, algo records and storage.
 *
 * The engine index owns the algo records; every record is freed before the
 * indexes that reference it.
 *
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

    if (engine->algo) {
        while (c_bytemap_ex_len(engine->algo)) {
            bytemap_entry* entry = engine->algo->first;
            dma_algo*      algo  = *(dma_algo**) entry->value;
            uuid_t         key;

            memcpy(key, entry->key, sizeof(uuid_t));
            c_bytemap_ex_pop(engine->algo, (const char*) key, sizeof(uuid_t), 0, NULL, NULL);

            if (algo) {
                algo->DMA = NULL;  // The access may already be gone at teardown.
                c_dma_algo_free(algo);
            }
        }

        c_bytemap_ex_free(engine->algo);
        engine->algo = NULL;
    }

    if (engine->working_orders_by_id) {
        c_bytemap_ex_free(engine->working_orders_by_id);
        engine->working_orders_by_id = NULL;
    }

    if (engine->working_orders_by_ticker) {
        c_bytemap_ex_free(engine->working_orders_by_ticker);
        engine->working_orders_by_ticker = NULL;
    }

    if (engine->working_algo) {
        c_bytemap_ex_free(engine->working_algo);
        engine->working_algo = NULL;
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
 * @brief Pop a finished order from the engine's working indexes.
 *
 * The history entries (the engine's all-orders map, the algo's order map and
 * the registration's order map) keep the context; only the live aliases go.
 *
 * @param engine The engine (NULL-safe).
 * @param order_id The order identity (non-NULL).
 */
static inline void c_dma_algo_engine_pop_working_order(dma_algo_engine* engine, const long_md_id* order_id) {
    if (!engine || !engine->working_orders_by_id || !order_id) return;

    char* stored = NULL;

    if (c_bytemap_ex_get_ptr(engine->working_orders_by_id, (const char*) order_id, sizeof(long_md_id), &stored, NULL) != BYTEMAP_OK) return;

    dma_order_ctx* ctx = *(dma_order_ctx**) stored;

    c_bytemap_ex_pop(engine->working_orders_by_id, (const char*) order_id, sizeof(long_md_id), 0, NULL, NULL);

    if (ctx && ctx->order.meta_info.ticker && engine->working_orders_by_ticker) {
        const char* ticker = ctx->order.meta_info.ticker;
        stored             = NULL;

        if (c_bytemap_ex_get_ptr(engine->working_orders_by_ticker, ticker, strlen(ticker), &stored, NULL) == BYTEMAP_OK) {
            dma_order_ctx* head = *(dma_order_ctx**) stored;

            if (head == ctx) {
                if (ctx->next) c_bytemap_ex_set(engine->working_orders_by_ticker, ticker, strlen(ticker), (const char*) &ctx->next, sizeof(dma_order_ctx*), 0, NULL);
                else c_bytemap_ex_pop(engine->working_orders_by_ticker, ticker, strlen(ticker), 0, NULL, NULL);
            }
            else {
                dma_order_ctx* previous = head;
                while (previous && previous->next != ctx) previous = previous->next;
                if (previous) previous->next = ctx->next;
            }

            ctx->next = NULL;
        }
    }

    if (ctx && ctx->algo && ctx->algo->working_orders) c_bytemap_ex_pop(ctx->algo->working_orders, (const char*) order_id, sizeof(long_md_id), 0, NULL, NULL);
}

/**
 * @brief Create, fill and register a sent limit order owned by the engine.
 *
 * The order id is the context address, unique while the order lives.
 *
 * @param engine The engine (NULL-safe).
 * @param algo The algo (NULL-safe).
 * @param limit_price The limit price.
 * @param volume The volume to trade (must be positive).
 * @param timestamp The launch timestamp.
 * @return The borrowed instruction (the context's order), or NULL.
 */
static inline md_trade_instruction* c_dma_algo_engine_new_order(dma_algo_engine* engine, dma_algo* algo, double limit_price, double volume, double timestamp) {
    if (!engine || !engine->orders || !algo || volume <= 0.) return NULL;

    allocator_protocol* allocator = c_ap_protocol_from_ptr(engine->orders);

    // === Step 0: Allocate the order context ===
    // One allocator block carries the instruction and its report index.
    dma_order_ctx*      ctx = (dma_order_ctx*) c_ap_alloc(sizeof(dma_order_ctx), allocator);
    if (!ctx) return NULL;

    ctx->algo = algo;
    ctx->next = NULL;

    if (c_dma_order_ctx_init(ctx, allocator) != DMA_ALGO_OK) {
        c_ap_free(ctx);
        return NULL;
    }

    // === Step 1: Fill the instruction ===
    // The engine owns the instruction; the algo's ticker, side and multiplier
    // seed it, and the id is the context address.
    md_trade_instruction* order = &ctx->order;

    order->meta_info.dtype     = DTYPE_INSTRUCTION;
    order->meta_info.ticker    = algo->ticker;
    order->meta_info.timestamp = timestamp;

    order->limit_price = limit_price;
    order->volume      = volume;
    order->side        = algo->side;
    order->order_type  = ORDER_LIMIT;
    order->multiplier  = algo->multiplier;
    order->order_state = STATE_SENT;
    order->ts_placed   = timestamp;

    order->order_id.id_type               = MID_UINT64;
    ((uint64_t*) order->order_id.data)[0] = (uint64_t) (uintptr_t) ctx;

    // === Step 2: Index the order in the engine ===
    // Every order is in the all-orders map; a live order is also in the id map
    // and in its ticker's working list, so the event entries route without a
    // scan.
    const long_md_id* order_id = &order->order_id;

    if (c_bytemap_ex_set(engine->orders, (const char*) order_id, sizeof(long_md_id), (const char*) &ctx, sizeof(dma_order_ctx*), 0, NULL) != BYTEMAP_OK) {
        c_dma_order_ctx_dealloc(ctx);
        c_ap_free(ctx);
        return NULL;
    }

    if (!engine->working_orders_by_id || c_bytemap_ex_set(engine->working_orders_by_id, (const char*) order_id, sizeof(long_md_id), (const char*) &ctx, sizeof(dma_order_ctx*), 0, NULL) != BYTEMAP_OK) {
        c_bytemap_ex_pop(engine->orders, (const char*) order_id, sizeof(long_md_id), 0, NULL, NULL);
        c_dma_order_ctx_dealloc(ctx);
        c_ap_free(ctx);
        return NULL;
    }

    const char* ticker = order->meta_info.ticker;

    if (ticker && engine->working_orders_by_ticker) {
        char*          stored = NULL;
        dma_order_ctx* head   = NULL;

        if (c_bytemap_ex_get_ptr(engine->working_orders_by_ticker, ticker, strlen(ticker), &stored, NULL) == BYTEMAP_OK) head = *(dma_order_ctx**) stored;

        ctx->next = NULL;

        if (head) {
            dma_order_ctx* tail = head;
            while (tail->next) tail = tail->next;
            tail->next = ctx;
        }
        else {
            c_bytemap_ex_set(engine->working_orders_by_ticker, ticker, strlen(ticker), (const char*) &ctx, sizeof(dma_order_ctx*), 0, NULL);
        }
    }

    // === Step 3: Index the order on the algo and its registration ===
    // The algo knows every order it created and aliases the live ones; the
    // registration knows every order its algos created.
    c_bytemap_ex_set(algo->working_orders, (const char*) order_id, sizeof(long_md_id), (const char*) &order, sizeof(md_trade_instruction*), 0, NULL);
    c_bytemap_ex_set(algo->orders, (const char*) order_id, sizeof(long_md_id), (const char*) &order, sizeof(md_trade_instruction*), 0, NULL);

    dma_algo_ctx* factory_ctx = c_dma_algo_registry_get(engine->registry, algo->algo_reg_id);

    if (factory_ctx && factory_ctx->orders) {
        c_bytemap_ex_set(factory_ctx->orders, (const char*) order_id, sizeof(long_md_id), (const char*) &ctx, sizeof(dma_order_ctx*), 0, NULL);
    }

    return order;
}

/**
 * @brief Drop an algo's order contexts from the engine.
 *
 * The engine owns the order contexts; the algo record is owned by its position
 * and is not freed here.
 *
 * @param engine The engine (NULL-safe).
 * @param algo The algo (NULL-safe).
 */
static inline void c_dma_algo_engine_free_orders(dma_algo_engine* engine, dma_algo* algo) {
    if (!engine || !engine->orders || !algo) return;

    dma_algo_ctx* factory_ctx = engine->registry ? c_dma_algo_registry_get(engine->registry, algo->algo_reg_id) : NULL;

    int           found = 1;

    while (found) {
        found = 0;

        bytemap_entry* entry = engine->orders->first;

        while (entry) {
            dma_order_ctx* ctx = *(dma_order_ctx**) entry->value;

            if (ctx && ctx->algo == algo) {
                long_md_id order_id;
                memcpy(&order_id, &ctx->order.order_id, sizeof(long_md_id));

                c_dma_algo_engine_pop_working_order(engine, &order_id);

                if (factory_ctx && factory_ctx->orders) {
                    c_bytemap_ex_pop(factory_ctx->orders, (const char*) &order_id, sizeof(long_md_id), 0, NULL, NULL);
                }

                c_bytemap_ex_pop(engine->orders, (const char*) &order_id, sizeof(long_md_id), 0, NULL, NULL);
                c_dma_order_ctx_dealloc(ctx);
                c_ap_free(ctx);

                found = 1;
                break;
            }

            entry = entry->next;
        }
    }

    algo->working_volume = 0.;
}

/**
 * @brief Handle an order-state event for the algo that owns the order; runs once per event.
 *
 * A rejected or invalid order dispatches on_rejected, a canceled order
 * on_canceled (a venue may report the cancel as order data), and a placed
 * order on_listed. A terminal order leaves the working indexes, the algo's
 * working volume is refreshed from the live set, and only then is the callback
 * dispatched, so the algo sees the state it will act on.
 *
 * @param engine The engine (NULL-safe).
 * @param order The order (NULL-safe; its id must be live).
 * @return DMA_ALGO_OK, DMA_ALGO_ERR_INVALID_INPUT, or DMA_ALGO_ERR_NOT_FOUND.
 */
static inline int c_dma_algo_engine_on_order(dma_algo_engine* engine, const md_trade_instruction* order) {
    if (!engine || !order) return DMA_ALGO_ERR_INVALID_INPUT;

    char* stored = NULL;

    if (c_bytemap_ex_get_ptr(engine->working_orders_by_id, (const char*) &order->order_id, sizeof(long_md_id), &stored, NULL) != BYTEMAP_OK) return DMA_ALGO_ERR_NOT_FOUND;

    dma_order_ctx* ctx  = *(dma_order_ctx**) stored;
    dma_algo*      algo = ctx->algo;

    if (!algo) return DMA_ALGO_ERR_NOT_FOUND;

    if (order != &ctx->order) memcpy(&ctx->order, order, sizeof(md_trade_instruction));

    md_trade_instruction* tracked  = &ctx->order;
    dma_algo_order_func   callback = NULL;

    switch (tracked->order_state) {
        case STATE_REJECTED:
        case STATE_INVALID:
            callback = algo->callbacks.on_rejected;
            break;
        case STATE_CANCELED:
            callback = algo->callbacks.on_canceled;
            break;
        case STATE_PLACED:
            callback = algo->callbacks.on_listed;
            break;
        default:
            break;
    }

    if (c_md_state_done(tracked->order_state)) {
        c_dma_algo_engine_pop_working_order(engine, &tracked->order_id);

        double         working_volume = 0.;
        bytemap_entry* entry          = algo->working_orders->first;

        while (entry) {
            md_trade_instruction* working = *(md_trade_instruction**) entry->value;

            if (working) {
                double remaining = working->volume - working->filled_volume;

                if (remaining > 0.) working_volume += remaining;
            }

            entry = entry->next;
        }

        algo->working_volume = working_volume;
    }

    if (callback) callback(algo->DMA, algo, tracked, algo->callbacks.user_data);

    return DMA_ALGO_OK;
}

/**
 * @brief Handle a fill report for the algo that owns the order; runs once per applied report.
 *
 * The order context applies the report first: a report that arrives while the
 * order is canceling is the cancel confirmation and dispatches on_canceled;
 * an applied fill updates the algo aggregates (filled volume and notional, fee,
 * signed exposure and cash flow), leaves the working indexes when the order is
 * finished, and refreshes the algo's working volume from the live set. The
 * callback is dispatched last. A report that moves nothing (a repeated trade id
 * or a zero-volume report) dispatches nothing.
 *
 * @param engine The engine (NULL-safe).
 * @param trade The fill (NULL-safe; its order id must be live).
 * @return DMA_ALGO_OK, DMA_ALGO_ERR_INVALID_INPUT, or DMA_ALGO_ERR_NOT_FOUND.
 */
static inline int c_dma_algo_engine_on_trade(dma_algo_engine* engine, const md_trade_report* trade) {
    if (!engine || !trade) return DMA_ALGO_ERR_INVALID_INPUT;

    char* stored = NULL;

    if (c_bytemap_ex_get_ptr(engine->working_orders_by_id, (const char*) &trade->order_id, sizeof(long_md_id), &stored, NULL) != BYTEMAP_OK) return DMA_ALGO_ERR_NOT_FOUND;

    dma_order_ctx* ctx  = *(dma_order_ctx**) stored;
    dma_algo*      algo = ctx->algo;

    if (!algo) return DMA_ALGO_ERR_NOT_FOUND;

    md_trade_instruction* order = &ctx->order;

    double                filled_before = order->filled_volume;

    int                   ret_code = c_dma_order_ctx_fill(ctx, trade);
    if (ret_code != DMA_ALGO_OK) return ret_code;

    bool canceled = order->order_state == STATE_CANCELED;

    if (!canceled && order->filled_volume == filled_before) return DMA_ALGO_OK;

    if (canceled) {
        c_dma_algo_engine_pop_working_order(engine, &order->order_id);
    }
    else {
        algo->filled_volume += trade->volume;
        algo->filled_notional += fabs(trade->notional);
        algo->fee += trade->fee;
        algo->exposure_volume += trade->volume * c_md_side_sign(trade->side);
        algo->cash_flow = -algo->filled_notional * c_md_side_sign(algo->side);

        if (c_md_state_done(order->order_state)) c_dma_algo_engine_pop_working_order(engine, &order->order_id);
    }

    double         working_volume = 0.;
    bytemap_entry* entry          = algo->working_orders->first;

    while (entry) {
        md_trade_instruction* working = *(md_trade_instruction**) entry->value;

        if (working) {
            double remaining = working->volume - working->filled_volume;

            if (remaining > 0.) working_volume += remaining;
        }

        entry = entry->next;
    }

    algo->working_volume = working_volume;

    if (canceled) {
        if (algo->callbacks.on_canceled) algo->callbacks.on_canceled(algo->DMA, algo, order, algo->callbacks.user_data);

        return DMA_ALGO_OK;
    }

    if (algo->callbacks.on_filled) algo->callbacks.on_filled(algo->DMA, algo, trade, algo->callbacks.user_data);

    return DMA_ALGO_OK;
}

/**
 * @brief Dispatch a market-data observation to every algo in the working set of the ticker.
 *
 * The engine keeps one working list per ticker, linked through the algo's
 * prev/next and driven by c_dma_algo_set_status. The list holds every algo from
 * creation until it completes or errors - STOPPING and REJECTED keep their seat
 * - so each listed algo is dispatched once per observation regardless of how
 * many orders it has.
 *
 * @param engine The engine (NULL-safe).
 * @param market_data The observation (NULL-safe).
 */
static inline void c_dma_algo_engine_on_md(dma_algo_engine* engine, const md_variant* market_data) {
    if (!engine || !engine->working_algo || !market_data) return;

    const char* ticker = market_data->meta_info.ticker;

    if (!ticker) return;

    char* stored = NULL;

    if (c_bytemap_ex_get_ptr(engine->working_algo, ticker, strlen(ticker), &stored, NULL) != BYTEMAP_OK) return;

    dma_algo* algo = *(dma_algo**) stored;

    while (algo) {
        dma_algo* next = algo->next;  // The callback may stop and unlink the algo.

        if (algo->callbacks.on_md) algo->callbacks.on_md(algo->DMA, algo, market_data, algo->callbacks.user_data);

        algo = next;
    }
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

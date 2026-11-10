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
 * @param reg_id The id of the registration that builds the algo.
 * @param allocator Allocator for the record.
 * @return The new record, or NULL.
 */
typedef dma_algo* (*dma_algo_new)(dma_algo_engine* engine, dma* DMA, const char* ticker, md_side side, double target_volume, const void* params, uint32_t reg_id, allocator_protocol* allocator);

/**
 * @struct dma_algo_ctx
 * @brief One registration of an algo kind.
 *
 * The context owns the maps of every algo built under this registration and
 * every order those algos created. An override preserves the replaced context
 * in the registry's `overrided` map, so its maps are never dropped.
 */
struct dma_algo_ctx {
    dma_algo_new algo_new_fn;  // NOT owned - the factory.
    bytemap*     orders;       // OWNED - order id -> borrowed dma_order_ctx*.
    bytemap*     algos;        // OWNED - algo id -> borrowed dma_algo*.
    uint32_t     reg_id;
};

/**
 * @struct dma_order_ctx
 * @brief One engine-owned order.
 *
 * `order` is the instruction the algo launched; `next` links the orders of one
 * ticker so market data can be routed without a per-order probe.
 */
struct dma_order_ctx {
    md_trade_instruction  order;
    dma_algo*             algo;  // NOT owned - the algo that launched the order.
    struct dma_order_ctx* next;  // Ticker-list link.
};

/**
 * @struct dma_algo_registry
 * @brief Named algo registrations; re-registering a name overrides it and raises the id.
 */
struct dma_algo_registry {
    bytemap* entries;    // OWNED - name -> dma_algo_ctx.
    bytemap* overrided;  // OWNED - reg id -> replaced dma_algo_ctx.
    uint32_t seq_id;     // Id handed to the latest registration.
};

/**
 * @struct dma_algo_engine
 * @brief The process-global algo engine state.
 */
struct dma_algo_engine {
    mds*               MDS;                       // NOT owned - the market data source.
    dma_algo_registry* registry;                  // OWNED - named factories.
    bytemap*           orders;                    // OWNED - order id -> owned dma_order_ctx*.
    bytemap*           working_orders_by_id;      // OWNED - live order id -> owned dma_order_ctx* (alias of orders).
    bytemap*           working_orders_by_ticker;  // OWNED - ticker -> dma_order_ctx* list head.
    double             timestamp;                 // Latest observation timestamp.
};

// ========== Forward Declarations ==========

// Algo registry
static inline dma_algo_registry* c_dma_algo_registry_new(allocator_protocol* allocator);
static inline int                c_dma_algo_registry_init(dma_algo_registry* registry, allocator_protocol* allocator);
static inline void               c_dma_algo_registry_dealloc(dma_algo_registry* registry);
static inline void               c_dma_algo_registry_free(dma_algo_registry* registry);
static inline int                c_dma_algo_registry_reg(dma_algo_registry* registry, const char* name, dma_algo_new algo_new_fn, uint32_t* out_reg_id);
static inline dma_algo_ctx*      c_dma_algo_registry_get(const dma_algo_registry* registry, uint32_t reg_id);

// Algo engine
static inline dma_algo_engine*   c_dma_algo_engine_new(mds* MDS, allocator_protocol* allocator);
static inline int                c_dma_algo_engine_init(dma_algo_engine* engine, mds* MDS, allocator_protocol* allocator);
static inline void               c_dma_algo_engine_dealloc(dma_algo_engine* engine);
static inline void               c_dma_algo_engine_free(dma_algo_engine* engine);
static inline dma_algo*          c_dma_algo_engine_new_algo(dma_algo_engine* engine, const char* algo_name, dma* DMA, const char* ticker, md_side side, double target_volume, const void* params, allocator_protocol* allocator);
static inline double             c_dma_algo_engine_compare_price(md_side side, double limit_price, double original_limit, dma_algo_limit_mode mode);
static inline double             c_dma_algo_engine_calculate_limit(dma_algo_engine* engine, const char* ticker, md_side side, double limit_price, double limit_adjust_factor, dma_algo_limit_mode mode);

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

    registry->entries = c_bytemap_ex_new(0, sizeof(dma_algo_ctx), allocator);
    if (!registry->entries) return DMA_ALGO_ERR_OOM;

    registry->overrided = c_bytemap_ex_new(0, sizeof(dma_algo_ctx), allocator);
    if (!registry->overrided) {
        c_bytemap_ex_free(registry->entries);
        registry->entries = NULL;
        return DMA_ALGO_ERR_OOM;
    }

    return DMA_ALGO_OK;
}

/**
 * @brief Tear down a caller-allocated registry, releasing every context's maps.
 * @param registry The registry (NULL-safe).
 */
static inline void c_dma_algo_registry_dealloc(dma_algo_registry* registry) {
    if (!registry) return;

    if (registry->entries) {
        bytemap_entry* entry = registry->entries->first;

        while (entry) {
            bytemap_entry* next = entry->next;
            c_dma_algo_ctx_dealloc((dma_algo_ctx*) entry->value);
            entry = next;
        }

        c_bytemap_ex_free(registry->entries);
        registry->entries = NULL;
    }

    if (registry->overrided) {
        bytemap_entry* entry = registry->overrided->first;

        while (entry) {
            bytemap_entry* next = entry->next;
            c_dma_algo_ctx_dealloc((dma_algo_ctx*) entry->value);
            entry = next;
        }

        c_bytemap_ex_free(registry->overrided);
        registry->overrided = NULL;
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
 * An override keeps the replaced context in `overrided`, keyed by its
 * registration id; every call raises the id.
 *
 * @param registry The registry (NULL-safe).
 * @param name The algo name (non-empty).
 * @param algo_new_fn The factory (NULL is refused).
 * @param out_reg_id Optional: receives the new registration id.
 * @return DMA_ALGO_OK, DMA_ALGO_ERR_INVALID_INPUT, or DMA_ALGO_ERR_OOM.
 */
static inline int c_dma_algo_registry_reg(dma_algo_registry* registry, const char* name, dma_algo_new algo_new_fn, uint32_t* out_reg_id) {
    if (!registry || !registry->entries || !registry->overrided || !name || !algo_new_fn) return DMA_ALGO_ERR_INVALID_INPUT;

    allocator_protocol* allocator = c_ap_protocol_from_ptr(registry->entries);

    dma_algo_ctx        empty_ctx;
    memset(&empty_ctx, 0, sizeof(dma_algo_ctx));

    char* stored = NULL;

    int   ret_code = c_bytemap_ex_set_default(
        registry->entries,
        name,
        strlen(name),
        (const char*) &empty_ctx,
        sizeof(dma_algo_ctx),
        0,
        &stored,
        NULL
    );

    if (ret_code != BYTEMAP_OK) return DMA_ALGO_ERR_OOM;

    dma_algo_ctx* ctx = (dma_algo_ctx*) stored;

    if (ctx->algo_new_fn) {
        ret_code = c_bytemap_ex_set(
            registry->overrided,
            (const char*) &ctx->reg_id,
            sizeof(uint32_t),
            (const char*) ctx,
            sizeof(dma_algo_ctx),
            0,
            NULL
        );

        if (ret_code != BYTEMAP_OK) return DMA_ALGO_ERR_OOM;

        // The old maps now belong to the preserved copy; do not release them.
        memset(ctx, 0, sizeof(dma_algo_ctx));
    }

    registry->seq_id += 1;

    ret_code = c_dma_algo_ctx_init(ctx, algo_new_fn, registry->seq_id, allocator);
    if (ret_code != DMA_ALGO_OK) return ret_code;

    if (out_reg_id) *out_reg_id = ctx->reg_id;
    return DMA_ALGO_OK;
}

/**
 * @brief Find the registration context for an id, current or overridden.
 * @param registry The registry (NULL-safe).
 * @param reg_id The registration id.
 * @return The borrowed context, or NULL.
 */
static inline dma_algo_ctx* c_dma_algo_registry_get(const dma_algo_registry* registry, uint32_t reg_id) {
    if (!registry) return NULL;

    bytemap* maps[2] = {registry->entries, registry->overrided};

    for (int map_index = 0; map_index < 2; map_index++) {
        bytemap* map = maps[map_index];

        if (!map) continue;

        bytemap_entry* entry = map->first;

        while (entry) {
            dma_algo_ctx* ctx = (dma_algo_ctx*) entry->value;

            if (ctx->reg_id == reg_id) return ctx;
            entry = entry->next;
        }
    }

    return NULL;
}

// ========== Public APIs (Algo Engine) ==========

/**
 * @brief Allocate an engine with its registry and order indexes.
 * @param MDS The market data source (borrowed).
 * @param allocator Allocator for the block and its children.
 * @return The new engine, or NULL.
 */
static inline dma_algo_engine* c_dma_algo_engine_new(mds* MDS, allocator_protocol* allocator) {
    dma_algo_engine* engine = (dma_algo_engine*) c_ap_alloc(sizeof(dma_algo_engine), allocator);
    if (!engine) return NULL;

    if (c_dma_algo_engine_init(engine, MDS, allocator) != DMA_ALGO_OK) {
        c_dma_algo_engine_dealloc(engine);
        c_ap_free(engine);
        return NULL;
    }

    return engine;
}

/**
 * @brief Initialize a caller-allocated, zeroed engine.
 * @param engine The zeroed engine.
 * @param MDS The market data source (borrowed).
 * @param allocator Allocator for the children; derived from engine when NULL.
 * @return DMA_ALGO_OK, DMA_ALGO_ERR_INVALID_INPUT, or DMA_ALGO_ERR_OOM.
 */
static inline int c_dma_algo_engine_init(dma_algo_engine* engine, mds* MDS, allocator_protocol* allocator) {
    if (!engine) return DMA_ALGO_ERR_INVALID_INPUT;
    if (!allocator) allocator = c_ap_protocol_from_ptr(engine);

    engine->MDS      = MDS;
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

            if (ctx) c_ap_free(ctx);
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
 * @brief Build an algo of the named kind through the registry.
 *
 * The algo identity is generated by the factory; the entry's registration id
 * is handed to the factory so the algo carries it.
 *
 * @param engine The engine (NULL-safe).
 * @param algo_name The registered algo name.
 * @param DMA The process-global access (borrowed).
 * @param ticker The instrument.
 * @param side The trade side.
 * @param target_volume The volume to trade.
 * @param params Algo-kind parameters (borrowed).
 * @param allocator Allocator for the record.
 * @return The new record, or NULL.
 */
static inline dma_algo* c_dma_algo_engine_new_algo(dma_algo_engine* engine, const char* algo_name, dma* DMA, const char* ticker, md_side side, double target_volume, const void* params, allocator_protocol* allocator) {
    if (!engine || !engine->registry || !engine->registry->entries || !algo_name) return NULL;

    char* stored = NULL;

    if (c_bytemap_ex_get_ptr(engine->registry->entries, algo_name, strlen(algo_name), &stored, NULL) != BYTEMAP_OK) return NULL;

    const dma_algo_ctx* ctx = (const dma_algo_ctx*) stored;

    if (!ctx->algo_new_fn) return NULL;

    return ctx->algo_new_fn(engine, DMA, ticker, side, target_volume, params, ctx->reg_id, allocator);
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
static inline double c_dma_algo_engine_compare_price(md_side side, double limit_price, double original_limit, dma_algo_limit_mode mode) {
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

/**
 * @brief Calculate the order limit from the market price and the algo limits.
 *
 * The absolute limit, the market factor and the market price are combined;
 * a missing market price yields NaN.
 *
 * @param engine The engine (NULL-safe).
 * @param ticker The instrument (non-empty).
 * @param side The trade side.
 * @param limit_price The absolute limit (NaN when unset).
 * @param limit_adjust_factor The market-price factor (NaN when unset).
 * @param mode Strict or loose.
 * @return The calculated limit, or NaN.
 */
static inline double c_dma_algo_engine_calculate_limit(dma_algo_engine* engine, const char* ticker, md_side side, double limit_price, double limit_adjust_factor, dma_algo_limit_mode mode) {
    if (!engine || !ticker) return NAN;

    double market_price = c_mds_get_market_price(engine->MDS, ticker);
    if (isnan(market_price)) return NAN;

    double calculated = NAN;

    calculated = c_dma_algo_engine_compare_price(side, limit_price, calculated, mode);

    if (!isnan(limit_adjust_factor)) {
        double adjusted = market_price * (1. + limit_adjust_factor * c_md_side_sign(side));
        calculated      = c_dma_algo_engine_compare_price(side, adjusted, calculated, mode);
    }

    calculated = c_dma_algo_engine_compare_price(side, market_price, calculated, mode);
    return calculated;
}

#endif /* C_DMA_ALGO_H */

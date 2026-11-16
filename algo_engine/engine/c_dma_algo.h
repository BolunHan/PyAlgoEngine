#ifndef C_DMA_ALGO_H
#define C_DMA_ALGO_H

#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <algo_engine/base/c_market_data/c_market_data.h>
#include <algo_engine/engine/c_market_engine.h>
#include <cbase/allocator_protocol/c_allocator_protocol.h>
#include <cbase/bytemap/c_bytemap.h>

typedef struct dma          dma;
typedef struct dma_algo     dma_algo;
typedef struct dma_position dma_position;

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

/** @enum dma_order_flag
 *  @brief The order transitions that move working volume, one bit each.
 *
 * The listed bit is immutable once set; every higher bit is a stop flag. A
 * stop flag records that the order's remaining volume left the host, so any
 * stop bit already set makes a repeated terminal update a no-op. */
typedef enum dma_order_flag {
    DMA_ORDER_LISTED   = 1 << 0,  // Immutable: the remaining volume entered the host.
    DMA_ORDER_CANCELED = 1 << 1,  // Stop flag: the cancel released the remaining volume.
    DMA_ORDER_REJECTED = 1 << 2,  // Stop flag: the rejection released the remaining volume.
    DMA_ORDER_ERROR    = 1 << 3,  // Stop flag: the invalid transition released the remaining volume.
} dma_order_flag;

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
    uint8_t               flags;    // Applied transitions; dma_order_flag bits.
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
    bytemap*           position;                  // OWNED - algo id -> borrowed dma_position* (the position that hosts the record).
} dma_algo_engine;

// ========== Forward Declarations ==========

// Order context
static inline int                   c_dma_order_ctx_init(dma_order_ctx* ctx, allocator_protocol* allocator);
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
static inline int                   c_dma_algo_engine_clear(dma_algo_engine* engine);
static inline void                  c_dma_algo_engine_dealloc(dma_algo_engine* engine);
static inline void                  c_dma_algo_engine_free(dma_algo_engine* engine);

// Engine entries
static inline void                  c_dma_algo_engine_pop_working_order(dma_algo_engine* engine, const long_md_id* order_id);
static inline md_trade_instruction* c_dma_algo_engine_new_order(dma_algo_engine* engine, dma_algo* algo, double limit_price, double volume);
static inline void                  c_dma_algo_engine_free_orders(dma_algo_engine* engine, dma_algo* algo);
static inline int                   c_dma_algo_engine_on_order(dma_algo_engine* engine, const md_trade_instruction* order);
static inline int                   c_dma_algo_engine_on_order_inplace(dma_algo_engine* engine, dma_order_ctx* ctx);
static inline int                   c_dma_algo_engine_on_trade(dma_algo_engine* engine, const md_trade_report* trade);
static inline void                  c_dma_algo_engine_on_md(dma_algo_engine* engine, const md_variant* market_data);

// Limit helpers
static inline double                c_dma_algo_compare_limit(md_side side, double limit_price, double original_limit, dma_algo_limit_mode mode);

#include <algo_engine/engine/algo/c_algo_template.h>
#include <algo_engine/engine/c_dma.h>
#include <algo_engine/engine/c_position.h>

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

    engine->position = c_bytemap_ex_new(8, sizeof(dma_position*), allocator);
    if (!engine->position) return DMA_ALGO_ERR_OOM;

    return DMA_ALGO_OK;
}

/**
 * @brief Remove every algo record, order context and stored trade the engine owns.
 *
 * The order contexts go first: each frees its owned trade-report copies and
 * pops its id from the engine's working index, the owning algo's history and
 * live indexes, and the registration's order index. The algo records follow,
 * each leaving the working list, the engine index and its hosting position.
 * The engine's maps, the registry and every registration survive, so the
 * engine can build new records after the sweep.
 *
 * @param engine The engine (NULL-safe).
 * @return DMA_ALGO_OK, or DMA_ALGO_ERR_INVALID_INPUT.
 */
static inline int c_dma_algo_engine_clear(dma_algo_engine* engine) {
    if (!engine) return DMA_ALGO_ERR_INVALID_INPUT;

    if (engine->orders) {
        bytemap_entry* entry = engine->orders->first;

        while (entry) {
            bytemap_entry* next = entry->next;
            dma_order_ctx* ctx  = *(dma_order_ctx**) entry->value;

            if (ctx) {
                const long_md_id* order_id = &ctx->order.order_id;

                if (engine->working_orders_by_id) c_bytemap_ex_pop(engine->working_orders_by_id, (const char*) order_id, sizeof(long_md_id), 0, NULL, NULL);

                if (ctx->algo) {
                    if (ctx->algo->working_orders) c_bytemap_ex_pop(ctx->algo->working_orders, (const char*) order_id, sizeof(long_md_id), 0, NULL, NULL);
                    if (ctx->algo->orders) c_bytemap_ex_pop(ctx->algo->orders, (const char*) order_id, sizeof(long_md_id), 0, NULL, NULL);

                    dma_algo_ctx* factory_ctx = engine->registry ? c_dma_algo_registry_get(engine->registry, ctx->algo->algo_reg_id) : NULL;

                    if (factory_ctx && factory_ctx->orders) c_bytemap_ex_pop(factory_ctx->orders, (const char*) order_id, sizeof(long_md_id), 0, NULL, NULL);
                }

                c_dma_order_ctx_dealloc(ctx);
                c_ap_free(ctx);
            }

            entry = next;
        }

        c_bytemap_ex_clear(engine->orders);
    }

    if (engine->working_orders_by_id) c_bytemap_ex_clear(engine->working_orders_by_id);
    if (engine->working_orders_by_ticker) c_bytemap_ex_clear(engine->working_orders_by_ticker);

    if (engine->algo) {
        while (c_bytemap_ex_len(engine->algo)) {
            bytemap_entry* entry = engine->algo->first;
            dma_algo*      algo  = *(dma_algo**) entry->value;
            uuid_t         key;

            memcpy(key, entry->key, sizeof(uuid_t));

            // The hosting position indexes the record freed below; drop the
            // reference and its routing entry in the same pass.
            if (algo) {
                dma_position* position        = NULL;
                char*         position_stored = NULL;

                if (engine->position && c_bytemap_ex_get_ptr(engine->position, (const char*) algo->algo_id, sizeof(uuid_t), &position_stored, NULL) == BYTEMAP_OK) position = *(dma_position**) position_stored;

                if (position) c_dma_position_pop_algo(position, key);
            }

            c_bytemap_ex_pop(engine->algo, (const char*) key, sizeof(uuid_t), 0, NULL, NULL);

            if (algo) {
                algo->DMA = NULL;  // The working list is cleared wholesale below.
                c_dma_algo_free(algo);
            }
        }
    }

    if (engine->working_algo) c_bytemap_ex_clear(engine->working_algo);

    if (engine->position) c_bytemap_ex_clear(engine->position);

    engine->error_algo = NULL;

    // The registration contexts survive; their borrowed indexes are emptied so
    // a new record of the same kind starts from a clean map.
    if (engine->registry && engine->registry->id_map) {
        bytemap_entry* entry = engine->registry->id_map->first;

        while (entry) {
            bytemap_entry* next = entry->next;
            dma_algo_ctx*  ctx  = *(dma_algo_ctx**) entry->value;

            if (ctx) {
                if (ctx->orders) c_bytemap_ex_clear(ctx->orders);
                if (ctx->algos) c_bytemap_ex_clear(ctx->algos);
            }

            entry = next;
        }
    }

    return DMA_ALGO_OK;
}

/**
 * @brief Tear down a caller-allocated engine, leaving it zeroed.
 *
 * The record sweep runs first; the engine's maps, the registry and every
 * registration context are then released.
 *
 * @param engine The engine (NULL-safe).
 */
static inline void c_dma_algo_engine_dealloc(dma_algo_engine* engine) {
    if (!engine) return;

    c_dma_algo_engine_clear(engine);

    if (engine->orders) {
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

    if (engine->algo) {
        c_bytemap_ex_free(engine->algo);
        engine->algo = NULL;
    }

    if (engine->working_algo) {
        c_bytemap_ex_free(engine->working_algo);
        engine->working_algo = NULL;
    }

    if (engine->position) {
        c_bytemap_ex_free(engine->position);
        engine->position = NULL;
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
    if (!engine || !order_id) return;

    char* stored = NULL;

    if (c_bytemap_ex_get_ptr(engine->working_orders_by_id, (const char*) order_id, sizeof(long_md_id), &stored, NULL) != BYTEMAP_OK) return;

    dma_order_ctx* ctx = *(dma_order_ctx**) stored;

    c_bytemap_ex_pop(engine->working_orders_by_id, (const char*) order_id, sizeof(long_md_id), 0, NULL, NULL);

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

    c_bytemap_ex_pop(ctx->algo->working_orders, (const char*) order_id, sizeof(long_md_id), 0, NULL, NULL);
}

/**
 * @brief Create, fill and register a sent limit order owned by the engine.
 *
 * The order id is the context address, unique while the order lives. The
 * order's session datetime is copied whole from the access's MDS
 * (`algo->DMA->mds->ts`); `ts_placed` is that pack's unix_ts.
 *
 * @param engine The engine (NULL-safe).
 * @param algo The algo (NULL-safe).
 * @param limit_price The limit price.
 * @param volume The volume to trade (must be positive).
 * @return The borrowed instruction (the context's order), or NULL.
 */
static inline md_trade_instruction* c_dma_algo_engine_new_order(dma_algo_engine* engine, dma_algo* algo, double limit_price, double volume) {
    if (!engine || !algo || !isfinite(volume) || volume <= 0.) return NULL;

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

    order->meta_info.dtype  = DTYPE_INSTRUCTION;
    order->meta_info.ticker = algo->ticker;
    order->meta_info.dt     = algo->DMA->mds->ts;

    order->limit_price = limit_price;
    order->volume      = volume;
    order->side        = algo->side;
    order->order_type  = ORDER_LIMIT;
    order->multiplier  = algo->multiplier;
    order->order_state = STATE_IDLE;
    order->ts_placed   = order->meta_info.dt.unix_ts;

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

    if (c_bytemap_ex_set(engine->working_orders_by_id, (const char*) order_id, sizeof(long_md_id), (const char*) &ctx, sizeof(dma_order_ctx*), 0, NULL) != BYTEMAP_OK) {
        c_bytemap_ex_pop(engine->orders, (const char*) order_id, sizeof(long_md_id), 0, NULL, NULL);
        c_dma_order_ctx_dealloc(ctx);
        c_ap_free(ctx);
        return NULL;
    }

    const char*    ticker = order->meta_info.ticker;

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

    // === Step 3: Index the order on the algo and its registration ===
    // The algo knows every order it created and aliases the live ones; the
    // registration knows every order its algos created.
    c_bytemap_ex_set(algo->working_orders, (const char*) order_id, sizeof(long_md_id), (const char*) &order, sizeof(md_trade_instruction*), 0, NULL);
    c_bytemap_ex_set(algo->orders, (const char*) order_id, sizeof(long_md_id), (const char*) &order, sizeof(md_trade_instruction*), 0, NULL);

    dma_algo_ctx* factory_ctx = c_dma_algo_registry_get(engine->registry, algo->algo_reg_id);

    if (factory_ctx) {
        c_bytemap_ex_set(factory_ctx->orders, (const char*) order_id, sizeof(long_md_id), (const char*) &ctx, sizeof(dma_order_ctx*), 0, NULL);
    }

    // The live set grew by this order, whose remaining volume is the volume sent.
    algo->working_volume += volume;

    // The order's remaining volume entered the algo; the immutable listed bit
    // records it once, and the hosting position mirrors the same delta. The
    // bit is set even without a host: a later host takes the volume through
    // add_algo, so no order event may add it again.
    ctx->flags |= DMA_ORDER_LISTED;

    // === Step 4: Mirror the sent volume into the hosting position ===
    dma_position* position        = NULL;
    char*         position_stored = NULL;

    if (c_bytemap_ex_get_ptr(engine->position, (const char*) algo->algo_id, sizeof(uuid_t), &position_stored, NULL) == BYTEMAP_OK) position = *(dma_position**) position_stored;

    if (position) {
        dma_position_ticker defaults;
        memset(&defaults, 0, sizeof(defaults));

        char* ticker_stored = NULL;

        if (c_bytemap_ex_set_default(position->tickers, algo->ticker, strlen(algo->ticker), (const char*) &defaults, sizeof(defaults), 0, &ticker_stored, NULL) == BYTEMAP_OK) {
            dma_position_ticker* cache = (dma_position_ticker*) ticker_stored;
            int                  sign  = c_md_side_sign(algo->side);

            if (sign > 0) cache->working_long += volume;
            else if (sign < 0) cache->working_short += volume;
        }

        if (c_dma_algo_status_is_active(algo->status)) c_bytemap_ex_set(position->working_algos, (const char*) algo->algo_id, sizeof(uuid_t), (const char*) &algo, sizeof(dma_algo*), 0, NULL);
        else c_bytemap_ex_pop(position->working_algos, (const char*) algo->algo_id, sizeof(uuid_t), 0, NULL, NULL);
    }

    return order;
}

/**
 * @brief Drop an algo's order contexts from the engine.
 * The engine owns the order contexts; the algo record is owned by the engine's
 * algo index and is not freed here. The hosting position keeps its cached view
 * until the record leaves its host or the position is cleared.
 * @param engine The engine (NULL-safe).
 * @param algo The algo (NULL-safe).
 */
static inline void c_dma_algo_engine_free_orders(dma_algo_engine* engine, dma_algo* algo) {
    if (!engine || !algo) return;

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

                if (factory_ctx) {
                    c_bytemap_ex_pop(factory_ctx->orders, (const char*) &order_id, sizeof(long_md_id), 0, NULL, NULL);
                }

                // The algo's own history index points at the context: the key
                // must leave before the block is freed.
                c_bytemap_ex_pop(algo->orders, (const char*) &order_id, sizeof(long_md_id), 0, NULL, NULL);

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
 * @brief Handle an order update for the algo that owns the order; runs once per update.
 *
 * The main entry. An exchange order-state event arrives in its own buffer and
 * is classified against the stored order's state column: a tolerated event is
 * a no-op, a banned event is refused, and an applied event is copied into the
 * stored order. An internal update arrives in the stored order itself, already
 * carrying its new state; it prints one line to stderr and routes to
 * `c_dma_algo_engine_on_order_inplace`, because its pre-state is gone.
 *
 * The applied update moves the one working-volume delta, leaves the working
 * indexes when the order is done, dispatches the state's callback and mirrors
 * the delta into the hosting position.
 *
 * @param engine The engine (NULL-safe).
 * @param order The update (NULL-safe; its id must be live).
 * @return DMA_ALGO_OK, DMA_ALGO_ERR_INVALID_INPUT, or DMA_ALGO_ERR_NOT_FOUND.
 */
static inline int c_dma_algo_engine_on_order(dma_algo_engine* engine, const md_trade_instruction* order) {
    if (!engine || !order) return DMA_ALGO_ERR_INVALID_INPUT;

    // === Step 0 ===
    // Resolve the tracked order context.
    char* stored = NULL;

    if (c_bytemap_ex_get_ptr(engine->working_orders_by_id, (const char*) &order->order_id, sizeof(long_md_id), &stored, NULL) != BYTEMAP_OK) return DMA_ALGO_ERR_NOT_FOUND;

    dma_order_ctx* ctx = *(dma_order_ctx**) stored;

    // === Step 0a ===
    // The access writes the stored order in place; an exchange event arrives in
    // its own buffer. An in-place update owns its pre-state and routes past the
    // guard to the entry that applies a written state.
    if (order == &ctx->order) {
        fprintf(
            stderr,
            "[DMA_ALGO] in-place order update; order <%s> state <%s>; routed to c_dma_algo_engine_on_order_inplace\n",
            ctx->order.meta_info.ticker,
            c_md_state_name(order->order_state)
        );
        return c_dma_algo_engine_on_order_inplace(engine, ctx);
    }

    dma_algo*      algo = ctx->algo;

    // === Step 0b ===
    // Classify the reported state against the tracked one. A tolerated report
    // is a no-op; a banned report is refused and leaves the order unchanged.
    md_order_state from = ctx->order.order_state;
    md_order_state to   = order->order_state;

    if (to == STATE_IDLE || to == STATE_UNKNOWN) {
        c_dma_refuse(algo->DMA, &ctx->order, "order report");
        return DMA_ALGO_ERR_INVALID_INPUT;
    }

    if (to == STATE_PLACED) {
        switch (from) {
            case STATE_IDLE:
            case STATE_PENDING:
            case STATE_SENT:
                break;
            case STATE_PLACED:
            case STATE_PARTFILLED:
            case STATE_FILLED:
            case STATE_CANCELING:
            case STATE_CANCELED:
                return DMA_ALGO_OK;
            default:
                c_dma_refuse(algo->DMA, &ctx->order, "placed");
                return DMA_ALGO_ERR_INVALID_INPUT;
        }
    }
    else if (to == STATE_REJECTED) {
        if (from == STATE_REJECTED) return DMA_ALGO_OK;
        if (from != STATE_SENT) {
            c_dma_refuse(algo->DMA, &ctx->order, "exchange reject");
            return DMA_ALGO_ERR_INVALID_INPUT;
        }
    }
    else if (to == STATE_CANCELED) {
        if (from == STATE_CANCELED) return DMA_ALGO_OK;
        if (from != STATE_CANCELING && from != STATE_SENT && from != STATE_PLACED && from != STATE_PARTFILLED) {
            c_dma_refuse(algo->DMA, &ctx->order, "cancel confirmation");
            return DMA_ALGO_ERR_INVALID_INPUT;
        }
    }

    // === Step 1 ===
    // Apply the exchange's order-state event to the stored order.
    memcpy(&ctx->order, order, sizeof(md_trade_instruction));
    ctx->order.meta_info.ticker = algo->ticker;

    md_trade_instruction* tracked       = &ctx->order;
    dma_algo_order_func   callback      = NULL;
    uint8_t               terminal_flag = 0;

    // === Step 2 ===
    // Classify the event and its callback.

    switch (tracked->order_state) {
        case STATE_REJECTED:
            callback      = algo->callbacks.on_rejected;
            terminal_flag = DMA_ORDER_REJECTED;
            break;
        case STATE_INVALID:
            callback      = algo->callbacks.on_rejected;
            terminal_flag = DMA_ORDER_ERROR;
            break;
        case STATE_CANCELED:
            callback      = algo->callbacks.on_canceled;
            terminal_flag = DMA_ORDER_CANCELED;
            break;
        case STATE_PLACED:
            callback = algo->callbacks.on_listed;
            break;
        default:
            break;
    }

    // === Step 3 ===
    // Apply the one working-volume delta to the context bits and the algo.
    // Any working state - pending included - opens the remaining volume once,
    // whatever state the exchange reports first; a stop bit closes it once, and a
    // stop recorded before the volume enters still blocks a later reopen.

    bool   working_state = c_md_state_working(tracked->order_state) || tracked->order_state == STATE_PENDING;
    bool   stopped       = ctx->flags > DMA_ORDER_LISTED;
    double working_delta = 0.;

    if (working_state) {
        if (!(ctx->flags & DMA_ORDER_LISTED) && !stopped) {
            ctx->flags |= DMA_ORDER_LISTED;
            working_delta = tracked->volume - tracked->filled_volume;
        }
    }
    else if (terminal_flag && !stopped) {
        ctx->flags |= terminal_flag;
        if (ctx->flags & DMA_ORDER_LISTED) working_delta = -(tracked->volume - tracked->filled_volume);
    }

    algo->working_volume += working_delta;

    // === Step 4 ===
    // Leave the working indexes when the order is done.
    if (c_md_state_done(tracked->order_state)) c_dma_algo_engine_pop_working_order(engine, &tracked->order_id);

    // === Step 5 ===
    // Dispatch the algo callback.
    if (callback) callback(algo->DMA, algo, tracked, algo->callbacks.user_data);

    // === Step 6 ===
    // Update the hosting position from the same delta: its ticker cache,
    // active set and strategy callbacks.
    dma_position* position        = NULL;
    char*         position_stored = NULL;

    if (c_bytemap_ex_get_ptr(engine->position, (const char*) algo->algo_id, sizeof(uuid_t), &position_stored, NULL) == BYTEMAP_OK) position = *(dma_position**) position_stored;

    if (position) {
        dma_position_ticker defaults;
        memset(&defaults, 0, sizeof(defaults));

        char* ticker_stored = NULL;

        if (c_bytemap_ex_set_default(position->tickers, algo->ticker, strlen(algo->ticker), (const char*) &defaults, sizeof(defaults), 0, &ticker_stored, NULL) == BYTEMAP_OK) {
            dma_position_ticker* cache = (dma_position_ticker*) ticker_stored;

            if (working_delta != 0.) {
                int sign = c_md_side_sign(algo->side);

                if (sign > 0) cache->working_long += working_delta;
                else if (sign < 0) cache->working_short += working_delta;
            }
        }

        if (c_dma_algo_status_is_active(algo->status)) c_bytemap_ex_set(position->working_algos, (const char*) algo->algo_id, sizeof(uuid_t), (const char*) &algo, sizeof(dma_algo*), 0, NULL);
        else c_bytemap_ex_pop(position->working_algos, (const char*) algo->algo_id, sizeof(uuid_t), 0, NULL, NULL);

        switch (tracked->order_state) {
            case STATE_REJECTED:
            case STATE_INVALID:
                c_dma_position_invoke_callbacks(position, on_rejected, algo, tracked, NULL);
                break;
            case STATE_CANCELED:
                c_dma_position_invoke_callbacks(position, on_canceled, algo, tracked, NULL);
                break;
            case STATE_PLACED:
                c_dma_position_invoke_callbacks(position, on_listed, algo, tracked, NULL);
                break;
            default:
                break;
        }
    }

    return DMA_ALGO_OK;
}

/**
 * @brief Apply a stored order's already-written state; runs when the main entry routes.
 *
 * The access writes the state in place before calling the main entry. The
 * pre-state is gone and the access's own operation guard already classified
 * it, so this entry runs no transition guard: the callback, the one
 * working-volume delta, the working-index departure and the position mirror
 * follow the state now in the stored order.
 *
 * @param engine The engine (NULL-safe).
 * @param ctx The tracked order context (NULL-safe).
 * @return DMA_ALGO_OK, or DMA_ALGO_ERR_INVALID_INPUT.
 */
static inline int c_dma_algo_engine_on_order_inplace(dma_algo_engine* engine, dma_order_ctx* ctx) {
    if (!engine || !ctx) return DMA_ALGO_ERR_INVALID_INPUT;

    dma_algo*             algo    = ctx->algo;
    md_trade_instruction* tracked = &ctx->order;

    // === Step 1 ===
    // Classify the state and its callback.

    dma_algo_order_func   callback      = NULL;
    uint8_t               terminal_flag = 0;

    switch (tracked->order_state) {
        case STATE_REJECTED:
            callback      = algo->callbacks.on_rejected;
            terminal_flag = DMA_ORDER_REJECTED;
            break;
        case STATE_INVALID:
            callback      = algo->callbacks.on_rejected;
            terminal_flag = DMA_ORDER_ERROR;
            break;
        case STATE_CANCELED:
            callback      = algo->callbacks.on_canceled;
            terminal_flag = DMA_ORDER_CANCELED;
            break;
        case STATE_PLACED:
            callback = algo->callbacks.on_listed;
            break;
        default:
            break;
    }

    // === Step 2 ===
    // Apply the one working-volume delta to the context bits and the algo.
    // Any working state - pending included - opens the remaining volume once,
    // whatever state the exchange reports first; a stop bit closes it once, and a
    // stop recorded before the volume enters still blocks a later reopen.

    bool   working_state = c_md_state_working(tracked->order_state) || tracked->order_state == STATE_PENDING;
    bool   stopped       = ctx->flags > DMA_ORDER_LISTED;
    double working_delta = 0.;

    if (working_state) {
        if (!(ctx->flags & DMA_ORDER_LISTED) && !stopped) {
            ctx->flags |= DMA_ORDER_LISTED;
            working_delta = tracked->volume - tracked->filled_volume;
        }
    }
    else if (terminal_flag && !stopped) {
        ctx->flags |= terminal_flag;
        if (ctx->flags & DMA_ORDER_LISTED) working_delta = -(tracked->volume - tracked->filled_volume);
    }

    algo->working_volume += working_delta;

    // === Step 3 ===
    // Leave the working indexes when the order is done.
    if (c_md_state_done(tracked->order_state)) c_dma_algo_engine_pop_working_order(engine, &tracked->order_id);

    // === Step 4 ===
    // Dispatch the algo callback.
    if (callback) callback(algo->DMA, algo, tracked, algo->callbacks.user_data);

    // === Step 5 ===
    // Update the hosting position from the same delta: its ticker cache,
    // active set and strategy callbacks.
    dma_position* position        = NULL;
    char*         position_stored = NULL;

    if (c_bytemap_ex_get_ptr(engine->position, (const char*) algo->algo_id, sizeof(uuid_t), &position_stored, NULL) == BYTEMAP_OK) position = *(dma_position**) position_stored;

    if (position) {
        dma_position_ticker defaults;
        memset(&defaults, 0, sizeof(defaults));

        char* ticker_stored = NULL;

        if (c_bytemap_ex_set_default(position->tickers, algo->ticker, strlen(algo->ticker), (const char*) &defaults, sizeof(defaults), 0, &ticker_stored, NULL) == BYTEMAP_OK) {
            dma_position_ticker* cache = (dma_position_ticker*) ticker_stored;

            if (working_delta != 0.) {
                int sign = c_md_side_sign(algo->side);

                if (sign > 0) cache->working_long += working_delta;
                else if (sign < 0) cache->working_short += working_delta;
            }
        }

        if (c_dma_algo_status_is_active(algo->status)) c_bytemap_ex_set(position->working_algos, (const char*) algo->algo_id, sizeof(uuid_t), (const char*) &algo, sizeof(dma_algo*), 0, NULL);
        else c_bytemap_ex_pop(position->working_algos, (const char*) algo->algo_id, sizeof(uuid_t), 0, NULL, NULL);

        switch (tracked->order_state) {
            case STATE_REJECTED:
            case STATE_INVALID:
                c_dma_position_invoke_callbacks(position, on_rejected, algo, tracked, NULL);
                break;
            case STATE_CANCELED:
                c_dma_position_invoke_callbacks(position, on_canceled, algo, tracked, NULL);
                break;
            case STATE_PLACED:
                c_dma_position_invoke_callbacks(position, on_listed, algo, tracked, NULL);
                break;
            default:
                break;
        }
    }

    return DMA_ALGO_OK;
}

/**
 * @brief Handle an exchange report for the order's algo; runs once per report.
 *
 * The report is classified by its side offset, never by its volume: a side
 * carrying OFFSET_CANCEL is a cancel report (its volume, possibly positive, is
 * the canceled amount) and any other side is a fill. The trade id is the gate:
 * a repeated report returns DMA_ALGO_OK before any effect, and a new report is
 * copied into the order context before it is applied.
 *
 * A fill books the volume, notional and fee, releases the filled volume from
 * the working set while the order is live, dispatches on_filled and mirrors the
 * fill into the hosting position. It writes no order state: the state column
 * follows the exchange's order-state events.
 *
 * A cancel report completes the order from the live states, releases the
 * remaining working volume once, leaves the working indexes and dispatches
 * on_canceled.
 *
 * @param engine The engine (NULL-safe).
 * @param trade The report (NULL-safe).
 * @return DMA_ALGO_OK, DMA_ALGO_ERR_INVALID_INPUT, DMA_ALGO_ERR_OOM, or
 *         DMA_ALGO_ERR_NOT_FOUND.
 */
static inline int c_dma_algo_engine_on_trade(dma_algo_engine* engine, const md_trade_report* trade) {
    if (!engine || !trade) return DMA_ALGO_ERR_INVALID_INPUT;

    // === Step 0: Resolve the order from the all-orders map, in one probe. ===
    // The map holds every context, live or done, so a late report resolves
    // without a second lookup.
    char* stored = NULL;

    if (c_bytemap_ex_get_ptr(engine->orders, (const char*) &trade->order_id, sizeof(long_md_id), &stored, NULL) != BYTEMAP_OK) return DMA_ALGO_ERR_NOT_FOUND;

    dma_order_ctx*        ctx   = *(dma_order_ctx**) stored;
    dma_algo*             algo  = ctx->algo;
    md_trade_instruction* order = &ctx->order;

    bool                  is_cancel = c_md_side_offset(trade->side) == OFFSET_CANCEL;

    // === Step 1: Classify the cancel report against the stored state. ===
    // A repeated confirmation is a no-op; a banned pre-state refuses and leaves
    // the order unchanged.
    if (is_cancel) {
        md_order_state state = order->order_state;

        if (state == STATE_CANCELED) return DMA_ALGO_OK;

        if (state != STATE_CANCELING && state != STATE_SENT && state != STATE_PLACED && state != STATE_PARTFILLED) {
            c_dma_refuse(algo->DMA, order, "cancel confirmation");
            return DMA_ALGO_ERR_INVALID_INPUT;
        }
    }

    // === Step 2: The trade-id gate. ===
    // A repeated report is a no-op; a new report is copied into the context
    // before it is applied.
    char* report_stored = NULL;

    if (c_bytemap_ex_get_ptr(ctx->reports, (const char*) &trade->trade_id, sizeof(long_md_id), &report_stored, NULL) == BYTEMAP_OK) return DMA_ALGO_OK;

    allocator_protocol* allocator = c_ap_protocol_from_ptr(ctx->reports);
    md_trade_report*    owned     = (md_trade_report*) c_ap_alloc(sizeof(md_trade_report), allocator);

    if (!owned) return DMA_ALGO_ERR_OOM;

    memcpy(owned, trade, sizeof(md_trade_report));

    if (c_bytemap_ex_set(ctx->reports, (const char*) &trade->trade_id, sizeof(long_md_id), (const char*) &owned, sizeof(md_trade_report*), 0, NULL) != BYTEMAP_OK) {
        c_ap_free(owned);
        return DMA_ALGO_ERR_OOM;
    }

    dma_position* position        = NULL;
    char*         position_stored = NULL;

    if (c_bytemap_ex_get_ptr(engine->position, (const char*) algo->algo_id, sizeof(uuid_t), &position_stored, NULL) == BYTEMAP_OK) position = *(dma_position**) position_stored;

    // === Step 3: Cancel report ===
    if (is_cancel) {
        order->order_state = STATE_CANCELED;
        order->ts_canceled = trade->meta_info.dt.unix_ts;

        if (ctx->flags > DMA_ORDER_LISTED) return DMA_ALGO_OK;

        ctx->flags |= DMA_ORDER_CANCELED;

        double working_delta = -(order->volume - order->filled_volume);

        algo->working_volume += working_delta;

        c_dma_algo_engine_pop_working_order(engine, &order->order_id);

        if (algo->callbacks.on_canceled) algo->callbacks.on_canceled(algo->DMA, algo, order, algo->callbacks.user_data);

        if (position) {
            dma_position_ticker defaults;
            memset(&defaults, 0, sizeof(defaults));

            char* ticker_stored = NULL;

            if (c_bytemap_ex_set_default(position->tickers, algo->ticker, strlen(algo->ticker), (const char*) &defaults, sizeof(defaults), 0, &ticker_stored, NULL) == BYTEMAP_OK) {
                dma_position_ticker* cache = (dma_position_ticker*) ticker_stored;
                int                  sign  = c_md_side_sign(algo->side);

                if (sign > 0) cache->working_long += working_delta;
                else if (sign < 0) cache->working_short += working_delta;
            }

            if (c_dma_algo_status_is_active(algo->status)) c_bytemap_ex_set(position->working_algos, (const char*) algo->algo_id, sizeof(uuid_t), (const char*) &algo, sizeof(dma_algo*), 0, NULL);
            else c_bytemap_ex_pop(position->working_algos, (const char*) algo->algo_id, sizeof(uuid_t), 0, NULL, NULL);

            c_dma_position_invoke_callbacks(position, on_canceled, algo, order, NULL);
        }

        return DMA_ALGO_OK;
    }

    // === Step 4: Fill ===
    if (isnan(order->multiplier)) order->multiplier = trade->multiplier;

    order->filled_volume += trade->volume;
    order->filled_notional += fabs(trade->notional);
    order->fee += trade->fee;

    double working_delta = ctx->flags == DMA_ORDER_LISTED ? -trade->volume : 0.;

    algo->working_volume += working_delta;
    algo->filled_volume += trade->volume;
    algo->filled_notional += fabs(trade->notional);
    algo->fee += trade->fee;
    algo->exposure_volume += trade->volume * c_md_side_sign(trade->side);
    algo->cash_flow = -algo->filled_notional * c_md_side_sign(algo->side);

    if (algo->callbacks.on_filled) algo->callbacks.on_filled(algo->DMA, algo, trade, algo->callbacks.user_data);

    if (position) {
        dma_position_ticker defaults;
        memset(&defaults, 0, sizeof(defaults));

        char* ticker_stored = NULL;

        if (c_bytemap_ex_set_default(position->tickers, algo->ticker, strlen(algo->ticker), (const char*) &defaults, sizeof(defaults), 0, &ticker_stored, NULL) == BYTEMAP_OK) {
            dma_position_ticker* cache = (dma_position_ticker*) ticker_stored;

            if (working_delta != 0.) {
                int sign = c_md_side_sign(algo->side);

                if (sign > 0) cache->working_long += working_delta;
                else if (sign < 0) cache->working_short += working_delta;
            }

            double exposure_delta = trade->volume * c_md_side_sign(trade->side);
            double notional_delta = fabs(trade->notional);
            double cash_delta     = -notional_delta * c_md_side_sign(algo->side);

            cache->exposure_volume += exposure_delta;
            cache->filled_notional += notional_delta;
            cache->cash_flow += cash_delta;
        }

        if (c_dma_algo_status_is_active(algo->status)) c_bytemap_ex_set(position->working_algos, (const char*) algo->algo_id, sizeof(uuid_t), (const char*) &algo, sizeof(dma_algo*), 0, NULL);
        else c_bytemap_ex_pop(position->working_algos, (const char*) algo->algo_id, sizeof(uuid_t), 0, NULL, NULL);

        c_dma_position_invoke_callbacks(position, on_filled, algo, order, trade);
    }

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
    if (!engine || !market_data) return;

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

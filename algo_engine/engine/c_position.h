#ifndef C_POSITION_H
#define C_POSITION_H

#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <cbase/allocator_protocol/c_allocator_protocol.h>
#include <cbase/bytemap/c_bytemap.h>
#include <cbase/bytemap/xxh3.h>

#include "algo_engine/base/c_market_data/c_market_data.h"

// The algo and access records are defined by headers that include this one;
// the position only holds pointers to them.
typedef struct dma             dma;
typedef struct dma_algo        dma_algo;
typedef struct dma_algo_engine dma_algo_engine;
typedef struct dma_position    dma_position;

// The 16-byte identity shared with the algo records (c_algo_template.h).
typedef unsigned char          uuid_t[16];

// ========== Constants ==========

#ifndef DMA_POSITION_DEFAULT_ALGO
#define DMA_POSITION_DEFAULT_ALGO "passive"
#endif

// ========== Enums ==========

/** @enum position_ret_code
 *  @brief 0 is success, negatives are errors. */
typedef enum position_ret_code {
    POSITION_OK                = 0,
    POSITION_ERR_INVALID_INPUT = -1,
    POSITION_ERR_OOM           = -2,
    POSITION_ERR_NOT_FOUND     = -3,
} position_ret_code;

/** @enum dma_callback_event
 *  @brief The strategy-facing position events. */
typedef enum dma_callback_event {
    on_canceled = 1,
    on_rejected = 2,
    on_listed   = 3,
    on_filled   = 4,
} dma_callback_event;

// ========== Structs ==========

/**
 * @brief Invoked when a hosted algo's order changes state.
 * @param position The position (borrowed).
 * @param algo The algo that owns the order (borrowed).
 * @param order The order (borrowed).
 * @param user_data The value registered on the callback ctx.
 * @return 0 to continue the dispatch, nonzero to end it.
 */
typedef int (*dma_position_order_func)(dma_position* position, const dma_algo* algo, const md_trade_instruction* order, void* user_data);

/**
 * @brief Invoked when a fill reaches a hosted algo.
 * @param position The position (borrowed).
 * @param algo The algo that owns the order (borrowed).
 * @param order The filled order (borrowed).
 * @param trade The fill (borrowed).
 * @param user_data The value registered on the callback ctx.
 * @return 0 to continue the dispatch, nonzero to end it.
 */
typedef int (*dma_position_trade_func)(dma_position* position, const dma_algo* algo, const md_trade_instruction* order, const md_trade_report* trade, void* user_data);

/**
 * @struct dma_position_callback_ctx
 * @brief One registered strategy callback; a calloc'ed linked-list node.
 *
 * Exactly one slot is set, the one named by the event given at registration.
 */
typedef struct dma_position_callback_ctx {
    dma_position_order_func           on_canceled;  // NOT owned - canceled-order callback.
    dma_position_order_func           on_rejected;  // NOT owned - rejected-order callback.
    dma_position_order_func           on_listed;    // NOT owned - listed-order callback.
    dma_position_trade_func           on_filled;    // NOT owned - fill callback.
    void*                             user_data;    // NOT owned - passed to the callback.
    uintptr_t                         id;           // The id used to unregister.
    struct dma_position_callback_ctx* next;
} dma_position_callback_ctx;

/**
 * @struct dma_position_ticker
 * @brief The running aggregates of one ticker.
 *
 * Every event updates this record in place; a query reads it without walking
 * the hosted algos. `multiplier` is the ticker's contract multiplier, taken
 * from the first hosted record or from an exposure add.
 */
typedef struct dma_position_ticker {
    double exposure_volume;  // Signed net exposure volume.
    double working_long;     // Long-side working volume.
    double working_short;    // Short-side working volume.
    double filled_notional;  // Accumulated filled notional.
    double cash_flow;        // Accumulated signed cash flow.
    double multiplier;       // Contract multiplier.
} dma_position_ticker;

/**
 * @struct dma_position
 * @brief One strategy's view of the algo engine.
 *
 * `algos` indexes the hosted subset by algo id; `working_algos` holds only the
 * active records, so a stop sweep never touches a completed one. `tickers`
 * caches every aggregate a strategy reads, updated in place by the engine's
 * event entries: the position never manages an algo's status, and no read
 * refreshes it. `callbacks` are the strategy-facing notifications.
 */
typedef struct dma_position {
    dma*                       dma;            // NOT owned - the access this position trades through.
    dma_algo_engine*           algo_engine;    // NOT owned - the engine that owns the hosted records; outlives this position.
    uuid_t                     position_id;    // 16-byte identity generated at init.
    char*                      default_algo;   // OWNED - default algo name.
    bytemap*                   algos;          // OWNED - algo id -> borrowed dma_algo*.
    bytemap*                   working_algos;  // OWNED - algo id -> borrowed dma_algo* (the active subset).
    bytemap*                   tickers;        // OWNED - ticker -> dma_position_ticker aggregates.
    dma_position_callback_ctx* callbacks;      // OWNED - calloc'ed strategy callback nodes.
} dma_position;

// ========== Forward Declarations ==========

// Lifecycle
static inline dma_position* c_dma_position_new(dma* access, dma_algo_engine* engine, const char* default_algo, allocator_protocol* allocator);
static inline int           c_dma_position_init(dma_position* position, dma* access, dma_algo_engine* engine, const char* default_algo, allocator_protocol* allocator);
static inline void          c_dma_position_dealloc(dma_position* position);
static inline void          c_dma_position_free(dma_position* position);
static inline int           c_dma_position_clear(dma_position* position);

// Identity
static inline void          c_dma_position_gen_uid(dma_position* position);

// Algo registry
static inline int           c_dma_position_add_algo(dma_position* position, dma_algo* algo);
static inline int           c_dma_position_add_algo_by_id(dma_position* position, const unsigned char* algo_id);
static inline int           c_dma_position_pop_algo(dma_position* position, const unsigned char* algo_id);
static inline dma_algo*     c_dma_position_get_algo(const dma_position* position, const unsigned char* algo_id);

// Queries
static inline size_t        c_dma_position_n_algos(const dma_position* position);
static inline size_t        c_dma_position_n_working_algos(const dma_position* position);
static inline double        c_dma_position_get_exposure(const dma_position* position, const char* ticker);
static inline int           c_dma_position_get_working(const dma_position* position, const char* ticker, double* out_long, double* out_short);
static inline double        c_dma_position_working_net(const dma_position* position, const char* ticker);
static inline double        c_dma_position_notional(const dma_position* position, const char* ticker);
static inline double        c_dma_position_pnl(const dma_position* position, const char* ticker);

// Exposure
static inline int           c_dma_position_add_exposure(dma_position* position, const char* ticker, double volume, double notional, double multiplier, md_side side);

// Callbacks (only for hosted dma_algo)
static inline int           c_dma_position_register_callback(dma_position* position, dma_callback_event event, void* callback, void* user_data, uintptr_t* out_id);
static inline int           c_dma_position_unregister_callback(dma_position* position, uintptr_t callback_id);

// Trading (c_dma_new_algo is defined in c_dma.h)
static inline dma_algo*     c_dma_new_algo(dma* DMA, uint32_t algo_reg_id, const char* ticker, md_side side, double target_volume, const void* params, allocator_protocol* allocator);
static inline int           c_dma_position_fully_stop(dma_position* position);
static inline int           c_dma_position_fully_unwind(dma_position* position);

// Hot path - runs once per routed event
static inline int           c_dma_position_invoke_callbacks(dma_position* position, dma_callback_event event, const dma_algo* algo, const md_trade_instruction* order, const md_trade_report* trade);

#include "algo_engine/engine/c_dma_algo.h"

// ========== Public APIs (Lifecycle) ==========

/**
 * @brief Allocate a position and register it in the access's ownership map.
 *
 * The access owns the block and frees it at dealloc; a position id already in
 * the map is refused.
 *
 * @param access The access this position trades through (borrowed; its positions map must exist).
 * @param engine The engine that owns the hosted records (borrowed).
 * @param default_algo The default algo name (copied; NULL selects DMA_POSITION_DEFAULT_ALGO).
 * @param allocator Allocator for the block and its children.
 * @return The new position, or NULL on allocation or registration failure.
 */
static inline dma_position* c_dma_position_new(dma* access, dma_algo_engine* engine, const char* default_algo, allocator_protocol* allocator) {
    if (!access || !access->positions) return NULL;

    dma_position* position = (dma_position*) c_ap_alloc(sizeof(dma_position), allocator);
    if (!position) return NULL;

    if (c_dma_position_init(position, access, engine, default_algo, allocator) != POSITION_OK) {
        c_dma_position_dealloc(position);
        c_ap_free(position);
        return NULL;
    }

    dma_position* empty  = NULL;
    char*         stored = NULL;

    if (c_bytemap_ex_set_default(access->positions, (const char*) position->position_id, sizeof(uuid_t), (const char*) &empty, sizeof(dma_position*), 0, &stored, NULL) != BYTEMAP_OK || *(dma_position**) stored) {
        c_dma_position_dealloc(position);
        c_ap_free(position);
        return NULL;
    }

    *(dma_position**) stored = position;
    return position;
}

/**
 * @brief Initialize a caller-allocated, zeroed position.
 *
 * The identity is generated from the block address and the default algo name.
 *
 * @param position The zeroed position to initialize.
 * @param access The access this position trades through (borrowed).
 * @param engine The engine that owns the hosted records (borrowed).
 * @param default_algo The default algo name (copied; NULL selects DMA_POSITION_DEFAULT_ALGO).
 * @param allocator Allocator for the children; derived from position when NULL.
 * @return POSITION_OK, POSITION_ERR_INVALID_INPUT, or POSITION_ERR_OOM.
 */
static inline int c_dma_position_init(dma_position* position, dma* access, dma_algo_engine* engine, const char* default_algo, allocator_protocol* allocator) {
    if (!position) return POSITION_ERR_INVALID_INPUT;
    if (!allocator) allocator = c_ap_protocol_from_ptr(position);

    position->dma         = access;
    position->algo_engine = engine;

    if (!default_algo) default_algo = DMA_POSITION_DEFAULT_ALGO;

    position->default_algo = c_ap_strdup(default_algo, allocator);
    if (!position->default_algo) return POSITION_ERR_OOM;
    c_ap_acquire_ownership(position->default_algo, position);

    position->algos = c_bytemap_ex_new(0, sizeof(dma_algo*), allocator);
    if (!position->algos) return POSITION_ERR_OOM;

    position->working_algos = c_bytemap_ex_new(0, sizeof(dma_algo*), allocator);
    if (!position->working_algos) goto oom;

    position->tickers = c_bytemap_ex_new(0, sizeof(dma_position_ticker), allocator);
    if (!position->tickers) goto oom;

    c_dma_position_gen_uid(position);
    return POSITION_OK;

oom:
    c_dma_position_dealloc(position);
    return POSITION_ERR_OOM;
}

/**
 * @brief Tear down a caller-allocated position, leaving it zeroed.
 *
 * The hosted records are borrowed; the engine owns them and frees them. The
 * engine must outlive the position because the teardown drops the position's
 * routing entries from it.
 *
 * @param position The position (NULL-safe).
 */
static inline void c_dma_position_dealloc(dma_position* position) {
    if (!position) return;

    c_dma_position_clear(position);

    dma_position_callback_ctx* callback = position->callbacks;

    while (callback) {
        dma_position_callback_ctx* next = callback->next;
        free(callback);
        callback = next;
    }

    position->callbacks = NULL;

    if (position->default_algo) {
        c_ap_free((void*) position->default_algo);
        position->default_algo = NULL;
    }

    if (position->algos) {
        c_bytemap_ex_free(position->algos);
        position->algos = NULL;
    }

    if (position->working_algos) {
        c_bytemap_ex_free(position->working_algos);
        position->working_algos = NULL;
    }

    if (position->tickers) {
        c_bytemap_ex_free(position->tickers);
        position->tickers = NULL;
    }

    memset(position, 0, sizeof(dma_position));
}

/**
 * @brief Unregister a position from its access and free it.
 * @param position The position (NULL-safe).
 */
static inline void c_dma_position_free(dma_position* position) {
    if (!position) return;

    if (position->dma && position->dma->positions) {
        char* stored = NULL;

        if (c_bytemap_ex_get_ptr(position->dma->positions, (const char*) position->position_id, sizeof(uuid_t), &stored, NULL) == BYTEMAP_OK && *(dma_position**) stored == position) {
            c_bytemap_ex_pop(position->dma->positions, (const char*) position->position_id, sizeof(uuid_t), 0, NULL, NULL);
        }
    }

    c_dma_position_dealloc(position);
    c_ap_free(position);
}

/**
 * @brief Drop every hosted algo reference and aggregate; the engine keeps the records.
 *
 * The engine routes events by algo id, so the routing entries of the hosted
 * records leave with the references.
 *
 * @param position The position (NULL-safe).
 * @return POSITION_OK, or POSITION_ERR_INVALID_INPUT.
 */
static inline int c_dma_position_clear(dma_position* position) {
    if (!position || !position->algos) return POSITION_ERR_INVALID_INPUT;

    if (position->algo_engine && position->algo_engine->position) {
        bytemap_entry* entry = position->algos->first;

        while (entry) {
            bytemap_entry* next   = entry->next;
            char*          stored = NULL;

            if (c_bytemap_ex_get_ptr(position->algo_engine->position, entry->key, sizeof(uuid_t), &stored, NULL) == BYTEMAP_OK && *(dma_position**) stored == position) {
                c_bytemap_ex_pop(position->algo_engine->position, entry->key, sizeof(uuid_t), 0, NULL, NULL);
            }

            entry = next;
        }
    }

    c_bytemap_ex_clear(position->algos);

    if (position->working_algos) c_bytemap_ex_clear(position->working_algos);
    if (position->tickers) c_bytemap_ex_clear(position->tickers);

    return POSITION_OK;
}

// ========== Public APIs (Identity) ==========

/**
 * @brief Generate the 16-byte position identity from the block address and the default algo.
 *
 * The result is laid out as a UUID (version 4 / RFC variant).
 *
 * @param position The position whose id is written (NULL-safe).
 */
static inline void c_dma_position_gen_uid(dma_position* position) {
    if (!position) return;

    XXH3_state_t state;
    XXH3_128bits_reset(&state);

    uintptr_t address = (uintptr_t) position;
    XXH3_128bits_update(&state, &address, sizeof(address));
    if (position->default_algo) XXH3_128bits_update(&state, position->default_algo, strlen(position->default_algo));

    XXH128_hash_t hash = XXH3_128bits_digest(&state);
    memcpy(position->position_id, &hash.low64, 8);
    memcpy(position->position_id + 8, &hash.high64, 8);

    position->position_id[6] = (position->position_id[6] & 0x0F) | 0x40;
    position->position_id[8] = (position->position_id[8] & 0x3F) | 0x80;
}

// ========== Public APIs (Algo Registry) ==========

/**
 * @brief Host an engine-owned algo and route its events to this position.
 *
 * An algo is hosted by one position; an unknown record, a duplicate, or a
 * record already hosted elsewhere is refused. The record's current aggregates
 * enter the ticker cache in the same pass.
 *
 * @param position The position (NULL-safe).
 * @param algo The engine-owned record (NULL is refused).
 * @return POSITION_OK, POSITION_ERR_INVALID_INPUT, POSITION_ERR_NOT_FOUND, or POSITION_ERR_OOM.
 */
static inline int c_dma_position_add_algo(dma_position* position, dma_algo* algo) {
    if (!position || !position->algos || !position->working_algos || !position->tickers || !algo || !algo->ticker) return POSITION_ERR_INVALID_INPUT;

    char* stored = NULL;

    if (c_bytemap_ex_get_ptr(position->algos, (const char*) algo->algo_id, sizeof(uuid_t), &stored, NULL) == BYTEMAP_OK) return POSITION_ERR_INVALID_INPUT;

    if (position->algo_engine) {
        if (!position->algo_engine->algo) return POSITION_ERR_INVALID_INPUT;
        if (c_bytemap_ex_get_ptr(position->algo_engine->algo, (const char*) algo->algo_id, sizeof(uuid_t), &stored, NULL) != BYTEMAP_OK) return POSITION_ERR_NOT_FOUND;

        // A record has one host: a second position would steal its events.
        if (position->algo_engine->position && c_bytemap_ex_get_ptr(position->algo_engine->position, (const char*) algo->algo_id, sizeof(uuid_t), &stored, NULL) == BYTEMAP_OK) return POSITION_ERR_INVALID_INPUT;
    }

    if (c_bytemap_ex_set(position->algos, (const char*) algo->algo_id, sizeof(uuid_t), (const char*) &algo, sizeof(dma_algo*), 0, NULL) != BYTEMAP_OK) return POSITION_ERR_OOM;

    if (position->algo_engine && position->algo_engine->position) {
        dma_position* host = position;

        if (c_bytemap_ex_set(position->algo_engine->position, (const char*) algo->algo_id, sizeof(uuid_t), (const char*) &host, sizeof(dma_position*), 0, NULL) != BYTEMAP_OK) {
            c_bytemap_ex_pop(position->algos, (const char*) algo->algo_id, sizeof(uuid_t), 0, NULL, NULL);
            return POSITION_ERR_OOM;
        }
    }

    dma_position_ticker defaults;
    memset(&defaults, 0, sizeof(defaults));

    char* ticker_stored = NULL;
    bool  new_ticker    = c_bytemap_ex_get_ptr(position->tickers, algo->ticker, strlen(algo->ticker), &ticker_stored, NULL) != BYTEMAP_OK;

    if (new_ticker && c_bytemap_ex_set_default(position->tickers, algo->ticker, strlen(algo->ticker), (const char*) &defaults, sizeof(defaults), 0, &ticker_stored, NULL) != BYTEMAP_OK) {
        if (position->algo_engine && position->algo_engine->position) c_bytemap_ex_pop(position->algo_engine->position, (const char*) algo->algo_id, sizeof(uuid_t), 0, NULL, NULL);
        c_bytemap_ex_pop(position->algos, (const char*) algo->algo_id, sizeof(uuid_t), 0, NULL, NULL);
        return POSITION_ERR_OOM;
    }

    dma_position_ticker* ticker = (dma_position_ticker*) ticker_stored;

    if (new_ticker) ticker->multiplier = algo->multiplier;

    ticker->exposure_volume += algo->exposure_volume;
    ticker->filled_notional += algo->filled_notional;
    ticker->cash_flow += algo->cash_flow;

    int sign = c_md_side_sign(algo->side);

    if (sign > 0) ticker->working_long += algo->working_volume;
    else if (sign < 0) ticker->working_short += algo->working_volume;

    if (c_dma_algo_status_is_active(algo->status) && c_bytemap_ex_set(position->working_algos, (const char*) algo->algo_id, sizeof(uuid_t), (const char*) &algo, sizeof(dma_algo*), 0, NULL) != BYTEMAP_OK) {
        if (position->algo_engine && position->algo_engine->position) c_bytemap_ex_pop(position->algo_engine->position, (const char*) algo->algo_id, sizeof(uuid_t), 0, NULL, NULL);
        c_bytemap_ex_pop(position->algos, (const char*) algo->algo_id, sizeof(uuid_t), 0, NULL, NULL);
        return POSITION_ERR_OOM;
    }

    return POSITION_OK;
}

/**
 * @brief Host an engine-owned algo resolved by id.
 * @param position The position (NULL-safe).
 * @param algo_id The 16-byte algo id (NULL is refused).
 * @return POSITION_OK, POSITION_ERR_INVALID_INPUT, POSITION_ERR_NOT_FOUND, or POSITION_ERR_OOM.
 */
static inline int c_dma_position_add_algo_by_id(dma_position* position, const unsigned char* algo_id) {
    if (!position || !position->algo_engine || !position->algo_engine->algo || !algo_id) return POSITION_ERR_INVALID_INPUT;

    char* stored = NULL;

    if (c_bytemap_ex_get_ptr(position->algo_engine->algo, (const char*) algo_id, sizeof(uuid_t), &stored, NULL) != BYTEMAP_OK) return POSITION_ERR_NOT_FOUND;

    return c_dma_position_add_algo(position, *(dma_algo**) stored);
}

/**
 * @brief Drop a hosted algo reference and its routing entry; the engine keeps the record.
 *
 * The record's own aggregates leave the ticker cache in the same pass.
 *
 * @param position The position (NULL-safe).
 * @param algo_id The 16-byte algo id (non-NULL).
 * @return POSITION_OK, POSITION_ERR_INVALID_INPUT, or POSITION_ERR_NOT_FOUND.
 */
static inline int c_dma_position_pop_algo(dma_position* position, const unsigned char* algo_id) {
    if (!position || !position->algos || !algo_id) return POSITION_ERR_INVALID_INPUT;

    char* stored = NULL;

    if (c_bytemap_ex_get_ptr(position->algos, (const char*) algo_id, sizeof(uuid_t), &stored, NULL) != BYTEMAP_OK) return POSITION_ERR_NOT_FOUND;

    const dma_algo* algo = *(const dma_algo**) stored;

    if (algo && algo->ticker && position->tickers) {
        char* ticker_stored = NULL;

        if (c_bytemap_ex_get_ptr(position->tickers, algo->ticker, strlen(algo->ticker), &ticker_stored, NULL) == BYTEMAP_OK) {
            dma_position_ticker* ticker = (dma_position_ticker*) ticker_stored;

            ticker->exposure_volume -= algo->exposure_volume;
            ticker->filled_notional -= algo->filled_notional;
            ticker->cash_flow -= algo->cash_flow;

            int sign = c_md_side_sign(algo->side);

            if (sign > 0) ticker->working_long -= algo->working_volume;
            else if (sign < 0) ticker->working_short -= algo->working_volume;

            // The ticker has no aggregate once its last hosted record leaves.
            bool           hosted = false;
            bytemap_entry* entry  = position->algos->first;

            while (entry) {
                const dma_algo* other = *(const dma_algo**) entry->value;

                if (other && other != algo && other->ticker && strcmp(other->ticker, algo->ticker) == 0) {
                    hosted = true;
                    break;
                }

                entry = entry->next;
            }

            if (!hosted) c_bytemap_ex_pop(position->tickers, algo->ticker, strlen(algo->ticker), 0, NULL, NULL);
        }
    }

    if (position->working_algos) c_bytemap_ex_pop(position->working_algos, (const char*) algo_id, sizeof(uuid_t), 0, NULL, NULL);

    if (position->algo_engine && position->algo_engine->position) {
        char* host_stored = NULL;

        if (c_bytemap_ex_get_ptr(position->algo_engine->position, (const char*) algo_id, sizeof(uuid_t), &host_stored, NULL) == BYTEMAP_OK && *(dma_position**) host_stored == position) {
            c_bytemap_ex_pop(position->algo_engine->position, (const char*) algo_id, sizeof(uuid_t), 0, NULL, NULL);
        }
    }

    c_bytemap_ex_pop(position->algos, (const char*) algo_id, sizeof(uuid_t), 0, NULL, NULL);
    return POSITION_OK;
}

/**
 * @brief Look up a hosted algo record by id.
 * @param position The position (NULL-safe).
 * @param algo_id The 16-byte algo id (non-NULL).
 * @return The borrowed record, or NULL.
 */
static inline dma_algo* c_dma_position_get_algo(const dma_position* position, const unsigned char* algo_id) {
    if (!position || !position->algos || !algo_id) return NULL;

    char* stored = NULL;

    if (c_bytemap_ex_get_ptr(position->algos, (const char*) algo_id, sizeof(uuid_t), &stored, NULL) != BYTEMAP_OK) return NULL;

    return *(dma_algo**) stored;
}

// ========== Public APIs (Queries) ==========

/**
 * @brief Number of hosted algo records.
 * @param position The position (NULL-safe).
 * @return The count, or 0.
 */
static inline size_t c_dma_position_n_algos(const dma_position* position) {
    return position && position->algos ? c_bytemap_ex_len(position->algos) : 0;
}

/**
 * @brief Number of active hosted algos; completed ones are not in the set.
 * @param position The position (NULL-safe).
 * @return The count, or 0.
 */
static inline size_t c_dma_position_n_working_algos(const dma_position* position) {
    return position && position->working_algos ? c_bytemap_ex_len(position->working_algos) : 0;
}

/**
 * @brief Cached signed net exposure volume of a ticker.
 * @param position The position (NULL-safe).
 * @param ticker The instrument (non-empty).
 * @return The exposure volume, or 0.
 */
static inline double c_dma_position_get_exposure(const dma_position* position, const char* ticker) {
    if (!position || !position->tickers || !ticker) return 0.;

    char* stored = NULL;

    if (c_bytemap_ex_get_ptr(position->tickers, ticker, strlen(ticker), &stored, NULL) != BYTEMAP_OK) return 0.;

    return ((const dma_position_ticker*) stored)->exposure_volume;
}

/**
 * @brief Cached working volume of a ticker on both sides.
 *
 * A missing ticker reports 0 on both sides with POSITION_OK.
 *
 * @param position The position (NULL-safe).
 * @param ticker The instrument (non-empty).
 * @param out_long Optional: receives the long working volume.
 * @param out_short Optional: receives the short working volume.
 * @return POSITION_OK, or POSITION_ERR_INVALID_INPUT.
 */
static inline int c_dma_position_get_working(const dma_position* position, const char* ticker, double* out_long, double* out_short) {
    if (out_long) *out_long = 0.;
    if (out_short) *out_short = 0.;
    if (!position || !position->tickers || !ticker) return POSITION_ERR_INVALID_INPUT;

    char* stored = NULL;

    if (c_bytemap_ex_get_ptr(position->tickers, ticker, strlen(ticker), &stored, NULL) != BYTEMAP_OK) return POSITION_OK;

    const dma_position_ticker* cache = (const dma_position_ticker*) stored;

    if (out_long) *out_long = cache->working_long;
    if (out_short) *out_short = cache->working_short;
    return POSITION_OK;
}

/**
 * @brief Cached signed net working volume of a ticker.
 * @param position The position (NULL-safe).
 * @param ticker The instrument (non-empty).
 * @return The net working volume, or 0.
 */
static inline double c_dma_position_working_net(const dma_position* position, const char* ticker) {
    double long_volume  = 0.;
    double short_volume = 0.;

    if (c_dma_position_get_working(position, ticker, &long_volume, &short_volume) != POSITION_OK) return 0.;

    return long_volume - short_volume;
}

/**
 * @brief Cached filled notional of a ticker.
 * @param position The position (NULL-safe).
 * @param ticker The instrument (non-empty).
 * @return The filled notional, or 0.
 */
static inline double c_dma_position_notional(const dma_position* position, const char* ticker) {
    if (!position || !position->tickers || !ticker) return 0.;

    char* stored = NULL;

    if (c_bytemap_ex_get_ptr(position->tickers, ticker, strlen(ticker), &stored, NULL) != BYTEMAP_OK) return 0.;

    return ((const dma_position_ticker*) stored)->filled_notional;
}

/**
 * @brief Cached PnL of a ticker.
 *
 * The market price comes from the position's access. An exposed ticker
 * without a price reports NaN; a flat ticker contributes its cash flow.
 *
 * @param position The position (NULL-safe).
 * @param ticker The instrument (non-empty).
 * @return The PnL, NaN when exposed without a price, or 0.
 */
static inline double c_dma_position_pnl(const dma_position* position, const char* ticker) {
    if (!position || !position->tickers || !ticker) return 0.;

    char* stored = NULL;

    if (c_bytemap_ex_get_ptr(position->tickers, ticker, strlen(ticker), &stored, NULL) != BYTEMAP_OK) return 0.;

    const dma_position_ticker* cache = (const dma_position_ticker*) stored;

    if (!cache->exposure_volume) return cache->cash_flow;

    double market_price = position->dma && position->dma->MDS ? c_mds_get_market_price(position->dma->MDS, ticker) : NAN;

    if (isnan(market_price)) return NAN;

    return market_price * cache->exposure_volume * cache->multiplier + cache->cash_flow;
}

// ========== Public APIs (Exposure) ==========

/**
 * @brief Add synthetic exposure to a ticker's cache.
 *
 * The volume signs the exposure by the side; the notional accrues to the
 * filled notional, and its absolute value leaves the cash flow on the side's
 * sign. The multiplier is stored on the ticker cache.
 *
 * @param position The position (NULL-safe).
 * @param ticker The instrument (NULL is refused).
 * @param volume The exposure volume.
 * @param notional The exposure notional.
 * @param multiplier The contract multiplier stored on the ticker cache.
 * @param side The trade side.
 * @return POSITION_OK, POSITION_ERR_INVALID_INPUT, or POSITION_ERR_OOM.
 */
static inline int c_dma_position_add_exposure(dma_position* position, const char* ticker, double volume, double notional, double multiplier, md_side side) {
    if (!position || !position->tickers || !ticker) return POSITION_ERR_INVALID_INPUT;

    dma_position_ticker defaults = {0};

    char*               stored = NULL;

    if (c_bytemap_ex_set_default(position->tickers, ticker, strlen(ticker), (const char*) &defaults, sizeof(defaults), 0, &stored, NULL) != BYTEMAP_OK) return POSITION_ERR_OOM;

    dma_position_ticker* cache = (dma_position_ticker*) stored;
    int                  sign  = c_md_side_sign(side);

    cache->multiplier = multiplier;
    cache->exposure_volume += volume * sign;
    cache->filled_notional += fabs(notional);
    cache->cash_flow -= fabs(notional) * sign;

    return POSITION_OK;
}

// ========== Public APIs (Callbacks) ==========

/**
 * @brief Register one strategy callback for an event.
 * @param position The position (NULL-safe).
 * @param event The event the callback answers (an unknown value is refused).
 * @param callback The callback (NULL is refused).
 * @param user_data Opaque value handed to the callback.
 * @param out_id Optional: receives the id used to unregister.
 * @return POSITION_OK, POSITION_ERR_INVALID_INPUT, or POSITION_ERR_OOM.
 */
static inline int c_dma_position_register_callback(dma_position* position, dma_callback_event event, void* callback, void* user_data, uintptr_t* out_id) {
    if (out_id) *out_id = 0;
    if (!position || !callback) return POSITION_ERR_INVALID_INPUT;

    if (event != on_canceled && event != on_rejected && event != on_listed && event != on_filled) return POSITION_ERR_INVALID_INPUT;

    dma_position_callback_ctx* node = (dma_position_callback_ctx*) calloc(1, sizeof(dma_position_callback_ctx));
    if (!node) return POSITION_ERR_OOM;

    switch (event) {
        case on_canceled:
            node->on_canceled = (dma_position_order_func) callback;
            break;
        case on_rejected:
            node->on_rejected = (dma_position_order_func) callback;
            break;
        case on_listed:
            node->on_listed = (dma_position_order_func) callback;
            break;
        case on_filled:
            node->on_filled = (dma_position_trade_func) callback;
            break;
        default:
            break;
    }

    node->user_data = user_data;
    node->id        = (uintptr_t) node;

    if (position->callbacks) {
        dma_position_callback_ctx* tail = position->callbacks;
        while (tail->next) tail = tail->next;
        tail->next = node;
    }
    else {
        position->callbacks = node;
    }

    if (out_id) *out_id = node->id;
    return POSITION_OK;
}

/**
 * @brief Unregister a strategy callback by the id registration returned.
 * @param position The position (NULL-safe).
 * @param callback_id The id from registration (0 is refused).
 * @return POSITION_OK, POSITION_ERR_INVALID_INPUT, or POSITION_ERR_NOT_FOUND.
 */
static inline int c_dma_position_unregister_callback(dma_position* position, uintptr_t callback_id) {
    if (!position || !callback_id) return POSITION_ERR_INVALID_INPUT;

    dma_position_callback_ctx* previous = NULL;
    dma_position_callback_ctx* current  = position->callbacks;

    while (current) {
        if (current->id == callback_id) {
            if (previous) previous->next = current->next;
            else position->callbacks = current->next;
            free(current);
            return POSITION_OK;
        }

        previous = current;
        current  = current->next;
    }

    return POSITION_ERR_NOT_FOUND;
}

// ========== Public APIs (Trading) ==========

/**
 * @brief Stop every active hosted algo.
 *
 * Only the working set is walked, so completed records cost nothing. Each
 * stopped algo cancels its live orders through its own stop hook.
 *
 * @param position The position (NULL-safe).
 * @return POSITION_OK, or POSITION_ERR_INVALID_INPUT.
 */
static inline int c_dma_position_fully_stop(dma_position* position) {
    if (!position || !position->working_algos) return POSITION_ERR_INVALID_INPUT;

    bytemap_entry* entry = position->working_algos->first;

    while (entry) {
        bytemap_entry* next = entry->next;
        dma_algo*      algo = *(dma_algo**) entry->value;

        if (algo && algo->status <= DMA_ALGO_STATUS_WORKING && algo->handlers.stop) {
            algo->handlers.stop(algo->DMA, algo, algo->handlers.user_data);
        }

        entry = next;
    }

    return POSITION_OK;
}

/**
 * @brief Open a closing algo for every exposed ticker through the unwind kind.
 *
 * The position must be an allocator block. The kind is
 * `DMA_POSITION_DEFAULT_ALGO` (passive by default). Each exposed ticker builds
 * one algo on the opposite side with the absolute exposure as its target,
 * hosts it, and launches it.
 *
 * @param position The position (NULL-safe; its access, engine and tickers must be bound).
 * @return POSITION_OK, POSITION_ERR_INVALID_INPUT, POSITION_ERR_NOT_FOUND, or POSITION_ERR_OOM.
 */
static inline int c_dma_position_fully_unwind(dma_position* position) {
    if (!position || !position->dma || !position->algo_engine || !position->tickers) return POSITION_ERR_INVALID_INPUT;

    dma_algo_registry* registry = position->algo_engine->registry;

    if (!registry || !registry->entries) return POSITION_ERR_INVALID_INPUT;

    char* stored = NULL;

    if (c_bytemap_ex_get_ptr(registry->entries, DMA_POSITION_DEFAULT_ALGO, strlen(DMA_POSITION_DEFAULT_ALGO), &stored, NULL) != BYTEMAP_OK) return POSITION_ERR_NOT_FOUND;

    dma_algo_ctx* registration = *(dma_algo_ctx**) stored;

    if (!registration || !registration->algo_new_fn) return POSITION_ERR_NOT_FOUND;

    size_t n_tickers = c_bytemap_ex_len(position->tickers);

    if (!n_tickers) return POSITION_OK;

    // Building an algo updates the ticker cache, so the exposed tickers are
    // snapshotted first: one block holds the pointer array and the key bytes.
    allocator_protocol* allocator = c_ap_protocol_from_ptr(position);

    size_t              key_bytes = 0;
    bytemap_entry*      entry     = position->tickers->first;

    while (entry) {
        key_bytes += entry->key_length + 1;
        entry = entry->next;
    }

    char* block = (char*) c_ap_alloc(n_tickers * sizeof(char*) + key_bytes, allocator);
    if (!block) return POSITION_ERR_OOM;

    char** tickers = (char**) block;
    char*  cursor  = block + n_tickers * sizeof(char*);
    size_t count   = 0;

    entry = position->tickers->first;

    while (entry && count < n_tickers) {
        const dma_position_ticker* cache = (const dma_position_ticker*) entry->value;

        if (cache->exposure_volume) {
            memcpy(cursor, entry->key, entry->key_length);
            cursor[entry->key_length] = '\0';

            tickers[count++] = cursor;
            cursor += entry->key_length + 1;
        }

        entry = entry->next;
    }

    int ret_code = POSITION_OK;

    for (size_t i = 0; i < count; ++i) {
        double exposure_volume = c_dma_position_get_exposure(position, tickers[i]);

        if (!exposure_volume) continue;

        md_side   side = exposure_volume > 0. ? SIDE_SHORT_CLOSE : SIDE_LONG_CLOSE;
        dma_algo* algo = c_dma_new_algo(position->dma, registration->reg_id, tickers[i], side, fabs(exposure_volume), NULL, allocator);

        if (!algo) {
            ret_code = POSITION_ERR_OOM;
            continue;
        }

        int add_code = c_dma_position_add_algo(position, algo);

        if (add_code != POSITION_OK) {
            ret_code = add_code;
            continue;
        }

        if (algo->handlers.launch && algo->handlers.launch(algo->DMA, algo, algo->handlers.user_data) != DMA_ALGO_OK) ret_code = POSITION_ERR_INVALID_INPUT;
    }

    c_ap_free(block);
    return ret_code;
}

// ========== Public APIs (Hot Path) ==========

/**
 * @brief Invoke every registered callback for an event.
 *
 * The traversal snapshots `next`, so a callback may unregister itself. A
 * nonzero callback result ends the dispatch and is returned.
 *
 * @param position The position (NULL-safe).
 * @param event The event to dispatch.
 * @param algo The hosted algo (borrowed).
 * @param order The order (borrowed).
 * @param trade The fill (borrowed; NULL for an order event).
 * @return POSITION_OK, or the first nonzero callback result.
 */
static inline int c_dma_position_invoke_callbacks(dma_position* position, dma_callback_event event, const dma_algo* algo, const md_trade_instruction* order, const md_trade_report* trade) {
    if (!position) return POSITION_ERR_INVALID_INPUT;

    dma_position_callback_ctx* callback = position->callbacks;

    while (callback) {
        dma_position_callback_ctx* next     = callback->next;
        int                        ret_code = 0;

        switch (event) {
            case on_canceled:
                if (callback->on_canceled) ret_code = callback->on_canceled(position, algo, order, callback->user_data);
                break;
            case on_rejected:
                if (callback->on_rejected) ret_code = callback->on_rejected(position, algo, order, callback->user_data);
                break;
            case on_listed:
                if (callback->on_listed) ret_code = callback->on_listed(position, algo, order, callback->user_data);
                break;
            case on_filled:
                if (callback->on_filled) ret_code = callback->on_filled(position, algo, order, trade, callback->user_data);
                break;
            default:
                break;
        }

        if (ret_code) return ret_code;

        callback = next;
    }

    return POSITION_OK;
}

#endif /* C_POSITION_H */

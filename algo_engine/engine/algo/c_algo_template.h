#ifndef C_ALGO_TEMPLATE_H
#define C_ALGO_TEMPLATE_H

#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <algo_engine/base/c_market_data/c_market_data.h>
#include <cbase/allocator_protocol/c_allocator_protocol.h>
#include <cbase/bytemap/c_bytemap.h>
#include <cbase/bytemap/xxh3.h>

typedef struct dma      dma;
typedef struct dma_algo dma_algo;

// ========== Enums ==========

/** @enum dma_algo_status
 *  @brief The algo lifecycle states. */
typedef enum dma_algo_status {
    DMA_ALGO_STATUS_IDLE      = 0,
    DMA_ALGO_STATUS_PREPARING = 1,
    DMA_ALGO_STATUS_READY     = 2,
    DMA_ALGO_STATUS_WORKING   = 3,
    DMA_ALGO_STATUS_DONE      = 4,
    DMA_ALGO_STATUS_CLOSED    = 5,
    DMA_ALGO_STATUS_STOPPING  = 6,
    DMA_ALGO_STATUS_REJECTED  = 7,
    DMA_ALGO_STATUS_ERROR     = 8,
} dma_algo_status;

// ========== Structs ==========

/** @brief A 16-byte identity, laid out as a UUID. */
typedef unsigned char uuid_t[16];

/**
 * @brief Invoked when an order of the algo changes state.
 * @param DMA The process-global access (borrowed).
 * @param algo The algo (borrowed).
 * @param order The order (borrowed).
 * @param user_data The value registered on the callback ctx.
 */
typedef void (*dma_algo_order_func)(dma* DMA, dma_algo* algo, const md_trade_instruction* order, void* user_data);

/**
 * @brief Invoked when a fill reaches the algo.
 * @param DMA The process-global access (borrowed).
 * @param algo The algo (borrowed).
 * @param trade The fill (borrowed).
 * @param user_data The value registered on the callback ctx.
 */
typedef void (*dma_algo_trade_func)(dma* DMA, dma_algo* algo, const md_trade_report* trade, void* user_data);

/**
 * @brief Invoked when market data reaches the algo.
 * @param DMA The process-global access (borrowed).
 * @param algo The algo (borrowed).
 * @param market_data The observation (borrowed).
 * @param user_data The value registered on the callback ctx.
 */
typedef void (*dma_algo_md_func)(dma* DMA, dma_algo* algo, const md_variant* market_data, void* user_data);

/**
 * @struct dma_algo_callback_ctx
 * @brief The five callbacks of an algo and their shared user data.
 *
 * The callbacks are NOT owned. A NULL slot is skipped by the engine dispatch.
 */
typedef struct dma_algo_callback_ctx {
    dma_algo_order_func on_canceled;  // NOT owned - canceled-order callback.
    dma_algo_order_func on_rejected;  // NOT owned - rejected-order callback.
    dma_algo_order_func on_listed;    // NOT owned - listed-order callback.
    dma_algo_trade_func on_filled;    // NOT owned - fill callback.
    dma_algo_md_func    on_md;        // NOT owned - market-data callback.
    void*               user_data;    // NOT owned - passed to every callback.
} dma_algo_callback_ctx;

/**
 * @brief Launch the algo. Returns 0 on success, negative on error.
 * @param DMA The process-global access (borrowed).
 * @param algo The algo (borrowed).
 * @param user_data The value registered with the hook.
 */
typedef int (*dma_algo_launch_func)(dma* DMA, dma_algo* algo, void* user_data);

/**
 * @brief Stop the algo. Returns 0 on success, negative on error.
 * @param DMA The process-global access (borrowed).
 * @param algo The algo (borrowed).
 * @param user_data The value registered with the hook.
 */
typedef int (*dma_algo_stop_func)(dma* DMA, dma_algo* algo, void* user_data);

/**
 * @struct dma_algo_handler_ctx
 * @brief The launch and stop hooks of an algo and their shared user data.
 *
 * The hooks are NOT owned. A NULL slot is refused by the engine ops.
 */
typedef struct dma_algo_handler_ctx {
    dma_algo_launch_func launch;     // NOT owned - algo launch hook.
    dma_algo_stop_func   stop;       // NOT owned - algo stop hook.
    void*                user_data;  // NOT owned - passed to both hooks.
} dma_algo_handler_ctx;

/**
 * @struct dma_algo
 * @brief The C state of one registered algo.
 *
 * The numeric fields are the algo's own state. `orders` indexes every order
 * the algo created and `working_orders` the live ones; every entry is the
 * address of the engine's dma_order_ctx block, whose instruction starts at
 * offset 0, so the index answers both the instruction and the order context.
 * `prev` and `next` link the algo into the engine's per-ticker working list
 * from creation until it completes or errors; STOPPING and REJECTED keep their
 * seat. An algo record is an allocator block owned by the engine's algo index.
 */
typedef struct dma_algo {
    dma*                  DMA;              // NOT owned - the process-global access.
    uuid_t                algo_id;          // The registry key.
    uint32_t              algo_reg_id;      // The id of the registration that built it.
    const char*           ticker;           // OWNED - the instrument.
    md_side               side;             // Full direction and offset specification.
    dma_algo_status       status;           // Current lifecycle state.
    double                target_volume;    // Volume the algo was asked to trade.
    double                working_volume;   // Volume still working.
    double                exposure_volume;  // Signed filled volume.
    double                filled_volume;    // Absolute filled volume.
    double                filled_notional;  // Filled volume times price.
    double                fee;              // Accumulated fee.
    double                cash_flow;        // Accumulated cash flow.
    double                multiplier;       // Contract multiplier.
    dma_algo_callback_ctx callbacks;        // on_canceled/on_rejected/on_listed/on_filled/on_md callbacks.
    dma_algo_handler_ctx  handlers;         // launch/stop hooks.
    bytemap*              orders;           // NOT owned - order id -> borrowed instruction (dma_order_ctx address; engine-owned).
    bytemap*              working_orders;   // NOT owned - live order id -> borrowed instruction alias.
    session_datetime_t    ts_started;       // Session datetime the algo started working.
    session_datetime_t    ts_finished;      // Session datetime the algo completed.
    struct dma_algo*      prev;             // Ticker linked list.
    struct dma_algo*      next;             // Ticker linked list.
} dma_algo;

// ========== Forward Declarations ==========

// Lifecycle
static inline int                         c_dma_algo_init(dma_algo* algo, dma* DMA, const char* ticker, md_side side, double target_volume, double multiplier, allocator_protocol* allocator);
static inline void                        c_dma_algo_gen_uid(dma_algo* algo);
static inline int                         c_dma_algo_set_status(dma_algo* algo, dma_algo_status status);
static inline void                        c_dma_algo_dealloc(dma_algo* algo);
static inline void                        c_dma_algo_free(dma_algo* algo);

// Queries
static inline const md_trade_instruction* c_dma_algo_get_order(const dma_algo* algo, const long_md_id* order_id);
static inline const md_trade_instruction* c_dma_algo_get_working_order(const dma_algo* algo, const long_md_id* order_id);
static inline bool                         c_dma_algo_status_is_active(dma_algo_status status);

// Hooks
static inline int                         c_dma_algo_launch(dma_algo* algo);
static inline int                         c_dma_algo_stop(dma_algo* algo);

#include <algo_engine/engine/c_dma.h>
#include <algo_engine/engine/c_dma_algo.h>

// ========== Public APIs (Algo Lifecycle) ==========

/**
 * @brief Initialize a caller-allocated, zeroed algo record.
 *
 * When the access carries an algo engine, the record is registered and linked
 * into its ticker's working list; the initial IDLE status is inside the
 * working set.
 *
 * @param algo The zeroed record to initialize.
 * @param DMA The process-global access (borrowed).
 * @param ticker The instrument (copied; NULL is refused).
 * @param side The full direction and offset specification.
 * @param target_volume The volume the algo was asked to trade.
 * @param multiplier The contract multiplier.
 * @param allocator Allocator for the children; derived from algo when NULL.
 * @return DMA_ALGO_OK, DMA_ALGO_ERR_INVALID_INPUT, or DMA_ALGO_ERR_OOM.
 */
static inline int c_dma_algo_init(dma_algo* algo, dma* DMA, const char* ticker, md_side side, double target_volume, double multiplier, allocator_protocol* allocator) {
    if (!algo || !ticker) return DMA_ALGO_ERR_INVALID_INPUT;
    if (!allocator) allocator = c_ap_protocol_from_ptr(algo);

    algo->DMA = DMA;

    algo->ticker = c_ap_strdup(ticker, allocator);
    if (!algo->ticker) return DMA_ALGO_ERR_OOM;

    c_ap_acquire_ownership((const void*) algo->ticker, (const void*) algo);

    c_dma_algo_gen_uid(algo);

    dma_algo_engine* engine = DMA ? DMA->algo_engine : NULL;

    if (engine && engine->algo) {
        int ret_code = c_bytemap_ex_set(engine->algo, (const char*) algo->algo_id, sizeof(uuid_t), (const char*) &algo, sizeof(dma_algo*), 0, NULL);
        if (ret_code != BYTEMAP_OK) return DMA_ALGO_ERR_OOM;
    }

    algo->orders = c_bytemap_ex_new(16, sizeof(md_trade_instruction*), allocator);
    if (!algo->orders) return DMA_ALGO_ERR_OOM;

    algo->working_orders = c_bytemap_ex_new(16, sizeof(md_trade_instruction*), allocator);
    if (!algo->working_orders) return DMA_ALGO_ERR_OOM;

    algo->side          = side;
    algo->target_volume = target_volume;
    algo->multiplier    = multiplier;

    // The record starts in IDLE, inside the working set: join its ticker's list.
    if (engine && engine->working_algo) {
        size_t    ticker_len = strlen(ticker);
        char*     stored     = NULL;
        dma_algo* head       = NULL;

        if (c_bytemap_ex_get_ptr(engine->working_algo, ticker, ticker_len, &stored, NULL) == BYTEMAP_OK) head = *(dma_algo**) stored;

        algo->prev = NULL;
        algo->next = head;

        if (head) head->prev = algo;

        c_bytemap_ex_set(engine->working_algo, ticker, ticker_len, (const char*) &algo, sizeof(dma_algo*), 0, NULL);
    }

    return DMA_ALGO_OK;
}

/**
 * @brief Generate the 16-byte algo identity from the block address and ticker.
 *
 * The result is laid out as a UUID (version 4 / RFC variant). When the access
 * is bound to an engine, an id already held by a registered algo is re-rolled
 * with an incrementing salt until it is unique.
 *
 * @param algo The record whose id is written (NULL-safe).
 */
static inline void c_dma_algo_gen_uid(dma_algo* algo) {
    if (!algo) return;

    uintptr_t address = (uintptr_t) algo;
    uint64_t  salt    = 0;

    for (;;) {
        XXH3_state_t state;
        XXH3_128bits_reset(&state);

        XXH3_128bits_update(&state, &address, sizeof(address));
        if (algo->ticker) XXH3_128bits_update(&state, algo->ticker, strlen(algo->ticker));
        if (salt) XXH3_128bits_update(&state, &salt, sizeof(salt));

        XXH128_hash_t hash = XXH3_128bits_digest(&state);
        memcpy(algo->algo_id, &hash.low64, 8);
        memcpy(algo->algo_id + 8, &hash.high64, 8);

        algo->algo_id[6] = (algo->algo_id[6] & 0x0F) | 0x40;
        algo->algo_id[8] = (algo->algo_id[8] & 0x3F) | 0x80;

        dma_algo_engine* engine = algo->DMA ? algo->DMA->algo_engine : NULL;

        if (!engine || !engine->algo) return;

        char* stored = NULL;

        if (c_bytemap_ex_get_ptr(engine->algo, (const char*) algo->algo_id, sizeof(uuid_t), &stored, NULL) != BYTEMAP_OK) return;

        salt++;
    }
}

/**
 * @brief Apply a lifecycle status and maintain the working list and timestamps.
 *
 * The record is linked at creation and keeps its seat through every live
 * status: IDLE, PREPARING, READY, WORKING, STOPPING and REJECTED. DONE and
 * CLOSED release it; the status is recorded first, so a refused exit keeps
 * its seat and reports DMA_ALGO_ERR_WORKING_ORDERS until the order set
 * empties. ERROR reports to stderr, releases it and parks it on the engine's
 * error slot. Reaching WORKING copies the access's market-data timestamp into
 * ts_started; completion from WORKING copies it into ts_finished.
 *
 * @param algo The record.
 * @param status The status to enter.
 * @return DMA_ALGO_OK, DMA_ALGO_ERR_WORKING_ORDERS when completion still has a
 *         live order, or DMA_ALGO_ERR_OOM when the working list cannot be indexed.
 */
static inline int c_dma_algo_set_status(dma_algo* algo, dma_algo_status status) {
    dma_algo_status prev_status = algo->status;
    algo->status                = status;
    dma_algo_engine* engine     = algo->DMA->algo_engine;

    switch (status) {
        case DMA_ALGO_STATUS_WORKING: {
            if (prev_status != DMA_ALGO_STATUS_WORKING && algo->DMA->mds) algo->ts_started = algo->DMA->mds->ts;
        }
        // fall through
        // Case 1: Start working
        case DMA_ALGO_STATUS_IDLE:
        case DMA_ALGO_STATUS_PREPARING:
        case DMA_ALGO_STATUS_READY: {
            // One hash: a missing ticker inserts a NULL head, and the stored
            // slot is rewritten in place with this algo as the new head.
            dma_algo* empty  = NULL;
            char*     stored = NULL;
            dma_algo* head   = NULL;

            if (c_bytemap_ex_set_default(engine->working_algo, algo->ticker, strlen(algo->ticker), (const char*) &empty, sizeof(dma_algo*), 0, &stored, NULL) != BYTEMAP_OK) return DMA_ALGO_ERR_OOM;

            head = *(dma_algo**) stored;

            if (head != algo && !algo->prev) {
                algo->prev = NULL;
                algo->next = head;

                if (head) head->prev = algo;
                *(dma_algo**) stored = algo;
            }

            break;
        }
        // Case 2: Complete
        case DMA_ALGO_STATUS_DONE:
        case DMA_ALGO_STATUS_CLOSED: {
            if (algo->working_orders && c_bytemap_ex_len(algo->working_orders)) return DMA_ALGO_ERR_WORKING_ORDERS;

            // A finish time only exists for an algo that was working.
            if (prev_status == DMA_ALGO_STATUS_WORKING && algo->DMA->mds) algo->ts_finished = algo->DMA->mds->ts;

            if (algo->prev) algo->prev->next = algo->next;
            else {
                char*     stored = NULL;
                dma_algo* head   = NULL;

                if (c_bytemap_ex_get_ptr(engine->working_algo, algo->ticker, strlen(algo->ticker), &stored, NULL) == BYTEMAP_OK) head = *(dma_algo**) stored;

                if (head == algo) {
                    if (algo->next) *(dma_algo**) stored = algo->next;
                    else c_bytemap_ex_pop(engine->working_algo, algo->ticker, strlen(algo->ticker), 0, NULL, NULL);
                }
            }

            if (algo->next) algo->next->prev = algo->prev;

            algo->prev = NULL;
            algo->next = NULL;
            break;
        }
        // Case 3: Stopping
        case DMA_ALGO_STATUS_STOPPING: {
            // Stopping is a state of working: keep the list seat.
            break;
        }
        // Case 4: Rejected
        case DMA_ALGO_STATUS_REJECTED: {
            // The algo may retry: keep the list seat.
            break;
        }
        case DMA_ALGO_STATUS_ERROR: {
            fprintf(stderr, "[dma_algo] %s (registration %u) entered ERROR\n", algo->ticker, algo->algo_reg_id);

            if (algo->prev) algo->prev->next = algo->next;
            else {
                char*     stored = NULL;
                dma_algo* head   = NULL;

                if (c_bytemap_ex_get_ptr(engine->working_algo, algo->ticker, strlen(algo->ticker), &stored, NULL) == BYTEMAP_OK) head = *(dma_algo**) stored;

                if (head == algo) {
                    if (algo->next) *(dma_algo**) stored = algo->next;
                    else c_bytemap_ex_pop(engine->working_algo, algo->ticker, strlen(algo->ticker), 0, NULL, NULL);
                }
            }

            if (algo->next) algo->next->prev = algo->prev;

            algo->prev = NULL;
            algo->next = NULL;

            engine->error_algo = algo;
            break;
        }
    }

    return DMA_ALGO_OK;
}

/**
 * @brief Tear down a caller-allocated algo record, leaving it zeroed.
 *
 * The record leaves the engine's working list and its algo index; every order
 * block is freed and the maps are released. Teardown detaches unconditionally:
 * a freed record must never stay in the list, and the engine frees the order
 * contexts before the records.
 *
 * @param algo The record (NULL-safe).
 */
static inline void c_dma_algo_dealloc(dma_algo* algo) {
    if (!algo) return;

    dma_algo_engine* engine = algo->DMA ? algo->DMA->algo_engine : NULL;

    // Teardown detaches unconditionally: a freed block must not stay in the list.
    if (engine && engine->working_algo && algo->ticker) {
        const char* ticker     = algo->ticker;
        size_t      ticker_len = strlen(ticker);
        char*       stored     = NULL;
        dma_algo*   head       = NULL;

        if (c_bytemap_ex_get_ptr(engine->working_algo, ticker, ticker_len, &stored, NULL) == BYTEMAP_OK) head = *(dma_algo**) stored;

        if (algo->prev) algo->prev->next = algo->next;
        else if (head == algo) {
            if (algo->next) c_bytemap_ex_set(engine->working_algo, ticker, ticker_len, (const char*) &algo->next, sizeof(dma_algo*), 0, NULL);
            else c_bytemap_ex_pop(engine->working_algo, ticker, ticker_len, 0, NULL, NULL);
        }

        if (algo->next) algo->next->prev = algo->prev;

        algo->prev = NULL;
        algo->next = NULL;
    }

    if (engine && engine->algo) c_bytemap_ex_pop(engine->algo, (const char*) algo->algo_id, sizeof(uuid_t), 0, NULL, NULL);

    if (algo->orders) {
        c_bytemap_ex_free(algo->orders);
        algo->orders = NULL;
    }

    if (algo->working_orders) {
        c_bytemap_ex_free(algo->working_orders);
        algo->working_orders = NULL;
    }

    if (algo->ticker) c_ap_free((void*) algo->ticker);

    memset(algo, 0, sizeof(dma_algo));
}

/**
 * @brief Free an algo record allocated by its factory.
 * @param algo The record (NULL-safe).
 */
static inline void c_dma_algo_free(dma_algo* algo) {
    if (!algo) return;

    c_dma_algo_dealloc(algo);
    c_ap_free(algo);
}

// ========== Public APIs (Queries) ==========
//
// Order state is owned by the engine entries: `c_dma_algo_engine_new_order`
// creates and indexes an order, and `c_dma_algo_engine_on_order` / `on_trade`
// apply every later state change. No set/pop/register entry point is exposed:
// a caller writing the indexes directly would skip the working volume, the
// aggregates and the callbacks, so the only safe way to move an order is to
// feed the engine.

/**
 * @brief Look up an order the algo owns.
 * @param algo The record (NULL-safe).
 * @param order_id The order identity (non-NULL).
 * @return The borrowed order, or NULL.
 */
static inline const md_trade_instruction* c_dma_algo_get_order(const dma_algo* algo, const long_md_id* order_id) {
    if (!algo || !algo->orders || !order_id) return NULL;

    char* stored = NULL;

    if (c_bytemap_ex_get_ptr(algo->orders, (const char*) order_id, sizeof(long_md_id), &stored, NULL) != BYTEMAP_OK) return NULL;

    return *(const md_trade_instruction**) stored;
}

/**
 * @brief Look up a live order by its id.
 * @param algo The record (NULL-safe).
 * @param order_id The order identity (non-NULL).
 * @return The borrowed order, or NULL.
 */
static inline const md_trade_instruction* c_dma_algo_get_working_order(const dma_algo* algo, const long_md_id* order_id) {
    if (!algo || !algo->working_orders || !order_id) return NULL;

    char* stored = NULL;

    if (c_bytemap_ex_get_ptr(algo->working_orders, (const char*) order_id, sizeof(long_md_id), &stored, NULL) != BYTEMAP_OK) return NULL;

    return *(const md_trade_instruction**) stored;
}

/**
 * @brief Whether an algo status may still act.
 *
 * A completed, closed, rejected or errored record is not stopped again.
 *
 * @param status The algo status.
 * @return true while the record may still act.
 */
static inline bool c_dma_algo_status_is_active(dma_algo_status status) {
    return status != DMA_ALGO_STATUS_DONE && status != DMA_ALGO_STATUS_CLOSED && status != DMA_ALGO_STATUS_REJECTED && status != DMA_ALGO_STATUS_ERROR;
}

// ========== Public APIs (Hooks) ==========

/**
 * @brief Invoke the algo's launch hook.
 * @param algo The record (NULL-safe).
 * @return The hook's return code, or DMA_ALGO_ERR_INVALID_INPUT.
 */
static inline int c_dma_algo_launch(dma_algo* algo) {
    if (!algo || !algo->handlers.launch) return DMA_ALGO_ERR_INVALID_INPUT;

    return algo->handlers.launch(algo->DMA, algo, algo->handlers.user_data);
}

/**
 * @brief Invoke the algo's stop hook.
 * @param algo The record (NULL-safe).
 * @return The hook's return code, or DMA_ALGO_ERR_INVALID_INPUT.
 */
static inline int c_dma_algo_stop(dma_algo* algo) {
    if (!algo || !algo->handlers.stop) return DMA_ALGO_ERR_INVALID_INPUT;

    return algo->handlers.stop(algo->DMA, algo, algo->handlers.user_data);
}

#endif /* C_ALGO_TEMPLATE_H */

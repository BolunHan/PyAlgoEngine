#ifndef C_PASSIVE_H
#define C_PASSIVE_H

#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include <algo_engine/engine/algo/c_algo_template.h>
#include <algo_engine/engine/c_dma.h>

// ========== Constants ==========

static const char* const DMA_ALGO_PASSIVE_NAME   = "passive";
static const uint32_t    DMA_ALGO_PASSIVE_REG_ID = 0x0001;

// ========== Structs ==========

/**
 * @struct dma_algo_passive_params
 * @brief The passive-family configuration.
 */
typedef struct dma_algo_passive_params {
    double              limit_price;          // Absolute limit (NaN when unset).
    double              limit_adjust_factor;  // Market-price factor (NaN when unset).
    double              timeout;              // Seconds before cancelling; 0 disables.
    dma_algo_limit_mode limit_mode;
    bool                relaunch;  // true relaunches after a fill or cancel.
} dma_algo_passive_params;

/**
 * @struct dma_algo_passive
 * @brief The passive algo state; `base` is at offset 0.
 */
typedef struct dma_algo_passive {
    dma_algo            base;
    double              limit_price;
    double              limit_adjust_factor;
    double              timeout;
    dma_algo_limit_mode limit_mode;
    bool                relaunch;
} dma_algo_passive;

// ========== Forward Declarations ==========

static inline int       c_dma_algo_passive_launch(dma* DMA, dma_algo* base, void* user_data);
static inline int       c_dma_algo_passive_stop(dma* DMA, dma_algo* base, void* user_data);

static inline void      c_dma_algo_passive_on_canceled(dma* DMA, dma_algo* base, const md_trade_instruction* order, void* user_data);
static inline void      c_dma_algo_passive_on_rejected(dma* DMA, dma_algo* base, const md_trade_instruction* order, void* user_data);
static inline void      c_dma_algo_passive_on_filled(dma* DMA, dma_algo* base, const md_trade_report* trade, void* user_data);
static inline void      c_dma_algo_passive_on_md(dma* DMA, dma_algo* base, const md_variant* market_data, void* user_data);

static inline dma_algo* c_dma_algo_passive_new(dma* DMA, const char* ticker, md_side side, double target_volume, const dma_algo_passive_params* params, allocator_protocol* allocator);

// ========== Public APIs (Handlers) ==========

/**
 * @brief Launch one limit order for the remaining target volume.
 *
 * A relaunch is accepted while the algo is working and has no live order; a
 * launch with no finite computed limit is refused.
 *
 * @param DMA The process-global access (borrowed).
 * @param base The algo record (NULL-safe).
 * @param user_data The callback ctx value (unused).
 * @return DMA_ALGO_OK, DMA_ALGO_ERR_INVALID_INPUT, or DMA_ALGO_ERR_OOM.
 */
static inline int c_dma_algo_passive_launch(dma* DMA, dma_algo* base, void* user_data) {
    (void) user_data;

    if (!base || !DMA || !DMA->algo_engine) return DMA_ALGO_ERR_INVALID_INPUT;

    dma_algo_passive* algo   = (dma_algo_passive*) base;
    dma_algo_engine*  engine = DMA->algo_engine;

    if (base->status == DMA_ALGO_STATUS_STOPPING || base->status == DMA_ALGO_STATUS_CLOSED || base->status == DMA_ALGO_STATUS_DONE) return DMA_ALGO_OK;
    if (c_bytemap_ex_len(base->working_orders)) return DMA_ALGO_OK;

    // The engine owns the working volume; the algo only reads the live total.
    double volume = base->target_volume - base->filled_volume - base->working_volume;
    if (!isfinite(volume)) return DMA_ALGO_ERR_INVALID_INPUT;
    if (volume <= 0.) return DMA_ALGO_OK;

    double limit = c_dma_calculate_limit(DMA, base->ticker, base->side, algo->limit_price, algo->limit_adjust_factor, algo->limit_mode);
    if (!isfinite(limit)) return DMA_ALGO_ERR_INVALID_INPUT;

    md_trade_instruction* order = c_dma_algo_engine_new_order(engine, base, limit, volume);
    if (!order) return DMA_ALGO_ERR_OOM;

    c_dma_algo_set_status(base, DMA_ALGO_STATUS_WORKING);

    c_dma_launch_order(DMA, order);
    return DMA_ALGO_OK;
}

/**
 * @brief Stop the algo and cancel every live order.
 * @param DMA The process-global access (borrowed).
 * @param base The algo record (NULL-safe).
 * @param user_data The callback ctx value (unused).
 * @return DMA_ALGO_OK, or DMA_ALGO_ERR_INVALID_INPUT.
 */
static inline int c_dma_algo_passive_stop(dma* DMA, dma_algo* base, void* user_data) {
    (void) user_data;

    if (!base) return DMA_ALGO_ERR_INVALID_INPUT;

    // Stopping is a state of working: the record keeps its working-list seat
    // while the cancels resolve through the order index.
    c_dma_algo_set_status(base, DMA_ALGO_STATUS_STOPPING);

    // Every live order moves to canceling before its cancel is dispatched.
    size_t         canceled = 0;
    bytemap_entry* entry    = base->working_orders ? base->working_orders->first : NULL;

    while (entry) {
        bytemap_entry*        next  = entry->next;
        md_trade_instruction* order = *(md_trade_instruction**) entry->value;

        if (order && c_md_state_cancellable(order->order_state)) {
            c_dma_cancel_order(DMA, order);
            canceled++;
        }

        entry = next;
    }

    if (canceled == 0 && base->filled_volume <= 0.) c_dma_algo_set_status(base, DMA_ALGO_STATUS_CLOSED);

    return DMA_ALGO_OK;
}

// ========== Public APIs (Callbacks) ==========

/**
 * @brief React to a canceled order; the engine has already released it.
 * @param DMA The process-global access (borrowed).
 * @param base The algo record (NULL-safe).
 * @param order The canceled order (NULL-safe).
 * @param user_data The callback ctx value (unused).
 */
static inline void c_dma_algo_passive_on_canceled(dma* DMA, dma_algo* base, const md_trade_instruction* order, void* user_data) {
    (void) order;

    if (!base) return;

    dma_algo_passive* algo = (dma_algo_passive*) base;

    if (base->filled_volume >= base->target_volume) {
        c_dma_algo_set_status(base, DMA_ALGO_STATUS_DONE);
        return;
    }

    if (base->status == DMA_ALGO_STATUS_STOPPING) {
        c_dma_algo_set_status(base, base->filled_volume > 0. ? DMA_ALGO_STATUS_DONE : DMA_ALGO_STATUS_CLOSED);
        return;
    }

    if (algo->relaunch && base->status == DMA_ALGO_STATUS_WORKING) {
        c_dma_algo_passive_launch(DMA, base, user_data);
    }
}

/**
 * @brief React to a rejected order; the engine has already released it.
 * @param DMA The process-global access (borrowed).
 * @param base The algo record (NULL-safe).
 * @param order The rejected order (NULL-safe).
 * @param user_data The callback ctx value (unused).
 */
static inline void c_dma_algo_passive_on_rejected(dma* DMA, dma_algo* base, const md_trade_instruction* order, void* user_data) {
    (void) DMA;
    (void) order;
    (void) user_data;

    if (!base) return;

    c_dma_algo_set_status(base, base->filled_volume > 0. ? DMA_ALGO_STATUS_DONE : DMA_ALGO_STATUS_CLOSED);
}

/**
 * @brief Continue or complete the algo after a fill.
 *
 * The engine has already applied the report to the order context, updated the
 * algo aggregates, refreshed the working volume and released a finished order
 * from the working set.
 *
 * @param DMA The process-global access (borrowed).
 * @param base The algo record (NULL-safe).
 * @param trade The fill (NULL-safe).
 * @param user_data The callback ctx value (unused).
 */
static inline void c_dma_algo_passive_on_filled(dma* DMA, dma_algo* base, const md_trade_report* trade, void* user_data) {
    if (!base || !trade) return;

    dma_algo_passive* algo = (dma_algo_passive*) base;

    if (base->filled_volume >= base->target_volume) {
        c_dma_algo_set_status(base, DMA_ALGO_STATUS_DONE);
        return;
    }

    if (base->status == DMA_ALGO_STATUS_STOPPING) return;

    c_dma_algo_set_status(base, DMA_ALGO_STATUS_WORKING);

    if (algo->relaunch && c_bytemap_ex_len(base->working_orders) == 0) c_dma_algo_passive_launch(DMA, base, user_data);
}

/**
 * @brief Cancel the algo when the timeout elapses.
 * @param DMA The process-global access (borrowed).
 * @param base The algo record (NULL-safe).
 * @param market_data The observation (NULL-safe).
 * @param user_data The callback ctx value (unused).
 */
static inline void c_dma_algo_passive_on_md(dma* DMA, dma_algo* base, const md_variant* market_data, void* user_data) {
    if (!base || !market_data) return;

    dma_algo_passive* algo = (dma_algo_passive*) base;

    if (!algo->timeout || base->status != DMA_ALGO_STATUS_WORKING) return;

    double         timestamp = market_data->meta_info.dt.unix_ts;
    double         oldest    = 0.;
    bytemap_entry* entry     = base->working_orders ? base->working_orders->first : NULL;

    while (entry) {
        const md_trade_instruction* order = *(const md_trade_instruction**) entry->value;

        if (order && c_md_state_working(order->order_state)) {
            if (oldest == 0. || order->ts_placed < oldest) oldest = order->ts_placed;
        }

        entry = entry->next;
    }

    if (oldest <= 0.) return;

    if (timestamp - oldest > algo->timeout) c_dma_algo_passive_stop(DMA, base, user_data);
}

// ========== Public APIs (Constructor) ==========

/**
 * @brief Allocate and configure a passive-family algo.
 * @param DMA The process-global access (borrowed).
 * @param ticker The instrument.
 * @param side The trade side.
 * @param target_volume The volume to trade.
 * @param params Optional configuration; NULL uses the defaults.
 * @param allocator Allocator for the block and its children.
 * @return The algo record, or NULL.
 */
static inline dma_algo* c_dma_algo_passive_new(dma* DMA, const char* ticker, md_side side, double target_volume, const dma_algo_passive_params* params, allocator_protocol* allocator) {
    if (!DMA || !ticker) return NULL;

    dma_algo_passive* algo = (dma_algo_passive*) c_ap_alloc(sizeof(dma_algo_passive), allocator);
    if (!algo) return NULL;

    if (c_dma_algo_init(&algo->base, DMA, ticker, side, target_volume, 1., allocator) != DMA_ALGO_OK) {
        c_dma_algo_dealloc(&algo->base);
        c_ap_free(algo);
        return NULL;
    }

    algo->limit_price         = params ? params->limit_price : NAN;
    algo->limit_adjust_factor = params ? params->limit_adjust_factor : NAN;
    algo->limit_mode          = params ? params->limit_mode : DMA_ALGO_LIMIT_STRICT;
    algo->timeout             = params ? params->timeout : 0.;
    algo->relaunch            = params ? params->relaunch : false;

    algo->base.handlers.launch    = c_dma_algo_passive_launch;
    algo->base.handlers.stop      = c_dma_algo_passive_stop;
    algo->base.handlers.user_data = NULL;

    algo->base.callbacks.on_canceled = c_dma_algo_passive_on_canceled;
    algo->base.callbacks.on_rejected = c_dma_algo_passive_on_rejected;
    algo->base.callbacks.on_filled   = c_dma_algo_passive_on_filled;
    algo->base.callbacks.on_md       = c_dma_algo_passive_on_md;
    algo->base.callbacks.user_data   = NULL;

    return &algo->base;
}

#endif /* C_PASSIVE_H */

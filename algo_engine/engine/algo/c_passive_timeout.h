#ifndef C_PASSIVE_TIMEOUT_H
#define C_PASSIVE_TIMEOUT_H

#include <stddef.h>

#include <algo_engine/engine/algo/c_passive.h>

// ========== Constants ==========

static const char* const DMA_ALGO_PASSIVE_TIMEOUT_NAME   = "passive_timeout";
static const uint32_t    DMA_ALGO_PASSIVE_TIMEOUT_REG_ID = 0x0002;

// ========== Forward Declarations ==========

static inline dma_algo* c_dma_algo_passive_timeout_new(dma* DMA, const char* ticker, md_side side, double target_volume, const dma_algo_passive_params* params, allocator_protocol* allocator);

// ========== Public APIs (Constructor) ==========

/**
 * @brief Allocate a passive algo that cancels its order after the params timeout.
 * @param DMA The process-global access (borrowed).
 * @param ticker The instrument.
 * @param side The trade side.
 * @param target_volume The volume to trade.
 * @param params Optional passive configuration; NULL uses the defaults and a zero timeout.
 * @param allocator Allocator for the block and its children.
 * @return The algo record, or NULL.
 */
static inline dma_algo* c_dma_algo_passive_timeout_new(dma* DMA, const char* ticker, md_side side, double target_volume, const dma_algo_passive_params* params, allocator_protocol* allocator) {
    dma_algo_passive_params merged = {
        .limit_price         = NAN,
        .limit_adjust_factor = NAN,
        .timeout             = 0.,
        .limit_mode          = DMA_ALGO_LIMIT_STRICT,
        .relaunch            = false,
    };

    if (params) merged = *params;

    merged.relaunch = false;

    return c_dma_algo_passive_new(DMA, ticker, side, target_volume, &merged, allocator);
}

#endif /* C_PASSIVE_TIMEOUT_H */

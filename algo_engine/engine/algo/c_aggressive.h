#ifndef C_AGGRESSIVE_H
#define C_AGGRESSIVE_H

#include <stddef.h>

#include <algo_engine/engine/algo/c_passive.h>

// ========== Constants ==========

static const char* const DMA_ALGO_AGGRESSIVE_NAME   = "aggressive";
static const uint32_t    DMA_ALGO_AGGRESSIVE_REG_ID = 0x0003;

// ========== Forward Declarations ==========

static inline dma_algo* c_dma_algo_aggressive_new(dma* DMA, const char* ticker, md_side side, double target_volume, const dma_algo_passive_params* params, allocator_protocol* allocator);

// ========== Public APIs (Constructor) ==========

/**
 * @brief Allocate a passive algo that relaunches after a fill or cancel.
 * @param DMA The process-global access (borrowed).
 * @param ticker The instrument.
 * @param side The trade side.
 * @param target_volume The volume to trade.
 * @param params Optional passive configuration; NULL uses the defaults.
 * @param allocator Allocator for the block and its children.
 * @return The algo record, or NULL.
 */
static inline dma_algo* c_dma_algo_aggressive_new(dma* DMA, const char* ticker, md_side side, double target_volume, const dma_algo_passive_params* params, allocator_protocol* allocator) {
    dma_algo_passive_params merged = {
        .limit_price = NAN,
        .limit_adjust_factor = NAN,
        .timeout = 0.,
        .limit_mode = DMA_ALGO_LIMIT_STRICT,
        .relaunch = false,
    };

    if (params) merged = *params;

    merged.timeout = 0.;
    merged.relaunch = true;

    return c_dma_algo_passive_new(DMA, ticker, side, target_volume, &merged, allocator);
}

#endif /* C_AGGRESSIVE_H */

#ifndef C_BALANCE_H
#define C_BALANCE_H

#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include <cbase/allocator_protocol/c_allocator_protocol.h>

#include <algo_engine/base/c_market_data/c_market_data.h>

// ========== Enums ==========

// clang-format off

/** @enum dma_balance_flag
 *  @brief The fields a host has supplied; an unflagged field constrains nothing. */
typedef enum dma_balance_flag {
    DMA_BALANCE_F_BALANCE              = 1ULL << 0,
    DMA_BALANCE_F_EQUITY               = 1ULL << 1,
    DMA_BALANCE_F_NET_ASSET_VALUE      = 1ULL << 2,

    DMA_BALANCE_F_TOTAL_ASSET          = 1ULL << 3,
    DMA_BALANCE_F_TOTAL_LIABILITY      = 1ULL << 4,

    DMA_BALANCE_F_UNREALIZED_PNL       = 1ULL << 5,
    DMA_BALANCE_F_REALIZED_PNL         = 1ULL << 6,

    DMA_BALANCE_F_CASH_BALANCE         = 1ULL << 7,
    DMA_BALANCE_F_AVAILABLE_CASH       = 1ULL << 8,
    DMA_BALANCE_F_FROZEN_CASH          = 1ULL << 9,

    DMA_BALANCE_F_DEPOSIT              = 1ULL << 10,
    DMA_BALANCE_F_WITHDRAWAL           = 1ULL << 11,
    DMA_BALANCE_F_CASH_IN              = 1ULL << 12,
    DMA_BALANCE_F_CASH_OUT             = 1ULL << 13,

    DMA_BALANCE_F_MARGIN               = 1ULL << 14,
    DMA_BALANCE_F_INITIAL_MARGIN       = 1ULL << 15,
    DMA_BALANCE_F_MAINTENANCE_MARGIN   = 1ULL << 16,

    DMA_BALANCE_F_AVAILABLE_MARGIN     = 1ULL << 17,
    DMA_BALANCE_F_FROZEN_MARGIN        = 1ULL << 18,

    DMA_BALANCE_F_COLLATERAL           = 1ULL << 19,
    DMA_BALANCE_F_AVAILABLE_COLLATERAL = 1ULL << 20,

    DMA_BALANCE_F_BORROWED             = 1ULL << 21,
    DMA_BALANCE_F_INTEREST             = 1ULL << 22,

    DMA_BALANCE_F_BUYING_POWER         = 1ULL << 23,
    DMA_BALANCE_F_LEVERAGE             = 1ULL << 24,
} dma_balance_flag;

/** @enum dma_balance_ret_code
 *  @brief 0 is success, negatives are errors. */
typedef enum dma_balance_ret_code {
    BALANCE_OK                =  0,
    BALANCE_ERR_INVALID_INPUT = -1,
    BALANCE_ERR_OOM           = -2,
} dma_balance_ret_code;

// clang-format on

// ========== Structs ==========

/**
 * @struct dma_balance
 * @brief The account balance a host supplies; `flags` names the live fields.
 */
typedef struct dma_balance {
    double   balance;
    double   equity;
    double   net_asset_value;

    double   total_asset;
    double   total_liability;

    double   unrealized_pnl;
    double   realized_pnl;

    double   cash_balance;
    double   available_cash;
    double   frozen_cash;

    double   deposit;
    double   withdrawal;

    double   cash_in;
    double   cash_out;

    double   margin;
    double   initial_margin;
    double   maintenance_margin;

    double   available_margin;
    double   frozen_margin;

    double   collateral;
    double   available_collateral;

    double   borrowed;
    double   interest;

    double   buying_power;
    double   leverage;

    uint64_t flags;
} dma_balance;

// ========== Forward Declarations ==========

// Lifecycle
static inline dma_balance* c_balance_new(allocator_protocol* allocator);
static inline int          c_balance_init(dma_balance* balance);
static inline void         c_balance_dealloc(dma_balance* balance);
static inline void         c_balance_free(dma_balance* balance);
static inline int          c_balance_clear(dma_balance* balance);

// Checks
static inline bool         c_balance_check(const dma_balance* balance, md_side side, double volume, double price, double multiplier, double margin_rate);

// ========== Public APIs ==========

/**
 * @brief Allocate a zeroed balance.
 * @param allocator Allocator for the block.
 * @return The new balance, or NULL on allocation failure.
 */
static inline dma_balance* c_balance_new(allocator_protocol* allocator) {
    dma_balance* balance = (dma_balance*) c_ap_alloc(sizeof(dma_balance), allocator);
    if (!balance) return NULL;

    if (c_balance_init(balance) != BALANCE_OK) {
        c_ap_free(balance);
        return NULL;
    }

    return balance;
}

/**
 * @brief Accept a caller-allocated, zeroed balance.
 * @param balance The zeroed balance.
 * @return BALANCE_OK, or BALANCE_ERR_INVALID_INPUT.
 */
static inline int c_balance_init(dma_balance* balance) {
    if (!balance) return BALANCE_ERR_INVALID_INPUT;

    return BALANCE_OK;
}

/**
 * @brief Drop every supplied field, leaving the struct zeroed.
 * @param balance The balance (NULL-safe).
 */
static inline void c_balance_dealloc(dma_balance* balance) {
    if (!balance) return;

    memset(balance, 0, sizeof(dma_balance));
}

/**
 * @brief Free a balance allocated by `c_balance_new`.
 * @param balance The balance (NULL-safe).
 */
static inline void c_balance_free(dma_balance* balance) {
    if (!balance) return;

    c_balance_dealloc(balance);
    c_ap_free(balance);
}

/**
 * @brief Drop every supplied field without releasing the block.
 * @param balance The balance (NULL-safe).
 * @return BALANCE_OK, or BALANCE_ERR_INVALID_INPUT.
 */
static inline int c_balance_clear(dma_balance* balance) {
    if (!balance) return BALANCE_ERR_INVALID_INPUT;

    c_balance_dealloc(balance);
    return BALANCE_OK;
}

/**
 * @brief Tell whether an order is affordable; an absent or unsupplied balance passes.
 *
 * A cash trade (NaN margin rate) constrains the buy side only: a sale releases
 * cash. A margin trade (any finite rate) consumes margin in either direction.
 * The selected field must be flagged, so an unsupplied balance passes with no
 * fallback.
 *
 * @param balance The balance (NULL-safe; NULL passes).
 * @param side The trade side.
 * @param volume The order volume.
 * @param price The per-unit price.
 * @param multiplier The contract multiplier.
 * @param margin_rate The designated margin rate; NaN means the trade is not on margin.
 * @return false only when the order exceeds the selected available field.
 */
static inline bool c_balance_check(const dma_balance* balance, md_side side, double volume, double price, double multiplier, double margin_rate) {
    if (!balance) return true;

    double notional = fabs(volume * price * multiplier);

    if (isnan(margin_rate)) {
        if (c_md_side_sign(side) <= 0) return true;
        if (!(balance->flags & DMA_BALANCE_F_AVAILABLE_CASH)) return true;
        return notional <= balance->available_cash;
    }

    if (!(balance->flags & DMA_BALANCE_F_AVAILABLE_MARGIN)) return true;
    return notional * margin_rate <= balance->available_margin;
}

#endif /* C_BALANCE_H */

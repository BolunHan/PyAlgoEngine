#ifndef C_INVENTORY_H
#define C_INVENTORY_H

#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include <algo_engine/exchange_profile/c_ex_profile_base.h>
#include <cbase/allocator_protocol/c_allocator_protocol.h>
#include <cbase/bytemap/c_bytemap.h>

// ========== Enums ==========

/** @enum inventory_ret_code
 *  @brief 0 is success, negatives are errors. */
typedef enum inventory_ret_code {
    INVENTORY_OK = 0,
    INVENTORY_ERR_INVALID_INPUT = -1,
    INVENTORY_ERR_OOM = -2,
    INVENTORY_ERR_NOT_FOUND = -3,
} inventory_ret_code;

/** @enum dma_security_type
 *  @brief The security classes an inventory entry can carry. */
typedef enum dma_security_type {
    DMA_SECURITY_COMMODITY = 0,
    DMA_SECURITY_CURRENCY_SWAP = 1,
    DMA_SECURITY_CRYPTO = 2,
    DMA_SECURITY_INDEX_FUTURE = 3,
    DMA_SECURITY_STOCK = 4,
} dma_security_type;

// ========== Structs ==========

/**
 * @struct dma_inventory_cash_dividend
 * @brief A cash dividend applied to a lot.
 */
typedef struct dma_inventory_cash_dividend {
    session_date_t                      record_date;
    session_date_t                      ex_date;
    double                              dividend_per_share;  // Cash per held share.
    bool                                applied;             // true once applied to a lot.
    struct dma_inventory_cash_dividend* next;
} dma_inventory_cash_dividend;

/**
 * @struct dma_inventory_stock_dividend
 * @brief A stock dividend applied to a lot.
 */
typedef struct dma_inventory_stock_dividend {
    session_date_t                       record_date;
    session_date_t                       ex_date;
    double                               dividend_per_share;  // Shares received per held share.
    bool                                 applied;             // true once applied to a lot.
    struct dma_inventory_stock_dividend* next;
} dma_inventory_stock_dividend;

/**
 * @struct dma_inventory_stock_split
 * @brief A stock split applied to a lot.
 */
typedef struct dma_inventory_stock_split {
    session_date_t                    record_date;
    session_date_t                    ex_date;
    double                            multiplier;
    bool                              applied;  // true once applied to a lot.
    struct dma_inventory_stock_split* next;
} dma_inventory_stock_split;

/**
 * @struct dma_inventory_stock_conversion
 * @brief A stock conversion applied to a lot.
 */
typedef struct dma_inventory_stock_conversion {
    session_date_t                         record_date;
    session_date_t                         ex_date;
    const char*                            convert_from;  // NOT owned - source ticker.
    const char*                            convert_to;    // NOT owned - target ticker.
    double                                 multiplier;
    bool                                   applied;  // true once applied to a lot.
    struct dma_inventory_stock_conversion* next;
} dma_inventory_stock_conversion;

/**
 * @struct dma_inventory_entry
 * @brief One inventory lot; a stored lot is owned by its node.
 */
typedef struct dma_inventory_entry {
    double                          volume;  // Normally positive.
    double                          price;
    double                          notional;
    double                          fee;
    double                          recalled;
    dma_security_type               security_type;
    dma_inventory_cash_dividend*    cash_dividends;     // OWNED - applied cash dividends.
    dma_inventory_stock_dividend*   stock_dividends;    // OWNED - applied stock dividends.
    dma_inventory_stock_split*      stock_splits;       // OWNED - applied splits.
    dma_inventory_stock_conversion* stock_conversions;  // OWNED - applied conversions.
    struct dma_inventory_entry*     next;
} dma_inventory_entry;

/**
 * @struct dma_inventory_node
 * @brief All inventory state of one ticker, its own allocator block.
 */
typedef struct dma_inventory_node {
    const char*          ticker;     // OWNED - the node's own copy of the ticker.
    dma_inventory_entry* long_inv;   // OWNED - long lot list.
    dma_inventory_entry* short_inv;  // OWNED - short lot list.
    double               long_used;
    double               short_used;
} dma_inventory_node;

/**
 * @struct dma_inventory
 * @brief The security lending inventory, keyed by ticker.
 */
typedef struct dma_inventory {
    bytemap* nodes;  // OWNED - ticker -> dma_inventory_node* (owned blocks).
} dma_inventory;

// ========== Forward Declarations ==========

static inline dma_inventory*                  c_dma_inventory_new(allocator_protocol* allocator);
static inline int                             c_dma_inventory_init(dma_inventory* inventory, allocator_protocol* allocator);
static inline void                            c_dma_inventory_dealloc(dma_inventory* inventory);
static inline void                            c_dma_inventory_free(dma_inventory* inventory);

static inline int                             c_dma_inventory_clear(dma_inventory* inventory);
static inline double                          c_dma_inventory_entry_available(const dma_inventory_entry* entry);

static inline dma_inventory_entry*            c_dma_inventory_entry_new(allocator_protocol* allocator);
static inline void                            c_dma_inventory_entry_free(dma_inventory_entry* entry);
static inline void                            c_dma_inventory_entry_clear(dma_inventory_entry* entry);

static inline dma_inventory_cash_dividend*    c_dma_inventory_cash_dividend_new(allocator_protocol* allocator);
static inline void                            c_dma_inventory_cash_dividend_free(dma_inventory_cash_dividend* event);
static inline dma_inventory_stock_dividend*   c_dma_inventory_stock_dividend_new(allocator_protocol* allocator);
static inline void                            c_dma_inventory_stock_dividend_free(dma_inventory_stock_dividend* event);
static inline dma_inventory_stock_split*      c_dma_inventory_stock_split_new(allocator_protocol* allocator);
static inline void                            c_dma_inventory_stock_split_free(dma_inventory_stock_split* event);
static inline dma_inventory_stock_conversion* c_dma_inventory_stock_conversion_new(allocator_protocol* allocator);
static inline void                            c_dma_inventory_stock_conversion_free(dma_inventory_stock_conversion* event);

static inline int                             c_dma_inventory_cash_dividend(dma_inventory* inventory, session_date_t record_date, session_date_t ex_date, double dividend_per_share);
static inline int                             c_dma_inventory_stock_dividend(dma_inventory* inventory, session_date_t record_date, session_date_t ex_date, double dividend_per_share);
static inline int                             c_dma_inventory_stock_split(dma_inventory* inventory, session_date_t record_date, session_date_t ex_date, double multiplier);
static inline int                             c_dma_inventory_stock_conversion(dma_inventory* inventory, session_date_t record_date, session_date_t ex_date, const char* convert_from, const char* convert_to, double multiplier);
static inline int                             c_dma_inventory_add_inv_long(dma_inventory* inventory, const char* ticker, double price, double volume, double notional, double fee, dma_security_type security_type);
static inline int                             c_dma_inventory_add_inv_short(dma_inventory* inventory, const char* ticker, double price, double volume, double notional, double fee, dma_security_type security_type);
static inline int                             c_dma_inventory_get_inv_long(const dma_inventory* inventory, const char* ticker, const dma_inventory_entry** out);
static inline int                             c_dma_inventory_get_inv_short(const dma_inventory* inventory, const char* ticker, const dma_inventory_entry** out);
static inline int                             c_dma_inventory_use_inv_long(dma_inventory* inventory, const char* ticker, double volume);
static inline int                             c_dma_inventory_use_inv_short(dma_inventory* inventory, const char* ticker, double volume);
static inline int                             c_dma_inventory_recall_inv_long(dma_inventory* inventory, const char* ticker, double volume);
static inline int                             c_dma_inventory_recall_inv_short(dma_inventory* inventory, const char* ticker, double volume);
static inline int                             c_dma_inventory_available(const dma_inventory* inventory, const char* ticker, double* out_long, double* out_short);

// ========== Utility Functions ==========

/**
 * @brief Free the four event lists of a lot.
 * @param entry The lot (NULL-safe).
 */
static inline void c_dma_inventory_entry_clear(dma_inventory_entry* entry) {
    if (!entry) return;

    dma_inventory_cash_dividend* cash = entry->cash_dividends;

    while (cash) {
        dma_inventory_cash_dividend* next = cash->next;
        c_ap_free(cash);
        cash = next;
    }

    dma_inventory_stock_dividend* stock = entry->stock_dividends;

    while (stock) {
        dma_inventory_stock_dividend* next = stock->next;
        c_ap_free(stock);
        stock = next;
    }

    dma_inventory_stock_split* split = entry->stock_splits;

    while (split) {
        dma_inventory_stock_split* next = split->next;
        c_ap_free(split);
        split = next;
    }

    dma_inventory_stock_conversion* conversion = entry->stock_conversions;

    while (conversion) {
        dma_inventory_stock_conversion* next = conversion->next;
        c_ap_free(conversion);
        conversion = next;
    }

    entry->cash_dividends = NULL;
    entry->stock_dividends = NULL;
    entry->stock_splits = NULL;
    entry->stock_conversions = NULL;
}

/**
 * @brief Free the ticker copy and both lot lists of a node, leaving it zeroed.
 * @param node The node (NULL-safe).
 */
static inline void c_dma_inventory_node_clear(dma_inventory_node* node) {
    if (!node) return;

    dma_inventory_entry* lot = node->long_inv;

    while (lot) {
        dma_inventory_entry* next = lot->next;
        c_dma_inventory_entry_clear(lot);
        c_ap_free(lot);
        lot = next;
    }

    lot = node->short_inv;

    while (lot) {
        dma_inventory_entry* next = lot->next;
        c_dma_inventory_entry_clear(lot);
        c_ap_free(lot);
        lot = next;
    }

    if (node->ticker) c_ap_free((void*) node->ticker);

    memset(node, 0, sizeof(dma_inventory_node));
}

// ========== Public APIs ==========

/**
 * @brief Allocate and initialize an inventory.
 * @param allocator Allocator for the inventory and its node map.
 * @return The new inventory, or NULL on allocation failure.
 */
static inline dma_inventory* c_dma_inventory_new(allocator_protocol* allocator) {
    dma_inventory* inventory = (dma_inventory*) c_ap_alloc(sizeof(dma_inventory), allocator);
    if (!inventory) return NULL;

    int ret_code = c_dma_inventory_init(inventory, allocator);
    if (ret_code != INVENTORY_OK) {
        c_ap_free(inventory);
        return NULL;
    }

    return inventory;
}

/**
 * @brief Initialize a caller-allocated, zeroed inventory.
 * @param inventory The zeroed inventory to initialize.
 * @param allocator Allocator for the node map; derived from inventory when NULL.
 * @return INVENTORY_OK, INVENTORY_ERR_INVALID_INPUT, or INVENTORY_ERR_OOM.
 */
static inline int c_dma_inventory_init(dma_inventory* inventory, allocator_protocol* allocator) {
    if (!inventory) return INVENTORY_ERR_INVALID_INPUT;
    if (!allocator) allocator = c_ap_protocol_from_ptr(inventory);

    memset(inventory, 0, sizeof(dma_inventory));

    inventory->nodes = c_bytemap_ex_new(0, sizeof(dma_inventory_node*), allocator);
    if (!inventory->nodes) return INVENTORY_ERR_OOM;

    return INVENTORY_OK;
}

/**
 * @brief Tear down a caller-allocated inventory, leaving it zeroed.
 * @param inventory The inventory (NULL-safe).
 */
static inline void c_dma_inventory_dealloc(dma_inventory* inventory) {
    if (!inventory) return;

    if (inventory->nodes) {
        // Free every node block before the map releases the keys.
        bytemap_entry* entry = inventory->nodes->first;

        while (entry) {
            bytemap_entry*      next = entry->next;
            dma_inventory_node* node = *(dma_inventory_node**) entry->value;

            if (node) {
                c_dma_inventory_node_clear(node);
                c_ap_free(node);
            }

            entry = next;
        }

        c_bytemap_ex_free(inventory->nodes);
        inventory->nodes = NULL;
    }

    memset(inventory, 0, sizeof(dma_inventory));
}

/**
 * @brief Free an inventory allocated by `c_dma_inventory_new`.
 * @param inventory The inventory (NULL-safe).
 */
static inline void c_dma_inventory_free(dma_inventory* inventory) {
    if (!inventory) return;

    c_dma_inventory_dealloc(inventory);
    c_ap_free(inventory);
}

/**
 * @brief Remove every lot, every consumed-volume record and every ticker node.
 * @param inventory The inventory (NULL-safe).
 * @return INVENTORY_OK, or INVENTORY_ERR_INVALID_INPUT.
 */
static inline int c_dma_inventory_clear(dma_inventory* inventory) {
    if (!inventory || !inventory->nodes) return INVENTORY_ERR_INVALID_INPUT;

    bytemap_entry* entry = inventory->nodes->first;

    while (entry) {
        bytemap_entry*      next = entry->next;
        dma_inventory_node* node = *(dma_inventory_node**) entry->value;

        if (node) {
            c_dma_inventory_node_clear(node);
            c_ap_free(node);
        }

        entry = next;
    }

    c_bytemap_ex_clear(inventory->nodes);
    return INVENTORY_OK;
}

/**
 * @brief The volume of a lot that is not recalled.
 * @param entry The lot (NULL-safe).
 * @return max(volume - recalled, 0).
 */
static inline double c_dma_inventory_entry_available(const dma_inventory_entry* entry) {
    if (!entry) return 0.;

    double available = entry->volume - entry->recalled;
    return available > 0 ? available : 0.;
}

/**
 * @brief Allocate a zeroed lot block.
 * @param allocator Allocator for the lot block.
 * @return The new lot, or NULL on allocation failure.
 */
static inline dma_inventory_entry* c_dma_inventory_entry_new(allocator_protocol* allocator) {
    return (dma_inventory_entry*) c_ap_alloc(sizeof(dma_inventory_entry), allocator);
}

/**
 * @brief Free a lot allocated by `c_dma_inventory_entry_new`.
 * @param entry The lot (NULL-safe).
 */
static inline void c_dma_inventory_entry_free(dma_inventory_entry* entry) {
    if (!entry) return;

    c_dma_inventory_entry_clear(entry);
    c_ap_free(entry);
}

/**
 * @brief Allocate a zeroed cash dividend event block.
 * @param allocator Allocator for the event block.
 * @return The new event, or NULL on allocation failure.
 */
static inline dma_inventory_cash_dividend* c_dma_inventory_cash_dividend_new(allocator_protocol* allocator) {
    return (dma_inventory_cash_dividend*) c_ap_alloc(sizeof(dma_inventory_cash_dividend), allocator);
}

/**
 * @brief Free a cash dividend event allocated by `..._new`.
 * @param event The event (NULL-safe).
 */
static inline void c_dma_inventory_cash_dividend_free(dma_inventory_cash_dividend* event) {
    if (!event) return;

    c_ap_free(event);
}

/**
 * @brief Apply and record a cash dividend across every lot.
 *
 * Each lot price is marked down by the dividend per share.
 *
 * @return INVENTORY_OK, INVENTORY_ERR_INVALID_INPUT, or INVENTORY_ERR_OOM.
 */
static inline int c_dma_inventory_cash_dividend(dma_inventory* inventory, session_date_t record_date, session_date_t ex_date, double dividend_per_share) {
    if (!inventory || !inventory->nodes) return INVENTORY_ERR_INVALID_INPUT;

    dma_inventory_cash_dividend event;
    memset(&event, 0, sizeof(dma_inventory_cash_dividend));

    event.record_date = record_date;
    event.ex_date = ex_date;
    event.dividend_per_share = dividend_per_share;
    event.applied = true;

    bytemap_entry* node_entry = inventory->nodes->first;

    while (node_entry) {
        dma_inventory_node* node = *(dma_inventory_node**) node_entry->value;

        if (node) {
            for (dma_inventory_entry* lot = node->long_inv; lot; lot = lot->next) {
                dma_inventory_cash_dividend* record = (dma_inventory_cash_dividend*) c_ap_alloc_child(sizeof(dma_inventory_cash_dividend), c_ap_protocol_from_ptr(lot), lot);
                if (!record) return INVENTORY_ERR_OOM;

                *record = event;
                record->next = NULL;

                dma_inventory_cash_dividend** tail = &lot->cash_dividends;
                while (*tail) tail = &(*tail)->next;
                *tail = record;

                lot->price -= dividend_per_share;
                lot->notional = lot->price * lot->volume;
            }

            for (dma_inventory_entry* lot = node->short_inv; lot; lot = lot->next) {
                dma_inventory_cash_dividend* record = (dma_inventory_cash_dividend*) c_ap_alloc_child(sizeof(dma_inventory_cash_dividend), c_ap_protocol_from_ptr(lot), lot);
                if (!record) return INVENTORY_ERR_OOM;

                *record = event;
                record->next = NULL;

                dma_inventory_cash_dividend** tail = &lot->cash_dividends;
                while (*tail) tail = &(*tail)->next;
                *tail = record;

                lot->price -= dividend_per_share;
                lot->notional = lot->price * lot->volume;
            }
        }

        node_entry = node_entry->next;
    }

    return INVENTORY_OK;
}

/**
 * @brief Allocate a zeroed stock dividend event block.
 * @param allocator Allocator for the event block.
 * @return The new event, or NULL on allocation failure.
 */
static inline dma_inventory_stock_dividend* c_dma_inventory_stock_dividend_new(allocator_protocol* allocator) {
    return (dma_inventory_stock_dividend*) c_ap_alloc(sizeof(dma_inventory_stock_dividend), allocator);
}

/**
 * @brief Free a stock dividend event allocated by `..._new`.
 * @param event The event (NULL-safe).
 */
static inline void c_dma_inventory_stock_dividend_free(dma_inventory_stock_dividend* event) {
    if (!event) return;

    c_ap_free(event);
}

/**
 * @brief Apply and record a stock dividend across every lot.
 *
 * Volume grows by `dividend_per_share` per held share and the price scales to
 * keep the notional.
 *
 * @return INVENTORY_OK, INVENTORY_ERR_INVALID_INPUT, or INVENTORY_ERR_OOM.
 */
static inline int c_dma_inventory_stock_dividend(dma_inventory* inventory, session_date_t record_date, session_date_t ex_date, double dividend_per_share) {
    if (!inventory || !inventory->nodes) return INVENTORY_ERR_INVALID_INPUT;

    double factor = 1. + dividend_per_share;
    if (factor <= 0) return INVENTORY_ERR_INVALID_INPUT;

    dma_inventory_stock_dividend event;
    memset(&event, 0, sizeof(dma_inventory_stock_dividend));

    event.record_date = record_date;
    event.ex_date = ex_date;
    event.dividend_per_share = dividend_per_share;
    event.applied = true;

    bytemap_entry* node_entry = inventory->nodes->first;

    while (node_entry) {
        dma_inventory_node* node = *(dma_inventory_node**) node_entry->value;

        if (node) {
            for (dma_inventory_entry* lot = node->long_inv; lot; lot = lot->next) {
                dma_inventory_stock_dividend* record = (dma_inventory_stock_dividend*) c_ap_alloc_child(sizeof(dma_inventory_stock_dividend), c_ap_protocol_from_ptr(lot), lot);
                if (!record) return INVENTORY_ERR_OOM;

                *record = event;
                record->next = NULL;

                dma_inventory_stock_dividend** tail = &lot->stock_dividends;
                while (*tail) tail = &(*tail)->next;
                *tail = record;

                lot->volume *= factor;
                lot->price /= factor;
            }

            for (dma_inventory_entry* lot = node->short_inv; lot; lot = lot->next) {
                dma_inventory_stock_dividend* record = (dma_inventory_stock_dividend*) c_ap_alloc_child(sizeof(dma_inventory_stock_dividend), c_ap_protocol_from_ptr(lot), lot);
                if (!record) return INVENTORY_ERR_OOM;

                *record = event;
                record->next = NULL;

                dma_inventory_stock_dividend** tail = &lot->stock_dividends;
                while (*tail) tail = &(*tail)->next;
                *tail = record;

                lot->volume *= factor;
                lot->price /= factor;
            }
        }

        node_entry = node_entry->next;
    }

    return INVENTORY_OK;
}

/**
 * @brief Allocate a zeroed split event block.
 * @param allocator Allocator for the event block.
 * @return The new event, or NULL on allocation failure.
 */
static inline dma_inventory_stock_split* c_dma_inventory_stock_split_new(allocator_protocol* allocator) {
    return (dma_inventory_stock_split*) c_ap_alloc(sizeof(dma_inventory_stock_split), allocator);
}

/**
 * @brief Free a split event allocated by `..._new`.
 * @param event The event (NULL-safe).
 */
static inline void c_dma_inventory_stock_split_free(dma_inventory_stock_split* event) {
    if (!event) return;

    c_ap_free(event);
}

/**
 * @brief Apply and record a split across every lot.
 *
 * Volume is multiplied and price divided.
 *
 * @return INVENTORY_OK, INVENTORY_ERR_INVALID_INPUT, or INVENTORY_ERR_OOM.
 */
static inline int c_dma_inventory_stock_split(dma_inventory* inventory, session_date_t record_date, session_date_t ex_date, double multiplier) {
    if (!inventory || !inventory->nodes || multiplier <= 0) return INVENTORY_ERR_INVALID_INPUT;

    dma_inventory_stock_split event;
    memset(&event, 0, sizeof(dma_inventory_stock_split));

    event.record_date = record_date;
    event.ex_date = ex_date;
    event.multiplier = multiplier;
    event.applied = true;

    bytemap_entry* node_entry = inventory->nodes->first;

    while (node_entry) {
        dma_inventory_node* node = *(dma_inventory_node**) node_entry->value;

        if (node) {
            for (dma_inventory_entry* lot = node->long_inv; lot; lot = lot->next) {
                dma_inventory_stock_split* record = (dma_inventory_stock_split*) c_ap_alloc_child(sizeof(dma_inventory_stock_split), c_ap_protocol_from_ptr(lot), lot);
                if (!record) return INVENTORY_ERR_OOM;

                *record = event;
                record->next = NULL;

                dma_inventory_stock_split** tail = &lot->stock_splits;
                while (*tail) tail = &(*tail)->next;
                *tail = record;

                lot->volume *= multiplier;
                lot->price /= multiplier;
            }

            for (dma_inventory_entry* lot = node->short_inv; lot; lot = lot->next) {
                dma_inventory_stock_split* record = (dma_inventory_stock_split*) c_ap_alloc_child(sizeof(dma_inventory_stock_split), c_ap_protocol_from_ptr(lot), lot);
                if (!record) return INVENTORY_ERR_OOM;

                *record = event;
                record->next = NULL;

                dma_inventory_stock_split** tail = &lot->stock_splits;
                while (*tail) tail = &(*tail)->next;
                *tail = record;

                lot->volume *= multiplier;
                lot->price /= multiplier;
            }
        }

        node_entry = node_entry->next;
    }

    return INVENTORY_OK;
}

/**
 * @brief Allocate a zeroed conversion event block.
 * @param allocator Allocator for the event block.
 * @return The new event, or NULL on allocation failure.
 */
static inline dma_inventory_stock_conversion* c_dma_inventory_stock_conversion_new(allocator_protocol* allocator) {
    return (dma_inventory_stock_conversion*) c_ap_alloc(sizeof(dma_inventory_stock_conversion), allocator);
}

/**
 * @brief Free a conversion event allocated by `..._new`.
 * @param event The event (NULL-safe).
 */
static inline void c_dma_inventory_stock_conversion_free(dma_inventory_stock_conversion* event) {
    if (!event) return;

    c_ap_free(event);
}

/**
 * @brief Apply and record a conversion across every lot.
 *
 * Volume is multiplied and price divided; moving the lots to the target
 * ticker is the caller's job.
 *
 * @return INVENTORY_OK, INVENTORY_ERR_INVALID_INPUT, or INVENTORY_ERR_OOM.
 */
static inline int c_dma_inventory_stock_conversion(dma_inventory* inventory, session_date_t record_date, session_date_t ex_date, const char* convert_from, const char* convert_to, double multiplier) {
    if (!inventory || !inventory->nodes || multiplier <= 0) return INVENTORY_ERR_INVALID_INPUT;

    dma_inventory_stock_conversion event;
    memset(&event, 0, sizeof(dma_inventory_stock_conversion));

    event.record_date = record_date;
    event.ex_date = ex_date;
    event.convert_from = convert_from;
    event.convert_to = convert_to;
    event.multiplier = multiplier;
    event.applied = true;

    bytemap_entry* node_entry = inventory->nodes->first;

    while (node_entry) {
        dma_inventory_node* node = *(dma_inventory_node**) node_entry->value;

        if (node) {
            for (dma_inventory_entry* lot = node->long_inv; lot; lot = lot->next) {
                dma_inventory_stock_conversion* record = (dma_inventory_stock_conversion*) c_ap_alloc_child(sizeof(dma_inventory_stock_conversion), c_ap_protocol_from_ptr(lot), lot);
                if (!record) return INVENTORY_ERR_OOM;

                *record = event;
                record->next = NULL;

                dma_inventory_stock_conversion** tail = &lot->stock_conversions;
                while (*tail) tail = &(*tail)->next;
                *tail = record;

                lot->volume *= multiplier;
                lot->price /= multiplier;
            }

            for (dma_inventory_entry* lot = node->short_inv; lot; lot = lot->next) {
                dma_inventory_stock_conversion* record = (dma_inventory_stock_conversion*) c_ap_alloc_child(sizeof(dma_inventory_stock_conversion), c_ap_protocol_from_ptr(lot), lot);
                if (!record) return INVENTORY_ERR_OOM;

                *record = event;
                record->next = NULL;

                dma_inventory_stock_conversion** tail = &lot->stock_conversions;
                while (*tail) tail = &(*tail)->next;
                *tail = record;

                lot->volume *= multiplier;
                lot->price /= multiplier;
            }
        }

        node_entry = node_entry->next;
    }

    return INVENTORY_OK;
}

/**
 * @brief Store one long lot with explicit fields.
 * @return INVENTORY_OK, INVENTORY_ERR_INVALID_INPUT, or INVENTORY_ERR_OOM.
 */
static inline int c_dma_inventory_add_inv_long(dma_inventory* inventory, const char* ticker, double price, double volume, double notional, double fee, dma_security_type security_type) {
    if (!inventory || !inventory->nodes || !ticker) return INVENTORY_ERR_INVALID_INPUT;

    dma_inventory_node* default_node = NULL;
    char*               stored = NULL;

    int                 ret_code = c_bytemap_ex_set_default(
        inventory->nodes,
        ticker,
        strlen(ticker),
        (const char*) &default_node,
        sizeof(dma_inventory_node*),
        0,
        &stored,
        NULL
    );

    if (ret_code != BYTEMAP_OK) return INVENTORY_ERR_OOM;

    dma_inventory_node* node = *(dma_inventory_node**) stored;

    if (!node) {
        allocator_protocol* allocator = c_ap_protocol_from_ptr(inventory->nodes);

        node = (dma_inventory_node*) c_ap_alloc(sizeof(dma_inventory_node), allocator);
        if (!node) return INVENTORY_ERR_OOM;

        char* copy = c_ap_strdup(ticker, allocator);
        if (!copy) {
            c_ap_free(node);
            return INVENTORY_ERR_OOM;
        }

        node->ticker = copy;
        *(dma_inventory_node**) stored = node;
    }

    allocator_protocol*  allocator = c_ap_protocol_from_ptr(node);

    dma_inventory_entry* lot = (dma_inventory_entry*) c_ap_alloc_child(sizeof(dma_inventory_entry), allocator, node);
    if (!lot) return INVENTORY_ERR_OOM;

    lot->price = price;
    lot->volume = volume;
    lot->notional = notional;
    lot->fee = fee;
    lot->security_type = security_type;

    if (!node->long_inv) {
        node->long_inv = lot;
    }
    else {
        dma_inventory_entry* tail = node->long_inv;
        while (tail->next) tail = tail->next;
        tail->next = lot;
    }

    return INVENTORY_OK;
}

/**
 * @brief Store one short lot with explicit fields.
 * @return INVENTORY_OK, INVENTORY_ERR_INVALID_INPUT, or INVENTORY_ERR_OOM.
 */
static inline int c_dma_inventory_add_inv_short(dma_inventory* inventory, const char* ticker, double price, double volume, double notional, double fee, dma_security_type security_type) {
    if (!inventory || !inventory->nodes || !ticker) return INVENTORY_ERR_INVALID_INPUT;

    dma_inventory_node* default_node = NULL;
    char*               stored = NULL;

    int                 ret_code = c_bytemap_ex_set_default(
        inventory->nodes,
        ticker,
        strlen(ticker),
        (const char*) &default_node,
        sizeof(dma_inventory_node*),
        0,
        &stored,
        NULL
    );

    if (ret_code != BYTEMAP_OK) return INVENTORY_ERR_OOM;

    dma_inventory_node* node = *(dma_inventory_node**) stored;

    if (!node) {
        allocator_protocol* allocator = c_ap_protocol_from_ptr(inventory->nodes);

        node = (dma_inventory_node*) c_ap_alloc(sizeof(dma_inventory_node), allocator);
        if (!node) return INVENTORY_ERR_OOM;

        char* copy = c_ap_strdup(ticker, allocator);
        if (!copy) {
            c_ap_free(node);
            return INVENTORY_ERR_OOM;
        }

        node->ticker = copy;
        *(dma_inventory_node**) stored = node;
    }

    allocator_protocol*  allocator = c_ap_protocol_from_ptr(node);

    dma_inventory_entry* lot = (dma_inventory_entry*) c_ap_alloc_child(sizeof(dma_inventory_entry), allocator, node);
    if (!lot) return INVENTORY_ERR_OOM;

    lot->price = price;
    lot->volume = volume;
    lot->notional = notional;
    lot->fee = fee;
    lot->security_type = security_type;

    if (!node->short_inv) {
        node->short_inv = lot;
    }
    else {
        dma_inventory_entry* tail = node->short_inv;
        while (tail->next) tail = tail->next;
        tail->next = lot;
    }

    return INVENTORY_OK;
}

/**
 * @brief The first stored long lot of a ticker.
 *
 * The pointer is a live view owned by the node.
 *
 * @param out Optional: receives the lot pointer.
 * @return INVENTORY_OK, INVENTORY_ERR_INVALID_INPUT, or INVENTORY_ERR_NOT_FOUND.
 */
static inline int c_dma_inventory_get_inv_long(const dma_inventory* inventory, const char* ticker, const dma_inventory_entry** out) {
    if (!inventory || !inventory->nodes || !ticker) return INVENTORY_ERR_INVALID_INPUT;

    char* stored = NULL;

    int   ret_code = c_bytemap_ex_get_ptr(inventory->nodes, ticker, strlen(ticker), &stored, NULL);
    if (ret_code == BYTEMAP_ERR_NOT_FOUND) return INVENTORY_ERR_NOT_FOUND;
    if (ret_code != BYTEMAP_OK) return INVENTORY_ERR_INVALID_INPUT;

    const dma_inventory_node* node = *(const dma_inventory_node**) stored;
    if (!node || !node->long_inv) return INVENTORY_ERR_NOT_FOUND;

    if (out) *out = node->long_inv;
    return INVENTORY_OK;
}

/**
 * @brief The first stored short lot of a ticker.
 *
 * The pointer is a live view owned by the node.
 *
 * @param out Optional: receives the lot pointer.
 * @return INVENTORY_OK, INVENTORY_ERR_INVALID_INPUT, or INVENTORY_ERR_NOT_FOUND.
 */
static inline int c_dma_inventory_get_inv_short(const dma_inventory* inventory, const char* ticker, const dma_inventory_entry** out) {
    if (!inventory || !inventory->nodes || !ticker) return INVENTORY_ERR_INVALID_INPUT;

    char* stored = NULL;

    int   ret_code = c_bytemap_ex_get_ptr(inventory->nodes, ticker, strlen(ticker), &stored, NULL);
    if (ret_code == BYTEMAP_ERR_NOT_FOUND) return INVENTORY_ERR_NOT_FOUND;
    if (ret_code != BYTEMAP_OK) return INVENTORY_ERR_INVALID_INPUT;

    const dma_inventory_node* node = *(const dma_inventory_node**) stored;
    if (!node || !node->short_inv) return INVENTORY_ERR_NOT_FOUND;

    if (out) *out = node->short_inv;
    return INVENTORY_OK;
}

/**
 * @brief Record consumed long volume for a ticker.
 * @return INVENTORY_OK, INVENTORY_ERR_INVALID_INPUT, or INVENTORY_ERR_OOM.
 */
static inline int c_dma_inventory_use_inv_long(dma_inventory* inventory, const char* ticker, double volume) {
    if (!inventory || !inventory->nodes || !ticker) return INVENTORY_ERR_INVALID_INPUT;

    dma_inventory_node* default_node = NULL;
    char*               stored = NULL;

    int                 ret_code = c_bytemap_ex_set_default(
        inventory->nodes,
        ticker,
        strlen(ticker),
        (const char*) &default_node,
        sizeof(dma_inventory_node*),
        0,
        &stored,
        NULL
    );

    if (ret_code != BYTEMAP_OK) return INVENTORY_ERR_OOM;

    dma_inventory_node* node = *(dma_inventory_node**) stored;

    if (!node) {
        allocator_protocol* allocator = c_ap_protocol_from_ptr(inventory->nodes);

        node = (dma_inventory_node*) c_ap_alloc(sizeof(dma_inventory_node), allocator);
        if (!node) return INVENTORY_ERR_OOM;

        char* copy = c_ap_strdup(ticker, allocator);
        if (!copy) {
            c_ap_free(node);
            return INVENTORY_ERR_OOM;
        }

        node->ticker = copy;
        *(dma_inventory_node**) stored = node;
    }

    node->long_used += volume;
    return INVENTORY_OK;
}

/**
 * @brief Record consumed short volume for a ticker.
 * @return INVENTORY_OK, INVENTORY_ERR_INVALID_INPUT, or INVENTORY_ERR_OOM.
 */
static inline int c_dma_inventory_use_inv_short(dma_inventory* inventory, const char* ticker, double volume) {
    if (!inventory || !inventory->nodes || !ticker) return INVENTORY_ERR_INVALID_INPUT;

    dma_inventory_node* default_node = NULL;
    char*               stored = NULL;

    int                 ret_code = c_bytemap_ex_set_default(
        inventory->nodes,
        ticker,
        strlen(ticker),
        (const char*) &default_node,
        sizeof(dma_inventory_node*),
        0,
        &stored,
        NULL
    );

    if (ret_code != BYTEMAP_OK) return INVENTORY_ERR_OOM;

    dma_inventory_node* node = *(dma_inventory_node**) stored;

    if (!node) {
        allocator_protocol* allocator = c_ap_protocol_from_ptr(inventory->nodes);

        node = (dma_inventory_node*) c_ap_alloc(sizeof(dma_inventory_node), allocator);
        if (!node) return INVENTORY_ERR_OOM;

        char* copy = c_ap_strdup(ticker, allocator);
        if (!copy) {
            c_ap_free(node);
            return INVENTORY_ERR_OOM;
        }

        node->ticker = copy;
        *(dma_inventory_node**) stored = node;
    }

    node->short_used += volume;
    return INVENTORY_OK;
}

/**
 * @brief Recall long volume from the earliest lots.
 *
 * Fully recalled lots are removed from the list.
 *
 * @return INVENTORY_OK, INVENTORY_ERR_INVALID_INPUT, or INVENTORY_ERR_NOT_FOUND.
 */
static inline int c_dma_inventory_recall_inv_long(dma_inventory* inventory, const char* ticker, double volume) {
    if (!inventory || !inventory->nodes || !ticker) return INVENTORY_ERR_INVALID_INPUT;

    char* stored = NULL;

    int   ret_code = c_bytemap_ex_get_ptr(inventory->nodes, ticker, strlen(ticker), &stored, NULL);
    if (ret_code == BYTEMAP_ERR_NOT_FOUND) return INVENTORY_ERR_NOT_FOUND;
    if (ret_code != BYTEMAP_OK) return INVENTORY_ERR_INVALID_INPUT;

    dma_inventory_node* node = *(dma_inventory_node**) stored;
    if (!node) return INVENTORY_ERR_NOT_FOUND;

    dma_inventory_entry* lot = node->long_inv;
    dma_inventory_entry* previous = NULL;
    double               remaining = volume > 0 ? volume : 0.;

    while (lot && remaining > 0) {
        dma_inventory_entry* next = lot->next;
        double               available = lot->volume - lot->recalled;
        double               recalled = available < remaining ? available : remaining;

        if (recalled > 0) {
            lot->recalled += recalled;
            remaining -= recalled;
        }

        if ((lot->volume - lot->recalled) <= 0) {
            if (previous) previous->next = next;
            else node->long_inv = next;

            c_dma_inventory_entry_clear(lot);
            c_ap_free(lot);
        }
        else {
            previous = lot;
        }

        lot = next;
    }

    return INVENTORY_OK;
}

/**
 * @brief Recall short volume from the earliest lots.
 *
 * Fully recalled lots are removed from the list.
 *
 * @return INVENTORY_OK, INVENTORY_ERR_INVALID_INPUT, or INVENTORY_ERR_NOT_FOUND.
 */
static inline int c_dma_inventory_recall_inv_short(dma_inventory* inventory, const char* ticker, double volume) {
    if (!inventory || !inventory->nodes || !ticker) return INVENTORY_ERR_INVALID_INPUT;

    char* stored = NULL;

    int   ret_code = c_bytemap_ex_get_ptr(inventory->nodes, ticker, strlen(ticker), &stored, NULL);
    if (ret_code == BYTEMAP_ERR_NOT_FOUND) return INVENTORY_ERR_NOT_FOUND;
    if (ret_code != BYTEMAP_OK) return INVENTORY_ERR_INVALID_INPUT;

    dma_inventory_node* node = *(dma_inventory_node**) stored;
    if (!node) return INVENTORY_ERR_NOT_FOUND;

    dma_inventory_entry* lot = node->short_inv;
    dma_inventory_entry* previous = NULL;
    double               remaining = volume > 0 ? volume : 0.;

    while (lot && remaining > 0) {
        dma_inventory_entry* next = lot->next;
        double               available = lot->volume - lot->recalled;
        double               recalled = available < remaining ? available : remaining;

        if (recalled > 0) {
            lot->recalled += recalled;
            remaining -= recalled;
        }

        if ((lot->volume - lot->recalled) <= 0) {
            if (previous) previous->next = next;
            else node->short_inv = next;

            c_dma_inventory_entry_clear(lot);
            c_ap_free(lot);
        }
        else {
            previous = lot;
        }

        lot = next;
    }

    return INVENTORY_OK;
}

/**
 * @brief Available volume of a ticker on both directions.
 *
 * A missing ticker reports 0 on both sides with INVENTORY_OK.
 *
 * @param out_long Optional: receives the available long volume.
 * @param out_short Optional: receives the available short volume.
 * @return INVENTORY_OK, or INVENTORY_ERR_INVALID_INPUT.
 */
static inline int c_dma_inventory_available(const dma_inventory* inventory, const char* ticker, double* out_long, double* out_short) {
    if (!inventory || !inventory->nodes || !ticker) return INVENTORY_ERR_INVALID_INPUT;

    double long_available = 0.;
    double short_available = 0.;

    char*  stored = NULL;

    int    ret_code = c_bytemap_ex_get_ptr(inventory->nodes, ticker, strlen(ticker), &stored, NULL);
    if (ret_code != BYTEMAP_OK && ret_code != BYTEMAP_ERR_NOT_FOUND) return INVENTORY_ERR_INVALID_INPUT;

    if (ret_code == BYTEMAP_OK) {
        const dma_inventory_node* node = *(const dma_inventory_node**) stored;

        if (node) {
            const dma_inventory_entry* lot = node->long_inv;
            double                     volume = 0.;
            double                     recalled = 0.;

            for (; lot; lot = lot->next) {
                volume += lot->volume;
                recalled += lot->recalled;
            }

            long_available = volume - recalled - node->long_used;

            lot = node->short_inv;
            volume = 0.;
            recalled = 0.;

            for (; lot; lot = lot->next) {
                volume += lot->volume;
                recalled += lot->recalled;
            }

            short_available = volume - recalled - node->short_used;
        }
    }

    if (out_long) *out_long = long_available > 0 ? long_available : 0.;
    if (out_short) *out_short = short_available > 0 ? short_available : 0.;

    return INVENTORY_OK;
}

#endif /* C_INVENTORY_H */

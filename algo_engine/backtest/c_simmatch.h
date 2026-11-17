#ifndef C_SIMMATCH_H
#define C_SIMMATCH_H

#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(_WIN32) || defined(_WIN64)
#include <process.h>
#else
#include <unistd.h>
#endif

#include <cbase/allocator_protocol/c_allocator_protocol.h>
#include <cbase/bytemap/c_bytemap.h>

#include <algo_engine/base/c_market_data/c_market_data.h>
#include <algo_engine/exchange_profile/c_ex_profile_base.h>

// ========== Constants ==========

/** @brief UUIDv4 version nibble written into byte 7 of a generated trade id. */
static const uint8_t  SMM_UUID_VERSION_NIBBLE = 0x40U;

/** @brief xorshift64* output multiplier. */
static const uint64_t SMM_RNG_MULTIPLIER = 0x2545F4914F6CDD1DULL;

/** @brief 2^-53, the scale turning the top 53 raw bits into [0, 1). */
static const double   SMM_RNG_DOUBLE_SCALE = 1.0 / 9007199254740992.0;

#ifndef SMM_ORDER_RING_DEFAULT_CAPACITY
#define SMM_ORDER_RING_DEFAULT_CAPACITY 32U
#endif

#ifndef SMM_ORDER_RING_GROWTH_FACTOR
#define SMM_ORDER_RING_GROWTH_FACTOR 2U
#endif

/** @brief Canonical bytemap key width of an order id (id type + id bytes). */
#define SMM_ID_KEY_SIZE (sizeof(md_id_type) + LONG_ID_SIZE)

#if SMM_ORDER_RING_GROWTH_FACTOR < 2U
#error "SMM_ORDER_RING_GROWTH_FACTOR must be >= 2"
#endif

#if (SMM_ORDER_RING_DEFAULT_CAPACITY & (SMM_ORDER_RING_DEFAULT_CAPACITY - 1U)) != 0U
#error "SMM_ORDER_RING_DEFAULT_CAPACITY must be a power of two"
#endif

// ========== Enums ==========

// clang-format off

/** @enum smm_match_event
 *  @brief The notification a registered listener receives. */
typedef enum smm_match_event {
    SMM_EVENT_MATCH  = 0,  // The order filled; the report is valid.
    SMM_EVENT_CANCEL = 1,  // The order was canceled; the report is NULL.
    SMM_EVENT_PLACED = 2,  // The order entered the registry; the report is NULL.
} smm_match_event;

/** @enum smm_ret_code
 *  @brief 0 is success, negatives are errors. */
typedef enum smm_ret_code {
    SMM_OK                 =  0,
    SMM_ERR_INVALID_ARG    = -1,
    SMM_ERR_DUPLICATE      = -2,  // The order id is already in the working registry.
    SMM_ERR_NOT_FOUND      = -3,  // The order id is not in the working registry.
    SMM_ERR_NO_VALID_PRICE = -4,
    SMM_ERR_INVALID_SIDE   = -5,
    SMM_ERR_OOM            = -6,
} smm_ret_code;

// clang-format on

// ========== Structs ==========

typedef struct smm_match_ctx         smm_match_ctx;
typedef struct smm_order_entry       smm_order_entry;
typedef struct smm_order_ring        smm_order_ring;
typedef struct smm_order_ring_cursor smm_order_ring_cursor;
typedef struct smm_listener          smm_listener;

/** @brief Fire-and-forget match / cancel notification; `user_data` is the registration value. */
typedef void (*smm_match_callback)(smm_match_ctx* ctx, smm_match_event event, const smm_order_entry* entry, const md_trade_report* report, void* user_data);

/**
 * @struct smm_order_entry
 * @brief One slot of the working-order ring.
 *
 * The registry owns a copy of the launched instruction: every fill and state
 * transition updates the copy, never the caller's instruction.
 */
struct smm_order_entry {
    md_trade_instruction* order;  // OWNED - the registry's copy of the launched instruction.
    uint64_t              transaction_count_at_placement;
    long_md_id            order_id;  // The registry key, an owned copy.
    bool                  occupied;  // The slot holds a live order.
};

/**
 * @struct smm_order_ring
 * @brief Contiguous ring-buffer registry of working orders.
 *
 * New orders append at the tail; the window `[head, head + span)` covers every
 * occupied slot. Canceled slots become holes and the dead ends of the window
 * are trimmed; a full window is compacted (or grown) on the next push.
 *
 * `index` maps an order id's bytes to its slot index + 1 (0 never names a
 * slot), which is how a cancel finds its slot without a scan.
 */
struct smm_order_ring {
    smm_order_entry* slots;       // OWNED - contiguous slot storage.
    bytemap*         index;       // OWNED - order_id bytes -> slot index + 1.
    size_t           capacity;    // Slot count, a power of two.
    size_t           head;        // Physical index of logical slot 0.
    size_t           span;        // Occupied window length (head .. tail).
    size_t           n_working;   // Live entries.
    uint64_t         generation;  // Bumped on rebuild; invalidates cursors.
};

/**
 * @struct smm_order_ring_cursor
 * @brief Newest-first, removal-safe scan cursor over a ring.
 *
 * The cursor walks physical slots backwards from the current tail and skips
 * unoccupied slots. A rebuild changes the layout, so a generation mismatch
 * ends the scan instead of walking stale indices.
 */
struct smm_order_ring_cursor {
    smm_order_ring* ring;
    size_t          slot;
    size_t          remaining;
    uint64_t        generation;
};

/**
 * @struct smm_listener
 * @brief One registered match / cancel notification listener.
 *
 * A linked list of callback nodes, each carrying its own user_data and a
 * unique id, so any number of wrappers can bind independently.
 */
struct smm_listener {
    smm_match_callback callback;
    void*              user_data;
    uintptr_t          id;  // The address of the node.
    smm_listener*      next;
};

/**
 * @struct smm_match_ctx
 * @brief Core state of the simulation matcher.
 *
 * The context is an allocator-protocol block: `c_ap_protocol_from_ptr` on it
 * is the one way to reach the allocator. Allocate it with `c_smm_match_new`;
 * a caller-allocated block must still be a block start handed out by
 * `c_ap_alloc`.
 */
struct smm_match_ctx {
    // === market state ===
    double         timestamp;               // Last market-data timestamp.
    double         last_price;              // Last market price, NAN if unset.
    uint64_t       last_transaction_count;  // Transaction counter.

    // === registry ===
    smm_order_ring orders;  // Contiguous ring buffer of working orders.

    // === matching config ===
    double         fee_rate;           // Fee as a fraction of notional.
    double         lag_ts;             // Minimum elapsed time (s) before a fill.
    uint64_t       lag_n_transaction;  // Minimum transaction count before a fill.
    double         hit_prob;           // Probability of a fill attempt succeeding.
    double         slippery_rate;      // Slippage as a fraction of price.

    // === random state ===
    uint64_t       seed;       // Configured seed (0 = entropy-derived).
    uint64_t       rng_state;  // xorshift64* state.

    // === listeners ===
    smm_listener*  listeners;  // Linked list of notification listeners.

    // === workspace ===
    md_variant*    ws_report;  // OWNED - allocator block reused for report construction.

    // === flags ===
    bool           instant_fill;              // Fill immediately on launch when no lag is set.
    bool           incremental_order_volume;  // Order data volume is incremental (SH style); when false,
                                              // order data is not matched (SZ depth reports carry resting volume).
};

// ========== Forward Declarations ==========

// --- Utility ---
static inline uint64_t              c_smm_pid(void);
static inline void                  c_smm_id_key(const long_md_id* id, char* out);
static inline bool                  c_smm_has_limit_price(const md_trade_instruction* order);
static inline void                  c_smm_rng_seed(smm_match_ctx* ctx, uint64_t seed);
static inline uint64_t              c_smm_rng_next_raw(smm_match_ctx* ctx);
static inline double                c_smm_rng_next(smm_match_ctx* ctx);
static inline void                  c_smm_gen_uuid(smm_match_ctx* ctx, long_md_id* out);
static inline bool                  c_smm_lag_allowed(const smm_match_ctx* ctx, const smm_order_entry* entry);
static inline bool                  c_smm_hit_ok(smm_match_ctx* ctx);
static inline double                c_smm_apply_slippage(const smm_match_ctx* ctx, double price, md_side side);
static inline bool                  c_smm_short_circuit(const smm_match_ctx* ctx, const md_trade_instruction* order);
static inline bool                  c_smm_order_eligible(const smm_match_ctx* ctx, const smm_order_entry* entry, double timestamp);

// --- Ring ---
static inline size_t                c_smm_ring_mask(const smm_order_ring* ring);
static inline int                   c_smm_ring_init(smm_order_ring* ring, allocator_protocol* allocator);
static inline void                  c_smm_ring_dealloc(smm_order_ring* ring);
static inline int                   c_smm_ring_rebuild(smm_order_ring* ring, allocator_protocol* allocator, size_t new_capacity);
static inline void                  c_smm_ring_clear(smm_order_ring* ring);
static inline void                  c_smm_ring_free_orders(smm_order_ring* ring);
static inline smm_order_entry*      c_smm_ring_find(smm_order_ring* ring, const long_md_id* order_id);
static inline smm_order_entry*      c_smm_ring_push(smm_order_ring* ring, const md_trade_instruction* order);
static inline void                  c_smm_ring_remove(smm_order_ring* ring, smm_order_entry* entry);
static inline smm_order_ring_cursor c_smm_ring_cursor_newest(smm_order_ring* ring);
static inline smm_order_entry*      c_smm_ring_next_newest(smm_order_ring_cursor* cursor);
static inline smm_order_entry*      c_smm_ring_newest(smm_order_ring* ring);

// --- Lifecycle ---
static inline smm_match_ctx*        c_smm_match_new(double fee_rate, bool instant_fill, double lag_ts, uint64_t lag_n_transaction, double hit_prob, double slippery_rate, bool incremental_order_volume, allocator_protocol* allocator);
static inline int                   c_smm_match_init(smm_match_ctx* ctx);
static inline void                  c_smm_match_dealloc(smm_match_ctx* ctx);
static inline void                  c_smm_match_free(smm_match_ctx* ctx);
static inline void                  c_smm_match_clear(smm_match_ctx* ctx);

// --- Configuration ---
static inline int                   c_smm_match_set_config(smm_match_ctx* ctx, double fee_rate, bool instant_fill, double lag_ts, uint64_t lag_n_transaction, double hit_prob, double slippery_rate, bool incremental_order_volume);
static inline int                   c_smm_match_set_seed(smm_match_ctx* ctx, uint64_t seed);

// --- Listeners ---
static inline int                   c_smm_match_register_listener(smm_match_ctx* ctx, smm_match_callback callback, void* user_data, uintptr_t* out_id);
static inline int                   c_smm_match_deregister_listener(smm_match_ctx* ctx, uintptr_t listener_id);

// --- Queries ---
static inline int                   c_smm_match_best_price(const double* prices, size_t n, int8_t sign, double* out);
static inline int                   c_smm_match_worst_price(const double* prices, size_t n, int8_t sign, double* out);

// --- Trading ---
static inline int                   c_smm_match_launch(smm_match_ctx* ctx, const md_trade_instruction* order);
static inline int                   c_smm_match_cancel(smm_match_ctx* ctx, const long_md_id* order_id);
static inline void                  c_smm_match_eod(smm_match_ctx* ctx);

// --- Hot path ---
// Runs once per market-data event.
static inline int                   c_smm_match_process(smm_match_ctx* ctx, const md_variant* market_data);

// --- Matching internals ---
static inline void                  c_smm_try_match(smm_match_ctx* ctx, smm_order_entry* entry, double match_volume, double match_price);
static inline void                  c_smm_check_bar(smm_match_ctx* ctx, const md_candlestick* bar);
static inline void                  c_smm_check_tick(smm_match_ctx* ctx, const md_tick_data* tick);
static inline void                  c_smm_check_tick_lite(smm_match_ctx* ctx, const md_tick_data_lite* lite);
static inline void                  c_smm_check_order(smm_match_ctx* ctx, const md_order_data* order_data);
static inline void                  c_smm_check_trade(smm_match_ctx* ctx, const md_transaction_data* transaction);

// ========== Utility Functions ==========

/**
 * @brief The current process id.
 * @return The pid as an unsigned value.
 */
static inline uint64_t c_smm_pid(void) {
#if defined(_WIN32) || defined(_WIN64)
    return (uint64_t) _getpid();
#else
    return (uint64_t) getpid();
#endif
}

/**
 * @brief Build the canonical bytemap key of an order id.
 *
 * The key is the id type followed by the id bytes, with no struct padding, so
 * a lookup is independent of uninitialized trailing bytes.
 * @param id  Order id.
 * @param out Key buffer of SMM_ID_KEY_SIZE bytes.
 */
static inline void c_smm_id_key(const long_md_id* id, char* out) {
    if (!id || !out) return;

    memcpy(out, &id->id_type, sizeof(md_id_type));
    memcpy(out + sizeof(md_id_type), id->data, LONG_ID_SIZE);
}

/**
 * @brief Whether the order carries an effective limit price.
 *
 * A NAN or infinite limit price is market-like: it constrains no price.
 * @param order Order instruction.
 * @return true when the limit price is finite.
 */
static inline bool c_smm_has_limit_price(const md_trade_instruction* order) {
    return order && isfinite(order->limit_price);
}

/**
 * @brief Seed the internal xorshift64* PRNG.
 * @param ctx  Matcher context.
 * @param seed Seed value; 0 derives from time and pid.
 */
static inline void c_smm_rng_seed(smm_match_ctx* ctx, uint64_t seed) {
    if (!ctx) return;

    ctx->seed = seed;
    if (seed == 0u) {
        uint64_t t = (uint64_t) time(NULL);
        uint64_t p = c_smm_pid();
        seed       = t ^ (p << 32u) ^ (p << 16u);
        if (seed == 0u) seed = SMM_RNG_MULTIPLIER;
    }
    ctx->rng_state = seed;
}

/**
 * @brief Advance the PRNG and return the next raw 64-bit value.
 * @param ctx Matcher context.
 * @return Next raw output.
 */
static inline uint64_t c_smm_rng_next_raw(smm_match_ctx* ctx) {
    if (!ctx) return 0u;

    uint64_t x = ctx->rng_state;
    x ^= x << 13u;
    x ^= x >> 7u;
    x ^= x << 17u;
    ctx->rng_state = x;
    return x * SMM_RNG_MULTIPLIER;
}

/**
 * @brief Advance the PRNG and return a double in [0, 1).
 * @param ctx Matcher context.
 * @return Next uniform value.
 */
static inline double c_smm_rng_next(smm_match_ctx* ctx) {
    return (double) (c_smm_rng_next_raw(ctx) >> 11u) * SMM_RNG_DOUBLE_SCALE;
}

/**
 * @brief Generate a UUIDv4 into an id buffer (bytes_le wire format).
 * @param ctx Matcher context (PRNG source).
 * @param out Target id; `id_type` is set to MID_UUID and 16 data bytes are written.
 */
static inline void c_smm_gen_uuid(smm_match_ctx* ctx, long_md_id* out) {
    if (!ctx || !out) return;

    uint64_t a = c_smm_rng_next_raw(ctx);
    uint64_t b = c_smm_rng_next_raw(ctx);
    uint8_t  bytes[16];
    memcpy(bytes, &a, 8);
    memcpy(bytes + 8u, &b, 8);

    bytes[7]     = (uint8_t) ((bytes[7] & 0x0Fu) | SMM_UUID_VERSION_NIBBLE);
    bytes[9]     = (uint8_t) ((bytes[9] & 0x3Fu) | 0x80u);
    out->id_type = MID_UUID;
    memcpy(out->data, bytes, 16);
}

/**
 * @brief Check whether the time and transaction lags allow a fill.
 * @param ctx   Matcher context.
 * @param entry Working order entry.
 * @return true when the fill may proceed.
 */
static inline bool c_smm_lag_allowed(const smm_match_ctx* ctx, const smm_order_entry* entry) {
    if (!ctx || !entry || !entry->order) return true;
    if (ctx->lag_ts <= 0 && ctx->lag_n_transaction == 0u) return true;

    double time_elapsed = ctx->timestamp - entry->order->meta_info.dt.unix_ts;
    if (ctx->lag_ts > 0 && time_elapsed < ctx->lag_ts) return false;

    uint64_t transactions_since = ctx->last_transaction_count >= entry->transaction_count_at_placement
                                      ? ctx->last_transaction_count - entry->transaction_count_at_placement
                                      : 0u;
    if (ctx->lag_n_transaction > 0 && transactions_since < ctx->lag_n_transaction) return false;
    return true;
}

/**
 * @brief Roll the hit-probability dice.
 * @param ctx Matcher context.
 * @return true when the fill attempt succeeds.
 */
static inline bool c_smm_hit_ok(smm_match_ctx* ctx) {
    if (!ctx) return false;
    if (ctx->hit_prob >= 1.0) return true;
    return c_smm_rng_next(ctx) < ctx->hit_prob;
}

/**
 * @brief Apply slippage to an execution price for the given side.
 * @param ctx   Matcher context.
 * @param price Base price.
 * @param side  Order side.
 * @return Buy: price * (1 + rate); sell: price * (1 - rate); an unknown side leaves the price.
 */
static inline double c_smm_apply_slippage(const smm_match_ctx* ctx, double price, md_side side) {
    if (!ctx) return price;

    double slippage = price * ctx->slippery_rate;
    int8_t sign     = c_md_side_sign(side);
    if (sign > 0) return price + slippage;
    if (sign < 0) return price - slippage;
    return price;
}

/**
 * @brief Check the instant-fill short circuit.
 * @param ctx   Matcher context.
 * @param order Order instruction.
 * @return true when the order should be matched immediately at launch.
 */
static inline bool c_smm_short_circuit(const smm_match_ctx* ctx, const md_trade_instruction* order) {
    if (!ctx || !order) return false;
    if (!c_smm_has_limit_price(order) && isnan(ctx->last_price)) return false;
    return ctx->instant_fill && ctx->lag_ts == 0 && ctx->lag_n_transaction == 0u;
}

/**
 * @brief Gate a working order for market-data processing.
 * @param ctx       Matcher context.
 * @param entry     Working order entry.
 * @param timestamp Incoming market-data timestamp.
 * @return true when the order is eligible for matching.
 */
static inline bool c_smm_order_eligible(const smm_match_ctx* ctx, const smm_order_entry* entry, double timestamp) {
    if (!ctx || !entry || !entry->order) return false;
    if (!c_md_state_working(entry->order->order_state)) return false;
    if (entry->order->meta_info.dt.unix_ts > timestamp) return false;
    return true;
}

// ========== Order Ring ==========

/**
 * @brief Physical slot mask; the capacity is always a power of two.
 * @param ring Order ring.
 * @return The wrap-around mask.
 */
static inline size_t c_smm_ring_mask(const smm_order_ring* ring) {
    return ring->capacity - 1u;
}

/**
 * @brief Initialize an empty order ring.
 * @param ring      The zeroed ring to initialize.
 * @param allocator Allocator for the slot storage and index.
 * @return SMM_OK, SMM_ERR_INVALID_ARG or SMM_ERR_OOM.
 */
static inline int c_smm_ring_init(smm_order_ring* ring, allocator_protocol* allocator) {
    if (!ring) return SMM_ERR_INVALID_ARG;

    size_t           capacity = (size_t) SMM_ORDER_RING_DEFAULT_CAPACITY;
    smm_order_entry* slots    = (smm_order_entry*) c_ap_alloc(capacity * sizeof(smm_order_entry), allocator);
    if (!slots) return SMM_ERR_OOM;

    bytemap* index = c_bytemap_new(capacity, allocator);
    if (!index) {
        c_ap_free(slots);
        return SMM_ERR_OOM;
    }

    ring->slots    = slots;
    ring->index    = index;
    ring->capacity = capacity;
    return SMM_OK;
}

/**
 * @brief Free the owned order copies of every occupied slot.
 * @param ring Order ring (NULL-safe).
 */
static inline void c_smm_ring_free_orders(smm_order_ring* ring) {
    if (!ring || !ring->slots) return;

    for (size_t i = 0u; i < ring->span; i++) {
        smm_order_entry* entry = &ring->slots[(ring->head + i) & c_smm_ring_mask(ring)];
        if (entry->order) {
            c_ap_free(entry->order);
            entry->order = NULL;
        }
    }
}

/**
 * @brief Tear down an order ring and leave it zeroed.
 * @param ring Order ring (NULL-safe).
 */
static inline void c_smm_ring_dealloc(smm_order_ring* ring) {
    if (!ring) return;

    c_smm_ring_free_orders(ring);
    if (ring->index) c_bytemap_free(ring->index);
    if (ring->slots) c_ap_free(ring->slots);
    memset(ring, 0, sizeof(smm_order_ring));
}

/**
 * @brief Rebuild the ring at a new capacity, compacting live entries.
 *
 * Live entries are copied oldest-first to the front of the new block and the
 * index is rewritten for the new slot indices. On failure the old ring is
 * left untouched.
 * @param ring         Order ring.
 * @param allocator    Allocator for the replacement blocks.
 * @param new_capacity New capacity (a power of two, >= 1).
 * @return SMM_OK, SMM_ERR_INVALID_ARG or SMM_ERR_OOM.
 */
static inline int c_smm_ring_rebuild(smm_order_ring* ring, allocator_protocol* allocator, size_t new_capacity) {
    if (!ring || new_capacity == 0u) return SMM_ERR_INVALID_ARG;

    smm_order_entry* slots = (smm_order_entry*) c_ap_alloc(new_capacity * sizeof(smm_order_entry), allocator);
    if (!slots) return SMM_ERR_OOM;

    bytemap* index = c_bytemap_new(new_capacity, allocator);
    if (!index) {
        c_ap_free(slots);
        return SMM_ERR_OOM;
    }

    size_t live = 0u;
    for (size_t i = 0u; i < ring->span; i++) {
        smm_order_entry* source = &ring->slots[(ring->head + i) & c_smm_ring_mask(ring)];
        if (!source->occupied) continue;

        char key[SMM_ID_KEY_SIZE];
        slots[live] = *source;
        c_smm_id_key(&slots[live].order_id, key);
        if (c_bytemap_set(index, key, SMM_ID_KEY_SIZE, (void*) (uintptr_t) (live + 1u), NULL) != BYTEMAP_OK) {
            c_bytemap_free(index);
            c_ap_free(slots);
            return SMM_ERR_OOM;
        }
        live++;
    }

    c_bytemap_free(ring->index);
    c_ap_free(ring->slots);

    ring->slots      = slots;
    ring->index      = index;
    ring->capacity   = new_capacity;
    ring->head       = 0u;
    ring->span       = live;
    ring->n_working  = live;
    ring->generation = ring->generation + 1u;
    return SMM_OK;
}

/**
 * @brief Drop every working entry, keeping the allocated capacity.
 * @param ring Order ring (NULL-safe).
 */
static inline void c_smm_ring_clear(smm_order_ring* ring) {
    if (!ring) return;

    if (ring->index) c_bytemap_clear(ring->index);
    c_smm_ring_free_orders(ring);
    if (ring->slots) memset(ring->slots, 0, ring->capacity * sizeof(smm_order_entry));

    ring->head       = 0u;
    ring->span       = 0u;
    ring->n_working  = 0u;
    ring->generation = ring->generation + 1u;
}

/**
 * @brief Find the working entry for an order id.
 * @param ring     Order ring.
 * @param order_id Order id to look up.
 * @return The live entry, or NULL.
 */
static inline smm_order_entry* c_smm_ring_find(smm_order_ring* ring, const long_md_id* order_id) {
    if (!ring || !ring->index || !ring->slots || !order_id) return NULL;

    void* value = NULL;
    char  key[SMM_ID_KEY_SIZE];
    c_smm_id_key(order_id, key);
    if (c_bytemap_get(ring->index, key, SMM_ID_KEY_SIZE, &value) != BYTEMAP_OK || !value) return NULL;

    size_t index = (size_t) (uintptr_t) value - 1u;
    if (index >= ring->capacity) return NULL;

    smm_order_entry* entry = &ring->slots[index];
    if (!entry->occupied) return NULL;
    if (!c_md_long_id_equal(&entry->order_id, order_id)) return NULL;
    return entry;
}

/**
 * @brief Append a copy of one order to the ring, compacting or growing when full.
 * @param ring  Order ring.
 * @param order Order instruction to copy into the registry.
 * @return The new entry, or NULL on OOM.
 */
static inline smm_order_entry* c_smm_ring_push(smm_order_ring* ring, const md_trade_instruction* order) {
    if (!ring || !ring->slots || !ring->index || !order) return NULL;

    if (ring->span == ring->capacity) {
        size_t              new_capacity      = ring->n_working == ring->capacity ? ring->capacity * SMM_ORDER_RING_GROWTH_FACTOR : ring->capacity;
        allocator_protocol* rebuild_allocator = c_ap_protocol_from_ptr((const void*) ring->slots);
        if (c_smm_ring_rebuild(ring, rebuild_allocator, new_capacity) != SMM_OK) return NULL;
    }

    // Re-derive after a rebuild: the replacement slots own a new protocol block.
    allocator_protocol*   allocator = c_ap_protocol_from_ptr((const void*) ring->slots);
    md_trade_instruction* owned     = (md_trade_instruction*) c_ap_alloc(sizeof(md_trade_instruction), allocator);
    if (!owned) return NULL;
    *owned = *order;

    char             key[SMM_ID_KEY_SIZE];
    size_t           tail  = (ring->head + ring->span) & c_smm_ring_mask(ring);
    smm_order_entry* entry = &ring->slots[tail];
    memset(entry, 0, sizeof(smm_order_entry));
    entry->order    = owned;
    entry->order_id = order->order_id;
    entry->occupied = true;

    c_smm_id_key(&entry->order_id, key);
    if (c_bytemap_set(ring->index, key, SMM_ID_KEY_SIZE, (void*) (uintptr_t) (tail + 1u), NULL) != BYTEMAP_OK) {
        c_ap_free(owned);
        memset(entry, 0, sizeof(smm_order_entry));
        return NULL;
    }

    ring->span++;
    ring->n_working++;
    return entry;
}

/**
 * @brief Remove one working entry from the ring.
 *
 * The slot becomes a hole; the dead ends of the window are trimmed. The entry
 * must belong to the ring.
 * @param ring  Order ring.
 * @param entry Entry to remove.
 */
static inline void c_smm_ring_remove(smm_order_ring* ring, smm_order_entry* entry) {
    if (!ring || !entry || !ring->slots || !ring->index) return;

    uintptr_t base = (uintptr_t) ring->slots;
    uintptr_t addr = (uintptr_t) entry;
    if (addr < base || addr >= base + ring->capacity * sizeof(smm_order_entry)) return;
    if ((addr - base) % sizeof(smm_order_entry) != 0u) return;

    char   key[SMM_ID_KEY_SIZE];
    size_t index = (size_t) (addr - base) / sizeof(smm_order_entry);
    entry        = &ring->slots[index];
    if (!entry->occupied) return;

    c_smm_id_key(&entry->order_id, key);
    c_bytemap_pop(ring->index, key, SMM_ID_KEY_SIZE, NULL);
    entry->occupied = false;
    if (entry->order) c_ap_free(entry->order);
    entry->order = NULL;
    if (ring->n_working > 0u) ring->n_working--;

    while (ring->span > 0u) {
        smm_order_entry* head_entry = &ring->slots[ring->head];
        if (head_entry->occupied) break;
        ring->head = (ring->head + 1u) & c_smm_ring_mask(ring);
        ring->span--;
    }
    while (ring->span > 0u) {
        smm_order_entry* tail_entry = &ring->slots[(ring->head + ring->span - 1u) & c_smm_ring_mask(ring)];
        if (tail_entry->occupied) break;
        ring->span--;
    }

    if (ring->span == 0u) ring->head = 0u;
}

/**
 * @brief Create a newest-first scan cursor.
 * @param ring Order ring.
 * @return The cursor (empty when the ring is empty).
 */
static inline smm_order_ring_cursor c_smm_ring_cursor_newest(smm_order_ring* ring) {
    smm_order_ring_cursor cursor;
    cursor.ring       = ring;
    cursor.generation = ring ? ring->generation : 0u;
    cursor.remaining  = ring ? ring->span : 0u;
    cursor.slot       = (ring && ring->span > 0u) ? ((ring->head + ring->span - 1u) & c_smm_ring_mask(ring)) : 0u;
    return cursor;
}

/**
 * @brief Yield the next occupied entry, newest first.
 *
 * Unoccupied slots are skipped. A rebuild invalidates the cursor, so a
 * callback that grows the registry ends the scan instead of walking a stale
 * layout.
 * @param cursor Scan cursor.
 * @return The next entry, or NULL when the scan is done.
 */
static inline smm_order_entry* c_smm_ring_next_newest(smm_order_ring_cursor* cursor) {
    if (!cursor || !cursor->ring) return NULL;

    smm_order_ring* ring = cursor->ring;
    if (cursor->generation != ring->generation) return NULL;

    while (cursor->remaining > 0u) {
        smm_order_entry* entry = &ring->slots[cursor->slot];
        cursor->slot           = (cursor->slot + ring->capacity - 1u) & c_smm_ring_mask(ring);
        cursor->remaining--;
        if (entry->occupied) return entry;
    }
    return NULL;
}

/**
 * @brief Newest occupied entry in the ring.
 * @param ring Order ring.
 * @return The newest entry, or NULL.
 */
static inline smm_order_entry* c_smm_ring_newest(smm_order_ring* ring) {
    smm_order_ring_cursor cursor = c_smm_ring_cursor_newest(ring);
    return c_smm_ring_next_newest(&cursor);
}

// ========== Listener APIs ==========

/**
 * @brief Register a match / cancel notification listener.
 *
 * The listener id is the address of the node and must be passed to
 * c_smm_match_deregister_listener.
 * @param ctx       Matcher context.
 * @param callback  Listener callback.
 * @param user_data Opaque data passed to the callback.
 * @param out_id    Receives the listener id (may be NULL).
 * @return SMM_OK, SMM_ERR_INVALID_ARG or SMM_ERR_OOM.
 */
static inline int c_smm_match_register_listener(smm_match_ctx* ctx, smm_match_callback callback, void* user_data, uintptr_t* out_id) {
    if (!ctx || !callback) return SMM_ERR_INVALID_ARG;

    smm_listener* node = (smm_listener*) calloc(1, sizeof(smm_listener));
    if (!node) return SMM_ERR_OOM;
    node->callback  = callback;
    node->user_data = user_data;
    node->id        = (uintptr_t) node;

    smm_listener** tail = &ctx->listeners;
    while (*tail) tail = &(*tail)->next;
    *tail = node;

    if (out_id) *out_id = node->id;
    return SMM_OK;
}

/**
 * @brief Deregister a notification listener by id.
 * @param ctx         Matcher context.
 * @param listener_id Id returned by c_smm_match_register_listener.
 * @return SMM_OK, SMM_ERR_INVALID_ARG or SMM_ERR_NOT_FOUND.
 */
static inline int c_smm_match_deregister_listener(smm_match_ctx* ctx, uintptr_t listener_id) {
    if (!ctx) return SMM_ERR_INVALID_ARG;

    smm_listener** link = &ctx->listeners;
    while (*link) {
        smm_listener* node = *link;
        if (node->id == listener_id) {
            *link = node->next;
            free(node);
            return SMM_OK;
        }
        link = &node->next;
    }
    return SMM_ERR_NOT_FOUND;
}

/**
 * @brief Fire an event to every registered listener.
 *
 * The next pointer is snapshotted before each call, so listeners may
 * deregister themselves from inside the callback. The entry is valid for the
 * duration of the call.
 * @param ctx    Matcher context.
 * @param event  Event type.
 * @param entry  Working order entry.
 * @param report Trade report for SMM_EVENT_MATCH, NULL otherwise.
 */
static inline void c_smm_match_fire(smm_match_ctx* ctx, smm_match_event event, const smm_order_entry* entry, const md_trade_report* report) {
    if (!ctx) return;

    smm_listener* listener = ctx->listeners;
    while (listener) {
        smm_listener* next = listener->next;
        if (listener->callback) listener->callback(ctx, event, entry, report, listener->user_data);
        listener = next;
    }
}

// ========== Public APIs ==========

/**
 * @brief Initialize a caller-allocated, zeroed matcher context to defaults.
 *
 * The block must be an allocator-protocol block start because the registry
 * derives the allocator with `c_ap_protocol_from_ptr`.
 *
 * Defaults: fee_rate 0, instant_fill false, lag 0, hit_prob 1, slippery 1e-4,
 * incremental_order_volume false, PRNG seeded from time and pid. The report
 * workspace is bound here so the matching hot path never allocates.
 * @param ctx The zeroed context.
 * @return SMM_OK, SMM_ERR_INVALID_ARG or SMM_ERR_OOM.
 */
static inline int c_smm_match_init(smm_match_ctx* ctx) {
    if (!ctx) return SMM_ERR_INVALID_ARG;

    ctx->last_price    = NAN;
    ctx->hit_prob      = 1.0;
    ctx->slippery_rate = 0.0001;
    c_smm_rng_seed(ctx, 0u);

    allocator_protocol* allocator = c_ap_protocol_from_ptr((const void*) ctx);
    ctx->ws_report                = (md_variant*) c_ap_alloc(sizeof(md_variant), allocator);
    if (!ctx->ws_report) return SMM_ERR_OOM;

    return c_smm_ring_init(&ctx->orders, allocator);
}

/**
 * @brief Allocate and initialize a matcher context.
 * @param fee_rate                 Fee as a fraction of notional.
 * @param instant_fill             Fill immediately at launch when no lag is set.
 * @param lag_ts                   Minimum elapsed time (s) before a fill.
 * @param lag_n_transaction        Minimum transaction count before a fill.
 * @param hit_prob                 Probability of a fill attempt succeeding.
 * @param slippery_rate            Slippage as a fraction of price.
 * @param incremental_order_volume Match order-data volume as incremental (SH
 *                                 style); when false order data is not matched
 *                                 (SZ depth reports carry resting volume).
 * @param allocator                Allocator for the context block; NULL uses the heap.
 * @return The new context, or NULL on allocation failure.
 */
static inline smm_match_ctx* c_smm_match_new(double fee_rate, bool instant_fill, double lag_ts, uint64_t lag_n_transaction, double hit_prob, double slippery_rate, bool incremental_order_volume, allocator_protocol* allocator) {
    smm_match_ctx* ctx = (smm_match_ctx*) c_ap_alloc(sizeof(smm_match_ctx), allocator);
    if (!ctx) return NULL;

    if (c_smm_match_init(ctx) != SMM_OK) {
        c_smm_match_dealloc(ctx);
        c_ap_free(ctx);
        return NULL;
    }

    c_smm_match_set_config(ctx, fee_rate, instant_fill, lag_ts, lag_n_transaction, hit_prob, slippery_rate, incremental_order_volume);
    return ctx;
}

/**
 * @brief Release every owned resource of a matcher context, leaving it zeroed.
 *
 * The context block itself is not freed; the registry, the report workspace
 * and every listener node are released.
 * @param ctx Matcher context (NULL-safe).
 */
static inline void c_smm_match_dealloc(smm_match_ctx* ctx) {
    if (!ctx) return;

    c_smm_ring_dealloc(&ctx->orders);

    smm_listener* listener = ctx->listeners;
    while (listener) {
        smm_listener* next = listener->next;
        free(listener);
        listener = next;
    }

    if (ctx->ws_report) c_ap_free(ctx->ws_report);
    memset(ctx, 0, sizeof(smm_match_ctx));
}

/**
 * @brief Free a matcher context.
 * @param ctx Matcher context (NULL-safe).
 */
static inline void c_smm_match_free(smm_match_ctx* ctx) {
    if (!ctx) return;

    c_smm_match_dealloc(ctx);
    c_ap_free(ctx);
}

/**
 * @brief Overwrite the matching configuration.
 * @param ctx                      Matcher context.
 * @param fee_rate                 Fee as a fraction of notional.
 * @param instant_fill             Fill immediately at launch when no lag is set.
 * @param lag_ts                   Minimum elapsed time (s) before a fill.
 * @param lag_n_transaction        Minimum transaction count before a fill.
 * @param hit_prob                 Probability of a fill attempt succeeding.
 * @param slippery_rate            Slippage as a fraction of price.
 * @param incremental_order_volume Match order-data volume as incremental (SH style).
 * @return SMM_OK or SMM_ERR_INVALID_ARG.
 */
static inline int c_smm_match_set_config(smm_match_ctx* ctx, double fee_rate, bool instant_fill, double lag_ts, uint64_t lag_n_transaction, double hit_prob, double slippery_rate, bool incremental_order_volume) {
    if (!ctx) return SMM_ERR_INVALID_ARG;

    ctx->fee_rate                 = fee_rate;
    ctx->instant_fill             = instant_fill;
    ctx->lag_ts                   = lag_ts;
    ctx->lag_n_transaction        = lag_n_transaction;
    ctx->hit_prob                 = hit_prob;
    ctx->slippery_rate            = slippery_rate;
    ctx->incremental_order_volume = incremental_order_volume;
    return SMM_OK;
}

/**
 * @brief (Re)seed the PRNG.
 * @param ctx  Matcher context.
 * @param seed Seed value; 0 derives from time and pid.
 * @return SMM_OK or SMM_ERR_INVALID_ARG.
 */
static inline int c_smm_match_set_seed(smm_match_ctx* ctx, uint64_t seed) {
    if (!ctx) return SMM_ERR_INVALID_ARG;

    c_smm_rng_seed(ctx, seed);
    return SMM_OK;
}

/**
 * @brief Reset market state and the registry, keeping config, listeners, the
 *        report workspace and the allocated ring capacity.
 *
 * Working orders are dropped; timestamp, last price and counters are reset;
 * the PRNG is re-seeded from the configured seed.
 * @param ctx Matcher context (NULL-safe).
 */
static inline void c_smm_match_clear(smm_match_ctx* ctx) {
    if (!ctx) return;

    c_smm_ring_clear(&ctx->orders);

    ctx->timestamp              = 0.0;
    ctx->last_price             = NAN;
    ctx->last_transaction_count = 0u;
    c_smm_rng_seed(ctx, ctx->seed);
}

/**
 * @brief Launch an order into the working registry.
 *
 * The registry copies the instruction and owns the copy; the caller's
 * instruction is never modified. Fires the SMM_EVENT_PLACED listener event and
 * applies the instant-fill short circuit.
 * @param ctx   Matcher context.
 * @param order Trade instruction to copy into the registry.
 * @return SMM_OK, SMM_ERR_INVALID_ARG, SMM_ERR_DUPLICATE or SMM_ERR_OOM.
 */
static inline int c_smm_match_launch(smm_match_ctx* ctx, const md_trade_instruction* order) {
    if (!ctx || !order) return SMM_ERR_INVALID_ARG;
    if (c_smm_ring_find(&ctx->orders, &order->order_id)) return SMM_ERR_DUPLICATE;

    smm_order_entry* entry = c_smm_ring_push(&ctx->orders, order);
    if (!entry) return SMM_ERR_OOM;

    entry->order->order_state             = STATE_PLACED;
    entry->order->ts_placed               = ctx->timestamp;
    entry->transaction_count_at_placement = ctx->last_transaction_count;

    c_smm_match_fire(ctx, SMM_EVENT_PLACED, entry, NULL);

    if (!c_smm_short_circuit(ctx, order)) return SMM_OK;

    // A listener may have rebuilt the ring (or canceled the order), so
    // resolve the entry again before matching it.
    entry = c_smm_ring_find(&ctx->orders, &order->order_id);
    if (!entry || !entry->order) return SMM_OK;

    double worst;
    if (!c_smm_has_limit_price(entry->order)) {
        worst = ctx->last_price;
    }
    else {
        int8_t sign = c_md_side_sign(entry->order->side);
        worst       = (sign > 0) ? fmax(entry->order->limit_price, ctx->last_price)
                                 : fmin(entry->order->limit_price, ctx->last_price);
    }
    c_smm_try_match(ctx, entry, NAN, worst);
    return SMM_OK;
}

/**
 * @brief Cancel a working order by id.
 *
 * Sets the order state to CANCELED (unless already FILLED), fires the
 * SMM_EVENT_CANCEL listeners and removes the entry.
 * @param ctx      Matcher context.
 * @param order_id Order id to cancel.
 * @return SMM_OK, SMM_ERR_INVALID_ARG or SMM_ERR_NOT_FOUND.
 */
static inline int c_smm_match_cancel(smm_match_ctx* ctx, const long_md_id* order_id) {
    if (!ctx || !order_id) return SMM_ERR_INVALID_ARG;

    smm_order_entry* entry = c_smm_ring_find(&ctx->orders, order_id);
    if (!entry) return SMM_ERR_NOT_FOUND;

    md_trade_instruction* order = entry->order;
    if (order && order->order_state != STATE_FILLED) {
        order->order_state = STATE_CANCELED;
        order->ts_canceled = ctx->timestamp;
    }
    c_smm_match_fire(ctx, SMM_EVENT_CANCEL, entry, NULL);
    c_smm_ring_remove(&ctx->orders, entry);
    return SMM_OK;
}

/**
 * @brief Cancel every working order (end of day).
 * @param ctx Matcher context (NULL-safe).
 */
static inline void c_smm_match_eod(smm_match_ctx* ctx) {
    if (!ctx) return;

    while (ctx->orders.n_working > 0u) {
        smm_order_entry* entry = c_smm_ring_newest(&ctx->orders);
        if (!entry) break;

        long_md_id order_id = entry->order_id;
        c_smm_match_cancel(ctx, &order_id);
    }
}

/**
 * @brief Process one market-data update (hot path: runs once per market-data event).
 *
 * Updates timestamp / last_price / transaction count and runs the matching
 * scan for the data type. Fires SMM_EVENT_MATCH listeners for each fill.
 * @param ctx         Matcher context.
 * @param market_data Market data header.
 * @return SMM_OK or SMM_ERR_INVALID_ARG.
 */
static inline int c_smm_match_process(smm_match_ctx* ctx, const md_variant* market_data) {
    if (!ctx || !market_data) return SMM_ERR_INVALID_ARG;

    ctx->timestamp  = market_data->meta_info.dt.unix_ts;
    ctx->last_price = c_md_get_price(market_data);

    switch (market_data->meta_info.dtype) {
        case DTYPE_TRANSACTION:
            ctx->last_transaction_count += 1u;
            c_smm_check_trade(ctx, &market_data->transaction_data);
            break;
        case DTYPE_BAR:
            c_smm_check_bar(ctx, &market_data->bar_data);
            break;
        case DTYPE_TICK:
            c_smm_check_tick(ctx, &market_data->tick_data_full);
            break;
        case DTYPE_TICK_LITE:
            c_smm_check_tick_lite(ctx, &market_data->tick_data_lite);
            break;
        case DTYPE_ORDER:
            // SZ-style depth reports carry resting volume, not incremental
            // fills - only match when incremental order volume is enabled.
            if (ctx->incremental_order_volume) c_smm_check_order(ctx, &market_data->order_data);
            break;
        default:
            break;
    }
    return SMM_OK;
}

/**
 * @brief Best price for a side (buy: min, sell: max).
 * @param prices Price array; non-finite entries are skipped.
 * @param n      Number of entries.
 * @param sign   Side sign (positive buy, negative sell).
 * @param out    Receives the best price.
 * @return SMM_OK, SMM_ERR_INVALID_ARG, SMM_ERR_NO_VALID_PRICE or SMM_ERR_INVALID_SIDE.
 */
static inline int c_smm_match_best_price(const double* prices, size_t n, int8_t sign, double* out) {
    if (!prices || !out) return SMM_ERR_INVALID_ARG;
    if (sign == 0) return SMM_ERR_INVALID_SIDE;

    double best = NAN;
    for (size_t i = 0u; i < n; i++) {
        double price = prices[i];
        if (!isfinite(price)) continue;
        if (isnan(best)) best = price;
        else if (sign > 0 ? price < best : price > best) best = price;
    }
    if (isnan(best)) return SMM_ERR_NO_VALID_PRICE;

    *out = best;
    return SMM_OK;
}

/**
 * @brief Worst price for a side (buy: max, sell: min).
 * @param prices Price array; non-finite entries are skipped.
 * @param n      Number of entries.
 * @param sign   Side sign (positive buy, negative sell).
 * @param out    Receives the worst price.
 * @return SMM_OK, SMM_ERR_INVALID_ARG, SMM_ERR_NO_VALID_PRICE or SMM_ERR_INVALID_SIDE.
 */
static inline int c_smm_match_worst_price(const double* prices, size_t n, int8_t sign, double* out) {
    if (!prices || !out) return SMM_ERR_INVALID_ARG;
    if (sign == 0) return SMM_ERR_INVALID_SIDE;

    double worst = NAN;
    for (size_t i = 0u; i < n; i++) {
        double price = prices[i];
        if (!isfinite(price)) continue;
        if (isnan(worst)) worst = price;
        else if (sign > 0 ? price > worst : price < worst) worst = price;
    }
    if (isnan(worst)) return SMM_ERR_NO_VALID_PRICE;

    *out = worst;
    return SMM_OK;
}

// ========== Matching Internals ==========

/**
 * @brief Attempt to match a working order at the given volume and price.
 *
 * Applies lag, hit probability, slippage and limit-price validation, then
 * builds a TradeReport, applies the fill to the order header and fires the
 * SMM_EVENT_MATCH listeners. Fully filled orders are removed from the
 * registry.
 * @param ctx          Matcher context.
 * @param entry        Working order entry (may be removed on full fill).
 * @param match_volume Desired volume; NAN = full working volume.
 * @param match_price  Desired price; NAN = the limit price.
 */
static inline void c_smm_try_match(smm_match_ctx* ctx, smm_order_entry* entry, double match_volume, double match_price) {
    if (!ctx || !entry || !entry->order) return;

    md_trade_instruction* order = entry->order;
    int8_t                sign  = c_md_side_sign(order->side);
    if (sign == 0) return;
    if (!c_smm_lag_allowed(ctx, entry)) return;
    if (!c_smm_hit_ok(ctx)) return;

    double working_volume = order->volume - order->filled_volume;
    if (isnan(match_volume) || match_volume > working_volume) match_volume = working_volume;
    if (match_volume <= 0) return;

    if (isnan(match_price)) {
        if (!c_smm_has_limit_price(order)) return;
        match_price = order->limit_price;
    }
    else {
        match_price = c_smm_apply_slippage(ctx, match_price, order->side);
    }

    if (c_smm_has_limit_price(order)) {
        if (sign > 0 && match_price > order->limit_price) match_price = order->limit_price;
        else if (sign < 0 && match_price < order->limit_price) match_price = order->limit_price;
    }
    if (!isfinite(match_price)) return;

    if (!ctx->ws_report) return;

    md_trade_report* rpt = &ctx->ws_report->trade_report;
    memset(ctx->ws_report, 0, sizeof(md_variant));
    rpt->meta_info.dtype  = DTYPE_REPORT;
    rpt->meta_info.ticker = order->meta_info.ticker;
    if (c_ex_profile_session_datetime_from_unix(ctx->timestamp, &rpt->meta_info.dt) != 0) {
        memset(&rpt->meta_info.dt, 0, sizeof(session_datetime_t));
    }
    rpt->meta_info.dt.date.stype = SESSION_TYPE_NORMINAL;
    rpt->price                   = match_price;
    rpt->volume                  = match_volume;
    rpt->side                    = order->side;
    rpt->multiplier              = order->multiplier;
    rpt->notional                = match_volume * match_price * order->multiplier;
    rpt->fee                     = ctx->fee_rate * rpt->notional;
    rpt->order_id                = order->order_id;
    c_smm_gen_uuid(ctx, &rpt->trade_id);

    order->filled_volume += match_volume;
    order->filled_notional += fabs(rpt->notional);
    order->fee += rpt->fee;
    if (order->filled_volume >= order->volume) {
        order->order_state = STATE_FILLED;
        order->ts_finished = ctx->timestamp;
    }
    else if (order->filled_volume > 0) {
        order->order_state = STATE_PARTFILLED;
    }

    c_smm_match_fire(ctx, SMM_EVENT_MATCH, entry, rpt);

    if (order->order_state == STATE_FILLED) c_smm_ring_remove(&ctx->orders, entry);
}

/**
 * @brief Matching scan for bar data.
 * @param ctx Matcher context.
 * @param bar BarData payload.
 */
static inline void c_smm_check_bar(smm_match_ctx* ctx, const md_candlestick* bar) {
    if (!ctx || !bar) return;

    smm_order_ring_cursor cursor = c_smm_ring_cursor_newest(&ctx->orders);
    smm_order_entry*      entry  = NULL;
    while ((entry = c_smm_ring_next_newest(&cursor))) {
        if (!c_smm_order_eligible(ctx, entry, bar->meta_info.dt.unix_ts)) continue;

        md_trade_instruction* order       = entry->order;
        int8_t                sign        = c_md_side_sign(order->side);
        double                match_price = NAN;

        if (!c_smm_has_limit_price(order)) {
            if (bar->volume > 0) match_price = bar->notional / bar->volume;
        }
        else if (sign > 0) {
            if (bar->high_price < order->limit_price) match_price = bar->high_price;
            else if (bar->low_price < order->limit_price) match_price = order->limit_price;
        }
        else if (sign < 0) {
            if (bar->low_price > order->limit_price) match_price = bar->low_price;
            else if (bar->high_price > order->limit_price) match_price = order->limit_price;
        }

        if (!isnan(match_price)) c_smm_try_match(ctx, entry, NAN, match_price);
    }
}

/**
 * @brief Matching scan for tick data (order book).
 * @param ctx  Matcher context.
 * @param tick TickData payload.
 */
static inline void c_smm_check_tick(smm_match_ctx* ctx, const md_tick_data* tick) {
    if (!ctx || !tick) return;

    smm_order_ring_cursor cursor = c_smm_ring_cursor_newest(&ctx->orders);
    smm_order_entry*      entry  = NULL;
    while ((entry = c_smm_ring_next_newest(&cursor))) {
        if (!c_smm_order_eligible(ctx, entry, tick->lite.meta_info.dt.unix_ts)) continue;

        md_trade_instruction* order = entry->order;
        int8_t                sign  = c_md_side_sign(order->side);
        if (sign == 0) continue;

        bool                has_limit = c_smm_has_limit_price(order);
        double              limit     = order->limit_price;
        const md_orderbook* book      = (sign > 0) ? tick->ask : tick->bid;
        bool                enter     = !has_limit;

        if (has_limit) {
            if (book->size > 0) enter = (sign > 0) ? (book->entries[0].price <= limit) : (book->entries[0].price >= limit);
        }

        if (!enter || book->size == 0) continue;

        double match_volume   = 0.0;
        double match_notional = 0.0;
        double working_volume = order->volume - order->filled_volume;

        for (size_t i = 0u; i < book->size; i++) {
            const md_orderbook_entry* level    = &book->entries[i];
            bool                      in_price = !has_limit || (sign > 0 ? level->price <= limit : level->price >= limit);
            if (!in_price) break;
            if (match_volume >= working_volume) break;

            double addition  = level->volume;
            double remaining = working_volume - match_volume;
            if (addition > remaining) addition = remaining;
            match_volume += addition;
            match_notional += addition * level->price;
        }

        if (match_volume > 0) c_smm_try_match(ctx, entry, match_volume, match_notional / match_volume);
    }
}

/**
 * @brief Matching scan for tick-lite data.
 * @param ctx  Matcher context.
 * @param lite TickDataLite payload.
 */
static inline void c_smm_check_tick_lite(smm_match_ctx* ctx, const md_tick_data_lite* lite) {
    if (!ctx || !lite) return;

    smm_order_ring_cursor cursor = c_smm_ring_cursor_newest(&ctx->orders);
    smm_order_entry*      entry  = NULL;
    while ((entry = c_smm_ring_next_newest(&cursor))) {
        if (!c_smm_order_eligible(ctx, entry, lite->meta_info.dt.unix_ts)) continue;

        md_trade_instruction* order = entry->order;
        int8_t                sign  = c_md_side_sign(order->side);
        if (sign == 0) continue;

        double vol   = 0.0;
        double price = NAN;
        if (!c_smm_has_limit_price(order)) {
            if (sign > 0) {
                vol   = lite->ask_volume;
                price = lite->ask_price;
            }
            else {
                vol   = lite->bid_volume;
                price = lite->bid_price;
            }
        }
        else if (sign > 0 && lite->ask_price <= order->limit_price) {
            vol   = lite->ask_volume;
            price = lite->ask_price;
        }
        else if (sign < 0 && lite->bid_price >= order->limit_price) {
            vol   = lite->bid_volume;
            price = lite->bid_price;
        }

        if (vol > 0) c_smm_try_match(ctx, entry, vol, price);
    }
}

/**
 * @brief Matching scan for order data.
 * @param ctx        Matcher context.
 * @param order_data OrderData payload.
 */
static inline void c_smm_check_order(smm_match_ctx* ctx, const md_order_data* order_data) {
    if (!ctx || !order_data) return;

    smm_order_ring_cursor cursor = c_smm_ring_cursor_newest(&ctx->orders);
    smm_order_entry*      entry  = NULL;
    while ((entry = c_smm_ring_next_newest(&cursor))) {
        if (!c_smm_order_eligible(ctx, entry, order_data->meta_info.dt.unix_ts)) continue;

        md_trade_instruction* order = entry->order;
        int8_t                sign  = c_md_side_sign(order->side);
        int8_t                taker = c_md_side_sign(order_data->side);
        if (sign == 0) continue;

        double vol = 0.0;
        if (!c_smm_has_limit_price(order)) {
            if (sign * taker < 0) vol = order_data->volume;
        }
        else if (sign > 0 && order_data->price <= order->limit_price) {
            vol = order_data->volume;
        }
        else if (sign < 0 && order_data->price >= order->limit_price) {
            vol = order_data->volume;
        }

        if (vol > 0) c_smm_try_match(ctx, entry, vol, order_data->price);
    }
}

/**
 * @brief Matching scan for transaction / trade data.
 * @param ctx         Matcher context.
 * @param transaction TransactionData payload.
 */
static inline void c_smm_check_trade(smm_match_ctx* ctx, const md_transaction_data* transaction) {
    if (!ctx || !transaction) return;

    smm_order_ring_cursor cursor = c_smm_ring_cursor_newest(&ctx->orders);
    smm_order_entry*      entry  = NULL;
    while ((entry = c_smm_ring_next_newest(&cursor))) {
        if (!c_smm_order_eligible(ctx, entry, transaction->meta_info.dt.unix_ts)) continue;

        md_trade_instruction* order = entry->order;
        int8_t                sign  = c_md_side_sign(order->side);
        if (sign == 0) continue;

        if (!c_smm_has_limit_price(order)) {
            if (sign * c_md_side_sign(transaction->side) > 0) c_smm_try_match(ctx, entry, transaction->volume, transaction->price);
        }
        else if (sign > 0 && transaction->price < order->limit_price) {
            c_smm_try_match(ctx, entry, transaction->volume, transaction->price);
        }
        else if (sign < 0 && transaction->price > order->limit_price) {
            c_smm_try_match(ctx, entry, transaction->volume, transaction->price);
        }
    }
}

#endif /* C_SIMMATCH_H */

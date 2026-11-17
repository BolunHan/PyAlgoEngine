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

/** @brief Bytemap key width of a long id (id type + id bytes, no padding). */
#define SMM_ID_KEY_SIZE (sizeof(md_id_type) + LONG_ID_SIZE)

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
    SMM_ERR_DUPLICATE      = -2,  // The order id is already in the working registry or the history.
    SMM_ERR_NOT_FOUND      = -3,  // The order id is not in the working registry.
    SMM_ERR_NO_VALID_PRICE = -4,
    SMM_ERR_INVALID_SIDE   = -5,
    SMM_ERR_OOM            = -6,
} smm_ret_code;

// clang-format on

// ========== Structs ==========

typedef struct smm_match_ctx   smm_match_ctx;
typedef struct smm_order_entry smm_order_entry;
typedef struct smm_listener    smm_listener;

/** @brief Fire-and-forget match / cancel notification; `user_data` is the registration value. */
typedef void (*smm_match_callback)(smm_match_ctx* ctx, smm_match_event event, const smm_order_entry* entry, const md_trade_report* report, void* user_data);

/**
 * @struct smm_order_entry
 * @brief One registry-owned order.
 *
 * `order` is the registry's copy of the launched instruction and sits at
 * offset 0, so the entry address is also the instruction address. Every fill
 * and state transition updates this copy, never the caller's instruction.
 * `reports` owns a copy of every applied report, so the complete fill history
 * is recoverable from the entry alone.
 *
 * `newer` / `older` link the live working orders newest-first. A completed
 * order keeps its links: a running scan may still hold the entry, and the
 * entry outlives its removal in the history registry.
 */
struct smm_order_entry {
    md_trade_instruction order;  // The registry's own copy of the launched instruction.
    uint64_t             transaction_count_at_launch;
    bytemap*             reports;  // OWNED - trade id -> owned md_trade_report copy; NULL until the first fill.
    smm_order_entry*     newer;
    smm_order_entry*     older;
};

_Static_assert(offsetof(smm_order_entry, order) == 0, "a smm_order_entry is addressable as its instruction");

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
 *
 * The registries are bytemaps of owned entries: `working` holds the live
 * orders, `history` holds every completed order until clear. Both are exposed
 * to Python as read-only views, so the maps are the single source of truth.
 */
struct smm_match_ctx {
    // === market state ===
    double           timestamp;               // Last market-data timestamp.
    double           last_price;              // Last market price, NAN if unset.
    uint64_t         last_transaction_count;  // Transaction counter.

    // === registry ===
    bytemap*         working;  // OWNED - order id -> owned smm_order_entry*.
    bytemap*         history;  // OWNED - completed order id -> owned smm_order_entry*.
    smm_order_entry* newest;   // Newest working entry (NULL when idle).
    smm_order_entry* oldest;   // Oldest working entry (NULL when idle).

    // === matching config ===
    double           fee_rate;           // Fee as a fraction of notional.
    double           lag_ts;             // Minimum elapsed time (s) before a fill.
    uint64_t         lag_n_transaction;  // Minimum transaction count before a fill.
    double           hit_prob;           // Probability of a fill attempt succeeding.
    double           slippery_rate;      // Slippage as a fraction of price.

    // === random state ===
    uint64_t         seed;       // Configured seed (0 = entropy-derived).
    uint64_t         rng_state;  // xorshift64* state.

    // === listeners ===
    smm_listener*    listeners;  // Linked list of notification listeners.

    // === flags ===
    bool             instant_fill;              // Fill immediately on launch when no lag is set.
    bool             incremental_order_volume;  // Order data volume is incremental (SH style); when false,
                                                // order data is not matched (SZ depth reports carry resting volume).
};

// ========== Forward Declarations ==========

// --- Utility ---
static inline uint64_t         c_smm_pid(void);
static inline void             c_smm_id_key(const long_md_id* id, char* out);
static inline bool             c_smm_has_limit_price(const md_trade_instruction* order);
static inline void             c_smm_rng_seed(smm_match_ctx* ctx, uint64_t seed);
static inline uint64_t         c_smm_rng_next_raw(smm_match_ctx* ctx);
static inline double           c_smm_rng_next(smm_match_ctx* ctx);
static inline void             c_smm_gen_uuid(smm_match_ctx* ctx, long_md_id* out);
static inline bool             c_smm_lag_allowed(const smm_match_ctx* ctx, const smm_order_entry* entry);
static inline bool             c_smm_hit_ok(smm_match_ctx* ctx);
static inline double           c_smm_apply_slippage(const smm_match_ctx* ctx, double price, md_side side);
static inline bool             c_smm_short_circuit(const smm_match_ctx* ctx, const md_trade_instruction* order);
static inline bool             c_smm_order_eligible(const smm_match_ctx* ctx, const smm_order_entry* entry, double timestamp);

// --- Registry ---
static inline smm_order_entry* c_smm_entry_new(allocator_protocol* allocator, const md_trade_instruction* order);
static inline void             c_smm_entry_dealloc(smm_order_entry* entry);
static inline int              c_smm_entry_report_add(smm_order_entry* entry, md_trade_report* report);
static inline void             c_smm_entries_clear(bytemap* registry);
static inline smm_order_entry* c_smm_working_find(const smm_match_ctx* ctx, const long_md_id* order_id);
static inline smm_order_entry* c_smm_working_push(smm_match_ctx* ctx, const md_trade_instruction* order);
static inline void             c_smm_working_remove(smm_match_ctx* ctx, smm_order_entry* entry);
static inline void             c_smm_working_touch(smm_match_ctx* ctx, smm_order_entry* entry);
static inline void             c_smm_working_clear(smm_match_ctx* ctx);
static inline smm_order_entry* c_smm_history_find(const smm_match_ctx* ctx, const long_md_id* order_id);
static inline int              c_smm_history_insert(smm_match_ctx* ctx, smm_order_entry* entry);
static inline void             c_smm_history_clear(smm_match_ctx* ctx);

// --- Lifecycle ---
static inline smm_match_ctx*   c_smm_match_new(double fee_rate, bool instant_fill, double lag_ts, uint64_t lag_n_transaction, double hit_prob, double slippery_rate, bool incremental_order_volume, allocator_protocol* allocator);
static inline int              c_smm_match_init(smm_match_ctx* ctx);
static inline void             c_smm_match_dealloc(smm_match_ctx* ctx);
static inline void             c_smm_match_free(smm_match_ctx* ctx);
static inline void             c_smm_match_clear(smm_match_ctx* ctx);

// --- Configuration ---
static inline int              c_smm_match_set_config(smm_match_ctx* ctx, double fee_rate, bool instant_fill, double lag_ts, uint64_t lag_n_transaction, double hit_prob, double slippery_rate, bool incremental_order_volume);
static inline int              c_smm_match_set_seed(smm_match_ctx* ctx, uint64_t seed);

// --- Listeners ---
static inline int              c_smm_match_register_listener(smm_match_ctx* ctx, smm_match_callback callback, void* user_data, uintptr_t* out_id);
static inline int              c_smm_match_deregister_listener(smm_match_ctx* ctx, uintptr_t listener_id);

// --- Queries ---
static inline int              c_smm_match_best_price(const double* prices, size_t n, int8_t sign, double* out);
static inline int              c_smm_match_worst_price(const double* prices, size_t n, int8_t sign, double* out);

// --- Trading ---
static inline int              c_smm_match_launch(smm_match_ctx* ctx, const md_trade_instruction* order);
static inline int              c_smm_match_cancel(smm_match_ctx* ctx, const long_md_id* order_id);
static inline void             c_smm_match_eod(smm_match_ctx* ctx);

// --- Hot path ---
// Runs once per market-data event.
static inline int              c_smm_match_process(smm_match_ctx* ctx, const md_variant* market_data);

// --- Matching internals ---
static inline void             c_smm_try_match(smm_match_ctx* ctx, smm_order_entry* entry, double match_volume, double match_price);
static inline void             c_smm_check_bar(smm_match_ctx* ctx, const md_candlestick* bar);
static inline void             c_smm_check_tick(smm_match_ctx* ctx, const md_tick_data* tick);
static inline void             c_smm_check_tick_lite(smm_match_ctx* ctx, const md_tick_data_lite* lite);
static inline void             c_smm_check_order(smm_match_ctx* ctx, const md_order_data* order_data);
static inline void             c_smm_check_trade(smm_match_ctx* ctx, const md_transaction_data* transaction);

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
    if (!ctx || !entry) return true;
    if (ctx->lag_ts <= 0 && ctx->lag_n_transaction == 0u) return true;

    double time_elapsed = ctx->timestamp - entry->order.meta_info.dt.unix_ts;
    if (ctx->lag_ts > 0 && time_elapsed < ctx->lag_ts) return false;

    uint64_t transactions_since = ctx->last_transaction_count >= entry->transaction_count_at_launch
                                      ? ctx->last_transaction_count - entry->transaction_count_at_launch
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
    if (!ctx || !entry) return false;
    if (!c_md_state_working(entry->order.order_state)) return false;
    if (entry->order.meta_info.dt.unix_ts > timestamp) return false;
    return true;
}

// ========== Registry ==========

/**
 * @brief Allocate an entry holding the registry's copy of an order.
 * @param allocator Allocator for the entry block.
 * @param order     Order instruction to copy.
 * @return The new entry, or NULL on OOM.
 */
static inline smm_order_entry* c_smm_entry_new(allocator_protocol* allocator, const md_trade_instruction* order) {
    if (!allocator || !order) return NULL;

    smm_order_entry* entry = (smm_order_entry*) c_ap_alloc(sizeof(smm_order_entry), allocator);
    if (!entry) return NULL;

    entry->order = *order;
    return entry;
}

/**
 * @brief Free one entry block and its report copies.
 * @param entry Entry to free (NULL-safe).
 */
static inline void c_smm_entry_dealloc(smm_order_entry* entry) {
    if (!entry) return;

    if (entry->reports) {
        bytemap_entry* node = entry->reports->first;
        while (node) {
            md_trade_report* report = *(md_trade_report**) node->value;
            node                    = node->next;
            if (report) c_ap_free(report);
        }
        c_bytemap_free(entry->reports);
    }
    c_ap_free(entry);
}

/**
 * @brief Store an owned copy of one applied report on the entry.
 *
 * The report map is created on the first fill, so an order that never fills
 * pays no map allocation. The registry takes ownership of `report` only when
 * this returns SMM_OK.
 * @param entry  Order entry.
 * @param report Owned report copy.
 * @return SMM_OK or SMM_ERR_OOM.
 */
static inline int c_smm_entry_report_add(smm_order_entry* entry, md_trade_report* report) {
    if (!entry || !report) return SMM_ERR_INVALID_ARG;

    if (!entry->reports) {
        allocator_protocol* allocator = c_ap_protocol_from_ptr((const void*) entry);
        entry->reports                = c_bytemap_ex_new(0u, sizeof(md_trade_report*), allocator);
        if (!entry->reports) return SMM_ERR_OOM;
    }

    char key[SMM_ID_KEY_SIZE];
    c_smm_id_key(&report->trade_id, key);
    if (c_bytemap_set(entry->reports, key, SMM_ID_KEY_SIZE, (void*) report, NULL) != BYTEMAP_OK) return SMM_ERR_OOM;
    return SMM_OK;
}

/**
 * @brief Free every entry of a registry and clear its map.
 *
 * The map keeps its allocated capacity; the clear fires the bytemap CLEARED
 * notification, which is how a bound Python view drops its cached entries.
 * @param registry Registry map (NULL-safe).
 */
static inline void c_smm_entries_clear(bytemap* registry) {
    if (!registry) return;

    bytemap_entry* node = registry->first;
    while (node) {
        smm_order_entry* entry = *(smm_order_entry**) node->value;
        node                   = node->next;
        c_smm_entry_dealloc(entry);
    }
    c_bytemap_clear(registry);
}

/**
 * @brief Find a live working entry by order id.
 * @param ctx      Matcher context.
 * @param order_id Order id to look up.
 * @return The entry, or NULL.
 */
static inline smm_order_entry* c_smm_working_find(const smm_match_ctx* ctx, const long_md_id* order_id) {
    if (!ctx || !ctx->working || !order_id) return NULL;

    void* value = NULL;
    char  key[SMM_ID_KEY_SIZE];
    c_smm_id_key(order_id, key);
    if (c_bytemap_get(ctx->working, key, SMM_ID_KEY_SIZE, &value) != BYTEMAP_OK) return NULL;
    return (smm_order_entry*) value;
}

/**
 * @brief Append a copy of one order to the working registry.
 *
 * The entry owns the copied instruction; the map and the newest-first list
 * both take it, and the bytemap notifies bound views of the insertion.
 * @param ctx   Matcher context.
 * @param order Order instruction to copy into the registry.
 * @return The new entry, or NULL on OOM.
 */
static inline smm_order_entry* c_smm_working_push(smm_match_ctx* ctx, const md_trade_instruction* order) {
    if (!ctx || !ctx->working || !order) return NULL;

    allocator_protocol* allocator = c_ap_protocol_from_ptr((const void*) ctx);
    smm_order_entry*    entry     = c_smm_entry_new(allocator, order);
    if (!entry) return NULL;

    char key[SMM_ID_KEY_SIZE];
    c_smm_id_key(&entry->order.order_id, key);
    if (c_bytemap_set(ctx->working, key, SMM_ID_KEY_SIZE, (void*) entry, NULL) != BYTEMAP_OK) {
        c_smm_entry_dealloc(entry);
        return NULL;
    }

    entry->older = ctx->newest;
    if (ctx->newest) ctx->newest->newer = entry;
    else ctx->oldest = entry;
    ctx->newest = entry;
    return entry;
}

/**
 * @brief Remove one working entry from the registry.
 *
 * The entry leaves the map and the list but keeps its block and its links: a
 * scan cursor may hold it, and the caller moves it to the history registry.
 * @param ctx   Matcher context.
 * @param entry Working entry to remove.
 */
static inline void c_smm_working_remove(smm_match_ctx* ctx, smm_order_entry* entry) {
    if (!ctx || !ctx->working || !entry) return;

    if (entry->older) entry->older->newer = entry->newer;
    else ctx->oldest = entry->newer;

    if (entry->newer) entry->newer->older = entry->older;
    else ctx->newest = entry->older;

    char key[SMM_ID_KEY_SIZE];
    c_smm_id_key(&entry->order.order_id, key);
    c_bytemap_pop(ctx->working, key, SMM_ID_KEY_SIZE, NULL);
}

/**
 * @brief Re-publish a working entry so bound views re-read its record.
 *
 * A partial fill mutates the entry in place, which the registry map cannot
 * see; rewriting the same value fires the bytemap MODIFIED notification and
 * every bound view refreshes its cached order record.
 * @param ctx   Matcher context.
 * @param entry Working entry that changed.
 */
static inline void c_smm_working_touch(smm_match_ctx* ctx, smm_order_entry* entry) {
    if (!ctx || !ctx->working || !entry) return;

    char key[SMM_ID_KEY_SIZE];
    c_smm_id_key(&entry->order.order_id, key);
    c_bytemap_set(ctx->working, key, SMM_ID_KEY_SIZE, (void*) entry, NULL);
}

/**
 * @brief Free every working entry, leaving the registry empty.
 * @param ctx Matcher context (NULL-safe).
 */
static inline void c_smm_working_clear(smm_match_ctx* ctx) {
    if (!ctx) return;

    c_smm_entries_clear(ctx->working);
    ctx->newest = NULL;
    ctx->oldest = NULL;
}

/**
 * @brief Find a completed entry by order id.
 * @param ctx      Matcher context.
 * @param order_id Order id to look up.
 * @return The entry, or NULL.
 */
static inline smm_order_entry* c_smm_history_find(const smm_match_ctx* ctx, const long_md_id* order_id) {
    if (!ctx || !ctx->history || !order_id) return NULL;

    void* value = NULL;
    char  key[SMM_ID_KEY_SIZE];
    c_smm_id_key(order_id, key);
    if (c_bytemap_get(ctx->history, key, SMM_ID_KEY_SIZE, &value) != BYTEMAP_OK) return NULL;
    return (smm_order_entry*) value;
}

/**
 * @brief Move one completed entry into the history registry.
 * @param ctx   Matcher context.
 * @param entry Completed entry; the registry takes ownership.
 * @return SMM_OK or SMM_ERR_OOM.
 */
static inline int c_smm_history_insert(smm_match_ctx* ctx, smm_order_entry* entry) {
    if (!ctx || !ctx->history || !entry) return SMM_ERR_INVALID_ARG;

    char key[SMM_ID_KEY_SIZE];
    c_smm_id_key(&entry->order.order_id, key);
    if (c_bytemap_set(ctx->history, key, SMM_ID_KEY_SIZE, (void*) entry, NULL) != BYTEMAP_OK) return SMM_ERR_OOM;
    return SMM_OK;
}

/**
 * @brief Free every completed entry, leaving the history empty.
 * @param ctx Matcher context (NULL-safe).
 */
static inline void c_smm_history_clear(smm_match_ctx* ctx) {
    if (!ctx) return;
    c_smm_entries_clear(ctx->history);
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
 * @param entry  Working or completed order entry.
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
 * The block must be an allocator-protocol block start because the registries
 * derive the allocator with `c_ap_protocol_from_ptr`.
 *
 * Defaults: fee_rate 0, instant_fill false, lag 0, hit_prob 1, slippery 1e-4,
 * incremental_order_volume false, PRNG seeded from time and pid.
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

    ctx->working = c_bytemap_new(0u, allocator);
    if (!ctx->working) return SMM_ERR_OOM;

    ctx->history = c_bytemap_new(0u, allocator);
    if (!ctx->history) {
        c_bytemap_free(ctx->working);
        ctx->working = NULL;
        return SMM_ERR_OOM;
    }
    return SMM_OK;
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
 * The context block itself is not freed; the registries, the report workspace
 * and every listener node are released. Freeing a registry fires the bytemap
 * FREED notification, which husks a bound view.
 * @param ctx Matcher context (NULL-safe).
 */
static inline void c_smm_match_dealloc(smm_match_ctx* ctx) {
    if (!ctx) return;

    c_smm_working_clear(ctx);
    c_smm_history_clear(ctx);

    if (ctx->working) c_bytemap_free(ctx->working);
    if (ctx->history) c_bytemap_free(ctx->history);

    smm_listener* listener = ctx->listeners;
    while (listener) {
        smm_listener* next = listener->next;
        free(listener);
        listener = next;
    }

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
 * @brief Reset market state and the registries, keeping config, listeners, the
 *        report workspace and the allocated map capacity.
 *
 * Every entry is dropped; timestamp, last price and counters are reset; the
 * PRNG is re-seeded from the configured seed.
 * @param ctx Matcher context (NULL-safe).
 */
static inline void c_smm_match_clear(smm_match_ctx* ctx) {
    if (!ctx) return;

    c_smm_working_clear(ctx);
    c_smm_history_clear(ctx);

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
    if (c_smm_working_find(ctx, &order->order_id) || c_smm_history_find(ctx, &order->order_id)) return SMM_ERR_DUPLICATE;

    smm_order_entry* entry = c_smm_working_push(ctx, order);
    if (!entry) return SMM_ERR_OOM;

    entry->order.order_state           = STATE_PLACED;
    entry->order.ts_placed             = ctx->timestamp;
    entry->transaction_count_at_launch = ctx->last_transaction_count;

    c_smm_match_fire(ctx, SMM_EVENT_PLACED, entry, NULL);

    if (!c_smm_short_circuit(ctx, order)) return SMM_OK;

    // A listener may have canceled the order, so resolve the entry again.
    entry = c_smm_working_find(ctx, &order->order_id);
    if (!entry) return SMM_OK;

    double worst;
    if (!c_smm_has_limit_price(&entry->order)) {
        worst = ctx->last_price;
    }
    else {
        int8_t sign = c_md_side_sign(entry->order.side);
        worst       = (sign > 0) ? fmax(entry->order.limit_price, ctx->last_price)
                                 : fmin(entry->order.limit_price, ctx->last_price);
    }
    c_smm_try_match(ctx, entry, NAN, worst);
    return SMM_OK;
}

/**
 * @brief Cancel a working order by id.
 *
 * Sets the order state to CANCELED (unless already FILLED), moves the entry to
 * the history registry and fires the SMM_EVENT_CANCEL listeners.
 * @param ctx      Matcher context.
 * @param order_id Order id to cancel.
 * @return SMM_OK, SMM_ERR_INVALID_ARG, SMM_ERR_NOT_FOUND or SMM_ERR_OOM.
 */
static inline int c_smm_match_cancel(smm_match_ctx* ctx, const long_md_id* order_id) {
    if (!ctx || !order_id) return SMM_ERR_INVALID_ARG;

    smm_order_entry* entry = c_smm_working_find(ctx, order_id);
    if (!entry) return SMM_ERR_NOT_FOUND;

    if (entry->order.order_state != STATE_FILLED) {
        entry->order.order_state = STATE_CANCELED;
        entry->order.ts_canceled = ctx->timestamp;
    }

    c_smm_working_remove(ctx, entry);
    if (c_smm_history_insert(ctx, entry) != SMM_OK) {
        c_smm_entry_dealloc(entry);
        return SMM_ERR_OOM;
    }

    c_smm_match_fire(ctx, SMM_EVENT_CANCEL, entry, NULL);
    return SMM_OK;
}

/**
 * @brief Cancel every working order (end of day).
 * @param ctx Matcher context (NULL-safe).
 */
static inline void c_smm_match_eod(smm_match_ctx* ctx) {
    if (!ctx) return;

    while (ctx->newest) {
        smm_order_entry* entry = ctx->newest;
        if (c_smm_match_cancel(ctx, &entry->order.order_id) != SMM_OK) break;
    }
}

/**
 * @brief Process one market-data update (hot path: runs once per market-data event).
 *
 * Updates timestamp / last_price / transaction count and runs the matching
 * scan for the data type. Fires SMM_EVENT_MATCH listeners for each fill; a
 * fill is the only branch that allocates.
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
 * builds an owned TradeReport, applies the fill to the registry's order record
 * and fires the SMM_EVENT_MATCH listeners. A fully filled order moves to the
 * history registry before the event fires, so a listener already sees the
 * completed state of both registries. A partial fill re-publishes the working
 * entry so bound views re-read its record.
 * @param ctx          Matcher context.
 * @param entry        Working order entry (moves to history on a full fill).
 * @param match_volume Desired volume; NAN = full working volume.
 * @param match_price  Desired price; NAN = the limit price.
 */
static inline void c_smm_try_match(smm_match_ctx* ctx, smm_order_entry* entry, double match_volume, double match_price) {
    if (!ctx || !entry) return;

    md_trade_instruction* order = &entry->order;
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

    allocator_protocol* allocator = c_ap_protocol_from_ptr((const void*) entry);
    md_trade_report*    rpt       = (md_trade_report*) c_ap_alloc(sizeof(md_trade_report), allocator);
    if (!rpt) return;

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

    if (c_smm_entry_report_add(entry, rpt) != SMM_OK) {
        c_ap_free(rpt);
        return;
    }

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

    if (order->order_state == STATE_FILLED) {
        c_smm_working_remove(ctx, entry);
        if (c_smm_history_insert(ctx, entry) != SMM_OK) {
            c_smm_entry_dealloc(entry);
            return;
        }
    }
    else {
        c_smm_working_touch(ctx, entry);
    }

    c_smm_match_fire(ctx, SMM_EVENT_MATCH, entry, rpt);
}

/**
 * @brief Matching scan for bar data.
 * @param ctx Matcher context.
 * @param bar BarData payload.
 */
static inline void c_smm_check_bar(smm_match_ctx* ctx, const md_candlestick* bar) {
    if (!ctx || !bar) return;

    smm_order_entry* entry = ctx->newest;
    while (entry) {
        smm_order_entry* next = entry->older;  // Snapshot: a fill may unlink the entry.
        if (!c_smm_order_eligible(ctx, entry, bar->meta_info.dt.unix_ts)) {
            entry = next;
            continue;
        }

        md_trade_instruction* order       = &entry->order;
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
        entry = next;
    }
}

/**
 * @brief Matching scan for tick data (order book).
 * @param ctx  Matcher context.
 * @param tick TickData payload.
 */
static inline void c_smm_check_tick(smm_match_ctx* ctx, const md_tick_data* tick) {
    if (!ctx || !tick) return;

    smm_order_entry* entry = ctx->newest;
    while (entry) {
        smm_order_entry* next = entry->older;
        if (!c_smm_order_eligible(ctx, entry, tick->lite.meta_info.dt.unix_ts)) {
            entry = next;
            continue;
        }

        md_trade_instruction* order = &entry->order;
        int8_t                sign  = c_md_side_sign(order->side);
        if (sign == 0) {
            entry = next;
            continue;
        }

        bool                has_limit = c_smm_has_limit_price(order);
        double              limit     = order->limit_price;
        const md_orderbook* book      = (sign > 0) ? tick->ask : tick->bid;
        bool                enter     = !has_limit;

        if (has_limit) {
            if (book->size > 0) enter = (sign > 0) ? (book->entries[0].price <= limit) : (book->entries[0].price >= limit);
        }

        if (!enter || book->size == 0) {
            entry = next;
            continue;
        }

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
        entry = next;
    }
}

/**
 * @brief Matching scan for tick-lite data.
 * @param ctx  Matcher context.
 * @param lite TickDataLite payload.
 */
static inline void c_smm_check_tick_lite(smm_match_ctx* ctx, const md_tick_data_lite* lite) {
    if (!ctx || !lite) return;

    smm_order_entry* entry = ctx->newest;
    while (entry) {
        smm_order_entry* next = entry->older;
        if (!c_smm_order_eligible(ctx, entry, lite->meta_info.dt.unix_ts)) {
            entry = next;
            continue;
        }

        md_trade_instruction* order = &entry->order;
        int8_t                sign  = c_md_side_sign(order->side);
        if (sign == 0) {
            entry = next;
            continue;
        }

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
        entry = next;
    }
}

/**
 * @brief Matching scan for order data.
 * @param ctx        Matcher context.
 * @param order_data OrderData payload.
 */
static inline void c_smm_check_order(smm_match_ctx* ctx, const md_order_data* order_data) {
    if (!ctx || !order_data) return;

    smm_order_entry* entry = ctx->newest;
    while (entry) {
        smm_order_entry* next = entry->older;
        if (!c_smm_order_eligible(ctx, entry, order_data->meta_info.dt.unix_ts)) {
            entry = next;
            continue;
        }

        md_trade_instruction* order = &entry->order;
        int8_t                sign  = c_md_side_sign(order->side);
        int8_t                taker = c_md_side_sign(order_data->side);
        if (sign == 0) {
            entry = next;
            continue;
        }

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
        entry = next;
    }
}

/**
 * @brief Matching scan for transaction / trade data.
 * @param ctx         Matcher context.
 * @param transaction TransactionData payload.
 */
static inline void c_smm_check_trade(smm_match_ctx* ctx, const md_transaction_data* transaction) {
    if (!ctx || !transaction) return;

    smm_order_entry* entry = ctx->newest;
    while (entry) {
        smm_order_entry* next = entry->older;
        if (!c_smm_order_eligible(ctx, entry, transaction->meta_info.dt.unix_ts)) {
            entry = next;
            continue;
        }

        md_trade_instruction* order = &entry->order;
        int8_t                sign  = c_md_side_sign(order->side);
        if (sign == 0) {
            entry = next;
            continue;
        }

        if (!c_smm_has_limit_price(order)) {
            if (sign * c_md_side_sign(transaction->side) > 0) c_smm_try_match(ctx, entry, transaction->volume, transaction->price);
        }
        else if (sign > 0 && transaction->price < order->limit_price) {
            c_smm_try_match(ctx, entry, transaction->volume, transaction->price);
        }
        else if (sign < 0 && transaction->price > order->limit_price) {
            c_smm_try_match(ctx, entry, transaction->volume, transaction->price);
        }
        entry = next;
    }
}

#endif /* C_SIMMATCH_H */

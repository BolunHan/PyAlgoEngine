#ifndef C_SIMMATCH_EX_H
#define C_SIMMATCH_EX_H

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(_WIN32) || defined(_WIN64)
#include <process.h>
#else
#include <unistd.h>
#endif

#include <Python.h>

#include <cbase/allocator_protocol/c_allocator_protocol.h>
#include <event_engine/capi/c_engine.h>
#include <event_engine/capi/c_event.h>
#include <event_engine/capi/c_event_pypayload.h>
#include <event_engine/capi/c_topic.h>

#include <algo_engine/base/c_market_data/c_market_data.h>
#include <algo_engine/exchange_profile/c_ex_profile_base.h>

// ========== Constants ==========

#ifndef SMM_UUID_VERSION_NIBBLE
#define SMM_UUID_VERSION_NIBBLE 0x40U
#endif

#ifndef SMM_RNG_MULTIPLIER
#define SMM_RNG_MULTIPLIER 0x2545F4914F6CDD1DULL
#endif

// ========== Enums ==========

// clang-format off

typedef enum smm_match_event {
    SMM_EVENT_MATCH        = 0,  // Order filled (report is valid)
    SMM_EVENT_CANCEL       = 1,  // Order canceled (report is NULL)
    SMM_EVENT_PLACED       = 2   // Order placed in the registry (report is NULL)
} smm_match_event;

typedef enum smm_ret_code {
    SMM_OK                 =  0,
    SMM_ERR_INVALID_ARG    = -1,
    SMM_ERR_DUPLICATE      = -2,  // order_id already in the working registry
    SMM_ERR_NOT_FOUND      = -3,  // order_id not in the working registry
    SMM_ERR_NO_VALID_PRICE = -4,
    SMM_ERR_INVALID_SIDE   = -5,
    SMM_ERR_OOM            = -6
} smm_ret_code;

// clang-format on

// ========== Forward Declarations ==========

typedef struct smm_order_entry    smm_order_entry;
typedef struct smm_match_ctx      smm_match_ctx;
typedef struct smm_listener       smm_listener;
typedef struct smm_engine_binding smm_engine_binding;

typedef void (*smm_match_callback)(smm_match_ctx* ctx, smm_match_event event, const smm_order_entry* entry, const md_trade_report* report, void* user_data);

// Forward declarations of matching internals (defined in Matching Internals).
static inline void c_smm_try_match(smm_match_ctx* ctx, smm_order_entry* entry, double match_volume, double match_price);
static inline void c_smm_check_bar(smm_match_ctx* ctx, const md_variant* md);
static inline void c_smm_check_tick(smm_match_ctx* ctx, const md_variant* md);
static inline void c_smm_check_tick_lite(smm_match_ctx* ctx, const md_variant* md);
static inline void c_smm_check_order(smm_match_ctx* ctx, const md_variant* md);
static inline void c_smm_check_trade(smm_match_ctx* ctx, const md_variant* md);

// Forward declarations of the public APIs used by the engine bridge.
static inline int c_smm_match_register(smm_match_ctx* ctx, const smm_engine_binding* binding);
static inline int c_smm_match_unregister(smm_match_ctx* ctx);
static inline int c_smm_match_launch(smm_match_ctx* ctx, md_variant* order_header, PyObject* py_order);
static inline int c_smm_match_cancel(smm_match_ctx* ctx, const long_md_id* order_id);
static inline int c_smm_match_process(smm_match_ctx* ctx, const md_variant* market_data);
static inline int c_smm_match_set_config(smm_match_ctx* ctx, double fee_rate, bool instant_fill, double lag_ts, uint64_t lag_n_transaction, double hit_prob, double slippery_rate, bool incremental_order_volume);

// ========== Structs ==========

/**
 * @struct smm_order_entry
 * @brief Registry node for one working order.
 *
 * The header pointer is borrowed — the Python order object (py_order, when
 * present) keeps the TradeInstruction buffer alive for the lifetime of the
 * entry. Pure-C consumers pass NULL py_order and own the header lifetime
 * themselves.
 */
struct smm_order_entry {
    long_md_id       order_id;                        // < registry key (owned copy)
    md_variant*      header;                          // < NOT owned — TradeInstruction buffer
    PyObject*        py_order;                        // < OWNED — Python order object (may be NULL)
    uint64_t         transaction_count_at_placement;  // < snapshot at launch
    smm_order_entry* next;                            // < next node in the working linked list
};

/**
 * @struct smm_listener
 * @brief One registered match / cancel notification listener.
 *
 * Follows the cbase bytemap callback pattern: a linked list of callback
 * nodes, each carrying its own user_data and a unique id, so any number of
 * wrappers (simulators, viewers, monitors) can bind independently.
 */
struct smm_listener {
    smm_match_callback callback;  // < listener callback
    void*              user_data;
    uintptr_t          id;  // < unique listener id (address of the node)
    smm_listener*      next;
};

/**
 * @struct smm_engine_binding
 * @brief Native event engine interface bound to the matcher.
 *
 * All handles are borrowed (owned by the engine and the caller) except the
 * PyObject refs, which this module INCREFs on register and DECREFs on
 * unregister / free. Hooks must be EventHook instances created by the
 * Cython layer (the engine hook registry stores Python objects); callback
 * registration on them happens here in C.
 */
struct smm_engine_binding {
    // === engine handles ===
    message_queue* mq;                   // < engine message queue (publish target)
    bytemap*       exact_topic_hooks;    // < engine exact hook map
    bytemap*       generic_topic_hooks;  // < engine generic hook map

    // === bound hooks (launch / cancel / realtime) ===
    evt_topic* launch_topic;     // < borrowed topic header (also the bytemap key)
    evt_hook*  launch_hook;      // < borrowed hook
    PyObject*  launch_hook_obj;  // < OWNED — hook object stored in the engine map
    evt_topic* cancel_topic;
    evt_hook*  cancel_hook;
    PyObject*  cancel_hook_obj;
    evt_topic* realtime_topic;
    evt_hook*  realtime_hook;
    PyObject*  realtime_hook_obj;

    // === publish topics (on_order / on_report) ===
    evt_topic* on_order_topic;      // < borrowed topic header
    PyObject*  on_order_topic_obj;  // < OWNED — Topic object for payloads
    evt_topic* on_report_topic;
    PyObject*  on_report_topic_obj;

    // === helpers ===
    PyObject* market_data_class;  // < OWNED — MarketData class (from_bytes wrapper)
};

/**
 * @struct smm_match_ctx
 * @brief Core state of the simulation matcher.
 *
 * Heap-allocated via c_smm_match_new (allocator protocol) or
 * caller-allocated via c_smm_match_init. Order entries and listener nodes
 * are short-lived bookkeeping allocations owned by the Python side and use
 * calloc. The engine binding is optional — without it the matcher works
 * purely through the listener interface.
 */
struct smm_match_ctx {
    // === allocation ===
    allocator_protocol* allocator;  // < allocator (derived lazily from the first md buffer)

    // === market state ===
    double   timestamp;               // < last market data timestamp
    double   last_price;              // < last market price, NAN if unset
    uint64_t last_transaction_count;  // < transaction counter

    // === random state ===
    uint64_t seed;       // < configured seed (0 = entropy-derived)
    uint64_t rng_state;  // < xorshift64* state

    // === matching config ===
    double   fee_rate;                  // < fee as fraction of notional
    bool     instant_fill;              // < fill immediately on launch when no lag
    double   lag_ts;                    // < minimum time (s) before fill
    uint64_t lag_n_transaction;         // < minimum transactions before fill
    double   hit_prob;                  // < probability of a fill attempt succeeding
    double   slippery_rate;             // < slippage as fraction of price
    bool     incremental_order_volume;  // < order data volume is incremental
                                        //   (SH style); when false, order data
                                        //   is not matched (SZ style depth
                                        //   reports carry resting volume)

    // === registry ===
    smm_order_entry* working;  // < linked list of working orders
    size_t           n_working;
    size_t           n_history;

    // === listeners ===
    smm_listener* listeners;  // < linked list of notification listeners

    // === engine binding ===
    smm_engine_binding* engine;  // < OWNED — native event engine interface (may be NULL)

    // === workspace ===
    md_variant ws_report;  // < workspace buffer for report construction
};

// ========== Utility Functions ==========

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
        uint64_t p = (uint64_t) getpid();
        seed = t ^ (p << 32u) ^ (p << 16u);
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
    uint64_t raw = c_smm_rng_next_raw(ctx);
    return (double) (raw >> 11u) * (1.0 / 9007199254740992.0);
}

/**
 * @brief Generate a UUIDv4 into an id buffer (bytes_le wire format).
 * @param ctx Matcher context (PRNG source).
 * @param out Target id buffer (md_id or long_md_id; 16 bytes are written).
 */
static inline void c_smm_gen_uuid(smm_match_ctx* ctx, long_md_id* out) {
    if (!ctx || !out) return;
    uint64_t a = c_smm_rng_next_raw(ctx);
    uint64_t b = c_smm_rng_next_raw(ctx);
    uint8_t  bytes[16];
    memcpy(bytes, &a, 8);
    memcpy(bytes + 8u, &b, 8);
    bytes[7] = (uint8_t) ((bytes[7] & 0x0Fu) | SMM_UUID_VERSION_NIBBLE);
    bytes[9] = (uint8_t) ((bytes[9] & 0x3Fu) | 0x80u);
    out->id_type = MID_UUID;
    memcpy(out->data, bytes, 16);
}

/**
 * @brief Resolve the ctx allocator, deriving it from the first market data
 *        buffer seen (the allocator protocol is embedded in every buffer).
 * @param ctx    Matcher context.
 * @param header Any market data / instruction header (may be NULL).
 * @return The ctx allocator (may still be NULL before any buffer is seen).
 */
static inline allocator_protocol* c_smm_allocator(smm_match_ctx* ctx, const md_variant* header) {
    if (!ctx) return NULL;
    if (!ctx->allocator && header) ctx->allocator = c_ap_protocol_from_ptr(header);
    return ctx->allocator;
}

/**
 * @brief Check whether lag constraints allow a fill.
 * @param ctx   Matcher context.
 * @param entry Working order entry.
 * @return true if the fill may proceed.
 */
static inline bool c_smm_lag_allowed(const smm_match_ctx* ctx, const smm_order_entry* entry) {
    if (!ctx || !entry || !entry->header) return true;
    if (ctx->lag_ts <= 0 && ctx->lag_n_transaction <= 0) return true;

    double time_elapsed = ctx->timestamp - entry->header->meta_info.timestamp;
    if (ctx->lag_ts > 0 && time_elapsed < ctx->lag_ts) return false;

    uint64_t transactions_since = ctx->last_transaction_count - entry->transaction_count_at_placement;
    if (ctx->lag_n_transaction > 0 && transactions_since < ctx->lag_n_transaction) return false;
    return true;
}

/**
 * @brief Roll the hit-probability dice.
 * @param ctx Matcher context.
 * @return true if the fill attempt succeeds.
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
 * @return Slipped price (buy: price * (1 + rate), sell: price * (1 - rate)).
 */
static inline double c_smm_apply_slippage(const smm_match_ctx* ctx, double price, md_side side) {
    if (!ctx) return price;
    double slippage = price * ctx->slippery_rate;
    int8_t sign = c_md_side_sign(side);
    if (sign > 0) return price + slippage;
    if (sign < 0) return price - slippage;
    return price;
}

/**
 * @brief Check the instant-fill short circuit.
 * @param ctx   Matcher context.
 * @param order Order instruction header.
 * @return true when the order should be matched immediately at launch.
 */
static inline bool c_smm_short_circuit(const smm_match_ctx* ctx, const md_trade_instruction* order) {
    if (!ctx || !order) return false;
    if (isnan(order->limit_price) && isnan(ctx->last_price)) return false;
    if (ctx->instant_fill && ctx->lag_ts == 0 && ctx->lag_n_transaction == 0) return true;
    return false;
}

/**
 * @brief Gate a working order for market-data processing.
 * @param ctx          Matcher context.
 * @param entry        Working order entry.
 * @param market_data  Incoming market data.
 * @return true when the order is eligible for matching.
 */
static inline bool c_smm_order_eligible(const smm_match_ctx* ctx, const smm_order_entry* entry, const md_variant* market_data) {
    if (!ctx || !entry || !entry->header || !market_data) return false;
    const md_trade_instruction* order = &entry->header->trade_instruction;
    if (!c_md_state_working(order->order_state)) return false;
    if (order->meta_info.timestamp > market_data->meta_info.timestamp) return false;
    return true;
}

/**
 * @brief Find a working entry by order_id.
 * @param ctx      Matcher context.
 * @param order_id Order id to look up.
 * @return Matching entry or NULL.
 */
static inline smm_order_entry* c_smm_find_entry(smm_match_ctx* ctx, const long_md_id* order_id) {
    if (!ctx || !order_id) return NULL;
    smm_order_entry* entry = ctx->working;
    while (entry) {
        if (c_md_long_id_equal(&entry->order_id, order_id)) return entry;
        entry = entry->next;
    }
    return NULL;
}

/**
 * @brief Unlink and free a working entry, bumping the history counter.
 * @param ctx    Matcher context.
 * @param target Entry to remove (must be in the working list).
 */
static inline void c_smm_remove_entry(smm_match_ctx* ctx, smm_order_entry* target) {
    if (!ctx || !target) return;
    smm_order_entry** link = &ctx->working;
    while (*link && *link != target) link = &(*link)->next;
    if (!*link) return;
    *link = target->next;
    ctx->n_working--;
    ctx->n_history++;
    Py_XDECREF(target->py_order);
    free(target);
}

// ========== Listener APIs ==========

/**
 * @brief Register a match / cancel notification listener.
 *
 * Multiple listeners may be registered; each receives every match and
 * cancel event with its own user_data. The listener id is the address of
 * the node and must be passed to c_smm_match_deregister_listener.
 * @param ctx       Matcher context.
 * @param callback  Listener callback.
 * @param user_data Opaque user data passed to the callback.
 * @param out_id    Receives the listener id (may be NULL).
 * @return SMM_OK, SMM_ERR_INVALID_ARG or SMM_ERR_OOM.
 */
static inline int c_smm_match_register_listener(smm_match_ctx* ctx, smm_match_callback callback, void* user_data, uintptr_t* out_id) {
    if (!ctx || !callback) return SMM_ERR_INVALID_ARG;

    smm_listener* node = (smm_listener*) calloc(1, sizeof(smm_listener));
    if (!node) return SMM_ERR_OOM;
    node->callback = callback;
    node->user_data = user_data;
    node->id = (uintptr_t) node;

    if (!ctx->listeners) {
        ctx->listeners = node;
    }
    else {
        smm_listener* tail = ctx->listeners;
        while (tail->next) tail = tail->next;
        tail->next = node;
    }

    if (out_id) *out_id = node->id;
    return SMM_OK;
}

/**
 * @brief Deregister a notification listener by id.
 * @param ctx         Matcher context.
 * @param listener_id Listener id returned by c_smm_match_register_listener.
 * @return SMM_OK or SMM_ERR_NOT_FOUND.
 */
static inline int c_smm_match_deregister_listener(smm_match_ctx* ctx, uintptr_t listener_id) {
    if (!ctx) return SMM_ERR_INVALID_ARG;

    smm_listener* prev = NULL;
    smm_listener* curr = ctx->listeners;
    while (curr) {
        if (curr->id == listener_id) {
            if (prev) prev->next = curr->next;
            else ctx->listeners = curr->next;
            free(curr);
            return SMM_OK;
        }
        prev = curr;
        curr = curr->next;
    }
    return SMM_ERR_NOT_FOUND;
}

/**
 * @brief Fire an event to every registered listener.
 *
 * The next pointer is snapshotted before each call, so listeners may
 * deregister themselves from inside the callback. The entry is valid for
 * the duration of the call (PLACED / MATCH fire while the entry is in the
 * registry; CANCEL fires before removal).
 * @param ctx    Matcher context.
 * @param event  Event type.
 * @param entry  Working order entry (valid during the call).
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

// ========== Event Engine Bridge ==========

/**
 * @brief Extract a market data header from a Python market data object via
 *        its readonly data_addr attribute.
 * @param obj Market data / instruction / report object.
 * @return The md_variant header, or NULL on failure (error cleared).
 */
static inline md_variant* c_smm_header_from_pyobject(PyObject* obj) {
    if (!obj) return NULL;
    PyObject* addr = PyObject_GetAttrString(obj, "data_addr");
    if (!addr) {
        PyErr_Clear();
        return NULL;
    }
    unsigned long long value = PyLong_AsUnsignedLongLong(addr);
    Py_DECREF(addr);
    if (value == (unsigned long long) -1 && PyErr_Occurred()) {
        PyErr_Clear();
        return NULL;
    }
    return (md_variant*) (uintptr_t) value;
}

/**
 * @brief Publish a key/value event on an engine topic.
 *
 * Builds an engine payload via c_evt_pypayload_new (the engine's own C API:
 * it injects the topic field, aggregates kwargs and sets fn_dealloc) and
 * puts it on the bound engine's message queue. The loop thread dispatches
 * it to the topic hook and frees the payload through fn_dealloc.
 * @param ctx          Matcher context.
 * @param topic_header Topic header for dispatch.
 * @param topic_obj    Topic object for payloads (must be a Topic instance).
 * @param key          Payload kwarg key (may be NULL).
 * @param value        Payload kwarg value (may be NULL when key is NULL).
 * @return SMM_OK, SMM_ERR_INVALID_ARG or SMM_ERR_OOM.
 */
static inline int c_smm_publish(smm_match_ctx* ctx, evt_topic* topic_header, PyObject* topic_obj, const char* key, PyObject* value) {
    if (!ctx || !ctx->engine || !topic_header || !topic_obj) return SMM_ERR_INVALID_ARG;
    smm_engine_binding* eng = ctx->engine;
    allocator_protocol* allocator = c_smm_allocator(ctx, NULL);
    if (!allocator) return SMM_ERR_OOM;

    PyObject* kwargs = PyDict_New();
    if (!kwargs) {
        PyErr_Clear();
        return SMM_ERR_OOM;
    }
    if (key && value) PyDict_SetItemString(kwargs, key, value);

    evt_message_payload* payload = c_evt_pypayload_new((evt_py_topic*) topic_obj, NULL, kwargs, allocator);
    Py_DECREF(kwargs);
    if (!payload) {
        PyErr_Clear();
        return SMM_ERR_OOM;
    }

    int ret = c_mq_put(eng->mq, payload);
    if (ret != 0) {
        payload->fn_dealloc(payload);
        return SMM_ERR_OOM;
    }
    return SMM_OK;
}

/**
 * @brief Publish a trade report on the bound engine's on_report topic.
 *
 * The report is serialized and re-wrapped into an owning MarketData object
 * so Python consumers receive an independent buffer.
 * @param ctx    Matcher context.
 * @param report Report variant to publish.
 * @return SMM_OK, SMM_ERR_INVALID_ARG or SMM_ERR_OOM.
 */
static inline int c_smm_publish_report(smm_match_ctx* ctx, const md_variant* report) {
    if (!ctx || !ctx->engine || !report) return SMM_ERR_INVALID_ARG;
    smm_engine_binding* eng = ctx->engine;

    size_t              size = c_md_serialized_size(report);
    char*               buf = (char*) PyMem_Malloc(size);
    if (!buf) return SMM_ERR_OOM;
    c_md_serialize(report, buf);
    PyObject* data = PyBytes_FromStringAndSize(buf, (Py_ssize_t) size);
    PyMem_Free(buf);
    if (!data) return SMM_ERR_OOM;

    PyObject* report_obj = PyObject_CallMethod(eng->market_data_class, "from_bytes", "(O)", data);
    Py_DECREF(data);
    if (!report_obj) {
        PyErr_Clear();
        return SMM_ERR_OOM;
    }
    int ret = c_smm_publish(ctx, eng->on_report_topic, eng->on_report_topic_obj, "report", report_obj);
    Py_DECREF(report_obj);
    return ret;
}

/**
 * @brief Check whether two topic headers carry the same key.
 * @param a Topic header (may be NULL).
 * @param b Topic header (may be NULL).
 * @return true when both are non-NULL and key-identical.
 */
static inline bool c_smm_topic_key_equal(const evt_topic* a, const evt_topic* b) {
    if (!a || !b) return false;
    return a->key_len == b->key_len && memcmp(a->key, b->key, a->key_len) == 0;
}

/**
 * @brief Engine hook handler: dispatch one engine message into the matcher.
 *
 * Registered on the launch / cancel / realtime hooks with user_data = ctx.
 * The dispatch is topic-based, mirroring the legacy registration semantics:
 * an order arriving on the launch topic is launched, on the cancel topic it
 * is canceled, and market data on the realtime topic is matched. Headers
 * are extracted via the data_addr attribute and the pipeline runs natively.
 * @param payload   Engine message payload.
 * @param user_data smm_match_ctx pointer.
 */
static inline void c_smm_evt_handler(evt_message_payload* payload, void* user_data) {
    smm_match_ctx* ctx = (smm_match_ctx*) user_data;
    if (!ctx || !ctx->engine || !payload || !payload->args || !payload->topic) return;
    smm_engine_binding* eng = ctx->engine;
    evt_py_payload*     py = (evt_py_payload*) payload->args;
    PyObject*           kwargs = py->py_kwargs;
    if (!kwargs || !PyDict_Check(kwargs)) return;

    if (c_smm_topic_key_equal(payload->topic, eng->cancel_topic)) {
        PyObject* order_obj = PyDict_GetItemString(kwargs, "order");  // borrowed
        if (order_obj) {
            md_variant* header = c_smm_header_from_pyobject(order_obj);
            if (header) c_smm_match_cancel(ctx, &header->trade_instruction.order_id);
        }
    }
    else if (c_smm_topic_key_equal(payload->topic, eng->launch_topic)) {
        PyObject* order_obj = PyDict_GetItemString(kwargs, "order");  // borrowed
        if (order_obj) {
            md_variant* header = c_smm_header_from_pyobject(order_obj);
            if (header) c_smm_match_launch(ctx, header, order_obj);
        }
    }
    else {
        PyObject* md_obj = PyDict_GetItemString(kwargs, "market_data");  // borrowed
        if (md_obj) {
            md_variant* header = c_smm_header_from_pyobject(md_obj);
            if (header) c_smm_match_process(ctx, header);
        }
    }
}

// ========== Public APIs ==========

/**
 * @brief Initialize a caller-allocated matcher context to defaults.
 *
 * Defaults: fee_rate 0, instant_fill false, lag 0, hit_prob 1, slippery 1e-4,
 * incremental_order_volume false.
 * @param ctx Matcher context.
 */
static inline void c_smm_match_init(smm_match_ctx* ctx) {
    if (!ctx) return;
    memset(ctx, 0, sizeof(smm_match_ctx));
    ctx->last_price = NAN;
    ctx->hit_prob = 1.0;
    ctx->slippery_rate = 0.0001;
    ctx->rng_state = SMM_RNG_MULTIPLIER;
}

/**
 * @brief Allocate and initialize a matcher context.
 * @param fee_rate                 Fee as fraction of notional.
 * @param instant_fill             Fill immediately at launch when no lag is set.
 * @param lag_ts                   Minimum elapsed time (s) before a fill.
 * @param lag_n_transaction        Minimum transaction count before a fill.
 * @param hit_prob                 Probability of a fill attempt succeeding.
 * @param slippery_rate            Slippage as fraction of price.
 * @param incremental_order_volume Match order data volume as incremental
 *                                 (SH style); when false order data is not
 *                                 matched (SZ style depth reports carry
 *                                 resting volume).
 * @param allocator                Allocator protocol; NULL derives lazily
 *                                 from the first market data buffer.
 * @return Heap-allocated context, or NULL on allocation failure.
 */
static inline smm_match_ctx* c_smm_match_new(double fee_rate, bool instant_fill, double lag_ts, uint64_t lag_n_transaction, double hit_prob, double slippery_rate, bool incremental_order_volume, allocator_protocol* allocator) {
    smm_match_ctx* ctx = (smm_match_ctx*) c_ap_alloc(sizeof(smm_match_ctx), allocator);
    if (!ctx) return NULL;
    c_smm_match_init(ctx);
    ctx->allocator = allocator;
    c_smm_match_set_config(ctx, fee_rate, instant_fill, lag_ts, lag_n_transaction, hit_prob, slippery_rate, incremental_order_volume);
    return ctx;
}

/**
 * @brief Release all owned resources of a matcher context.
 *
 * Does not free the context itself (caller-allocated usage).
 * @param ctx Matcher context.
 */
static inline void c_smm_match_dealloc(smm_match_ctx* ctx) {
    if (!ctx) return;
    smm_order_entry* entry = ctx->working;
    while (entry) {
        smm_order_entry* next = entry->next;
        Py_XDECREF(entry->py_order);
        free(entry);
        entry = next;
    }
    ctx->working = NULL;
    ctx->n_working = 0;
    ctx->n_history = 0;

    smm_listener* listener = ctx->listeners;
    while (listener) {
        smm_listener* next = listener->next;
        free(listener);
        listener = next;
    }
    ctx->listeners = NULL;

    if (ctx->engine) c_smm_match_unregister(ctx);
}

/**
 * @brief Free a matcher context.
 * @param ctx Matcher context.
 */
static inline void c_smm_match_free(smm_match_ctx* ctx) {
    if (!ctx) return;
    c_smm_match_dealloc(ctx);
    c_ap_free(ctx);
}

/**
 * @brief Overwrite the matching configuration.
 * @param ctx                      Matcher context.
 * @param fee_rate                 Fee as fraction of notional.
 * @param instant_fill             Fill immediately at launch when no lag is set.
 * @param lag_ts                   Minimum elapsed time (s) before a fill.
 * @param lag_n_transaction        Minimum transaction count before a fill.
 * @param hit_prob                 Probability of a fill attempt succeeding.
 * @param slippery_rate            Slippage as fraction of price.
 * @param incremental_order_volume Match order data volume as incremental
 *                                 (SH style); when false order data is not
 *                                 matched (SZ style depth reports carry
 *                                 resting volume).
 * @return SMM_OK or SMM_ERR_INVALID_ARG.
 */
static inline int c_smm_match_set_config(smm_match_ctx* ctx, double fee_rate, bool instant_fill, double lag_ts, uint64_t lag_n_transaction, double hit_prob, double slippery_rate, bool incremental_order_volume) {
    if (!ctx) return SMM_ERR_INVALID_ARG;
    ctx->fee_rate = fee_rate;
    ctx->instant_fill = instant_fill;
    ctx->lag_ts = lag_ts;
    ctx->lag_n_transaction = lag_n_transaction;
    ctx->hit_prob = hit_prob;
    ctx->slippery_rate = slippery_rate;
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
 * @brief Reset market state and registry but keep config, listeners and the
 *        engine binding.
 *
 * Equivalent of the Python clear(): working/history reset, timestamp and
 * counters zeroed, last_price unset, PRNG re-seeded.
 * @param ctx Matcher context.
 */
static inline void c_smm_match_clear(smm_match_ctx* ctx) {
    if (!ctx) return;
    double              fee_rate = ctx->fee_rate;
    bool                instant_fill = ctx->instant_fill;
    double              lag_ts = ctx->lag_ts;
    uint64_t            lag_n_transaction = ctx->lag_n_transaction;
    double              hit_prob = ctx->hit_prob;
    double              slippery_rate = ctx->slippery_rate;
    bool                incremental_order_volume = ctx->incremental_order_volume;
    uint64_t            seed = ctx->seed;
    allocator_protocol* allocator = ctx->allocator;
    smm_listener*       listeners = ctx->listeners;  // keep listener bindings
    smm_engine_binding* engine = ctx->engine;        // keep engine binding

    smm_order_entry*    entry = ctx->working;  // free entries only
    while (entry) {
        smm_order_entry* next = entry->next;
        Py_XDECREF(entry->py_order);
        free(entry);
        entry = next;
    }
    ctx->working = NULL;
    ctx->n_working = 0;
    ctx->n_history = 0;

    c_smm_match_init(ctx);
    ctx->allocator = allocator;
    ctx->listeners = listeners;
    ctx->engine = engine;
    c_smm_match_set_config(ctx, fee_rate, instant_fill, lag_ts, lag_n_transaction, hit_prob, slippery_rate, incremental_order_volume);
    c_smm_rng_seed(ctx, seed);
}

/**
 * @brief Register the matcher on a native event engine.
 *
 * Registers the C dispatch handler on the launch / cancel / realtime hooks
 * (c_evt_hook_register_callback) and stores the engine interface for native
 * on_order / on_report publishing. The hooks must be EventHook instances
 * created by the Cython layer; the engine hook maps must be the engine's
 * exact / generic bytemaps. The binding is copied into the context.
 * @param ctx     Matcher context.
 * @param binding Engine interface to bind (copied).
 * @return SMM_OK, SMM_ERR_INVALID_ARG or SMM_ERR_OOM.
 */
static inline int c_smm_match_register(smm_match_ctx* ctx, const smm_engine_binding* binding) {
    if (!ctx || !binding) return SMM_ERR_INVALID_ARG;
    if (ctx->engine) return SMM_ERR_DUPLICATE;
    if (!binding->mq || !binding->exact_topic_hooks || !binding->generic_topic_hooks) return SMM_ERR_INVALID_ARG;
    if (!binding->launch_topic || !binding->launch_hook || !binding->cancel_topic || !binding->cancel_hook || !binding->realtime_topic || !binding->realtime_hook) return SMM_ERR_INVALID_ARG;

    smm_engine_binding* eng = (smm_engine_binding*) calloc(1, sizeof(smm_engine_binding));
    if (!eng) return SMM_ERR_OOM;
    *eng = *binding;

    // store the hook objects in the engine maps (exact vs generic by key)
    evt_topic* topics[3] = {binding->launch_topic, binding->cancel_topic, binding->realtime_topic};
    PyObject*  hook_objs[3] = {binding->launch_hook_obj, binding->cancel_hook_obj, binding->realtime_hook_obj};
    for (size_t i = 0; i < 3; i++) {
        if (!topics[i] || !hook_objs[i]) {
            free(eng);
            return SMM_ERR_INVALID_ARG;
        }
        bytemap* map = topics[i]->is_exact ? eng->exact_topic_hooks : eng->generic_topic_hooks;
        c_bytemap_set(map, topics[i]->key, topics[i]->key_len, (void*) hook_objs[i], NULL);
        Py_INCREF(hook_objs[i]);  // the map holds a reference
    }

    // register the C dispatch handler on each hook
    evt_hook* hooks[3] = {binding->launch_hook, binding->cancel_hook, binding->realtime_hook};
    for (size_t i = 0; i < 3; i++) {
        if (c_evt_hook_register_callback(hooks[i], (const void*) c_smm_evt_handler, EVT_CALLBACK_WITH_PAYLOAD_USERDATA, ctx, 1) != EVT_RET_OK) {
            goto fail;
        }
    }

    // hold Python refs used for payload construction
    Py_XINCREF(eng->on_order_topic_obj);
    Py_XINCREF(eng->on_report_topic_obj);
    PyObject* md_module = PyImport_ImportModule("algo_engine.base.c_market_data.c_market_data");
    if (!md_module) {
        PyErr_Clear();
        goto fail;
    }
    eng->market_data_class = PyObject_GetAttrString(md_module, "MarketData");
    Py_DECREF(md_module);
    if (!eng->market_data_class) {
        PyErr_Clear();
        goto fail;
    }

    ctx->engine = eng;
    return SMM_OK;

fail:
    for (size_t i = 0; i < 3; i++) {
        bytemap*  map = topics[i]->is_exact ? eng->exact_topic_hooks : eng->generic_topic_hooks;
        PyObject* hook_obj = NULL;
        c_bytemap_pop(map, topics[i]->key, topics[i]->key_len, (void**) &hook_obj);
        Py_XDECREF(hook_obj);
    }
    Py_XDECREF(eng->on_order_topic_obj);
    Py_XDECREF(eng->on_report_topic_obj);
    Py_XDECREF(eng->market_data_class);
    free(eng);
    return SMM_ERR_OOM;
}

/**
 * @brief Unregister from the bound event engine.
 *
 * Pops the C dispatch handlers from the hooks, removes the hook objects
 * from the engine maps and releases all Python refs held by the binding.
 * @param ctx Matcher context.
 * @return SMM_OK or SMM_ERR_INVALID_ARG.
 */
static inline int c_smm_match_unregister(smm_match_ctx* ctx) {
    if (!ctx || !ctx->engine) return SMM_ERR_INVALID_ARG;
    smm_engine_binding* eng = ctx->engine;

    // pop the C dispatch handler from each hook
    evt_hook* hooks[3] = {eng->launch_hook, eng->cancel_hook, eng->realtime_hook};
    for (size_t h = 0; h < 3; h++) {
        evt_hook* hook = hooks[h];
        if (!hook) continue;
        for (size_t i = 0; i < hook->n_callbacks; i++) {
            evt_callback* cb = &hook->callbacks[i];
            if (cb->type == EVT_CALLBACK_WITH_PAYLOAD_USERDATA && cb->fn.with_payload_userdata == c_smm_evt_handler && cb->user_data == ctx) {
                c_evt_hook_pop_callback(hook, i);
                break;
            }
        }
    }

    // remove the hook objects from the engine maps
    evt_topic* topics[3] = {eng->launch_topic, eng->cancel_topic, eng->realtime_topic};
    for (size_t t = 0; t < 3; t++) {
        evt_topic* topic = topics[t];
        if (!topic) continue;
        bytemap*  map = topic->is_exact ? eng->exact_topic_hooks : eng->generic_topic_hooks;
        PyObject* hook_obj = NULL;
        c_bytemap_pop(map, topic->key, topic->key_len, (void**) &hook_obj);
        Py_XDECREF(hook_obj);
    }

    Py_XDECREF(eng->on_order_topic_obj);
    Py_XDECREF(eng->on_report_topic_obj);
    Py_XDECREF(eng->market_data_class);
    free(eng);
    ctx->engine = NULL;
    return SMM_OK;
}

/**
 * @brief Launch an order into the working registry.
 *
 * The order header is NOT owned and must stay alive until the entry is
 * removed; when py_order is given it is INCREF'd and keeps the buffer
 * alive. Publishes the placed on_order event on the bound engine.
 * @param ctx          Matcher context.
 * @param order_header TradeInstruction header to register.
 * @param py_order     Python order object (may be NULL in pure-C usage).
 * @return SMM_OK, SMM_ERR_INVALID_ARG, SMM_ERR_DUPLICATE or SMM_ERR_OOM.
 */
static inline int c_smm_match_launch(smm_match_ctx* ctx, md_variant* order_header, PyObject* py_order) {
    if (!ctx || !order_header) return SMM_ERR_INVALID_ARG;
    md_trade_instruction* order = &order_header->trade_instruction;
    if (c_smm_find_entry(ctx, &order->order_id)) return SMM_ERR_DUPLICATE;
    c_smm_allocator(ctx, order_header);

    order->order_state = STATE_PLACED;
    order->ts_placed = ctx->timestamp;

    smm_order_entry* entry = (smm_order_entry*) calloc(1, sizeof(smm_order_entry));
    if (!entry) return SMM_ERR_OOM;
    entry->order_id = order->order_id;
    entry->header = order_header;
    entry->py_order = py_order;
    Py_XINCREF(py_order);
    entry->transaction_count_at_placement = ctx->last_transaction_count;
    entry->next = ctx->working;
    ctx->working = entry;
    ctx->n_working++;

    c_smm_match_fire(ctx, SMM_EVENT_PLACED, entry, NULL);

    if (ctx->engine && entry->py_order) {
        c_smm_publish(ctx, ctx->engine->on_order_topic, ctx->engine->on_order_topic_obj, "order", entry->py_order);
    }

    if (c_smm_short_circuit(ctx, order)) {
        double worst;
        if (isnan(order->limit_price)) {
            worst = ctx->last_price;
        }
        else {
            int8_t sign = c_md_side_sign(order->side);
            worst = (sign > 0) ? fmax(order->limit_price, ctx->last_price)
                               : fmin(order->limit_price, ctx->last_price);
        }
        c_smm_try_match(ctx, entry, NAN, worst);
    }
    return SMM_OK;
}

/**
 * @brief Cancel a working order by id.
 *
 * Sets the order state to CANCELED (unless already FILLED), moves the entry
 * to history, fires the SMM_EVENT_CANCEL listeners and publishes the
 * canceled on_order event on the bound engine.
 * @param ctx      Matcher context.
 * @param order_id Order id to cancel.
 * @return SMM_OK, SMM_ERR_INVALID_ARG or SMM_ERR_NOT_FOUND.
 */
static inline int c_smm_match_cancel(smm_match_ctx* ctx, const long_md_id* order_id) {
    if (!ctx || !order_id) return SMM_ERR_INVALID_ARG;
    smm_order_entry* entry = c_smm_find_entry(ctx, order_id);
    if (!entry) return SMM_ERR_NOT_FOUND;

    md_trade_instruction* order = &entry->header->trade_instruction;
    PyObject*             py_order = entry->py_order;
    Py_XINCREF(py_order);
    if (order->order_state != STATE_FILLED) {
        order->order_state = STATE_CANCELED;
        order->ts_canceled = ctx->timestamp;
    }
    c_smm_match_fire(ctx, SMM_EVENT_CANCEL, entry, NULL);
    c_smm_remove_entry(ctx, entry);

    if (ctx->engine && py_order) {
        c_smm_publish(ctx, ctx->engine->on_order_topic, ctx->engine->on_order_topic_obj, "order", py_order);
    }
    Py_XDECREF(py_order);
    return SMM_OK;
}

/**
 * @brief Cancel every working order (end-of-day).
 * @param ctx Matcher context.
 */
static inline void c_smm_match_eod(smm_match_ctx* ctx) {
    if (!ctx) return;
    smm_order_entry* entry = ctx->working;
    while (entry) {
        smm_order_entry* next = entry->next;
        c_smm_match_cancel(ctx, &entry->order_id);
        entry = next;
    }
}

/**
 * @brief Process one market data update.
 *
 * Updates timestamp / last_price / transaction count and runs the matching
 * scan for the data type. Fires SMM_EVENT_MATCH listeners and publishes
 * on_report / on_order events on the bound engine for each fill.
 * @param ctx          Matcher context.
 * @param market_data  Market data header (bar / tick / tick-lite / order /
 *                     transaction; other dtypes only update market state).
 * @return SMM_OK or SMM_ERR_INVALID_ARG.
 */
static inline int c_smm_match_process(smm_match_ctx* ctx, const md_variant* market_data) {
    if (!ctx || !market_data) return SMM_ERR_INVALID_ARG;
    c_smm_allocator(ctx, market_data);

    ctx->timestamp = market_data->meta_info.timestamp;
    ctx->last_price = c_md_get_price(market_data);

    switch (market_data->meta_info.dtype) {
        case DTYPE_TRANSACTION:
            ctx->last_transaction_count += 1;
            c_smm_check_trade(ctx, market_data);
            break;
        case DTYPE_BAR:
            c_smm_check_bar(ctx, market_data);
            break;
        case DTYPE_TICK:
            c_smm_check_tick(ctx, market_data);
            break;
        case DTYPE_TICK_LITE:
            c_smm_check_tick_lite(ctx, market_data);
            break;
        case DTYPE_ORDER:
            // SZ-style depth reports carry resting volume, not incremental
            // fills — only match when incremental order volume is enabled.
            if (ctx->incremental_order_volume) c_smm_check_order(ctx, market_data);
            break;
        default:
            break;
    }
    return SMM_OK;
}

/**
 * @brief Best price for a side (buy: min, sell: max).
 * @param prices Price array; NAN entries are skipped.
 * @param n      Number of entries.
 * @param sign   Side sign (1 buy, -1 sell).
 * @param out    Receives the best price.
 * @return SMM_OK, SMM_ERR_INVALID_ARG, SMM_ERR_NO_VALID_PRICE or
 *         SMM_ERR_INVALID_SIDE.
 */
static inline int c_smm_match_best_price(const double* prices, size_t n, int8_t sign, double* out) {
    if (!prices || !out) return SMM_ERR_INVALID_ARG;
    double best = NAN;
    size_t count = 0u;
    for (size_t i = 0u; i < n; i++) {
        double p = prices[i];
        if (!isfinite(p)) continue;
        if (count == 0u) {
            best = p;
            count = 1u;
        }
        else if (sign > 0) {
            if (p < best) best = p;
        }
        else if (sign < 0) {
            if (p > best) best = p;
        }
        else {
            return SMM_ERR_INVALID_SIDE;
        }
    }
    if (count == 0u) return SMM_ERR_NO_VALID_PRICE;
    *out = best;
    return SMM_OK;
}

/**
 * @brief Worst price for a side (buy: max, sell: min).
 * @param prices Price array; NAN entries are skipped.
 * @param n      Number of entries.
 * @param sign   Side sign (1 buy, -1 sell).
 * @param out    Receives the worst price.
 * @return SMM_OK, SMM_ERR_INVALID_ARG, SMM_ERR_NO_VALID_PRICE or
 *         SMM_ERR_INVALID_SIDE.
 */
static inline int c_smm_match_worst_price(const double* prices, size_t n, int8_t sign, double* out) {
    if (!prices || !out) return SMM_ERR_INVALID_ARG;
    double worst = NAN;
    size_t count = 0u;
    for (size_t i = 0u; i < n; i++) {
        double p = prices[i];
        if (!isfinite(p)) continue;
        if (count == 0u) {
            worst = p;
            count = 1u;
        }
        else if (sign > 0) {
            if (p > worst) worst = p;
        }
        else if (sign < 0) {
            if (p < worst) worst = p;
        }
        else {
            return SMM_ERR_INVALID_SIDE;
        }
    }
    if (count == 0u) return SMM_ERR_NO_VALID_PRICE;
    *out = worst;
    return SMM_OK;
}

// ========== Matching Internals ==========

/**
 * @brief Attempt to match a working order at the given volume and price.
 *
 * Applies lag, hit probability, slippage and limit-price validation, then
 * builds a TradeReport, applies the fill to the order header, fires the
 * SMM_EVENT_MATCH listeners and publishes on_report / on_order on the bound
 * engine. Fully filled orders are moved to history.
 * @param ctx          Matcher context.
 * @param entry        Working order entry (may be removed on full fill).
 * @param match_volume Desired volume; NAN = full working volume.
 * @param match_price  Desired price; NAN = use limit price.
 */
static inline void c_smm_try_match(smm_match_ctx* ctx, smm_order_entry* entry, double match_volume, double match_price) {
    if (!ctx || !entry || !entry->header) return;
    md_trade_instruction* order = &entry->header->trade_instruction;

    if (!c_smm_lag_allowed(ctx, entry)) return;
    if (!c_smm_hit_ok(ctx)) return;

    double working_volume = order->volume - order->filled_volume;
    if (isnan(match_volume)) {
        match_volume = working_volume;
    }
    else if (match_volume > working_volume) {
        match_volume = working_volume;
    }

    if (isnan(match_price) && !isnan(order->limit_price)) {
        match_price = order->limit_price;
    }
    else if (!isnan(match_price)) {
        match_price = c_smm_apply_slippage(ctx, match_price, order->side);
    }

    if (!isnan(order->limit_price)) {
        int8_t sign = c_md_side_sign(order->side);
        if (sign > 0 && match_price > order->limit_price) {
            match_price = order->limit_price;
        }
        else if (sign < 0 && match_price < order->limit_price) {
            match_price = order->limit_price;
        }
    }

    if (match_volume <= 0) return;

    md_trade_report* rpt = &ctx->ws_report.trade_report;
    memset(&ctx->ws_report, 0, sizeof(md_variant));
    rpt->meta_info.dtype = DTYPE_REPORT;
    rpt->meta_info.ticker = order->meta_info.ticker;
    rpt->meta_info.timestamp = ctx->timestamp;
    if (c_ex_profile_session_datetime_from_unix(ctx->timestamp, &rpt->meta_info.dt) != 0) {
        memset(&rpt->meta_info.dt, 0, sizeof(session_datetime_t));
    }
    rpt->meta_info.dt.date.stype = SESSION_TYPE_NORMINAL;
    rpt->price = match_price;
    rpt->volume = match_volume;
    rpt->side = order->side;
    rpt->multiplier = order->multiplier;
    rpt->notional = match_volume * match_price * order->multiplier;
    rpt->fee = ctx->fee_rate * rpt->notional;
    rpt->order_id = order->order_id;
    c_smm_gen_uuid(ctx, &rpt->trade_id);

    order->filled_volume += match_volume;
    order->filled_notional += fabs(rpt->notional);
    order->fee += rpt->fee;
    if (order->filled_volume == order->volume) {
        order->order_state = STATE_FILLED;
        order->ts_finished = ctx->timestamp;
    }
    else if (order->filled_volume > 0) {
        order->order_state = STATE_PARTFILLED;
    }

    c_smm_match_fire(ctx, SMM_EVENT_MATCH, entry, rpt);

    if (ctx->engine) {
        c_smm_publish_report(ctx, &ctx->ws_report);
        if (entry->py_order) {
            c_smm_publish(ctx, ctx->engine->on_order_topic, ctx->engine->on_order_topic_obj, "order", entry->py_order);
        }
    }

    if (order->order_state == STATE_FILLED) {
        c_smm_remove_entry(ctx, entry);
    }
}

/**
 * @brief Matching scan for bar data.
 * @param ctx Matcher context.
 * @param md  BarData header.
 */
static inline void c_smm_check_bar(smm_match_ctx* ctx, const md_variant* md) {
    if (!ctx || !md) return;
    const md_candlestick* bar = &md->bar_data;
    smm_order_entry*      entry = ctx->working;
    while (entry) {
        smm_order_entry* next = entry->next;
        if (c_smm_order_eligible(ctx, entry, md)) {
            md_trade_instruction* order = &entry->header->trade_instruction;
            int8_t                sign = c_md_side_sign(order->side);
            double                match_price = NAN;
            bool                  has_match = false;
            if (sign > 0) {
                if (isnan(order->limit_price)) {
                    match_price = (bar->volume > 0) ? bar->notional / bar->volume : NAN;
                    has_match = true;
                }
                else if (bar->high_price < order->limit_price) {
                    match_price = bar->high_price;
                    has_match = true;
                }
                else if (bar->low_price < order->limit_price) {
                    match_price = order->limit_price;
                    has_match = true;
                }
            }
            else if (sign < 0) {
                if (isnan(order->limit_price)) {
                    match_price = (bar->volume > 0) ? bar->notional / bar->volume : NAN;
                    has_match = true;
                }
                else if (bar->low_price > order->limit_price) {
                    match_price = bar->low_price;
                    has_match = true;
                }
                else if (bar->high_price > order->limit_price) {
                    match_price = order->limit_price;
                    has_match = true;
                }
            }
            if (has_match) {
                c_smm_try_match(ctx, entry, NAN, match_price);
            }
        }
        entry = next;
    }
}

/**
 * @brief Matching scan for tick data (order book).
 * @param ctx Matcher context.
 * @param md  TickData header.
 */
static inline void c_smm_check_tick(smm_match_ctx* ctx, const md_variant* md) {
    if (!ctx || !md) return;
    const md_tick_data* tick = &md->tick_data_full;
    smm_order_entry*    entry = ctx->working;
    while (entry) {
        smm_order_entry* next = entry->next;
        if (c_smm_order_eligible(ctx, entry, md)) {
            md_trade_instruction* order = &entry->header->trade_instruction;
            int8_t                sign = c_md_side_sign(order->side);
            double                limit = order->limit_price;
            const md_orderbook*   book = (sign > 0) ? tick->ask : tick->bid;
            bool                  enter = false;

            if (isnan(limit)) {
                enter = true;
            }
            else if (sign > 0) {
                enter = (book->size > 0 && book->entries[0].price <= limit);
            }
            else if (sign < 0) {
                enter = (book->size > 0 && book->entries[0].price >= limit);
            }

            double match_volume = 0.0;
            double match_notional = 0.0;
            double working_volume = order->volume - order->filled_volume;

            if (enter) {
                for (size_t i = 0u; i < book->size; i++) {
                    const md_orderbook_entry* e = &book->entries[i];
                    bool                      in_price = isnan(limit) || (sign > 0 ? e->price <= limit : e->price >= limit);
                    if (!in_price) break;
                    if (match_volume >= working_volume) break;
                    double addition = e->volume;
                    double remaining = working_volume - match_volume;
                    if (addition > remaining) addition = remaining;
                    match_volume += addition;
                    match_notional += addition * e->price;
                }
            }

            if (match_volume > 0) {
                c_smm_try_match(ctx, entry, match_volume, match_notional / match_volume);
            }
        }
        entry = next;
    }
}

/**
 * @brief Matching scan for tick-lite data.
 * @param ctx Matcher context.
 * @param md  TickDataLite header.
 */
static inline void c_smm_check_tick_lite(smm_match_ctx* ctx, const md_variant* md) {
    if (!ctx || !md) return;
    const md_tick_data_lite* lite = &md->tick_data_lite;
    smm_order_entry*         entry = ctx->working;
    while (entry) {
        smm_order_entry* next = entry->next;
        if (c_smm_order_eligible(ctx, entry, md)) {
            md_trade_instruction* order = &entry->header->trade_instruction;
            int8_t                sign = c_md_side_sign(order->side);
            double                limit = order->limit_price;
            double                vol = 0.0;
            double                price = NAN;

            if (isnan(limit)) {
                if (sign > 0) {
                    vol = lite->ask_volume;
                    price = lite->ask_price;
                }
                else if (sign < 0) {
                    vol = lite->bid_volume;
                    price = lite->bid_price;
                }
            }
            else if (sign > 0 && lite->ask_price <= limit) {
                vol = lite->ask_volume;
                price = lite->ask_price;
            }
            else if (sign < 0 && lite->bid_price >= limit) {
                vol = lite->bid_volume;
                price = lite->bid_price;
            }

            if (vol > 0) {
                c_smm_try_match(ctx, entry, vol, price);
            }
        }
        entry = next;
    }
}

/**
 * @brief Matching scan for order data.
 * @param ctx Matcher context.
 * @param md  OrderData header.
 */
static inline void c_smm_check_order(smm_match_ctx* ctx, const md_variant* md) {
    if (!ctx || !md) return;
    const md_order_data* od = &md->order_data;
    smm_order_entry*     entry = ctx->working;
    while (entry) {
        smm_order_entry* next = entry->next;
        if (c_smm_order_eligible(ctx, entry, md)) {
            md_trade_instruction* order = &entry->header->trade_instruction;
            int8_t                sign = c_md_side_sign(order->side);
            double                limit = order->limit_price;
            double                vol = 0.0;
            double                price = NAN;

            if (isnan(limit)) {
                if (sign > 0 && c_md_side_sign(od->side) < 0) {
                    vol = od->volume;
                    price = od->price;
                }
                else if (sign < 0 && c_md_side_sign(od->side) > 0) {
                    vol = od->volume;
                    price = od->price;
                }
            }
            else if (sign > 0 && od->price <= limit) {
                vol = od->volume;
                price = od->price;
            }
            else if (sign < 0 && od->price >= limit) {
                vol = od->volume;
                price = od->price;
            }

            if (vol > 0) {
                c_smm_try_match(ctx, entry, vol, price);
            }
        }
        entry = next;
    }
}

/**
 * @brief Matching scan for transaction / trade data.
 * @param ctx Matcher context.
 * @param md  TransactionData or TradeData header.
 */
static inline void c_smm_check_trade(smm_match_ctx* ctx, const md_variant* md) {
    if (!ctx || !md) return;
    const md_transaction_data* txn = &md->transaction_data;
    smm_order_entry*           entry = ctx->working;
    while (entry) {
        smm_order_entry* next = entry->next;
        if (c_smm_order_eligible(ctx, entry, md)) {
            md_trade_instruction* order = &entry->header->trade_instruction;
            int8_t                sign = c_md_side_sign(order->side);
            double                limit = order->limit_price;

            if (isnan(limit)) {
                if (sign * c_md_side_sign(txn->side) > 0) {
                    c_smm_try_match(ctx, entry, txn->volume, txn->price);
                }
            }
            else if (sign > 0 && txn->price < limit) {
                c_smm_try_match(ctx, entry, txn->volume, txn->price);
            }
            else if (sign < 0 && txn->price > limit) {
                c_smm_try_match(ctx, entry, txn->volume, txn->price);
            }
        }
        entry = next;
    }
}

#endif /* C_SIMMATCH_EX_H */

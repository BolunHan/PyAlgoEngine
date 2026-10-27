#ifndef C_EVENT_ENGINE_H
#define C_EVENT_ENGINE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include <algo_engine/base/c_market_data/c_market_data.h>
#include <cbase/allocator_protocol/c_allocator_protocol.h>
#include <cbase/bytemap/c_bytemap.h>
#include <event_engine/capi/c_topic.h>

// ========== Constants ==========

/* Stack buffer for composed keys; longer keys go through the allocator. */
#ifndef TOPIC_SET_KEY_MAX
#define TOPIC_SET_KEY_MAX 256U
#endif

/* 0: character switch over the names this set generates (O(1)).
   1: full comparison against every known dtype name. */
#ifndef TOPIC_SET_MODE_VIGILANT
#define TOPIC_SET_MODE_VIGILANT 0
#endif

/* Topic literals: the single source of every key the set composes. */
#ifndef TOPIC_SET_ON_ORDER
#define TOPIC_SET_ON_ORDER "on_order"
#endif

#ifndef TOPIC_SET_ON_REPORT
#define TOPIC_SET_ON_REPORT "on_report"
#endif

#ifndef TOPIC_SET_EOD
#define TOPIC_SET_EOD "eod"
#endif

#ifndef TOPIC_SET_EOD_DONE
#define TOPIC_SET_EOD_DONE "eod_done"
#endif

#ifndef TOPIC_SET_BOD
#define TOPIC_SET_BOD "bod"
#endif

#ifndef TOPIC_SET_BOD_DONE
#define TOPIC_SET_BOD_DONE "bod_done"
#endif

#ifndef TOPIC_SET_LAUNCH_ORDER_PATTERN
#define TOPIC_SET_LAUNCH_ORDER_PATTERN "launch_order.{ticker}"
#endif

#ifndef TOPIC_SET_CANCEL_ORDER_PATTERN
#define TOPIC_SET_CANCEL_ORDER_PATTERN "cancel_order.{ticker}"
#endif

#ifndef TOPIC_SET_REALTIME
#define TOPIC_SET_REALTIME "realtime"
#endif

#ifndef TOPIC_SET_REALTIME_PATTERN
#define TOPIC_SET_REALTIME_PATTERN TOPIC_SET_REALTIME ".{ticker}.{dtype}"
#endif

#ifndef TOPIC_SET_REALTIME_PREFIX
#define TOPIC_SET_REALTIME_PREFIX TOPIC_SET_REALTIME "."
#endif

// ========== Enums ==========

// clang-format off

/** @enum evt_topic_variant
 *  @brief Retained for the C-only topic set; no key carries a variant yet. */
typedef enum evt_topic_variant {
    EVT_TOPIC_VARIANT_C     = 0,        // .c  - pure C handler (no GIL).
    EVT_TOPIC_VARIANT_PY    = 1,        // .py - Python callback (requires the GIL).
    EVT_TOPIC_VARIANT_COUNT = 2,        // Number of variants and array size.
    EVT_TOPIC_VARIANT_MASK  = 0b1111,   // 4-bit field reserved for future variants.
} evt_topic_variant;

/** @enum topic_set_ret_code
 *  @brief 0 is success, negatives are errors. */
typedef enum topic_set_ret_code {
    TOPIC_SET_OK                =  0,
    TOPIC_SET_ERR_INVALID_INPUT = -1,
    TOPIC_SET_ERR_INVALID_TOPIC = -2,
    TOPIC_SET_ERR_UNKNOWN_DTYPE = -3,
    TOPIC_SET_ERR_OOM           = -4,
} topic_set_ret_code;

// clang-format on

// ========== Structs ==========

/**
 * @struct evt_topic_set
 * @brief The nine legacy-composed built-ins plus the dynamic registries.
 *
 * Keys carry no dispatch suffix. Every topic is allocator-backed and owned by
 * the set: the full-key registry holds one entry per topic and dealloc frees
 * them; the named fields alias the registry pointers. Every pointer is valid
 * until c_topic_set_free.
 *
 * rt_tick, rt_trade (DTYPE_TRANSACTION) and rt_order map a bare ticker to a
 * topic for the three hot dtypes; other dtypes resolve through the registry.
 */
typedef struct evt_topic_set {
    evt_topic* on_order;   // TOPIC_SET_ON_ORDER
    evt_topic* on_report;  // TOPIC_SET_ON_REPORT
    evt_topic* eod;        // TOPIC_SET_EOD
    evt_topic* eod_done;   // TOPIC_SET_EOD_DONE
    evt_topic* bod;        // TOPIC_SET_BOD
    evt_topic* bod_done;   // TOPIC_SET_BOD_DONE

    evt_topic* launch_order;  // TOPIC_SET_LAUNCH_ORDER_PATTERN
    evt_topic* cancel_order;  // TOPIC_SET_CANCEL_ORDER_PATTERN

    evt_topic* realtime;  // TOPIC_SET_REALTIME_PATTERN

    bytemap*   rt_tick;   // OWNED - DTYPE_TICK: ticker -> topic.
    bytemap*   rt_trade;  // OWNED - DTYPE_TRANSACTION: ticker -> topic.
    bytemap*   rt_order;  // OWNED - DTYPE_ORDER: ticker -> topic.

    bytemap*   registry;  // OWNED - full topic key -> evt_topic*.
} evt_topic_set;

// ========== Forward Declarations ==========

static inline const char*    c_topic_set_variant_suffix(evt_topic_variant variant);
static inline md_data_type   c_topic_set_dtype_from_name(const char* name, size_t name_len);

static inline int            c_topic_set_init(evt_topic_set* topic_set, allocator_protocol* allocator);
static inline void           c_topic_set_dealloc(evt_topic_set* topic_set);
static inline evt_topic_set* c_topic_set_new(allocator_protocol* allocator);
static inline void           c_topic_set_free(evt_topic_set* topic_set);

static inline bool           c_topic_set_contains(const evt_topic_set* topic_set, const char* topic_literal, size_t topic_len);
static inline bool           c_topic_set_contains_topic(const evt_topic_set* topic_set, const evt_topic* topic);
static inline evt_topic*     c_topic_set_get(evt_topic_set* topic_set, const char* topic_literal, size_t topic_len);
static inline evt_topic*     c_topic_set_register(evt_topic_set* topic_set, evt_topic* topic);

static inline evt_topic*     c_md_topic_push(evt_topic_set* topic_set, const md_variant* market_data);
static inline int            c_md_topic_parse(const evt_topic_set* topic_set, const evt_topic* topic, const char** out_ticker, md_data_type* out_dtype);

// ========== Utility Functions ==========

/**
 * @brief Suffix of a topic variant ("c" / "py"), or NULL when invalid.
 */
static inline const char* c_topic_set_variant_suffix(evt_topic_variant variant) {
    switch (variant) {
        case EVT_TOPIC_VARIANT_C:
            return "c";
        case EVT_TOPIC_VARIANT_PY:
            return "py";
        default:
            return NULL;
    }
}

/**
 * @brief Resolve a dtype name to its md_data_type; DTYPE_UNKNOWN when unknown.
 *
 * The body is chosen at compile time by TOPIC_SET_MODE_VIGILANT, so the hot
 * path pays no branch. Mode 0 reads the distinguishing characters of the nine
 * generated names (B, G, I, O, T+i|r); the two index reads are length-guarded.
 * Mode 1 compares every known name in full.
 */
static inline md_data_type c_topic_set_dtype_from_name(const char* name, size_t name_len) {
    if (!name || name_len == 0) return DTYPE_UNKNOWN;

#if TOPIC_SET_MODE_VIGILANT
    static const md_data_type dtypes[] = {
        DTYPE_MARKET_DATA,
        DTYPE_INTERNAL,
        DTYPE_TRANSACTION,
        DTYPE_ORDER,
        DTYPE_TICK_LITE,
        DTYPE_TICK,
        DTYPE_BAR,
        DTYPE_REPORT,
        DTYPE_INSTRUCTION,
    };

    for (size_t i = 0; i < sizeof(dtypes) / sizeof(dtypes[0]); i++) {
        const char* candidate = c_md_dtype_name(dtypes[i]);
        if (!candidate) continue;

        if (strlen(candidate) == name_len && memcmp(candidate, name, name_len) == 0) return dtypes[i];
    }

    return DTYPE_UNKNOWN;
#else
    switch (name[0]) {
        case 'B':
            return DTYPE_BAR;

        case 'G':
            return DTYPE_MARKET_DATA;

        case 'I':
            return DTYPE_INTERNAL;

        case 'O':
            return DTYPE_ORDER;

        case 'T':
            if (name_len < 2) return DTYPE_UNKNOWN;

            switch (name[1]) {
                case 'i':
                    // "TickData" (8) vs "TickDataLite" (12); index 8 is the
                    // terminator of the shorter name, never 'L'.
                    if (name_len < 8) return DTYPE_UNKNOWN;
                    return name[8] == 'L' ? DTYPE_TICK_LITE : DTYPE_TICK;

                case 'r':
                    // Index 5: 'R' TradeReport, 'I' TradeInstruction, else
                    // TransactionData.
                    if (name_len < 6) return DTYPE_UNKNOWN;
                    return name[5] == 'R'   ? DTYPE_REPORT
                           : name[5] == 'I' ? DTYPE_INSTRUCTION
                                            : DTYPE_TRANSACTION;
            }
            return DTYPE_UNKNOWN;
    }

    return DTYPE_UNKNOWN;
#endif
}

// ========== Lifecycle ==========

/**
 * @brief Initialize a caller-allocated set: registries plus the nine built-ins.
 */
static inline int c_topic_set_init(evt_topic_set* topic_set, allocator_protocol* allocator) {
    if (!topic_set) return TOPIC_SET_ERR_INVALID_INPUT;

    memset(topic_set, 0, sizeof(evt_topic_set));

    topic_set->registry = c_bytemap_new(0, allocator);
    if (!topic_set->registry) return TOPIC_SET_ERR_OOM;

    topic_set->rt_tick = c_bytemap_new(0, allocator);
    topic_set->rt_trade = c_bytemap_new(0, allocator);
    topic_set->rt_order = c_bytemap_new(0, allocator);

    if (!topic_set->rt_tick || !topic_set->rt_trade || !topic_set->rt_order) {
        c_topic_set_dealloc(topic_set);
        return TOPIC_SET_ERR_OOM;
    }

    struct builtin_binding {
        const char* key;
        evt_topic** slot;
    };

    struct builtin_binding bindings[] = {
        {TOPIC_SET_ON_ORDER, &topic_set->on_order},
        {TOPIC_SET_ON_REPORT, &topic_set->on_report},
        {TOPIC_SET_EOD, &topic_set->eod},
        {TOPIC_SET_EOD_DONE, &topic_set->eod_done},
        {TOPIC_SET_BOD, &topic_set->bod},
        {TOPIC_SET_BOD_DONE, &topic_set->bod_done},
        {TOPIC_SET_LAUNCH_ORDER_PATTERN, &topic_set->launch_order},
        {TOPIC_SET_CANCEL_ORDER_PATTERN, &topic_set->cancel_order},
        {TOPIC_SET_REALTIME_PATTERN, &topic_set->realtime},
    };

    for (size_t i = 0; i < sizeof(bindings) / sizeof(bindings[0]); i++) {
        evt_topic* canonical = c_topic_new(bindings[i].key, 0, allocator);
        if (!canonical) {
            c_topic_set_dealloc(topic_set);
            return TOPIC_SET_ERR_OOM;
        }

        if (c_bytemap_set(topic_set->registry, bindings[i].key, 0, (void*) canonical, NULL) != BYTEMAP_OK) {
            c_topic_free(canonical);
            c_topic_set_dealloc(topic_set);
            return TOPIC_SET_ERR_OOM;
        }

        *bindings[i].slot = canonical;
    }

    return TOPIC_SET_OK;
}

/**
 * @brief Free every topic in the registry and every map; does not free the set.
 */
static inline void c_topic_set_dealloc(evt_topic_set* topic_set) {
    if (!topic_set) return;

    if (topic_set->registry) {
        bytemap_entry* entry = c_bytemap_first(topic_set->registry);
        while (entry) {
            evt_topic* topic = (evt_topic*) c_bytemap_entry_value(entry);
            if (topic) c_topic_free(topic);
            entry = c_bytemap_next(entry);
        }

        c_bytemap_free(topic_set->registry);
        topic_set->registry = NULL;
    }

    if (topic_set->rt_tick) c_bytemap_free(topic_set->rt_tick);
    if (topic_set->rt_trade) c_bytemap_free(topic_set->rt_trade);
    if (topic_set->rt_order) c_bytemap_free(topic_set->rt_order);

    memset(topic_set, 0, sizeof(evt_topic_set));
}

/**
 * @brief Allocate and initialize a topic set; NULL on failure.
 */
static inline evt_topic_set* c_topic_set_new(allocator_protocol* allocator) {
    evt_topic_set* topic_set = (evt_topic_set*) c_ap_alloc(sizeof(evt_topic_set), allocator);
    if (!topic_set) return NULL;

    if (c_topic_set_init(topic_set, allocator) != TOPIC_SET_OK) {
        c_ap_free(topic_set);
        return NULL;
    }
    return topic_set;
}

/**
 * @brief Destroy a set created by c_topic_set_new.
 */
static inline void c_topic_set_free(evt_topic_set* topic_set) {
    if (!topic_set) return;
    c_topic_set_dealloc(topic_set);
    c_ap_free(topic_set);
}

// ========== Public APIs ==========

/**
 * @brief Test whether a topic literal is registered; pure lookup.
 */
static inline bool c_topic_set_contains(const evt_topic_set* topic_set, const char* topic_literal, size_t topic_len) {
    if (!topic_set || !topic_set->registry) return false;
    return c_bytemap_contains(topic_set->registry, topic_literal, topic_len) == BYTEMAP_OK;
}

/**
 * @brief Test whether this exact topic (key and pointer) is registered.
 */
static inline bool c_topic_set_contains_topic(const evt_topic_set* topic_set, const evt_topic* topic) {
    if (!topic_set || !topic_set->registry || !topic || !topic->key || topic->key_len == 0) return false;

    evt_topic* existing = NULL;
    return c_bytemap_get(topic_set->registry, topic->key, topic->key_len, (void**) &existing) == BYTEMAP_OK && existing == topic;
}

/**
 * @brief Get the topic registered under this literal, creating and registering
 *        it on first use (get-or-set).
 *
 * The literal is parsed by c_topic_new, so a dotted ticker is split into
 * several parts; c_md_topic_push builds such topics from parts instead.
 */
static inline evt_topic* c_topic_set_get(evt_topic_set* topic_set, const char* topic_literal, size_t topic_len) {
    if (!topic_set || !topic_set->registry || !topic_literal) return NULL;
    if (topic_len == 0) topic_len = strlen(topic_literal);
    if (topic_len == 0) return NULL;

    allocator_protocol* allocator = c_ap_protocol_from_ptr(topic_set->registry);
    evt_topic*          topic = c_topic_new(topic_literal, topic_len, allocator);
    if (!topic) return NULL;

    return c_topic_set_register(topic_set, topic);
}

/**
 * @brief Get the topic registered under this topic's key, or register it.
 *
 * Single probe (c_bytemap_ex_set_default): the stored topic wins on a hit, the
 * given topic is inserted as the default on a miss. Ownership passes to the
 * set only when the key is new; on a hit the caller keeps `topic`.
 *
 * A realtime topic with three exact parts is also inserted into its dtype's
 * ticker cache, so c_md_topic_push finds it without touching the registry.
 */
static inline evt_topic* c_topic_set_register(evt_topic_set* topic_set, evt_topic* topic) {
    if (!topic_set || !topic_set->registry || !topic || !topic->key || topic->key_len == 0) return NULL;

    char* stored = NULL;
    if (c_bytemap_ex_set_default(topic_set->registry, topic->key, topic->key_len, (const char*) &topic, sizeof(topic), 0, &stored, NULL) != BYTEMAP_OK) {
        return NULL;
    }

    evt_topic* registered = NULL;
    if (stored) memcpy(&registered, stored, sizeof(registered));
    if (!registered) return NULL;

    if (!registered->is_exact || registered->n != 3) return registered;

    const evt_topic_part_variant* part_0 = registered->parts;
    const evt_topic_part_variant* part_1 = part_0 ? part_0->header.next : NULL;
    const evt_topic_part_variant* part_2 = part_1 ? part_1->header.next : NULL;

    if (!part_2 || part_2->header.next) return registered;
    if (part_0->header.ttype != TOPIC_PART_EXACT ||
        part_1->header.ttype != TOPIC_PART_EXACT ||
        part_2->header.ttype != TOPIC_PART_EXACT) {
        return registered;
    }
    if (part_0->exact.part_len != sizeof(TOPIC_SET_REALTIME) - 1 ||
        memcmp(part_0->exact.part, TOPIC_SET_REALTIME, sizeof(TOPIC_SET_REALTIME) - 1) != 0) {
        return registered;
    }
    if (part_1->exact.part_len == 0) return registered;

    bytemap* map = NULL;
    switch (c_topic_set_dtype_from_name(part_2->exact.part, part_2->exact.part_len)) {
        case DTYPE_TICK:
            map = topic_set->rt_tick;
            break;
        case DTYPE_TRANSACTION:
            map = topic_set->rt_trade;
            break;
        case DTYPE_ORDER:
            map = topic_set->rt_order;
            break;
        default:
            break;
    }

    if (map) c_bytemap_set(map, part_1->exact.part, part_1->exact.part_len, (void*) registered, NULL);
    return registered;
}

/**
 * @brief Resolve the publish topic for a market data buffer.
 *
 * Tick, transaction and order resolve through their dedicated ticker cache
 * (one lookup); other dtypes have no cache and are looked up in the registry
 * by the composed literal (stack buffer, oversized tickers are rejected). A
 * miss builds realtime.<ticker>.<dtype> from parts and registers it, which
 * also fills the ticker cache.
 */
static inline evt_topic* c_md_topic_push(evt_topic_set* topic_set, const md_variant* market_data) {
    if (!topic_set || !topic_set->registry || !market_data) return NULL;

    const char* ticker = market_data->meta_info.ticker;
    if (!ticker) return NULL;

    size_t ticker_len = strlen(ticker);
    if (ticker_len == 0) return NULL;

    md_data_type dtype = market_data->meta_info.dtype;
    if (dtype == DTYPE_UNKNOWN) return NULL;

    bytemap* map = NULL;
    switch (dtype) {
        case DTYPE_TICK:
            map = topic_set->rt_tick;
            break;
        case DTYPE_TRANSACTION:
            map = topic_set->rt_trade;
            break;
        case DTYPE_ORDER:
            map = topic_set->rt_order;
            break;
        default:
            // No dedicated map; the registry path handles it.
            break;
    }

    // Hot dtypes: the ticker cache is authoritative (c_topic_set_register
    // keeps it in step), so a miss means the set does not have it.
    if (map) {
        evt_topic* cached = NULL;
        if (c_bytemap_get(map, ticker, ticker_len, (void**) &cached) == BYTEMAP_OK) return cached;
    }

    const char* dtype_name = c_md_dtype_name(dtype);
    if (!dtype_name) return NULL;

    // Other dtypes have no cache, so the registry serves as one and the key
    // has to be composed.
    if (!map) {
        const size_t prefix_len = sizeof(TOPIC_SET_REALTIME_PREFIX) - 1;
        size_t       dtype_len = strlen(dtype_name);
        size_t       key_len = prefix_len + ticker_len + 1 + dtype_len;

        if (key_len + 1 > TOPIC_SET_KEY_MAX) return NULL;

        char   key[TOPIC_SET_KEY_MAX];

        size_t pos = 0;
        memcpy(key + pos, TOPIC_SET_REALTIME_PREFIX, prefix_len);
        pos += prefix_len;
        memcpy(key + pos, ticker, ticker_len);
        pos += ticker_len;
        key[pos++] = DEFAULT_TOPIC_SEP;
        memcpy(key + pos, dtype_name, dtype_len);
        pos += dtype_len;
        key[pos] = '\0';

        evt_topic* cached = NULL;
        if (c_bytemap_get(topic_set->registry, key, key_len, (void**) &cached) == BYTEMAP_OK) return cached;
    }

    allocator_protocol* allocator = c_ap_protocol_from_ptr(topic_set->registry);

    evt_topic*          topic = c_topic_new(NULL, 0, allocator);
    if (!topic) return NULL;

    int ret_code = c_topic_append(topic, TOPIC_SET_REALTIME, 0, TOPIC_PART_EXACT);
    if (ret_code == EVT_RET_OK) ret_code = c_topic_append(topic, ticker, 0, TOPIC_PART_EXACT);
    if (ret_code == EVT_RET_OK) ret_code = c_topic_append(topic, dtype_name, 0, TOPIC_PART_EXACT);
    if (ret_code == EVT_RET_OK) ret_code = c_topic_update_literal(topic);

    if (ret_code != EVT_RET_OK) {
        c_topic_free(topic);
        return NULL;
    }

    return c_topic_set_register(topic_set, topic);
}

/**
 * @brief Parse realtime.<ticker>.<dtype> into (ticker, dtype).
 *
 * out_ticker is a borrowed pointer into the topic's ticker part; it lives as
 * long as the owning set.
 */
static inline int c_md_topic_parse(const evt_topic_set* topic_set, const evt_topic* topic, const char** out_ticker, md_data_type* out_dtype) {
    if (!topic_set || !topic) return TOPIC_SET_ERR_INVALID_INPUT;
    if (!topic->is_exact || topic->n != 3) return TOPIC_SET_ERR_INVALID_TOPIC;

    const evt_topic_part_variant* part_0 = topic->parts;
    const evt_topic_part_variant* part_1 = part_0 ? part_0->header.next : NULL;
    const evt_topic_part_variant* part_2 = part_1 ? part_1->header.next : NULL;

    if (!part_2 || part_2->header.next) return TOPIC_SET_ERR_INVALID_TOPIC;
    if (part_0->header.ttype != TOPIC_PART_EXACT ||
        part_1->header.ttype != TOPIC_PART_EXACT ||
        part_2->header.ttype != TOPIC_PART_EXACT) {
        return TOPIC_SET_ERR_INVALID_TOPIC;
    }

    if (part_0->exact.part_len != sizeof(TOPIC_SET_REALTIME) - 1 ||
        memcmp(part_0->exact.part, TOPIC_SET_REALTIME, sizeof(TOPIC_SET_REALTIME) - 1) != 0) {
        return TOPIC_SET_ERR_INVALID_TOPIC;
    }

    if (part_1->exact.part_len == 0) return TOPIC_SET_ERR_INVALID_TOPIC;

    md_data_type dtype = c_topic_set_dtype_from_name(part_2->exact.part, part_2->exact.part_len);
    if (dtype == DTYPE_UNKNOWN) return TOPIC_SET_ERR_UNKNOWN_DTYPE;

    if (out_ticker) *out_ticker = part_1->exact.part;
    if (out_dtype) *out_dtype = dtype;
    return TOPIC_SET_OK;
}

#endif  // C_EVENT_ENGINE_H

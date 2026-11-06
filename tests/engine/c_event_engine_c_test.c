/*
 * Pure-C test for algo_engine/engine/c_event_engine.h.
 *
 * Covers:
 *   1. The retained evt_topic_variant enum and suffix helper.
 *   2. Topic set lifecycle and the nine built-in topics: every named field is
 *      the allocator-backed heap topic the registry owns, and the keys are the
 *      legacy composition with no dispatch suffix.
 *   3. Template matching.
 *   4. c_md_topic_push / c_md_topic_parse for every md_data_type, with direct
 *      assertions on the rt_tick / rt_trade / rt_order ticker registries.
 *   5. Dynamic topic registration through the full-key bytemap registry.
 *   6. dtype-name resolution under both settings of the compile-time
 *      TOPIC_SET_MODE_VIGILANT flag.
 *   7. Ownership: the set frees its topics; a fresh set can register the same
 *      keys afterwards.
 *   8. Publish + dispatch in the pure C event engine layer (no GIL), using the
 *      built-in heap topics directly.
 *   9. A push/parse throughput measurement.
 *
 * Build & run with run_c_event_engine_c_test.sh, which builds the same
 * source twice: once with the default fast resolution and once with
 * -DTOPIC_SET_MODE_VIGILANT=1.
 */

// Feature-test macro for CPU_ZERO / CPU_SET / pthread_setaffinity_np. Must be
// defined before any system header is pulled in.
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <algo_engine/engine/c_event_engine.h>
#include <event_engine/capi/c_engine.h>

// ========== Assertion Helper ==========

static size_t          g_checks = 0;
static size_t          g_failures = 0;
static _Atomic size_t  g_payload_frees = 0;

#define CHECK(cond)                                                          \
    do {                                                                     \
        g_checks++;                                                          \
        if (!(cond)) {                                                       \
            g_failures++;                                                    \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);  \
        }                                                                    \
    } while (0)

// ========== 1. Retained Variant Surface ==========

static void test_variant_surface(void) {
    // Retained for the C-only topic set that follows; the set does not put a
    // variant suffix into any key it composes.
    CHECK(EVT_TOPIC_VARIANT_C == 0);
    CHECK(EVT_TOPIC_VARIANT_PY == 1);
    CHECK(EVT_TOPIC_VARIANT_COUNT == 2);
    CHECK(EVT_TOPIC_VARIANT_MASK == 0b1111);

    CHECK(strcmp(c_topic_set_variant_suffix(EVT_TOPIC_VARIANT_C), "c") == 0);
    CHECK(strcmp(c_topic_set_variant_suffix(EVT_TOPIC_VARIANT_PY), "py") == 0);
    CHECK(c_topic_set_variant_suffix((evt_topic_variant) 2) == NULL);
}

// ========== 2. Built-ins: Named Field + Registry ==========

typedef struct builtin_case {
    const char* key;        // Legacy key, no dispatch suffix.
    evt_topic** slot;       // Address of the named field in evt_topic_set.
    evt_topic*  topic;      // Allocator-backed heap topic held there.
    evt_topic*  canonical;  // Same pointer, as resolved from the registry.
} builtin_case;

static size_t make_builtin_cases(evt_topic_set* set, builtin_case* cases, size_t capacity) {
    builtin_case all[] = {
        {"on_order",                   &set->on_order,     NULL, NULL},
        {"on_report",                  &set->on_report,    NULL, NULL},
        {"eod",                        &set->eod,          NULL, NULL},
        {"eod_done",                   &set->eod_done,     NULL, NULL},
        {"bod",                        &set->bod,          NULL, NULL},
        {"bod_done",                   &set->bod_done,     NULL, NULL},
        {"launch_order.{ticker}",      &set->launch_order, NULL, NULL},
        {"cancel_order.{ticker}",      &set->cancel_order, NULL, NULL},
        {"realtime.{ticker}.{dtype}",  &set->realtime,     NULL, NULL},
    };

    size_t n = sizeof(all) / sizeof(all[0]);
    if (n > capacity) n = capacity;

    for (size_t i = 0; i < n; i++) {
        cases[i] = all[i];
        cases[i].topic = *all[i].slot;
        cases[i].canonical = cases[i].topic;
    }
    return n;
}

static bool key_is_template(const char* key) {
    return strstr(key, "{ticker}") != NULL;
}

static void test_topic_set_basics(evt_topic_set* set) {
    CHECK(set != NULL);
    if (!set) return;

    builtin_case cases[9];
    size_t       n_cases = make_builtin_cases(set, cases, sizeof(cases) / sizeof(cases[0]));
    CHECK(n_cases == 9);

    for (size_t i = 0; i < n_cases; i++) {
        evt_topic* topic = cases[i].topic;
        evt_topic* canonical = cases[i].canonical;

        CHECK(topic != NULL);
        CHECK(canonical != NULL);
        if (!topic || !canonical) continue;

        // One heap topic per built-in: the named field and the registry hold
        // the same pointer, and the key carries no dispatch suffix.
        CHECK(topic == canonical);

        CHECK(topic->key != NULL);
        CHECK(topic->key_len == strlen(cases[i].key));
        CHECK(topic->key && memcmp(topic->key, cases[i].key, topic->key_len) == 0);
        CHECK(topic->is_exact == (key_is_template(cases[i].key) ? 0 : 1));

        // Allocator-backed: c_topic_match recovers the allocator from the
        // topic pointer.
        CHECK(c_topic_match_bool(topic, topic));
        CHECK(c_topic_match_bool(topic, canonical));
        evt_topic_match* match = c_topic_match(topic, topic, NULL);
        CHECK(match != NULL);
        if (match) {
            CHECK(match->matched);
            CHECK(match->literal == topic->key);
            c_topic_match_free(match);
        }

        // The full-key registry resolves the same topic.
        CHECK(c_topic_set_register(set, topic) == topic);
        CHECK(c_topic_set_contains(set, cases[i].key, 0));
        CHECK(c_topic_set_contains_topic(set, topic));
    }

    // Exact built-ins do not match each other.
    for (size_t i = 0; i < n_cases; i++) {
        if (key_is_template(cases[i].key)) continue;
        for (size_t j = i + 1; j < n_cases; j++) {
            if (key_is_template(cases[j].key)) continue;
            CHECK(!c_topic_match_bool(cases[i].topic, cases[j].topic));
        }
    }

    // Invalid lookups are NULL-safe.
    CHECK(!c_topic_set_contains(set, "not.registered.topic", 0));
    CHECK(!c_topic_set_contains(set, "x", 0));
    CHECK(!c_topic_set_contains(NULL, "x", 0));
}

// ========== 3. Template Resolution ==========

static void test_templates(evt_topic_set* set) {
    allocator_protocol* allocator = c_ap_protocol_from_ptr(set->registry);

    evt_topic* launch = c_topic_new("launch_order.RB2510", 0, allocator);
    evt_topic* cancel = c_topic_new("cancel_order.RB2510", 0, allocator);
    CHECK(launch != NULL);
    CHECK(cancel != NULL);
    if (!launch || !cancel) return;

    CHECK(launch->is_exact);
    CHECK(c_topic_match_bool(set->launch_order, launch));
    CHECK(!c_topic_match_bool(set->cancel_order, launch));
    CHECK(!c_topic_match_bool(set->launch_order, cancel));
    CHECK(c_topic_match_bool(set->cancel_order, cancel));

    md_variant md;
    memset(&md, 0, sizeof(md));
    md.meta_info.ticker = "RB2510";
    md.meta_info.dtype = DTYPE_TICK;

    evt_topic* realtime = c_md_topic_push(set, &md);
    CHECK(realtime != NULL);
    if (realtime) {
        CHECK(realtime->is_exact);
        CHECK(c_topic_match_bool(set->realtime, realtime));
        CHECK(!c_topic_match_bool(set->launch_order, realtime));
        CHECK(!c_topic_match_bool(set->realtime, launch));
        CHECK(realtime->n == 3);
    }

    // The legacy composition: no variant suffix anywhere.
    CHECK(strstr(launch->key, ".c") == NULL);
    CHECK(strstr(launch->key, ".py") == NULL);
}

// ========== 4. Push / Parse for Every dtype ==========

static bytemap* named_rt_map(evt_topic_set* set, md_data_type dtype) {
    switch (dtype) {
        case DTYPE_TICK:        return set->rt_tick;
        case DTYPE_TRANSACTION: return set->rt_trade;
        case DTYPE_ORDER:       return set->rt_order;
        default:                return NULL;
    }
}

static void test_push_parse_all_dtypes(evt_topic_set* set) {
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

    static const char* const tickers[] = {
        "RB2510",
        "600010.SH",
        "IF2412",
        "au2512",
        "SR501",
    };

    for (size_t t = 0; t < sizeof(tickers) / sizeof(tickers[0]); t++) {
        for (size_t i = 0; i < sizeof(dtypes) / sizeof(dtypes[0]); i++) {
            md_variant md;
            memset(&md, 0, sizeof(md));
            md.meta_info.ticker = tickers[t];
            md.meta_info.dtype = dtypes[i];

            evt_topic* topic = c_md_topic_push(set, &md);

            CHECK(topic != NULL);
            if (!topic) continue;

            char expected[512];
            int  expected_len = snprintf(expected, sizeof(expected), "realtime.%s.%s", tickers[t], c_md_dtype_name(dtypes[i]));
            CHECK(expected_len > 0);
            CHECK(topic->key_len == (size_t) expected_len);
            CHECK(topic->key && memcmp(topic->key, expected, (size_t) expected_len) == 0);
            CHECK(topic->n == 3);
            CHECK(topic->is_exact);

            // Repeated pushes return the same canonical topic.
            CHECK(c_md_topic_push(set, &md) == topic);

            // Dedicated maps exist for tick/trade/order; other dtypes use
            // only the full-key registry.
            bytemap* named = named_rt_map(set, dtypes[i]);
            if (named) {
                evt_topic* mapped = NULL;
                CHECK(c_bytemap_get(named, tickers[t], 0, (void**) &mapped) == BYTEMAP_OK);
                CHECK(mapped == topic);
            }

            // The full-key registry resolves it for every dtype.
            CHECK(c_topic_set_contains(set, expected, 0));
            CHECK(c_topic_set_contains_topic(set, topic));

            const char*  out_ticker = NULL;
            md_data_type out_dtype = DTYPE_UNKNOWN;
            int ret = c_md_topic_parse(set, topic, &out_ticker, &out_dtype);
            CHECK(ret == TOPIC_SET_OK);
            CHECK(out_ticker && strcmp(out_ticker, tickers[t]) == 0);
            CHECK(out_dtype == dtypes[i]);

            // Optional outputs are NULL-safe.
            CHECK(c_md_topic_parse(set, topic, NULL, NULL) == TOPIC_SET_OK);
            CHECK(c_md_topic_parse(set, topic, &out_ticker, NULL) == TOPIC_SET_OK);
            CHECK(c_md_topic_parse(set, topic, NULL, &out_dtype) == TOPIC_SET_OK);
        }
    }

    // A long ticker still fits the stack-composed key.
    char long_ticker[200];
    memset(long_ticker, 'T', sizeof(long_ticker) - 1);
    long_ticker[sizeof(long_ticker) - 1] = '\0';

    md_variant md_long;
    memset(&md_long, 0, sizeof(md_long));
    md_long.meta_info.ticker = long_ticker;
    md_long.meta_info.dtype = DTYPE_TICK;

    evt_topic* long_topic = c_md_topic_push(set, &md_long);
    CHECK(long_topic != NULL);
    if (long_topic) {
        const char*  out_ticker = NULL;
        md_data_type out_dtype = DTYPE_UNKNOWN;
        CHECK(c_md_topic_parse(set, long_topic, &out_ticker, &out_dtype) == TOPIC_SET_OK);
        CHECK(out_ticker && strcmp(out_ticker, long_ticker) == 0);
        CHECK(out_dtype == DTYPE_TICK);

        // The dedicated ticker map is keyed by the raw ticker.
        evt_topic* mapped = NULL;
        CHECK(c_bytemap_get(set->rt_tick, long_ticker, 0, (void**) &mapped) == BYTEMAP_OK);
        CHECK(mapped == long_topic);

        CHECK(c_topic_set_contains(set, long_topic->key, long_topic->key_len));
        CHECK(c_topic_set_contains_topic(set, long_topic));
    }

    // A key above the 256-byte stack buffer uses the malloc'd key path on the
    // registry path; hot dtypes build from parts and compose no key at all.
    char oversized_ticker[400];
    memset(oversized_ticker, 'T', sizeof(oversized_ticker) - 1);
    oversized_ticker[sizeof(oversized_ticker) - 1] = '\0';

    md_variant md_oversized;
    memset(&md_oversized, 0, sizeof(md_oversized));
    md_oversized.meta_info.ticker = oversized_ticker;

    md_oversized.meta_info.dtype = DTYPE_BAR;
    evt_topic* oversized_bar = c_md_topic_push(set, &md_oversized);
    CHECK(oversized_bar != NULL);
    if (oversized_bar) {
        const char*  out_ticker = NULL;
        md_data_type out_dtype = DTYPE_UNKNOWN;
        CHECK(c_md_topic_parse(set, oversized_bar, &out_ticker, &out_dtype) == TOPIC_SET_OK);
        CHECK(out_ticker && strcmp(out_ticker, oversized_ticker) == 0);
        CHECK(out_dtype == DTYPE_BAR);

        // A repeat hit resolves through the malloc'd key path as well.
        CHECK(c_md_topic_push(set, &md_oversized) == oversized_bar);
    }

    md_oversized.meta_info.dtype = DTYPE_TICK;
    evt_topic* oversized_hot = c_md_topic_push(set, &md_oversized);
    CHECK(oversized_hot != NULL);
    if (oversized_hot) {
        const char*  out_ticker = NULL;
        md_data_type out_dtype = DTYPE_UNKNOWN;
        CHECK(c_md_topic_parse(set, oversized_hot, &out_ticker, &out_dtype) == TOPIC_SET_OK);
        CHECK(out_ticker && strcmp(out_ticker, oversized_ticker) == 0);
    }

    // Invalid market data is rejected.
    md_variant md_bad;
    memset(&md_bad, 0, sizeof(md_bad));
    md_bad.meta_info.ticker = NULL;
    md_bad.meta_info.dtype = DTYPE_TICK;
    CHECK(c_md_topic_push(set, &md_bad) == NULL);

    md_bad.meta_info.ticker = "RB2510";
    md_bad.meta_info.dtype = DTYPE_UNKNOWN;
    CHECK(c_md_topic_push(set, &md_bad) == NULL);

    md_bad.meta_info.ticker = "";
    md_bad.meta_info.dtype = DTYPE_TICK;
    CHECK(c_md_topic_push(set, &md_bad) == NULL);

    md_variant md_valid;
    memset(&md_valid, 0, sizeof(md_valid));
    md_valid.meta_info.ticker = "RB2510";
    md_valid.meta_info.dtype = DTYPE_TICK;

    CHECK(c_md_topic_push(NULL, &md_valid) == NULL);
    CHECK(c_md_topic_push(set, NULL) == NULL);

    // Parse rejects non-realtime shapes.
    const char*  out_ticker = NULL;
    md_data_type out_dtype = DTYPE_UNKNOWN;

    CHECK(c_md_topic_parse(set, NULL, &out_ticker, &out_dtype) == TOPIC_SET_ERR_INVALID_INPUT);
    CHECK(c_md_topic_parse(NULL, set->realtime, &out_ticker, &out_dtype) == TOPIC_SET_ERR_INVALID_INPUT);

    // An exact built-in (on_order) is not a realtime topic.
    CHECK(c_md_topic_parse(set, set->on_order, &out_ticker, &out_dtype) == TOPIC_SET_ERR_INVALID_TOPIC);

    // The realtime template is not exact.
    CHECK(c_md_topic_parse(set, set->realtime, &out_ticker, &out_dtype) == TOPIC_SET_ERR_INVALID_TOPIC);

    // Wrong literal prefix.
    evt_topic* wrong_prefix = c_topic_new("notrealtime.RB2510.TickData", 0, c_ap_protocol_from_ptr(set->registry));
    CHECK(wrong_prefix != NULL);
    if (wrong_prefix) {
        CHECK(c_md_topic_parse(set, wrong_prefix, &out_ticker, &out_dtype) == TOPIC_SET_ERR_INVALID_TOPIC);
    }

    // Missing dtype part.
    evt_topic* short_key = c_topic_new("realtime.RB2510", 0, c_ap_protocol_from_ptr(set->registry));
    CHECK(short_key != NULL);
    if (short_key) {
        CHECK(c_md_topic_parse(set, short_key, &out_ticker, &out_dtype) == TOPIC_SET_ERR_INVALID_TOPIC);
    }

    // The old suffixed composition is not a realtime topic any more.
    evt_topic* legacy_suffix = c_topic_new("realtime.RB2510.TickData.c", 0, c_ap_protocol_from_ptr(set->registry));
    CHECK(legacy_suffix != NULL);
    if (legacy_suffix) {
        CHECK(c_md_topic_parse(set, legacy_suffix, &out_ticker, &out_dtype) == TOPIC_SET_ERR_INVALID_TOPIC);
    }
}

// ========== 5. Dynamic Registration ==========

static evt_topic* register_key(evt_topic_set* set, const char* key) {
    evt_topic* topic = c_topic_new(key, 0, c_ap_protocol_from_ptr(set->registry));
    if (!topic) return NULL;
    return c_topic_set_register(set, topic);
}

static void test_dynamic_registration(evt_topic_set* set) {
    allocator_protocol* allocator = c_ap_protocol_from_ptr(set->registry);
    size_t              before = c_bytemap_len(set->registry);

    evt_topic* topic = c_topic_new("custom.alpha", 0, allocator);
    CHECK(topic != NULL);
    if (!topic) return;

    CHECK(!c_topic_set_contains(set, "custom.alpha", 0));
    CHECK(!c_topic_set_contains_topic(set, topic));

    CHECK(c_topic_set_register(set, topic) == topic);
    CHECK(c_bytemap_len(set->registry) == before + 1);
    CHECK(c_topic_set_contains(set, "custom.alpha", 0));
    CHECK(c_topic_set_contains(set, "custom.alpha", strlen("custom.alpha")));
    CHECK(c_topic_set_get(set, "custom.alpha", 0) == topic);
    CHECK(c_topic_set_contains_topic(set, topic));

    // Re-registering the same topic returns it and does not grow the set.
    CHECK(c_topic_set_register(set, topic) == topic);
    CHECK(c_bytemap_len(set->registry) == before + 1);

    // Registering a key already pushed by the realtime path returns the same
    // canonical topic pointer.
    md_variant md;
    memset(&md, 0, sizeof(md));
    md.meta_info.ticker = "RB2510";
    md.meta_info.dtype = DTYPE_BAR;

    evt_topic* pushed = c_md_topic_push(set, &md);
    CHECK(pushed != NULL);
    evt_topic* same_key = c_topic_new("realtime.RB2510.BarData", 0, allocator);
    CHECK(same_key == pushed);  // Internalized: the push created both.
    CHECK(c_topic_set_register(set, same_key) == pushed);
    CHECK(c_topic_set_contains_topic(set, pushed));

    // A realtime key registered before its first push is reused, not rebuilt.
    evt_topic* pre_registered = c_topic_new("realtime.PF2512.TickData", 0, allocator);
    CHECK(pre_registered != NULL);
    CHECK(c_topic_set_register(set, pre_registered) == pre_registered);
    md_variant md_pre;
    memset(&md_pre, 0, sizeof(md_pre));
    md_pre.meta_info.ticker = "PF2512";
    md_pre.meta_info.dtype = DTYPE_TICK;
    CHECK(c_md_topic_push(set, &md_pre) == pre_registered);

    // A dynamic template can be registered and used with c_topic_match_bool.
    evt_topic* custom_template = register_key(set, "custom.order.{ticker}");
    evt_topic* custom_exact = register_key(set, "custom.order.RB2510");
    CHECK(custom_template != NULL);
    CHECK(custom_exact != NULL);
    if (custom_template && custom_exact) {
        CHECK(!custom_template->is_exact);
        CHECK(custom_exact->is_exact);
        CHECK(c_topic_match_bool(custom_template, custom_exact));
    }

    // get() on an existing literal returns it and registers nothing.
    size_t before_get = c_bytemap_len(set->registry);
    CHECK(c_topic_set_get(set, "custom.alpha", 0) == topic);
    CHECK(c_bytemap_len(set->registry) == before_get);

    // get() on an unknown literal creates, registers and returns it; a repeat
    // call returns the same topic and does not grow the set.
    evt_topic* lazy = c_topic_set_get(set, "lazy.one", 0);
    CHECK(lazy != NULL);
    if (lazy) {
        CHECK(strcmp(lazy->key, "lazy.one") == 0);
        CHECK(c_bytemap_len(set->registry) == before_get + 1);
        CHECK(c_topic_set_contains(set, "lazy.one", 0));
        CHECK(c_topic_set_contains_topic(set, lazy));
        CHECK(c_topic_set_get(set, "lazy.one", 0) == lazy);
        CHECK(c_bytemap_len(set->registry) == before_get + 1);
    }

    // Empty or missing literals resolve to NULL and register nothing.
    CHECK(c_topic_set_get(set, NULL, 0) == NULL);
    CHECK(c_topic_set_get(set, "", 0) == NULL);
    CHECK(c_topic_set_get(NULL, "lazy.one", 0) == NULL);
    CHECK(!c_topic_set_contains(set, "", 0));

    // A topic without a key is refused by the get-set; the caller keeps it.
    evt_topic* keyless = c_topic_new(NULL, 0, allocator);
    CHECK(keyless != NULL);
    if (keyless) {
        CHECK(c_topic_set_register(set, keyless) == NULL);
        CHECK(!c_topic_set_contains_topic(set, keyless));
        c_topic_free(keyless);
    }

    // Negative lookups.
    CHECK(!c_topic_set_contains(set, "not.registered", 0));
    CHECK(!c_topic_set_contains(set, NULL, 0));
    CHECK(!c_topic_set_contains_topic(set, NULL));
    CHECK(!c_topic_set_contains(NULL, "custom.alpha", 0));
    CHECK(c_topic_set_register(NULL, topic) == NULL);
    CHECK(c_topic_set_register(set, NULL) == NULL);
}

// ========== 6. Registry -> Cache Sync ==========

static void test_registry_cache_sync(evt_topic_set* topic_set) {
    allocator_protocol* allocator = c_ap_protocol_from_ptr(topic_set->registry);
    evt_topic*          mapped = NULL;

    // get() of a realtime literal fills the matching ticker cache.
    evt_topic* tick_topic = c_topic_set_get(topic_set, "realtime.SYNC1.TickData", 0);
    CHECK(tick_topic != NULL);
    CHECK(c_bytemap_get(topic_set->rt_tick, "SYNC1", 0, (void**) &mapped) == BYTEMAP_OK);
    CHECK(mapped == tick_topic);

    md_variant md;
    memset(&md, 0, sizeof(md));
    md.meta_info.ticker = "SYNC1";
    md.meta_info.dtype = DTYPE_TICK;
    CHECK(c_md_topic_push(topic_set, &md) == tick_topic);

    evt_topic* trade_topic = c_topic_set_get(topic_set, "realtime.SYNC2.TransactionData", 0);
    CHECK(trade_topic != NULL);
    CHECK(c_bytemap_get(topic_set->rt_trade, "SYNC2", 0, (void**) &mapped) == BYTEMAP_OK);
    CHECK(mapped == trade_topic);
    md.meta_info.ticker = "SYNC2";
    md.meta_info.dtype = DTYPE_TRANSACTION;
    CHECK(c_md_topic_push(topic_set, &md) == trade_topic);

    evt_topic* order_topic = c_topic_set_get(topic_set, "realtime.SYNC3.OrderData", 0);
    CHECK(order_topic != NULL);
    CHECK(c_bytemap_get(topic_set->rt_order, "SYNC3", 0, (void**) &mapped) == BYTEMAP_OK);
    CHECK(mapped == order_topic);

    // Registering an evt_topic* directly syncs the same way.
    evt_topic* direct = c_topic_new("realtime.SYNC4.TickData", 0, allocator);
    CHECK(direct != NULL);
    CHECK(c_topic_set_register(topic_set, direct) == direct);
    CHECK(c_bytemap_get(topic_set->rt_tick, "SYNC4", 0, (void**) &mapped) == BYTEMAP_OK);
    CHECK(mapped == direct);

    // A dtype without a dedicated cache never touches the ticker maps.
    evt_topic* bar_topic = c_topic_set_get(topic_set, "realtime.SYNC5.BarData", 0);
    CHECK(bar_topic != NULL);
    CHECK(c_bytemap_contains(topic_set->rt_tick, "SYNC5", 0) != BYTEMAP_OK);
    CHECK(c_bytemap_contains(topic_set->rt_trade, "SYNC5", 0) != BYTEMAP_OK);
    CHECK(c_bytemap_contains(topic_set->rt_order, "SYNC5", 0) != BYTEMAP_OK);

    // push resolves it through the registry and returns the same topic.
    md.meta_info.ticker = "SYNC5";
    md.meta_info.dtype = DTYPE_BAR;
    CHECK(c_md_topic_push(topic_set, &md) == bar_topic);

    // Only realtime literals are synced.
    evt_topic* custom = c_topic_set_get(topic_set, "other.SYNC6.TickData", 0);
    CHECK(custom != NULL);
    CHECK(c_bytemap_contains(topic_set->rt_tick, "SYNC6", 0) != BYTEMAP_OK);

    // A dotted ticker pushed from parts lands in the cache under the full ticker.
    md.meta_info.ticker = "600010.SH";
    md.meta_info.dtype = DTYPE_TICK;
    evt_topic* dotted = c_md_topic_push(topic_set, &md);
    CHECK(dotted != NULL);
    if (dotted) {
        CHECK(c_bytemap_get(topic_set->rt_tick, "600010.SH", 0, (void**) &mapped) == BYTEMAP_OK);
        CHECK(mapped == dotted);
    }
}

// ========== 7. Set Union ==========

static void test_union(evt_topic_set* set) {
    CHECK(c_topic_set_union(NULL, set, NULL) == NULL);
    CHECK(c_topic_set_union(set, NULL, NULL) == NULL);

    evt_topic_set* merged = c_topic_set_union(set, set, NULL);
    CHECK(merged != NULL);
    if (!merged) return;

    CHECK(!merged->owns_topics);
    CHECK(c_topic_set_contains(merged, "on_order", 0));
    CHECK(c_topic_set_get(merged, "on_order", 0) == set->on_order);
    CHECK(c_topic_set_contains(merged, "realtime.RB2510.TickData", 0));
    CHECK(c_bytemap_contains(merged->rt_tick, "RB2510", 0) == BYTEMAP_OK);

    // A merged set is read-only: get() never creates.
    CHECK(c_topic_set_get(merged, "borrowed.missing", 0) == NULL);
    CHECK(!c_topic_set_contains(merged, "borrowed.missing", 0));

    // Freeing the merged set frees only its own maps; the source topics stay.
    c_topic_set_free(merged);

    md_variant md;
    memset(&md, 0, sizeof(md));
    md.meta_info.ticker = "RB2510";
    md.meta_info.dtype = DTYPE_TICK;
    CHECK(c_md_topic_push(set, &md) != NULL);
    CHECK(set->on_order != NULL);
    CHECK(set->on_order->key != NULL);
    CHECK(strcmp(set->on_order->key, "on_order") == 0);
}

// ========== 8. dtype-name Resolution ==========

static void test_dtype_name_resolution(evt_topic_set* set) {
    CHECK(set != NULL);
    if (!set) return;

    static const struct {
        const char*  name;
        md_data_type dtype;
    } names[] = {
        {"BarData", DTYPE_BAR},
        {"GenericMarketData", DTYPE_MARKET_DATA},
        {"InternalData", DTYPE_INTERNAL},
        {"OrderData", DTYPE_ORDER},
        {"TickDataLite", DTYPE_TICK_LITE},
        {"TickData", DTYPE_TICK},
        {"TradeReport", DTYPE_REPORT},
        {"TradeInstruction", DTYPE_INSTRUCTION},
        {"TransactionData", DTYPE_TRANSACTION},
    };

    // Every generated name resolves, under either compile-time method.
    for (size_t i = 0; i < sizeof(names) / sizeof(names[0]); i++) {
        CHECK(c_topic_set_dtype_from_name(names[i].name, strlen(names[i].name)) == names[i].dtype);
    }

#if TOPIC_SET_MODE_VIGILANT
    // VIGILANT compares in full: a name the set never generates is unknown.
    CHECK(c_topic_set_dtype_from_name("Bogus", 5) == DTYPE_UNKNOWN);
    CHECK(c_topic_set_dtype_from_name("TickDoodle", 10) == DTYPE_UNKNOWN);
    CHECK(c_topic_set_dtype_from_name("Trades", 6) == DTYPE_UNKNOWN);
#else
    // FAST trusts the distinguishing characters.
    CHECK(c_topic_set_dtype_from_name("Bogus", 5) == DTYPE_BAR);
    CHECK(c_topic_set_dtype_from_name("TickDoodle", 10) == DTYPE_TICK);
    CHECK(c_topic_set_dtype_from_name("Trades", 6) == DTYPE_TRANSACTION);
#endif

    // Short or empty names never read past their bytes, in either method.
    CHECK(c_topic_set_dtype_from_name("Ti", 2) == DTYPE_UNKNOWN);
    CHECK(c_topic_set_dtype_from_name("Tr", 2) == DTYPE_UNKNOWN);
    CHECK(c_topic_set_dtype_from_name("T", 1) == DTYPE_UNKNOWN);
    CHECK(c_topic_set_dtype_from_name("", 0) == DTYPE_UNKNOWN);
    CHECK(c_topic_set_dtype_from_name(NULL, 0) == DTYPE_UNKNOWN);
    CHECK(c_topic_set_dtype_from_name(NULL, 4) == DTYPE_UNKNOWN);

    // The compile-time flag picks the method for the parse path.
    evt_topic* bogus = register_key(set, "realtime.RB2510.Bogus");
    CHECK(bogus != NULL);
    if (bogus) {
        const char*  ticker = NULL;
        md_data_type dtype = DTYPE_UNKNOWN;
        int          ret = c_md_topic_parse(set, bogus, &ticker, &dtype);

#if TOPIC_SET_MODE_VIGILANT
        CHECK(ret == TOPIC_SET_ERR_UNKNOWN_DTYPE);
        CHECK(dtype == DTYPE_UNKNOWN);
#else
        CHECK(ret == TOPIC_SET_OK);
        CHECK(dtype == DTYPE_BAR);
        CHECK(ticker && strcmp(ticker, "RB2510") == 0);
#endif
    }

    // Every generated name still parses end to end under either mode.
    static const md_data_type dtypes[] = {
        DTYPE_MARKET_DATA, DTYPE_INTERNAL, DTYPE_TRANSACTION, DTYPE_ORDER,
        DTYPE_TICK_LITE, DTYPE_TICK, DTYPE_BAR, DTYPE_REPORT, DTYPE_INSTRUCTION,
    };

    md_variant md;
    memset(&md, 0, sizeof(md));
    md.meta_info.ticker = "RB2510";

    for (size_t i = 0; i < sizeof(dtypes) / sizeof(dtypes[0]); i++) {
        md.meta_info.dtype = dtypes[i];

        evt_topic*   topic = c_md_topic_push(set, &md);
        const char*  ticker = NULL;
        md_data_type dtype = DTYPE_UNKNOWN;
        CHECK(topic != NULL);
        if (!topic) continue;

        CHECK(c_md_topic_parse(set, topic, &ticker, &dtype) == TOPIC_SET_OK);
        CHECK(dtype == dtypes[i]);
    }
}

// ========== 9. Ownership Lifecycle ==========

static void test_ownership_lifecycle(void) {
    md_variant md;
    memset(&md, 0, sizeof(md));
    md.meta_info.ticker = "RB2510";
    md.meta_info.dtype = DTYPE_TICK;

    evt_topic_set* first = c_topic_set_new(NULL);
    CHECK(first != NULL);
    if (!first) return;

    evt_topic* first_topic = c_md_topic_push(first, &md);
    CHECK(first_topic != NULL);
    CHECK(first->on_order != NULL);
    CHECK(first->on_order->key != NULL);
    CHECK(strcmp(first->on_order->key, "on_order") == 0);

    // Frees every topic (built-ins, dynamic and resolved realtime) and zeroes
    // the named fields.
    c_topic_set_free(first);

    // A fresh set must be able to internalize the same keys again.
    evt_topic_set* second = c_topic_set_new(NULL);
    CHECK(second != NULL);
    if (!second) return;

    evt_topic* second_topic = c_md_topic_push(second, &md);
    CHECK(second_topic != NULL);
    CHECK(c_md_topic_parse(second, second_topic, NULL, NULL) == TOPIC_SET_OK);
    CHECK(second->realtime != NULL);
    CHECK(second->realtime->key != NULL);
    CHECK(strcmp(second->realtime->key, "realtime.{ticker}.{dtype}") == 0);
    CHECK(c_bytemap_len(second->registry) == 10);  // 9 built-ins + one realtime.

    c_topic_set_free(second);
}

// ========== 10. Pure-C Publish / Dispatch ==========

typedef struct handler_ctx {
    _Atomic size_t count;
    _Atomic size_t errors;
    uint32_t       magic;
    evt_topic*     expected_topic;
} handler_ctx;

static void test_handler(void* args, evt_topic* topic, void* user_data) {
    handler_ctx* ctx = (handler_ctx*) user_data;

    if (!args || !topic) {
        atomic_fetch_add_explicit(&ctx->errors, 1, memory_order_relaxed);
        return;
    }
    if (*(const uint32_t*) args != ctx->magic) {
        atomic_fetch_add_explicit(&ctx->errors, 1, memory_order_relaxed);
        return;
    }
    if (ctx->expected_topic && !c_topic_match_bool(ctx->expected_topic, topic)) {
        atomic_fetch_add_explicit(&ctx->errors, 1, memory_order_relaxed);
        return;
    }
    atomic_fetch_add_explicit(&ctx->count, 1, memory_order_release);
}

static void payload_dealloc(evt_message_payload* payload) {
    atomic_fetch_add_explicit(&g_payload_frees, 1, memory_order_relaxed);
    free(payload->args);
    free(payload);
}

static int publish_magic(evt_engine* engine, evt_topic* topic, uint32_t magic) {
    evt_message_payload* payload = (evt_message_payload*) calloc(1, sizeof(evt_message_payload));
    uint32_t*            args = (uint32_t*) malloc(sizeof(uint32_t));
    if (!payload || !args) {
        free(payload);
        free(args);
        return EVT_RET_ERR_OOM;
    }

    *args = magic;
    payload->args = args;
    payload->topic = topic;
    payload->seq_id = 0;
    payload->fn_dealloc = payload_dealloc;
    return c_evt_engine_publish(engine, payload, true, 0, 1.0);
}

typedef struct loop_ctx {
    evt_engine* engine;
} loop_ctx;

static void* engine_loop_main(void* arg) {
    c_evt_engine_loop(((loop_ctx*) arg)->engine);
    return NULL;
}

static double monotonic_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double) ts.tv_sec + (double) ts.tv_nsec / 1e9;
}

static bool wait_counts(handler_ctx* ctxs, size_t n, double timeout_seconds) {
    double deadline = monotonic_seconds() + timeout_seconds;

    for (;;) {
        size_t ready = 0;
        for (size_t i = 0; i < n; i++) {
            if (atomic_load_explicit(&ctxs[i].count, memory_order_acquire) > 0) ready++;
        }
        if (ready == n) return true;
        if (monotonic_seconds() >= deadline) return false;

        struct timespec interval = {0, 200000};  // 200 us
        nanosleep(&interval, NULL);
    }
}

static void test_engine_dispatch(evt_topic_set* set) {
    evt_engine* engine = c_evt_engine_new(NULL);
    CHECK(engine != NULL);
    if (!engine) return;

    builtin_case cases[9];
    size_t       n_cases = make_builtin_cases(set, cases, sizeof(cases) / sizeof(cases[0]));

    static handler_ctx ctx[9];
    memset(ctx, 0, sizeof(ctx));

    for (size_t i = 0; i < n_cases; i++) {
        handler_ctx* h = &ctx[i];
        atomic_init(&h->count, 0);
        atomic_init(&h->errors, 0);
        h->magic = (uint32_t) (i + 1);
        h->expected_topic = cases[i].topic;

        // Register on the canonical built-in topic the set owns.
        int ret = c_evt_engine_register_handler(
            engine,
            cases[i].topic,
            (const void*) test_handler,
            EVT_CALLBACK_WITH_ARGS_TOPIC_USERDATA,
            h,
            false
        );
        CHECK(ret == EVT_RET_OK);
    }

    // A template topic cannot be published directly (not exact).
    {
        evt_message_payload* payload = (evt_message_payload*) calloc(1, sizeof(evt_message_payload));
        uint32_t*            args = (uint32_t*) malloc(sizeof(uint32_t));
        CHECK(payload != NULL && args != NULL);
        if (payload && args) {
            *args = 0;
            payload->args = args;
            payload->topic = set->realtime;
            payload->fn_dealloc = payload_dealloc;
            CHECK(c_evt_engine_publish(engine, payload, true, 0, 1.0) == EVT_RET_ERR_INVALID_TOPIC);
            payload_dealloc(payload);
        }
    }

    // Resolve the template instances that will be published.
    allocator_protocol* allocator = c_ap_protocol_from_ptr(set->registry);
    evt_topic*          launch = c_topic_new("launch_order.RB2510", 0, allocator);
    evt_topic*          cancel = c_topic_new("cancel_order.RB2510", 0, allocator);

    md_variant md;
    memset(&md, 0, sizeof(md));
    md.meta_info.ticker = "RB2510";
    md.meta_info.dtype = DTYPE_TICK;

    evt_topic* realtime = c_md_topic_push(set, &md);
    CHECK(launch && cancel && realtime);

    // Start the pure C dispatch loop on a separate thread (no GIL involvement).
    atomic_store_explicit(&engine->active, true, memory_order_release);
    loop_ctx  lc = {.engine = engine};
    pthread_t thread;
    CHECK(pthread_create(&thread, NULL, engine_loop_main, &lc) == 0);

    size_t published = 0;

    for (size_t i = 0; i < n_cases; i++) {
        evt_topic* topic = cases[i].topic;
        if (strncmp(cases[i].key, "launch_order.", 13) == 0) {
            topic = launch;
        } else if (strncmp(cases[i].key, "cancel_order.", 13) == 0) {
            topic = cancel;
        } else if (strncmp(cases[i].key, "realtime.", 9) == 0) {
            topic = realtime;
        }

        CHECK(topic != NULL);
        if (!topic) continue;

        CHECK(publish_magic(engine, topic, ctx[i].magic) == EVT_RET_OK);
        published++;
    }

    bool all_seen = wait_counts(ctx, n_cases, 5.0);
    CHECK(all_seen);

    // Give any erroneous extra delivery a chance to show up.
    struct timespec settle = {0, 100000000};  // 100 ms
    nanosleep(&settle, NULL);

    for (size_t i = 0; i < n_cases; i++) {
        CHECK(atomic_load_explicit(&ctx[i].count, memory_order_acquire) == 1);
        CHECK(atomic_load_explicit(&ctx[i].errors, memory_order_acquire) == 0);
    }

    // Stop the loop and drain.
    atomic_store_explicit(&engine->active, false, memory_order_release);
    pthread_join(thread, NULL);

    size_t handled = 0;
    for (size_t i = 0; i < n_cases; i++) {
        handled += atomic_load_explicit(&ctx[i].count, memory_order_acquire);
    }

    CHECK(published == 9);
    CHECK(handled == published);
    // The published payloads plus the one rejected (generic topic) payload.
    CHECK(atomic_load_explicit(&g_payload_frees, memory_order_acquire) == published + 1);

    printf("[c_event_engine] topics=%zu handled=%zu payload_frees=%zu\n",
           published, handled, atomic_load_explicit(&g_payload_frees, memory_order_acquire));

    c_evt_engine_free(engine);
}

// ========== 11. Throughput ==========

#define BENCH_N_TICKERS 5
#define BENCH_N_DTYPES 8

static void run_benchmark(evt_topic_set* set) {
    static const char* const tickers[BENCH_N_TICKERS] = {"RB2510", "600010.SH", "IF2412", "au2512", "SR501"};
    static const md_data_type dtypes[BENCH_N_DTYPES] = {
        DTYPE_INTERNAL, DTYPE_TRANSACTION, DTYPE_ORDER, DTYPE_TICK_LITE,
        DTYPE_TICK, DTYPE_BAR, DTYPE_REPORT, DTYPE_INSTRUCTION,
    };

    const size_t iterations = 2000000;
    md_variant   md[BENCH_N_TICKERS][BENCH_N_DTYPES];
    evt_topic*   topics[BENCH_N_TICKERS][BENCH_N_DTYPES];

    for (size_t i = 0; i < BENCH_N_TICKERS; i++) {
        for (size_t j = 0; j < BENCH_N_DTYPES; j++) {
            memset(&md[i][j], 0, sizeof(md_variant));
            md[i][j].meta_info.ticker = tickers[i];
            md[i][j].meta_info.dtype = dtypes[j];
            topics[i][j] = c_md_topic_push(set, &md[i][j]);
        }
    }

    // The inputs vary each iteration and the asm barrier prevents loop-invariant
    // hoisting, so the numbers reflect real bytemap work.
    volatile uintptr_t sink = 0;
    size_t             n_cases = BENCH_N_TICKERS * BENCH_N_DTYPES;

    double start = monotonic_seconds();
    for (size_t i = 0; i < iterations; i++) {
        size_t     idx = i % n_cases;
        evt_topic* topic = c_md_topic_push(set, &md[idx / BENCH_N_DTYPES][idx % BENCH_N_DTYPES]);
        sink ^= (uintptr_t) topic;
        __asm__ __volatile__("" ::: "memory");
    }
    double push_seconds = monotonic_seconds() - start;

    start = monotonic_seconds();
    for (size_t i = 0; i < iterations; i++) {
        size_t      idx = i % n_cases;
        const char* ticker = NULL;
        md_data_type dtype = DTYPE_UNKNOWN;
        int          ret = c_md_topic_parse(set, topics[idx / BENCH_N_DTYPES][idx % BENCH_N_DTYPES], &ticker, &dtype);
        sink ^= (uintptr_t) ticker ^ (uintptr_t) ret ^ (uintptr_t) dtype;
        __asm__ __volatile__("" ::: "memory");
    }
    double parse_seconds = monotonic_seconds() - start;

    // Production hot path: one symbol per handler, always a ticker-map hit.
    start = monotonic_seconds();
    for (size_t i = 0; i < iterations; i++) {
        evt_topic* topic = c_md_topic_push(set, &md[0][4]);  // DTYPE_TICK
        sink ^= (uintptr_t) topic;
        __asm__ __volatile__("" ::: "memory");
    }
    double hot_push_seconds = monotonic_seconds() - start;

    printf(
        "[bench] push (mixed %zu cases): %.1f ns/op (%.2f M ops/s)\n"
        "[bench] push (single symbol)  : %.1f ns/op (%.2f M ops/s)\n"
        "[bench] parse (%zu topics)    : %.1f ns/op (%.2f M ops/s)\n",
        n_cases, push_seconds * 1e9 / (double) iterations, (double) iterations / push_seconds / 1e6,
        hot_push_seconds * 1e9 / (double) iterations, (double) iterations / hot_push_seconds / 1e6,
        n_cases, parse_seconds * 1e9 / (double) iterations, (double) iterations / parse_seconds / 1e6
    );

    (void) sink;
}

// ========== Main ==========

int main(void) {
    printf("[c_event_engine] TOPIC_SET_MODE_VIGILANT=%d\n", TOPIC_SET_MODE_VIGILANT);

    test_variant_surface();
    test_ownership_lifecycle();

    evt_topic_set* set = c_topic_set_new(NULL);
    if (!set) {
        fprintf(stderr, "FATAL: c_topic_set_new returned NULL\n");
        return 1;
    }

    test_topic_set_basics(set);
    test_templates(set);
    test_push_parse_all_dtypes(set);
    test_dynamic_registration(set);
    test_registry_cache_sync(set);
    test_union(set);
    test_dtype_name_resolution(set);
    test_engine_dispatch(set);
    run_benchmark(set);

    c_topic_set_free(set);

    printf("[c_event_engine] %zu checks, %zu failure(s)\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}

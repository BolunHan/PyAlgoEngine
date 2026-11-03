/*
 * Pure-C test for algo_engine/engine/c_market_engine.h.
 *
 * The test links nothing but libc/libpthread/libm: every C function under test
 * is static inline in the headers. It covers:
 *
 *   1. The mds_monitor base: lifecycle, uid generation, generic md_filter.
 *   2. The mds_monitor_manager base: registry, observer callbacks with
 *      self-removal during dispatch, enabled flag.
 *   3. C-level inheritance: a derived manager embeds the manager base at
 *      offset 0 and recovers itself from the base pointer.
 *   4. The mds core: subscriptions in the bytemap, session_datetime_t ts,
 *      both registries, clear/free.
 *   5. The hot path: c_mds_on_market_data inlines the subscription update and
 *      returns MDS_OK_SUBSCRIBED on first sight, with no out-parameter.
 *   6. Rejections: internal data and unknown dtypes.
 *   7. Bytemap growth beyond DEFAULT_BYTEMAP_CAPACITY.
 *
 * Build & run with run_c_market_engine_c_test.sh.
 */

// Feature-test macro for the POSIX declarations (strnlen / usleep) pulled in
// by the allocator and market-data headers. Must precede every system include.
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <algo_engine/engine/c_market_engine.h>

// ========== Assertion Helper ==========

static size_t g_checks   = 0;
static size_t g_failures = 0;

/* Root allocator for the stack-embedded managers in this harness. */
static allocator_protocol* g_allocator = NULL;

#define CHECK(cond)                                                          \
    do {                                                                     \
        g_checks++;                                                          \
        if (!(cond)) {                                                       \
            g_failures++;                                                    \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);  \
        }                                                                    \
    } while (0)

// ========== C-Level Inherited Manager ==========

/* A derived manager in the C sense: the base must be the first member, and
   the callback recovers the derived pointer from the base pointer alone. */
typedef struct test_manager {
    mds_monitor_manager base;
    size_t              n_feeds;
    const char*         last_ticker;
    double              last_price;
    size_t              n_user_data_hits;
} test_manager;

static void test_manager_on_feed(mds_monitor_manager* manager, const md_variant* market_data, void* user_data) {
    test_manager* self = (test_manager*) manager;
    self->n_feeds++;
    self->last_ticker = market_data->meta_info.ticker;
    self->last_price  = c_md_get_price(market_data);
    if (user_data == (void*) self) self->n_user_data_hits++;
}

/* A callback that removes itself while the manager is walking the list. */
typedef struct self_unregister_manager {
    mds_monitor_manager base;
    uintptr_t           callback_id;
    size_t              n_calls;
} self_unregister_manager;

static void self_unregister_on_feed(mds_monitor_manager* manager, const md_variant* market_data, void* user_data) {
    self_unregister_manager* self = (self_unregister_manager*) manager;
    (void) market_data;
    (void) user_data;
    self->n_calls++;
    if (self->callback_id) {
        c_mds_monitor_manager_unregister_callback(manager, self->callback_id);
        self->callback_id = 0;
    }
}

// ========== Observation Builders ==========

static session_datetime_t make_dt(double timestamp) {
    session_datetime_t dt;
    memset(&dt, 0, sizeof(dt));
    dt.unix_ts = timestamp;
    dt.ts      = timestamp;
    dt.ordinal = (uint32_t) timestamp;
    dt.date.year  = 2024;
    dt.date.month = 11;
    dt.date.day   = 11;
    return dt;
}

static md_variant make_transaction(const char* ticker, double price, double timestamp) {
    md_variant market_data;
    memset(&market_data, 0, sizeof(market_data));
    market_data.meta_info.dtype     = DTYPE_TRANSACTION;
    market_data.meta_info.ticker    = ticker;
    market_data.meta_info.timestamp = timestamp;
    market_data.meta_info.dt        = make_dt(timestamp);
    market_data.transaction_data.price = price;
    return market_data;
}

static md_variant make_tick_lite(const char* ticker, double price, double timestamp) {
    md_variant market_data;
    memset(&market_data, 0, sizeof(market_data));
    market_data.meta_info.dtype     = DTYPE_TICK_LITE;
    market_data.meta_info.ticker    = ticker;
    market_data.meta_info.timestamp = timestamp;
    market_data.meta_info.dt        = make_dt(timestamp);
    market_data.tick_data_lite.last_price = price;
    return market_data;
}

/* The subscription value lives in the bytemap; read it through get_ptr. */
static const mds_subscription* subscription_at(mds* service, const char* ticker) {
    char*  stored = NULL;
    size_t stored_len = 0;

    if (c_bytemap_ex_get_ptr(service->subscriptions, ticker, strlen(ticker), &stored, &stored_len) != BYTEMAP_OK) return NULL;
    if (stored_len != sizeof(mds_subscription)) return NULL;

    return (const mds_subscription*) stored;
}

static mds_monitor make_monitor(const char* name) {
    mds_monitor monitor;
    memset(&monitor, 0, sizeof(monitor));
    c_mds_monitor_init(&monitor, name, 0, (md_filter_flag) 0, g_allocator);
    c_mds_monitor_gen_uid(&monitor, name, strlen(name));
    return monitor;
}

// ========== 1. Monitor Base ==========

static void test_monitor_base(void) {
    mds_monitor monitor;
    memset(&monitor, 0, sizeof(monitor));

    CHECK(c_mds_monitor_init(&monitor, "BASE", 0, (md_filter_flag) 0, g_allocator) == MDS_OK);
    CHECK(monitor.enabled);
    CHECK(monitor.is_ready);
    CHECK(monitor.name != NULL && strcmp(monitor.name, "BASE") == 0);

    c_mds_monitor_gen_uid(&monitor, monitor.name, strlen(monitor.name));

    mds_monitor_id zero_id = {0};
    CHECK(memcmp(monitor.monitor_id, zero_id, sizeof(mds_monitor_id)) != 0);

    md_variant trade = make_transaction("BASE", 1.0, 1.0);
    CHECK(c_mds_monitor_filter_md(&monitor, &trade));

    monitor.md_filter = MD_FILTER_NO_TRADE;
    CHECK(!c_mds_monitor_filter_md(&monitor, &trade));

    c_mds_monitor_set_enabled(&monitor, false);
    CHECK(!c_mds_monitor_is_enabled(&monitor));

    c_mds_monitor_dealloc(&monitor);
    CHECK(monitor.name == NULL);

    mds_monitor* allocated = c_mds_monitor_new("ALLOCATED", 0, (md_filter_flag) 0, NULL);
    CHECK(allocated != NULL);
    c_mds_monitor_free(allocated);
    c_mds_monitor_free(NULL);
}

// ========== 2. Manager Base ==========

static void test_manager_base(void) {
    test_manager manager;
    memset(&manager, 0, sizeof(manager));

    CHECK(offsetof(test_manager, base) == 0);
    CHECK(&manager.base == (mds_monitor_manager*) &manager);
    CHECK(c_mds_monitor_manager_init(&manager.base, g_allocator) == MDS_OK);
    CHECK(c_mds_monitor_manager_is_enabled(&manager.base));
    CHECK(c_bytemap_ex_len(manager.base.monitors) == 0);

    uintptr_t callback_id = 0;
    CHECK(c_mds_monitor_manager_register_callback(&manager.base, test_manager_on_feed, &manager, &callback_id) == MDS_OK);
    CHECK(callback_id != 0);

    md_variant market_data = make_transaction("AAA", 12.5, 100.0);
    c_mds_monitor_manager_feed(&manager.base, &market_data);
    CHECK(manager.n_feeds == 1);
    CHECK(strcmp(manager.last_ticker, "AAA") == 0);
    CHECK(manager.last_price == 12.5);
    CHECK(manager.n_user_data_hits == 1);

    CHECK(c_mds_monitor_manager_unregister_callback(&manager.base, callback_id) == MDS_OK);
    CHECK(c_mds_monitor_manager_unregister_callback(&manager.base, callback_id) == MDS_ERR_NOT_FOUND);
    c_mds_monitor_manager_feed(&manager.base, &market_data);
    CHECK(manager.n_feeds == 1);

    // Disabled managers drop the feed.
    c_mds_monitor_manager_register_callback(&manager.base, test_manager_on_feed, &manager, NULL);
    c_mds_monitor_manager_set_enabled(&manager.base, false);
    c_mds_monitor_manager_feed(&manager.base, &market_data);
    CHECK(manager.n_feeds == 1);
    c_mds_monitor_manager_set_enabled(&manager.base, true);
    c_mds_monitor_manager_feed(&manager.base, &market_data);
    CHECK(manager.n_feeds == 2);

    c_mds_monitor_manager_dealloc(&manager.base);
    CHECK(manager.base.callbacks == NULL);
    CHECK(manager.base.monitors == NULL);
}

static void test_self_unregister_during_feed(void) {
    self_unregister_manager manager;
    memset(&manager, 0, sizeof(manager));
    c_mds_monitor_manager_init(&manager.base, g_allocator);

    CHECK(c_mds_monitor_manager_register_callback(&manager.base, self_unregister_on_feed, NULL, &manager.callback_id) == MDS_OK);

    md_variant market_data = make_transaction("SELF", 2.0, 1.0);
    c_mds_monitor_manager_feed(&manager.base, &market_data);
    CHECK(manager.n_calls == 1);
    CHECK(manager.callback_id == 0);
    CHECK(manager.base.callbacks == NULL);

    c_mds_monitor_manager_feed(&manager.base, &market_data);
    CHECK(manager.n_calls == 1);

    c_mds_monitor_manager_dealloc(&manager.base);
}

static void test_manager_registry(void) {
    mds_monitor_manager* manager = c_mds_monitor_manager_new(NULL);
    CHECK(manager != NULL);

    mds_monitor monitor = make_monitor("REG");
    CHECK(c_mds_monitor_manager_register_monitor(manager, &monitor) == MDS_OK);
    CHECK(c_bytemap_ex_len(manager->monitors) == 1);
    CHECK(c_mds_monitor_manager_get_monitor(manager, monitor.monitor_id) == &monitor);
    CHECK(c_mds_monitor_manager_unregister_monitor(manager, &monitor) == MDS_OK);
    CHECK(c_bytemap_ex_len(manager->monitors) == 0);
    CHECK(c_mds_monitor_manager_get_monitor(manager, monitor.monitor_id) == NULL);
    CHECK(c_mds_monitor_manager_unregister_monitor(manager, &monitor) == MDS_ERR_NOT_FOUND);

    c_mds_monitor_dealloc(&monitor);
    c_mds_monitor_manager_free(manager);
    CHECK(c_mds_monitor_manager_init(NULL, NULL) == MDS_ERR_INVALID_INPUT);
    c_mds_monitor_manager_free(NULL);
}

// ========== 3. Service Lifecycle ==========

static void test_service_lifecycle(void) {
    mds* service = c_mds_new(NULL);
    CHECK(service != NULL);
    CHECK(service->subscriptions != NULL);
    CHECK(service->monitors != NULL);
    CHECK(c_bytemap_ex_len(service->subscriptions) == 0);
    CHECK(c_mds_n_monitors(service) == 0);
    CHECK(service->ts.date.year == 0);
    CHECK(service->monitor_manager == NULL);

    CHECK(c_mds_clear(service) == MDS_OK);
    CHECK(service->ts.date.year == 0);

    c_mds_free(service);
    CHECK(c_mds_clear(NULL) == MDS_ERR_INVALID_INPUT);
    c_mds_free(NULL);
}

// ========== 4. Subscription Registration & Hot Path ==========

static void test_subscription_registration(void) {
    mds* service = c_mds_new(NULL);
    CHECK(service != NULL);

    md_variant first = make_transaction("AAA", 10.5, 100.0);

    // The hot path returns the new-subscription status, no out-parameter.
    CHECK(c_mds_on_market_data(service, &first) == MDS_OK_SUBSCRIBED);
    CHECK(c_bytemap_ex_len(service->subscriptions) == 1);
    CHECK(service->ts.unix_ts == 100.0);
    CHECK(service->ts.date.year == 2024);
    CHECK(c_mds_get_market_price(service, "AAA") == 10.5);
    CHECK(isnan(c_mds_get_market_price(service, "NO_SUCH_TICKER")));
    CHECK(isnan(c_mds_get_market_price(NULL, "AAA")));
    CHECK(isnan(c_mds_get_market_price(service, NULL)));

    const mds_subscription* subscription = subscription_at(service, "AAA");
    CHECK(subscription != NULL);
    CHECK(subscription->n_feeds == 1);
    CHECK(subscription->last_price == 10.5);
    CHECK(subscription->last_update.unix_ts == 100.0);
    CHECK(subscription->last_update.date.year == 2024);
    CHECK(subscription->is_subscribed_td);
    CHECK(!subscription->is_subscribed_tk_lite);

    md_variant second = make_transaction("AAA", 11.5, 101.0);
    CHECK(c_mds_on_market_data(service, &second) == MDS_OK);
    CHECK(c_bytemap_ex_len(service->subscriptions) == 1);
    CHECK(service->ts.unix_ts == 101.0);
    CHECK(c_mds_get_market_price(service, "AAA") == 11.5);
    CHECK(subscription_at(service, "AAA")->n_feeds == 2);

    md_variant tick = make_tick_lite("AAA", 12.5, 102.0);
    CHECK(c_mds_on_market_data(service, &tick) == MDS_OK);
    subscription = subscription_at(service, "AAA");
    CHECK(subscription->is_subscribed_td);
    CHECK(subscription->is_subscribed_tk_lite);
    CHECK(subscription->n_feeds == 3);
    CHECK(subscription->last_price == 12.5);

    md_variant other = make_transaction("BBB", 20.0, 103.0);
    CHECK(c_mds_on_market_data(service, &other) == MDS_OK_SUBSCRIBED);
    CHECK(c_bytemap_ex_len(service->subscriptions) == 2);
    CHECK(c_mds_get_market_price(service, "BBB") == 20.0);

    // The standalone update entry agrees with the inlined hot path.
    md_variant third = make_transaction("CCC", 30.0, 104.0);
    CHECK(c_mds_update_subscription(service, &third) == MDS_OK_SUBSCRIBED);
    CHECK(c_mds_update_subscription(service, &third) == MDS_OK);
    CHECK(c_mds_get_market_price(service, "CCC") == 30.0);

    c_mds_free(service);
}

// ========== 5. Rejections ==========

static void test_rejections(void) {
    mds* service = c_mds_new(NULL);

    md_variant internal_data;
    memset(&internal_data, 0, sizeof(internal_data));
    internal_data.meta_info.dtype  = DTYPE_INTERNAL;
    internal_data.meta_info.ticker = "INT";

    CHECK(c_mds_on_market_data(service, &internal_data) == MDS_ERR_INTERNAL_DATA);
    CHECK(c_bytemap_ex_len(service->subscriptions) == 0);

    md_variant unknown;
    memset(&unknown, 0, sizeof(unknown));
    unknown.meta_info.dtype  = DTYPE_UNKNOWN;
    unknown.meta_info.ticker = "UNKNOWN";
    CHECK(c_mds_on_market_data(service, &unknown) == MDS_ERR_UNKNOWN_DTYPE);
    CHECK(c_bytemap_ex_len(service->subscriptions) == 0);

    md_variant no_ticker;
    memset(&no_ticker, 0, sizeof(no_ticker));
    no_ticker.meta_info.dtype  = DTYPE_TRANSACTION;
    no_ticker.meta_info.ticker = NULL;
    CHECK(c_mds_on_market_data(service, &no_ticker) == MDS_ERR_INVALID_INPUT);

    // Internal data does not subscribe, but reaches the manager.
    test_manager manager;
    memset(&manager, 0, sizeof(manager));
    c_mds_monitor_manager_init(&manager.base, g_allocator);
    c_mds_monitor_manager_register_callback(&manager.base, test_manager_on_feed, &manager, NULL);
    c_mds_set_manager(service, &manager.base);

    CHECK(c_mds_on_internal_data(service, (const md_internal*) &internal_data) == MDS_OK);
    CHECK(manager.n_feeds == 1);
    CHECK(strcmp(manager.last_ticker, "INT") == 0);
    CHECK(c_bytemap_ex_len(service->subscriptions) == 0);

    c_mds_free(service);
}

// ========== 6. Manager Dispatch & Registries ==========

static void test_manager_dispatch(void) {
    mds* service = c_mds_new(NULL);

    test_manager manager;
    memset(&manager, 0, sizeof(manager));
    c_mds_monitor_manager_init(&manager.base, g_allocator);
    c_mds_monitor_manager_register_callback(&manager.base, test_manager_on_feed, &manager, NULL);

    CHECK(c_mds_set_manager(service, &manager.base) == MDS_OK);
    CHECK(c_mds_get_manager(service) == &manager.base);

    mds_monitor monitor = make_monitor("DISPATCH");
    CHECK(c_mds_register_monitor(service, &monitor) == MDS_OK);
    CHECK(c_mds_register_monitor(service, &monitor) == MDS_OK);
    CHECK(c_mds_n_monitors(service) == 1);
    CHECK(c_mds_get_monitor(service, monitor.monitor_id) == &monitor);

    md_variant market_data = make_transaction("DISPATCH", 7.5, 50.0);
    CHECK(c_mds_on_market_data(service, &market_data) == MDS_OK_SUBSCRIBED);
    CHECK(manager.n_feeds == 1);
    CHECK(strcmp(manager.last_ticker, "DISPATCH") == 0);
    CHECK(manager.last_price == 7.5);

    CHECK(c_mds_unregister_monitor(service, &monitor) == MDS_OK);
    CHECK(c_mds_n_monitors(service) == 0);
    CHECK(c_mds_unregister_monitor(service, &monitor) == MDS_ERR_NOT_FOUND);

    CHECK(c_mds_set_manager(service, NULL) == MDS_OK);
    CHECK(c_mds_get_manager(service) == NULL);
    CHECK(c_mds_on_market_data(service, &market_data) == MDS_OK);
    CHECK(manager.n_feeds == 1);

    c_mds_monitor_dealloc(&monitor);
    c_mds_free(service);
}

// ========== 7. Clear & Reuse ==========

static void test_clear_and_reuse(void) {
    mds* service = c_mds_new(NULL);

    mds_monitor monitor = make_monitor("REUSE");
    c_mds_register_monitor(service, &monitor);

    md_variant market_data = make_transaction("REUSE", 5.0, 10.0);
    CHECK(c_mds_on_market_data(service, &market_data) == MDS_OK_SUBSCRIBED);
    CHECK(c_bytemap_ex_len(service->subscriptions) == 1);

    CHECK(c_mds_clear(service) == MDS_OK);
    CHECK(c_bytemap_ex_len(service->subscriptions) == 0);
    CHECK(c_mds_n_monitors(service) == 0);
    CHECK(service->ts.date.year == 0);
    CHECK(isnan(c_mds_get_market_price(service, "REUSE")));

    CHECK(c_mds_on_market_data(service, &market_data) == MDS_OK_SUBSCRIBED);
    CHECK(c_bytemap_ex_len(service->subscriptions) == 1);
    CHECK(c_mds_get_market_price(service, "REUSE") == 5.0);

    // _dealloc leaves a ZEROED struct; _init turns it back into a live service.
    c_mds_dealloc(service);
    CHECK(service->subscriptions == NULL);
    CHECK(service->monitors == NULL);
    CHECK(service->ts.date.year == 0);

    CHECK(c_mds_init(service, NULL) == MDS_OK);
    CHECK(service->subscriptions != NULL);
    CHECK(service->monitors != NULL);
    CHECK(service->ts.date.year == 0);

    c_mds_dealloc(service);
    c_ap_free(service);

    c_mds_monitor_dealloc(&monitor);
}

// ========== 8. Bytemap Growth ==========

static void test_subscription_growth(void) {
    mds* service = c_mds_new(NULL);
    CHECK(service != NULL);

    char ticker[32];
    size_t n_tickers = DEFAULT_BYTEMAP_CAPACITY + 2;

    for (size_t index = 0; index < n_tickers; index++) {
        snprintf(ticker, sizeof(ticker), "T%zu", index);
        md_variant market_data = make_transaction(ticker, (double) index + 1.0, (double) index);
        CHECK(c_mds_on_market_data(service, &market_data) == MDS_OK_SUBSCRIBED);
    }

    CHECK(c_bytemap_ex_len(service->subscriptions) == n_tickers);

    for (size_t index = 0; index < n_tickers; index++) {
        snprintf(ticker, sizeof(ticker), "T%zu", index);
        CHECK(c_mds_get_market_price(service, ticker) == (double) index + 1.0);
    }

    c_mds_free(service);
}

// ========== Main ==========

int main(void) {
    void* root = c_ap_alloc(1, NULL);
    g_allocator = c_ap_protocol_from_ptr(root);

    test_monitor_base();
    test_manager_base();
    test_self_unregister_during_feed();
    test_manager_registry();
    test_service_lifecycle();
    test_subscription_registration();
    test_rejections();
    test_manager_dispatch();
    test_clear_and_reuse();
    test_subscription_growth();

    c_ap_free(root);

    printf("%zu checks, %zu failures\n", g_checks, g_failures);
    return g_failures ? 1 : 0;
}

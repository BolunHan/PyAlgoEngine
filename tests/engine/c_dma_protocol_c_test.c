/*
 * Pure-C test for algo_engine/engine/c_dma_protocol.h.
 *
 * Covers:
 *   1. Lifecycle, the two launch/cancel topic caches, and configuration.
 *   2. Direct callback registration/unregistration with the event engine.
 *   3. The single static payload: a second in-flight send is refused as BUSY.
 *   4. Topic routing: order data to realtime, a trade instruction to on_order,
 *      an exchange report to on_report.
 *   5. Ownership: a heap source is copied, an in-shm source is transferred.
 *   6. The DMA-compatible launch/cancel interface on launch_order/cancel_order.
 *   7. The ret-code enum values.
 *
 * Build & run with run_c_dma_protocol_c_test.sh. The only headers involved
 * are the static-inline C headers of this repo, PyCyBase and PyEventEngine.
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
#include <unistd.h>

#include <algo_engine/engine/c_dma.h>
#include <algo_engine/engine/c_dma_protocol.h>
#include <cbase/allocator_protocol/c_allocator_protocol.h>
#include <cbase/allocator_protocol/c_shm_comp.h>
#include <cbase/intern_string/c_intern_string.h>
#include <event_engine/capi/c_engine.h>

// ========== Assertion Helper ==========

static size_t g_checks = 0;
static size_t g_failures = 0;

#define CHECK(cond)                                                          \
    do {                                                                     \
        g_checks++;                                                          \
        if (!(cond)) {                                                       \
            g_failures++;                                                    \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);  \
        }                                                                    \
    } while (0)

// ========== Shared State ==========

static _Atomic size_t g_ticks = 0;
static _Atomic size_t g_orders = 0;
static _Atomic size_t g_instructions = 0;
static _Atomic size_t g_reports = 0;
static _Atomic size_t g_launches = 0;
static _Atomic size_t g_cancels = 0;
static _Atomic size_t g_errors = 0;

static char   g_last_ticker[64];
static double g_last_price = 0.;

static void record(const md_variant* md) {
    const char* ticker = md->meta_info.ticker;

    if (ticker) {
        size_t len = strlen(ticker);
        if (len >= sizeof(g_last_ticker)) len = sizeof(g_last_ticker) - 1;
        memcpy(g_last_ticker, ticker, len);
        g_last_ticker[len] = '\0';
    }
    else {
        g_last_ticker[0] = '\0';
    }
}

static void tick_listener(const md_variant* md, const evt_topic* topic, void* user_data) {
    (void) user_data;
    if (!md || !topic) {
        atomic_fetch_add_explicit(&g_errors, 1, memory_order_relaxed);
        return;
    }
    if (md->meta_info.dtype != DTYPE_TICK_LITE) return;

    record(md);
    g_last_price = md->tick_data_lite.last_price;
    atomic_fetch_add_explicit(&g_ticks, 1, memory_order_release);
}

static void order_listener(const md_variant* md, const evt_topic* topic, void* user_data) {
    (void) user_data;
    if (!md || !topic) {
        atomic_fetch_add_explicit(&g_errors, 1, memory_order_relaxed);
        return;
    }
    if (md->meta_info.dtype != DTYPE_ORDER) return;

    record(md);
    g_last_price = md->order_data.price;
    atomic_fetch_add_explicit(&g_orders, 1, memory_order_release);
}

static void instruction_listener(const md_variant* md, const evt_topic* topic, void* user_data) {
    (void) user_data;
    if (!md || !topic) {
        atomic_fetch_add_explicit(&g_errors, 1, memory_order_relaxed);
        return;
    }
    if (md->meta_info.dtype != DTYPE_INSTRUCTION) return;

    record(md);
    g_last_price = md->trade_instruction.volume;
    atomic_fetch_add_explicit(&g_instructions, 1, memory_order_release);
}

static void report_listener(const md_variant* md, const evt_topic* topic, void* user_data) {
    (void) user_data;
    if (!md || !topic) {
        atomic_fetch_add_explicit(&g_errors, 1, memory_order_relaxed);
        return;
    }
    if (md->meta_info.dtype != DTYPE_REPORT) return;

    record(md);
    g_last_price = md->trade_report.price;
    atomic_fetch_add_explicit(&g_reports, 1, memory_order_release);
}

static void launch_listener(const md_variant* md, const evt_topic* topic, void* user_data) {
    (void) user_data;
    if (!md || !topic) {
        atomic_fetch_add_explicit(&g_errors, 1, memory_order_relaxed);
        return;
    }
    if (md->meta_info.dtype != DTYPE_INSTRUCTION) return;

    record(md);
    g_last_price = md->trade_instruction.volume;
    atomic_fetch_add_explicit(&g_launches, 1, memory_order_release);
}

static void cancel_listener(const md_variant* md, const evt_topic* topic, void* user_data) {
    (void) user_data;
    if (!md || !topic) {
        atomic_fetch_add_explicit(&g_errors, 1, memory_order_relaxed);
        return;
    }
    if (md->meta_info.dtype != DTYPE_INSTRUCTION) return;

    record(md);
    g_last_price = md->trade_instruction.volume;
    atomic_fetch_add_explicit(&g_cancels, 1, memory_order_release);
}

// ========== Helpers ==========

static double monotonic_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double) ts.tv_sec + (double) ts.tv_nsec / 1e9;
}

static bool wait_for(_Atomic size_t* counter, size_t expected, double timeout_seconds) {
    const double deadline = monotonic_seconds() + timeout_seconds;

    for (;;) {
        if (atomic_load_explicit(counter, memory_order_acquire) >= expected) return true;
        if (monotonic_seconds() >= deadline) return false;

        struct timespec interval = {0, 200000};  // 200 us
        nanosleep(&interval, NULL);
    }
}

typedef struct loop_ctx {
    evt_engine* engine;
} loop_ctx;

static void* engine_loop_main(void* arg) {
    c_evt_engine_loop(((loop_ctx*) arg)->engine);
    return NULL;
}

// ========== Test ==========

static void test_handler_compatibility(void) {
    dma_order_launch_func launch_handler = c_dma_protocol_launch_order;
    dma_order_cancel_func cancel_handler = c_dma_protocol_cancel_order;

    CHECK(launch_handler != NULL);
    CHECK(cancel_handler != NULL);
}

static void test_ret_codes(void) {
    CHECK(DMA_PROTOCOL_OK == 0);
    CHECK(DMA_PROTOCOL_ERR_INVALID_INPUT == -1);
    CHECK(DMA_PROTOCOL_ERR_OOM == -2);
    CHECK(DMA_PROTOCOL_ERR_NO_SHM == -3);
    CHECK(DMA_PROTOCOL_ERR_NO_TOPIC == -4);
    CHECK(DMA_PROTOCOL_ERR_PUBLISH == -5);
    CHECK(DMA_PROTOCOL_ERR_ENGINE == -6);
    CHECK(DMA_PROTOCOL_ERR_NOT_FOUND == -7);
    CHECK(DMA_PROTOCOL_ERR_BUSY == -8);
}

static void test_dma_topics(evt_topic_set* topics) {
    md_trade_instruction order;
    memset(&order, 0, sizeof(order));
    order.meta_info.ticker = "CDMA1";

    evt_topic* launch = c_md_topic_dma_launch(topics, &order);
    CHECK(launch != NULL);
    if (launch) {
        CHECK(launch->key_len == strlen("launch_order.CDMA1"));
        CHECK(memcmp(launch->key, "launch_order.CDMA1", launch->key_len) == 0);
        CHECK(c_md_topic_dma_launch(topics, &order) == launch);
        CHECK(c_bytemap_contains(topics->evt_launch, "CDMA1", 0) == BYTEMAP_OK);
    }

    evt_topic* cancel = c_md_topic_dma_cancel(topics, &order);
    CHECK(cancel != NULL);
    if (cancel) {
        CHECK(cancel->key_len == strlen("cancel_order.CDMA1"));
        CHECK(memcmp(cancel->key, "cancel_order.CDMA1", cancel->key_len) == 0);
        CHECK(c_md_topic_dma_cancel(topics, &order) == cancel);
        CHECK(c_bytemap_contains(topics->evt_cancel, "CDMA1", 0) == BYTEMAP_OK);
    }

    const char* ticker = NULL;
    md_side     side = SIDE_UNKNOWN;
    if (launch) {
        CHECK(c_md_topic_dma_parse(topics, launch, &ticker, &side) == TOPIC_SET_OK);
        CHECK(ticker && strcmp(ticker, "CDMA1") == 0);
        CHECK((int) side == OFFSET_ORDER);
    }
    if (cancel) {
        ticker = NULL;
        side   = SIDE_UNKNOWN;
        CHECK(c_md_topic_dma_parse(topics, cancel, &ticker, &side) == TOPIC_SET_OK);
        CHECK(ticker && strcmp(ticker, "CDMA1") == 0);
        CHECK((int) side == OFFSET_CANCEL);
    }

    // Registering an exact launch topic fills the cache without a DMA call.
    evt_topic* registered = c_topic_set_get(topics, "launch_order.CDMA2", 0);
    CHECK(registered != NULL);
    CHECK(c_bytemap_contains(topics->evt_launch, "CDMA2", 0) == BYTEMAP_OK);

    // A realtime topic is not a DMA topic.
    md_variant md;
    memset(&md, 0, sizeof(md));
    md.meta_info.ticker = "CDMA1";
    md.meta_info.dtype = DTYPE_TICK;
    evt_topic* realtime = c_md_topic_push(topics, &md);
    CHECK(realtime != NULL);
    if (realtime) {
        CHECK(c_md_topic_dma_parse(topics, realtime, NULL, NULL) == TOPIC_SET_ERR_INVALID_TOPIC);
    }
}

static void test_transport(void) {
    const size_t region_size = (size_t) 1 << 26;  // 64 MiB
    char         prefix[64];

    snprintf(prefix, sizeof(prefix), "c_dma_protocol_test_%ld", (long) getpid());

    shm_allocator_ctx* shm_ctx = c_shm_allocator_new(region_size, prefix);
    CHECK(shm_ctx != NULL);
    if (!shm_ctx) return;

    allocator_protocol* shm_allocator = c_ap_allocator_protocol_new(4096, shm_ctx, NULL, true);
    CHECK(shm_allocator != NULL);
    if (!shm_allocator) {
        c_shm_allocator_free(shm_ctx);
        return;
    }

    istr_map*      pool = c_istr_map_new(0, shm_allocator);
    evt_topic_set* topics = c_topic_set_new(NULL);
    evt_engine*    engine = c_evt_engine_new(NULL);
    CHECK(pool && topics && engine);
    if (!pool || !topics || !engine) {
        if (pool) c_istr_map_free(pool);
        if (topics) c_topic_set_free(topics);
        if (engine) c_evt_engine_free(engine);
        c_ap_allocator_protocol_free(shm_allocator);
        c_shm_allocator_free(shm_ctx);
        return;
    }

    dma_protocol_ctx* protocol = c_dma_protocol_new(engine, topics, shm_allocator, pool, NULL);
    CHECK(protocol != NULL);
    if (!protocol) {
        c_istr_map_free(pool);
        c_topic_set_free(topics);
        c_evt_engine_free(engine);
        c_ap_allocator_protocol_free(shm_allocator);
        c_shm_allocator_free(shm_ctx);
        return;
    }

    CHECK(protocol->ENGINE == engine);
    CHECK(protocol->TOPICS == topics);
    CHECK(protocol->SHM_ALLOCATOR == shm_allocator);
    CHECK(protocol->SHM_POOL == pool);
    CHECK(topics->evt_launch != NULL);
    CHECK(topics->evt_cancel != NULL);
    test_dma_topics(topics);
    CHECK(protocol->block == (bool) DMA_PROTOCOL_DEFAULT_BLOCK);
    CHECK(protocol->max_spin == (size_t) DMA_PROTOCOL_DEFAULT_MAX_SPIN);
    CHECK(protocol->timeout == (double) DMA_PROTOCOL_DEFAULT_TIMEOUT);

    CHECK(c_dma_protocol_config(protocol, false, 128, 0.5) == DMA_PROTOCOL_OK);
    CHECK(protocol->block == false);
    CHECK(protocol->max_spin == 128);
    CHECK(protocol->timeout == 0.5);
    CHECK(c_dma_protocol_config(protocol, true, 0, 1.0) == DMA_PROTOCOL_OK);

    // Direct registration: the callback is the engine handler.
    CHECK(c_dma_protocol_register_callback(protocol, topics->realtime, tick_listener, NULL) == DMA_PROTOCOL_OK);
    CHECK(c_dma_protocol_register_callback(protocol, topics->realtime, order_listener, NULL) == DMA_PROTOCOL_OK);
    CHECK(c_dma_protocol_register_callback(protocol, topics->on_order, instruction_listener, NULL) == DMA_PROTOCOL_OK);
    CHECK(c_dma_protocol_register_callback(protocol, topics->on_report, report_listener, NULL) == DMA_PROTOCOL_OK);
    CHECK(c_dma_protocol_register_callback(protocol, topics->launch_order, launch_listener, NULL) == DMA_PROTOCOL_OK);
    CHECK(c_dma_protocol_register_callback(protocol, topics->cancel_order, cancel_listener, NULL) == DMA_PROTOCOL_OK);

    // Invalid arguments.
    CHECK(c_dma_protocol_register_callback(NULL, topics->realtime, tick_listener, NULL) == DMA_PROTOCOL_ERR_INVALID_INPUT);
    CHECK(c_dma_protocol_register_callback(protocol, NULL, tick_listener, NULL) == DMA_PROTOCOL_ERR_INVALID_INPUT);
    CHECK(c_dma_protocol_register_callback(protocol, topics->realtime, NULL, NULL) == DMA_PROTOCOL_ERR_INVALID_INPUT);
    CHECK(c_dma_protocol_send(NULL, NULL) == DMA_PROTOCOL_ERR_INVALID_INPUT);

    // Market data without a ticker has no publish topic.
    md_variant* no_ticker_md = c_md_new(DTYPE_TICK_LITE, NULL);
    CHECK(no_ticker_md != NULL);
    if (no_ticker_md) {
        CHECK(c_dma_protocol_send(protocol, no_ticker_md) == DMA_PROTOCOL_ERR_NO_TOPIC);
        c_ap_free(no_ticker_md);
    }

    // The static payload is single-use: a second send before the loop drains
    // the first is refused.
    md_variant* busy_md = c_md_new(DTYPE_TICK_LITE, NULL);
    CHECK(busy_md != NULL);
    if (busy_md) {
        busy_md->meta_info.ticker = "CBUSY";
        busy_md->tick_data_lite.last_price = 1.5;

        CHECK(c_dma_protocol_send(protocol, busy_md) == DMA_PROTOCOL_OK);
        CHECK(c_dma_protocol_send(protocol, busy_md) == DMA_PROTOCOL_ERR_BUSY);
        c_ap_free(busy_md);
    }

    // Start the pure C dispatch loop; it drains CBUSY and its cleanup.
    atomic_store_explicit(&engine->active, true, memory_order_release);
    loop_ctx  lc = {.engine = engine};
    pthread_t thread;
    CHECK(pthread_create(&thread, NULL, engine_loop_main, &lc) == 0);

    CHECK(wait_for(&g_ticks, 1, 3.0));
    CHECK(atomic_load_explicit(&g_errors, memory_order_acquire) == 0);
    CHECK(strcmp(g_last_ticker, "CBUSY") == 0);
    CHECK(g_last_price == 1.5);

    // A heap source is copied into shm; the source is freed immediately after
    // the send, so the listener's data must come from the copy.
    md_variant* heap_md = c_md_new(DTYPE_TICK_LITE, NULL);
    CHECK(heap_md != NULL);
    if (heap_md) {
        heap_md->meta_info.ticker = "CTEST1";
        heap_md->tick_data_lite.last_price = 42.5;

        CHECK(c_dma_protocol_send(protocol, heap_md) == DMA_PROTOCOL_OK);
        c_ap_free(heap_md);

        CHECK(wait_for(&g_ticks, 2, 3.0));
        CHECK(atomic_load_explicit(&g_errors, memory_order_acquire) == 0);
        CHECK(strcmp(g_last_ticker, "CTEST1") == 0);
        CHECK(g_last_price == 42.5);
    }

    // An in-shm source is copied too: the source stays the caller's and is
    // freed right after the send, so the listener must read the copy.
    md_variant* shm_md = c_md_new(DTYPE_TICK_LITE, shm_allocator);
    CHECK(shm_md != NULL);
    if (shm_md) {
        shm_md->meta_info.ticker = "CTEST2";
        shm_md->tick_data_lite.last_price = 7.5;

        CHECK(c_dma_protocol_send(protocol, shm_md) == DMA_PROTOCOL_OK);
        c_ap_free(shm_md);

        CHECK(wait_for(&g_ticks, 3, 3.0));
        CHECK(atomic_load_explicit(&g_errors, memory_order_acquire) == 0);
        CHECK(strcmp(g_last_ticker, "CTEST2") == 0);
        CHECK(g_last_price == 7.5);
    }

    // Order data routes to realtime through c_md_topic_push.
    md_variant* order = c_md_new(DTYPE_ORDER, NULL);
    CHECK(order != NULL);
    if (order) {
        order->meta_info.ticker = "CTEST3";
        order->order_data.price = 5.25;

        CHECK(c_dma_protocol_send(protocol, order) == DMA_PROTOCOL_OK);
        c_ap_free(order);

        CHECK(wait_for(&g_orders, 1, 3.0));
        CHECK(atomic_load_explicit(&g_errors, memory_order_acquire) == 0);
        CHECK(strcmp(g_last_ticker, "CTEST3") == 0);
        CHECK(g_last_price == 5.25);
    }

    // A trade instruction routes to on_order.
    md_variant* instruction = c_md_new(DTYPE_INSTRUCTION, NULL);
    CHECK(instruction != NULL);
    if (instruction) {
        instruction->meta_info.ticker = "CTEST4";
        instruction->trade_instruction.volume = 4.5;

        CHECK(c_dma_protocol_send(protocol, instruction) == DMA_PROTOCOL_OK);
        c_ap_free(instruction);

        CHECK(wait_for(&g_instructions, 1, 3.0));
        CHECK(atomic_load_explicit(&g_errors, memory_order_acquire) == 0);
        CHECK(strcmp(g_last_ticker, "CTEST4") == 0);
        CHECK(g_last_price == 4.5);
    }

    // An exchange report routes to on_report.
    md_variant* report = c_md_new(DTYPE_REPORT, NULL);
    CHECK(report != NULL);
    if (report) {
        report->meta_info.ticker = "CTEST5";
        report->trade_report.price = 10.25;

        CHECK(c_dma_protocol_send(protocol, report) == DMA_PROTOCOL_OK);
        c_ap_free(report);

        CHECK(wait_for(&g_reports, 1, 3.0));
        CHECK(atomic_load_explicit(&g_errors, memory_order_acquire) == 0);
        CHECK(strcmp(g_last_ticker, "CTEST5") == 0);
        CHECK(g_last_price == 10.25);
    }

    // The dedicated launch/cancel interface sends a heap instruction through
    // c_md_send_to_shm; the source is copied and stays owned by the caller.
    md_variant* order_md = c_md_new(DTYPE_INSTRUCTION, NULL);
    CHECK(order_md != NULL);
    if (order_md) {
        order_md->meta_info.ticker = "CTEST6";
        order_md->trade_instruction.volume = 3.5;

        c_dma_protocol_launch_order((md_trade_instruction*) order_md, 1, protocol);
        CHECK(wait_for(&g_launches, 1, 3.0));
        CHECK(atomic_load_explicit(&g_errors, memory_order_acquire) == 0);
        CHECK(strcmp(g_last_ticker, "CTEST6") == 0);
        CHECK(g_last_price == 3.5);

        c_dma_protocol_cancel_order((const md_trade_instruction*) order_md, 2, protocol);
        CHECK(wait_for(&g_cancels, 1, 3.0));
        CHECK(atomic_load_explicit(&g_errors, memory_order_acquire) == 0);
        CHECK(strcmp(g_last_ticker, "CTEST6") == 0);
        CHECK(g_last_price == 3.5);

        c_ap_free(order_md);
    }

    // Stop the loop.
    atomic_store_explicit(&engine->active, false, memory_order_release);
    pthread_join(thread, NULL);

    // Unregister; a second attempt is not found.
    CHECK(c_dma_protocol_unregister_callback(protocol, topics->realtime, tick_listener, NULL) == DMA_PROTOCOL_OK);
    CHECK(c_dma_protocol_unregister_callback(protocol, topics->realtime, tick_listener, NULL) == DMA_PROTOCOL_ERR_NOT_FOUND);
    CHECK(c_dma_protocol_unregister_callback(protocol, topics->realtime, order_listener, NULL) == DMA_PROTOCOL_OK);
    CHECK(c_dma_protocol_unregister_callback(protocol, topics->on_order, instruction_listener, NULL) == DMA_PROTOCOL_OK);
    CHECK(c_dma_protocol_unregister_callback(protocol, topics->on_report, report_listener, NULL) == DMA_PROTOCOL_OK);
    CHECK(c_dma_protocol_unregister_callback(protocol, topics->launch_order, launch_listener, NULL) == DMA_PROTOCOL_OK);
    CHECK(c_dma_protocol_unregister_callback(protocol, topics->cancel_order, cancel_listener, NULL) == DMA_PROTOCOL_OK);

    c_dma_protocol_free(protocol);
    c_istr_map_free(pool);
    c_topic_set_free(topics);
    c_evt_engine_free(engine);
    c_ap_allocator_protocol_free(shm_allocator);
    c_shm_allocator_free(shm_ctx);
}

int main(void) {
    test_handler_compatibility();
    test_ret_codes();
    test_transport();

    printf("[c_dma_protocol] checks=%zu failures=%zu\n", g_checks, g_failures);
    return g_failures ? 1 : 0;
}

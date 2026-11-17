/*
 * Pure-C test for algo_engine/backtest/c_simmatch.h.
 *
 * The test links nothing but libc/libm: every C function under test is static
 * inline in the header. It covers:
 *
 *   1. Lifecycle: _new/_init/_dealloc/_free defaults, the zeroed-buffer
 *      contract and the bound report workspace.
 *   2. The price helpers: side selection, non-finite skipping, ret codes.
 *   3. The registry: bytemap lookup, the newest-first list, growth, the
 *      working/history split, eod, clear, and the owned order copies.
 *   4. Matching: bar (limit / VWAP / infinite limit / zero volume), tick
 *      accumulation, tick-lite, order data and transactions.
 *   5. Lag, hit probability and the seeded trade id.
 *   6. Listener registration, delivery and survival across clear.
 *   7. Ownership: launch copies the instruction and never mutates the caller's.
 *
 * Build & run with run_c_simmatch_c_test.sh.
 */

// Feature-test macro for the POSIX declarations pulled in by the allocator
// and market-data headers. Must precede every system include.
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <algo_engine/backtest/c_simmatch.h>

// ========== Exchange Profile Globals ==========

// The profile globals live in the c_exchange_profile extension on POSIX; a
// pure-C harness supplies the NULL defaults the report timestamp uses.
const exchange_profile*         EX_PROFILE                      = NULL;
const session_date_range_t*     EX_TRADE_CALENDAR_CACHE         = NULL;
ex_profile_activation_listener* EX_PROFILE_ACTIVATION_LISTENERS = NULL;
const exchange_profile          EX_PROFILE_DEFAULT              = {0};

// ========== Assertion Helper ==========

static size_t g_checks   = 0;
static size_t g_failures = 0;

#define CHECK(cond)                                                          \
    do {                                                                     \
        g_checks++;                                                          \
        if (!(cond)) {                                                       \
            g_failures++;                                                    \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);  \
        }                                                                    \
    } while (0)

// ========== Listener ==========

typedef struct test_listener {
    size_t         n_placed;
    size_t         n_match;
    size_t         n_cancel;
    double         last_price;
    double         last_volume;
    double         last_fee;
    double         last_filled_volume;
    md_order_state last_state;
    long_md_id     last_trade_id;
} test_listener;

static void listener_cb(smm_match_ctx* ctx, smm_match_event event,
                        const smm_order_entry* entry, const md_trade_report* report,
                        void* user_data) {
    test_listener* listener = (test_listener*) user_data;
    (void) ctx;

    if (entry) {
        listener->last_state         = entry->order.order_state;
        listener->last_filled_volume = entry->order.filled_volume;
        listener->last_fee           = entry->order.fee;
    }

    if (event == SMM_EVENT_PLACED) {
        listener->n_placed++;
    }
    else if (event == SMM_EVENT_MATCH) {
        listener->n_match++;
        if (report) {
            listener->last_price    = report->price;
            listener->last_volume   = report->volume;
            listener->last_trade_id = report->trade_id;
        }
    }
    else {
        listener->n_cancel++;
    }
}

// ========== Payload Builders ==========

static md_trade_instruction make_order(md_side side, double limit_price, double volume, uint8_t id) {
    md_trade_instruction order;
    memset(&order, 0, sizeof(order));
    order.meta_info.dtype      = DTYPE_INSTRUCTION;
    order.meta_info.ticker     = "TEST";
    order.meta_info.dt.unix_ts = 100.0;
    order.limit_price          = limit_price;
    order.volume               = volume;
    order.side                 = side;
    order.order_type           = ORDER_LIMIT;
    order.multiplier           = 1.0;
    order.order_id.id_type     = MID_UUID;
    order.order_id.data[0]     = (char) id;
    return order;
}

static md_variant make_bar(double ts, double high, double low, double close, double volume, double notional) {
    md_variant bar;
    memset(&bar, 0, sizeof(bar));
    bar.meta_info.dtype      = DTYPE_BAR;
    bar.meta_info.ticker     = "TEST";
    bar.meta_info.dt.unix_ts = ts;
    bar.bar_data.bar_span    = 60.0;
    bar.bar_data.high_price  = high;
    bar.bar_data.low_price   = low;
    bar.bar_data.open_price  = close;
    bar.bar_data.close_price = close;
    bar.bar_data.volume      = volume;
    bar.bar_data.notional    = notional;
    return bar;
}

static md_variant make_tick_lite(double ts, double last, double bid_price, double bid_volume, double ask_price, double ask_volume) {
    md_variant lite;
    memset(&lite, 0, sizeof(lite));
    lite.meta_info.dtype           = DTYPE_TICK_LITE;
    lite.meta_info.ticker          = "TEST";
    lite.meta_info.dt.unix_ts      = ts;
    lite.tick_data_lite.last_price = last;
    lite.tick_data_lite.bid_price  = bid_price;
    lite.tick_data_lite.bid_volume = bid_volume;
    lite.tick_data_lite.ask_price  = ask_price;
    lite.tick_data_lite.ask_volume = ask_volume;
    return lite;
}

static md_variant make_txn(double ts, double price, double volume, md_side side) {
    md_variant txn;
    memset(&txn, 0, sizeof(txn));
    txn.meta_info.dtype             = DTYPE_TRANSACTION;
    txn.meta_info.ticker            = "TEST";
    txn.meta_info.dt.unix_ts        = ts;
    txn.transaction_data.price      = price;
    txn.transaction_data.volume     = volume;
    txn.transaction_data.side       = side;
    txn.transaction_data.multiplier = 1.0;
    return txn;
}

static md_variant make_order_data(double ts, double price, double volume, md_side side) {
    md_variant data;
    memset(&data, 0, sizeof(data));
    data.meta_info.dtype      = DTYPE_ORDER;
    data.meta_info.ticker     = "TEST";
    data.meta_info.dt.unix_ts = ts;
    data.order_data.price     = price;
    data.order_data.volume    = volume;
    data.order_data.side      = side;
    return data;
}

static md_variant make_tick(double ts, double last, size_t n_ask, const double* ask_prices, const double* ask_volumes) {
    md_variant tick;
    memset(&tick, 0, sizeof(tick));
    tick.meta_info.dtype                = DTYPE_TICK;
    tick.meta_info.ticker               = "TEST";
    tick.meta_info.dt.unix_ts           = ts;
    tick.tick_data_full.lite.last_price = last;
    tick.tick_data_full.bid             = c_md_orderbook_new(n_ask, NULL);
    tick.tick_data_full.ask             = c_md_orderbook_new(n_ask, NULL);

    tick.tick_data_full.bid->direction = DIRECTION_LONG;
    tick.tick_data_full.ask->direction = DIRECTION_SHORT;

    tick.tick_data_full.ask->size = n_ask;
    for (size_t i = 0u; i < n_ask; i++) {
        tick.tick_data_full.ask->entries[i].price  = ask_prices[i];
        tick.tick_data_full.ask->entries[i].volume = ask_volumes[i];
    }
    return tick;
}

static void free_tick(md_variant* tick) {
    c_md_orderbook_free(tick->tick_data_full.ask);
    c_md_orderbook_free(tick->tick_data_full.bid);
}

// ========== Registry Accessors ==========

// The registry's owned order record, or NULL while the order is absent.
static md_trade_instruction* working_order(smm_match_ctx* ctx, const long_md_id* order_id) {
    smm_order_entry* entry = c_smm_working_find(ctx, order_id);
    return entry ? &entry->order : NULL;
}

static md_trade_instruction* history_order(smm_match_ctx* ctx, const long_md_id* order_id) {
    smm_order_entry* entry = c_smm_history_find(ctx, order_id);
    return entry ? &entry->order : NULL;
}

// The stored report count of an order, 0 while nothing filled.
static size_t report_count(smm_match_ctx* ctx, const long_md_id* order_id) {
    smm_order_entry* entry = c_smm_history_find(ctx, order_id);
    if (!entry) entry = c_smm_working_find(ctx, order_id);
    if (!entry || !entry->reports) return 0u;
    return c_bytemap_len(entry->reports);
}

// ========== Lifecycle ==========

static void test_lifecycle(void) {
    smm_match_ctx* ctx = c_smm_match_new(0.001, false, 0.0, 0u, 1.0, 0.0001, false, NULL);
    CHECK(ctx != NULL);
    CHECK(ctx->fee_rate == 0.001);
    CHECK(ctx->hit_prob == 1.0);
    CHECK(ctx->slippery_rate == 0.0001);
    CHECK(isnan(ctx->last_price));
    CHECK(ctx->working != NULL);
    CHECK(ctx->history != NULL);
    CHECK(ctx->newest == NULL);
    CHECK(ctx->oldest == NULL);
    CHECK(c_bytemap_len(ctx->working) == 0u);
    CHECK(c_bytemap_len(ctx->history) == 0u);
    c_smm_match_free(ctx);

    char* block = (char*) c_ap_alloc(sizeof(smm_match_ctx), NULL);
    CHECK(block != NULL);
    smm_match_ctx* manual = (smm_match_ctx*) block;
    CHECK(c_smm_match_init(manual) == SMM_OK);
    CHECK(manual->working != NULL);
    CHECK(manual->history != NULL);
    CHECK(isnan(manual->last_price));
    c_smm_match_dealloc(manual);

    bool zeroed = true;
    for (size_t i = 0u; i < sizeof(smm_match_ctx); i++) {
        if (block[i] != 0) {
            zeroed = false;
            break;
        }
    }
    CHECK(zeroed);
    c_ap_free(block);

    CHECK(c_smm_match_init(NULL) == SMM_ERR_INVALID_ARG);
    CHECK(c_smm_match_process(NULL, NULL) == SMM_ERR_INVALID_ARG);
    CHECK(c_smm_match_launch(NULL, NULL) == SMM_ERR_INVALID_ARG);
    CHECK(c_smm_match_cancel(NULL, NULL) == SMM_ERR_INVALID_ARG);
    CHECK(c_smm_match_set_seed(NULL, 1u) == SMM_ERR_INVALID_ARG);
    CHECK(c_smm_match_set_config(NULL, 0.0, false, 0.0, 0u, 1.0, 0.0, false) == SMM_ERR_INVALID_ARG);
}

// ========== Price Helpers ==========

static void test_price_helpers(void) {
    double prices[] = {3.0, NAN, 1.0, INFINITY, 2.0};
    double out      = 0.0;

    CHECK(c_smm_match_best_price(prices, 5u, 1, &out) == SMM_OK && out == 1.0);
    CHECK(c_smm_match_worst_price(prices, 5u, 1, &out) == SMM_OK && out == 3.0);
    CHECK(c_smm_match_best_price(prices, 5u, -1, &out) == SMM_OK && out == 3.0);
    CHECK(c_smm_match_worst_price(prices, 5u, -1, &out) == SMM_OK && out == 1.0);

    CHECK(c_smm_match_best_price(prices, 5u, 0, &out) == SMM_ERR_INVALID_SIDE);
    CHECK(c_smm_match_worst_price(prices, 5u, 0, &out) == SMM_ERR_INVALID_SIDE);

    double invalid[] = {NAN, INFINITY};
    CHECK(c_smm_match_best_price(invalid, 2u, 1, &out) == SMM_ERR_NO_VALID_PRICE);
    CHECK(c_smm_match_worst_price(invalid, 2u, -1, &out) == SMM_ERR_NO_VALID_PRICE);

    CHECK(c_smm_match_best_price(NULL, 5u, 1, &out) == SMM_ERR_INVALID_ARG);
    CHECK(c_smm_match_best_price(prices, 5u, 1, NULL) == SMM_ERR_INVALID_ARG);
}

// ========== Registry ==========

static void test_registry(void) {
    smm_match_ctx* ctx = c_smm_match_new(0.0, false, 0.0, 0u, 1.0, 0.0, false, NULL);
    CHECK(ctx != NULL);

    // An entry is addressable as its instruction.
    md_trade_instruction orders[200];
    for (uint8_t i = 0u; i < 200u; i++) orders[i] = make_order(SIDE_BID, 20.0, 1.0, i);

    for (size_t i = 0u; i < 5u; i++) CHECK(c_smm_match_launch(ctx, &orders[i]) == SMM_OK);
    CHECK(c_bytemap_len(ctx->working) == 5u);

    // Newest-first list: the last launch is the scan head.
    smm_order_entry* node = ctx->newest;
    for (uint8_t expected = 4u;; expected--) {
        CHECK(node != NULL);
        if (!node) break;
        CHECK(node->order.order_id.data[0] == (char) expected);
        if (expected == 0u) break;
        node = node->older;
    }
    CHECK(ctx->oldest != NULL && ctx->oldest->order.order_id.data[0] == 0);

    md_trade_instruction* copy = working_order(ctx, &orders[0].order_id);
    CHECK(copy != NULL);
    CHECK(copy != &orders[0]);
    CHECK(copy->order_state == STATE_PLACED);
    CHECK(orders[0].order_state == STATE_UNKNOWN);
    CHECK(ctx->oldest->order.order_id.data[0] == 0);

    // The transaction count at launch feeds the transaction lag. The print is
    // dated before the orders, so it advances the counter without matching.
    md_variant txn = make_txn(50.0, 19.0, 1.0, SIDE_LONG_OPEN);
    CHECK(c_smm_match_process(ctx, &txn) == SMM_OK);
    CHECK(c_smm_match_launch(ctx, &orders[5]) == SMM_OK);
    CHECK(c_smm_working_find(ctx, &orders[5].order_id)->transaction_count_at_launch == 1u);
    CHECK(c_smm_working_find(ctx, &orders[0].order_id)->transaction_count_at_launch == 0u);

    CHECK(c_smm_match_launch(ctx, &orders[0]) == SMM_ERR_DUPLICATE);

    // Growth has no fixed capacity: 200 live orders all stay reachable.
    for (size_t i = 6u; i < 200u; i++) CHECK(c_smm_match_launch(ctx, &orders[i]) == SMM_OK);
    CHECK(c_bytemap_len(ctx->working) == 200u);
    CHECK(working_order(ctx, &orders[199].order_id) != NULL);
    CHECK(working_order(ctx, &orders[0].order_id) != NULL);

    for (size_t i = 0u; i < 200u; i += 2) CHECK(c_smm_match_cancel(ctx, &orders[i].order_id) == SMM_OK);
    CHECK(c_bytemap_len(ctx->working) == 100u);
    CHECK(c_bytemap_len(ctx->history) == 100u);
    CHECK(working_order(ctx, &orders[0].order_id) == NULL);
    CHECK(history_order(ctx, &orders[0].order_id) != NULL);
    CHECK(history_order(ctx, &orders[0].order_id)->order_state == STATE_CANCELED);
    CHECK(orders[0].order_state == STATE_UNKNOWN);
    CHECK(c_smm_match_launch(ctx, &orders[0]) == SMM_ERR_DUPLICATE);

    c_smm_match_eod(ctx);
    CHECK(c_bytemap_len(ctx->working) == 0u);
    CHECK(c_bytemap_len(ctx->history) == 200u);
    CHECK(ctx->newest == NULL);
    CHECK(ctx->oldest == NULL);
    CHECK(c_smm_match_cancel(ctx, &orders[0].order_id) == SMM_ERR_NOT_FOUND);

    c_smm_match_clear(ctx);
    CHECK(c_bytemap_len(ctx->working) == 0u);
    CHECK(c_bytemap_len(ctx->history) == 0u);
    CHECK(ctx->working != NULL);
    CHECK(ctx->history != NULL);

    // The cleared registries keep working after growth.
    CHECK(c_smm_match_launch(ctx, &orders[0]) == SMM_OK);
    md_variant bar = make_bar(200.0, 19.5, 19.0, 19.3, 1000.0, 19300.0);
    CHECK(c_smm_match_process(ctx, &bar) == SMM_OK);
    CHECK(history_order(ctx, &orders[0].order_id) != NULL);

    c_smm_match_free(ctx);
}

// ========== Bar Matching ==========

static void test_bar_matching(void) {
    smm_match_ctx* ctx = c_smm_match_new(0.0, false, 0.0, 0u, 1.0, 0.0, false, NULL);
    test_listener  listener;
    memset(&listener, 0, sizeof(listener));

    uintptr_t listener_id = 0u;
    CHECK(c_smm_match_register_listener(ctx, listener_cb, &listener, &listener_id) == SMM_OK);

    md_trade_instruction buy = make_order(SIDE_BID, 50.0, 100.0, 1);
    CHECK(c_smm_match_launch(ctx, &buy) == SMM_OK);
    CHECK(listener.n_placed == 1u);
    CHECK(listener.last_state == STATE_PLACED);
    CHECK(working_order(ctx, &buy.order_id)->order_state == STATE_PLACED);
    CHECK(buy.order_state == STATE_UNKNOWN);

    md_variant bar = make_bar(200.0, 49.5, 49.0, 49.3, 1000.0, 49200.0);
    CHECK(c_smm_match_process(ctx, &bar) == SMM_OK);
    CHECK(listener.last_state == STATE_FILLED);
    CHECK(listener.last_filled_volume == 100.0);
    CHECK(listener.n_match == 1u);
    CHECK(listener.last_price == 49.5);
    CHECK(listener.last_volume == 100.0);
    CHECK(c_smm_working_find(ctx, &buy.order_id) == NULL);
    CHECK(history_order(ctx, &buy.order_id) != NULL);
    CHECK(report_count(ctx, &buy.order_id) == 1u);

    // The caller's instruction is untouched by the fill.
    CHECK(buy.order_state == STATE_UNKNOWN);
    CHECK(buy.filled_volume == 0.0);
    CHECK(buy.ts_placed == 0.0);

    md_trade_instruction sell = make_order(SIDE_ASK, 50.0, 100.0, 2);
    CHECK(c_smm_match_launch(ctx, &sell) == SMM_OK);
    md_variant bar_sell = make_bar(201.0, 50.8, 50.5, 50.7, 1000.0, 50700.0);
    CHECK(c_smm_match_process(ctx, &bar_sell) == SMM_OK);
    CHECK(listener.last_state == STATE_FILLED);
    CHECK(listener.last_price == 50.5);
    CHECK(sell.order_state == STATE_UNKNOWN);

    md_trade_instruction no_fill = make_order(SIDE_BID, 50.0, 100.0, 3);
    CHECK(c_smm_match_launch(ctx, &no_fill) == SMM_OK);
    md_variant bar_no_fill = make_bar(202.0, 50.5, 50.2, 50.4, 1000.0, 50400.0);
    CHECK(c_smm_match_process(ctx, &bar_no_fill) == SMM_OK);
    CHECK(working_order(ctx, &no_fill.order_id)->order_state == STATE_PLACED);
    CHECK(listener.n_match == 2u);
    CHECK(c_smm_match_cancel(ctx, &no_fill.order_id) == SMM_OK);

    md_trade_instruction market = make_order(SIDE_BID, NAN, 100.0, 4);
    CHECK(c_smm_match_launch(ctx, &market) == SMM_OK);
    md_variant bar_zero = make_bar(203.0, 49.5, 49.0, 49.3, 0.0, 0.0);
    CHECK(c_smm_match_process(ctx, &bar_zero) == SMM_OK);
    CHECK(working_order(ctx, &market.order_id)->order_state == STATE_PLACED);
    CHECK(listener.n_match == 2u);

    md_trade_instruction market_like = make_order(SIDE_BID, INFINITY, 100.0, 5);
    CHECK(c_smm_match_launch(ctx, &market_like) == SMM_OK);
    md_variant bar_vwap = make_bar(204.0, 50.5, 49.5, 50.2, 1000.0, 50000.0);
    CHECK(c_smm_match_process(ctx, &bar_vwap) == SMM_OK);
    CHECK(listener.last_state == STATE_FILLED);
    CHECK(listener.last_price == 50.0);
    CHECK(history_order(ctx, &market_like.order_id) != NULL);

    CHECK(c_smm_match_deregister_listener(ctx, listener_id) == SMM_OK);
    CHECK(c_smm_match_deregister_listener(ctx, listener_id) == SMM_ERR_NOT_FOUND);

    c_smm_match_free(ctx);
}

// ========== Tick Matching ==========

static void test_tick_matching(void) {
    smm_match_ctx* ctx = c_smm_match_new(0.0, false, 0.0, 0u, 1.0, 0.0, false, NULL);
    test_listener  listener;
    memset(&listener, 0, sizeof(listener));
    CHECK(c_smm_match_register_listener(ctx, listener_cb, &listener, NULL) == SMM_OK);

    md_trade_instruction buy = make_order(SIDE_BID, 50.0, 150.0, 1);
    CHECK(c_smm_match_launch(ctx, &buy) == SMM_OK);

    double ask_prices[2]  = {49.9, 50.0};
    double ask_volumes[2] = {100.0, 100.0};
    md_variant tick = make_tick(200.0, 49.5, 2u, ask_prices, ask_volumes);
    CHECK(c_smm_match_process(ctx, &tick) == SMM_OK);
    free_tick(&tick);

    CHECK(listener.last_state == STATE_FILLED);
    CHECK(listener.last_filled_volume == 150.0);
    CHECK(fabs(listener.last_price - (100.0 * 49.9 + 50.0 * 50.0) / 150.0) < 1e-12);
    CHECK(buy.filled_volume == 0.0);

    md_trade_instruction partial = make_order(SIDE_BID, 50.0, 100.0, 2);
    CHECK(c_smm_match_launch(ctx, &partial) == SMM_OK);
    double one_price[1]  = {49.9};
    double one_volume[1] = {60.0};
    md_variant tick_partial = make_tick(201.0, 49.5, 1u, one_price, one_volume);
    CHECK(c_smm_match_process(ctx, &tick_partial) == SMM_OK);
    free_tick(&tick_partial);

    md_trade_instruction* partial_copy = working_order(ctx, &partial.order_id);
    CHECK(partial_copy != NULL);
    CHECK(partial_copy->order_state == STATE_PARTFILLED);
    CHECK(partial_copy->filled_volume == 60.0);
    CHECK(report_count(ctx, &partial.order_id) == 1u);

    md_variant tick_rest = make_tick(202.0, 49.5, 1u, one_price, one_volume);
    CHECK(c_smm_match_process(ctx, &tick_rest) == SMM_OK);
    free_tick(&tick_rest);
    CHECK(listener.last_state == STATE_FILLED);
    CHECK(listener.last_filled_volume == 100.0);
    CHECK(working_order(ctx, &partial.order_id) == NULL);
    CHECK(history_order(ctx, &partial.order_id) != NULL);
    CHECK(report_count(ctx, &partial.order_id) == 2u);
    CHECK(partial.order_state == STATE_UNKNOWN);

    md_trade_instruction no_fill = make_order(SIDE_BID, 50.0, 100.0, 3);
    CHECK(c_smm_match_launch(ctx, &no_fill) == SMM_OK);
    double high_price[1] = {50.5};
    md_variant tick_high = make_tick(203.0, 50.5, 1u, high_price, one_volume);
    CHECK(c_smm_match_process(ctx, &tick_high) == SMM_OK);
    free_tick(&tick_high);
    CHECK(working_order(ctx, &no_fill.order_id)->order_state == STATE_PLACED);

    c_smm_match_free(ctx);
}

// ========== Tick-Lite, Order and Transaction Matching ==========

static void test_bookless_matching(void) {
    smm_match_ctx* ctx = c_smm_match_new(0.0, false, 0.0, 0u, 1.0, 0.0, false, NULL);
    test_listener  listener;
    memset(&listener, 0, sizeof(listener));
    CHECK(c_smm_match_register_listener(ctx, listener_cb, &listener, NULL) == SMM_OK);

    md_trade_instruction lite_buy = make_order(SIDE_BID, 50.0, 100.0, 1);
    CHECK(c_smm_match_launch(ctx, &lite_buy) == SMM_OK);
    md_variant lite = make_tick_lite(200.0, 49.5, 49.0, 100.0, 49.9, 100.0);
    CHECK(c_smm_match_process(ctx, &lite) == SMM_OK);
    CHECK(listener.last_state == STATE_FILLED);
    CHECK(listener.last_price == 49.9);
    CHECK(lite_buy.order_state == STATE_UNKNOWN);

    md_trade_instruction txn_buy = make_order(SIDE_BID, NAN, 100.0, 2);
    CHECK(c_smm_match_launch(ctx, &txn_buy) == SMM_OK);
    md_variant txn = make_txn(201.0, 49.5, 30.0, SIDE_LONG_OPEN);
    CHECK(c_smm_match_process(ctx, &txn) == SMM_OK);
    md_trade_instruction* txn_copy = working_order(ctx, &txn_buy.order_id);
    CHECK(txn_copy != NULL);
    CHECK(txn_copy->order_state == STATE_PARTFILLED);
    CHECK(txn_copy->filled_volume == 30.0);
    CHECK(ctx->last_transaction_count == 1u);

    md_trade_instruction order_buy = make_order(SIDE_BID, 50.0, 100.0, 3);
    CHECK(c_smm_match_launch(ctx, &order_buy) == SMM_OK);
    md_variant order_data = make_order_data(202.0, 49.9, 50.0, SIDE_ASK);
    CHECK(c_smm_match_process(ctx, &order_data) == SMM_OK);
    CHECK(working_order(ctx, &order_buy.order_id)->order_state == STATE_PLACED);

    c_smm_match_set_config(ctx, 0.0, false, 0.0, 0u, 1.0, 0.0, true);
    CHECK(c_smm_match_process(ctx, &order_data) == SMM_OK);
    md_trade_instruction* order_copy = working_order(ctx, &order_buy.order_id);
    CHECK(order_copy != NULL);
    CHECK(order_copy->order_state == STATE_PARTFILLED);
    CHECK(order_copy->filled_volume == 50.0);
    CHECK(order_buy.order_state == STATE_UNKNOWN);

    c_smm_match_free(ctx);
}

// ========== Lag ==========

static void test_lag(void) {
    smm_match_ctx* ctx = c_smm_match_new(0.0, false, 5.0, 0u, 1.0, 0.0, false, NULL);
    md_trade_instruction order = make_order(SIDE_BID, 50.0, 100.0, 1);
    CHECK(c_smm_match_launch(ctx, &order) == SMM_OK);

    md_variant early = make_bar(103.0, 49.5, 49.0, 49.3, 1000.0, 49200.0);
    CHECK(c_smm_match_process(ctx, &early) == SMM_OK);
    CHECK(working_order(ctx, &order.order_id)->order_state == STATE_PLACED);

    md_variant late = make_bar(105.0, 49.5, 49.0, 49.3, 1000.0, 49200.0);
    CHECK(c_smm_match_process(ctx, &late) == SMM_OK);
    CHECK(working_order(ctx, &order.order_id) == NULL);
    c_smm_match_free(ctx);

    ctx = c_smm_match_new(0.0, false, 0.0, 2u, 1.0, 0.0, false, NULL);
    md_trade_instruction market = make_order(SIDE_BID, NAN, 100.0, 2);
    CHECK(c_smm_match_launch(ctx, &market) == SMM_OK);

    md_variant txn_first = make_txn(200.0, 49.5, 10.0, SIDE_LONG_OPEN);
    CHECK(c_smm_match_process(ctx, &txn_first) == SMM_OK);
    CHECK(working_order(ctx, &market.order_id)->order_state == STATE_PLACED);

    md_variant txn_second = make_txn(201.0, 49.5, 10.0, SIDE_LONG_OPEN);
    CHECK(c_smm_match_process(ctx, &txn_second) == SMM_OK);
    md_trade_instruction* market_copy = working_order(ctx, &market.order_id);
    CHECK(market_copy != NULL);
    CHECK(market_copy->order_state == STATE_PARTFILLED);
    CHECK(market_copy->filled_volume == 10.0);
    c_smm_match_free(ctx);
}

// ========== Seed and Listener Survival ==========

static long_md_id seeded_trade_id(uint64_t seed) {
    smm_match_ctx* ctx = c_smm_match_new(0.0, false, 0.0, 0u, 1.0, 0.0, false, NULL);
    test_listener  listener;
    memset(&listener, 0, sizeof(listener));
    CHECK(c_smm_match_set_seed(ctx, seed) == SMM_OK);
    CHECK(c_smm_match_register_listener(ctx, listener_cb, &listener, NULL) == SMM_OK);

    md_trade_instruction order = make_order(SIDE_BID, 50.0, 100.0, 1);
    CHECK(c_smm_match_launch(ctx, &order) == SMM_OK);
    md_variant bar = make_bar(200.0, 49.5, 49.0, 49.3, 1000.0, 49200.0);
    CHECK(c_smm_match_process(ctx, &bar) == SMM_OK);
    CHECK(listener.n_match == 1u);

    long_md_id trade_id = listener.last_trade_id;
    c_smm_match_free(ctx);
    return trade_id;
}

static void test_seed_and_clear(void) {
    long_md_id first  = seeded_trade_id(42u);
    long_md_id second = seeded_trade_id(42u);
    long_md_id third  = seeded_trade_id(43u);

    CHECK(first.id_type == MID_UUID);
    CHECK(memcmp(&first, &second, sizeof(long_md_id)) == 0);
    CHECK(memcmp(&first, &third, sizeof(long_md_id)) != 0);

    smm_match_ctx* ctx = c_smm_match_new(0.0, false, 0.0, 0u, 1.0, 0.0, false, NULL);
    test_listener  listener;
    memset(&listener, 0, sizeof(listener));
    uintptr_t listener_id = 0u;
    CHECK(c_smm_match_register_listener(ctx, listener_cb, &listener, &listener_id) == SMM_OK);

    md_trade_instruction first_order = make_order(SIDE_BID, 50.0, 100.0, 1);
    CHECK(c_smm_match_launch(ctx, &first_order) == SMM_OK);
    md_variant first_bar = make_bar(200.0, 49.5, 49.0, 49.3, 1000.0, 49200.0);
    CHECK(c_smm_match_process(ctx, &first_bar) == SMM_OK);
    CHECK(listener.n_match == 1u);

    c_smm_match_clear(ctx);
    CHECK(ctx->timestamp == 0.0);
    CHECK(isnan(ctx->last_price));
    CHECK(ctx->last_transaction_count == 0u);
    CHECK(c_bytemap_len(ctx->working) == 0u);
    CHECK(c_bytemap_len(ctx->history) == 0u);

    md_trade_instruction second_order = make_order(SIDE_BID, 50.0, 100.0, 2);
    CHECK(c_smm_match_launch(ctx, &second_order) == SMM_OK);
    md_variant second_bar = make_bar(201.0, 49.5, 49.0, 49.3, 1000.0, 49200.0);
    CHECK(c_smm_match_process(ctx, &second_bar) == SMM_OK);
    CHECK(listener.n_match == 2u);
    CHECK(listener.n_placed == 2u);
    CHECK(listener.n_cancel == 0u);
    CHECK(history_order(ctx, &second_order.order_id) != NULL);

    CHECK(c_smm_match_deregister_listener(ctx, listener_id) == SMM_OK);
    c_smm_match_free(ctx);
}

// ========== Main ==========

int main(void) {
    test_lifecycle();
    test_price_helpers();
    test_registry();
    test_bar_matching();
    test_tick_matching();
    test_bookless_matching();
    test_lag();
    test_seed_and_clear();

    if (g_failures) {
        fprintf(stderr, "[FAIL] %zu/%zu checks failed\n", g_failures, g_checks);
        return 1;
    }

    printf("[PASS] %zu checks\n", g_checks);
    return 0;
}

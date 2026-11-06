/*
 * Pure-C test for algo_engine/engine/c_dma.h, c_inventory.h, c_position.h and
 * c_risk_profile.h.
 *
 * The test links nothing but libc/libm: every C function under test is static
 * inline in the headers. It covers:
 *
 *   1. The risk profile: rule set/get, missing-rule detection, clear.
 *   2. The risk check: incomplete-profile fail-closed, price validity, trade,
 *      exposure, percentile, notional, portfolio net and portfolio total.
 *   3. The direct market access: interface binding, enabled/risk gating,
 *      interface dispatch, position binding, the buffered request queue.
 *   4. The inventory: add, merge, available, use, recall, clear.
 *   5. The position management state: algo and working-algo registries.
 *
 * Build & run with run_c_dma_c_test.sh.
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

#include <algo_engine/engine/c_dma.h>
#include <algo_engine/engine/c_inventory.h>
#include <algo_engine/engine/c_position.h>

// ========== Assertion Helper ==========

static size_t g_checks   = 0;
static size_t g_failures = 0;

/* Root allocator for the heap blocks in this harness. */
static allocator_protocol* g_allocator = NULL;

#define CHECK(cond)                                                          \
    do {                                                                     \
        g_checks++;                                                          \
        if (!(cond)) {                                                       \
            g_failures++;                                                    \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);  \
        }                                                                    \
    } while (0)

// ========== Test Interface Context ==========

typedef struct test_interface {
    size_t    n_launch;
    size_t    n_cancel;
    size_t    n_reject;
    size_t    n_filled;
    uint32_t  last_seq_id;
    const md_trade_instruction* last_order;
    md_trade_report*            last_report;
    void*     last_user_data;
} test_interface;

static void test_on_launch(md_trade_instruction* order, uint32_t seq_id, void* user_data) {
    test_interface* ctx = (test_interface*) user_data;
    ctx->n_launch++;
    ctx->last_seq_id = seq_id;
    ctx->last_order = order;
    ctx->last_user_data = user_data;
}

static void test_on_cancel(const md_trade_instruction* order, void* user_data) {
    test_interface* ctx = (test_interface*) user_data;
    ctx->n_cancel++;
    ctx->last_order = order;
    ctx->last_user_data = user_data;
}

static void test_on_reject(const md_trade_instruction* order, void* user_data) {
    test_interface* ctx = (test_interface*) user_data;
    ctx->n_reject++;
    ctx->last_order = order;
    ctx->last_user_data = user_data;
}

static void test_on_filled(const md_trade_instruction* order, md_trade_report* report, void* user_data) {
    test_interface* ctx = (test_interface*) user_data;
    ctx->n_filled++;
    ctx->last_order = order;
    ctx->last_report = report;
    ctx->last_user_data = user_data;
}

/* A handler that unregisters itself while the access is walking the list. */
typedef struct self_unregister_interface {
    dma*      access;
    uintptr_t handler_id;
    size_t    n_calls;
} self_unregister_interface;

static void self_unregister_on_launch(md_trade_instruction* order, uint32_t seq_id, void* user_data) {
    self_unregister_interface* ctx = (self_unregister_interface*) user_data;
    ctx->n_calls++;
    c_dma_unregister_order_handler(ctx->access, ctx->handler_id);
    (void) order;
    (void) seq_id;
}

static void self_unregister_on_cancel(const md_trade_instruction* order, void* user_data) {
    (void) order;
    (void) user_data;
}

static void self_unregister_on_reject(const md_trade_instruction* order, void* user_data) {
    (void) order;
    (void) user_data;
}

static void self_unregister_on_filled(const md_trade_instruction* order, md_trade_report* report, void* user_data) {
    (void) order;
    (void) report;
    (void) user_data;
}

// ========== Helpers ==========

static md_variant make_order(const char* ticker, double volume, md_side side) {
    md_variant order;
    memset(&order, 0, sizeof(md_variant));
    order.meta_info.ticker = ticker;
    order.meta_info.dtype = DTYPE_INSTRUCTION;
    order.trade_instruction.volume = volume;
    order.trade_instruction.side = side;
    order.trade_instruction.order_state = STATE_PENDING;
    return order;
}

static dma_risk_query make_query(const char* ticker, double volume, int side_sign, double price) {
    dma_risk_query query;
    memset(&query, 0, sizeof(dma_risk_query));
    query.ticker = ticker;
    query.volume = volume;
    query.side_sign = side_sign;
    query.market_price = price;
    return query;
}

static void set_full_ticker_rules(dma_risk* risk, const char* ticker, double trade_long, double trade_short, double exposure_long, double exposure_short, double notional_long, double notional_short) {
    CHECK(c_dma_risk_set_limit(risk, ticker, DMA_RULE_MAX_PERCENTILE, INFINITY) == DMA_RISK_OK_RET);
    CHECK(c_dma_risk_set_limit(risk, ticker, DMA_RULE_MAX_TRADE_LONG, trade_long) == DMA_RISK_OK_RET);
    CHECK(c_dma_risk_set_limit(risk, ticker, DMA_RULE_MAX_TRADE_SHORT, trade_short) == DMA_RISK_OK_RET);
    CHECK(c_dma_risk_set_limit(risk, ticker, DMA_RULE_MAX_EXPOSURE_LONG, exposure_long) == DMA_RISK_OK_RET);
    CHECK(c_dma_risk_set_limit(risk, ticker, DMA_RULE_MAX_EXPOSURE_SHORT, exposure_short) == DMA_RISK_OK_RET);
    CHECK(c_dma_risk_set_limit(risk, ticker, DMA_RULE_MAX_NOTIONAL_LONG, notional_long) == DMA_RISK_OK_RET);
    CHECK(c_dma_risk_set_limit(risk, ticker, DMA_RULE_MAX_NOTIONAL_SHORT, notional_short) == DMA_RISK_OK_RET);
}

static void set_full_globals(dma_risk* risk, double ttl_long, double ttl_short, double net_long, double net_short) {
    CHECK(c_dma_risk_set_limit(risk, NULL, DMA_RULE_MAX_TTL_NOTIONAL_LONG, ttl_long) == DMA_RISK_OK_RET);
    CHECK(c_dma_risk_set_limit(risk, NULL, DMA_RULE_MAX_TTL_NOTIONAL_SHORT, ttl_short) == DMA_RISK_OK_RET);
    CHECK(c_dma_risk_set_limit(risk, NULL, DMA_RULE_MAX_NET_NOTIONAL_LONG, net_long) == DMA_RISK_OK_RET);
    CHECK(c_dma_risk_set_limit(risk, NULL, DMA_RULE_MAX_NET_NOTIONAL_SHORT, net_short) == DMA_RISK_OK_RET);
}

// ========== Risk Profile: lifecycle and rules ==========

static void test_risk_profile_rules(void) {
    dma_risk* risk = c_dma_risk_new(g_allocator);
    CHECK(risk != NULL);

    // A fresh profile defines nothing.
    CHECK(c_dma_risk_missing_flags(risk, "AAPL") == (DMA_RISK_FLAGS_TICKER_ALL | DMA_RISK_FLAGS_GLOBAL_ALL));
    CHECK(c_dma_risk_get_entry(risk, "AAPL") == NULL);
    CHECK(!c_dma_risk_has_limit(risk, "AAPL", DMA_RULE_MAX_TRADE_LONG));
    CHECK(!c_dma_risk_has_limit(risk, NULL, DMA_RULE_MAX_NET_NOTIONAL_LONG));

    // Scope errors.
    CHECK(c_dma_risk_set_limit(risk, NULL, DMA_RULE_MAX_TRADE_LONG, 1.) == DMA_RISK_ERR_RULE_SCOPE);
    CHECK(c_dma_risk_set_limit(risk, "AAPL", DMA_RULE_MAX_NET_NOTIONAL_LONG, 1.) == DMA_RISK_ERR_RULE_SCOPE);

    // Ticker rules are stored in one entry and flagged individually.
    CHECK(c_dma_risk_set_limit(risk, "AAPL", DMA_RULE_MAX_TRADE_LONG, 100.) == DMA_RISK_OK_RET);
    CHECK(c_dma_risk_has_limit(risk, "AAPL", DMA_RULE_MAX_TRADE_LONG));
    CHECK(!c_dma_risk_has_limit(risk, "AAPL", DMA_RULE_MAX_TRADE_SHORT));
    CHECK(c_dma_risk_missing_flags(risk, "AAPL") == ((DMA_RISK_FLAGS_TICKER_ALL & ~DMA_RISK_FLAG_MAX_TRADE_LONG) | DMA_RISK_FLAGS_GLOBAL_ALL));

    double value = 0.;
    CHECK(c_dma_risk_get_limit(risk, "AAPL", DMA_RULE_MAX_TRADE_LONG, &value));
    CHECK(value == 100.);
    CHECK(c_dma_risk_get_limit(risk, "AAPL", DMA_RULE_MAX_TRADE_LONG, NULL));

    // Global rules.
    CHECK(c_dma_risk_set_limit(risk, NULL, DMA_RULE_MAX_NET_NOTIONAL_LONG, 1e6) == DMA_RISK_OK_RET);
    CHECK(c_dma_risk_has_limit(risk, NULL, DMA_RULE_MAX_NET_NOTIONAL_LONG));
    CHECK(c_dma_risk_get_limit(risk, NULL, DMA_RULE_MAX_NET_NOTIONAL_LONG, &value));
    CHECK(value == 1e6);

    // Rule names and scope helpers.
    CHECK(strcmp(c_dma_risk_rule_name(DMA_RULE_MAX_TRADE_LONG), "max_trade_long") == 0);
    CHECK(c_dma_risk_rule_name((dma_risk_rule) 99) == NULL);
    CHECK(!c_dma_risk_rule_is_global(DMA_RULE_MAX_PERCENTILE));
    CHECK(c_dma_risk_rule_is_global(DMA_RULE_MAX_NET_NOTIONAL_SHORT));
    CHECK(c_dma_risk_rule_flag(DMA_RULE_MAX_TRADE_SHORT) == DMA_RISK_FLAG_MAX_TRADE_SHORT);

    // Missing-rule listing.
    char   missing[DMA_RISK_MESSAGE_SIZE];
    size_t n_missing = c_dma_risk_missing_rules(risk, "AAPL", missing, sizeof(missing));
    CHECK(n_missing == 9);  // 6 ticker rules + 3 global rules missing.
    CHECK(strstr(missing, "max_trade_short") != NULL);
    CHECK(strstr(missing, "max_net_notional_short") != NULL);

    // clear() drops the ticker rules but keeps the globals defined.
    CHECK(c_dma_risk_clear(risk) == DMA_RISK_OK_RET);
    CHECK(c_dma_risk_get_entry(risk, "AAPL") == NULL);
    CHECK(c_dma_risk_has_limit(risk, NULL, DMA_RULE_MAX_TTL_NOTIONAL_LONG));
    CHECK(c_dma_risk_missing_flags(risk, "AAPL") == DMA_RISK_FLAGS_TICKER_ALL);
    CHECK(c_dma_risk_missing_rules(risk, "AAPL", NULL, 0) == 7);

    c_dma_risk_free(risk);
}

// ========== Risk Check ==========

static void test_risk_check_incomplete(void) {
    dma_risk* risk = c_dma_risk_new(g_allocator);
    dma_risk_violation violation;
    dma_risk_query query = make_query("AAPL", 10., 1, 10.);

    // No rules at all: fail closed with an explicit message.
    CHECK(c_dma_risk_check(risk, &query, &violation) == DMA_RISK_INCOMPLETE);
    CHECK(violation.type == DMA_RISK_INCOMPLETE);
    CHECK(violation.code == 1000);
    CHECK(strstr(violation.msg, "no transaction is allowed") != NULL);
    CHECK(strstr(violation.msg, "max_trade_long") != NULL);
    CHECK(strcmp(c_dma_risk_type_name(violation.type), "RiskProfile.Incomplete") == 0);

    // Ticker complete, globals missing: still refused.
    set_full_ticker_rules(risk, "AAPL", 100., 100., 100., 100., 1000., 1000.);
    CHECK(c_dma_risk_check(risk, &query, &violation) == DMA_RISK_INCOMPLETE);
    CHECK(strstr(violation.msg, "max_ttl_notional_long") != NULL);

    // Globals complete, ticker complete: pass.
    set_full_globals(risk, INFINITY, INFINITY, INFINITY, INFINITY);
    CHECK(c_dma_risk_check(risk, &query, &violation) == DMA_RISK_PASS);
    CHECK(violation.msg[0] == '\0');

    // Unknown ticker: refused.
    dma_risk_query other = make_query("MSFT", 10., 1, 10.);
    CHECK(c_dma_risk_check(risk, &other, &violation) == DMA_RISK_INCOMPLETE);

    c_dma_risk_free(risk);
}

static void test_risk_check_limits(void) {
    dma_risk* risk = c_dma_risk_new(g_allocator);
    dma_risk_violation violation;

    set_full_ticker_rules(risk, "AAPL", 100., 100., 100., 100., 1000., 1000.);
    set_full_globals(risk, INFINITY, INFINITY, INFINITY, INFINITY);

    // Price validity.
    dma_risk_query query = make_query("AAPL", 10., 1, NAN);
    CHECK(c_dma_risk_check(risk, &query, &violation) == DMA_RISK_PRICE);
    CHECK(violation.code == 100);

    // Trade limit.
    CHECK(c_dma_risk_set_limit(risk, "AAPL", DMA_RULE_MAX_TRADE_LONG, 5.) == DMA_RISK_OK_RET);
    query = make_query("AAPL", 10., 1, 10.);
    CHECK(c_dma_risk_check(risk, &query, &violation) == DMA_RISK_TRADE_LONG);
    CHECK(violation.code == 1001);
    CHECK(strstr(violation.msg, "lmt=5") != NULL);
    CHECK(c_dma_risk_set_limit(risk, "AAPL", DMA_RULE_MAX_TRADE_LONG, 100.) == DMA_RISK_OK_RET);

    // Exposure limit: a 10-lot order against a 5-lot long exposure limit.
    CHECK(c_dma_risk_set_limit(risk, "AAPL", DMA_RULE_MAX_EXPOSURE_LONG, 5.) == DMA_RISK_OK_RET);
    CHECK(c_dma_risk_check(risk, &query, &violation) == DMA_RISK_EXPOSURE_LONG);
    CHECK(violation.code == 2001);
    CHECK(c_dma_risk_set_limit(risk, "AAPL", DMA_RULE_MAX_EXPOSURE_LONG, 100.) == DMA_RISK_OK_RET);

    // Notional limit: a 10-lot order at 10 is worth 100.
    CHECK(c_dma_risk_set_limit(risk, "AAPL", DMA_RULE_MAX_NOTIONAL_LONG, 50.) == DMA_RISK_OK_RET);
    CHECK(c_dma_risk_check(risk, &query, &violation) == DMA_RISK_NOTIONAL_LONG);
    CHECK(violation.code == 4001);
    CHECK(c_dma_risk_set_limit(risk, "AAPL", DMA_RULE_MAX_NOTIONAL_LONG, 1000.) == DMA_RISK_OK_RET);

    // Percentile limit: a 50% share of a 1000 notional portfolio is 1000 notional / 10 = 100 lots.
    CHECK(c_dma_risk_set_limit(risk, "AAPL", DMA_RULE_MAX_TRADE_LONG, 1000.) == DMA_RISK_OK_RET);
    CHECK(c_dma_risk_set_limit(risk, "AAPL", DMA_RULE_MAX_EXPOSURE_LONG, 1000.) == DMA_RISK_OK_RET);
    CHECK(c_dma_risk_set_limit(risk, "AAPL", DMA_RULE_MAX_PERCENTILE, 0.5) == DMA_RISK_OK_RET);
    dma_risk_query big = make_query("AAPL", 200., 1, 10.);
    big.portfolio_exposure_notional_abs = 1000.;
    CHECK(c_dma_risk_check(risk, &big, &violation) == DMA_RISK_PERCENTILE_LONG);
    CHECK(violation.code == 3001);
    CHECK(c_dma_risk_set_limit(risk, "AAPL", DMA_RULE_MAX_PERCENTILE, INFINITY) == DMA_RISK_OK_RET);

    // Portfolio net limit.
    set_full_globals(risk, INFINITY, INFINITY, 100., 100.);
    dma_risk_query net = make_query("AAPL", 10., 1, 10.);
    net.portfolio_exposure_notional = 200.;
    CHECK(c_dma_risk_check(risk, &net, &violation) == DMA_RISK_PORTFOLIO_NET_LONG);
    CHECK(violation.code == 5001);

    // Portfolio total limit.
    set_full_globals(risk, 100., 100., INFINITY, INFINITY);
    dma_risk_query ttl = make_query("AAPL", 10., 1, 10.);
    ttl.ttl_exposure_notional_long = 200.;
    CHECK(c_dma_risk_check(risk, &ttl, &violation) == DMA_RISK_PORTFOLIO_TOTAL_LONG);
    CHECK(violation.code == 5003);

    // Short side: a 10-lot sell at 10 against a 5-lot short exposure limit.
    set_full_globals(risk, INFINITY, INFINITY, INFINITY, INFINITY);
    CHECK(c_dma_risk_set_limit(risk, "AAPL", DMA_RULE_MAX_EXPOSURE_SHORT, 5.) == DMA_RISK_OK_RET);
    dma_risk_query short_query = make_query("AAPL", 10., -1, 10.);
    CHECK(c_dma_risk_check(risk, &short_query, &violation) == DMA_RISK_EXPOSURE_SHORT);
    CHECK(violation.code == 2002);

    // Neutral side skips the directional checks.
    dma_risk_query neutral = make_query("AAPL", 10., 0, 10.);
    CHECK(c_dma_risk_check(risk, &neutral, &violation) == DMA_RISK_PASS);

    c_dma_risk_free(risk);
}

// ========== Direct Market Access ==========

static void test_dma_lifecycle_and_gate(void) {
    dma_risk* risk = c_dma_risk_new(g_allocator);
    set_full_ticker_rules(risk, "AAPL", 100., 100., 100., 100., 1000., 1000.);
    set_full_globals(risk, INFINITY, INFINITY, INFINITY, INFINITY);

    dma* access = c_dma_new(NULL, risk, 0.5, g_allocator);
    CHECK(access != NULL);
    CHECK(!c_dma_is_enabled(access));
    CHECK(access->cool_down == 0.5);

    test_interface interface;
    uintptr_t      handler_id = 0;
    memset(&interface, 0, sizeof(test_interface));

    CHECK(c_dma_register_order_handler(access, test_on_launch, test_on_cancel, test_on_reject, test_on_filled, &interface, 7, &handler_id) == DMA_OK);
    CHECK(handler_id != 0);
    CHECK(c_dma_register_order_handler(access, NULL, test_on_cancel, test_on_reject, test_on_filled, &interface, 7, NULL) == DMA_ERR_INVALID_INPUT);
    CHECK(c_dma_register_order_handler(access, test_on_launch, test_on_cancel, test_on_reject, NULL, &interface, 7, NULL) == DMA_ERR_INVALID_INPUT);

    // The position this access trades through is a borrowed, bindable slot.
    dma_position* position = c_dma_position_new(access, NULL, "access-position", NULL, false, g_allocator);
    CHECK(position != NULL);
    CHECK(c_dma_get_position(access) == NULL);
    c_dma_set_position(access, position);
    CHECK(c_dma_get_position(access) == position);
    c_dma_set_position(access, NULL);
    CHECK(c_dma_get_position(access) == NULL);
    c_dma_position_free(position);

    dma_risk_query     query = make_query("AAPL", 10., 1, 10.);
    dma_risk_violation violation;

    // Disabled: refused before the risk check.
    CHECK(c_dma_check_order(access, &query, &violation) == DMA_ERR_DISABLED);

    c_dma_set_enabled(access, true);
    CHECK(c_dma_is_enabled(access));
    CHECK(c_dma_check_order(access, &query, &violation) == DMA_OK);

    // A missing risk profile refuses every order.
    dma* no_risk = c_dma_new(NULL, NULL, 0., g_allocator);
    c_dma_set_enabled(no_risk, true);
    CHECK(c_dma_check_order(no_risk, &query, &violation) == DMA_ERR_NO_RISK);
    c_dma_free(no_risk);

    // Incomplete profile refuses the order.
    dma_risk* empty = c_dma_risk_new(g_allocator);
    dma*     incomplete = c_dma_new(NULL, empty, 0., g_allocator);
    c_dma_set_enabled(incomplete, true);
    CHECK(c_dma_check_order(incomplete, &query, &violation) == DMA_ERR_RISK);
    CHECK(violation.type == DMA_RISK_INCOMPLETE);
    CHECK(strstr(violation.msg, "no transaction is allowed") != NULL);
    c_dma_free(incomplete);
    c_dma_risk_free(empty);

    // Each action invokes its own handler; only the launch carries a seq id.
    md_variant order = make_order("AAPL", 10., SIDE_LONG_OPEN);

    c_dma_invoke_launch_order(access, &order.trade_instruction);
    CHECK(interface.n_launch == 1);
    CHECK(interface.last_seq_id == 7);
    CHECK(interface.last_order == &order.trade_instruction);
    CHECK(interface.last_user_data == &interface);

    c_dma_invoke_cancel_order(access, &order.trade_instruction);
    CHECK(interface.n_cancel == 1);
    CHECK(interface.last_order == &order.trade_instruction);

    c_dma_invoke_reject_order(access, &order.trade_instruction);
    CHECK(interface.n_reject == 1);
    CHECK(interface.last_order == &order.trade_instruction);

    md_trade_report report;
    memset(&report, 0, sizeof(md_trade_report));

    c_dma_invoke_filled_order(access, &order.trade_instruction, &report);
    CHECK(interface.n_filled == 1);
    CHECK(interface.last_order == &order.trade_instruction);
    CHECK(interface.last_report == &report);

    // Unregistering stops the dispatch; a second unregister misses.
    CHECK(c_dma_unregister_order_handler(access, handler_id) == DMA_OK);
    c_dma_invoke_launch_order(access, &order.trade_instruction);
    CHECK(interface.n_launch == 1);
    c_dma_invoke_filled_order(access, &order.trade_instruction, &report);
    CHECK(interface.n_filled == 1);
    CHECK(c_dma_unregister_order_handler(access, handler_id) == DMA_ERR_NOT_FOUND);

    c_dma_free(access);
    c_dma_risk_free(risk);
}

static void test_dma_request_queue(void) {
    dma* access = c_dma_new(NULL, NULL, 0.1, g_allocator);
    CHECK(access != NULL);

    md_variant order_a = make_order("AAPL", 1., SIDE_LONG_OPEN);
    md_variant order_b = make_order("AAPL", 2., SIDE_SHORT_OPEN);
    md_variant order_c = make_order("AAPL", 3., SIDE_LONG_OPEN);

    CHECK(c_dma_request_order(access, DMA_ACTION_LAUNCH, &order_a.trade_instruction) == DMA_OK);
    CHECK(c_dma_request_order(access, DMA_ACTION_CANCEL, &order_b.trade_instruction) == DMA_OK);
    CHECK(c_dma_request_order(access, DMA_ACTION_LAUNCH, &order_c.trade_instruction) == DMA_OK);
    CHECK(access->n_requests == 3);

    dma_order_action      action;
    md_trade_instruction* order = NULL;

    CHECK(c_dma_pop_request(access, &action, &order) == DMA_OK);
    CHECK(action == DMA_ACTION_LAUNCH);
    CHECK(order == &order_a.trade_instruction);
    CHECK(access->n_requests == 2);

    CHECK(c_dma_pop_request(access, &action, &order) == DMA_OK);
    CHECK(action == DMA_ACTION_CANCEL);
    CHECK(order == &order_b.trade_instruction);

    CHECK(c_dma_pop_request(access, &action, &order) == DMA_OK);
    CHECK(order == &order_c.trade_instruction);

    CHECK(c_dma_pop_request(access, &action, &order) == DMA_ERR_EMPTY);
    CHECK(access->n_requests == 0);
    CHECK(access->head == NULL && access->tail == NULL);

    // clear_requests drops queued nodes; later teardown must stay clean.
    CHECK(c_dma_request_order(access, DMA_ACTION_LAUNCH, &order_a.trade_instruction) == DMA_OK);
    c_dma_clear_requests(access);
    CHECK(access->n_requests == 0);

    CHECK(c_dma_request_order(access, DMA_ACTION_LAUNCH, &order_a.trade_instruction) == DMA_OK);
    CHECK(c_dma_inventory_add(&access->inventory, &(dma_inventory_entry) {
        .ticker = "AAPL",
        .volume = 1.,
        .price = 1.,
        .notional = 1.,
        .direction = DIRECTION_LONG,
        .security_type = DMA_SECURITY_STOCK,
    }) == INVENTORY_OK);
    CHECK(c_dma_inventory_available(&access->inventory, "AAPL", DIRECTION_LONG) == 1.);
    c_dma_free(access);
}

static void test_dma_order_handler_self_unregister(void) {
    dma* access = c_dma_new(NULL, NULL, 0., g_allocator);
    CHECK(access != NULL);

    self_unregister_interface interface;
    memset(&interface, 0, sizeof(interface));
    interface.access = access;

    CHECK(c_dma_register_order_handler(access, self_unregister_on_launch, self_unregister_on_cancel, self_unregister_on_reject, self_unregister_on_filled, &interface, 0, &interface.handler_id) == DMA_OK);

    md_variant order = make_order("AAPL", 1., SIDE_LONG_OPEN);

    c_dma_invoke_launch_order(access, &order.trade_instruction);
    CHECK(interface.n_calls == 1);
    CHECK(access->handlers == NULL);

    c_dma_invoke_launch_order(access, &order.trade_instruction);
    CHECK(interface.n_calls == 1);

    c_dma_free(access);
}

// ========== Inventory ==========

static void test_inventory(void) {
    dma_inventory* inventory = c_dma_inventory_new(g_allocator);
    CHECK(inventory != NULL);

    CHECK(c_dma_inventory_available(inventory, "AAPL", DIRECTION_LONG) == 0.);
    CHECK(c_dma_inventory_get(inventory, "AAPL", DIRECTION_LONG, NULL) == INVENTORY_ERR_NOT_FOUND);

    dma_inventory_entry lot_a = {
        .ticker = "AAPL",
        .volume = 100.,
        .price = 10.,
        .notional = 1000.,
        .fee = 1.,
        .recalled = 0.,
        .direction = DIRECTION_LONG,
        .security_type = DMA_SECURITY_STOCK,
    };

    dma_inventory_entry lot_b = {
        .ticker = "AAPL",
        .volume = 100.,
        .price = 12.,
        .notional = 1200.,
        .fee = 2.,
        .recalled = 0.,
        .direction = DIRECTION_LONG,
        .security_type = DMA_SECURITY_STOCK,
    };

    CHECK(c_dma_inventory_add(inventory, &lot_a) == INVENTORY_OK);
    CHECK(c_dma_inventory_add(inventory, &lot_b) == INVENTORY_OK);

    // The stored node owns its ticker copy, not the caller's pointer.
    bytemap_entry* bucket_entry = inventory->buckets->first;
    CHECK(bucket_entry != NULL);
    dma_inventory_bucket* bucket = (dma_inventory_bucket*) bucket_entry->value;
    CHECK(bucket->counts[DIRECTION_LONG] == 2);
    CHECK(bucket->heads[DIRECTION_LONG]->entry.ticker != lot_a.ticker);
    CHECK(strcmp(bucket->heads[DIRECTION_LONG]->entry.ticker, "AAPL") == 0);

    dma_inventory_entry merged;
    CHECK(c_dma_inventory_get(inventory, "AAPL", DIRECTION_LONG, &merged) == INVENTORY_OK);
    CHECK(merged.volume == 200.);
    CHECK(merged.notional == 2200.);
    CHECK(merged.fee == 3.);
    CHECK(merged.recalled == 0.);
    CHECK(merged.price == 11.);
    CHECK(merged.direction == DIRECTION_LONG);
    CHECK(merged.security_type == DMA_SECURITY_STOCK);
    CHECK(c_dma_inventory_entry_available(&merged) == 200.);

    // The short direction is a separate list.
    CHECK(c_dma_inventory_get(inventory, "AAPL", DIRECTION_SHORT, NULL) == INVENTORY_ERR_NOT_FOUND);

    // Consumed volume reduces availability.
    CHECK(c_dma_inventory_use(inventory, "AAPL", DIRECTION_LONG, 50.) == INVENTORY_OK);
    CHECK(c_dma_inventory_available(inventory, "AAPL", DIRECTION_LONG) == 150.);
    CHECK(c_dma_inventory_available(inventory, "AAPL", DIRECTION_SHORT) == 0.);

    // Recall 120: the first lot is fully recalled and removed, the second keeps 20.
    CHECK(c_dma_inventory_recall(inventory, "AAPL", DIRECTION_LONG, 120.) == INVENTORY_OK);
    CHECK(bucket->counts[DIRECTION_LONG] == 1);

    CHECK(c_dma_inventory_get(inventory, "AAPL", DIRECTION_LONG, &merged) == INVENTORY_OK);
    CHECK(merged.volume == 100.);
    CHECK(merged.recalled == 20.);
    CHECK(c_dma_inventory_available(inventory, "AAPL", DIRECTION_LONG) == 30.);

    // Recalling past the held volume removes the remaining lot.
    CHECK(c_dma_inventory_recall(inventory, "AAPL", DIRECTION_LONG, 1000.) == INVENTORY_OK);
    CHECK(bucket->counts[DIRECTION_LONG] == 0);
    CHECK(c_dma_inventory_get(inventory, "AAPL", DIRECTION_LONG, NULL) == INVENTORY_ERR_NOT_FOUND);

    CHECK(c_dma_inventory_clear(inventory) == INVENTORY_OK);
    CHECK(c_bytemap_ex_len(inventory->buckets) == 0);

    c_dma_inventory_free(inventory);
}

// ========== Position Management ==========

static void test_position(void) {
    dma* access = c_dma_new(NULL, NULL, 0., g_allocator);

    int algo_a = 1;
    int algo_b = 2;

    dma_position* position = c_dma_position_new(access, (void*) &algo_a, "position-1", "passive", false, g_allocator);
    CHECK(position != NULL);
    CHECK(position->dma == access);
    CHECK(position->algo_engine == (void*) &algo_a);
    CHECK(strcmp(position->position_id, "position-1") == 0);
    CHECK(strcmp(position->default_algo, "passive") == 0);
    CHECK(!position->no_cache);

    CHECK(c_dma_position_set_algo(position, "algo-a", &algo_a) == POSITION_OK);
    CHECK(c_dma_position_set_algo(position, "algo-b", &algo_b) == POSITION_OK);
    CHECK(c_dma_position_get_algo(position, "algo-a") == &algo_a);
    CHECK(c_dma_position_get_algo(position, "algo-b") == &algo_b);
    CHECK(c_dma_position_get_algo(position, "algo-c") == NULL);

    CHECK(c_dma_position_set_working_algo(position, "algo-a", &algo_a) == POSITION_OK);
    CHECK(c_bytemap_ex_len(position->algos) == 2);
    CHECK(c_bytemap_ex_len(position->working_algos) == 1);

    CHECK(c_dma_position_pop_working_algo(position, "algo-a") == POSITION_OK);
    CHECK(c_dma_position_pop_working_algo(position, "algo-a") == POSITION_ERR_NOT_FOUND);
    CHECK(c_bytemap_ex_len(position->working_algos) == 0);

    CHECK(c_dma_position_pop_algo(position, "algo-b") == POSITION_OK);
    CHECK(c_dma_position_pop_algo(position, "algo-b") == POSITION_ERR_NOT_FOUND);
    CHECK(c_bytemap_ex_len(position->algos) == 1);

    CHECK(c_dma_position_clear(position) == POSITION_OK);
    CHECK(c_bytemap_ex_len(position->algos) == 0);

    c_dma_position_free(position);
    c_dma_free(access);
}

// ========== Main ==========

int main(void) {
    void* root = c_ap_alloc(1, NULL);
    g_allocator = c_ap_protocol_from_ptr(root);

    test_risk_profile_rules();
    test_risk_check_incomplete();
    test_risk_check_limits();
    test_dma_lifecycle_and_gate();
    test_dma_request_queue();
    test_dma_order_handler_self_unregister();
    test_inventory();
    test_position();

    c_ap_free(root);

    printf("%zu checks, %zu failures\n", g_checks, g_failures);
    return g_failures ? 1 : 0;
}

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
 *      interface dispatch, the buffered request queue.
 *   4. The inventory: add, merge, available, use, recall, clear.
 *   5. The position: algo hosting, event routing, aggregates and callbacks.
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
#include <time.h>

#include <algo_engine/engine/c_dma.h>
#include <algo_engine/engine/c_inventory.h>
#include <algo_engine/engine/c_position.h>
#include <algo_engine/engine/risk_profile/c_refusal_window.h>

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

// ========== Session Time Helper ==========

/* A session time with its sortable elapsed-seconds field filled in. */
static session_time_t make_session_time(uint8_t hour, uint8_t minute) {
    session_time_t time;
    memset(&time, 0, sizeof(time));
    time.hour = hour;
    time.minute = minute;
    time.elapsed_seconds = hour * 3600. + minute * 60.;
    return time;
}

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

static void test_on_cancel(const md_trade_instruction* order, uint32_t seq_id, void* user_data) {
    test_interface* ctx = (test_interface*) user_data;
    ctx->n_cancel++;
    ctx->last_seq_id = seq_id;
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

/* A handler that detaches itself from inside the launch dispatch. */
typedef struct self_detach_interface {
    dma*   access;
    size_t n_calls;
} self_detach_interface;

static void self_detach_on_launch(md_trade_instruction* order, uint32_t seq_id, void* user_data) {
    self_detach_interface* ctx = (self_detach_interface*) user_data;
    ctx->n_calls++;
    c_dma_detach_protocol_handler(ctx->access);
    (void) order;
    (void) seq_id;
}

static void self_detach_on_cancel(const md_trade_instruction* order, uint32_t seq_id, void* user_data) {
    (void) order;
    (void) seq_id;
    (void) user_data;
}

static void self_detach_on_reject(const md_trade_instruction* order, void* user_data) {
    (void) order;
    (void) user_data;
}

static void self_detach_on_filled(const md_trade_instruction* order, md_trade_report* report, void* user_data) {
    (void) order;
    (void) report;
    (void) user_data;
}

/* A balance query that supplies a fixed cash and margin capacity, or declines. */
typedef struct test_balance_ctx {
    double available_cash;
    double available_margin;
    int    supply;
} test_balance_ctx;

static int test_balance_query(const void* DMA, dma_balance* out, void* user_data) {
    (void) DMA;

    test_balance_ctx* ctx = (test_balance_ctx*) user_data;

    if (!ctx->supply) return -1;

    out->available_cash   = ctx->available_cash;
    out->available_margin = ctx->available_margin;
    out->flags |= DMA_BALANCE_F_AVAILABLE_CASH | DMA_BALANCE_F_AVAILABLE_MARGIN;
    return DMA_OK;
}

// ========== Helpers ==========

static md_variant make_order(const char* ticker, double volume, md_side side) {
    md_variant order;
    memset(&order, 0, sizeof(md_variant));
    order.meta_info.ticker = ticker;
    order.meta_info.dtype = DTYPE_INSTRUCTION;
    order.trade_instruction.volume = volume;
    order.trade_instruction.side = side;
    order.trade_instruction.order_state = STATE_IDLE;
    return order;
}

static dma_risk_check_state make_query(const char* ticker, double volume, int side_sign, double price) {
    dma_risk_check_state check_state;
    memset(&check_state, 0, sizeof(dma_risk_check_state));
    check_state.ticker = ticker;
    check_state.volume = volume;
    check_state.side_sign = side_sign;
    check_state.market_price = price;
    return check_state;
}

static void set_full_ticker_rules(dma_risk_profile* risk, const char* ticker, double trade_long, double trade_short, double exposure_long, double exposure_short, double notional_long, double notional_short) {
    CHECK(c_dma_risk_profile_set_limit(risk, ticker, DMA_RULE_MAX_PERCENTILE, INFINITY) == DMA_RISK_OK_RET);
    CHECK(c_dma_risk_profile_set_limit(risk, ticker, DMA_RULE_MAX_TRADE_LONG, trade_long) == DMA_RISK_OK_RET);
    CHECK(c_dma_risk_profile_set_limit(risk, ticker, DMA_RULE_MAX_TRADE_SHORT, trade_short) == DMA_RISK_OK_RET);
    CHECK(c_dma_risk_profile_set_limit(risk, ticker, DMA_RULE_MAX_EXPOSURE_LONG, exposure_long) == DMA_RISK_OK_RET);
    CHECK(c_dma_risk_profile_set_limit(risk, ticker, DMA_RULE_MAX_EXPOSURE_SHORT, exposure_short) == DMA_RISK_OK_RET);
    CHECK(c_dma_risk_profile_set_limit(risk, ticker, DMA_RULE_MAX_NOTIONAL_LONG, notional_long) == DMA_RISK_OK_RET);
    CHECK(c_dma_risk_profile_set_limit(risk, ticker, DMA_RULE_MAX_NOTIONAL_SHORT, notional_short) == DMA_RISK_OK_RET);
}

static void set_full_globals(dma_risk_profile* risk, double ttl_long, double ttl_short, double net_long, double net_short) {
    CHECK(c_dma_risk_profile_set_limit(risk, NULL, DMA_RULE_MAX_TTL_NOTIONAL_LONG, ttl_long) == DMA_RISK_OK_RET);
    CHECK(c_dma_risk_profile_set_limit(risk, NULL, DMA_RULE_MAX_TTL_NOTIONAL_SHORT, ttl_short) == DMA_RISK_OK_RET);
    CHECK(c_dma_risk_profile_set_limit(risk, NULL, DMA_RULE_MAX_NET_NOTIONAL_LONG, net_long) == DMA_RISK_OK_RET);
    CHECK(c_dma_risk_profile_set_limit(risk, NULL, DMA_RULE_MAX_NET_NOTIONAL_SHORT, net_short) == DMA_RISK_OK_RET);
}

// ========== Risk Profile: lifecycle and rules ==========

static void test_risk_profile_rules(void) {
    dma_risk_profile* risk = c_dma_risk_profile_new(g_allocator);
    CHECK(risk != NULL);

    // A fresh profile enables nothing: every rule is disabled and stores 0.
    CHECK(c_dma_risk_profile_get_limits(risk, "AAPL") == NULL);
    CHECK(!c_dma_risk_profile_get_limit(risk, "AAPL", DMA_RULE_MAX_TRADE_LONG, NULL));
    CHECK(!c_dma_risk_profile_get_limit(risk, NULL, DMA_RULE_MAX_NET_NOTIONAL_LONG, NULL));
    CHECK(risk->globals.enabled == 0);
    CHECK(risk->globals.max_ttl_notional_long == 0.);

    // Scope errors.
    CHECK(c_dma_risk_profile_set_limit(risk, NULL, DMA_RULE_MAX_TRADE_LONG, 1.) == DMA_RISK_ERR_RULE_SCOPE);
    CHECK(c_dma_risk_profile_set_limit(risk, "AAPL", DMA_RULE_MAX_NET_NOTIONAL_LONG, 1.) == DMA_RISK_ERR_RULE_SCOPE);

    // NaN and negative values are refused.
    CHECK(c_dma_risk_profile_set_limit(risk, "AAPL", DMA_RULE_MAX_TRADE_LONG, NAN) == DMA_RISK_ERR_INVALID_INPUT);
    CHECK(c_dma_risk_profile_set_limit(risk, "AAPL", DMA_RULE_MAX_TRADE_LONG, -1.) == DMA_RISK_ERR_INVALID_INPUT);
    CHECK(c_dma_risk_profile_set_limit(risk, NULL, DMA_RULE_MAX_TTL_NOTIONAL_LONG, -1.) == DMA_RISK_ERR_INVALID_INPUT);
    CHECK(c_dma_risk_profile_get_limits(risk, "AAPL") == NULL);

    // Writing one ticker rule creates the limit set; the others stay disabled.
    CHECK(c_dma_risk_profile_set_limit(risk, "AAPL", DMA_RULE_MAX_TRADE_LONG, 100.) == DMA_RISK_OK_RET);
    CHECK(c_dma_risk_profile_get_limits(risk, "AAPL") != NULL);

    // A limit set is its own allocator block: the pointer stays valid and
    // keeps its value when the map grows underneath it.
    dma_risk_limits* aapl_limits = c_dma_risk_profile_get_limits(risk, "AAPL");
    CHECK(aapl_limits != NULL);
    CHECK(aapl_limits->enabled == c_dma_risk_rule_bit(DMA_RULE_MAX_TRADE_LONG));
    CHECK(aapl_limits->max_trade_short == 0.);

    char grow_ticker[32];
    for (int i = 0; i < 256; i++) {
        snprintf(grow_ticker, sizeof(grow_ticker), "GROW%03d.SH", i);
        CHECK(c_dma_risk_profile_set_limit(risk, grow_ticker, DMA_RULE_MAX_TRADE_LONG, 1.) == DMA_RISK_OK_RET);
    }

    CHECK(c_dma_risk_profile_get_limits(risk, "AAPL") == aapl_limits);
    CHECK(aapl_limits->max_trade_long == 100.);

    // An infinite limit disables the rule: the value resets to 0 and the bit
    // clears, so the rule neither constrains nor reads.
    CHECK(c_dma_risk_profile_set_limit(risk, "AAPL", DMA_RULE_MAX_TRADE_LONG, INFINITY) == DMA_RISK_OK_RET);
    CHECK(aapl_limits->max_trade_long == 0.);
    CHECK(!(aapl_limits->enabled & c_dma_risk_rule_bit(DMA_RULE_MAX_TRADE_LONG)));
    CHECK(!c_dma_risk_profile_get_limit(risk, "AAPL", DMA_RULE_MAX_TRADE_LONG, NULL));

    // A percentile of one or more disables the rule too.
    CHECK(c_dma_risk_profile_set_limit(risk, "AAPL", DMA_RULE_MAX_PERCENTILE, 1.) == DMA_RISK_OK_RET);
    CHECK(!c_dma_risk_profile_get_limit(risk, "AAPL", DMA_RULE_MAX_PERCENTILE, NULL));

    CHECK(c_dma_risk_profile_set_limit(risk, "AAPL", DMA_RULE_MAX_TRADE_LONG, 100.) == DMA_RISK_OK_RET);
    CHECK(aapl_limits->max_trade_long == 100.);

    double value = 0.;
    CHECK(c_dma_risk_profile_get_limit(risk, "AAPL", DMA_RULE_MAX_TRADE_LONG, &value));
    CHECK(value == 100.);
    CHECK(c_dma_risk_profile_get_limit(risk, "AAPL", DMA_RULE_MAX_TRADE_LONG, NULL));
    CHECK(!c_dma_risk_profile_get_limit(risk, "AAPL", DMA_RULE_MAX_TRADE_SHORT, NULL));
    CHECK(!c_dma_risk_profile_get_limit(risk, "AAPL", DMA_RULE_MAX_PERCENTILE, NULL));

    // Global rules.
    CHECK(c_dma_risk_profile_set_limit(risk, NULL, DMA_RULE_MAX_NET_NOTIONAL_LONG, 1e6) == DMA_RISK_OK_RET);
    CHECK(c_dma_risk_profile_get_limit(risk, NULL, DMA_RULE_MAX_NET_NOTIONAL_LONG, &value));
    CHECK(value == 1e6);
    CHECK(!c_dma_risk_profile_get_limit(risk, NULL, DMA_RULE_MAX_NET_NOTIONAL_SHORT, NULL));

    // Rule names, scope and side applicability.
    for (int rule = DMA_RULE_MAX_PERCENTILE; rule <= DMA_RULE_MAX_NET_NOTIONAL_SHORT; rule++) {
        const char* name = c_dma_risk_rule_name((dma_risk_rule) rule);
        CHECK(name != NULL);
        CHECK(name[0] != '\0');
        CHECK(c_dma_risk_rule_is_global((dma_risk_rule) rule) == (rule >= DMA_RULE_MAX_TTL_NOTIONAL_LONG));
        CHECK(c_dma_risk_rule_applies_to((dma_risk_rule) rule, 1) || c_dma_risk_rule_applies_to((dma_risk_rule) rule, -1));
        CHECK(c_dma_risk_rule_applies_to((dma_risk_rule) rule, 0));
    }

    CHECK(c_dma_risk_rule_applies_to(DMA_RULE_MAX_TRADE_LONG, 1));
    CHECK(!c_dma_risk_rule_applies_to(DMA_RULE_MAX_TRADE_LONG, -1));
    CHECK(c_dma_risk_rule_applies_to(DMA_RULE_MAX_PERCENTILE, 1));
    CHECK(c_dma_risk_rule_applies_to(DMA_RULE_MAX_PERCENTILE, -1));
    CHECK(strcmp(c_dma_risk_rule_name(DMA_RULE_MAX_TRADE_LONG), "max_trade_long") == 0);
    CHECK(c_dma_risk_rule_name((dma_risk_rule) 99) == NULL);

    // Every violation type has a name and a code.
    for (int type = DMA_RISK_PASS; type <= DMA_RISK_CUSTOM; type++) {
        CHECK(c_dma_risk_type_name((dma_risk_type) type) != NULL);
        CHECK(strncmp(c_dma_risk_type_name((dma_risk_type) type), "RiskProfile.", 12) == 0);
    }

    CHECK(c_dma_risk_violation_code(DMA_RISK_PASS) == 0);
    CHECK(c_dma_risk_violation_code(DMA_RISK_PRICE) == 100);
    CHECK(c_dma_risk_violation_code(DMA_RISK_TRADE_LONG) == 1001);
    CHECK(c_dma_risk_violation_code(DMA_RISK_TRADE_SHORT) == 1002);
    CHECK(c_dma_risk_violation_code(DMA_RISK_TRADE_INVALID) == 1003);
    CHECK(c_dma_risk_violation_code(DMA_RISK_EXPOSURE_LONG) == 2001);
    CHECK(c_dma_risk_violation_code(DMA_RISK_EXPOSURE_SHORT) == 2002);
    CHECK(c_dma_risk_violation_code(DMA_RISK_PERCENTILE_LONG) == 3001);
    CHECK(c_dma_risk_violation_code(DMA_RISK_PERCENTILE_SHORT) == 3002);
    CHECK(c_dma_risk_violation_code(DMA_RISK_NOTIONAL_LONG) == 4001);
    CHECK(c_dma_risk_violation_code(DMA_RISK_NOTIONAL_SHORT) == 4002);
    CHECK(c_dma_risk_violation_code(DMA_RISK_PORTFOLIO_NET_LONG) == 5001);
    CHECK(c_dma_risk_violation_code(DMA_RISK_PORTFOLIO_NET_SHORT) == 5002);
    CHECK(c_dma_risk_violation_code(DMA_RISK_PORTFOLIO_TOTAL_LONG) == 5003);
    CHECK(c_dma_risk_violation_code(DMA_RISK_PORTFOLIO_TOTAL_SHORT) == 5004);
    CHECK(c_dma_risk_violation_code(DMA_RISK_TICKER_BLOCKED) == 1004);
    CHECK(c_dma_risk_violation_code(DMA_RISK_TICKER_UNLISTED) == 1005);
    CHECK(c_dma_risk_violation_code(DMA_RISK_CUSTOM) == 0);
    CHECK(c_dma_risk_violation_code(DMA_RISK_INVALID) == 0);

    // Margin rates: a finite rate designates the ticker, NaN removes it, and
    // clear() frees the map.
    CHECK(isnan(c_dma_risk_profile_get_margin_rate(risk, "AAPL")));
    CHECK(c_dma_risk_profile_set_margin_rate(risk, "AAPL", 0.5) == DMA_RISK_OK_RET);
    CHECK(c_dma_risk_profile_get_margin_rate(risk, "AAPL") == 0.5);
    CHECK(c_dma_risk_profile_set_margin_rate(risk, "AAPL", 1.) == DMA_RISK_OK_RET);
    CHECK(c_dma_risk_profile_get_margin_rate(risk, "AAPL") == 1.);
    CHECK(c_dma_risk_profile_set_margin_rate(risk, "AAPL", -1.) == DMA_RISK_ERR_INVALID_INPUT);
    CHECK(c_dma_risk_profile_set_margin_rate(risk, "AAPL", INFINITY) == DMA_RISK_ERR_INVALID_INPUT);
    CHECK(c_dma_risk_profile_set_margin_rate(risk, "AAPL", NAN) == DMA_RISK_OK_RET);
    CHECK(isnan(c_dma_risk_profile_get_margin_rate(risk, "AAPL")));
    CHECK(c_dma_risk_profile_set_margin_rate(NULL, "AAPL", 0.5) == DMA_RISK_ERR_INVALID_INPUT);
    CHECK(isnan(c_dma_risk_profile_get_margin_rate(NULL, "AAPL")));

    // clear() drops the ticker rules and disables the portfolio limits: the
    // profile constrains nothing until a rule is set again.
    CHECK(c_dma_risk_profile_clear(risk) == DMA_RISK_OK_RET);
    CHECK(c_dma_risk_profile_get_limits(risk, "AAPL") == NULL);
    CHECK(!c_dma_risk_profile_get_limit(risk, NULL, DMA_RULE_MAX_TTL_NOTIONAL_LONG, NULL));
    CHECK(risk->globals.enabled == 0);
    CHECK(risk->globals.max_ttl_notional_long == 0.);
    CHECK(risk->margin_rate == NULL);
    CHECK(isnan(c_dma_risk_profile_get_margin_rate(risk, "AAPL")));

    c_dma_risk_profile_free(risk);
}

static void test_risk_ticker_gates(void) {
    dma_risk_profile* risk = c_dma_risk_profile_new(g_allocator);

    // The gates are absent until a ticker is listed: every ticker may trade.
    CHECK(risk->whitelist == NULL);
    CHECK(risk->blacklist == NULL);
    CHECK(c_dma_risk_profile_is_ticker_allowed(risk, "AAPL"));
    CHECK(c_dma_risk_profile_is_ticker_allowed(risk, "MSFT"));

    // The whitelist is created on first add and allows only what it lists.
    CHECK(c_dma_risk_profile_whitelist_add(risk, "AAPL") == DMA_RISK_OK_RET);
    CHECK(risk->whitelist != NULL);
    CHECK(risk->blacklist == NULL);
    CHECK(c_dma_risk_profile_is_ticker_allowed(risk, "AAPL"));
    CHECK(!c_dma_risk_profile_is_ticker_allowed(risk, "MSFT"));

    // Removing a missing ticker is a no-op.
    CHECK(c_dma_risk_profile_whitelist_remove(risk, "MSFT") == DMA_RISK_OK_RET);

    // A ticker cannot be on both lists: the add refuses the conflict.
    CHECK(c_dma_risk_profile_blacklist_add(risk, "AAPL") == DMA_RISK_ERR_CONFLICT);
    CHECK(risk->blacklist == NULL);

    // A blacklist alone refuses its listed tickers.
    CHECK(c_dma_risk_profile_blacklist_add(risk, "MSFT") == DMA_RISK_OK_RET);
    CHECK(risk->blacklist != NULL);
    CHECK(c_dma_risk_profile_is_ticker_allowed(risk, "AAPL"));
    CHECK(!c_dma_risk_profile_is_ticker_allowed(risk, "MSFT"));

    // Dropping the last whitelisted ticker frees the whitelist; the gate is
    // skipped from then on.
    CHECK(c_dma_risk_profile_whitelist_remove(risk, "AAPL") == DMA_RISK_OK_RET);
    CHECK(risk->whitelist == NULL);
    CHECK(c_dma_risk_profile_is_ticker_allowed(risk, "XYZ"));
    CHECK(!c_dma_risk_profile_is_ticker_allowed(risk, "MSFT"));

    // The reverse conflict is refused too.
    CHECK(c_dma_risk_profile_whitelist_add(risk, "XYZ") == DMA_RISK_OK_RET);
    CHECK(c_dma_risk_profile_whitelist_add(risk, "MSFT") == DMA_RISK_ERR_CONFLICT);
    CHECK(!c_dma_risk_profile_is_ticker_allowed(risk, "AAPL"));

    // clear() frees both gates: every ticker may trade again.
    CHECK(c_dma_risk_profile_clear(risk) == DMA_RISK_OK_RET);
    CHECK(risk->whitelist == NULL);
    CHECK(risk->blacklist == NULL);
    CHECK(c_dma_risk_profile_is_ticker_allowed(risk, "AAPL"));
    CHECK(c_dma_risk_profile_is_ticker_allowed(risk, "MSFT"));

    // Invalid input.
    CHECK(c_dma_risk_profile_whitelist_add(NULL, "AAPL") == DMA_RISK_ERR_INVALID_INPUT);
    CHECK(c_dma_risk_profile_whitelist_add(risk, NULL) == DMA_RISK_ERR_INVALID_INPUT);
    CHECK(c_dma_risk_profile_whitelist_add(risk, "") == DMA_RISK_ERR_INVALID_INPUT);
    CHECK(c_dma_risk_profile_blacklist_add(risk, NULL) == DMA_RISK_ERR_INVALID_INPUT);
    CHECK(c_dma_risk_profile_whitelist_remove(risk, NULL) == DMA_RISK_ERR_INVALID_INPUT);
    CHECK(c_dma_risk_profile_blacklist_remove(risk, NULL) == DMA_RISK_ERR_INVALID_INPUT);

    c_dma_risk_profile_free(risk);
}

// ========== Risk Check ==========

static void test_risk_check_incomplete(void) {
    dma_risk_profile*          risk = c_dma_risk_profile_new(g_allocator);
    dma_risk_violation violation;
    dma_risk_check_state     check_state = make_query("AAPL", 10., 1, 10.);
    dma_risk_check_state     short_query = make_query("AAPL", 10., -1, 10.);

    // A fresh profile constrains nothing: any transaction is refused.
    CHECK(c_dma_risk_profile_check(risk, &check_state, &violation) == DMA_RISK_INCOMPLETE);
    CHECK(violation.type == DMA_RISK_INCOMPLETE);
    CHECK(violation.code == 1000);
    CHECK(strstr(violation.msg, "no rule constrains a LONG order") != NULL);
    CHECK(strstr(violation.msg, "max_trade_long") != NULL);
    CHECK(strstr(violation.msg, "max_exposure_long") != NULL);
    CHECK(strstr(violation.msg, "max_net_notional_short") == NULL);
    CHECK(strcmp(c_dma_risk_type_name(violation.type), "RiskProfile.Incomplete") == 0);

    // The message names the side's rules.
    CHECK(c_dma_risk_profile_check(risk, &short_query, &violation) == DMA_RISK_INCOMPLETE);
    CHECK(strstr(violation.msg, "SHORT") != NULL);
    CHECK(strstr(violation.msg, "max_trade_short") != NULL);
    CHECK(strstr(violation.msg, "max_trade_long") == NULL);

    // Rules on another ticker do not constrain this one, and no global rule
    // is defined.
    set_full_ticker_rules(risk, "MSFT", 100., 100., 100., 100., 1000., 1000.);
    CHECK(c_dma_risk_profile_check(risk, &check_state, &violation) == DMA_RISK_INCOMPLETE);

    // The percentile limit is relative to the portfolio: defined alone it
    // does not constrain the side.
    dma_risk_profile* percentile_only = c_dma_risk_profile_new(g_allocator);
    CHECK(percentile_only != NULL);
    CHECK(c_dma_risk_profile_set_limit(percentile_only, "AAPL", DMA_RULE_MAX_PERCENTILE, 0.5) == DMA_RISK_OK_RET);
    CHECK(c_dma_risk_profile_check(percentile_only, &check_state, &violation) == DMA_RISK_INCOMPLETE);
    CHECK(strstr(violation.msg, "no rule constrains a LONG order") != NULL);
    CHECK(strstr(violation.msg, "max_percentile") == NULL);
    CHECK(c_dma_risk_profile_check(percentile_only, &short_query, &violation) == DMA_RISK_INCOMPLETE);
    CHECK(strstr(violation.msg, "no rule constrains a SHORT order") != NULL);
    c_dma_risk_profile_free(percentile_only);

    // One defined long rule constrains the long side ...
    CHECK(c_dma_risk_profile_set_limit(risk, "AAPL", DMA_RULE_MAX_TRADE_LONG, 100.) == DMA_RISK_OK_RET);
    CHECK(c_dma_risk_profile_check(risk, &check_state, &violation) == DMA_RISK_PASS);

    // ... while the short side of the ticker has none.
    CHECK(c_dma_risk_profile_check(risk, &short_query, &violation) == DMA_RISK_INCOMPLETE);
    CHECK(strstr(violation.msg, "max_trade_short") != NULL);

    // A global rule alone constrains its side for any ticker.
    dma_risk_profile* global_only = c_dma_risk_profile_new(g_allocator);
    CHECK(c_dma_risk_profile_set_limit(global_only, NULL, DMA_RULE_MAX_TTL_NOTIONAL_LONG, 1e9) == DMA_RISK_OK_RET);
    CHECK(c_dma_risk_profile_check(global_only, &check_state, &violation) == DMA_RISK_PASS);
    CHECK(c_dma_risk_profile_check(global_only, &short_query, &violation) == DMA_RISK_INCOMPLETE);
    c_dma_risk_profile_free(global_only);

    // A missing market price is refused directly, even when only a
    // volume-only rule is enabled.
    dma_risk_check_state no_price = make_query("AAPL", 10., 1, NAN);
    CHECK(c_dma_risk_profile_check(risk, &no_price, &violation) == DMA_RISK_PRICE);
    CHECK(violation.code == 100);

    c_dma_risk_profile_free(risk);
}

static void test_risk_check_limits(void) {
    dma_risk_profile* risk = c_dma_risk_profile_new(g_allocator);
    dma_risk_violation violation;

    set_full_ticker_rules(risk, "AAPL", 100., 100., 100., 100., 1000., 1000.);
    set_full_globals(risk, INFINITY, INFINITY, INFINITY, INFINITY);

    // Price validity.
    dma_risk_check_state check_state = make_query("AAPL", 10., 1, NAN);
    CHECK(c_dma_risk_profile_check(risk, &check_state, &violation) == DMA_RISK_PRICE);
    CHECK(violation.code == 100);

    // Trade limit.
    CHECK(c_dma_risk_profile_set_limit(risk, "AAPL", DMA_RULE_MAX_TRADE_LONG, 5.) == DMA_RISK_OK_RET);
    check_state = make_query("AAPL", 10., 1, 10.);
    CHECK(c_dma_risk_profile_check(risk, &check_state, &violation) == DMA_RISK_TRADE_LONG);
    CHECK(violation.code == 1001);
    CHECK(strstr(violation.msg, "lmt=5") != NULL);
    CHECK(c_dma_risk_profile_set_limit(risk, "AAPL", DMA_RULE_MAX_TRADE_LONG, 100.) == DMA_RISK_OK_RET);

    // Exposure limit: a 10-lot order against a 5-lot long exposure limit.
    CHECK(c_dma_risk_profile_set_limit(risk, "AAPL", DMA_RULE_MAX_EXPOSURE_LONG, 5.) == DMA_RISK_OK_RET);
    CHECK(c_dma_risk_profile_check(risk, &check_state, &violation) == DMA_RISK_EXPOSURE_LONG);
    CHECK(violation.code == 2001);
    CHECK(c_dma_risk_profile_set_limit(risk, "AAPL", DMA_RULE_MAX_EXPOSURE_LONG, 100.) == DMA_RISK_OK_RET);

    // Notional limit: a 10-lot order at 10 is worth 100.
    CHECK(c_dma_risk_profile_set_limit(risk, "AAPL", DMA_RULE_MAX_NOTIONAL_LONG, 50.) == DMA_RISK_OK_RET);
    CHECK(c_dma_risk_profile_check(risk, &check_state, &violation) == DMA_RISK_NOTIONAL_LONG);
    CHECK(violation.code == 4001);
    CHECK(c_dma_risk_profile_set_limit(risk, "AAPL", DMA_RULE_MAX_NOTIONAL_LONG, 1000.) == DMA_RISK_OK_RET);

    // Percentile limit: a 50% share of a 1000 notional portfolio is 1000 notional / 10 = 100 lots.
    CHECK(c_dma_risk_profile_set_limit(risk, "AAPL", DMA_RULE_MAX_TRADE_LONG, 1000.) == DMA_RISK_OK_RET);
    CHECK(c_dma_risk_profile_set_limit(risk, "AAPL", DMA_RULE_MAX_EXPOSURE_LONG, 1000.) == DMA_RISK_OK_RET);
    CHECK(c_dma_risk_profile_set_limit(risk, "AAPL", DMA_RULE_MAX_PERCENTILE, 0.5) == DMA_RISK_OK_RET);
    dma_risk_check_state big = make_query("AAPL", 200., 1, 10.);
    big.portfolio_exposure_notional_abs = 1000.;
    CHECK(c_dma_risk_profile_check(risk, &big, &violation) == DMA_RISK_PERCENTILE_LONG);
    CHECK(violation.code == 3001);
    CHECK(c_dma_risk_profile_set_limit(risk, "AAPL", DMA_RULE_MAX_PERCENTILE, INFINITY) == DMA_RISK_OK_RET);

    // Portfolio net limit.
    set_full_globals(risk, INFINITY, INFINITY, 100., 100.);
    dma_risk_check_state net = make_query("AAPL", 10., 1, 10.);
    net.portfolio_exposure_notional = 200.;
    CHECK(c_dma_risk_profile_check(risk, &net, &violation) == DMA_RISK_PORTFOLIO_NET_LONG);
    CHECK(violation.code == 5001);

    // Portfolio total limit.
    set_full_globals(risk, 100., 100., INFINITY, INFINITY);
    dma_risk_check_state ttl = make_query("AAPL", 10., 1, 10.);
    ttl.ttl_exposure_notional_long = 200.;
    CHECK(c_dma_risk_profile_check(risk, &ttl, &violation) == DMA_RISK_PORTFOLIO_TOTAL_LONG);
    CHECK(violation.code == 5003);

    // Short side: a 10-lot sell at 10 against a 5-lot short exposure limit.
    set_full_globals(risk, INFINITY, INFINITY, INFINITY, INFINITY);
    CHECK(c_dma_risk_profile_set_limit(risk, "AAPL", DMA_RULE_MAX_EXPOSURE_SHORT, 5.) == DMA_RISK_OK_RET);
    dma_risk_check_state short_query = make_query("AAPL", 10., -1, 10.);
    CHECK(c_dma_risk_profile_check(risk, &short_query, &violation) == DMA_RISK_EXPOSURE_SHORT);
    CHECK(violation.code == 2002);

    // Short-side trade limit.
    CHECK(c_dma_risk_profile_set_limit(risk, "AAPL", DMA_RULE_MAX_EXPOSURE_SHORT, 1000.) == DMA_RISK_OK_RET);
    CHECK(c_dma_risk_profile_set_limit(risk, "AAPL", DMA_RULE_MAX_TRADE_SHORT, 5.) == DMA_RISK_OK_RET);
    CHECK(c_dma_risk_profile_check(risk, &short_query, &violation) == DMA_RISK_TRADE_SHORT);
    CHECK(violation.code == 1002);
    CHECK(c_dma_risk_profile_set_limit(risk, "AAPL", DMA_RULE_MAX_TRADE_SHORT, 1000.) == DMA_RISK_OK_RET);

    // Short-side notional limit.
    CHECK(c_dma_risk_profile_set_limit(risk, "AAPL", DMA_RULE_MAX_NOTIONAL_SHORT, 50.) == DMA_RISK_OK_RET);
    CHECK(c_dma_risk_profile_check(risk, &short_query, &violation) == DMA_RISK_NOTIONAL_SHORT);
    CHECK(violation.code == 4002);
    CHECK(c_dma_risk_profile_set_limit(risk, "AAPL", DMA_RULE_MAX_NOTIONAL_SHORT, 1000.) == DMA_RISK_OK_RET);

    // Short-side percentile limit.
    CHECK(c_dma_risk_profile_set_limit(risk, "AAPL", DMA_RULE_MAX_PERCENTILE, 0.5) == DMA_RISK_OK_RET);
    dma_risk_check_state short_big = make_query("AAPL", 200., -1, 10.);
    short_big.portfolio_exposure_notional_abs = 1000.;
    CHECK(c_dma_risk_profile_check(risk, &short_big, &violation) == DMA_RISK_PERCENTILE_SHORT);
    CHECK(violation.code == 3002);
    CHECK(c_dma_risk_profile_set_limit(risk, "AAPL", DMA_RULE_MAX_PERCENTILE, INFINITY) == DMA_RISK_OK_RET);

    // Short-side portfolio net limit.
    set_full_globals(risk, INFINITY, INFINITY, 100., 100.);
    dma_risk_check_state net_short = make_query("AAPL", 10., -1, 10.);
    net_short.portfolio_exposure_notional = -200.;
    CHECK(c_dma_risk_profile_check(risk, &net_short, &violation) == DMA_RISK_PORTFOLIO_NET_SHORT);
    CHECK(violation.code == 5002);

    // Short-side portfolio total limit.
    set_full_globals(risk, 100., 100., INFINITY, INFINITY);
    dma_risk_check_state ttl_short = make_query("AAPL", 10., -1, 10.);
    ttl_short.ttl_exposure_notional_short = 200.;
    CHECK(c_dma_risk_profile_check(risk, &ttl_short, &violation) == DMA_RISK_PORTFOLIO_TOTAL_SHORT);
    CHECK(violation.code == 5004);

    // Neutral side skips the directional checks.
    dma_risk_check_state neutral = make_query("AAPL", 10., 0, 10.);
    CHECK(c_dma_risk_profile_check(risk, &neutral, &violation) == DMA_RISK_PASS);

    c_dma_risk_profile_free(risk);
}

static void test_risk_check_gates(void) {
    dma_risk_profile*          risk = c_dma_risk_profile_new(g_allocator);
    dma_risk_violation violation;
    dma_risk_check_state     check_state = make_query("AAPL", 10., 1, 10.);

    set_full_ticker_rules(risk, "AAPL", 100., 100., 100., 100., 1000., 1000.);
    set_full_globals(risk, INFINITY, INFINITY, INFINITY, INFINITY);

    // A complete profile passes before any gate is set.
    CHECK(c_dma_risk_profile_check(risk, &check_state, &violation) == DMA_RISK_PASS);

    // A non-empty whitelist refuses a ticker it does not list.
    CHECK(c_dma_risk_profile_whitelist_add(risk, "MSFT") == DMA_RISK_OK_RET);
    CHECK(c_dma_risk_profile_check(risk, &check_state, &violation) == DMA_RISK_TICKER_UNLISTED);
    CHECK(violation.type == DMA_RISK_TICKER_UNLISTED);
    CHECK(violation.code == 1005);
    CHECK(strcmp(c_dma_risk_type_name(violation.type), "RiskProfile.TickerLimit.Unlisted") == 0);
    CHECK(strstr(violation.msg, "not whitelisted") != NULL);

    // Listing the ticker restores the pass.
    CHECK(c_dma_risk_profile_whitelist_add(risk, "AAPL") == DMA_RISK_OK_RET);
    CHECK(c_dma_risk_profile_check(risk, &check_state, &violation) == DMA_RISK_PASS);

    // A ticker cannot be on both lists: blacklisting a whitelisted ticker is
    // refused with the conflict code.
    CHECK(c_dma_risk_profile_blacklist_add(risk, "AAPL") == DMA_RISK_ERR_CONFLICT);

    // Dropping it from the whitelist makes the blacklist add succeed; the
    // blacklisted ticker is refused.
    CHECK(c_dma_risk_profile_whitelist_remove(risk, "AAPL") == DMA_RISK_OK_RET);
    CHECK(c_dma_risk_profile_blacklist_add(risk, "AAPL") == DMA_RISK_OK_RET);
    CHECK(c_dma_risk_profile_check(risk, &check_state, &violation) == DMA_RISK_TICKER_BLOCKED);
    CHECK(violation.type == DMA_RISK_TICKER_BLOCKED);
    CHECK(violation.code == 1004);
    CHECK(strcmp(c_dma_risk_type_name(violation.type), "RiskProfile.TickerLimit.Blocked") == 0);

    // The gate runs before the incomplete test: an unconfigured ticker is
    // reported as blocked, not incomplete.
    dma_risk_check_state other = make_query("XYZ", 10., 1, 10.);
    CHECK(c_dma_risk_profile_blacklist_add(risk, "XYZ") == DMA_RISK_OK_RET);
    CHECK(c_dma_risk_profile_check(risk, &other, &violation) == DMA_RISK_TICKER_BLOCKED);

    // Invalid input reports itself instead of a pass.
    CHECK(c_dma_risk_profile_check(NULL, &check_state, &violation) == DMA_RISK_INVALID);
    CHECK(violation.type == DMA_RISK_INVALID);

    check_state.ticker = NULL;
    CHECK(c_dma_risk_profile_check(risk, &check_state, &violation) == DMA_RISK_INVALID);
    CHECK(violation.type == DMA_RISK_INVALID);

    c_dma_risk_profile_free(risk);
}

static dma_algo* test_algo_factory(dma* DMA, const char* ticker, md_side side, double target_volume, const void* params, allocator_protocol* allocator);

static void test_risk_access_checks(void) {
    mds* market_data = c_mds_new(g_allocator);
    CHECK(market_data != NULL);

    market_data->ts.unix_ts    = 1720000000.;
    market_data->ts.ts         = 43200.;
    market_data->ts.date.year  = 2024;
    market_data->ts.date.month = 2;
    market_data->ts.date.day   = 8;
    market_data->ts.date.stype = SESSION_TYPE_NORMINAL;
    market_data->ts.time = make_session_time(10, 0);

    md_variant quote;
    memset(&quote, 0, sizeof(quote));
    quote.meta_info.dtype         = DTYPE_TRANSACTION;
    quote.meta_info.ticker        = "AAPL";
    quote.meta_info.dt            = market_data->ts;
    quote.transaction_data.price  = 10.;
    quote.transaction_data.volume = 100.;
    quote.transaction_data.side   = SIDE_LONG_OPEN;

    CHECK(c_mds_update_subscription(market_data, &quote) == MDS_OK_SUBSCRIBED);
    CHECK(fabs(c_mds_get_market_price(market_data, "AAPL") - 10.) < 1e-9);

    dma* access = c_dma_new(market_data, g_allocator);
    CHECK(access != NULL);

    dma_algo_engine* engine = access->ALGO_ENGINE;
    CHECK(engine != NULL);

    dma_risk_profile* risk = access->RISK_PROFILE;
    set_full_ticker_rules(risk, "AAPL", 100., 100., 100., 100., 250., 250.);
    set_full_ticker_rules(risk, "NOQ", 100., 100., 100., 100., 250., 250.);
    set_full_globals(risk, INFINITY, INFINITY, INFINITY, INFINITY);


    dma_position* position = c_dma_position_new(access, engine, NULL, g_allocator);
    CHECK(position != NULL);

    dma_algo_ctx* registration = c_dma_algo_registry_reg(engine->registry, "risk_check", test_algo_factory, 0);
    CHECK(registration != NULL);

    dma_algo* algo = c_dma_new_algo(access, registration->reg_id, "AAPL", SIDE_LONG_OPEN, 10., NULL, g_allocator);
    CHECK(algo != NULL);
    CHECK(c_dma_position_add_algo(position, algo) == POSITION_OK);

    // The algo check reads the algo's remaining target at the market price.
    dma_risk_violation violation = c_dma_risk_profile_check_algo(access, risk, algo);
    CHECK(violation.type == DMA_RISK_PASS);

    // A free instruction is checked at its limit price, or at the market
    // price when it has none.
    md_variant order = make_order("AAPL", 10., SIDE_LONG_OPEN);
    CHECK(c_dma_risk_profile_check_order(access, risk, &order.trade_instruction).type == DMA_RISK_PASS);

    order.trade_instruction.limit_price = 20.;
    violation = c_dma_risk_profile_check_order(access, risk, &order.trade_instruction);
    CHECK(violation.type == DMA_RISK_PASS);

    // The engine order starts idle and carries no working volume.
    md_trade_instruction* tracked = c_dma_algo_engine_new_order(engine, algo, 10., 10.);
    CHECK(tracked != NULL);
    CHECK(tracked->order_state == STATE_IDLE);
    CHECK(((dma_order_ctx*) tracked)->flags == 0);

    double working_long  = 0.;
    double working_short = 0.;
    CHECK(c_dma_position_get_working(position, "AAPL", &working_long, &working_short) == POSITION_OK);
    CHECK(working_long == 0.);

    // The first working state opens the volume: the unchecked launch moves the
    // order to sent and the position's working cache takes the remaining.
    CHECK(c_dma_launch_order_unchecked(access, tracked) == DMA_OK);
    CHECK(tracked->order_state == STATE_SENT);
    CHECK(c_dma_position_get_working(position, "AAPL", &working_long, &working_short) == POSITION_OK);
    CHECK(working_long == 10.);

    // The working volume is part of the trade count: 10 working + 95 > 100.
    violation = c_dma_risk_profile_check_volume(access, risk, "AAPL", SIDE_LONG_OPEN, 95.);
    CHECK(violation.type == DMA_RISK_TRADE_LONG);
    CHECK(violation.code == 1001);

    // The short side carries its own limit.
    violation = c_dma_risk_profile_check_volume(access, risk, "AAPL", SIDE_SHORT_OPEN, 105.);
    CHECK(violation.type == DMA_RISK_TRADE_SHORT);
    CHECK(violation.code == 1002);

    // The explicit price drives the notional check: 10 lots at 30 is 300
    // against a 250 limit.
    violation = c_dma_risk_profile_check_limit(access, risk, "AAPL", SIDE_LONG_OPEN, 10., 30.);
    CHECK(violation.type == DMA_RISK_NOTIONAL_LONG);
    CHECK(violation.code == 4001);

    // At the market price the same volume fits.
    violation = c_dma_risk_profile_check_volume(access, risk, "AAPL", SIDE_LONG_OPEN, 10.);
    CHECK(violation.type == DMA_RISK_PASS);

    // A configured ticker without a quote is refused on price validity.
    violation = c_dma_risk_profile_check_volume(access, risk, "NOQ", SIDE_LONG_OPEN, 1.);
    CHECK(violation.type == DMA_RISK_PRICE);
    CHECK(violation.code == 100);

    // A fill moves the volume from working to traded: 10 traded + 95 > 100.
    md_trade_report fill;
    memset(&fill, 0, sizeof(fill));
    fill.meta_info.dt = market_data->ts;
    fill.multiplier   = 1.;
    fill.order_id     = tracked->order_id;
    fill.volume       = 10.;
    fill.notional     = 100.;
    fill.side         = SIDE_LONG_OPEN;
    strcpy(fill.trade_id.data, "risk-1");

    CHECK(c_dma_algo_engine_on_trade(engine, &fill) == DMA_ALGO_OK);
    CHECK(c_dma_position_get_working(position, "AAPL", &working_long, &working_short) == POSITION_OK);
    CHECK(working_long == 0.);

    violation = c_dma_risk_profile_check_volume(access, risk, "AAPL", SIDE_LONG_OPEN, 95.);
    CHECK(violation.type == DMA_RISK_TRADE_LONG);

    violation = c_dma_risk_profile_check_volume(access, risk, "AAPL", SIDE_LONG_OPEN, 10.);
    CHECK(violation.type == DMA_RISK_PASS);

    // The gates refuse both entry points.
    CHECK(c_dma_risk_profile_blacklist_add(risk, "AAPL") == DMA_RISK_OK_RET);
    CHECK(c_dma_risk_profile_check_algo(access, risk, algo).type == DMA_RISK_TICKER_BLOCKED);
    CHECK(c_dma_risk_profile_check_order(access, risk, &order.trade_instruction).type == DMA_RISK_TICKER_BLOCKED);
    CHECK(c_dma_risk_profile_blacklist_remove(risk, "AAPL") == DMA_RISK_OK_RET);

    // Invalid input.
    CHECK(c_dma_risk_profile_check_order(NULL, risk, &order.trade_instruction).type == DMA_RISK_INVALID);
    CHECK(c_dma_risk_profile_check_order(access, NULL, &order.trade_instruction).type == DMA_RISK_INVALID);
    CHECK(c_dma_risk_profile_check_order(access, risk, NULL).type == DMA_RISK_INVALID);
    CHECK(c_dma_risk_profile_check_algo(access, risk, NULL).type == DMA_RISK_INVALID);

    c_dma_free(access);
    c_mds_free(market_data);
}

static void test_risk_custom_rules(void) {
    mds* market_data = c_mds_new(g_allocator);
    CHECK(market_data != NULL);

    market_data->ts.unix_ts     = 1720000000.;
    market_data->ts.date.year   = 2024;
    market_data->ts.date.month  = 2;
    market_data->ts.date.day    = 8;
    market_data->ts.time = make_session_time(10, 0);

    dma* access = c_dma_new(market_data, g_allocator);
    CHECK(access != NULL);

    dma_algo_engine* engine = access->ALGO_ENGINE;
    CHECK(engine != NULL);

    dma_risk_profile* risk = access->RISK_PROFILE;

    // A trade needs its ticker's market price.
    md_variant quote;
    memset(&quote, 0, sizeof(quote));
    quote.meta_info.dtype         = DTYPE_TRANSACTION;
    quote.meta_info.dt            = market_data->ts;
    quote.transaction_data.price  = 10.;
    quote.transaction_data.volume = 100.;
    quote.transaction_data.side   = SIDE_LONG_OPEN;

    quote.meta_info.ticker = "AAPL";
    CHECK(c_mds_update_subscription(market_data, &quote) == MDS_OK_SUBSCRIBED);

    quote.meta_info.ticker = "MSFT";
    CHECK(c_mds_update_subscription(market_data, &quote) == MDS_OK_SUBSCRIBED);

    session_time_t nine_forty_five = make_session_time(9, 45);

    refusal_window* ban = c_refusal_window_new(NULL, &nine_forty_five, g_allocator);
    CHECK(ban != NULL);

    // The profile composes and owns the rule node; the node adopts the state.
    dma_risk_rule_ctx* node = c_refusal_window_register(risk, ban, "AAPL");
    CHECK(node != NULL);
    CHECK(node->rule_id == (uintptr_t) node);
    CHECK(node->risk_rule_code == REFUSAL_WINDOW_RULE_CODE);
    CHECK(node->check_fn == c_refusal_window_check);
    CHECK(node->user_data == ban);
    CHECK(node->name != NULL);
    CHECK(strcmp(node->name, REFUSAL_WINDOW_RULE_NAME) == 0);
    CHECK(node->ticker != NULL);
    CHECK(strcmp(node->ticker, "AAPL") == 0);
    CHECK(node->enabled);

    // The state is a child of the node, so the node's teardown frees it too.
    CHECK(c_ap_protocol_from_ptr(ban)->parent == c_ap_protocol_from_ptr(node));

    // A custom rule never completes a profile: with no absolute risk value
    // defined, both sides stay incomplete.
    dma_risk_limits* limits = c_dma_risk_profile_get_limits(risk, "AAPL");
    CHECK(limits != NULL);
    CHECK(limits->rules == node);
    CHECK(limits->enabled == 0);
    CHECK(limits->max_percentile == 0.);
    CHECK(!c_dma_risk_profile_get_limit(risk, "AAPL", DMA_RULE_MAX_TRADE_LONG, NULL));
    CHECK(c_dma_risk_profile_check_volume(access, risk, "AAPL", SIDE_LONG_OPEN, 10.).type == DMA_RISK_INCOMPLETE);
    CHECK(c_dma_risk_profile_check_volume(access, risk, "AAPL", SIDE_SHORT_OPEN, 10.).type == DMA_RISK_INCOMPLETE);

    // An absolute rule per side constrains it; at 10:00 the ban is over.
    CHECK(c_dma_risk_profile_set_limit(risk, "AAPL", DMA_RULE_MAX_TRADE_LONG, 1e9) == DMA_RISK_OK_RET);
    CHECK(c_dma_risk_profile_set_limit(risk, "AAPL", DMA_RULE_MAX_TRADE_SHORT, 1e9) == DMA_RISK_OK_RET);

    CHECK(c_dma_risk_profile_check_volume(access, risk, "AAPL", SIDE_LONG_OPEN, 10.).type == DMA_RISK_PASS);
    CHECK(c_dma_risk_profile_check_volume(access, risk, "AAPL", SIDE_SHORT_OPEN, 10.).type == DMA_RISK_PASS);

    // Before 09:45 the rule rejects through both entries.
    market_data->ts.time = make_session_time(9, 30);

    dma_risk_violation violation = c_dma_risk_profile_check_volume(access, risk, "AAPL", SIDE_LONG_OPEN, 10.);
    CHECK(violation.type == DMA_RISK_CUSTOM);
    CHECK(violation.code == REFUSAL_WINDOW_RULE_CODE);
    CHECK(strcmp(c_dma_risk_type_name(violation.type), "RiskProfile.Custom") == 0);
    CHECK(strstr(violation.msg, REFUSAL_WINDOW_RULE_NAME) != NULL);
    CHECK(strstr(violation.msg, "09:45:00") != NULL);

    md_variant order = make_order("AAPL", 10., SIDE_LONG_OPEN);
    CHECK(c_dma_risk_profile_check_order(access, risk, &order.trade_instruction).type == DMA_RISK_CUSTOM);

    // A disabled rule stays linked but the check skips it.
    CHECK(c_dma_risk_profile_disable_rule(risk, "AAPL", node->rule_id) == DMA_RISK_OK_RET);
    CHECK(!node->enabled);
    CHECK(c_dma_risk_profile_check_volume(access, risk, "AAPL", SIDE_LONG_OPEN, 10.).type == DMA_RISK_PASS);
    CHECK(c_dma_risk_profile_disable_rule(risk, "AAPL", 1) == DMA_RISK_ERR_NOT_FOUND);

    CHECK(c_dma_risk_profile_enable_rule(risk, "AAPL", node->rule_id) == DMA_RISK_OK_RET);
    CHECK(node->enabled);
    CHECK(c_dma_risk_profile_check_volume(access, risk, "AAPL", SIDE_LONG_OPEN, 10.).type == DMA_RISK_CUSTOM);

    // Another ticker is not hit by the AAPL rule; it has no rule at all.
    CHECK(c_dma_risk_profile_check_volume(access, risk, "MSFT", SIDE_LONG_OPEN, 10.).type == DMA_RISK_INCOMPLETE);

    // A second rule appends to the chain as its own node.
    session_time_t ten_fifteen = make_session_time(10, 15);

    refusal_window* later = c_refusal_window_new(NULL, &ten_fifteen, g_allocator);
    CHECK(later != NULL);

    dma_risk_rule_ctx* later_node = c_refusal_window_register(risk, later, "AAPL");
    CHECK(later_node != NULL);
    CHECK(later_node != node);
    CHECK(node->next == later_node);

    market_data->ts.time = make_session_time(10, 0);
    CHECK(c_dma_risk_profile_check_volume(access, risk, "AAPL", SIDE_LONG_OPEN, 10.).type == DMA_RISK_CUSTOM);

    market_data->ts.time = make_session_time(10, 20);
    CHECK(c_dma_risk_profile_check_volume(access, risk, "AAPL", SIDE_LONG_OPEN, 10.).type == DMA_RISK_PASS);

    // Unregistering the later rule frees its node and, with it, the state;
    // the first rule is left alone.
    CHECK(c_dma_risk_profile_unregister_rule(risk, later_node->ticker, later_node->rule_id) == DMA_RISK_OK_RET);
    CHECK(node->next == NULL);
    CHECK(c_dma_risk_profile_check_volume(access, risk, "AAPL", SIDE_LONG_OPEN, 10.).type == DMA_RISK_PASS);

    // A global rule applies to every ticker; a ticker needs an absolute rule
    // to reach the custom checks.
    market_data->ts.time = make_session_time(10, 0);

    refusal_window* global_ban = c_refusal_window_new(NULL, &ten_fifteen, g_allocator);
    CHECK(global_ban != NULL);

    dma_risk_rule_ctx* global_node = c_refusal_window_register_global(risk, global_ban);
    CHECK(global_node != NULL);
    CHECK(risk->globals.rules == global_node);
    CHECK(global_node->ticker == NULL);
    CHECK(global_node->enabled);
    CHECK(c_ap_protocol_from_ptr(global_ban)->parent == c_ap_protocol_from_ptr(global_node));

    CHECK(c_dma_risk_profile_set_limit(risk, "MSFT", DMA_RULE_MAX_TRADE_LONG, 1e9) == DMA_RISK_OK_RET);
    CHECK(c_dma_risk_profile_check_volume(access, risk, "MSFT", SIDE_LONG_OPEN, 10.).type == DMA_RISK_CUSTOM);
    CHECK(c_dma_risk_profile_check_volume(access, risk, "AAPL", SIDE_LONG_OPEN, 10.).type == DMA_RISK_CUSTOM);

    // A disabled global rule is skipped.
    CHECK(c_dma_risk_profile_disable_global_rule(risk, global_node->rule_id) == DMA_RISK_OK_RET);
    CHECK(c_dma_risk_profile_check_volume(access, risk, "MSFT", SIDE_LONG_OPEN, 10.).type == DMA_RISK_PASS);
    CHECK(c_dma_risk_profile_enable_global_rule(risk, global_node->rule_id) == DMA_RISK_OK_RET);
    CHECK(c_dma_risk_profile_check_volume(access, risk, "MSFT", SIDE_LONG_OPEN, 10.).type == DMA_RISK_CUSTOM);

    // Unregistering the global rule frees it and its adopted state.
    CHECK(c_dma_risk_profile_unregister_global_rule(risk, global_node->rule_id) == DMA_RISK_OK_RET);
    CHECK(risk->globals.rules == NULL);
    CHECK(c_dma_risk_profile_check_volume(access, risk, "MSFT", SIDE_LONG_OPEN, 10.).type == DMA_RISK_PASS);
    CHECK(c_dma_risk_profile_check_volume(access, risk, "AAPL", SIDE_LONG_OPEN, 10.).type == DMA_RISK_PASS);

    // A start-only window refuses from its start onward.
    refusal_window* cutoff = c_refusal_window_new(&ten_fifteen, NULL, g_allocator);
    CHECK(cutoff != NULL);

    dma_risk_rule_ctx* cutoff_node = c_refusal_window_register(risk, cutoff, "AAPL");
    CHECK(cutoff_node != NULL);

    // The clock is 10:00: before the start, trading passes.
    CHECK(c_dma_risk_profile_check_volume(access, risk, "AAPL", SIDE_LONG_OPEN, 10.).type == DMA_RISK_PASS);

    market_data->ts.time = make_session_time(10, 20);
    CHECK(c_dma_risk_profile_check_volume(access, risk, "AAPL", SIDE_LONG_OPEN, 10.).type == DMA_RISK_CUSTOM);

    CHECK(c_dma_risk_profile_unregister_rule(risk, cutoff_node->ticker, cutoff_node->rule_id) == DMA_RISK_OK_RET);

    // A registered rule without a session clock fails closed.
    session_time_t ten_thirty = make_session_time(10, 30);

    refusal_window* clockless_ban = c_refusal_window_new(NULL, &ten_thirty, g_allocator);
    CHECK(clockless_ban != NULL);

    dma_risk_rule_ctx* clockless_node = c_refusal_window_register(risk, clockless_ban, "AAPL");
    CHECK(clockless_node != NULL);

    dma* clockless = c_dma_new(NULL, g_allocator);
    CHECK(clockless != NULL);

    md_variant clockless_order = make_order("AAPL", 10., SIDE_LONG_OPEN);
    clockless_order.trade_instruction.limit_price = 10.;

    violation = c_dma_risk_profile_check_order(clockless, risk, &clockless_order.trade_instruction);
    CHECK(violation.type == DMA_RISK_CUSTOM);
    CHECK(strstr(violation.msg, "session clock") != NULL);

    CHECK(c_dma_risk_profile_unregister_rule(risk, clockless_node->ticker, clockless_node->rule_id) == DMA_RISK_OK_RET);

    // Unregistering the last custom rule leaves the absolute rules in place:
    // the side stays constrained and the check passes.
    CHECK(c_dma_risk_profile_unregister_rule(risk, node->ticker, node->rule_id) == DMA_RISK_OK_RET);
    CHECK(c_dma_risk_profile_check_volume(access, risk, "AAPL", SIDE_LONG_OPEN, 10.).type == DMA_RISK_PASS);

    // The profile's teardown frees a rule that is still registered.
    refusal_window* leftover = c_refusal_window_new(NULL, &ten_thirty, g_allocator);
    CHECK(leftover != NULL);
    CHECK(c_refusal_window_register(risk, leftover, "AAPL") != NULL);

    c_dma_free(clockless);
    c_dma_free(access);
    c_mds_free(market_data);
}

// ========== Direct Market Access ==========

static void test_dma_lifecycle_and_gate(void) {
    dma* access = c_dma_new(NULL, g_allocator);
    CHECK(access != NULL);

    dma_algo_engine* engine = access->ALGO_ENGINE;
    CHECK(engine != NULL);
    CHECK(access->positions != NULL);
    CHECK(c_bytemap_ex_len(access->positions) == 0);

    dma_risk_profile* risk = access->RISK_PROFILE;
    set_full_ticker_rules(risk, "AAPL", 100., 100., 100., 100., 1000., 1000.);
    set_full_globals(risk, INFINITY, INFINITY, INFINITY, INFINITY);

    test_interface interface;
    memset(&interface, 0, sizeof(test_interface));

    CHECK(c_dma_attach_protocol_handler(access, test_on_launch, test_on_cancel, NULL, &interface) == DMA_OK);
    CHECK(c_dma_attach_protocol_callback(access, test_on_reject, test_on_filled) == DMA_OK);
    CHECK(c_dma_attach_protocol_handler(access, NULL, test_on_cancel, NULL, &interface) == DMA_ERR_INVALID_INPUT);
    CHECK(c_dma_attach_protocol_handler(access, test_on_launch, NULL, NULL, &interface) == DMA_ERR_INVALID_INPUT);
    CHECK(c_dma_attach_protocol_handler(access, test_on_launch, test_on_cancel, NULL, &interface) == DMA_OK);

    dma_risk_violation violation;
    md_variant         order = make_order("AAPL", 10., SIDE_LONG_OPEN);

    order.trade_instruction.limit_price = 10.;

    // The dry run passes and dispatches nothing.
    CHECK(c_dma_check_order(access, &order.trade_instruction, &violation) == DMA_OK);
    CHECK(interface.n_launch == 0 && interface.n_cancel == 0 && interface.n_reject == 0);

    // A caller-initialized access owns its blocks; its fresh profile refuses
    // the order as incomplete.
    dma init_only;
    memset(&init_only, 0, sizeof(init_only));
    CHECK(c_dma_init(&init_only, NULL, g_allocator) == DMA_OK);
    CHECK(c_dma_check_order(&init_only, &order.trade_instruction, &violation) == DMA_ERR_RISK);
    CHECK(violation.type == DMA_RISK_INCOMPLETE);
    c_dma_dealloc(&init_only);

    // A fresh profile (c_dma_new) constrains nothing and refuses the order.
    dma* incomplete = c_dma_new(NULL, g_allocator);
    CHECK(c_dma_check_order(incomplete, &order.trade_instruction, &violation) == DMA_ERR_RISK);
    CHECK(violation.type == DMA_RISK_INCOMPLETE);
    CHECK(strstr(violation.msg, "no rule constrains") != NULL);
    c_dma_free(incomplete);

    // The fill hook is an event notification, not a dispatch entry.
    md_trade_report report;
    memset(&report, 0, sizeof(md_trade_report));

    if (access->protocol.callbacks.order_filled_fn) access->protocol.callbacks.order_filled_fn(&order.trade_instruction, &report, access->protocol.user_data);
    CHECK(interface.n_filled == 1);
    CHECK(interface.last_order == &order.trade_instruction);
    CHECK(interface.last_report == &report);

    // Detaching makes the access inert; a second detach is a no-op.
    CHECK(c_dma_detach_protocol_handler(access) == DMA_OK);

    md_variant detached = make_order("AAPL", 10., SIDE_LONG_OPEN);
    CHECK(c_dma_launch_order_unchecked(access, &detached.trade_instruction) == DMA_OK);
    CHECK(detached.trade_instruction.order_state == STATE_SENT);
    CHECK(interface.n_launch == 0);

    CHECK(c_dma_detach_protocol_handler(access) == DMA_OK);

    c_dma_free(access);
}

static void test_dma_order_state_contract(void) {
    dma* access = c_dma_new(NULL, g_allocator);
    CHECK(access != NULL);

    dma_algo_engine* engine = access->ALGO_ENGINE;
    CHECK(engine != NULL);

    dma_risk_profile* risk = access->RISK_PROFILE;
    set_full_ticker_rules(risk, "AAPL", 100., 100., 100., 100., 1000., 1000.);
    set_full_globals(risk, INFINITY, INFINITY, INFINITY, INFINITY);

    test_interface interface;
    memset(&interface, 0, sizeof(test_interface));
    CHECK(c_dma_attach_protocol_handler(access, test_on_launch, test_on_cancel, NULL, &interface) == DMA_OK);
    CHECK(c_dma_attach_protocol_callback(access, test_on_reject, test_on_filled) == DMA_OK);

    md_variant order = make_order("AAPL", 10., SIDE_LONG_OPEN);
    order.trade_instruction.limit_price = 10.;

    // The dry run leaves the explicit IDLE state alone.
    CHECK(order.trade_instruction.order_state == STATE_IDLE);
    CHECK(c_dma_check_order(access, &order.trade_instruction, NULL) == DMA_OK);
    CHECK(order.trade_instruction.order_state == STATE_IDLE);

#if DMA_NO_QUEUE_MODE
    // Direct launch: IDLE -> SENT; a relaunch is banned.
    CHECK(c_dma_launch_order(access, &order.trade_instruction) == DMA_OK);
    CHECK(order.trade_instruction.order_state == STATE_SENT);
    CHECK(interface.n_launch == 1);
    CHECK(interface.last_seq_id == 1);

    CHECK(c_dma_launch_order(access, &order.trade_instruction) == DMA_ERR_INVALID_INPUT);
    CHECK(order.trade_instruction.order_state == STATE_SENT);
    CHECK(interface.n_launch == 1);

    // Direct cancel: SENT -> CANCELING; a repeated cancel is tolerated.
    CHECK(c_dma_cancel_order(access, &order.trade_instruction) == DMA_OK);
    CHECK(order.trade_instruction.order_state == STATE_CANCELING);
    CHECK(interface.n_cancel == 1);
    CHECK(interface.last_seq_id == 2);

    CHECK(c_dma_cancel_order(access, &order.trade_instruction) == DMA_OK_NOOP);
    CHECK(order.trade_instruction.order_state == STATE_CANCELING);
    CHECK(interface.n_cancel == 1);

    // PENDING is not a direct-launch pre-state.
    md_variant direct_pending = make_order("AAPL", 10., SIDE_LONG_OPEN);
    direct_pending.trade_instruction.order_state = STATE_PENDING;
    CHECK(c_dma_launch_order(access, &direct_pending.trade_instruction) == DMA_ERR_INVALID_INPUT);
    CHECK(direct_pending.trade_instruction.order_state == STATE_PENDING);
#else
    // Queued launch: IDLE -> PENDING until process, then SENT.
    CHECK(c_dma_launch_order(access, &order.trade_instruction) == DMA_OK_BUFFERED);
    CHECK(order.trade_instruction.order_state == STATE_PENDING);
    CHECK(interface.n_launch == 0);

    // A repeat on PENDING is tolerated in queue mode.
    CHECK(c_dma_launch_order(access, &order.trade_instruction) == DMA_OK_NOOP);
    CHECK(order.trade_instruction.order_state == STATE_PENDING);
    CHECK(interface.n_launch == 0);

    CHECK(c_dma_process_queue(access) == 1);
    CHECK(order.trade_instruction.order_state == STATE_SENT);
    CHECK(interface.n_launch == 1);
    CHECK(interface.last_seq_id == 1);

    CHECK(c_dma_launch_order(access, &order.trade_instruction) == DMA_ERR_INVALID_INPUT);
    CHECK(order.trade_instruction.order_state == STATE_SENT);

    // Queued cancel: CANCELING at enqueue; process sends it.
    CHECK(c_dma_cancel_order(access, &order.trade_instruction) == DMA_OK_BUFFERED);
    CHECK(order.trade_instruction.order_state == STATE_CANCELING);
    CHECK(interface.n_cancel == 0);
    CHECK(c_dma_process_queue(access) == 1);
    CHECK(order.trade_instruction.order_state == STATE_CANCELING);
    CHECK(interface.n_cancel == 1);
    CHECK(interface.last_seq_id == 2);

    CHECK(c_dma_cancel_order(access, &order.trade_instruction) == DMA_OK_NOOP);
    CHECK(interface.n_cancel == 1);
#endif

    // Every cancellable state moves to CANCELING; terminal states refuse.
    md_order_state cancellable_states[] = {STATE_PLACED, STATE_PARTFILLED};

    for (size_t index = 0; index < sizeof(cancellable_states) / sizeof(cancellable_states[0]); index++) {
        md_variant live = make_order("AAPL", 10., SIDE_LONG_OPEN);
        live.trade_instruction.order_state = cancellable_states[index];

#if DMA_NO_QUEUE_MODE
        CHECK(c_dma_cancel_order(access, &live.trade_instruction) == DMA_OK);
        CHECK(live.trade_instruction.order_state == STATE_CANCELING);
#else
        CHECK(c_dma_cancel_order(access, &live.trade_instruction) == DMA_OK_BUFFERED);
        CHECK(live.trade_instruction.order_state == STATE_CANCELING);
        CHECK(c_dma_process_queue(access) == 1);
        CHECK(live.trade_instruction.order_state == STATE_CANCELING);
#endif
    }

    md_variant filled = make_order("AAPL", 10., SIDE_LONG_OPEN);
    filled.trade_instruction.order_state = STATE_FILLED;
    CHECK(c_dma_cancel_order(access, &filled.trade_instruction) == DMA_ERR_INVALID_INPUT);
    CHECK(filled.trade_instruction.order_state == STATE_FILLED);

    md_variant rejected = make_order("AAPL", 10., SIDE_LONG_OPEN);
    rejected.trade_instruction.order_state = STATE_REJECTED;
    CHECK(c_dma_launch_order(access, &rejected.trade_instruction) == DMA_ERR_INVALID_INPUT);
    CHECK(rejected.trade_instruction.order_state == STATE_REJECTED);

    // A refused launch: PENDING -> REJECTED through the reject handler.
    md_variant too_large = make_order("AAPL", 5000., SIDE_LONG_OPEN);
    too_large.trade_instruction.limit_price = 10.;
    CHECK(too_large.trade_instruction.order_state == STATE_IDLE);

#if DMA_NO_QUEUE_MODE
    CHECK(c_dma_launch_order(access, &too_large.trade_instruction) == DMA_ERR_RISK);
    CHECK(too_large.trade_instruction.order_state == STATE_REJECTED);
    CHECK(interface.n_reject == 1);
#else
    CHECK(c_dma_launch_order(access, &too_large.trade_instruction) == DMA_OK_BUFFERED);
    CHECK(too_large.trade_instruction.order_state == STATE_PENDING);
    CHECK(c_dma_process_queue(access) == 1);
    CHECK(too_large.trade_instruction.order_state == STATE_REJECTED);
    CHECK(interface.n_reject == 1);
#endif

    // Unchecked accepts only PENDING and moves to SENT.
    md_variant unchecked = make_order("AAPL", 5000., SIDE_LONG_OPEN);
    CHECK(c_dma_launch_order_unchecked(access, &unchecked.trade_instruction) == DMA_OK);
    CHECK(unchecked.trade_instruction.order_state == STATE_SENT);
    CHECK(interface.n_launch == 2);

    CHECK(c_dma_launch_order_unchecked(access, &unchecked.trade_instruction) == DMA_ERR_INVALID_INPUT);
    CHECK(unchecked.trade_instruction.order_state == STATE_SENT);
    CHECK(interface.n_launch == 2);

    // IDLE and PENDING have not left the access: the cancel is local.
    md_variant idle_cancel = make_order("AAPL", 10., SIDE_LONG_OPEN);
    CHECK(c_dma_cancel_order(access, &idle_cancel.trade_instruction) == DMA_OK);
    CHECK(idle_cancel.trade_instruction.order_state == STATE_CANCELED);

    md_variant pending_cancel = make_order("AAPL", 10., SIDE_LONG_OPEN);
    pending_cancel.trade_instruction.order_state = STATE_PENDING;
    CHECK(c_dma_cancel_order(access, &pending_cancel.trade_instruction) == DMA_OK);
    CHECK(pending_cancel.trade_instruction.order_state == STATE_CANCELED);

    // UNKNOWN is not a valid launch or cancel pre-state.
    md_variant unknown = make_order("AAPL", 10., SIDE_LONG_OPEN);
    unknown.trade_instruction.order_state = STATE_UNKNOWN;
    CHECK(c_dma_launch_order(access, &unknown.trade_instruction) == DMA_ERR_INVALID_INPUT);
    CHECK(unknown.trade_instruction.order_state == STATE_UNKNOWN);
    CHECK(c_dma_cancel_order(access, &unknown.trade_instruction) == DMA_ERR_INVALID_INPUT);
    CHECK(unknown.trade_instruction.order_state == STATE_UNKNOWN);

#if !DMA_NO_QUEUE_MODE
    // The queue drop paths: a canceled/placed launch entry and a resolved
    // cancel entry are tolerated; an IDLE launch entry is banned.
    md_variant tombstone = make_order("AAPL", 10., SIDE_LONG_OPEN);
    tombstone.trade_instruction.order_state = STATE_CANCELED;
    md_variant placed_entry = make_order("AAPL", 10., SIDE_LONG_OPEN);
    placed_entry.trade_instruction.order_state = STATE_PLACED;

    CHECK(c_dma_order_queue_push(&access->requests, DMA_ACTION_LAUNCH, &tombstone.trade_instruction) == DMA_OK);
    CHECK(c_dma_order_queue_push(&access->requests, DMA_ACTION_LAUNCH, &placed_entry.trade_instruction) == DMA_OK);
    CHECK(c_dma_process_queue(access) == 2);
    CHECK(tombstone.trade_instruction.order_state == STATE_CANCELED);
    CHECK(placed_entry.trade_instruction.order_state == STATE_PLACED);

    md_variant resolved_cancel = make_order("AAPL", 10., SIDE_LONG_OPEN);
    resolved_cancel.trade_instruction.order_state = STATE_FILLED;
    CHECK(c_dma_order_queue_push(&access->requests, DMA_ACTION_CANCEL, &resolved_cancel.trade_instruction) == DMA_OK);
    CHECK(c_dma_process_queue(access) == 1);
    CHECK(resolved_cancel.trade_instruction.order_state == STATE_FILLED);

    md_variant idle_entry = make_order("AAPL", 10., SIDE_LONG_OPEN);
    CHECK(c_dma_order_queue_push(&access->requests, DMA_ACTION_LAUNCH, &idle_entry.trade_instruction) == DMA_OK);
    CHECK(c_dma_process_queue(access) == DMA_ERR_INVALID_INPUT);
    CHECK(idle_entry.trade_instruction.order_state == STATE_IDLE);
#endif

    c_dma_free(access);
}

static void test_dma_request_queue(void) {
    dma* access = c_dma_new(NULL, g_allocator);
    CHECK(access != NULL);

    dma_algo_engine* engine = access->ALGO_ENGINE;
    CHECK(engine != NULL);

    md_variant order_a = make_order("AAPL", 1., SIDE_LONG_OPEN);
    md_variant order_b = make_order("AAPL", 2., SIDE_SHORT_OPEN);
    md_variant order_c = make_order("AAPL", 3., SIDE_LONG_OPEN);

    CHECK(c_dma_order_queue_push(&access->requests, DMA_ACTION_LAUNCH, &order_a.trade_instruction) == DMA_OK);
    CHECK(c_dma_order_queue_push(&access->requests, DMA_ACTION_CANCEL, &order_b.trade_instruction) == DMA_OK);
    CHECK(c_dma_order_queue_push(&access->requests, DMA_ACTION_LAUNCH, &order_c.trade_instruction) == DMA_OK);
    CHECK(access->requests.size == 3);

    dma_order_action      action = DMA_ACTION_LAUNCH;
    md_trade_instruction* order  = NULL;

    CHECK(c_dma_order_queue_pop(&access->requests, &action, &order) == DMA_OK);
    CHECK(action == DMA_ACTION_LAUNCH);
    CHECK(order == &order_a.trade_instruction);
    CHECK(access->requests.size == 2);

    CHECK(c_dma_order_queue_pop(&access->requests, &action, &order) == DMA_OK);
    CHECK(action == DMA_ACTION_CANCEL);
    CHECK(order == &order_b.trade_instruction);

    CHECK(c_dma_order_queue_pop(&access->requests, &action, &order) == DMA_OK);
    CHECK(order == &order_c.trade_instruction);

    CHECK(c_dma_order_queue_pop(&access->requests, &action, &order) == DMA_ERR_EMPTY);
    CHECK(access->requests.size == 0);

    // clear_requests drops queued entries; the ring buffer stays allocated.
    CHECK(c_dma_order_queue_push(&access->requests, DMA_ACTION_LAUNCH, &order_a.trade_instruction) == DMA_OK);
    c_dma_order_queue_clear(&access->requests);
    CHECK(access->requests.size == 0);
    CHECK(access->requests.head == 0);

    // The ring doubles its contiguous buffer when full and keeps FIFO order.
    size_t capacity = access->requests.capacity;

    for (size_t index = 0; index < capacity + 1; index++) {
        dma_order_action queued = index % 2 ? DMA_ACTION_CANCEL : DMA_ACTION_LAUNCH;
        CHECK(c_dma_order_queue_push(&access->requests, queued, &order_a.trade_instruction) == DMA_OK);
    }

    CHECK(access->requests.size == capacity + 1);
    CHECK(access->requests.capacity > capacity);

    for (size_t index = 0; index < capacity + 1; index++) {
        dma_order_action expected = index % 2 ? DMA_ACTION_CANCEL : DMA_ACTION_LAUNCH;
        CHECK(c_dma_order_queue_pop(&access->requests, &action, &order) == DMA_OK);
        CHECK(action == expected);
        CHECK(order == &order_a.trade_instruction);
    }

    CHECK(c_dma_order_queue_pop(&access->requests, &action, &order) == DMA_ERR_EMPTY);

    // process_queue drains FIFO and applies the launch check; without a risk
    // profile the launch reaches the reject handler.
    test_interface interface;
    memset(&interface, 0, sizeof(test_interface));
    CHECK(c_dma_attach_protocol_handler(access, test_on_launch, test_on_cancel, NULL, &interface) == DMA_OK);
    CHECK(c_dma_attach_protocol_callback(access, test_on_reject, test_on_filled) == DMA_OK);

    order_a.trade_instruction.order_state = STATE_CANCELING;
    order_b.trade_instruction.order_state = STATE_PENDING;
    CHECK(c_dma_order_queue_push(&access->requests, DMA_ACTION_CANCEL, &order_a.trade_instruction) == DMA_OK);
    CHECK(c_dma_order_queue_push(&access->requests, DMA_ACTION_LAUNCH, &order_b.trade_instruction) == DMA_OK);

    CHECK(c_dma_process_queue(access) == 2);
    CHECK(interface.n_cancel == 1);
    CHECK(interface.n_reject == 1);
    CHECK(interface.n_launch == 0);
    CHECK(order_a.trade_instruction.order_state == STATE_CANCELING);
    CHECK(order_b.trade_instruction.order_state == STATE_REJECTED);
    CHECK(c_dma_process_queue(access) == 0);

    CHECK(c_dma_order_queue_push(&access->requests, DMA_ACTION_LAUNCH, &order_a.trade_instruction) == DMA_OK);
    CHECK(c_dma_inventory_add_inv_long(access->INVENTORY, "AAPL", 1., 1., 1., 0., DMA_SECURITY_STOCK) == INVENTORY_OK);

    double long_available = 0.;
    double short_available = 0.;
    CHECK(c_dma_inventory_available(access->INVENTORY, "AAPL", &long_available, &short_available) == INVENTORY_OK);
    CHECK(long_available == 1.);
    CHECK(short_available == 0.);
    c_dma_free(access);
}

static void test_dma_order_handler_self_detach(void) {
    dma* access = c_dma_new(NULL, g_allocator);
    CHECK(access != NULL);

    dma_algo_engine* engine = access->ALGO_ENGINE;
    CHECK(engine != NULL);

    self_detach_interface interface;
    memset(&interface, 0, sizeof(interface));
    interface.access = access;

    CHECK(c_dma_attach_protocol_handler(access, self_detach_on_launch, self_detach_on_cancel, NULL, &interface) == DMA_OK);
    CHECK(c_dma_attach_protocol_callback(access, self_detach_on_reject, self_detach_on_filled) == DMA_OK);

    md_variant first = make_order("AAPL", 1., SIDE_LONG_OPEN);

    CHECK(c_dma_launch_order_unchecked(access, &first.trade_instruction) == DMA_OK);
    CHECK(interface.n_calls == 1);
    CHECK(access->protocol.handlers.order_launch_fn == NULL);

    md_variant second = make_order("AAPL", 1., SIDE_LONG_OPEN);

    CHECK(c_dma_launch_order_unchecked(access, &second.trade_instruction) == DMA_OK);
    CHECK(interface.n_calls == 1);

    c_dma_free(access);
}

static void test_dma_edge_contracts(void) {
    // Every entry refuses a NULL argument and leaves its out-parameter alone.
    CHECK(c_dma_init(NULL, NULL, NULL) == DMA_ERR_INVALID_INPUT);
    CHECK(c_dma_clear(NULL) == DMA_ERR_INVALID_INPUT);
    c_dma_dealloc(NULL);
    c_dma_free(NULL);

    CHECK(c_dma_attach_protocol_handler(NULL, test_on_launch, test_on_cancel, NULL, NULL) == DMA_ERR_INVALID_INPUT);
    CHECK(c_dma_detach_protocol_handler(NULL) == DMA_ERR_INVALID_INPUT);

    dma_risk_violation violation;
    memset(&violation, 0xff, sizeof(violation));
    CHECK(c_dma_check_order(NULL, NULL, &violation) == DMA_ERR_INVALID_INPUT);
    CHECK(violation.type == DMA_RISK_PASS);
    CHECK(violation.code == 0);
    CHECK(c_dma_check_order(NULL, NULL, NULL) == DMA_ERR_INVALID_INPUT);

    CHECK(c_dma_launch_order(NULL, NULL) == DMA_ERR_INVALID_INPUT);
    CHECK(c_dma_cancel_order(NULL, NULL) == DMA_ERR_INVALID_INPUT);
    CHECK(c_dma_launch_order_unchecked(NULL, NULL) == DMA_ERR_INVALID_INPUT);
    CHECK(c_dma_process_queue(NULL) == DMA_ERR_INVALID_INPUT);

    CHECK(c_dma_new_algo(NULL, 0, NULL, SIDE_LONG_OPEN, 1., NULL, g_allocator) == NULL);
    CHECK(isnan(c_dma_calculate_limit(NULL, NULL, SIDE_LONG_OPEN, 1., NAN, DMA_ALGO_LIMIT_STRICT)));

    CHECK(c_dma_order_queue_init(NULL, g_allocator) == DMA_ERR_INVALID_INPUT);
    c_dma_order_queue_dealloc(NULL);
    c_dma_order_queue_clear(NULL);
    CHECK(c_dma_order_queue_grow(NULL) == DMA_ERR_INVALID_INPUT);
    CHECK(c_dma_order_queue_push(NULL, DMA_ACTION_LAUNCH, NULL) == DMA_ERR_INVALID_INPUT);
    CHECK(c_dma_order_queue_pop(NULL, NULL, NULL) == DMA_ERR_INVALID_INPUT);

    dma_order_queue queue;
    memset(&queue, 0, sizeof(queue));
    CHECK(c_dma_order_queue_grow(&queue) == DMA_ERR_INVALID_INPUT);

    CHECK(c_dma_order_queue_init(&queue, g_allocator) == DMA_OK);
    CHECK(queue.capacity == (size_t) DMA_ORDER_QUEUE_INITIAL_CAPACITY);
    CHECK(queue.head == 0);
    CHECK(queue.size == 0);
    CHECK(c_dma_order_queue_push(&queue, DMA_ACTION_LAUNCH, NULL) == DMA_ERR_INVALID_INPUT);
    CHECK(c_dma_order_queue_pop(&queue, NULL, NULL) == DMA_ERR_EMPTY);

    c_dma_order_queue_dealloc(&queue);
    CHECK(queue.entries == NULL);
    CHECK(queue.capacity == 0);
    CHECK(queue.head == 0);
    CHECK(queue.size == 0);

    dma* access = c_dma_new(NULL, g_allocator);
    CHECK(access != NULL);

    dma_algo_engine* engine = access->ALGO_ENGINE;
    CHECK(engine != NULL);

    // The per-send sequence advances and never hands out 0, even across the wrap.
    CHECK(c_dma_next_seq_id(access) == 1);
    CHECK(c_dma_next_seq_id(access) == 2);

    access->seq_id = UINT32_MAX;
    CHECK(c_dma_next_seq_id(access) == 1);
    CHECK(c_dma_next_seq_id(access) == 2);

    access->seq_id = 41;
    CHECK(c_dma_next_seq_id(access) == 42);

    // Attaching a second handler set replaces the first.
    test_interface first;
    test_interface second;

    memset(&first, 0, sizeof(first));
    memset(&second, 0, sizeof(second));

    CHECK(c_dma_attach_protocol_handler(access, test_on_launch, test_on_cancel, NULL, &first) == DMA_OK);
    CHECK(c_dma_attach_protocol_handler(access, test_on_launch, test_on_cancel, NULL, &second) == DMA_OK);

    md_variant order = make_order("AAPL", 1., SIDE_LONG_OPEN);
    CHECK(c_dma_launch_order_unchecked(access, &order.trade_instruction) == DMA_OK);
    CHECK(first.n_launch == 0);
    CHECK(second.n_launch == 1);
    CHECK(second.last_user_data == &second);

    // The limit combines the absolute limit, the factor-adjusted market price
    // and the market price; strict tightens the long side and lifts the short.
    mds* market_data = c_mds_new(g_allocator);
    CHECK(market_data != NULL);

    market_data->ts.unix_ts    = 1720000000.;
    market_data->ts.ts         = 43200.;
    market_data->ts.date.year  = 2024;
    market_data->ts.date.month = 2;
    market_data->ts.date.day   = 8;
    market_data->ts.date.stype = SESSION_TYPE_NORMINAL;
    market_data->ts.time       = make_session_time(10, 0);

    md_variant quote;
    memset(&quote, 0, sizeof(quote));
    quote.meta_info.dtype         = DTYPE_TRANSACTION;
    quote.meta_info.ticker        = "AAPL";
    quote.meta_info.dt            = market_data->ts;
    quote.transaction_data.price  = 10.5;
    quote.transaction_data.volume = 100.;
    quote.transaction_data.side   = SIDE_LONG_OPEN;

    CHECK(c_mds_update_subscription(market_data, &quote) == MDS_OK_SUBSCRIBED);
    access->MDS = market_data;

    CHECK(fabs(c_dma_calculate_limit(access, "AAPL", SIDE_LONG_OPEN, NAN, NAN, DMA_ALGO_LIMIT_STRICT) - 10.5) < 1e-9);
    CHECK(fabs(c_dma_calculate_limit(access, "AAPL", SIDE_LONG_OPEN, 10., NAN, DMA_ALGO_LIMIT_STRICT) - 10.) < 1e-9);
    CHECK(fabs(c_dma_calculate_limit(access, "AAPL", SIDE_LONG_OPEN, 10., NAN, DMA_ALGO_LIMIT_LOOSE) - 10.5) < 1e-9);
    CHECK(fabs(c_dma_calculate_limit(access, "AAPL", SIDE_LONG_OPEN, NAN, 0.1, DMA_ALGO_LIMIT_LOOSE) - 11.55) < 1e-9);

    CHECK(fabs(c_dma_calculate_limit(access, "AAPL", SIDE_SHORT_OPEN, 11., NAN, DMA_ALGO_LIMIT_STRICT) - 11.) < 1e-9);
    CHECK(fabs(c_dma_calculate_limit(access, "AAPL", SIDE_SHORT_OPEN, 11., NAN, DMA_ALGO_LIMIT_LOOSE) - 10.5) < 1e-9);
    CHECK(fabs(c_dma_calculate_limit(access, "AAPL", SIDE_SHORT_OPEN, NAN, 0.1, DMA_ALGO_LIMIT_LOOSE) - 9.45) < 1e-9);

    CHECK(isnan(c_dma_calculate_limit(access, "NOPRICE", SIDE_LONG_OPEN, 10., NAN, DMA_ALGO_LIMIT_STRICT)));
    CHECK(isnan(c_dma_calculate_limit(access, NULL, SIDE_LONG_OPEN, 10., NAN, DMA_ALGO_LIMIT_STRICT)));

    c_dma_free(access);
    c_mds_free(market_data);
}

// ========== Balance ==========

static void test_balance(void) {
    dma_balance* balance = c_balance_new(g_allocator);
    CHECK(balance != NULL);
    CHECK(balance->flags == 0);

    // An unfilled balance and an absent one both pass.
    CHECK(c_balance_check(balance, SIDE_LONG_OPEN, 10., 10., 1., NAN));
    CHECK(c_balance_check(NULL, SIDE_LONG_OPEN, 10., 10., 1., NAN));

    // No margin (NaN): the notional is checked against available cash.
    balance->available_cash = 100.;
    balance->flags |= DMA_BALANCE_F_AVAILABLE_CASH;

    CHECK(c_balance_check(balance, SIDE_LONG_OPEN, 10., 10., 1., NAN));
    CHECK(!c_balance_check(balance, SIDE_LONG_OPEN, 11., 10., 1., NAN));
    CHECK(c_balance_check(balance, SIDE_SHORT_OPEN, 11., 10., 1., NAN));

    // The check reads available cash only; another field stands in for nothing.
    balance->flags = DMA_BALANCE_F_CASH_BALANCE;
    balance->cash_balance = 250.;
    CHECK(c_balance_check(balance, SIDE_LONG_OPEN, 10., 10., 1., NAN));

    // Any finite margin rate checks the notional times the rate against
    // available margin; a rate of 1 is a margin trade too.
    balance->available_cash   = 0.;
    balance->available_margin = 100.;
    balance->flags            = DMA_BALANCE_F_AVAILABLE_CASH | DMA_BALANCE_F_AVAILABLE_MARGIN;

    CHECK(c_balance_check(balance, SIDE_LONG_OPEN, 20., 10., 1., 0.5));
    CHECK(!c_balance_check(balance, SIDE_LONG_OPEN, 21., 10., 1., 0.5));
    CHECK(c_balance_check(balance, SIDE_LONG_OPEN, 10., 10., 1., 1.));
    CHECK(!c_balance_check(balance, SIDE_LONG_OPEN, 11., 10., 1., 1.));

    // The margin branch constrains the short side too.
    CHECK(c_balance_check(balance, SIDE_SHORT_OPEN, 20., 10., 1., 0.5));
    CHECK(!c_balance_check(balance, SIDE_SHORT_OPEN, 21., 10., 1., 0.5));

    // Without the margin field supplied, the margin branch passes.
    balance->flags = DMA_BALANCE_F_AVAILABLE_CASH;
    CHECK(c_balance_check(balance, SIDE_LONG_OPEN, 20., 10., 1., 0.5));

    CHECK(c_balance_clear(balance) == BALANCE_OK);
    CHECK(balance->flags == 0);
    CHECK(balance->available_cash == 0.);
    CHECK(c_balance_clear(NULL) == BALANCE_ERR_INVALID_INPUT);
    CHECK(c_balance_init(NULL) == BALANCE_ERR_INVALID_INPUT);

    dma_balance stack_balance;
    memset(&stack_balance, 0, sizeof(stack_balance));
    CHECK(c_balance_init(&stack_balance) == BALANCE_OK);

    c_balance_dealloc(NULL);
    c_balance_free(NULL);
    c_balance_free(balance);
}

static void test_dma_balance_gate(void) {
    dma* access = c_dma_new(NULL, g_allocator);
    CHECK(access != NULL);

    dma_algo_engine* engine = access->ALGO_ENGINE;
    CHECK(engine != NULL);

    dma_risk_profile* risk = access->RISK_PROFILE;
    set_full_ticker_rules(risk, "AAPL", 1e9, 1e9, 1e9, 1e9, 1e12, 1e12);
    set_full_globals(risk, INFINITY, INFINITY, INFINITY, INFINITY);

    test_balance_ctx ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.supply           = 1;
    ctx.available_cash   = 500.;
    ctx.available_margin = 500.;

    CHECK(c_dma_attach_protocol_handler(access, test_on_launch, test_on_cancel, test_balance_query, &ctx) == DMA_OK);

    md_variant order = make_order("AAPL", 100., SIDE_LONG_OPEN);
    order.trade_instruction.limit_price  = 10.;
    order.trade_instruction.multiplier   = 1.;

    dma_risk_violation violation;
    CHECK(c_dma_check_order(access, &order.trade_instruction, &violation) == DMA_ERR_RISK);
    CHECK(violation.type == DMA_RISK_BALANCE);
    CHECK(violation.code == c_dma_risk_violation_code(DMA_RISK_BALANCE));
    CHECK(violation.code == 6000);
    CHECK(strstr(violation.msg, "available cash") != NULL);
    CHECK(access->balance_valid);

    // Sufficient cash passes.
    ctx.available_cash = 5000.;
    CHECK(c_dma_check_order(access, &order.trade_instruction, &violation) == DMA_OK);
    CHECK(violation.type == DMA_RISK_PASS);

    // A declined query leaves the gate passing.
    ctx.supply = 0;
    CHECK(c_dma_query_balance(access) < 0);
    CHECK(!access->balance_valid);
    CHECK(c_dma_check_order(access, &order.trade_instruction, &violation) == DMA_OK);

    // No balance handler at all: the gate passes.
    CHECK(c_dma_attach_protocol_handler(access, test_on_launch, test_on_cancel, NULL, &ctx) == DMA_OK);
    ctx.supply         = 1;
    ctx.available_cash = 1.;
    CHECK(c_dma_check_order(access, &order.trade_instruction, &violation) == DMA_OK);
    CHECK(!access->balance_valid);

    // The sell side raises no cash requirement.
    CHECK(c_dma_attach_protocol_handler(access, test_on_launch, test_on_cancel, test_balance_query, &ctx) == DMA_OK);
    md_variant sell = make_order("AAPL", 100., SIDE_SHORT_OPEN);
    sell.trade_instruction.limit_price = 10.;
    sell.trade_instruction.multiplier  = 1.;
    CHECK(c_dma_check_order(access, &sell.trade_instruction, &violation) == DMA_OK);

    // A designated margin rate moves the gate to available margin: the same
    // order needs 500 at 0.5 and the cash field no longer matters.
    CHECK(c_dma_risk_profile_set_margin_rate(risk, "AAPL", 0.5) == DMA_RISK_OK_RET);
    CHECK(c_dma_risk_profile_get_margin_rate(risk, "AAPL") == 0.5);

    ctx.available_cash   = 0.;
    ctx.available_margin = 500.;
    CHECK(c_dma_check_order(access, &order.trade_instruction, &violation) == DMA_OK);

    ctx.available_margin = 400.;
    CHECK(c_dma_check_order(access, &order.trade_instruction, &violation) == DMA_ERR_RISK);
    CHECK(violation.type == DMA_RISK_BALANCE);
    CHECK(strstr(violation.msg, "available margin") != NULL);

    // The margin gate constrains the short side as well: the same 1000
    // notional needs 500 at 0.5 whichever direction it trades.
    ctx.available_margin = 500.;
    CHECK(c_dma_check_order(access, &sell.trade_instruction, &violation) == DMA_OK);
    ctx.available_margin = 400.;
    CHECK(c_dma_check_order(access, &sell.trade_instruction, &violation) == DMA_ERR_RISK);
    CHECK(violation.type == DMA_RISK_BALANCE);
    CHECK(strstr(violation.msg, "available margin") != NULL);

    // Removing the designation returns the ticker to the cash check.
    ctx.available_cash = 5000.;
    CHECK(c_dma_risk_profile_set_margin_rate(risk, "AAPL", NAN) == DMA_RISK_OK_RET);
    CHECK(isnan(c_dma_risk_profile_get_margin_rate(risk, "AAPL")));
    CHECK(c_dma_check_order(access, &order.trade_instruction, &violation) == DMA_OK);

    CHECK(c_dma_query_balance(NULL) == DMA_ERR_INVALID_INPUT);

    c_dma_free(access);
}

// ========== Inventory ==========

static void test_inventory(void) {
    dma_inventory* inventory = c_dma_inventory_new(g_allocator);
    CHECK(inventory != NULL);

    double                     long_available = -1.;
    double                     short_available = -1.;
    const dma_inventory_entry* lot = NULL;

    CHECK(c_dma_inventory_available(inventory, "AAPL", &long_available, &short_available) == INVENTORY_OK);
    CHECK(long_available == 0.);
    CHECK(short_available == 0.);
    CHECK(c_dma_inventory_get_inv_long(inventory, "AAPL", &lot) == INVENTORY_ERR_NOT_FOUND);
    CHECK(lot == NULL);

    CHECK(c_dma_inventory_add_inv_long(inventory, "AAPL", 10., 100., 1000., 1., DMA_SECURITY_STOCK) == INVENTORY_OK);
    CHECK(c_dma_inventory_add_inv_long(inventory, "AAPL", 12., 100., 1200., 2., DMA_SECURITY_STOCK) == INVENTORY_OK);

    // One node per ticker: the ticker lives on the node, the lots on its lists.
    bytemap_entry* node_entry = inventory->nodes->first;
    CHECK(node_entry != NULL);
    dma_inventory_node* node = *(dma_inventory_node**) node_entry->value;
    CHECK(strcmp(node->ticker, "AAPL") == 0);
    CHECK(node->long_inv != NULL);
    CHECK(node->long_inv->next != NULL);
    CHECK(node->short_inv == NULL);
    CHECK(node->long_used == 0.);
    CHECK(node->short_used == 0.);

    // The getter returns the live first lot of the side.
    CHECK(c_dma_inventory_get_inv_long(inventory, "AAPL", &lot) == INVENTORY_OK);
    CHECK(lot == node->long_inv);
    CHECK(lot->volume == 100.);
    CHECK(lot->price == 10.);
    CHECK(lot->notional == 1000.);
    CHECK(lot->fee == 1.);
    CHECK(lot->recalled == 0.);
    CHECK(lot->security_type == DMA_SECURITY_STOCK);
    CHECK(lot->next->volume == 100.);
    CHECK(lot->next->price == 12.);
    CHECK(lot->next->fee == 2.);

    // The short list is separate from the long list.
    CHECK(c_dma_inventory_get_inv_short(inventory, "AAPL", &lot) == INVENTORY_ERR_NOT_FOUND);
    CHECK(c_dma_inventory_add_inv_short(inventory, "AAPL", 10., 100., 1000., 1., DMA_SECURITY_STOCK) == INVENTORY_OK);
    CHECK(c_dma_inventory_get_inv_short(inventory, "AAPL", &lot) == INVENTORY_OK);
    CHECK(lot == node->short_inv);
    CHECK(lot->volume == 100.);

    // Consumed volume is kept per direction.
    CHECK(c_dma_inventory_use_inv_long(inventory, "AAPL", 50.) == INVENTORY_OK);
    CHECK(c_dma_inventory_available(inventory, "AAPL", &long_available, &short_available) == INVENTORY_OK);
    CHECK(long_available == 150.);
    CHECK(short_available == 100.);

    // Recall 120: the first long lot is fully recalled and removed, the second keeps 20.
    CHECK(c_dma_inventory_recall_inv_long(inventory, "AAPL", 120.) == INVENTORY_OK);
    CHECK(node->long_inv != NULL);
    CHECK(node->long_inv->next == NULL);
    CHECK(node->short_inv != NULL);

    CHECK(c_dma_inventory_get_inv_long(inventory, "AAPL", &lot) == INVENTORY_OK);
    CHECK(lot->volume == 100.);
    CHECK(lot->recalled == 20.);
    CHECK(c_dma_inventory_available(inventory, "AAPL", &long_available, &short_available) == INVENTORY_OK);
    CHECK(long_available == 30.);
    CHECK(short_available == 100.);

    // Recalling past the held volume removes the remaining long lot.
    CHECK(c_dma_inventory_recall_inv_long(inventory, "AAPL", 1000.) == INVENTORY_OK);
    CHECK(node->long_inv == NULL);
    CHECK(c_dma_inventory_get_inv_long(inventory, "AAPL", &lot) == INVENTORY_ERR_NOT_FOUND);
    CHECK(c_dma_inventory_get_inv_short(inventory, "AAPL", &lot) == INVENTORY_OK);

    // A use-only ticker still creates its node and ticker copy.
    CHECK(c_dma_inventory_use_inv_short(inventory, "MSFT", 5.) == INVENTORY_OK);
    CHECK(c_bytemap_ex_len(inventory->nodes) == 2);

    CHECK(c_dma_inventory_clear(inventory) == INVENTORY_OK);
    CHECK(c_bytemap_ex_len(inventory->nodes) == 0);

    // A standalone lot block is allocator-zeroed and frees its event lists.
    dma_inventory_entry* standalone = c_dma_inventory_entry_new(g_allocator);
    CHECK(standalone != NULL);
    CHECK(standalone->volume == 0.);
    standalone->volume = 1.;
    c_dma_inventory_entry_free(standalone);

    c_dma_inventory_free(inventory);
}

static void test_inventory_corporate_actions(void) {
    dma_inventory* inventory = c_dma_inventory_new(g_allocator);
    CHECK(inventory != NULL);

    CHECK(c_dma_inventory_add_inv_long(inventory, "AAPL", 10., 100., 1000., 0., DMA_SECURITY_STOCK) == INVENTORY_OK);
    CHECK(c_dma_inventory_add_inv_short(inventory, "AAPL", 10., 100., 1000., 0., DMA_SECURITY_STOCK) == INVENTORY_OK);

    bytemap_entry* node_entry = inventory->nodes->first;
    CHECK(node_entry != NULL);
    dma_inventory_node* node = *(dma_inventory_node**) node_entry->value;

    dma_inventory_entry* long_lot = node->long_inv;
    dma_inventory_entry* short_lot = node->short_inv;
    CHECK(long_lot != NULL);
    CHECK(short_lot != NULL);

    session_date_t record_date = {.year = 2024, .month = 11, .day = 1};
    session_date_t ex_date = {.year = 2024, .month = 11, .day = 11};

    // Cash dividend: price marked down on both sides and recorded per lot.
    CHECK(c_dma_inventory_cash_dividend(inventory, record_date, ex_date, 0.5) == INVENTORY_OK);

    CHECK(long_lot->volume == 100.);
    CHECK(long_lot->price == 9.5);
    CHECK(long_lot->notional == 950.);
    CHECK(short_lot->price == 9.5);
    CHECK(short_lot->notional == 950.);

    CHECK(long_lot->cash_dividends != NULL);
    CHECK(long_lot->cash_dividends->next == NULL);
    CHECK(long_lot->cash_dividends->applied);
    CHECK(long_lot->cash_dividends->dividend_per_share == 0.5);
    CHECK(long_lot->cash_dividends->record_date.day == 1);
    CHECK(long_lot->cash_dividends->ex_date.day == 11);
    CHECK(short_lot->cash_dividends != NULL);
    CHECK(short_lot->cash_dividends->applied);

    // Stock dividend: volume grows and the price scales, on both sides.
    CHECK(c_dma_inventory_stock_dividend(inventory, record_date, ex_date, 0.2) == INVENTORY_OK);
    CHECK(long_lot->volume == 120.);
    CHECK(fabs(long_lot->price - (9.5 / 1.2)) < 1e-12);
    CHECK(long_lot->notional == 950.);
    CHECK(long_lot->stock_dividends != NULL);
    CHECK(long_lot->stock_dividends->applied);
    CHECK(short_lot->stock_dividends != NULL);

    // Split: volume doubles and the price halves.
    CHECK(c_dma_inventory_stock_split(inventory, record_date, ex_date, 2.) == INVENTORY_OK);
    CHECK(long_lot->volume == 240.);
    CHECK(fabs(long_lot->price - (9.5 / 2.4)) < 1e-12);
    CHECK(long_lot->notional == 950.);
    CHECK(long_lot->stock_splits != NULL);
    CHECK(long_lot->stock_splits->applied);
    CHECK(long_lot->stock_splits->multiplier == 2.);
    CHECK(long_lot->stock_splits->record_date.day == 1);
    CHECK(long_lot->stock_splits->ex_date.day == 11);
    CHECK(short_lot->stock_splits != NULL);

    // Conversion: volume scales, price divides and the record keeps the tickers.
    CHECK(c_dma_inventory_stock_conversion(inventory, record_date, ex_date, "AAPL", "NEW.TICKER", 1.5) == INVENTORY_OK);
    CHECK(long_lot->volume == 360.);
    CHECK(fabs(long_lot->price - (9.5 / 3.6)) < 1e-12);
    CHECK(long_lot->notional == 950.);
    CHECK(long_lot->stock_conversions != NULL);
    CHECK(long_lot->stock_conversions->applied);
    CHECK(strcmp(long_lot->stock_conversions->convert_from, "AAPL") == 0);
    CHECK(strcmp(long_lot->stock_conversions->convert_to, "NEW.TICKER") == 0);
    CHECK(short_lot->stock_conversions != NULL);

    // A second event of the same kind appends to each lot's list.
    CHECK(c_dma_inventory_cash_dividend(inventory, record_date, ex_date, 0.25) == INVENTORY_OK);
    CHECK(long_lot->cash_dividends->next != NULL);
    CHECK(long_lot->cash_dividends->next->next == NULL);

    // The getters expose the live lots the events were recorded on.
    const dma_inventory_entry* got = NULL;
    CHECK(c_dma_inventory_get_inv_long(inventory, "AAPL", &got) == INVENTORY_OK);
    CHECK(got == long_lot);
    CHECK(c_dma_inventory_get_inv_short(inventory, "AAPL", &got) == INVENTORY_OK);
    CHECK(got == short_lot);

    // Invalid inputs are refused.
    CHECK(c_dma_inventory_cash_dividend(NULL, record_date, ex_date, 0.5) == INVENTORY_ERR_INVALID_INPUT);
    CHECK(c_dma_inventory_stock_dividend(inventory, record_date, ex_date, -1.) == INVENTORY_ERR_INVALID_INPUT);
    CHECK(c_dma_inventory_stock_split(inventory, record_date, ex_date, 0.) == INVENTORY_ERR_INVALID_INPUT);
    CHECK(c_dma_inventory_stock_conversion(inventory, record_date, ex_date, "AAPL", "X", 0.) == INVENTORY_ERR_INVALID_INPUT);

    CHECK(c_dma_inventory_clear(inventory) == INVENTORY_OK);
    CHECK(c_bytemap_ex_len(inventory->nodes) == 0);

    c_dma_inventory_free(inventory);
}

// ========== Algo Engine ==========

static dma_algo* test_algo_factory(dma* DMA, const char* ticker, md_side side, double target_volume, const void* params, allocator_protocol* allocator) {
    (void) params;

    dma_algo* algo = (dma_algo*) c_ap_alloc(sizeof(dma_algo), allocator);
    if (!algo) return NULL;

    if (c_dma_algo_init(algo, DMA, ticker, side, target_volume, 1., allocator) != DMA_ALGO_OK) {
        c_dma_algo_dealloc(algo);
        c_ap_free(algo);
        return NULL;
    }

    return algo;
}

typedef struct test_algo_ctx {
    int listed_hits;
    int canceled_hits;
    int rejected_hits;
    int filled_hits;
    int md_hits;
    int launch_hits;
    int stop_hits;
} test_algo_ctx;

static void test_algo_on_listed(dma* access, dma_algo* algo, const md_trade_instruction* order, void* user_data) {
    (void) access;
    (void) algo;
    (void) order;

    ((test_algo_ctx*) user_data)->listed_hits += 1;
}

static void test_algo_on_canceled(dma* access, dma_algo* algo, const md_trade_instruction* order, void* user_data) {
    (void) access;
    (void) algo;
    (void) order;

    ((test_algo_ctx*) user_data)->canceled_hits += 1;
}

static void test_algo_on_rejected(dma* access, dma_algo* algo, const md_trade_instruction* order, void* user_data) {
    (void) access;
    (void) algo;
    (void) order;

    ((test_algo_ctx*) user_data)->rejected_hits += 1;
}

static void test_algo_on_filled(dma* access, dma_algo* algo, const md_trade_report* trade, void* user_data) {
    (void) access;
    (void) algo;
    (void) trade;

    ((test_algo_ctx*) user_data)->filled_hits += 1;
}

static void test_algo_on_md(dma* access, dma_algo* algo, const md_variant* market_data, void* user_data) {
    (void) access;
    (void) algo;
    (void) market_data;

    ((test_algo_ctx*) user_data)->md_hits += 1;
}

static int test_algo_launch(dma* access, dma_algo* algo, void* user_data) {
    (void) access;
    (void) algo;

    ((test_algo_ctx*) user_data)->launch_hits += 1;
    return 7;
}

static int test_algo_stop(dma* access, dma_algo* algo, void* user_data) {
    (void) access;
    (void) algo;

    ((test_algo_ctx*) user_data)->stop_hits += 1;
    return 9;
}

static void test_algo_engine(void) {
    dma_algo_registry* registry = c_dma_algo_registry_new(g_allocator);
    CHECK(registry != NULL);
    CHECK(c_bytemap_ex_len(registry->entries) == 0);
    CHECK(c_bytemap_ex_len(registry->id_map) == 0);

    dma_algo_ctx* reg_one = c_dma_algo_registry_reg(registry, "passive", test_algo_factory, 0);
    CHECK(reg_one != NULL);
    CHECK(reg_one->reg_id == 1);

    // Re-registering the name repoints it and keeps the older context by id.
    dma_algo_ctx* reg_two = c_dma_algo_registry_reg(registry, "passive", test_algo_factory, 0);
    CHECK(reg_two != NULL);
    CHECK(reg_two->reg_id == 2);
    CHECK(c_bytemap_ex_len(registry->entries) == 1);
    CHECK(c_bytemap_ex_len(registry->id_map) == 2);
    CHECK(c_dma_algo_registry_get(registry, 1) == reg_one);
    CHECK(c_dma_algo_registry_get(registry, 2) == reg_two);
    CHECK(c_dma_algo_registry_get(registry, 3) == NULL);

    // A caller id is refused on collision and accepted when free.
    CHECK(c_dma_algo_registry_reg(registry, "copy", test_algo_factory, 2) == NULL);

    dma_algo_ctx* reg_user = c_dma_algo_registry_reg(registry, "custom", test_algo_factory, 7);
    CHECK(reg_user != NULL);
    CHECK(reg_user->reg_id == 7);
    CHECK(c_dma_algo_registry_get(registry, 7) == reg_user);
    CHECK(c_dma_algo_registry_reg(NULL, "x", test_algo_factory, 0) == NULL);

    dma* access = c_dma_new(NULL, g_allocator);
    CHECK(access != NULL);

    dma_algo_engine* engine = access->ALGO_ENGINE;
    CHECK(engine != NULL);
    CHECK(engine->registry != NULL);
    CHECK(engine->orders != NULL);
    CHECK(engine->working_orders_by_id != NULL);
    CHECK(engine->working_orders_by_ticker != NULL);
    CHECK(engine->algo != NULL);
    CHECK(engine->working_algo != NULL);
    CHECK(engine->position != NULL);

    CHECK(c_dma_algo_compare_limit(SIDE_LONG_OPEN, 10., 11., DMA_ALGO_LIMIT_STRICT) == 10.);
    CHECK(c_dma_algo_compare_limit(SIDE_LONG_OPEN, 12., 11., DMA_ALGO_LIMIT_STRICT) == 11.);
    CHECK(c_dma_algo_compare_limit(SIDE_LONG_OPEN, 12., 11., DMA_ALGO_LIMIT_LOOSE) == 12.);
    CHECK(c_dma_algo_compare_limit(SIDE_SHORT_OPEN, 10., 11., DMA_ALGO_LIMIT_STRICT) == 11.);
    CHECK(c_dma_algo_compare_limit(SIDE_LONG_OPEN, 10., NAN, DMA_ALGO_LIMIT_STRICT) == 10.);

    dma_algo_ctx* engine_ctx = c_dma_algo_registry_reg(engine->registry, "test", test_algo_factory, 0);
    CHECK(engine_ctx != NULL);

    mds mds_state;
    memset(&mds_state, 0, sizeof(mds_state));
    mds_state.ts.unix_ts = 1.;

    access->MDS         = &mds_state;

    dma_algo* created = c_dma_new_algo(access, engine_ctx->reg_id, "AAPL", SIDE_LONG_OPEN, 10., NULL, g_allocator);
    CHECK(created != NULL);
    CHECK(created->algo_reg_id == engine_ctx->reg_id);
    CHECK(c_dma_new_algo(access, 9999, "AAPL", SIDE_LONG_OPEN, 10., NULL, g_allocator) == NULL);
    CHECK(c_bytemap_ex_len(engine_ctx->algos) == 1);
    CHECK(c_bytemap_ex_len(engine->algo) == 1);
    CHECK(c_bytemap_ex_len(engine->working_algo) == 1);  // IDLE is inside the working set.

    md_trade_instruction* order = c_dma_algo_engine_new_order(engine, created, 10., 10.);
    CHECK(order != NULL);
    CHECK(c_bytemap_ex_len(engine->orders) == 1);
    CHECK(c_bytemap_ex_len(engine->working_orders_by_id) == 1);
    CHECK(c_bytemap_ex_len(engine->working_orders_by_ticker) == 1);
    CHECK(c_bytemap_ex_len(engine_ctx->orders) == 1);
    CHECK(c_dma_algo_get_working_order(created, &order->order_id) == order);

    char* order_stored = NULL;
    CHECK(c_bytemap_ex_get_ptr(engine->orders, (const char*) &order->order_id, sizeof(long_md_id), &order_stored, NULL) == BYTEMAP_OK);

    dma_order_ctx* order_ctx = *(dma_order_ctx**) order_stored;

    // The order leaves IDLE through the sent entry; the fill flow accepts it.
    order->order_state = STATE_SENT;
    CHECK(c_dma_algo_engine_on_order_launched_inplace(engine, order_ctx) == DMA_ALGO_OK);
    CHECK(created->working_volume == 10.);
    CHECK((order_ctx->flags & DMA_ORDER_LISTED) != 0);

    md_trade_report fill_one;
    memset(&fill_one, 0, sizeof(fill_one));
    fill_one.meta_info.dt.unix_ts = 2.;
    fill_one.multiplier = 1.;
    fill_one.order_id = order->order_id;
    fill_one.side     = SIDE_LONG_OPEN;
    strcpy(fill_one.trade_id.data, "fill-1");
    fill_one.volume = 4.;
    fill_one.notional = 42.;
    fill_one.fee = .1;

    md_trade_report fill_two = fill_one;
    strcpy(fill_two.trade_id.data, "fill-2");
    fill_two.volume = 6.;
    fill_two.notional = 63.;

    CHECK(c_dma_algo_engine_on_trade(engine, &fill_one) == DMA_ALGO_OK);
    CHECK(order->filled_volume == 4.);
    CHECK(order->order_state == STATE_SENT);  // The fill books; the state column is the order report's.
    CHECK(c_bytemap_ex_len(order_ctx->reports) == 1);
    CHECK(created->working_volume == 6.);

    CHECK(c_dma_algo_engine_on_trade(engine, &fill_one) == DMA_ALGO_OK);  // The trade-id gate.
    CHECK(order->filled_volume == 4.);
    CHECK(c_bytemap_ex_len(order_ctx->reports) == 1);
    CHECK(created->working_volume == 6.);

    CHECK(c_dma_algo_engine_on_trade(engine, &fill_two) == DMA_ALGO_OK);
    CHECK(order->filled_volume == 10.);
    CHECK(order->order_state == STATE_SENT);
    CHECK(c_bytemap_ex_len(order_ctx->reports) == 2);
    CHECK(created->working_volume == 0.);

    md_trade_instruction* cancel_order = c_dma_algo_engine_new_order(engine, created, 10., 10.);
    CHECK(cancel_order != NULL);

    order_stored = NULL;
    CHECK(c_bytemap_ex_get_ptr(engine->orders, (const char*) &cancel_order->order_id, sizeof(long_md_id), &order_stored, NULL) == BYTEMAP_OK);

    dma_order_ctx* cancel_ctx = *(dma_order_ctx**) order_stored;

    md_trade_report cancel_report;
    memset(&cancel_report, 0, sizeof(cancel_report));
    cancel_report.meta_info.dt.unix_ts = 4.;
    cancel_report.order_id = cancel_order->order_id;
    strcpy(cancel_report.trade_id.data, "cancel-1");
    cancel_report.side   = SIDE_CANCEL;  // The side offset identifies the cancel report.
    cancel_report.volume = 10.;          // The exchange reports the canceled amount; it is not booked.

    cancel_order->order_state = STATE_SENT;
    CHECK(c_dma_algo_engine_on_order_launched_inplace(engine, cancel_ctx) == DMA_ALGO_OK);
    CHECK(created->working_volume == 10.);

    cancel_order->order_state = STATE_CANCELING;
    CHECK(c_dma_algo_engine_on_order(engine, cancel_order) == DMA_ALGO_OK);  // The stored-order view route.
    CHECK(created->working_volume == 10.);

    CHECK(c_dma_algo_engine_on_trade(engine, &cancel_report) == DMA_ALGO_OK);
    CHECK(cancel_order->order_state == STATE_CANCELED);
    CHECK(cancel_order->filled_volume == 0.);
    CHECK(created->working_volume == 0.);
    CHECK(c_bytemap_ex_len(cancel_ctx->reports) == 1);
    CHECK(c_bytemap_ex_len(engine->working_orders_by_id) == 1);

    CHECK(c_dma_algo_engine_on_order(engine, cancel_order) == DMA_ALGO_ERR_NOT_FOUND);

    md_trade_instruction filled = *order;
    filled.order_state = STATE_FILLED;
    CHECK(c_dma_algo_engine_on_order(engine, &filled) == DMA_ALGO_OK);
    CHECK(c_dma_algo_engine_on_order(engine, order) == DMA_ALGO_ERR_NOT_FOUND);
    CHECK(c_bytemap_ex_len(engine->working_orders_by_id) == 0);
    CHECK(c_bytemap_ex_len(engine->orders) == 2);
    CHECK(c_dma_algo_get_working_order(created, &order->order_id) == NULL);
    CHECK(c_dma_algo_get_order(created, &order->order_id) == order);

    // === The access's local cancel applies through the in-place entry ===
    // The order never left the access, so its volume never entered the working
    // set: the CANCELED entry leaves the working volume untouched and applies
    // the index departure and the callback.
    test_algo_ctx local_ctx;
    memset(&local_ctx, 0, sizeof(local_ctx));

    created->callbacks.on_canceled = test_algo_on_canceled;
    created->callbacks.user_data   = &local_ctx;

    md_trade_instruction* local_order = c_dma_algo_engine_new_order(engine, created, 10., 5.);
    CHECK(local_order != NULL);
    CHECK(created->working_volume == 0.);
    CHECK(((dma_order_ctx*) local_order)->flags == 0);

    CHECK(c_dma_cancel_order(access, local_order) == DMA_OK);
    CHECK(local_order->order_state == STATE_CANCELED);
    CHECK(created->working_volume == 0.);
    CHECK(((dma_order_ctx*) local_order)->flags == DMA_ORDER_CANCELED);
    CHECK(local_ctx.canceled_hits == 1);
    CHECK(c_dma_algo_get_working_order(created, &local_order->order_id) == NULL);
    CHECK(c_bytemap_ex_len(engine->working_orders_by_id) == 0);

    c_dma_algo_engine_free_orders(engine, created);
    CHECK(c_bytemap_ex_len(engine->orders) == 0);
    CHECK(c_bytemap_ex_len(created->orders) == 0);
    CHECK(c_bytemap_ex_len(created->working_orders) == 0);
    CHECK(c_bytemap_ex_len(engine_ctx->algos) == 1);
    CHECK(c_bytemap_ex_len(engine_ctx->orders) == 0);

    // The engine owns the algo record and releases it with the engine. The
    // record never left IDLE, so it is still on the working list.
    CHECK(c_bytemap_ex_len(engine->algo) == 1);
    CHECK(c_bytemap_ex_len(engine->working_algo) == 1);

    c_dma_free(access);
    c_dma_algo_registry_free(registry);
}

typedef struct test_position_ctx {
    size_t                      canceled_hits;
    size_t                      rejected_hits;
    size_t                      listed_hits;
    size_t                      filled_hits;
    dma_position*               last_position;
    const dma_algo*             last_algo;
    const md_trade_instruction* last_order;
    const md_trade_report*      last_trade;
} test_position_ctx;

static int test_position_on_canceled(dma_position* position, const dma_algo* algo, const md_trade_instruction* order, void* user_data) {
    test_position_ctx* ctx = (test_position_ctx*) user_data;

    ctx->canceled_hits++;
    ctx->last_position = position;
    ctx->last_algo     = algo;
    ctx->last_order    = order;
    return 0;
}

static int test_position_on_rejected(dma_position* position, const dma_algo* algo, const md_trade_instruction* order, void* user_data) {
    test_position_ctx* ctx = (test_position_ctx*) user_data;

    ctx->rejected_hits++;
    ctx->last_position = position;
    ctx->last_algo     = algo;
    ctx->last_order    = order;
    return 0;
}

static int test_position_on_listed(dma_position* position, const dma_algo* algo, const md_trade_instruction* order, void* user_data) {
    test_position_ctx* ctx = (test_position_ctx*) user_data;

    ctx->listed_hits++;
    ctx->last_position = position;
    ctx->last_algo     = algo;
    ctx->last_order    = order;
    return 0;
}

static int test_position_on_filled(dma_position* position, const dma_algo* algo, const md_trade_instruction* order, const md_trade_report* trade, void* user_data) {
    test_position_ctx* ctx = (test_position_ctx*) user_data;

    ctx->filled_hits++;
    ctx->last_position = position;
    ctx->last_algo     = algo;
    ctx->last_order    = order;
    ctx->last_trade    = trade;
    return 0;
}

static void test_position(void) {
    dma* access = c_dma_new(NULL, g_allocator);
    CHECK(access != NULL);

    dma_algo_engine* engine = access->ALGO_ENGINE;
    CHECK(engine != NULL);
    CHECK(engine->position != NULL);

    mds mds_state;
    memset(&mds_state, 0, sizeof(mds_state));
    mds_state.ts.unix_ts = 1.;

    access->MDS         = &mds_state;

    dma_algo_ctx* registration = c_dma_algo_registry_reg(engine->registry, "position", test_algo_factory, 0);
    CHECK(registration != NULL);

    dma_position* position = c_dma_position_new(access, engine, "position", g_allocator);
    CHECK(position != NULL);
    CHECK(position->dma == access);
    CHECK(position->algo_engine == engine);
    CHECK(strcmp(position->default_algo, "position") == 0);
    CHECK(position->callbacks == NULL);
    CHECK(c_bytemap_ex_len(access->positions) == 1);

    dma_position* defaulted = c_dma_position_new(access, engine, NULL, g_allocator);
    CHECK(defaulted != NULL);
    CHECK(strcmp(defaulted->default_algo, DMA_POSITION_DEFAULT_ALGO) == 0);
    c_dma_position_free(defaulted);
    CHECK(c_bytemap_ex_len(access->positions) == 1);

    static const unsigned char zero_id[16] = {0};
    CHECK(memcmp(position->position_id, zero_id, sizeof(uuid_t)) != 0);
    CHECK((position->position_id[6] & 0xF0) == 0x40);
    CHECK((position->position_id[8] & 0xC0) == 0x80);

    test_algo_ctx algo_ctx;
    memset(&algo_ctx, 0, sizeof(algo_ctx));

    test_position_ctx position_ctx;
    memset(&position_ctx, 0, sizeof(position_ctx));

    test_position_ctx other_ctx;
    memset(&other_ctx, 0, sizeof(other_ctx));

    uintptr_t callback_id = 0;

    CHECK(c_dma_position_register_callback(position, on_listed, (void*) test_position_on_listed, &position_ctx, &callback_id) == POSITION_OK);
    CHECK(callback_id != 0);
    CHECK(c_dma_position_register_callback(position, on_canceled, (void*) test_position_on_canceled, &position_ctx, NULL) == POSITION_OK);
    CHECK(c_dma_position_register_callback(position, on_rejected, (void*) test_position_on_rejected, &position_ctx, NULL) == POSITION_OK);
    CHECK(c_dma_position_register_callback(position, on_filled, (void*) test_position_on_filled, &position_ctx, NULL) == POSITION_OK);
    CHECK(c_dma_position_register_callback(position, (dma_callback_event) 99, (void*) test_position_on_listed, NULL, NULL) == POSITION_ERR_INVALID_INPUT);
    CHECK(c_dma_position_register_callback(position, on_listed, NULL, NULL, NULL) == POSITION_ERR_INVALID_INPUT);
    CHECK(c_dma_position_register_callback(NULL, on_listed, (void*) test_position_on_listed, NULL, NULL) == POSITION_ERR_INVALID_INPUT);

    dma_algo* algo_a = c_dma_new_algo(access, registration->reg_id, "AAPL", SIDE_LONG_OPEN, 100., NULL, g_allocator);
    dma_algo* algo_b = c_dma_new_algo(access, registration->reg_id, "AAPL", SIDE_SHORT_OPEN, 50., NULL, g_allocator);
    dma_algo* algo_c = c_dma_new_algo(access, registration->reg_id, "MSFT", SIDE_LONG_OPEN, 10., NULL, g_allocator);
    CHECK(algo_a != NULL);
    CHECK(algo_b != NULL);
    CHECK(algo_c != NULL);

    algo_a->handlers.stop      = test_algo_stop;
    algo_a->handlers.user_data = &algo_ctx;
    algo_b->handlers.stop      = test_algo_stop;
    algo_b->handlers.user_data = &algo_ctx;

    algo_a->callbacks.on_listed   = test_algo_on_listed;
    algo_a->callbacks.on_canceled = test_algo_on_canceled;
    algo_a->callbacks.on_rejected = test_algo_on_rejected;
    algo_a->callbacks.on_filled   = test_algo_on_filled;
    algo_a->callbacks.user_data   = &algo_ctx;
    algo_b->callbacks.on_canceled = test_algo_on_canceled;
    algo_b->callbacks.on_rejected = test_algo_on_rejected;
    algo_b->callbacks.user_data   = &algo_ctx;
    algo_a->handlers.launch       = test_algo_launch;
    CHECK(c_dma_algo_launch(algo_a) == 7);
    CHECK(algo_ctx.launch_hits == 1);

    // === Hosting ===
    CHECK(c_dma_position_add_algo(position, algo_a) == POSITION_OK);
    CHECK(c_dma_position_add_algo(position, algo_a) == POSITION_ERR_INVALID_INPUT);
    CHECK(c_dma_position_add_algo(position, algo_b) == POSITION_OK);
    CHECK(c_dma_position_add_algo_by_id(position, algo_a->algo_id) == POSITION_ERR_INVALID_INPUT);
    CHECK(c_dma_position_get_algo(position, algo_a->algo_id) == algo_a);
    CHECK(c_dma_position_get_algo(position, algo_b->algo_id) == algo_b);
    CHECK(c_bytemap_ex_len(position->algos) == 2);
    CHECK(c_bytemap_ex_len(position->working_algos) == 2);
    CHECK(c_bytemap_ex_len(position->tickers) == 1);
    CHECK(c_bytemap_ex_len(engine->position) == 2);
    CHECK(c_dma_position_n_algos(position) == 2);
    CHECK(c_dma_position_n_working_algos(position) == 2);
    CHECK(c_dma_position_get_exposure(position, "AAPL") == 0.);
    CHECK(c_dma_position_notional(position, "AAPL") == 0.);

    uuid_t missing_id = {0xff};
    CHECK(c_dma_position_get_algo(position, missing_id) == NULL);
    CHECK(c_dma_position_add_algo_by_id(position, missing_id) == POSITION_ERR_NOT_FOUND);
    CHECK(c_dma_position_pop_algo(position, missing_id) == POSITION_ERR_NOT_FOUND);

    // A second position hosts its own subset; a record cannot be hosted twice.
    dma_position* other = c_dma_position_new(access, engine, "position", g_allocator);
    CHECK(other != NULL);
    CHECK(c_bytemap_ex_len(access->positions) == 2);
    CHECK(c_dma_position_add_algo(other, algo_c) == POSITION_OK);
    CHECK(c_dma_position_add_algo(other, algo_a) == POSITION_ERR_INVALID_INPUT);
    CHECK(c_bytemap_ex_len(engine->position) == 3);

    CHECK(c_dma_position_register_callback(other, on_listed, (void*) test_position_on_listed, &other_ctx, NULL) == POSITION_OK);
    CHECK(c_dma_position_register_callback(other, on_filled, (void*) test_position_on_filled, &other_ctx, NULL) == POSITION_OK);

    // === In-place event updates ===
    // A new order is idle and carries no working volume; the first working
    // state enters it.
    md_trade_instruction* order = c_dma_algo_engine_new_order(engine, algo_a, 10., 10.);
    CHECK(order != NULL);
    CHECK(((dma_order_ctx*) order)->flags == 0);

    double working_long  = 0.;
    double working_short = 0.;

    CHECK(c_dma_position_get_working(position, "AAPL", &working_long, &working_short) == POSITION_OK);
    CHECK(working_long == 0.);
    CHECK(working_short == 0.);

    md_trade_instruction placed = *order;
    placed.order_state = STATE_PLACED;
    CHECK(c_dma_algo_engine_on_order(engine, &placed) == DMA_ALGO_OK);
    CHECK(position_ctx.listed_hits == 1);
    CHECK(position_ctx.last_position == position);
    CHECK(position_ctx.last_algo == algo_a);
    CHECK(other_ctx.listed_hits == 0);
    CHECK(algo_ctx.listed_hits == 1);
    CHECK(((dma_order_ctx*) order)->flags == DMA_ORDER_LISTED);
    CHECK(c_dma_position_get_working(position, "AAPL", &working_long, &working_short) == POSITION_OK);
    CHECK(working_long == 10.);

    // A fill adds its exposure, notional and cash flow directly.
    md_trade_report fill;
    memset(&fill, 0, sizeof(fill));
    fill.meta_info.dt.unix_ts = 2.;
    fill.multiplier           = 1.;
    fill.order_id             = order->order_id;
    fill.volume               = 10.;
    fill.notional             = 105.;
    fill.side                 = SIDE_LONG_OPEN;
    strcpy(fill.trade_id.data, "position-1");

    CHECK(c_dma_algo_engine_on_trade(engine, &fill) == DMA_ALGO_OK);
    CHECK(position_ctx.filled_hits == 1);
    CHECK(position_ctx.last_trade != NULL);
    CHECK(other_ctx.filled_hits == 0);
    CHECK(algo_ctx.filled_hits == 1);
    CHECK(c_dma_position_get_exposure(position, "AAPL") == 10.);
    CHECK(c_dma_position_get_working(position, "AAPL", &working_long, &working_short) == POSITION_OK);
    CHECK(working_long == 0.);
    CHECK(c_dma_position_notional(position, "AAPL") == 105.);
    CHECK(isnan(c_dma_position_pnl(position, "AAPL")));  // No market price on the bare MDS.

    // The exchange reports the order's fill state; the engine closes the order.
    md_trade_instruction filled_state = *order;
    filled_state.order_state = STATE_FILLED;
    CHECK(c_dma_algo_engine_on_order(engine, &filled_state) == DMA_ALGO_OK);

    // A canceled order opens its volume on the sent state and releases it on
    // the cancel, dispatching on_canceled.
    md_trade_instruction* canceled_order = c_dma_algo_engine_new_order(engine, algo_b, 10., 5.);
    CHECK(canceled_order != NULL);
    CHECK(c_dma_position_get_working(position, "AAPL", &working_long, &working_short) == POSITION_OK);
    CHECK(working_short == 0.);

    CHECK(c_dma_launch_order_unchecked(access, canceled_order) == DMA_OK);
    CHECK(c_dma_position_get_working(position, "AAPL", &working_long, &working_short) == POSITION_OK);
    CHECK(working_short == 5.);

    md_trade_instruction canceled = *canceled_order;
    canceled.order_state = STATE_CANCELED;
    CHECK(c_dma_algo_engine_on_order(engine, &canceled) == DMA_ALGO_OK);
    CHECK(position_ctx.canceled_hits == 1);
    CHECK(algo_ctx.canceled_hits == 1);
    CHECK(c_dma_position_get_working(position, "AAPL", &working_long, &working_short) == POSITION_OK);
    CHECK(working_short == 0.);

    // A rejected order does the same.
    md_trade_instruction* rejected_order = c_dma_algo_engine_new_order(engine, algo_b, 10., 5.);
    CHECK(rejected_order != NULL);
    CHECK(c_dma_position_get_working(position, "AAPL", &working_long, &working_short) == POSITION_OK);
    CHECK(working_short == 0.);

    CHECK(c_dma_launch_order_unchecked(access, rejected_order) == DMA_OK);
    CHECK(c_dma_position_get_working(position, "AAPL", &working_long, &working_short) == POSITION_OK);
    CHECK(working_short == 5.);

    md_trade_instruction rejected = *rejected_order;
    rejected.order_state = STATE_REJECTED;
    CHECK(c_dma_algo_engine_on_order(engine, &rejected) == DMA_ALGO_OK);
    CHECK(position_ctx.rejected_hits == 1);
    CHECK(algo_ctx.rejected_hits == 1);
    CHECK(c_dma_position_get_working(position, "AAPL", &working_long, &working_short) == POSITION_OK);
    CHECK(working_short == 0.);

    // === Completed algos leave the active set ===
    CHECK(c_dma_position_n_working_algos(position) == 2);
    CHECK(c_dma_algo_set_status(algo_a, DMA_ALGO_STATUS_DONE) == DMA_ALGO_OK);

    // A terminal update moves the completed record out of the stop sweep.
    md_trade_instruction* done_order = c_dma_algo_engine_new_order(engine, algo_a, 10., 1.);
    CHECK(done_order != NULL);
    CHECK(c_dma_position_n_working_algos(position) == 2);

    done_order->order_state = STATE_CANCELED;
    CHECK(c_dma_algo_engine_on_order_canceled_inplace(engine, (dma_order_ctx*) done_order) == DMA_ALGO_OK);
    CHECK(c_dma_position_n_working_algos(position) == 1);
    CHECK(c_bytemap_ex_len(position->working_algos) == 1);

    algo_ctx.stop_hits = 0;
    CHECK(c_dma_position_fully_stop(position) == POSITION_OK);
    CHECK(algo_ctx.stop_hits == 1);

    // === Unwind resolves the configured kind, not the position default ===
    CHECK(c_dma_position_fully_unwind(position) == POSITION_ERR_NOT_FOUND);
    CHECK(c_dma_algo_registry_reg(engine->registry, DMA_POSITION_DEFAULT_ALGO, test_algo_factory, 0) != NULL);
    CHECK(c_dma_position_fully_unwind(position) == POSITION_OK);
    CHECK(c_dma_position_n_algos(position) == 3);
    CHECK(c_bytemap_ex_len(engine->position) == 4);
    CHECK(c_dma_position_n_working_algos(position) == 2);

    CHECK(c_dma_position_fully_stop(NULL) == POSITION_ERR_INVALID_INPUT);
    CHECK(c_dma_position_fully_unwind(NULL) == POSITION_ERR_INVALID_INPUT);
    CHECK(c_dma_position_clear(NULL) == POSITION_ERR_INVALID_INPUT);

    CHECK(c_dma_position_unregister_callback(position, callback_id) == POSITION_OK);
    CHECK(c_dma_position_unregister_callback(position, callback_id) == POSITION_ERR_NOT_FOUND);

    // A synthetic exposure writes the ticker cache without an order.
    char* expo_stored = NULL;

    CHECK(c_dma_position_add_exposure(position, "EXPO", 100., 1050., 2.5, SIDE_LONG_OPEN) == POSITION_OK);
    CHECK(c_dma_position_get_exposure(position, "EXPO") == 100.);
    CHECK(c_dma_position_notional(position, "EXPO") == 1050.);
    CHECK(c_bytemap_ex_get_ptr(position->tickers, "EXPO", 4, &expo_stored, NULL) == BYTEMAP_OK);
    CHECK(((dma_position_ticker*) expo_stored)->multiplier == 2.5);
    CHECK(c_dma_position_add_exposure(NULL, "EXPO", 100., 1050., 1., SIDE_LONG_OPEN) == POSITION_ERR_INVALID_INPUT);
    CHECK(c_dma_position_add_exposure(position, NULL, 100., 1050., 1., SIDE_LONG_OPEN) == POSITION_ERR_INVALID_INPUT);

    // === Pop and clear ===
    CHECK(c_dma_position_pop_algo(position, algo_b->algo_id) == POSITION_OK);
    CHECK(c_dma_position_pop_algo(position, algo_b->algo_id) == POSITION_ERR_NOT_FOUND);
    CHECK(c_bytemap_ex_len(position->algos) == 2);
    CHECK(c_bytemap_ex_len(engine->position) == 3);

    CHECK(c_dma_position_clear(position) == POSITION_OK);
    CHECK(c_bytemap_ex_len(position->algos) == 0);
    CHECK(c_bytemap_ex_len(position->working_algos) == 0);
    CHECK(c_bytemap_ex_len(position->tickers) == 0);
    CHECK(c_bytemap_ex_len(engine->position) == 1);  // Only the other position's record remains.
    CHECK(position->callbacks != NULL);              // Clear keeps the callbacks.

    c_dma_position_free(position);
    CHECK(c_bytemap_ex_len(engine->position) == 1);
    CHECK(c_bytemap_ex_len(access->positions) == 1);

    c_dma_position_free(other);
    CHECK(c_bytemap_ex_len(engine->position) == 0);
    CHECK(c_bytemap_ex_len(access->positions) == 0);

    // The engine owns the algo records and releases them with the engine.
    CHECK(c_bytemap_ex_len(engine->algo) == 4);
    c_dma_free(access);
}

// ========== Position: Partial-Fill Flow ==========

/**
 * A 1000-volume order fills 100 at a time; each fill is followed by an exchange
 * order-state event carrying the cumulative fill, and the order is then canceled.
 * The working cache must fall by exactly the filled volume, whatever state the
 * exchange reports, and the cancel must release only the remainder.
 */
static void test_position_partial_fill_flow(void) {

    dma* access = c_dma_new(NULL, g_allocator);
    CHECK(access != NULL);

    dma_algo_engine* engine = access->ALGO_ENGINE;
    CHECK(engine != NULL);

    mds mds_state;
    memset(&mds_state, 0, sizeof(mds_state));
    mds_state.ts.unix_ts     = 1.;
    mds_state.subscriptions  = c_bytemap_ex_new(8, sizeof(mds_subscription), g_allocator);
    CHECK(mds_state.subscriptions != NULL);

    mds_subscription subscription;
    memset(&subscription, 0, sizeof(subscription));
    subscription.last_price = 10.;
    CHECK(c_bytemap_ex_set(mds_state.subscriptions, "AAPL", 4, (const char*) &subscription, sizeof(subscription), 0, NULL) == BYTEMAP_OK);

    access->MDS         = &mds_state;

    dma_algo_ctx* registration = c_dma_algo_registry_reg(engine->registry, "partial", test_algo_factory, 0);
    CHECK(registration != NULL);

    dma_position* position = c_dma_position_new(access, engine, "partial", g_allocator);
    CHECK(position != NULL);

    test_position_ctx position_ctx;
    memset(&position_ctx, 0, sizeof(position_ctx));

    CHECK(c_dma_position_register_callback(position, on_filled, (void*) test_position_on_filled, &position_ctx, NULL) == POSITION_OK);
    CHECK(c_dma_position_register_callback(position, on_canceled, (void*) test_position_on_canceled, &position_ctx, NULL) == POSITION_OK);

    dma_algo* algo = c_dma_new_algo(access, registration->reg_id, "AAPL", SIDE_LONG_OPEN, 1000., NULL, g_allocator);
    CHECK(algo != NULL);
    CHECK(c_dma_position_add_algo(position, algo) == POSITION_OK);

    md_trade_instruction* order = c_dma_algo_engine_new_order(engine, algo, 10., 1000.);
    CHECK(order != NULL);

    char* order_stored = NULL;
    CHECK(c_bytemap_ex_get_ptr(engine->orders, (const char*) &order->order_id, sizeof(long_md_id), &order_stored, NULL) == BYTEMAP_OK);

    dma_order_ctx* order_ctx = *(dma_order_ctx**) order_stored;

    // The order leaves IDLE through the sent entry; the fills accept it.
    order->order_state = STATE_SENT;
    CHECK(c_dma_algo_engine_on_order_launched_inplace(engine, order_ctx) == DMA_ALGO_OK);

    CHECK((order_ctx->flags & DMA_ORDER_LISTED) != 0);
    CHECK((order_ctx->flags & DMA_ORDER_CANCELED) == 0);

    double working_long  = 0.;
    double working_short = 0.;

    CHECK(c_dma_position_get_working(position, "AAPL", &working_long, &working_short) == POSITION_OK);
    CHECK(working_long == 1000.);
    CHECK(working_short == 0.);
    CHECK(c_dma_position_get_exposure(position, "AAPL") == 0.);
    CHECK(c_dma_position_notional(position, "AAPL") == 0.);
    CHECK(algo->working_volume == 1000.);

    // A placed order update is a working state; the volume is already in the
    // book and must not be counted again.
    md_trade_instruction placed;
    memcpy(&placed, order, sizeof(placed));
    placed.order_state   = STATE_PLACED;
    placed.filled_volume = 0.;
    CHECK(c_dma_algo_engine_on_order(engine, &placed) == DMA_ALGO_OK);
    CHECK(c_dma_position_get_working(position, "AAPL", &working_long, &working_short) == POSITION_OK);
    CHECK(working_long == 1000.);
    CHECK(algo->working_volume == 1000.);

    double cumulative = 0.;

    for (size_t i = 1; i <= 3; ++i) {
        md_trade_report fill;
        memset(&fill, 0, sizeof(fill));
        fill.meta_info.dt.unix_ts = 1. + (double) i;
        fill.multiplier           = 1.;
        fill.order_id             = order->order_id;
        fill.volume               = 100.;
        fill.notional             = 1000.;
        fill.side                 = SIDE_LONG_OPEN;
        snprintf(fill.trade_id.data, sizeof(fill.trade_id.data), "partial-%zu", i);

        CHECK(c_dma_algo_engine_on_trade(engine, &fill) == DMA_ALGO_OK);
        cumulative += 100.;

        CHECK(position_ctx.filled_hits == i);
        CHECK(c_dma_position_get_working(position, "AAPL", &working_long, &working_short) == POSITION_OK);
        CHECK(working_long == 1000. - cumulative);
        CHECK(working_short == 0.);
        CHECK(c_dma_position_get_exposure(position, "AAPL") == cumulative);
        CHECK(c_dma_position_notional(position, "AAPL") == cumulative * 10.);
        CHECK(c_dma_position_pnl(position, "AAPL") == 0.);  // Marked at the fill price.

        // The algo's own aggregates follow the same deltas.
        CHECK(algo->working_volume == 1000. - cumulative);
        CHECK(algo->filled_volume == cumulative);
        CHECK(algo->exposure_volume == cumulative);
        CHECK(algo->filled_notional == cumulative * 10.);
        CHECK(algo->cash_flow == -cumulative * 10.);

        // A repeated fill report is deduped by trade id: nothing moves.
        CHECK(c_dma_algo_engine_on_trade(engine, &fill) == DMA_ALGO_OK);
        CHECK(c_dma_position_get_working(position, "AAPL", &working_long, &working_short) == POSITION_OK);
        CHECK(working_long == 1000. - cumulative);
        CHECK(algo->filled_volume == cumulative);

        // The exchange's order-state event repeats the cumulative fill. Whether it
        // reports the partial state or repeats the sent state, and however
        // often it repeats, the working cache must not move: the fill was
        // already applied above.
        md_trade_instruction echo;
        memcpy(&echo, order, sizeof(echo));

        for (size_t repeat = 0; repeat < 2; ++repeat) {
            echo.order_state   = STATE_PARTFILLED;
            echo.filled_volume = cumulative;
            CHECK(c_dma_algo_engine_on_order(engine, &echo) == DMA_ALGO_OK);
            CHECK(c_dma_position_get_working(position, "AAPL", &working_long, &working_short) == POSITION_OK);
            CHECK(working_long == 1000. - cumulative);

            echo.order_state   = STATE_SENT;
            echo.filled_volume = cumulative;
            CHECK(c_dma_algo_engine_on_order(engine, &echo) == DMA_ALGO_OK);
            CHECK(c_dma_position_get_working(position, "AAPL", &working_long, &working_short) == POSITION_OK);
            CHECK(working_long == 1000. - cumulative);
        }
    }

    // The cancel carries the cumulative fill and releases only the remainder.
    md_trade_instruction canceled;
    memcpy(&canceled, order, sizeof(canceled));
    canceled.order_state   = STATE_CANCELED;
    canceled.filled_volume = cumulative;

    CHECK(c_dma_algo_engine_on_order(engine, &canceled) == DMA_ALGO_OK);
    CHECK((order_ctx->flags & DMA_ORDER_CANCELED) != 0);
    CHECK(position_ctx.canceled_hits == 1);
    CHECK(c_dma_position_get_working(position, "AAPL", &working_long, &working_short) == POSITION_OK);
    CHECK(working_long == 0.);
    CHECK(working_short == 0.);
    CHECK(c_dma_position_get_exposure(position, "AAPL") == 300.);
    CHECK(c_dma_position_notional(position, "AAPL") == 3000.);
    CHECK(c_dma_position_pnl(position, "AAPL") == 0.);
    CHECK(algo->working_volume == 0.);
    CHECK(algo->filled_volume == 300.);
    CHECK(algo->exposure_volume == 300.);
    CHECK(algo->filled_notional == 3000.);
    CHECK(algo->cash_flow == -3000.);

    // The order left the live set: a late update has nothing to apply.
    CHECK(c_dma_algo_engine_on_order(engine, &canceled) == DMA_ALGO_ERR_NOT_FOUND);
    CHECK(position_ctx.canceled_hits == 1);

    // A late stop update is out of the live set, so the flags cannot be
    // applied twice through the engine path.
    CHECK((order_ctx->flags & DMA_ORDER_LISTED) != 0);
    CHECK((order_ctx->flags & DMA_ORDER_CANCELED) != 0);

    // === A cancel that arrives as a trade report ===
    // The exchange confirms the cancel through a trade report; the release
    // happens once even when the confirmation is repeated.
    md_trade_instruction* late_order = c_dma_algo_engine_new_order(engine, algo, 10., 500.);
    CHECK(late_order != NULL);

    order_stored = NULL;
    CHECK(c_bytemap_ex_get_ptr(engine->orders, (const char*) &late_order->order_id, sizeof(long_md_id), &order_stored, NULL) == BYTEMAP_OK);

    dma_order_ctx* late_ctx = *(dma_order_ctx**) order_stored;

    // The order leaves IDLE through the sent entry; the fills accept it.
    late_order->order_state = STATE_SENT;
    CHECK(c_dma_algo_engine_on_order_launched_inplace(engine, late_ctx) == DMA_ALGO_OK);

    CHECK((late_ctx->flags & DMA_ORDER_LISTED) != 0);
    CHECK((late_ctx->flags & DMA_ORDER_CANCELED) == 0);
    CHECK(c_dma_position_get_working(position, "AAPL", &working_long, &working_short) == POSITION_OK);
    CHECK(working_long == 500.);

    md_trade_report partial;
    memset(&partial, 0, sizeof(partial));
    partial.meta_info.dt.unix_ts = 10.;
    partial.multiplier           = 1.;
    partial.order_id             = late_order->order_id;
    partial.volume               = 100.;
    partial.notional             = 1000.;
    partial.side                 = SIDE_LONG_OPEN;
    strcpy(partial.trade_id.data, "late-1");
    CHECK(c_dma_algo_engine_on_trade(engine, &partial) == DMA_ALGO_OK);
    CHECK(c_dma_position_get_working(position, "AAPL", &working_long, &working_short) == POSITION_OK);
    CHECK(working_long == 400.);

    // An order-state event moves the order to canceling; the working cache does not move.
    md_trade_instruction canceling;
    memcpy(&canceling, late_order, sizeof(canceling));
    canceling.order_state = STATE_CANCELING;
    CHECK(c_dma_algo_engine_on_order(engine, &canceling) == DMA_ALGO_OK);
    CHECK(c_dma_position_get_working(position, "AAPL", &working_long, &working_short) == POSITION_OK);
    CHECK(working_long == 400.);

    md_trade_report cancel_confirm;
    memset(&cancel_confirm, 0, sizeof(cancel_confirm));
    cancel_confirm.meta_info.dt.unix_ts = 11.;
    cancel_confirm.order_id             = late_order->order_id;
    cancel_confirm.side                 = SIDE_CANCEL;
    cancel_confirm.volume               = 400.;  // The canceled amount; not booked as a fill.
    strcpy(cancel_confirm.trade_id.data, "cancel-late");

    CHECK(c_dma_algo_engine_on_trade(engine, &cancel_confirm) == DMA_ALGO_OK);
    CHECK((late_ctx->flags & DMA_ORDER_CANCELED) != 0);
    CHECK(position_ctx.canceled_hits == 2);
    CHECK(c_dma_position_get_working(position, "AAPL", &working_long, &working_short) == POSITION_OK);
    CHECK(working_long == 0.);
    CHECK(algo->working_volume == 0.);

    // The duplicate confirmation is a tolerated no-op on the canceled order.
    CHECK(c_dma_algo_engine_on_trade(engine, &cancel_confirm) == DMA_ALGO_OK);
    CHECK(position_ctx.canceled_hits == 2);
    CHECK(c_dma_position_get_working(position, "AAPL", &working_long, &working_short) == POSITION_OK);
    CHECK(working_long == 0.);

    // A late fill on the canceled order runs the fill flow: the volume, the
    // callbacks and the exposure move; only the post-state stays CANCELED.
    md_trade_report late_fill;
    memset(&late_fill, 0, sizeof(late_fill));
    late_fill.meta_info.dt.unix_ts = 12.;
    late_fill.multiplier           = 1.;
    late_fill.order_id             = late_order->order_id;
    late_fill.volume               = 50.;
    late_fill.notional             = 500.;
    late_fill.side                 = SIDE_LONG_OPEN;
    strcpy(late_fill.trade_id.data, "late-2");

    CHECK(c_dma_algo_engine_on_trade(engine, &late_fill) == DMA_ALGO_OK);
    CHECK(late_order->order_state == STATE_CANCELED);
    CHECK(late_order->filled_volume == 150.);
    CHECK(position_ctx.filled_hits == 5);
    CHECK(c_dma_position_get_working(position, "AAPL", &working_long, &working_short) == POSITION_OK);
    CHECK(working_long == 0.);
    CHECK(algo->working_volume == 0.);
    CHECK(algo->filled_volume == 450.);

    CHECK(c_dma_position_get_exposure(position, "AAPL") == 450.);
    CHECK(c_dma_position_notional(position, "AAPL") == 4500.);
    CHECK(c_dma_position_pnl(position, "AAPL") == 0.);

    c_dma_free(access);
    c_bytemap_ex_free(mds_state.subscriptions);
}

// ========== Access In-Place Application ==========

static void test_access_inplace_updates(void) {

    mds mds_state;
    memset(&mds_state, 0, sizeof(mds_state));
    mds_state.ts.unix_ts = 1.;

    dma* access = c_dma_new(&mds_state, g_allocator);
    CHECK(access != NULL);

    dma_algo_engine* engine = access->ALGO_ENGINE;
    CHECK(engine != NULL);

    dma_risk_profile* risk = access->RISK_PROFILE;
    set_full_ticker_rules(risk, "AAPL", 20., 20., 20., 20., 200., 200.);
    set_full_globals(risk, INFINITY, INFINITY, INFINITY, INFINITY);

    test_interface interface;
    memset(&interface, 0, sizeof(interface));
    CHECK(c_dma_attach_protocol_handler(access, test_on_launch, test_on_cancel, NULL, &interface) == DMA_OK);
    CHECK(c_dma_attach_protocol_callback(access, test_on_reject, test_on_filled) == DMA_OK);

    dma_algo_ctx* registration = c_dma_algo_registry_reg(engine->registry, "inplace", test_algo_factory, 0);
    CHECK(registration != NULL);

    dma_position* position = c_dma_position_new(access, engine, "inplace", g_allocator);
    CHECK(position != NULL);

    dma_algo* algo = c_dma_new_algo(access, registration->reg_id, "AAPL", SIDE_LONG_OPEN, 20., NULL, g_allocator);
    CHECK(algo != NULL);
    CHECK(c_dma_position_add_algo(position, algo) == POSITION_OK);

    test_algo_ctx     algo_ctx;
    test_position_ctx position_ctx;

    memset(&algo_ctx, 0, sizeof(algo_ctx));
    memset(&position_ctx, 0, sizeof(position_ctx));

    algo->callbacks.on_canceled = test_algo_on_canceled;
    algo->callbacks.on_rejected = test_algo_on_rejected;
    algo->callbacks.user_data   = &algo_ctx;

    CHECK(c_dma_position_register_callback(position, on_canceled, (void*) test_position_on_canceled, &position_ctx, NULL) == POSITION_OK);
    CHECK(c_dma_position_register_callback(position, on_rejected, (void*) test_position_on_rejected, &position_ctx, NULL) == POSITION_OK);

    double working_long  = 0.;
    double working_short = 0.;

    // A passing launch opens the remaining volume on the first working state
    // and applies the written state in place; the order stays live.
    md_trade_instruction* order = c_dma_algo_engine_new_order(engine, algo, 10., 10.);
    CHECK(order != NULL);
    CHECK(((dma_order_ctx*) order)->flags == 0);
    CHECK(algo->working_volume == 0.);

    dma_order_ctx* order_ctx = (dma_order_ctx*) order;  // The instruction starts its context.

#if DMA_NO_QUEUE_MODE
    CHECK(c_dma_launch_order(access, order) == DMA_OK);
#else
    CHECK(c_dma_launch_order(access, order) == DMA_OK_BUFFERED);
    CHECK(order->order_state == STATE_PENDING);
    CHECK(algo->working_volume == 10.);  // The pending entry opened the volume.
    CHECK(c_dma_process_queue(access) == 1);
#endif
    CHECK(order->order_state == STATE_SENT);
    CHECK(interface.n_launch == 1);
    CHECK(algo->working_volume == 10.);
    CHECK((order_ctx->flags & DMA_ORDER_LISTED) != 0);
    CHECK(c_dma_algo_get_working_order(algo, &order->order_id) == order);

    // A refused launch that never reached a working state is never deducted:
    // the rejection reaches the algo and the host, the working volume is the
    // live order's alone.
    md_trade_instruction* refused = c_dma_algo_engine_new_order(engine, algo, 10., 30.);
    CHECK(refused != NULL);
    CHECK(((dma_order_ctx*) refused)->flags == 0);
    CHECK(algo->working_volume == 10.);

    dma_order_ctx* refused_ctx = (dma_order_ctx*) refused;

#if DMA_NO_QUEUE_MODE
    CHECK(c_dma_launch_order(access, refused) == DMA_ERR_RISK);
    CHECK((refused_ctx->flags & DMA_ORDER_LISTED) == 0);  // The volume was never opened.
#else
    CHECK(c_dma_launch_order(access, refused) == DMA_OK_BUFFERED);
    CHECK(algo->working_volume == 40.);  // The pending entry opened the volume.
    CHECK((refused_ctx->flags & DMA_ORDER_LISTED) != 0);
    CHECK(c_dma_process_queue(access) == 1);
#endif
    CHECK(refused->order_state == STATE_REJECTED);
    CHECK(interface.n_reject == 1);
    CHECK(algo->working_volume == 10.);
    CHECK((refused_ctx->flags & DMA_ORDER_REJECTED) != 0);
    CHECK(c_dma_algo_get_working_order(algo, &refused->order_id) == NULL);
    CHECK(c_bytemap_ex_len(engine->working_orders_by_id) == 1);
    CHECK(algo_ctx.rejected_hits == 1);
    CHECK(position_ctx.rejected_hits == 1);
    CHECK(c_dma_position_get_working(position, "AAPL", &working_long, &working_short) == POSITION_OK);
    CHECK(working_long == 10.);

    // A cancel moves the live order to canceling in place; the volume stays.
#if DMA_NO_QUEUE_MODE
    CHECK(c_dma_cancel_order(access, order) == DMA_OK);
#else
    CHECK(c_dma_cancel_order(access, order) == DMA_OK_BUFFERED);
    CHECK(c_dma_process_queue(access) == 1);
#endif
    CHECK(order->order_state == STATE_CANCELING);
    CHECK(interface.n_cancel == 1);
    CHECK(algo->working_volume == 10.);
    CHECK(c_dma_algo_get_working_order(algo, &order->order_id) == order);

    // The exchange confirms; the completion releases the remaining volume once.
    md_trade_instruction canceled = *order;
    canceled.order_state = STATE_CANCELED;
    CHECK(c_dma_algo_engine_on_order(engine, &canceled) == DMA_ALGO_OK);
    CHECK(order->order_state == STATE_CANCELED);
    CHECK(algo->working_volume == 0.);
    CHECK(algo_ctx.canceled_hits == 1);
    CHECK(position_ctx.canceled_hits == 1);
    CHECK(c_dma_position_get_working(position, "AAPL", &working_long, &working_short) == POSITION_OK);
    CHECK(working_long == 0.);
    CHECK(c_bytemap_ex_len(engine->working_orders_by_id) == 0);

    c_dma_free(access);
}

// ========== Algo In-Place Entries ==========

static void test_algo_inplace_entries(void) {

    mds mds_state;
    memset(&mds_state, 0, sizeof(mds_state));
    mds_state.ts.unix_ts = 1.;

    dma* access = c_dma_new(&mds_state, g_allocator);
    CHECK(access != NULL);

    dma_algo_engine* engine = access->ALGO_ENGINE;
    CHECK(engine != NULL);

    dma_algo_ctx* registration = c_dma_algo_registry_reg(engine->registry, "inplace", test_algo_factory, 0);
    CHECK(registration != NULL);

    dma_position* position = c_dma_position_new(access, engine, "inplace", g_allocator);
    CHECK(position != NULL);

    dma_algo* algo = c_dma_new_algo(access, registration->reg_id, "AAPL", SIDE_LONG_OPEN, 20., NULL, g_allocator);
    CHECK(algo != NULL);
    CHECK(c_dma_position_add_algo(position, algo) == POSITION_OK);

    test_algo_ctx     algo_ctx;
    test_position_ctx position_ctx;

    memset(&algo_ctx, 0, sizeof(algo_ctx));
    memset(&position_ctx, 0, sizeof(position_ctx));

    algo->callbacks.on_listed   = test_algo_on_listed;
    algo->callbacks.on_canceled = test_algo_on_canceled;
    algo->callbacks.on_rejected = test_algo_on_rejected;
    algo->callbacks.user_data   = &algo_ctx;

    CHECK(c_dma_position_register_callback(position, on_listed, (void*) test_position_on_listed, &position_ctx, NULL) == POSITION_OK);
    CHECK(c_dma_position_register_callback(position, on_canceled, (void*) test_position_on_canceled, &position_ctx, NULL) == POSITION_OK);
    CHECK(c_dma_position_register_callback(position, on_rejected, (void*) test_position_on_rejected, &position_ctx, NULL) == POSITION_OK);

    double working_long  = 0.;
    double working_short = 0.;

    // === PENDING / CANCELED ===
    md_trade_instruction* pending = c_dma_algo_engine_new_order(engine, algo, 10., 10.);
    CHECK(pending != NULL);
    CHECK(((dma_order_ctx*) pending)->flags == 0);
    CHECK(algo->working_volume == 0.);

    dma_order_ctx* pending_ctx = (dma_order_ctx*) pending;

    pending->order_state = STATE_PENDING;
    CHECK(c_dma_algo_engine_on_order_pending_inplace(engine, pending_ctx) == DMA_ALGO_OK);
    CHECK(pending->order_state == STATE_PENDING);
    CHECK(algo->working_volume == 10.);
    CHECK(pending_ctx->flags == DMA_ORDER_LISTED);
    CHECK(c_dma_algo_get_working_order(algo, &pending->order_id) == pending);
    CHECK(c_bytemap_ex_len(position->working_algos) == 1);
    CHECK(c_dma_position_get_working(position, "AAPL", &working_long, &working_short) == POSITION_OK);
    CHECK(working_long == 10.);

    pending->order_state = STATE_CANCELED;
    CHECK(c_dma_algo_engine_on_order_canceled_inplace(engine, pending_ctx) == DMA_ALGO_OK);
    CHECK(algo->working_volume == 0.);
    CHECK((pending_ctx->flags & DMA_ORDER_CANCELED) != 0);
    CHECK(c_dma_algo_get_working_order(algo, &pending->order_id) == NULL);
    CHECK(algo_ctx.canceled_hits == 1);
    CHECK(position_ctx.canceled_hits == 1);
    CHECK(c_dma_position_get_working(position, "AAPL", &working_long, &working_short) == POSITION_OK);
    CHECK(working_long == 0.);

    memset(&algo_ctx, 0, sizeof(algo_ctx));
    memset(&position_ctx, 0, sizeof(position_ctx));

    // === SENT / REJECTED ===
    md_trade_instruction* launched = c_dma_algo_engine_new_order(engine, algo, 10., 10.);
    CHECK(launched != NULL);
    CHECK(((dma_order_ctx*) launched)->flags == 0);
    CHECK(algo->working_volume == 0.);

    dma_order_ctx* launched_ctx = (dma_order_ctx*) launched;

    launched->order_state = STATE_SENT;
    CHECK(c_dma_algo_engine_on_order_launched_inplace(engine, launched_ctx) == DMA_ALGO_OK);
    CHECK(algo->working_volume == 10.);
    CHECK(c_dma_algo_get_working_order(algo, &launched->order_id) == launched);
    CHECK(c_bytemap_ex_len(position->working_algos) == 1);
    CHECK(c_dma_position_get_working(position, "AAPL", &working_long, &working_short) == POSITION_OK);
    CHECK(working_long == 10.);

    launched->order_state = STATE_REJECTED;
    CHECK(c_dma_algo_engine_on_order_rejected_inplace(engine, launched_ctx) == DMA_ALGO_OK);
    CHECK(algo->working_volume == 0.);
    CHECK((launched_ctx->flags & DMA_ORDER_REJECTED) != 0);
    CHECK(c_dma_algo_get_working_order(algo, &launched->order_id) == NULL);
    CHECK(algo_ctx.rejected_hits == 1);
    CHECK(position_ctx.rejected_hits == 1);
    CHECK(c_dma_position_get_working(position, "AAPL", &working_long, &working_short) == POSITION_OK);
    CHECK(working_long == 0.);

    memset(&algo_ctx, 0, sizeof(algo_ctx));
    memset(&position_ctx, 0, sizeof(position_ctx));

    // === PLACED / CANCELING / CANCELED ===
    md_trade_instruction* placed = c_dma_algo_engine_new_order(engine, algo, 10., 10.);
    CHECK(placed != NULL);
    CHECK(((dma_order_ctx*) placed)->flags == 0);
    CHECK(algo->working_volume == 0.);

    dma_order_ctx* placed_ctx = (dma_order_ctx*) placed;

    // A placed order is working: the sent entry opened its volume first.
    placed->order_state = STATE_SENT;
    CHECK(c_dma_algo_engine_on_order_launched_inplace(engine, placed_ctx) == DMA_ALGO_OK);
    CHECK(algo->working_volume == 10.);

    placed->order_state = STATE_PLACED;
    CHECK(c_dma_algo_engine_on_order_placed_inplace(engine, placed_ctx) == DMA_ALGO_OK);
    CHECK(algo->working_volume == 10.);
    CHECK(algo_ctx.listed_hits == 1);
    CHECK(position_ctx.listed_hits == 1);
    CHECK(c_dma_algo_get_working_order(algo, &placed->order_id) == placed);
    CHECK(c_dma_position_get_working(position, "AAPL", &working_long, &working_short) == POSITION_OK);
    CHECK(working_long == 10.);

    placed->order_state = STATE_CANCELING;
    CHECK(c_dma_algo_engine_on_order_canceling_inplace(engine, placed_ctx) == DMA_ALGO_OK);
    CHECK(algo->working_volume == 10.);
    CHECK(algo_ctx.canceled_hits == 0);

    placed->order_state = STATE_CANCELED;
    CHECK(c_dma_algo_engine_on_order_canceled_inplace(engine, placed_ctx) == DMA_ALGO_OK);
    CHECK(algo->working_volume == 0.);
    CHECK(algo_ctx.canceled_hits == 1);
    CHECK(position_ctx.canceled_hits == 1);

    memset(&algo_ctx, 0, sizeof(algo_ctx));
    memset(&position_ctx, 0, sizeof(position_ctx));

    // === PARTFILLED / FILLED ===
    md_trade_instruction* partial = c_dma_algo_engine_new_order(engine, algo, 10., 10.);
    CHECK(partial != NULL);
    CHECK(((dma_order_ctx*) partial)->flags == 0);
    CHECK(algo->working_volume == 0.);

    dma_order_ctx* partial_ctx = (dma_order_ctx*) partial;

    // A partially filled order is working: the sent entry opened its volume.
    partial->order_state = STATE_SENT;
    CHECK(c_dma_algo_engine_on_order_launched_inplace(engine, partial_ctx) == DMA_ALGO_OK);
    CHECK(algo->working_volume == 10.);

    partial->order_state = STATE_PARTFILLED;
    CHECK(c_dma_algo_engine_on_order_partially_filled_inplace(engine, partial_ctx) == DMA_ALGO_OK);
    CHECK(algo->working_volume == 10.);
    CHECK(c_dma_algo_get_working_order(algo, &partial->order_id) == partial);

    // The fill report releases the filled volume; the filled state then leaves
    // the working index without touching the volume.
    md_trade_report fill;
    memset(&fill, 0, sizeof(fill));
    fill.meta_info.dt.unix_ts = 2.;
    fill.multiplier           = 1.;
    fill.order_id             = partial->order_id;
    fill.volume               = 4.;
    fill.notional             = 40.;
    fill.side                 = SIDE_LONG_OPEN;
    strcpy(fill.trade_id.data, "inplace-fill");

    CHECK(c_dma_algo_engine_on_trade(engine, &fill) == DMA_ALGO_OK);
    CHECK(algo->working_volume == 6.);

    partial->order_state = STATE_FILLED;
    CHECK(c_dma_algo_engine_on_order_filled_inplace(engine, partial_ctx) == DMA_ALGO_OK);
    CHECK(algo->working_volume == 6.);
    CHECK(c_dma_algo_get_working_order(algo, &partial->order_id) == NULL);

    md_trade_report rest;
    memset(&rest, 0, sizeof(rest));
    rest.meta_info.dt.unix_ts = 3.;
    rest.multiplier           = 1.;
    rest.order_id             = partial->order_id;
    rest.volume               = 6.;
    rest.notional             = 60.;
    rest.side                 = SIDE_LONG_OPEN;
    strcpy(rest.trade_id.data, "inplace-fill-rest");

    CHECK(c_dma_algo_engine_on_trade(engine, &rest) == DMA_ALGO_OK);
    CHECK(algo->working_volume == 0.);
    CHECK(c_dma_position_get_working(position, "AAPL", &working_long, &working_short) == POSITION_OK);
    CHECK(working_long == 0.);

    memset(&algo_ctx, 0, sizeof(algo_ctx));
    memset(&position_ctx, 0, sizeof(position_ctx));

    // === INVALID ===
    md_trade_instruction* invalid = c_dma_algo_engine_new_order(engine, algo, 10., 10.);
    CHECK(invalid != NULL);
    CHECK(((dma_order_ctx*) invalid)->flags == 0);

    dma_order_ctx* invalid_ctx = (dma_order_ctx*) invalid;

    invalid->order_state = STATE_INVALID;
    CHECK(c_dma_algo_engine_on_order_invalid_inplace(engine, invalid_ctx) == DMA_ALGO_OK);
    CHECK(algo->working_volume == 0.);  // Never opened, so never released.
    CHECK((invalid_ctx->flags & DMA_ORDER_ERROR) != 0);
    CHECK((invalid_ctx->flags & DMA_ORDER_LISTED) == 0);
    CHECK(c_dma_algo_get_working_order(algo, &invalid->order_id) == NULL);
    CHECK(algo_ctx.rejected_hits == 1);
    CHECK(position_ctx.rejected_hits == 1);

    memset(&algo_ctx, 0, sizeof(algo_ctx));
    memset(&position_ctx, 0, sizeof(position_ctx));

    // === The dispatcher routes a handled state and refuses an unhandled one ===
    md_trade_instruction* routed = c_dma_algo_engine_new_order(engine, algo, 10., 10.);
    CHECK(routed != NULL);
    CHECK(((dma_order_ctx*) routed)->flags == 0);

    dma_order_ctx* routed_ctx = (dma_order_ctx*) routed;

    routed->order_state = STATE_SENT;
    CHECK(c_dma_algo_engine_on_order_inplace(engine, routed_ctx) == DMA_ALGO_OK);
    CHECK(algo->working_volume == 10.);

    routed->order_state = STATE_IDLE;
    CHECK(c_dma_algo_engine_on_order_inplace(engine, routed_ctx) == DMA_ALGO_ERR_INVALID_INPUT);
    CHECK(algo->working_volume == 10.);
    CHECK(c_dma_algo_get_working_order(algo, &routed->order_id) == routed);

    routed->order_state = STATE_CANCELED;
    CHECK(c_dma_algo_engine_on_order_canceled_inplace(engine, routed_ctx) == DMA_ALGO_OK);
    CHECK(algo->working_volume == 0.);

    c_dma_free(access);
}

// ========== Position Hot-Path Benchmark ==========

static double bench_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double) ts.tv_sec + (double) ts.tv_nsec * 1e-9;
}

static void test_position_bench(void) {

    dma* access = c_dma_new(NULL, g_allocator);
    CHECK(access != NULL);

    dma_algo_engine* engine = access->ALGO_ENGINE;
    CHECK(engine != NULL);

    mds mds_state;
    memset(&mds_state, 0, sizeof(mds_state));
    mds_state.ts.unix_ts = 1.;

    access->MDS         = &mds_state;

    dma_algo_ctx* registration = c_dma_algo_registry_reg(engine->registry, "bench", test_algo_factory, 0);
    CHECK(registration != NULL);

    dma_position* position = c_dma_position_new(access, engine, "passive", g_allocator);
    CHECK(position != NULL);

    dma_algo* algo = c_dma_new_algo(access, registration->reg_id, "AAPL", SIDE_LONG_OPEN, 100., NULL, g_allocator);
    CHECK(algo != NULL);
    CHECK(c_dma_position_add_algo(position, algo) == POSITION_OK);

    md_trade_instruction* order = c_dma_algo_engine_new_order(engine, algo, 10., 100.);
    CHECK(order != NULL);
    order->order_state = STATE_PLACED;

    // The exchange's order-state event is its own buffer, never the stored order.
    md_trade_instruction event;
    memcpy(&event, order, sizeof(event));
    event.order_state = STATE_PLACED;

    size_t n_events = 200000;
    int    checksum = 0;

    double start = bench_now();

    for (size_t i = 0; i < n_events; ++i) checksum += c_dma_algo_engine_on_order(engine, &event);

    double elapsed = bench_now() - start;

    CHECK(checksum == 0);
    printf("[bench] c_dma_algo_engine_on_order (1 hosted position): %.1f ns/op (%zu events, %.3f ms)\n", elapsed * 1e9 / (double) n_events, n_events, elapsed * 1e3);

    c_dma_position_free(position);
    c_dma_free(access);
}

static void test_risk_bench(void) {
    dma* access = c_dma_new(NULL, g_allocator);
    CHECK(access != NULL);

    dma_algo_engine* engine = access->ALGO_ENGINE;
    CHECK(engine != NULL);

    dma_risk_profile* risk = access->RISK_PROFILE;
    set_full_ticker_rules(risk, "AAPL", 1e9, 1e9, 1e9, 1e9, 1e12, 1e12);
    set_full_globals(risk, INFINITY, INFINITY, INFINITY, INFINITY);

    size_t n_checks = 200000;
    size_t n_pass   = 0;

    double start = bench_now();

    for (size_t i = 0; i < n_checks; ++i) {
        dma_risk_violation violation = c_dma_risk_profile_check_limit(access, risk, "AAPL", SIDE_LONG_OPEN, 10., 10.);
        n_pass += violation.type == DMA_RISK_PASS;
    }

    double elapsed = bench_now() - start;

    CHECK(n_pass == n_checks);
    printf("[bench] c_dma_risk_profile_check_limit (1 ticker rule set): %.1f ns/op (%zu checks, %.3f ms)\n", elapsed * 1e9 / (double) n_checks, n_checks, elapsed * 1e3);

    c_dma_free(access);
}

static void test_dma_bench(void) {
    dma* access = c_dma_new(NULL, g_allocator);
    CHECK(access != NULL);

    dma_algo_engine* engine = access->ALGO_ENGINE;
    CHECK(engine != NULL);

    dma_risk_profile* risk = access->RISK_PROFILE;
    set_full_ticker_rules(risk, "AAPL", 1e9, 1e9, 1e9, 1e9, 1e12, 1e12);
    set_full_globals(risk, INFINITY, INFINITY, INFINITY, INFINITY);

    md_variant order = make_order("AAPL", 10., SIDE_LONG_OPEN);
    order.trade_instruction.limit_price = 10.;

    dma_risk_violation violation;
    size_t             n_checks = 500000;
    size_t             n_pass   = 0;

    double start = bench_now();

    for (size_t i = 0; i < n_checks; ++i) n_pass += c_dma_check_order(access, &order.trade_instruction, &violation) == DMA_OK;

    double elapsed = bench_now() - start;

    CHECK(n_pass == n_checks);
    printf("[bench] c_dma_check_order (1 ticker rule set): %.1f ns/op (%zu checks, %.3f ms)\n", elapsed * 1e9 / (double) n_checks, n_checks, elapsed * 1e3);

    c_dma_free(access);
}

static void test_algo_status_working_set(void) {

    dma* access = c_dma_new(NULL, g_allocator);
    CHECK(access != NULL);

    dma_algo_engine* engine = access->ALGO_ENGINE;
    CHECK(engine != NULL);

    mds mds_state;
    memset(&mds_state, 0, sizeof(mds_state));

    access->MDS        = &mds_state;

    dma_algo_ctx* registration = c_dma_algo_registry_reg(engine->registry, "working", test_algo_factory, 0);
    CHECK(registration != NULL);

    test_algo_ctx counters;
    memset(&counters, 0, sizeof(counters));

    md_variant observation;
    memset(&observation, 0, sizeof(observation));
    observation.meta_info.ticker = "AAPL";

    // A new record starts in IDLE and is already on its ticker's working list.
    dma_algo* first = c_dma_new_algo(access, registration->reg_id, "AAPL", SIDE_LONG_OPEN, 10., NULL, g_allocator);
    CHECK(first != NULL);
    CHECK(first->status == DMA_ALGO_STATUS_IDLE);
    CHECK(c_bytemap_ex_len(engine->working_algo) == 1);

    first->callbacks.on_md     = test_algo_on_md;
    first->callbacks.user_data = &counters;

    c_dma_algo_engine_on_md(engine, &observation);
    CHECK(counters.md_hits == 1);  // An IDLE algo receives the observation.

    // PREPARING and READY stay in the working set.
    CHECK(c_dma_algo_set_status(first, DMA_ALGO_STATUS_PREPARING) == DMA_ALGO_OK);
    CHECK(c_bytemap_ex_len(engine->working_algo) == 1);

    c_dma_algo_engine_on_md(engine, &observation);
    CHECK(counters.md_hits == 2);

    CHECK(c_dma_algo_set_status(first, DMA_ALGO_STATUS_READY) == DMA_ALGO_OK);
    CHECK(c_bytemap_ex_len(engine->working_algo) == 1);

    // DONE leaves the working set and gets no market data. The previous state
    // was READY, not WORKING, so no finish time is stamped.
    mds_state.ts.unix_ts = 1720000000.;
    CHECK(c_dma_algo_set_status(first, DMA_ALGO_STATUS_DONE) == DMA_ALGO_OK);
    CHECK(first->ts_finished.unix_ts == 0.);
    CHECK(c_bytemap_ex_len(engine->working_algo) == 0);

    c_dma_algo_engine_on_md(engine, &observation);
    CHECK(counters.md_hits == 2);

    // Completion from WORKING stamps the finish time.
    CHECK(c_dma_algo_set_status(first, DMA_ALGO_STATUS_WORKING) == DMA_ALGO_OK);
    CHECK(c_bytemap_ex_len(engine->working_algo) == 1);

    mds_state.ts.unix_ts = 1725000000.;
    CHECK(c_dma_algo_set_status(first, DMA_ALGO_STATUS_DONE) == DMA_ALGO_OK);
    CHECK(first->ts_finished.unix_ts == 1725000000.);
    CHECK(c_bytemap_ex_len(engine->working_algo) == 0);

    // A transition from a terminal status back into the set re-links.
    CHECK(c_dma_algo_set_status(first, DMA_ALGO_STATUS_READY) == DMA_ALGO_OK);
    CHECK(c_bytemap_ex_len(engine->working_algo) == 1);

    c_dma_algo_engine_on_md(engine, &observation);
    CHECK(counters.md_hits == 3);

    // A second algo of the same ticker shares the list; one observation reaches each once.
    dma_algo* second = c_dma_new_algo(access, registration->reg_id, "AAPL", SIDE_SHORT_OPEN, 5., NULL, g_allocator);
    CHECK(second != NULL);
    CHECK(c_bytemap_ex_len(engine->working_algo) == 1);

    second->callbacks.on_md     = test_algo_on_md;
    second->callbacks.user_data = &counters;

    c_dma_algo_engine_on_md(engine, &observation);
    CHECK(counters.md_hits == 5);

    // Unlinking the non-head algo keeps the head in place.
    CHECK(c_dma_algo_set_status(first, DMA_ALGO_STATUS_CLOSED) == DMA_ALGO_OK);
    CHECK(c_bytemap_ex_len(engine->working_algo) == 1);

    c_dma_algo_engine_on_md(engine, &observation);
    CHECK(counters.md_hits == 6);

    // A terminal record freed while a sibling is still listed leaves the sibling's head intact.
    dma_algo* third = (dma_algo*) c_ap_alloc(sizeof(dma_algo), g_allocator);
    CHECK(third != NULL);
    CHECK(c_dma_algo_init(third, access, "AAPL", SIDE_LONG_OPEN, 5., 1., g_allocator) == DMA_ALGO_OK);
    CHECK(c_dma_algo_set_status(third, DMA_ALGO_STATUS_CLOSED) == DMA_ALGO_OK);
    CHECK(third->ts_finished.unix_ts == 0.);  // IDLE -> CLOSED never worked.

    c_dma_algo_free(third);
    CHECK(c_bytemap_ex_len(engine->working_algo) == 1);

    c_dma_algo_engine_on_md(engine, &observation);
    CHECK(counters.md_hits == 7);

    // STOPPING and REJECTED are states of working: the seat is kept.
    CHECK(c_dma_algo_set_status(second, DMA_ALGO_STATUS_STOPPING) == DMA_ALGO_OK);
    CHECK(c_bytemap_ex_len(engine->working_algo) == 1);

    c_dma_algo_engine_on_md(engine, &observation);
    CHECK(counters.md_hits == 8);

    CHECK(c_dma_algo_set_status(second, DMA_ALGO_STATUS_REJECTED) == DMA_ALGO_OK);
    CHECK(c_bytemap_ex_len(engine->working_algo) == 1);

    c_dma_algo_engine_on_md(engine, &observation);
    CHECK(counters.md_hits == 9);

    // A retry from REJECTED reaches WORKING and copies the MDS timestamp into ts_started.
    mds_state.ts.unix_ts = 1730000000.;
    CHECK(c_dma_algo_set_status(second, DMA_ALGO_STATUS_WORKING) == DMA_ALGO_OK);
    CHECK(second->ts_started.unix_ts == 1730000000.);
    CHECK(c_bytemap_ex_len(engine->working_algo) == 1);

    c_dma_algo_engine_on_md(engine, &observation);
    CHECK(counters.md_hits == 10);

    // ERROR releases the seat and parks the record for intervention.
    CHECK(c_dma_algo_set_status(second, DMA_ALGO_STATUS_ERROR) == DMA_ALGO_OK);
    CHECK(c_bytemap_ex_len(engine->working_algo) == 0);
    CHECK(engine->error_algo == second);

    c_dma_free(access);
}

static void test_algo_unlink_guard(void) {

    dma* access = c_dma_new(NULL, g_allocator);
    CHECK(access != NULL);

    dma_algo_engine* engine = access->ALGO_ENGINE;
    CHECK(engine != NULL);

    mds mds_state;
    memset(&mds_state, 0, sizeof(mds_state));
    mds_state.ts.unix_ts = 1.;

    access->MDS         = &mds_state;

    dma_algo_ctx* registration = c_dma_algo_registry_reg(engine->registry, "guard", test_algo_factory, 0);
    CHECK(registration != NULL);

    dma_algo* algo = c_dma_new_algo(access, registration->reg_id, "MSFT", SIDE_LONG_OPEN, 10., NULL, g_allocator);
    CHECK(algo != NULL);
    CHECK(c_bytemap_ex_len(engine->working_algo) == 1);

    md_trade_instruction* order = c_dma_algo_engine_new_order(engine, algo, 10., 10.);
    CHECK(order != NULL);
    CHECK(c_bytemap_ex_len(algo->working_orders) == 1);

    // A status outside the working set is recorded, but the record keeps its
    // seat while the order is live.
    CHECK(c_dma_algo_set_status(algo, DMA_ALGO_STATUS_CLOSED) == DMA_ALGO_ERR_WORKING_ORDERS);
    CHECK(algo->status == DMA_ALGO_STATUS_CLOSED);
    CHECK(c_bytemap_ex_len(engine->working_algo) == 1);

    // Move the order out of IDLE so the cancel transition is legal.
    CHECK(c_dma_launch_order_unchecked(access, order) == DMA_OK);
    CHECK(order->order_state == STATE_SENT);

    // Once the order is terminal, the next outside transition releases the seat.
    md_trade_instruction canceled = *order;
    canceled.order_state = STATE_CANCELED;
    CHECK(c_dma_algo_engine_on_order(engine, &canceled) == DMA_ALGO_OK);
    CHECK(c_bytemap_ex_len(algo->working_orders) == 0);
    CHECK(c_dma_algo_set_status(algo, DMA_ALGO_STATUS_CLOSED) == DMA_ALGO_OK);
    CHECK(c_bytemap_ex_len(engine->working_algo) == 0);

    // Re-entering the working set takes the seat again, exactly once.
    CHECK(c_dma_algo_set_status(algo, DMA_ALGO_STATUS_READY) == DMA_ALGO_OK);
    CHECK(c_bytemap_ex_len(engine->working_algo) == 1);

    c_dma_free(access);
}

static void test_algo_new_order_uses_mds_ts(void) {

    dma* access = c_dma_new(NULL, g_allocator);
    CHECK(access != NULL);

    dma_algo_engine* engine = access->ALGO_ENGINE;
    CHECK(engine != NULL);

    mds mds_state;
    memset(&mds_state, 0, sizeof(mds_state));
    mds_state.ts.unix_ts = 1720000000.;
    mds_state.ts.ts      = 43200.;
    mds_state.ts.ordinal = 19700;
    mds_state.ts.date.year  = 2024;
    mds_state.ts.date.month = 2;
    mds_state.ts.date.day   = 8;
    mds_state.ts.date.stype = SESSION_TYPE_NORMINAL;

    access->MDS         = &mds_state;

    dma_algo_ctx* registration = c_dma_algo_registry_reg(engine->registry, "mds_ts", test_algo_factory, 0);
    CHECK(registration != NULL);

    dma_algo* algo = c_dma_new_algo(access, registration->reg_id, "AAPL", SIDE_LONG_OPEN, 10., NULL, g_allocator);
    CHECK(algo != NULL);

    md_trade_instruction* order = c_dma_algo_engine_new_order(engine, algo, 10., 1.);
    CHECK(order != NULL);

    // The order's whole session datetime is the access's MDS pack.
    CHECK(order->meta_info.dt.unix_ts == mds_state.ts.unix_ts);
    CHECK(order->meta_info.dt.ts == mds_state.ts.ts);
    CHECK(order->meta_info.dt.ordinal == mds_state.ts.ordinal);
    CHECK(order->meta_info.dt.date.year == mds_state.ts.date.year);
    CHECK(order->meta_info.dt.date.month == mds_state.ts.date.month);
    CHECK(order->meta_info.dt.date.day == mds_state.ts.date.day);
    CHECK(order->meta_info.dt.date.stype == mds_state.ts.date.stype);
    CHECK(order->ts_placed == mds_state.ts.unix_ts);

    c_dma_free(access);
}

static void test_dma_teardown(void) {

    dma* access = c_dma_new(NULL, g_allocator);
    CHECK(access != NULL);

    dma_algo_engine* engine = access->ALGO_ENGINE;
    CHECK(engine != NULL);

    mds mds_state;
    memset(&mds_state, 0, sizeof(mds_state));

    access->MDS         = &mds_state;

    dma_algo_ctx* registration = c_dma_algo_registry_reg(engine->registry, "teardown", test_algo_factory, 0);
    CHECK(registration != NULL);

    dma_algo* algo = c_dma_new_algo(access, registration->reg_id, "AAPL", SIDE_LONG_OPEN, 10., NULL, g_allocator);
    CHECK(algo != NULL);
    CHECK(c_bytemap_ex_len(engine->algo) == 1);

    md_trade_instruction* order = c_dma_algo_engine_new_order(engine, algo, 10., 10.);
    CHECK(order != NULL);
    CHECK(c_bytemap_ex_len(engine->orders) == 1);

    char* stored = NULL;
    CHECK(c_bytemap_ex_get_ptr(engine->orders, (const char*) &order->order_id, sizeof(long_md_id), &stored, NULL) == BYTEMAP_OK);

    dma_order_ctx* order_ctx = *(dma_order_ctx**) stored;

    // The order leaves IDLE through the sent entry; the fill flow accepts it.
    order->order_state = STATE_SENT;
    CHECK(c_dma_algo_engine_on_order_launched_inplace(engine, order_ctx) == DMA_ALGO_OK);
    CHECK(algo->working_volume == 10.);

    md_trade_report fill;
    memset(&fill, 0, sizeof(fill));
    fill.meta_info.dt.unix_ts = 1.;
    fill.multiplier = 1.;
    fill.order_id = order->order_id;
    strcpy(fill.trade_id.data, "teardown-1");
    fill.side = SIDE_LONG_OPEN;
    fill.volume = 10.;
    fill.notional = 105.;

    CHECK(c_dma_algo_engine_on_trade(engine, &fill) == DMA_ALGO_OK);
    CHECK(c_bytemap_ex_len(order_ctx->reports) == 1);
    CHECK(algo->working_volume == 0.);

    md_variant buffered = make_order("AAPL", 1., SIDE_LONG_OPEN);
    CHECK(c_dma_order_queue_push(&access->requests, DMA_ACTION_LAUNCH, &buffered.trade_instruction) == DMA_OK);

    dma_position* position = c_dma_position_new(access, engine, "passive", g_allocator);
    CHECK(position != NULL);
    CHECK(c_dma_position_add_algo(position, algo) == POSITION_OK);
    CHECK(c_bytemap_ex_len(engine->position) == 1);

    // The sweep frees every algo record, order context and stored report, and
    // drops the hosted position's references before those records go.
    CHECK(c_dma_clear(access) == DMA_OK);
    CHECK(access->requests.size == 0);
    CHECK(c_bytemap_ex_len(engine->orders) == 0);
    CHECK(c_bytemap_ex_len(engine->working_orders_by_id) == 0);
    CHECK(c_bytemap_ex_len(engine->working_orders_by_ticker) == 0);
    CHECK(c_bytemap_ex_len(engine->algo) == 0);
    CHECK(c_bytemap_ex_len(engine->working_algo) == 0);
    CHECK(engine->error_algo == NULL);
    CHECK(c_bytemap_ex_len(registration->orders) == 0);
    CHECK(c_bytemap_ex_len(registration->algos) == 0);
    CHECK(c_bytemap_ex_len(position->algos) == 0);
    CHECK(c_bytemap_ex_len(position->working_algos) == 0);
    CHECK(c_bytemap_ex_len(position->tickers) == 0);
    CHECK(c_bytemap_ex_len(engine->position) == 0);

    // The registrations and the engine's maps survive the sweep, so the
    // engine builds records again.
    CHECK(engine->registry != NULL);
    CHECK(engine->orders != NULL);
    CHECK(engine->algo != NULL);

    dma_algo* rebuilt = c_dma_new_algo(access, registration->reg_id, "AAPL", SIDE_LONG_OPEN, 5., NULL, g_allocator);
    CHECK(rebuilt != NULL);
    CHECK(c_bytemap_ex_len(engine->algo) == 1);
    CHECK(c_bytemap_ex_len(registration->algos) == 1);

    md_trade_instruction* rebuilt_order = c_dma_algo_engine_new_order(engine, rebuilt, 10., 5.);
    CHECK(rebuilt_order != NULL);
    CHECK(c_bytemap_ex_len(engine->orders) == 1);

    // Dealloc runs the same sweep and releases the access's own subordinates,
    // including the owned engine, leaving the access zeroed.
    c_dma_dealloc(access);
    CHECK(access->MDS == NULL);
    CHECK(access->ALGO_ENGINE == NULL);
    CHECK(access->RISK_PROFILE == NULL);
    CHECK(access->INVENTORY == NULL);
    CHECK(access->BALANCE == NULL);
    CHECK(access->protocol.handlers.order_launch_fn == NULL);
    CHECK(access->requests.entries == NULL);
    CHECK(access->requests.size == 0);
    CHECK(access->requests.capacity == 0);
    CHECK(access->seq_id == 0);
    CHECK(access->INVENTORY == NULL);
    CHECK(access->positions == NULL);

    c_ap_free(access);

}

// ========== Main ==========

int main(void) {
    void* root = c_ap_alloc(1, NULL);
    g_allocator = c_ap_protocol_from_ptr(root);

    test_risk_profile_rules();
    test_risk_ticker_gates();
    test_risk_check_incomplete();
    test_risk_check_limits();
    test_risk_check_gates();
    test_risk_access_checks();
    test_risk_custom_rules();
    test_dma_lifecycle_and_gate();
    test_dma_order_state_contract();
    test_dma_request_queue();
    test_dma_order_handler_self_detach();
    test_dma_edge_contracts();
    test_balance();
    test_dma_balance_gate();
    test_inventory();
    test_inventory_corporate_actions();
    test_algo_engine();
    test_position();
    test_position_partial_fill_flow();
    test_access_inplace_updates();
    test_algo_inplace_entries();
    test_risk_bench();
    test_dma_bench();
    test_position_bench();
    test_algo_status_working_set();
    test_algo_unlink_guard();
    test_algo_new_order_uses_mds_ts();
    test_dma_teardown();

    c_ap_free(root);

    printf("%zu checks, %zu failures\n", g_checks, g_failures);
    return g_failures ? 1 : 0;
}

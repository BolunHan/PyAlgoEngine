#include "c_ex_profile_base.h"

static void default_on_activate(const exchange_profile* profile) {
    if (!profile) return;
    // ((exchange_profile*) profile)->tz_offset_seconds = c_utc_offset_seconds();
    (void) profile;
}

static void default_on_deactivate(const exchange_profile* profile) {
    (void) profile;
}

static session_date_range_t* default_trade_calendar(const session_date_t* start_date, const session_date_t* end_date) {
    // Default: every calendar day is a trading day, and every day is nominal.
    session_date_range_t* drange = c_ex_profile_date_range(start_date, end_date);
    if (!drange) return NULL;

    drange->start.stype = SESSION_TYPE_NORMINAL;
    drange->end.stype = SESSION_TYPE_NORMINAL;
    for (size_t i = 0; i < drange->n_days; ++i) {
        drange->dates[i].stype = SESSION_TYPE_NORMINAL;
    }

    return drange;
}

static auction_phase default_resolve_auction_phase(double ts) {
    (void) ts;
    return AUCTION_PHASE_DONE;
}

static session_phase default_resolve_session_phase(double ts) {
    (void) ts;
    return SESSION_PHASE_CONTINUOUS;
}

static session_type default_resolve_session_type(uint16_t year, uint8_t month, uint8_t day) {
    (void) year;
    (void) month;
    (void) day;
    return SESSION_TYPE_NORMINAL;
}

// Default profile instance

const exchange_profile EX_PROFILE_DEFAULT = {
    .profile_id = "UTC_NONSTOP_DEFAULT",

    .session_start = {
        .elapsed_seconds = 0.0,
        .hour = 0u,
        .minute = 0u,
        .second = 0u,
        .nanosecond = 0u,
        .phase = SESSION_PHASE_CONTINUOUS,
    },
    .session_end = {
        .elapsed_seconds = (double) SECONDS_PER_DAY,
        .hour = 23u,
        .minute = 59u,
        .second = 59u,
        .nanosecond = (uint32_t) NANOS_PER_SECOND - 1u,
        .phase = SESSION_PHASE_CONTINUOUS,
    },
    .session_start_ts = 0.0,
    .session_end_ts = (double) SECONDS_PER_DAY,
    .session_length_seconds = (double) SECONDS_PER_DAY,

    .open_call_auction = NULL,
    .close_call_auction = NULL,
    .session_breaks = NULL,

    .time_zone = "UTC",
    .tz_offset_seconds = 0.0,

    .on_activate_func = default_on_activate,
    .on_deactivate_func = default_on_deactivate,
    .trade_calendar_func = default_trade_calendar,
    .resolve_auction_phase_func = default_resolve_auction_phase,
    .resolve_session_phase_func = default_resolve_session_phase,
    .resolve_session_type_func = default_resolve_session_type,
};

// Externs from the header
const exchange_profile*     EX_PROFILE = &EX_PROFILE_DEFAULT;
const session_date_range_t* EX_TRADE_CALENDAR_CACHE = NULL;

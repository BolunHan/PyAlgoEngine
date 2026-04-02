#ifndef C_EX_PROFILE_BASE_H
#define C_EX_PROFILE_BASE_H

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>

// ========== Constants ==========

#ifndef SECONDS_PER_DAY
#define SECONDS_PER_DAY 86400
#endif

#ifndef SECONDS_PER_HOUR
#define SECONDS_PER_HOUR 3600
#endif

#ifndef SECONDS_PER_MINUTE
#define SECONDS_PER_MINUTE 60
#endif

#ifndef NANOS_PER_SECOND
#define NANOS_PER_SECOND 1000000000
#endif

#ifndef MICROS_PER_SECOND
#define MICROS_PER_SECOND 1000000
#endif

#ifndef UNIX_EPOCH_ORDINAL
#define UNIX_EPOCH_ORDINAL ((int64_t) 719163)
#endif

#ifndef EX_PROFILE_ID_SIZE
#define EX_PROFILE_ID_SIZE 64
#endif

#ifndef EX_PROFILE_MIN_YEAR
#define EX_PROFILE_MIN_YEAR ((uint16_t) 1)
#endif

#ifndef EX_PROFILE_MAX_YEAR
#define EX_PROFILE_MAX_YEAR ((uint16_t) 9999)
#endif

#ifndef EX_PROFILE_MAX_ORDINAL
#define EX_PROFILE_MAX_ORDINAL ((uint32_t) 3652059)
#endif

static const uint8_t DAYS_IN_MONTH_TABLE[12] = {
    31u, 28u, 31u, 30u, 31u, 30u, 31u, 31u, 30u, 31u, 30u, 31u
};

static const uint16_t DAYS_BEFORE_MONTH_TABLE[12] = {
    0u, 31u, 59u, 90u, 120u, 151u, 181u, 212u, 243u, 273u, 304u, 334u
};

// ========== Structs ==========

typedef enum session_type {
    SESSION_TYPE_NON_TRADING = 0,
    SESSION_TYPE_NORMINAL = 1 << 0,
    SESSION_TYPE_SUSPENDED = 1 << 1,
    SESSION_TYPE_HALF_DAY = 1 << 2,
    SESSION_TYPE_CIRCUIT_BREAK = 1 << 4,
} session_type;

typedef enum session_phase {
    SESSION_PHASE_UNKNOWN,
    SESSION_PHASE_PREOPEN,
    SESSION_PHASE_OPEN_AUCTION,
    SESSION_PHASE_CONTINUOUS,
    SESSION_PHASE_BREAK,
    SESSION_PHASE_SUSPENDED,
    SESSION_PHASE_CLOSE_AUCTION,
    SESSION_PHASE_CLOSED
} session_phase;

typedef enum auction_phase {
    AUCTION_PHASE_ACTIVE,
    AUCTION_PHASE_NO_CANCEL,
    AUCTION_PHASE_FROZEN,
    AUCTION_PHASE_UNCROSSING,
    AUCTION_PHASE_DONE,
} auction_phase;

typedef struct session_time_t {
    double elapsed_seconds;
    uint8_t hour;
    uint8_t minute;
    uint8_t second;
    uint32_t nanosecond;
    session_phase session_phase;
} session_time_t;

typedef struct session_date_t {
    uint16_t year;  // 0-9999
    uint8_t month;  // 1-12
    uint8_t day;    // 1-31
    session_type session_type;
} session_date_t;

typedef struct session_time_range_t {
    session_time_t start;
    session_time_t end;
    double elapsed_seconds;
} session_time_range_t;

typedef struct session_date_range_t {
    session_date_t start;
    session_date_t end;
    size_t n_days;
    session_date_t dates[];
} session_date_range_t;

typedef struct call_auction {
    session_time_t auction_start;
    session_time_range_t active;
    session_time_range_t no_cancel;
    session_time_range_t frozen;
    session_time_t uncross;
    session_time_t auction_end;
} call_auction;

typedef struct session_break {
    session_time_t break_start;
    session_time_t break_end;
    double break_start_ts;        // From midnight, not from session start, cached for break adjustment calculation
    double break_end_ts;          // From midnight, not from session start, cached for break adjustment calculation
    double break_length_seconds;  // Cached for break adjustment calculation
    struct session_break* next;
} session_break;

typedef void (*c_ex_profile_on_activate)(struct exchange_profile* profile);
typedef void (*c_ex_profile_on_deactivate)(struct exchange_profile* profile);
typedef session_date_range_t* (*c_ex_profile_trade_calendar)(session_date_t start_date, session_date_t end_date);
typedef auction_phase (*c_ex_profile_resolve_auction_phase)(double ts);
typedef session_phase (*c_ex_profile_resolve_session_phase)(double ts);
typedef session_type (*c_ex_profile_resolve_session_type)(uint16_t year, uint8_t month, uint8_t day);

typedef struct exchange_profile {
    char profile_id[EX_PROFILE_ID_SIZE];  // Unique identifier for the exchange profile (e.g., "NYSE", "NASDAQ")
    session_time_t session_start;         // Continuous session start time
    session_time_t session_end;           // Continuous session end time
    double session_start_ts;              // Continuous session start time in seconds from midnight, cached for session phase resolution and break adjustment calculation
    double session_end_ts;                // Continuous session end time in seconds from midnight. For continuous session filtering.
    double session_length_seconds;        // Continuous session length in seconds, cached for break adjustment calculation
    call_auction* open_call_auction;      // Pointer to open call auction details (if applicable)
    call_auction* close_call_auction;     // Pointer to close call auction details (if applicable)
    session_break* session_breaks;        // Linked list of session breaks (if applicable)
    const char* time_zone;                // IANA time zone string (e.g., "UTC", "America/New_York")
    double tz_offset_seconds;             // Time zone offset in seconds from UTC (e.g., -18000 for EST). Subject to machine local time zone to UTC conversion

    // Function pointers for exchange-specific logic
    c_ex_profile_on_activate on_activate_func;
    c_ex_profile_on_deactivate on_deactivate_func;
    c_ex_profile_trade_calendar trade_calendar_func;
    c_ex_profile_resolve_auction_phase resolve_auction_phase_func;
    c_ex_profile_resolve_session_phase resolve_session_phase_func;
    c_ex_profile_resolve_session_type resolve_session_type_func;
} exchange_profile;

// ========== Forward Declaration ==========

extern exchange_profile* EX_PROFILE;
extern session_date_range_t* EX_TRADE_CALENDAR_CACHE;

static inline double c_utc_offset_seconds(void);

static inline double c_ex_profile_time_to_ts(uint8_t hour, uint8_t minute, uint8_t second, uint32_t nanosecond);
static inline double c_ex_profile_unix_to_ts(double unix_ts);
static inline double c_ex_profile_ts_to_elapsed(double elapsed);

static inline int c_ex_profile_session_date_compare(const void* d1, const void* d2);
static inline uint32_t c_ex_profile_days_before_year(uint16_t year);
static inline bool c_ex_profile_is_leap_year(uint16_t year);
static inline uint8_t c_ex_profile_days_in_month(uint16_t year, uint8_t month);
static inline bool c_ex_profile_date_is_valid(const session_date_t* date);
static inline uint32_t c_ex_profile_date_to_ordinal(const session_date_t* date);
static inline int c_ex_profile_date_from_ordinal(uint32_t ordinal, session_date_t* out);
static inline int c_ex_profile_date_from_year_day(uint16_t year, uint32_t day_of_year, session_date_t* out);
static inline int c_ex_profile_next_day(const session_date_t* date, session_date_t* out);
static inline int c_ex_profile_previous_day(const session_date_t* date, session_date_t* out);
static inline int c_ex_profile_days_after(const session_date_t* date, size_t days_after, session_date_t* out);
static inline int c_ex_profile_days_before(const session_date_t* date, size_t days_before, session_date_t* out);
static inline session_date_range_t* c_ex_profile_date_range_between(const session_date_t* start_date, const session_date_t* end_date);

static inline void c_ex_profile_activate(exchange_profile* profile);

static inline session_time_t* c_ex_profile_session_time_new(uint8_t hour, uint8_t minute, uint8_t second, uint32_t nanosecond);
static inline int c_ex_profile_session_time_from_ts(double unix_ts, session_time_t* out);
static inline session_time_range_t* c_ex_profile_session_trange_between_time(session_time_t* start_time, session_time_t* end_time);
static inline session_time_range_t* c_ex_profile_session_trange_between_ts(double start_unix_ts, double end_unix_ts);

static inline session_date_t* c_ex_profile_session_date_new(uint16_t year, uint8_t month, uint8_t day);
static inline int c_ex_profile_session_date_from_ts(double unix_ts, session_date_t* out);
static inline size_t c_ex_profile_session_date_index(session_date_t* date, session_date_range_t* drange);
static inline session_date_range_t* c_ex_profile_session_date_between(session_date_t* start_date, session_date_t* end_date);
static inline int c_ex_profile_session_trading_days_before(session_date_t* market_date, size_t days, session_date_t* out);
static inline int c_ex_profile_session_trading_days_after(session_date_t* market_date, size_t days, session_date_t* out);
static inline int c_ex_profile_nearest_trading_date(session_date_t* market_date, bool previous, session_date_t* out);

// ========== Utilities Functions (session_time_t) ==========

static inline double c_utc_offset_seconds(void) {
    time_t now = time(NULL);
    struct tm local_tm;
    struct tm gmt_tm;
    time_t local_seconds = 0;
    time_t gmt_seconds = 0;
    double diff_seconds = 0.0;

    localtime_r(&now, &local_tm);
    gmtime_r(&now, &gmt_tm);

    local_seconds = mktime(&local_tm);
    gmt_seconds = mktime(&gmt_tm);
    diff_seconds = difftime(local_seconds, gmt_seconds);

    return diff_seconds;
}

static inline double c_ex_profile_time_to_ts(uint8_t hour, uint8_t minute, uint8_t second, uint32_t nanosecond) {
    double total_seconds = hour * SECONDS_PER_HOUR + minute * SECONDS_PER_MINUTE + second + (double) nanosecond / NANOS_PER_SECOND;
    return total_seconds;
}

static inline double c_ex_profile_unix_to_ts(double unix_ts) {
    // When no active profile is set, fall back to the machine-local Unix day fraction.
    const double tz_offset = EX_PROFILE ? EX_PROFILE->tz_offset_seconds : 0.0;
    double total_seconds = fmod(unix_ts + tz_offset, (double) SECONDS_PER_DAY);
    return total_seconds;
}

static inline double c_ex_profile_ts_to_elapsed(double ts) {
    if (!EX_PROFILE) return 0.0;

    if (ts <= EX_PROFILE->session_start_ts) return 0.0;
    if (ts >= EX_PROFILE->session_end_ts) return EX_PROFILE->session_length_seconds;

    double break_elapsed = 0;
    session_break* current_break = EX_PROFILE->session_breaks;
    while (current_break) {
        if (ts > current_break->break_end_ts) break_elapsed += current_break->break_length_seconds;
        else if (ts > current_break->break_start_ts) break_elapsed += ts - current_break->break_start_ts;
        else break;
        current_break = current_break->next;
    }

    return ts - break_elapsed - EX_PROFILE->session_start_ts;
}

// ========== Utilities Functions (session_date_t) ==========

static inline int c_ex_profile_session_date_compare(const void* d1, const void* d2) {
    // qsort function, returns 1, -1, or 0
    if (d1 == d2) return 0;
    if (!d1) return -1;
    if (!d2) return 1;

    const session_date_t* date1 = (const session_date_t*) d1;
    const session_date_t* date2 = (const session_date_t*) d2;
    if (date1->year < date2->year) return -1;
    if (date1->year > date2->year) return 1;
    if (date1->month < date2->month) return -1;
    if (date1->month > date2->month) return 1;
    if (date1->day < date2->day) return -1;
    if (date1->day > date2->day) return 1;
    return 0;
}

static inline uint32_t c_ex_profile_days_before_year(uint16_t year) {
    uint32_t y = (uint32_t) year - 1u;
    return 365u * y + (y / 4u) - (y / 100u) + (y / 400u);
}

static inline bool c_ex_profile_is_leap_year(uint16_t year) {
    if (year < EX_PROFILE_MIN_YEAR || year > EX_PROFILE_MAX_YEAR) return false;
    return ((year % 4u) == 0u) && (((year % 100u) != 0u) || ((year % 400u) == 0u));
}

static inline uint8_t c_ex_profile_days_in_month(uint16_t year, uint8_t month) {
    if (year < EX_PROFILE_MIN_YEAR || year > EX_PROFILE_MAX_YEAR || month < 1u || month > 12u) return 0u;
    uint8_t days = DAYS_IN_MONTH_TABLE[(uint8_t) (month - 1u)];
    if (month == 2u && c_ex_profile_is_leap_year(year)) days = 29u;
    return days;
}

static inline bool c_ex_profile_date_is_valid(const session_date_t* date) {
    uint8_t dim = 0u;
    if (!date) return false;
    dim = c_ex_profile_days_in_month(date->year, date->month);
    if (dim == 0u) return false;
    return date->day >= 1u && date->day <= dim;
}

static inline uint32_t c_ex_profile_date_to_ordinal(const session_date_t* date) {
    uint32_t ordinal = 0u;

    if (!date) return 0u;
    if (!c_ex_profile_date_is_valid(date)) return 0u;

    ordinal = c_ex_profile_days_before_year(date->year);
    ordinal += (uint32_t) DAYS_BEFORE_MONTH_TABLE[(uint8_t) (date->month - 1u)];
    if (date->month > 2u && c_ex_profile_is_leap_year(date->year)) {
        ordinal += 1u;
    }
    ordinal += (uint32_t) date->day;

    return ordinal;
}

static inline int c_ex_profile_date_from_ordinal(uint32_t ordinal, session_date_t* out) {
    // Ordinal is 1-based: 1 == 0001-01-01
    // Use 400/100/4/1-year cycle decomposition to avoid binary search.
    // This matches the proleptic Gregorian calendar used by c_ex_profile_days_before_year().
    uint32_t max_ordinal = 0u;
    uint32_t n = 0u;  // 0-based day index
    uint32_t n400 = 0u;
    uint32_t n100 = 0u;
    uint32_t n4 = 0u;
    uint32_t n1 = 0u;
    uint32_t day_of_year = 0u;  // 0-based day-of-year
    uint16_t year = 0u;
    uint8_t month = 1u;
    uint8_t is_leap = 0u;

    if (!out) return -1;

    // Keep bounds consistent even if EX_PROFILE_MAX_YEAR is overridden.
    max_ordinal = c_ex_profile_days_before_year((uint16_t) (EX_PROFILE_MAX_YEAR + 1u));
    if (ordinal < 1u || ordinal > max_ordinal) return -1;

    n = ordinal - 1u;

    // Decompose days since 0001-01-01 into Gregorian year and day-of-year.
    // 400-year cycle: 146097 days
    n400 = n / 146097u;
    n %= 146097u;

    // 100-year cycle: 36524 days, but clamp to 0..3 within a 400-year cycle
    n100 = n / 36524u;
    if (n100 == 4u) n100 = 3u;
    n -= n100 * 36524u;

    // 4-year cycle: 1461 days
    n4 = n / 1461u;
    n %= 1461u;

    // 1-year cycle: 365 days, clamp to 0..3 within a 4-year cycle
    n1 = n / 365u;
    if (n1 == 4u) n1 = 3u;
    n -= n1 * 365u;

    year = (uint16_t) (400u * n400 + 100u * n100 + 4u * n4 + n1 + 1u);
    day_of_year = n;

    is_leap = (uint8_t) (((year % 4u) == 0u) && (((year % 100u) != 0u) || ((year % 400u) == 0u)));

    // Convert day-of-year to month/day.
    // DAYS_BEFORE_MONTH_TABLE is for non-leap years; adjust for leap years after Feb.
    for (month = 1u; month <= 12u; ++month) {
        uint16_t next = 0u;
        if (month == 12u) {
            next = (uint16_t) (365u + is_leap);
        }
        else {
            next = DAYS_BEFORE_MONTH_TABLE[month];
            if (is_leap && month >= 2u) next = (uint16_t) (next + 1u);
        }
        if (day_of_year < (uint32_t) next) break;
    }

    {
        uint16_t start = DAYS_BEFORE_MONTH_TABLE[(uint8_t) (month - 1u)];
        if (is_leap && month > 2u) start = (uint16_t) (start + 1u);
        out->year = year;
        out->month = month;
        out->day = (uint8_t) ((uint16_t) (day_of_year - (uint32_t) start + 1u));
    }
    return 0;
}

static inline int c_ex_profile_date_from_year_day(uint16_t year, uint32_t day_of_year, session_date_t* out) {
    // day_of_year is 0-based: 0 == Jan 1
    uint8_t month = 1u;
    uint8_t is_leap = 0u;

    if (!out) return -1;
    if (year < EX_PROFILE_MIN_YEAR || year > EX_PROFILE_MAX_YEAR) return -1;

    is_leap = (uint8_t) (((year % 4u) == 0u) && (((year % 100u) != 0u) || ((year % 400u) == 0u)));
    if (day_of_year >= (uint32_t) (365u + is_leap)) return -1;

    // Convert day-of-year to month/day.
    // DAYS_BEFORE_MONTH_TABLE is for non-leap years; adjust for leap years after Feb.
    for (month = 1u; month <= 12u; ++month) {
        uint16_t next = 0u;
        if (month == 12u) {
            next = (uint16_t) (365u + is_leap);
        }
        else {
            next = DAYS_BEFORE_MONTH_TABLE[month];
            if (is_leap && month >= 2u) next = (uint16_t) (next + 1u);
        }
        if (day_of_year < (uint32_t) next) break;
    }

    {
        uint16_t start = DAYS_BEFORE_MONTH_TABLE[(uint8_t) (month - 1u)];
        if (is_leap && month > 2u) start = (uint16_t) (start + 1u);
        out->year = year;
        out->month = month;
        out->day = (uint8_t) ((uint16_t) (day_of_year - (uint32_t) start + 1u));
    }

    return 0;
}

static inline int c_ex_profile_next_day(const session_date_t* date, session_date_t* out) {
    // Ultra-fast path for consecutive date iteration.
    if (!date || !out) return -1;
    if (!c_ex_profile_date_is_valid(date)) return -1;

    uint16_t year = date->year;
    uint8_t month = date->month;
    uint8_t day = date->day;
    uint8_t dim = c_ex_profile_days_in_month(year, month);
    if (dim == 0u) return -1;

    if (day < dim) {
        *out = *date;
        out->day = (uint8_t) (day + 1u);
        return 0;
    }

    // End of month
    if (month < 12u) {
        *out = *date;
        out->month = (uint8_t) (month + 1u);
        out->day = 1u;
        return 0;
    }

    // End of year
    if (year >= EX_PROFILE_MAX_YEAR) return -1;
    *out = *date;
    out->year = (uint16_t) (year + 1u);
    out->month = 1u;
    out->day = 1u;
    return 0;
}

static inline int c_ex_profile_previous_day(const session_date_t* date, session_date_t* out) {
    // Ultra-fast path for consecutive date iteration.
    // Returns 0 on success; -1 on invalid input or if decrementing would exceed supported range.
    // Supports in-place operation: out may alias date.
    if (!date || !out) return -1;
    if (!c_ex_profile_date_is_valid(date)) return -1;

    const uint16_t year = date->year;
    const uint8_t month = date->month;
    const uint8_t day = date->day;

    if (day > 1u) {
        *out = *date;
        out->day = (uint8_t) (day - 1u);
        return 0;
    }

    // Beginning of month
    if (month > 1u) {
        const uint8_t prev_month = (uint8_t) (month - 1u);
        const uint8_t dim = c_ex_profile_days_in_month(year, prev_month);
        if (dim == 0u) return -1;
        *out = *date;
        out->month = prev_month;
        out->day = dim;
        return 0;
    }

    // Beginning of year
    if (year <= EX_PROFILE_MIN_YEAR) return -1;
    {
        const uint16_t prev_year = (uint16_t) (year - 1u);
        const uint8_t dim = c_ex_profile_days_in_month(prev_year, 12u);
        if (dim == 0u) return -1;
        *out = *date;
        out->year = prev_year;
        out->month = 12u;
        out->day = dim;
        return 0;
    }
}

static inline int c_ex_profile_days_after(const session_date_t* date, size_t days_after, session_date_t* out) {
    if (!date || !out) return -1;
    if (!c_ex_profile_date_is_valid(date)) return -1;

    if (days_after == 0u) {
        *out = *date;
        return 0;
    }

    // Ultra-fast small horizon: repeated next_day.
    if (days_after == 1u) return c_ex_profile_next_day(date, out);

    uint32_t ordinal = c_ex_profile_date_to_ordinal(date);
    if (ordinal == 0u) return -1;

    // Fast path: small horizon within the same year.
    {
        const uint16_t year = date->year;
        const uint32_t start_of_year = c_ex_profile_days_before_year(year);
        const uint32_t day_of_year = ordinal - start_of_year - 1u;
        const uint8_t is_leap = (uint8_t) (((year % 4u) == 0u) && (((year % 100u) != 0u) || ((year % 400u) == 0u)));
        const uint32_t year_len = (uint32_t) (365u + is_leap);

        if (day_of_year < year_len) {
            const uint32_t remaining = (uint32_t) (year_len - 1u - day_of_year);
            if (days_after <= (size_t) remaining) {
                return c_ex_profile_date_from_year_day(year, (uint32_t) (day_of_year + (uint32_t) days_after), out);
            }
        }
    }

    // Long horizon: ordinal math + from_ordinal.
    // Guard against uint32_t wrap.
    if (days_after > (size_t) (UINT32_MAX - ordinal)) return -1;

    ordinal += (uint32_t) days_after;
    return c_ex_profile_date_from_ordinal(ordinal, out);
}

static inline int c_ex_profile_days_before(const session_date_t* date, size_t days_before, session_date_t* out) {
    if (!out || !date) return -1;
    if (!c_ex_profile_date_is_valid(date)) return -1;

    if (days_before == 0u) {
        *out = *date;
        return 0;
    }

    // Ultra-fast small horizon: repeated previous_day.
    if (days_before == 1u) return c_ex_profile_previous_day(date, out);

    uint32_t ordinal = c_ex_profile_date_to_ordinal(date);
    if (ordinal == 0u) return -1;

    // Fast path: small horizon within the same year.
    {
        const uint16_t year = date->year;
        const uint32_t start_of_year = c_ex_profile_days_before_year(year);
        const uint32_t day_of_year = ordinal - start_of_year - 1u;
        const uint8_t is_leap = (uint8_t) (((year % 4u) == 0u) && (((year % 100u) != 0u) || ((year % 400u) == 0u)));
        const uint32_t year_len = (uint32_t) (365u + is_leap);

        if (day_of_year < year_len) {
            if (days_before <= (size_t) day_of_year) {
                return c_ex_profile_date_from_year_day(year, (uint32_t) (day_of_year - (uint32_t) days_before), out);
            }
        }
    }

    // Long horizon: ordinal math + from_ordinal.
    // Guard against size_t -> uint32_t truncation.
    if (days_before > (size_t) (ordinal - 1u)) return -1;

    ordinal -= (uint32_t) days_before;
    return c_ex_profile_date_from_ordinal(ordinal, out);
}

static inline session_date_range_t* c_ex_profile_date_range_between(const session_date_t* start_date, const session_date_t* end_date) {
    if (!start_date || !end_date) return NULL;

    uint32_t start_ordinal = c_ex_profile_date_to_ordinal(start_date);
    uint32_t end_ordinal = c_ex_profile_date_to_ordinal(end_date);
    if (start_ordinal == 0u || end_ordinal == 0u || end_ordinal < start_ordinal) return NULL;
    size_t n_days = (size_t) (end_ordinal - start_ordinal + 1u);
    session_date_range_t* drange = (session_date_range_t*) calloc(1, sizeof(session_date_range_t) + n_days * sizeof(session_date_t));
    if (!drange) return NULL;
    drange->start = *start_date;
    drange->end = *end_date;
    drange->n_days = n_days;

    // Fast fill: write dates[0], then repeated next_day into the array.
    drange->dates[0] = *start_date;
    for (size_t i = 1; i < n_days; ++i) {
        if (c_ex_profile_next_day(&drange->dates[i - 1u], &drange->dates[i]) != 0) {
            free(drange);
            return NULL;
        }
    }
    return drange;
}

// ========== Public APIs (exchange_profile) ==========

static inline void c_ex_profile_activate(exchange_profile* profile) {
    if (EX_PROFILE == profile) return;

    if (EX_PROFILE && EX_PROFILE->on_deactivate_func) EX_PROFILE->on_deactivate_func(EX_PROFILE);

    if (EX_TRADE_CALENDAR_CACHE) {
        free(EX_TRADE_CALENDAR_CACHE);
        EX_TRADE_CALENDAR_CACHE = NULL;
    }

    if (profile->on_activate_func) profile->on_activate_func(profile);

    EX_PROFILE = profile;
}

// ========== Public APIs (session_time_t) ==========

static inline session_time_t* c_ex_profile_session_time_new(uint8_t hour, uint8_t minute, uint8_t second, uint32_t nanosecond) {
    session_time_t* out = (session_time_t*) calloc(1, sizeof(session_time_t));
    if (!out) return NULL;

    out->hour = hour;
    out->minute = minute;
    out->second = second;
    out->nanosecond = nanosecond;

    double ts = c_ex_profile_time_to_ts(hour, minute, second, nanosecond);

    // No active exchange profile, return a default session time representation
    if (!EX_PROFILE) {
        out->elapsed_seconds = ts;
        out->session_phase = SESSION_PHASE_UNKNOWN;
        return out;
    }

    // Calculate elapsed seconds since continuous session start
    out->elapsed_seconds = c_ex_profile_ts_to_elapsed(ts);
    out->session_phase = EX_PROFILE->resolve_session_phase_func(ts);
    return out;
}

static inline int c_ex_profile_session_time_from_ts(double unix_ts, session_time_t* out) {
    if (!out) return -1;

    double ts = c_ex_profile_unix_to_ts(unix_ts);
    uint8_t hour = (uint8_t) (ts / SECONDS_PER_HOUR);
    uint8_t minute = (uint8_t) ((ts - hour * SECONDS_PER_HOUR) / SECONDS_PER_MINUTE);
    uint8_t second = (uint8_t) (ts - hour * SECONDS_PER_HOUR - minute * SECONDS_PER_MINUTE);
    uint32_t nanosecond = (uint32_t) ((ts - (uint64_t) ts) * NANOS_PER_SECOND);

    out->hour = hour;
    out->minute = minute;
    out->second = second;
    out->nanosecond = nanosecond;
    out->elapsed_seconds = c_ex_profile_ts_to_elapsed(ts);
    out->session_phase = EX_PROFILE ? EX_PROFILE->resolve_session_phase_func(ts) : SESSION_PHASE_UNKNOWN;
    return 0;
}

static inline session_time_range_t* c_ex_profile_session_trange_between_time(session_time_t* start_time, session_time_t* end_time) {
    session_time_range_t* trange = (session_time_range_t*) calloc(1, sizeof(session_time_range_t));
    if (!trange) return NULL;
    trange->start = *start_time;
    trange->end = *end_time;
    trange->elapsed_seconds = end_time->elapsed_seconds - start_time->elapsed_seconds;
    return trange;
}

static inline session_time_range_t* c_ex_profile_session_trange_between_ts(double start_unix_ts, double end_unix_ts) {
    session_time_range_t* trange = (session_time_range_t*) calloc(1, sizeof(session_time_range_t));
    if (!trange) return NULL;

    c_ex_profile_session_time_from_ts(start_unix_ts, &trange->start);
    c_ex_profile_session_time_from_ts(end_unix_ts, &trange->end);
    trange->elapsed_seconds = trange->end.elapsed_seconds - trange->start.elapsed_seconds;
    return trange;
}

// ========== Public APIs (session_date_t) ==========

static inline session_date_t* c_ex_profile_session_date_new(uint16_t year, uint8_t month, uint8_t day) {
    session_date_t* d = (session_date_t*) calloc(1, sizeof(session_date_t));
    if (!d) return NULL;
    d->year = year;
    d->month = month;
    d->day = day;
    d->session_type = EX_PROFILE ? EX_PROFILE->resolve_session_type_func(year, month, day) : SESSION_TYPE_NORMINAL;
    return d;
}

static inline int c_ex_profile_session_date_from_ts_sys(double unix_ts, session_date_t* out) {
    if (!out) return -1;
    if (!isfinite(unix_ts)) return -1;

    // If an exchange profile is active, interpret the date in that profile's local time
    // using its precomputed UTC offset. Otherwise, fall back to machine-local time.
    struct tm tm_local;
    if (EX_PROFILE) {
        const double adjusted = unix_ts + EX_PROFILE->tz_offset_seconds;
        const time_t sec = (time_t) adjusted;
        if (!gmtime_r(&sec, &tm_local)) return -1;
    }
    else {
        const time_t sec = (time_t) unix_ts;
        if (!localtime_r(&sec, &tm_local)) return -1;
    }

    out->year = (uint16_t) (tm_local.tm_year + 1900);
    out->month = (uint8_t) (tm_local.tm_mon + 1);
    out->day = (uint8_t) tm_local.tm_mday;
    out->session_type = EX_PROFILE ? EX_PROFILE->resolve_session_type_func(out->year, out->month, out->day) : SESSION_TYPE_NORMINAL;
    return 0;
}

static inline int c_ex_profile_session_date_from_ts(double unix_ts, session_date_t* out) {
    if (!out) return -1;
    if (!isfinite(unix_ts)) return -1;

    int64_t days = 0;
    int64_t ordinal = 0;

    double shifted_ts = unix_ts + (EX_PROFILE ? EX_PROFILE->tz_offset_seconds : 0.0);
    if (!isfinite(shifted_ts)) return -1;

    double days_floor = floor(shifted_ts / (double) SECONDS_PER_DAY);
    if (days_floor < (double) INT64_MIN || days_floor > (double) INT64_MAX) return -1;
    days = (int64_t) days_floor;

    ordinal = UNIX_EPOCH_ORDINAL + days;
    if (ordinal < 1 || ordinal > (int64_t) EX_PROFILE_MAX_ORDINAL) return -1;

    if (c_ex_profile_date_from_ordinal((uint32_t) ordinal, out) != 0) return -1;
    out->session_type = EX_PROFILE ? EX_PROFILE->resolve_session_type_func(out->year, out->month, out->day) : SESSION_TYPE_NORMINAL;
    return 0;
}

static inline size_t c_ex_profile_session_date_index(session_date_t* date, session_date_range_t* drange) {
    if (!date) return (size_t) -1;
    if (!drange) return (size_t) -1;
    if (drange->n_days == 0u) return (size_t) -1;
    if (c_ex_profile_session_date_compare(date, &drange->start) == -1) return 0;                // Start date
    if (c_ex_profile_session_date_compare(date, &drange->end) == 1) return drange->n_days - 1;  // End date

    // Return the largest index i such that drange->dates[i] <= date.
    // Use a half-open binary search to avoid size_t underflow.
    size_t lo = 0;
    size_t hi = drange->n_days;  // exclusive
    while (lo < hi) {
        size_t mid = lo + ((hi - lo) >> 1);
        int cmp = c_ex_profile_session_date_compare(&drange->dates[mid], date);
        if (cmp <= 0) lo = mid + 1u;
        else hi = mid;
    }
    // lo is the first index where drange->dates[lo] > date, so answer is lo-1.
    return lo - 1u;
}

static inline session_date_range_t* c_ex_profile_session_date_between(session_date_t* start_date, session_date_t* end_date) {
    if (!start_date || !end_date) return NULL;
    // Cache contains trading days only. If no cache, fall back to natural calendar day range.
    if (!EX_TRADE_CALENDAR_CACHE) return c_ex_profile_date_range_between(start_date, end_date);
    if (EX_TRADE_CALENDAR_CACHE->n_days == 0u) return NULL;

    // Normalize order.
    if (c_ex_profile_session_date_compare(start_date, end_date) == 1) {
        session_date_t* tmp = start_date;
        start_date = end_date;
        end_date = tmp;
    }

    size_t start_idx = c_ex_profile_session_date_index(start_date, EX_TRADE_CALENDAR_CACHE);
    size_t end_idx = c_ex_profile_session_date_index(end_date, EX_TRADE_CALENDAR_CACHE);
    if (start_idx == (size_t) -1 || end_idx == (size_t) -1) return NULL;

    // Convert floor-index into a [first >= start_date] index.
    if (c_ex_profile_session_date_compare(EX_TRADE_CALENDAR_CACHE->dates + start_idx, start_date) == -1) {
        if (start_idx + 1u >= EX_TRADE_CALENDAR_CACHE->n_days) return NULL;
        start_idx += 1u;
    }

    // c_ex_profile_session_date_index() is a floor/left indexer, but it clamps "date < start" to 0.
    // Reject the clamp-at-0 case where even dates[0] is after end_date (no cached date <= end_date).
    if (end_idx == 0u && c_ex_profile_session_date_compare(EX_TRADE_CALENDAR_CACHE->dates, end_date) == 1) return NULL;

    if (start_idx > end_idx) return NULL;

    size_t n_days = (size_t) (end_idx - start_idx + 1u);
    session_date_range_t* drange = (session_date_range_t*) calloc(1, sizeof(session_date_range_t) + n_days * sizeof(session_date_t));
    if (!drange) return NULL;

    drange->start = EX_TRADE_CALENDAR_CACHE->dates[start_idx];
    drange->end = EX_TRADE_CALENDAR_CACHE->dates[end_idx];
    drange->n_days = n_days;

    for (size_t i = 0u; i < n_days; ++i) {
        drange->dates[i] = EX_TRADE_CALENDAR_CACHE->dates[start_idx + i];
    }

    return drange;
}

static inline int c_ex_profile_session_trading_days_before(session_date_t* market_date, size_t days, session_date_t* out) {
    if (!market_date || !out || days == 0) return -1;

    if (!EX_TRADE_CALENDAR_CACHE) return c_ex_profile_days_before(market_date, days, out);
    if (EX_TRADE_CALENDAR_CACHE->n_days == 0u) return -1;

    size_t idx = c_ex_profile_session_date_index(market_date, EX_TRADE_CALENDAR_CACHE);
    if (idx == (size_t) -1) return -1;

    // If market_date is not a trading day, idx is the floor trading date (dates[idx] < market_date).
    // In that case, treat dates[idx] as "1 trading day before market_date".
    int cmp = c_ex_profile_session_date_compare(EX_TRADE_CALENDAR_CACHE->dates + idx, market_date);
    if (cmp == -1) days--;
    if (cmp == 1) return -1;  // market_date is before the first trading day in the cache

    size_t target = (idx > days) ? (idx - days) : 0u;
    *out = EX_TRADE_CALENDAR_CACHE->dates[target];
    return 0;
}

static inline int c_ex_profile_session_trading_days_after(session_date_t* market_date, size_t days, session_date_t* out) {
    if (!market_date || !out || days == 0) return -1;

    if (!EX_TRADE_CALENDAR_CACHE) return c_ex_profile_days_after(market_date, days, out);
    if (EX_TRADE_CALENDAR_CACHE->n_days == 0u) return -1;

    size_t idx = c_ex_profile_session_date_index(market_date, EX_TRADE_CALENDAR_CACHE);
    if (idx == (size_t) -1) return -1;
    int cmp = c_ex_profile_session_date_compare(EX_TRADE_CALENDAR_CACHE->dates + idx, market_date);
    if (cmp == 1) {
        size_t max_idx = EX_TRADE_CALENDAR_CACHE->n_days - 1u;
        size_t offset = days - 1u;
        size_t target = (offset > max_idx) ? max_idx : offset;
        *out = EX_TRADE_CALENDAR_CACHE->dates[target];
        return 0;
    }
    else {
        size_t max_idx = EX_TRADE_CALENDAR_CACHE->n_days - 1u;
        size_t target = (days > (max_idx - idx)) ? max_idx : (idx + days);
        *out = EX_TRADE_CALENDAR_CACHE->dates[target];
        return 0;
    }
}

static inline int c_ex_profile_nearest_trading_date(session_date_t* market_date, bool previous, session_date_t* out) {
    if (!market_date || !out) return -1;

    if (!EX_TRADE_CALENDAR_CACHE) {
        *out = *market_date;
        out->session_type = EX_PROFILE ? EX_PROFILE->resolve_session_type_func(out->year, out->month, out->day) : SESSION_TYPE_NORMINAL;
        return c_ex_profile_date_is_valid(out) ? 0 : -1;
    }
    if (EX_TRADE_CALENDAR_CACHE->n_days == 0u) return -1;

    size_t idx = c_ex_profile_session_date_index(market_date, EX_TRADE_CALENDAR_CACHE);
    if (idx == (size_t) -1) return -1;
    int cmp = c_ex_profile_session_date_compare(EX_TRADE_CALENDAR_CACHE->dates + idx, market_date);

    if (cmp == 0) {
        *out = EX_TRADE_CALENDAR_CACHE->dates[idx];
        return 0;
    }
    else if (cmp == -1) {
        // market_date is after dates[idx], so nearest is either idx or idx + 1
        if (previous) {
            *out = EX_TRADE_CALENDAR_CACHE->dates[idx];
            return 0;
        }
        else if (idx + 1u >= EX_TRADE_CALENDAR_CACHE->n_days) return -1;
        else {
            *out = EX_TRADE_CALENDAR_CACHE->dates[idx + 1u];
            return 0;
        }
    }
    else {
        // market_date is before dates[idx], the only possible case is that idx == 0
        if (previous) return -1;
        else {
            *out = *EX_TRADE_CALENDAR_CACHE->dates;
            return 0;
        }
    }
}

#endif /* C_EX_PROFILE_BASE_H */
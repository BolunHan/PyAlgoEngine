#ifndef C_MARKET_DATA_H
#define C_MARKET_DATA_H

#include <stdint.h>
#include "c_market_data_config.h"

#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
#else
#include <unistd.h>
#endif

// ========== Constants ==========

static const int8_t sign_lut[4] = {
    -1,  // 0b00 -> -1
    0,   // 0b01 -> 0
    1,   // 0b10 -> 1
    0    // 0b11 -> 0
};

// ========== Enums ==========

typedef enum direction_t {
    DIRECTION_SHORT     = 0,
    DIRECTION_UNKNOWN   = 1,
    DIRECTION_LONG      = 2,
    DIRECTION_NEUTRAL   = 3
} direction_t;

// Offset Enum
typedef enum offset_t {
    OFFSET_CANCEL       = 0,
    OFFSET_ORDER        = 4,
    OFFSET_OPEN         = 8,
    OFFSET_CLOSE        = 16
} offset_t;

// Side Enum (bitwise composition of direction + offset)
typedef enum side_t {
    // Long Side transaction
    SIDE_LONG_OPEN      = DIRECTION_LONG + OFFSET_OPEN,
    SIDE_LONG_CLOSE     = DIRECTION_LONG + OFFSET_CLOSE,
    SIDE_LONG_CANCEL    = DIRECTION_LONG + OFFSET_CANCEL,

    // Short Side transaction
    SIDE_SHORT_OPEN     = DIRECTION_SHORT + OFFSET_OPEN,
    SIDE_SHORT_CLOSE    = DIRECTION_SHORT + OFFSET_CLOSE,
    SIDE_SHORT_CANCEL   = DIRECTION_SHORT + OFFSET_CANCEL,

    // NEUTRAL transaction
    SIDE_NEUTRAL_OPEN   = DIRECTION_NEUTRAL + OFFSET_OPEN,
    SIDE_NEUTRAL_CLOSE  = DIRECTION_NEUTRAL + OFFSET_CLOSE,

    // Order
    SIDE_BID            = DIRECTION_LONG + OFFSET_ORDER,
    SIDE_ASK            = DIRECTION_SHORT + OFFSET_ORDER,

    // Generic Cancel
    SIDE_CANCEL         = DIRECTION_UNKNOWN + OFFSET_CANCEL,

    // Alias
    SIDE_UNKNOWN        = SIDE_CANCEL,
    SIDE_LONG           = SIDE_LONG_OPEN,
    SIDE_SHORT          = SIDE_SHORT_OPEN
} side_t;

typedef enum order_type_t {
    ORDER_UNKNOWN       = 2,
    ORDER_CANCEL        = 1,
    ORDER_GENERIC       = 0,
    ORDER_LIMIT         = 10,
    ORDER_LIMIT_MAKER   = 11,
    ORDER_MARKET        = 20,
    ORDER_FOK           = 21,
    ORDER_FAK           = 22,
    ORDER_IOC           = 23
} order_type_t;

typedef enum order_state_t {
    STATE_UNKNOWN       = 0,
    STATE_REJECTED      = 1,
    STATE_INVALID       = 2,
    STATE_PENDING       = 3,
    STATE_SENT          = 4,
    STATE_PLACED        = 5,
    STATE_PARTFILLED    = 6,
    STATE_FILLED        = 7,
    STATE_CANCELING     = 8,
    STATE_CANCELED      = 9
} order_state_t;

typedef enum data_type_t {
    DTYPE_UNKNOWN       = 0,
    DTYPE_INTERNAL      = 1,
    DTYPE_MARKET_DATA   = 10,
    DTYPE_TRANSACTION   = 20,
    DTYPE_ORDER         = 30,
    DTYPE_TICK_LITE     = 31,
    DTYPE_TICK          = 32,
    DTYPE_BAR           = 40,
    DTYPE_REPORT        = 50,
    DTYPE_INSTRUCTION   = 51
} data_type_t;

typedef enum filter_mode_t {
    NO_INTERNAL         = 1 << 0,
    NO_CANCEL           = 1 << 1,
    NO_AUCTION          = 1 << 2,
    NO_ORDER            = 1 << 3,
    NO_TRADE            = 1 << 4,
    NO_TICK             = 1 << 5
} filter_mode_t;

// ========== Structs ==========

typedef struct meta_info_t {
    uint8_t dtype;
    char* ticker;
    double timestamp;
} meta_info_t;

typedef struct id_t {
    uint8_t id_type;
    char data[ID_SIZE];
} id_t;

typedef struct long_id_t {
    uint8_t id_type;
    char data[LONG_ID_SIZE];
} long_id_t;

typedef struct internal_buffer_t {
    uint8_t dtype;
    char ticker[TICKER_SIZE];
    double timestamp;
    uint32_t code;
} internal_buffer_t;

typedef struct order_book_entry_t {
    double price;
    double volume;
    uint64_t n_orders;
} order_book_entry_t;

typedef struct order_book_buffer_t {
    order_book_entry_t entries[BOOK_SIZE];
} order_book_buffer_t;

typedef struct candlestick_buffer_t {
    meta_info_t meta_info;
    double bar_span;
    double high_price;
    double low_price;
    double open_price;
    double close_price;
    double volume;
    double notional;
    uint64_t trade_count;
} candlestick_buffer_t;

typedef struct tick_data_lite_buffer_t {
    meta_info_t meta_info;
    double bid_price;
    double bid_volume;
    double ask_price;
    double ask_volume;
    double last_price;
    double open_price;
    double prev_close;
    double total_traded_volume;
    double total_traded_notional;
    uint64_t total_trade_count;
} tick_data_lite_buffer_t;

typedef struct tick_data_buffer_t {
    tick_data_lite_buffer_t lite;
    double total_bid_volume;
    double total_ask_volume;
    double weighted_bid_price;
    double weighted_ask_price;
    order_book_buffer_t bid;
    order_book_buffer_t ask;
} tick_data_buffer_t;

typedef struct transaction_data_buffer_t {
    meta_info_t meta_info;
    double price;
    double volume;
    uint8_t side;
    double multiplier;
    double notional;
    id_t transaction_id;
    id_t buy_id;
    id_t sell_id;
} transaction_data_buffer_t;

typedef struct order_data_buffer_t {
    meta_info_t meta_info;
    double price;
    double volume;
    uint8_t side;
    id_t order_id;
    uint8_t order_type;
} order_data_buffer_t;

typedef struct trade_report_buffer_t {
    meta_info_t meta_info;
    double price;
    double volume;
    uint8_t side;
    double multiplier;
    double notional;
    double fee;
    long_id_t order_id;
    long_id_t trade_id;
} trade_report_buffer_t;

typedef struct trade_instruction_buffer_t {
    meta_info_t meta_info;
    double limit_price;
    double volume;
    uint8_t side;
    long_id_t order_id;
    int32_t order_type;
    double multiplier;
    uint8_t order_state;
    double filled_volume;
    double filled_notional;
    double fee;
    double ts_placed;
    double ts_canceled;
    double ts_finished;
} trade_instruction_buffer_t;

typedef union market_data_buffer_t {
    meta_info_t meta_info;
    internal_buffer_t internal;
    transaction_data_buffer_t transaction_data;
    order_data_buffer_t order_data;
    candlestick_buffer_t bar_data;
    tick_data_lite_buffer_t tick_data_lite;
    tick_data_buffer_t tick_data_full;
    trade_report_buffer_t trade_report;
    trade_instruction_buffer_t trade_instruction;
} market_data_buffer_t;

// ========== Utility Functions ==========

static inline int8_t direction_to_sign(uint8_t x) {
    return sign_lut[x & 0b11];
}

static inline void platform_usleep(unsigned int usec) {
#if defined(_WIN32) || defined(_WIN64)
    Sleep(usec / 1000);  // Windows: Sleep in milliseconds
#else
    usleep(usec);        // POSIX: Sleep in microseconds
#endif
}

static inline int compare_md_ptr(const void* a, const void* b) {
    const meta_info_t* md_a = *(const meta_info_t* const*) a;
    const meta_info_t* md_b = *(const meta_info_t* const*) b;

    if (md_a->timestamp < md_b->timestamp) return -1;
    if (md_a->timestamp > md_b->timestamp) return 1;
    return 0;
}

static inline int compare_entries_bid(const void* a, const void* b) {
    const order_book_entry_t* entry_a = (const order_book_entry_t*) a;
    const order_book_entry_t* entry_b = (const order_book_entry_t*) b;

    if (entry_a->n_orders == 0 && entry_b->n_orders == 0) return 0;
    if (entry_b->n_orders == 0) return -1;
    if (entry_a->n_orders == 0) return 1;

    if (entry_a->price < entry_b->price) return 1;
    if (entry_a->price > entry_b->price) return -1;
    return 0;
}

static inline int compare_entries_ask(const void* a, const void* b) {
    const order_book_entry_t* entry_a = (const order_book_entry_t*) a;
    const order_book_entry_t* entry_b = (const order_book_entry_t*) b;

    if (entry_a->n_orders == 0 && entry_b->n_orders == 0) return 0;
    if (entry_b->n_orders == 0) return -1;
    if (entry_a->n_orders == 0) return 1;

    if (entry_a->price < entry_b->price) return -1;
    if (entry_a->price > entry_b->price) return 1;
    return 0;
}

#endif // C_MARKET_DATA_H
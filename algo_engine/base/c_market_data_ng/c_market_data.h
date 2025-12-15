#ifndef C_MARKET_DATA_H
#define C_MARKET_DATA_H

#include <math.h>
#include <stdint.h>
#include <string.h>

#include "c_market_data_config.h"
#include "c_heap_allocator.h"
#include "c_shm_allocator.h"

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

static const char internal_dtype_name[]     = "InternalData";
static const char transaction_dtype_name[]  = "TransactionData";
static const char order_dtype_name[]        = "OrderData";
static const char tick_lite_dtype_name[]    = "TickDataLite";
static const char tick_dtype_name[]         = "TickData";
static const char bar_dtype_name[]          = "BarData";
static const char report_dtype_name[]       = "TradeReport";
static const char instruction_dtype_name[]  = "TradeInstruction";
static const char generic_dtype_name[]      = "GenericMarketData";

#define DTYPE_MIN_SIZE (sizeof(internal_t))
#define DTYPE_MAX_SIZE (sizeof(market_data_t))

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

typedef enum mid_type_t {
    MID_UNKNOWN         = 0,
    MID_INT             = 1,
    MID_STRING          = 2,
    MID_BYTE            = 3,
    MID_UUID            = 4
} mid_type_t;

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
    data_type_t dtype;
    const char* ticker;
    double timestamp;
    shm_allocator_t* shm_allocator;
    heap_allocator_t* heap_allocator;
} meta_info_t;

typedef struct mid_t {
    mid_type_t id_type;
    char data[ID_SIZE];
} mid_t;

typedef struct long_mid_t {
    mid_type_t id_type;
    char data[LONG_ID_SIZE];
} long_mid_t;

typedef struct internal_t {
    meta_info_t meta_info;
    uint32_t code;
} internal_t;

typedef struct order_book_entry_t {
    double price;
    double volume;
    uint64_t n_orders;
} order_book_entry_t;

typedef struct order_book_t {
    order_book_entry_t entries[BOOK_SIZE];
} order_book_t;

typedef struct candlestick_t {
    meta_info_t meta_info;
    double bar_span;
    double high_price;
    double low_price;
    double open_price;
    double close_price;
    double volume;
    double notional;
    uint64_t trade_count;
} candlestick_t;

typedef struct tick_data_lite_t {
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
} tick_data_lite_t;

typedef struct tick_data_t {
    tick_data_lite_t lite;
    double total_bid_volume;
    double total_ask_volume;
    double weighted_bid_price;
    double weighted_ask_price;
    order_book_t bid;
    order_book_t ask;
} tick_data_t;

typedef struct transaction_data_t {
    meta_info_t meta_info;
    double price;
    double volume;
    side_t side;
    double multiplier;
    double notional;
    mid_t transaction_id;
    mid_t buy_id;
    mid_t sell_id;
} transaction_data_t;

typedef struct order_data_t {
    meta_info_t meta_info;
    double price;
    double volume;
    side_t side;
    mid_t order_id;
    order_type_t order_type;
} order_data_t;

typedef struct trade_report_t {
    meta_info_t meta_info;
    double price;
    double volume;
    side_t side;
    double multiplier;
    double notional;
    double fee;
    long_mid_t order_id;
    long_mid_t trade_id;
} trade_report_t;

typedef struct trade_instruction_t {
    meta_info_t meta_info;
    double limit_price;
    double volume;
    side_t side;
    long_mid_t order_id;
    order_type_t order_type;
    double multiplier;
    order_state_t order_state;
    double filled_volume;
    double filled_notional;
    double fee;
    double ts_placed;
    double ts_canceled;
    double ts_finished;
} trade_instruction_t;

typedef union market_data_t {
    meta_info_t meta_info;
    internal_t internal;
    transaction_data_t transaction_data;
    order_data_t order_data;
    candlestick_t bar_data;
    tick_data_lite_t tick_data_lite;
    tick_data_t tick_data_full;
    trade_report_t trade_report;
    trade_instruction_t trade_instruction;
} market_data_t;

// ========== Forward Declarations (Public API) ==========

/**
 * @brief Sleep for a number of microseconds (cross-platform).
 * @param usec Microseconds to sleep.
 */
static inline void c_usleep(unsigned int usec);

/**
 * @brief Allocate a new market_data buffer of the specified dtype.
 * @param dtype Concrete data type to allocate.
 * @param shm_allocator Shared-memory allocator context, or NULL.
 * @param heap_allocator Heap allocator, or NULL.
 * @param with_lock Whether to lock allocator during allocation.
 * @return Pointer to allocated `market_data_t`, or NULL on failure.
 */
static inline market_data_t* c_md_new(data_type_t dtype, shm_allocator_ctx* shm_allocator, heap_allocator_t* heap_allocator, int with_lock);

/**
 * @brief Free or recycle a previously allocated market_data buffer.
 * @param market_data Pointer returned by `c_md_new`.
 * @param with_lock Whether to lock allocator during free.
 */
static inline void c_md_free(market_data_t* market_data, int with_lock);

/**
 * @brief Get the representative price from a market_data union.
 * @param market_data The data buffer.
 * @return Extracted price or 0.0 if unavailable.
 */
static inline double c_md_get_price(const market_data_t* market_data);

/**
 * @brief Extract the offset bits from a composed `side_t`.
 * @param side Composed side value.
 * @return `offset_t` component.
 */
static inline offset_t c_md_get_offset(side_t side);

/**
 * @brief Extract the direction bits from a composed `side_t`.
 * @param side Composed side value.
 * @return `direction_t` component.
 */
static inline direction_t c_md_get_direction(side_t side);

/**
 * @brief Map a direction to its sign (-1, 0, 1).
 * @param x Direction value.
 * @return Sign as int8_t.
 */
static inline int8_t c_md_get_sign(direction_t x);

/**
 * @brief Get the size in bytes of a concrete dtype.
 * @param dtype Data type.
 * @return Size of the corresponding struct/union.
 */
static inline size_t c_md_get_size(data_type_t dtype);

/**
 * @brief Get the human-readable name of a dtype.
 * @param dtype Data type.
 * @return Constant string name, or NULL if unknown.
 */
static inline const char* c_md_dtype_name(data_type_t dtype);

/**
 * @brief Compute serialized size of a market_data buffer.
 * @param market_data Buffer to measure.
 * @return Total bytes required for serialization.
 */
static inline size_t c_md_serialized_size(const market_data_t* market_data);

/**
 * @brief Serialize market_data into a contiguous buffer.
 * @param market_data Source buffer.
 * @param out Destination byte buffer (preallocated).
 * @return Number of bytes written.
 */
static inline size_t c_md_serialize(const market_data_t* market_data, char* out);

/**
 * @brief Deserialize market_data from a contiguous buffer.
 * @param src Source byte buffer.
 * @param shm_allocator Shared allocator context, or NULL.
 * @param heap_allocator Heap allocator, or NULL.
 * @param with_lock Whether to lock allocator during allocation.
 * @return Newly allocated `market_data_t*`, or NULL on failure.
 */
static inline market_data_t* c_md_deserialize(const char* src, shm_allocator_ctx* shm_allocator, heap_allocator_t* heap_allocator, int with_lock);

/**
 * @brief Compare two meta_info pointers by timestamp (ascending).
 * @param a Pointer to pointer of meta_info_t.
 * @param b Pointer to pointer of meta_info_t.
 * @return -1, 0, or 1 like strcmp semantics.
 */
static inline int c_md_compare_ptr(const void* a, const void* b);

/**
 * @brief Order-book comparator for bids (descending price, non-empty first).
 * @param a Pointer to order_book_entry_t.
 * @param b Pointer to order_book_entry_t.
 * @return -1, 0, or 1 for sorting.
 */
static inline int c_md_compare_bid(const void* a, const void* b);

/**
 * @brief Order-book comparator for asks (ascending price, non-empty first).
 * @param a Pointer to order_book_entry_t.
 * @param b Pointer to order_book_entry_t.
 * @return -1, 0, or 1 for sorting.
 */
static inline int c_md_compare_ask(const void* a, const void* b);

// ========== Utility Functions ==========

static inline void c_usleep(unsigned int usec) {
#if defined(_WIN32) || defined(_WIN64)
    Sleep(usec / 1000);  // Windows: Sleep in milliseconds
#else
    usleep(usec);        // POSIX: Sleep in microseconds
#endif
}

static inline market_data_t* c_md_new(data_type_t dtype, shm_allocator_ctx* shm_allocator, heap_allocator_t* heap_allocator, int with_lock) {
    size_t size = c_md_get_size(dtype);
    if (size == 0) return NULL;

    if (shm_allocator) {
        pthread_mutex_t* lock = with_lock ? &shm_allocator->shm_allocator->lock : NULL;
        market_data_t* market_data = (market_data_t*) c_shm_request(shm_allocator, size, 0, lock);
        market_data->meta_info.dtype = dtype;
        market_data->meta_info.shm_allocator = shm_allocator->shm_allocator;
        return market_data;
    }
    else if (heap_allocator) {
        pthread_mutex_t* lock = with_lock ? &heap_allocator->lock : NULL;
        market_data_t* market_data = (market_data_t*) c_heap_request(heap_allocator, size, 0, lock);
        market_data->meta_info.dtype = dtype;
        market_data->meta_info.heap_allocator = heap_allocator;
        return market_data;
    }
    else {
        market_data_t* market_data = (market_data_t*) calloc(1, size);
        market_data->meta_info.dtype = dtype;
        return market_data;
    }
}

static inline void c_md_free(market_data_t* market_data, int with_lock) {
    if (!market_data) return;

    shm_allocator_t* shm_allocator = market_data->meta_info.shm_allocator;
    heap_allocator_t* heap_allocator = market_data->meta_info.heap_allocator;

    if (shm_allocator) {
        pthread_mutex_t* lock = with_lock ? &shm_allocator->lock : NULL;
        c_shm_free((void*) market_data, lock);
    }
    else if (heap_allocator) {
        pthread_mutex_t* lock = with_lock ? &heap_allocator->lock : NULL;
        c_heap_free((void*) market_data, lock);
    }
    else {
        free((void*) market_data);
    }
}

static inline double c_md_get_price(const market_data_t* market_data) {
    if (!market_data) return 0.0;

    switch (market_data->meta_info.dtype) {
        case DTYPE_INTERNAL:
            return 0.0;
        case DTYPE_TRANSACTION:
            return market_data->transaction_data.price;
        case DTYPE_ORDER:
            return market_data->order_data.price;
        case DTYPE_TICK_LITE:
            return market_data->tick_data_lite.last_price;
        case DTYPE_TICK:
            return market_data->tick_data_full.lite.last_price;
        case DTYPE_BAR:
            return market_data->bar_data.close_price;
        case DTYPE_REPORT:
            return market_data->trade_report.price;
        case DTYPE_INSTRUCTION:
            return market_data->trade_instruction.limit_price;
        case DTYPE_UNKNOWN:
        case DTYPE_MARKET_DATA:
        default:
            return 0.0;
    }
}

static inline offset_t c_md_get_offset(side_t side) {
    return (offset_t) (side & 0xFC);
}

static inline direction_t c_md_get_direction(side_t side) {
    return (direction_t) (side & 0x03);
}

static inline int8_t c_md_get_sign(direction_t x) {
    return sign_lut[x & 0b11];
}

static inline size_t c_md_get_size(data_type_t dtype) {
    switch (dtype) {
        case DTYPE_INTERNAL:
            return sizeof(internal_t);
        case DTYPE_TRANSACTION:
            return sizeof(transaction_data_t);
        case DTYPE_ORDER:
            return sizeof(order_data_t);
        case DTYPE_TICK_LITE:
            return sizeof(tick_data_lite_t);
        case DTYPE_TICK:
            return sizeof(tick_data_t);
        case DTYPE_BAR:
            return sizeof(candlestick_t);
        case DTYPE_REPORT:
            return sizeof(trade_report_t);
        case DTYPE_INSTRUCTION:
            return sizeof(trade_instruction_t);
        case DTYPE_UNKNOWN:
        case DTYPE_MARKET_DATA:
            return sizeof(market_data_t);
        default:
            return 0;
    }
}

static inline const char* c_md_dtype_name(data_type_t dtype) {
    switch (dtype) {
        case DTYPE_INTERNAL:
            return internal_dtype_name;
        case DTYPE_TRANSACTION:
            return transaction_dtype_name;
        case DTYPE_ORDER:
            return order_dtype_name;
        case DTYPE_TICK_LITE:
            return tick_lite_dtype_name;
        case DTYPE_TICK:
            return tick_dtype_name;
        case DTYPE_BAR:
            return bar_dtype_name;
        case DTYPE_REPORT:
            return report_dtype_name;
        case DTYPE_INSTRUCTION:
            return instruction_dtype_name;
        case DTYPE_UNKNOWN:
        case DTYPE_MARKET_DATA:
            return generic_dtype_name;
        default:
            return NULL;
    }
}

static inline size_t c_md_serialized_size(const market_data_t* market_data) {
    if (!market_data) return 0;

    const meta_info_t* meta = &market_data->meta_info;
    const char* ticker = meta->ticker ? meta->ticker : "";
    const size_t ticker_len = strlen(ticker);

    size_t payload_size = 0;
    switch (meta->dtype) {
        case DTYPE_INTERNAL:      payload_size = sizeof(internal_t) - sizeof(meta_info_t); break;
        case DTYPE_TRANSACTION:   payload_size = sizeof(transaction_data_t) - sizeof(meta_info_t); break;
        case DTYPE_ORDER:         payload_size = sizeof(order_data_t) - sizeof(meta_info_t); break;
        case DTYPE_TICK_LITE:     payload_size = sizeof(tick_data_lite_t) - sizeof(meta_info_t); break;
        case DTYPE_TICK:          payload_size = sizeof(tick_data_t) - sizeof(meta_info_t); break;
        case DTYPE_BAR:           payload_size = sizeof(candlestick_t) - sizeof(meta_info_t); break;
        case DTYPE_REPORT:        payload_size = sizeof(trade_report_t) - sizeof(meta_info_t); break;
        case DTYPE_INSTRUCTION:   payload_size = sizeof(trade_instruction_t) - sizeof(meta_info_t); break;
        case DTYPE_UNKNOWN:
        case DTYPE_MARKET_DATA:
        default:
            payload_size = 0;
            break;
    }

    return sizeof(uint8_t)      /* dtype */
        + sizeof(double)        /* timestamp */
        + sizeof(uint32_t)      /* ticker length */
        + ticker_len  + 1       /* ticker bytes with nul terminator*/
        + payload_size;         /* payload without meta_info */
}

static inline size_t c_md_serialize(const market_data_t* market_data, char* out) {
    // [uint8 dtype][double timestamp][uint32 ticker_len][ticker bytes][payload w/o meta_info].
    if (!market_data || !out) return 0;

    const meta_info_t* meta = &market_data->meta_info;
    const char* ticker = meta->ticker ? meta->ticker : "";
    const uint32_t ticker_len = (uint32_t) strlen(ticker);

    char* cursor = out;

    const uint8_t dtype_byte = (uint8_t) meta->dtype;
    memcpy(cursor, &dtype_byte, sizeof(dtype_byte));
    cursor += sizeof(uint8_t);

    memcpy(cursor, &meta->timestamp, sizeof(meta->timestamp));
    cursor += sizeof(double);

    memcpy(cursor, &ticker_len, sizeof(ticker_len));
    cursor += sizeof(uint32_t);

    if (ticker_len) {
        memcpy(cursor, ticker, ticker_len);
        cursor += ticker_len;
    }

    // Null-terminate ticker
    cursor[0] = '\0';
    cursor++;

#define WRITE_PAYLOAD(member_type, member_name) \
    do { \
        const member_type* p = &market_data->member_name; \
        memcpy(cursor, ((const char*) p) + sizeof(meta_info_t), sizeof(member_type) - sizeof(meta_info_t)); \
        cursor += sizeof(member_type) - sizeof(meta_info_t); \
    } while (0)

    switch (meta->dtype) {
        case DTYPE_INTERNAL:      WRITE_PAYLOAD(internal_t, internal); break;
        case DTYPE_TRANSACTION:   WRITE_PAYLOAD(transaction_data_t, transaction_data); break;
        case DTYPE_ORDER:         WRITE_PAYLOAD(order_data_t, order_data); break;
        case DTYPE_TICK_LITE:     WRITE_PAYLOAD(tick_data_lite_t, tick_data_lite); break;
        case DTYPE_TICK:          WRITE_PAYLOAD(tick_data_t, tick_data_full); break;
        case DTYPE_BAR:           WRITE_PAYLOAD(candlestick_t, bar_data); break;
        case DTYPE_REPORT:        WRITE_PAYLOAD(trade_report_t, trade_report); break;
        case DTYPE_INSTRUCTION:   WRITE_PAYLOAD(trade_instruction_t, trade_instruction); break;
        case DTYPE_UNKNOWN:
        case DTYPE_MARKET_DATA:
        default:
            // Nothing else to copy for unknown/aggregate.
            break;
    }

#undef WRITE_PAYLOAD
    return (size_t) (cursor - out);
}

static inline market_data_t* c_md_deserialize(const char* src, shm_allocator_ctx* shm_allocator, heap_allocator_t* heap_allocator, int with_lock) {
    if (!src) return NULL;

    const char* cursor = src;

    // Step 1: Read dtype
    data_type_t dtype = (data_type_t) cursor[0];
    cursor += sizeof(uint8_t);

    // Step 2: Allocate market_data_t
    market_data_t* market_data = c_md_new(dtype, shm_allocator, heap_allocator, with_lock);
    if (!market_data) return NULL;

    // Step 3: Deserialize meta_info
    meta_info_t* meta = &market_data->meta_info;
    // dtype already assigned
    // meta->dtype = dtype;
    memcpy(&meta->timestamp, cursor, sizeof(double));
    cursor += sizeof(double);

    uint32_t ticker_len = 0;
    memcpy(&ticker_len, cursor, sizeof(uint32_t));
    cursor += sizeof(uint32_t);

    // Use a borrowed ptr for ticker, if it needs to be owned, user should intern or strdup it later.
    const char* ticker = cursor;
    meta->ticker = ticker_len > 0 ? ticker : NULL;
    cursor += ticker_len + 1;

    // Step 4: Deserialize payload
#define READ_PAYLOAD(member_type, member_name) \
    do { \
        member_type* p = &market_data->member_name; \
        memcpy(((char*) p) + sizeof(meta_info_t), cursor, sizeof(member_type) - sizeof(meta_info_t)); \
        cursor += sizeof(member_type) - sizeof(meta_info_t); \
    } while (0)
    switch (meta->dtype) {
        case DTYPE_INTERNAL:      READ_PAYLOAD(internal_t, internal); break;
        case DTYPE_TRANSACTION:   READ_PAYLOAD(transaction_data_t, transaction_data); break;
        case DTYPE_ORDER:         READ_PAYLOAD(order_data_t, order_data); break;
        case DTYPE_TICK_LITE:     READ_PAYLOAD(tick_data_lite_t, tick_data_lite); break;
        case DTYPE_TICK:          READ_PAYLOAD(tick_data_t, tick_data_full); break;
        case DTYPE_BAR:           READ_PAYLOAD(candlestick_t, bar_data); break;
        case DTYPE_REPORT:        READ_PAYLOAD(trade_report_t, trade_report); break;
        case DTYPE_INSTRUCTION:   READ_PAYLOAD(trade_instruction_t, trade_instruction); break;
        case DTYPE_UNKNOWN:
        case DTYPE_MARKET_DATA:
        default:
            // Nothing else to copy for unknown/aggregate.
            break;
    }
#undef READ_PAYLOAD
    return market_data;
}

static inline int c_md_compare_ptr(const void* a, const void* b) {
    const meta_info_t* md_a = *(const meta_info_t* const*) a;
    const meta_info_t* md_b = *(const meta_info_t* const*) b;

    if (md_a->timestamp < md_b->timestamp) return -1;
    if (md_a->timestamp > md_b->timestamp) return 1;
    return 0;
}

static inline int c_md_compare_bid(const void* a, const void* b) {
    const order_book_entry_t* entry_a = (const order_book_entry_t*) a;
    const order_book_entry_t* entry_b = (const order_book_entry_t*) b;

    if (entry_a->n_orders == 0 && entry_b->n_orders == 0) return 0;
    if (entry_b->n_orders == 0) return -1;
    if (entry_a->n_orders == 0) return 1;

    if (entry_a->price < entry_b->price) return 1;
    if (entry_a->price > entry_b->price) return -1;
    return 0;
}

static inline int c_md_compare_ask(const void* a, const void* b) {
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
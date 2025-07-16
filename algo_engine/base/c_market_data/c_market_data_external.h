#ifndef C_MARKET_DATA_H
#define C_MARKET_DATA_H

#include <stdint.h>
#include "c_market_data_config.h"

#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
#else
#include <unistd.h>
#endif

// Direction Enum
typedef enum {
    DIRECTION_SHORT = 0,
    DIRECTION_UNKNOWN = 1,
    DIRECTION_LONG = 2,
    DIRECTION_NEUTRAL = 3
} Direction;

// Offset Enum
typedef enum {
    OFFSET_CANCEL = 0,
    OFFSET_ORDER = 4,
    OFFSET_OPEN = 8,
    OFFSET_CLOSE = 16
} Offset;

// Side Enum (bitwise composition of direction + offset)
typedef enum {
    // Long Side transaction
    SIDE_LONG_OPEN = DIRECTION_LONG + OFFSET_OPEN,
    SIDE_LONG_CLOSE = DIRECTION_LONG + OFFSET_CLOSE,
    SIDE_LONG_CANCEL = DIRECTION_LONG + OFFSET_CANCEL,

    // Short Side transaction
    SIDE_SHORT_OPEN = DIRECTION_SHORT + OFFSET_OPEN,
    SIDE_SHORT_CLOSE = DIRECTION_SHORT + OFFSET_CLOSE,
    SIDE_SHORT_CANCEL = DIRECTION_SHORT + OFFSET_CANCEL,

    // NEUTRAL transaction
    SIDE_NEUTRAL_OPEN = DIRECTION_NEUTRAL + OFFSET_OPEN,
    SIDE_NEUTRAL_CLOSE = DIRECTION_NEUTRAL + OFFSET_CLOSE,

    // Order
    SIDE_BID = DIRECTION_LONG + OFFSET_ORDER,
    SIDE_ASK = DIRECTION_SHORT + OFFSET_ORDER,

    // Generic Cancel
    SIDE_CANCEL = DIRECTION_UNKNOWN + OFFSET_CANCEL,

    // Alias
    SIDE_UNKNOWN = SIDE_CANCEL,
    SIDE_LONG = SIDE_LONG_OPEN,
    SIDE_SHORT = SIDE_SHORT_OPEN
} Side;

// Structs
typedef struct {
    uint8_t dtype;
    char ticker[TICKER_SIZE];
    double timestamp;
} __attribute__((packed)) MetaInfo;

typedef struct {
    double price;
    double volume;
    uint64_t n_orders;
} __attribute__((packed)) Entry;

// Small performance-critical function as static inline in header
static const int8_t SIGN_LUT[4] = {
    -1,  // 0b00 → -1
    0,   // 0b01 → 0
    1,   // 0b10 → 1
    0    // 0b11 → 0
};

static inline int8_t direction_to_sign(uint8_t x) {
    return SIGN_LUT[x & 0b11];
};

static inline void platform_usleep(unsigned int usec) {
#if defined(_WIN32) || defined(_WIN64)
    Sleep(usec / 1000);  // Windows: Sleep in milliseconds
#else
    usleep(usec);        // POSIX: Sleep in microseconds
#endif
};

static inline int compare_md_ptr(const void *a, const void *b) {
    const MetaInfo *md_a = *(const MetaInfo **)a;
    const MetaInfo *md_b = *(const MetaInfo **)b;

    if (md_a->timestamp < md_b->timestamp) return -1;
    if (md_a->timestamp > md_b->timestamp) return 1;
    return 0;
};

static inline int compare_entries_bid(const void* a, const void* b) {
    const Entry* entry_a = (const Entry*)a;
    const Entry* entry_b = (const Entry*)b;

    if (entry_a->n_orders == 0 && entry_b->n_orders == 0) return 0;
    if (entry_b->n_orders == 0) return -1;
    if (entry_a->n_orders == 0) return 1;

    if (entry_a->price < entry_b->price) return 1;
    if (entry_a->price > entry_b->price) return -1;
    return 0;
};

static inline int compare_entries_ask(const void* a, const void* b) {
    const Entry* entry_a = (const Entry*)a;
    const Entry* entry_b = (const Entry*)b;

    if (entry_a->n_orders == 0 && entry_b->n_orders == 0) return 0;
    if (entry_b->n_orders == 0) return -1;
    if (entry_a->n_orders == 0) return 1;

    if (entry_a->price < entry_b->price) return -1;
    if (entry_a->price > entry_b->price) return 1;
    return 0;
};

#endif // C_MARKET_DATA_H
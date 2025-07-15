#ifndef C_MARKET_DATA_EXTERNAL_H
#define C_MARKET_DATA_EXTERNAL_H

#include <stdint.h>

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
    SIDE_LONG_OPEN = DIRECTION_LONG + OFFSET_OPEN,
    SIDE_LONG_CLOSE = DIRECTION_LONG + OFFSET_CLOSE,
    SIDE_LONG_CANCEL = DIRECTION_LONG + OFFSET_CANCEL,

    SIDE_SHORT_OPEN = DIRECTION_SHORT + OFFSET_OPEN,
    SIDE_SHORT_CLOSE = DIRECTION_SHORT + OFFSET_CLOSE,
    SIDE_SHORT_CANCEL = DIRECTION_SHORT + OFFSET_CANCEL,

    SIDE_NEUTRAL_OPEN = DIRECTION_NEUTRAL + OFFSET_OPEN,
    SIDE_NEUTRAL_CLOSE = DIRECTION_NEUTRAL + OFFSET_CLOSE,

    SIDE_BID = DIRECTION_LONG + OFFSET_ORDER,
    SIDE_ASK = DIRECTION_SHORT + OFFSET_ORDER,

    SIDE_CANCEL = DIRECTION_UNKNOWN + OFFSET_CANCEL,
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

// Lookup table for direction sign
extern const int8_t SIGN_LUT[4];

// Function Declarations
static inline int8_t direction_to_sign(uint8_t x);
static inline void platform_usleep(unsigned int usec);

int compare_md_ptr(const void *a, const void *b);
int compare_entries_bid(const void *a, const void *b);
int compare_entries_ask(const void *a, const void *b);

#endif // C_MARKET_DATA_EXTERNAL_H

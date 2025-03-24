#include <stdlib.h>

#define TICKER_SIZE 32
#define BOOK_SIZE 10
#define ID_SIZE 16

typedef enum {
    DIRECTION_SHORT = 0,
    DIRECTION_UNKNOWN = 1,
    DIRECTION_LONG = 2
} Direction;

typedef enum {
    OFFSET_CANCEL = 0,
    OFFSET_ORDER = 4,
    OFFSET_OPEN = 8,
    OFFSET_CLOSE = 16
} Offset;

typedef enum {
    // Long Side
    SIDE_LONG_OPEN = DIRECTION_LONG + OFFSET_OPEN,
    SIDE_LONG_CLOSE = DIRECTION_LONG + OFFSET_CLOSE,
    SIDE_LONG_CANCEL = DIRECTION_LONG + OFFSET_CANCEL,

    // Short Side
    SIDE_SHORT_OPEN = DIRECTION_SHORT + OFFSET_OPEN,
    SIDE_SHORT_CLOSE = DIRECTION_SHORT + OFFSET_CLOSE,
    SIDE_SHORT_CANCEL = DIRECTION_SHORT + OFFSET_CANCEL,

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

typedef struct {
    double price;
    double volume;
    unsigned int n_orders;
} Entry;

int compare_entries_bid(const void* a, const void* b) {
    Entry* entry_a = (Entry*) a;
    Entry* entry_b = (Entry*) b;
    if (entry_a->price < entry_b->price) {
        return 1;
    } else if (entry_a->price > entry_b->price) {
        return -1;
    }
    return 0;
}

int compare_entries_ask(const void* a, const void* b) {
    Entry* entry_a = (Entry*) a;
    Entry* entry_b = (Entry*) b;
    if (entry_a->price < entry_b->price) {
        return -1;
    } else if (entry_a->price > entry_b->price) {
        return 1;
    }
    return 0;
}

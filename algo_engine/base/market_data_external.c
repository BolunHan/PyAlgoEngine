#include <stdlib.h>

#define TICKER_SIZE 32
#define BOOK_SIZE 10
#define ID_SIZE 16
#define MAX_WORKERS 128

#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
#else
#include <unistd.h>
#endif

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
    uint8_t dtype;
    char ticker[TICKER_SIZE];
    double timestamp;
} __attribute__((packed)) MetaInfo;

typedef struct {
    double price;
    double volume;
    uint64_t n_orders;
} __attribute__((packed)) Entry;

static inline void platform_usleep(unsigned int usec) {
    #if defined(_WIN32) || defined(_WIN64)
    Sleep(usec / 1000);  // Windows Sleep uses milliseconds
    #else
    usleep(usec);         // POSIX usleep uses microseconds
    #endif
}

// Compare function for sorting market data by timestamp
int compare_md(const void* a, const void* b) {
    // The input is a pointer to a pointer to MetaInfo
    const MetaInfo* data_a = *((const MetaInfo**)a);
    const MetaInfo* data_b = *((const MetaInfo**)b);

    if (data_a->timestamp < data_b->timestamp) {
        return -1;
    } else if (data_a->timestamp > data_b->timestamp) {
        return 1;
    }
    return 0;
}

// Compare function for sorting market data pointers by timestamp
int compare_md_ptr(const void *a, const void *b) {
    const MetaInfo *md_a = *(const MetaInfo **)a;
    const MetaInfo *md_b = *(const MetaInfo **)b;

    if (md_a->timestamp < md_b->timestamp) return -1;
    if (md_a->timestamp > md_b->timestamp) return 1;
    return 0;
}

int compare_entries_bid(const void* a, const void* b) {
    Entry* entry_a = (Entry*) a;
    Entry* entry_b = (Entry*) b;

    // If both have n_orders == 0, they are equal
    if (entry_a->n_orders == 0 && entry_b->n_orders == 0) {
        return 0;
    }
    // If only entry_b has n_orders == 0, entry_a should be considered larger
    if (entry_b->n_orders == 0) {
        return -1;
    }
    // If only entry_a has n_orders == 0, entry_b should be considered larger
    if (entry_a->n_orders == 0) {
        return 1;
    }

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

    // If both have n_orders == 0, they are equal
    if (entry_a->n_orders == 0 && entry_b->n_orders == 0) {
        return 0;
    }
    // If only entry_b has n_orders == 0, entry_a should be considered larger
    if (entry_b->n_orders == 0) {
        return -1;
    }
    // If only entry_a has n_orders == 0, entry_b should be considered larger
    if (entry_a->n_orders == 0) {
        return 1;
    }

    if (entry_a->price < entry_b->price) {
        return -1;
    } else if (entry_a->price > entry_b->price) {
        return 1;
    }
    return 0;
}

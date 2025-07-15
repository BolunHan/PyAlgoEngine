#include "c_market_data_config.h"
#include "c_market_data_external.h"

// Lookup table definition
const int8_t SIGN_LUT[4] = {
    -1,  // 0b00 → -1
    0,   // 0b01 → 0
    1,   // 0b10 → 1
    0    // 0b11 → 0
};

// Convert Direction enum to signed int
static inline int8_t direction_to_sign(uint8_t x) {
    return SIGN_LUT[x & 0b11];  // Mask to 2 bits
}

// Cross-platform sleep in microseconds
static inline void platform_usleep(unsigned int usec) {
#if defined(_WIN32) || defined(_WIN64)
    Sleep(usec / 1000);  // Windows: Sleep in milliseconds
#else
    usleep(usec);        // POSIX: Sleep in microseconds
#endif
}

// Compare function for sorting MetaInfo pointers by timestamp
int compare_md_ptr(const void *a, const void *b) {
    const MetaInfo *md_a = *(const MetaInfo **)a;
    const MetaInfo *md_b = *(const MetaInfo **)b;

    if (md_a->timestamp < md_b->timestamp) return -1;
    if (md_a->timestamp > md_b->timestamp) return 1;
    return 0;
}

// Shared comparator logic for Entry sorting
static inline int compare_entry_common(const Entry *a, const Entry *b, int descending) {
    if (a->price < b->price) return descending ? 1 : -1;
    if (a->price > b->price) return descending ? -1 : 1;

    // Secondary tie-breaker by n_orders
    if (a->n_orders == 0 && b->n_orders > 0) return 1;
    if (a->n_orders > 0 && b->n_orders == 0) return -1;

    return 0;
}

// Descending order for bids (high price first)
int compare_entries_bid(const void *a, const void *b) {
    return compare_entry_common((const Entry *)a, (const Entry *)b, 1);
}

// Ascending order for asks (low price first)
int compare_entries_ask(const void *a, const void *b) {
    return compare_entry_common((const Entry *)a, (const Entry *)b, 0);
}

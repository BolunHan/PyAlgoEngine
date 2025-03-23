#include <stdlib.h>

#define TICKER_SIZE 32
#define BOOK_SIZE 10
#define ID_SIZE 16

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

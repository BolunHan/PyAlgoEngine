#ifndef C_MARKET_DATA_BUFFER_H
#define C_MARKET_DATA_BUFFER_H

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <pthread.h>
#include "c_shm_allocator.h"
#include "c_heap_allocator.h"
#include "c_intern_string.h"
#include "c_market_data.h"

// ========== MarketData Buffer Structs ==========

typedef struct md_block_buffer {
    shm_allocator_t* shm_allocator;
    heap_allocator_t* heap_allocator;
    int sorted;
    size_t ptr_capacity;
    size_t ptr_offset;
    size_t ptr_tail;
    size_t data_capacity;
    size_t data_offset;
    size_t data_tail;
    double current_timestamp;
    char buffer[];
} md_block_buffer;

typedef struct md_ring_buffer {
    shm_allocator_t* shm_allocator;
    heap_allocator_t* heap_allocator;
    size_t ptr_capacity;
    size_t ptr_offset;
    size_t ptr_head;
    size_t ptr_tail;
    size_t data_capacity;
    size_t data_offset;
    size_t data_tail;
    char buffer[];
} md_ring_buffer;

typedef struct md_concurrent_buffer_worker_t {
    size_t ptr_head;
    int enabled;
} md_concurrent_buffer_worker_t;

typedef struct md_concurrent_buffer {
    shm_allocator_t* shm_allocator;
    md_concurrent_buffer_worker_t* workers;
    size_t n_workers;
    market_data_t** buffer;
    size_t capacity;
    size_t tail;
} md_concurrent_buffer;

// ========== Utility Functions ==========

static inline int c_md_compare_serialized(const void* a, const void* b) {
    const char* md_serialized_a = *(const char* const*) a;
    const char* md_serialized_b = *(const char* const*) b;
    double timestamp_a = *(const double*) (md_serialized_a + sizeof(uint8_t));
    double timestamp_b = *(const double*) (md_serialized_b + sizeof(uint8_t));

    if (timestamp_a < timestamp_b) return -1;
    if (timestamp_a > timestamp_b) return 1;
    return 0;
}

static inline size_t c_md_total_buffer_size(market_data_t** md_array, size_t n_md) {
    market_data_t* md;
    size_t total_size = 0;

    for (size_t i = 0; i < n_md; i++) {
        md = md_array[i];
        total_size += c_md_serialized_size(md);
    }
    return total_size;
}

static inline market_data_t* c_md_send_to_shm(market_data_t* market_data, shm_allocator_ctx* shm_allocator, istr_map* shm_pool, int with_lock) {
    if (!market_data || !shm_allocator || !shm_pool) return NULL;

    // Step 1: Intern ticker string
    const char* interned_ticker;
    if (market_data->meta_info.ticker) {
        if (with_lock) {
            interned_ticker = c_istr_synced(shm_pool, market_data->meta_info.ticker);
        }
        else {
            interned_ticker = c_istr(shm_pool, market_data->meta_info.ticker);
        }
        if (!interned_ticker) return NULL;
    }
    else {
        interned_ticker = NULL;
    }

    // Step 2: Check if already in SHM
    if (market_data->meta_info.shm_allocator == shm_allocator->shm_allocator) {
        market_data->meta_info.ticker = interned_ticker;
        return market_data;
    }

    // Step 3: Initialize new market_data in SHM
    data_type_t dtype = market_data->meta_info.dtype;
    market_data_t* shm_md = c_md_new(dtype, shm_allocator, NULL, with_lock);
    size_t size = c_md_get_size(dtype);
    if (!shm_md) return NULL;

    memcpy((void*) shm_md, (void*) market_data, size);
    shm_md->meta_info.ticker = interned_ticker;

    // Step 4: Handle order book pointers if tick_data_t
    if (dtype == DTYPE_TICK) {
        tick_data_t* src_tick = &market_data->tick_data_full;
        tick_data_t* dst_tick = &shm_md->tick_data_full;

        // Bid order book
        if (src_tick->bid) {
            size_t ob_capacity = src_tick->bid->capacity;
            size_t ob_size = sizeof(order_book_t) + (ob_capacity * sizeof(order_book_entry_t));
            order_book_t* ob_shm = c_md_orderbook_new(ob_capacity, shm_allocator, NULL, with_lock);
            if (!ob_shm) {
                c_shm_free((void*) shm_md, NULL);
                return NULL;
            }
            memcpy((void*) ob_shm, (void*) src_tick->bid, ob_size);
            dst_tick->bid = ob_shm;
        }
        else {
            dst_tick->bid = NULL;
        }

        // Ask order book
        if (src_tick->ask) {
            size_t ob_capacity = src_tick->ask->capacity;
            size_t ob_size = sizeof(order_book_t) + (ob_capacity * sizeof(order_book_entry_t));
            order_book_t* ob_shm = c_md_orderbook_new(ob_capacity, shm_allocator, NULL, with_lock);
            if (!ob_shm) {
                if (dst_tick->bid) {
                    c_shm_free((void*) dst_tick->bid, NULL);
                }
                c_shm_free((void*) shm_md, NULL);
                return NULL;
            }
            memcpy((void*) ob_shm, (void*) src_tick->ask, ob_size);
            dst_tick->ask = ob_shm;
        }
        else {
            dst_tick->ask = NULL;
        }
    }

    return shm_md;
}

// ========== BlockBuffer API Functions ==========

static inline md_block_buffer* c_md_block_buffer_new(size_t ptr_capacity, size_t data_capacity, shm_allocator_ctx* shm_allocator, heap_allocator_t* heap_allocator, int with_lock) {
    size_t data_offset = ptr_capacity * sizeof(size_t);
    size_t size = sizeof(md_block_buffer)
        + data_offset                 /* pointer array */
        + data_capacity;              /* data buffer */

    if (size == 0) return NULL;

    if (shm_allocator) {
        pthread_mutex_t* lock = with_lock ? &shm_allocator->shm_allocator->lock : NULL;
        md_block_buffer* buffer = (md_block_buffer*) c_shm_request(shm_allocator, size, 0, lock);
        if (!buffer) return NULL;
        buffer->shm_allocator = shm_allocator->shm_allocator; /* ctx -> allocator */
        buffer->ptr_capacity = ptr_capacity;
        buffer->data_capacity = data_capacity;
        buffer->data_offset = data_offset;
        buffer->sorted = 1;
        return buffer;
    }
    else if (heap_allocator) {
        pthread_mutex_t* lock = with_lock ? &heap_allocator->lock : NULL;
        md_block_buffer* buffer = (md_block_buffer*) c_heap_request(heap_allocator, size, 0, lock);
        if (!buffer) return NULL;
        buffer->heap_allocator = heap_allocator;
        buffer->ptr_capacity = ptr_capacity;
        buffer->data_capacity = data_capacity;
        buffer->data_offset = data_offset;
        buffer->sorted = 1;
        return buffer;
    }
    else {
        md_block_buffer* buffer = (md_block_buffer*) calloc(1, size); /* zeroed */
        if (!buffer) return NULL;
        buffer->ptr_capacity = ptr_capacity;
        buffer->data_capacity = data_capacity;
        buffer->data_offset = data_offset;
        buffer->sorted = 1;
        return buffer;
    }
}

static inline int c_md_block_buffer_free(md_block_buffer* buffer, int with_lock) {
    if (!buffer) return -1;

    shm_allocator_t* shm_allocator = buffer->shm_allocator;
    heap_allocator_t* heap_allocator = buffer->heap_allocator;

    if (shm_allocator) {
        pthread_mutex_t* lock = with_lock ? &shm_allocator->lock : NULL;
        c_shm_free((void*) buffer, lock);
    }
    else if (heap_allocator) {
        pthread_mutex_t* lock = with_lock ? &heap_allocator->lock : NULL;
        c_heap_free((void*) buffer, lock);
    }
    else {
        free((void*) buffer);
    }
    return 0;
}

static inline md_block_buffer* c_md_block_buffer_extend(md_block_buffer* buffer, size_t new_ptr_capacity, size_t new_data_capacity, shm_allocator_ctx* shm_allocator, heap_allocator_t* heap_allocator, int with_lock) {
    if (!buffer) return NULL;

    size_t new_data_offset = new_ptr_capacity * sizeof(size_t);
    md_block_buffer* new_buffer = c_md_block_buffer_new(new_ptr_capacity, new_data_capacity, shm_allocator, heap_allocator, with_lock);
    if (!new_buffer) return NULL;

    new_buffer->ptr_offset = buffer->ptr_offset;
    new_buffer->ptr_capacity = new_ptr_capacity;;
    new_buffer->data_capacity = new_data_capacity;
    new_buffer->data_offset = new_data_offset;
    new_buffer->current_timestamp = buffer->current_timestamp;

    memcpy(new_buffer->buffer + new_buffer->ptr_offset, buffer->buffer + buffer->ptr_offset, buffer->ptr_tail * sizeof(size_t));
    memcpy(new_buffer->buffer + new_buffer->data_offset, buffer->buffer + buffer->data_offset, buffer->data_tail);

    new_buffer->ptr_tail = buffer->ptr_tail;
    new_buffer->data_tail = buffer->data_tail;
    new_buffer->sorted = buffer->sorted;

    return new_buffer;
}

static inline int c_md_block_buffer_put(md_block_buffer* buffer, market_data_t* market_data) {
    if (!buffer || !market_data) return -1;

    if (buffer->ptr_tail >= buffer->ptr_capacity) {
        return -1;
    }

    size_t serialized_size = c_md_serialized_size(market_data);
    if (buffer->data_tail + serialized_size > buffer->data_capacity) {
        return -1;
    }

    char* data_ptr = buffer->buffer + buffer->data_offset + buffer->data_tail;
    c_md_serialize(market_data, data_ptr);

    size_t* ptr_array = (size_t*) (buffer->buffer + buffer->ptr_offset);
    ptr_array[buffer->ptr_tail] = buffer->data_tail;

    double ts = market_data->meta_info.timestamp;
    if (buffer->ptr_tail) {
        if (ts < buffer->current_timestamp) {
            buffer->sorted = 0;
        }
        else {
            buffer->current_timestamp = ts;
        }
    }
    else {
        buffer->sorted = 1;
        buffer->current_timestamp = ts;
    }

    buffer->ptr_tail += 1;
    buffer->data_tail += serialized_size;

    return 0;
}

static inline const char* c_md_block_buffer_get(md_block_buffer* buffer, size_t index) {
    if (!buffer) return NULL;
    if (index >= buffer->ptr_tail) return NULL;

    size_t* ptr_array = (size_t*) (buffer->buffer + buffer->ptr_offset);
    size_t data_offset = ptr_array[index];

    const char* data_ptr = buffer->buffer + buffer->data_offset + data_offset;
    return data_ptr;
}

static inline int c_md_block_buffer_sort(md_block_buffer* buffer) {
    if (!buffer) return -1;
    if (buffer->sorted) return 0;
    if (buffer->ptr_tail <= 1) {
        buffer->sorted = 1;
        return 0;
    }

    size_t* offset_array = (size_t*) (buffer->buffer + buffer->ptr_offset);
    char* data_base = buffer->buffer + buffer->data_offset;
    market_data_t** ptr_array = (market_data_t**) malloc(buffer->ptr_tail * sizeof(market_data_t*));
    if (!ptr_array) return -1;

    for (size_t i = 0; i < buffer->ptr_tail; i++) {
        ptr_array[i] = (market_data_t*) (data_base + offset_array[i]);
    }

    qsort(ptr_array, buffer->ptr_tail, sizeof(market_data_t*), c_md_compare_serialized);

    for (size_t i = 0; i < buffer->ptr_tail; i++) {
        offset_array[i] = (size_t) ((char*) ptr_array[i] - data_base);
    }

    buffer->sorted = 1;
    free(ptr_array);
    return 0;
}

static inline int c_md_block_buffer_clear(md_block_buffer* buffer) {
    if (!buffer) return -1;

    buffer->ptr_tail = 0;
    buffer->data_tail = 0;
    buffer->current_timestamp = 0.0;
    buffer->sorted = 1;
    return 0;
}

static inline size_t c_md_block_buffer_serialized_size(md_block_buffer* buffer) {
    if (!buffer) return 0;

    return sizeof(md_block_buffer) + (buffer->ptr_tail * sizeof(size_t)) + buffer->data_tail;
}

static inline size_t c_md_block_buffer_serialize(md_block_buffer* buffer, char* out_buffer) {
    if (!buffer || !out_buffer) return 0;

    size_t offset = 0;
    memcpy(out_buffer + offset, buffer, sizeof(md_block_buffer));
    offset += sizeof(md_block_buffer);

    size_t ptr_array_size = buffer->ptr_tail * sizeof(size_t);
    memcpy(out_buffer + offset, buffer->buffer + buffer->ptr_offset, ptr_array_size);
    offset += ptr_array_size;

    memcpy(out_buffer + offset, buffer->buffer + buffer->data_offset, buffer->data_tail);
    offset += buffer->data_tail;

    return offset;
}

// ========== RingBuffer API Functions ==========

static inline md_ring_buffer* c_md_ring_buffer_new(size_t ptr_capacity, size_t data_capacity, shm_allocator_ctx* shm_allocator, heap_allocator_t* heap_allocator, int with_lock) {
    size_t data_offset = ptr_capacity * sizeof(size_t);
    size_t size = sizeof(md_ring_buffer) + data_offset + data_capacity;

    if (size == 0) return NULL;

    if (shm_allocator) {
        pthread_mutex_t* lock = with_lock ? &shm_allocator->shm_allocator->lock : NULL;
        md_ring_buffer* buffer = (md_ring_buffer*) c_shm_request(shm_allocator, size, 0, lock);
        if (!buffer) return NULL;
        buffer->shm_allocator = shm_allocator->shm_allocator;
        buffer->ptr_capacity = ptr_capacity;
        buffer->data_capacity = data_capacity;
        buffer->data_offset = data_offset;
        return buffer;
    }
    else if (heap_allocator) {
        pthread_mutex_t* lock = with_lock ? &heap_allocator->lock : NULL;
        md_ring_buffer* buffer = (md_ring_buffer*) c_heap_request(heap_allocator, size, 0, lock);
        if (!buffer) return NULL;
        buffer->heap_allocator = heap_allocator;
        buffer->ptr_capacity = ptr_capacity;
        buffer->data_capacity = data_capacity;
        buffer->data_offset = data_offset;
        return buffer;
    }
    else {
        md_ring_buffer* buffer = (md_ring_buffer*) calloc(1, size); /* zeroed */
        if (!buffer) return NULL;
        buffer->ptr_capacity = ptr_capacity;
        buffer->data_capacity = data_capacity;
        buffer->data_offset = data_offset;
        return buffer;
    }
}

static inline int c_md_ring_buffer_free(md_ring_buffer* buffer, int with_lock) {
    if (!buffer) return -1;

    shm_allocator_t* shm_allocator = buffer->shm_allocator;
    heap_allocator_t* heap_allocator = buffer->heap_allocator;

    if (shm_allocator) {
        pthread_mutex_t* lock = with_lock ? &shm_allocator->lock : NULL;
        c_shm_free((void*) buffer, lock);
    }
    else if (heap_allocator) {
        pthread_mutex_t* lock = with_lock ? &heap_allocator->lock : NULL;
        c_heap_free((void*) buffer, lock);
    }
    else {
        free((void*) buffer);
    }
    return 0;
}

static inline int c_md_ring_buffer_is_full(md_ring_buffer* buffer, market_data_t* market_data) {
    if (!buffer) return -1;

    size_t* offset_array = (size_t*) (buffer->buffer + buffer->ptr_offset);
    size_t ptr_head = buffer->ptr_head;
    size_t ptr_tail = buffer->ptr_tail;
    size_t ptr_next = (ptr_tail + 1) % buffer->ptr_capacity;

    if (ptr_head == ptr_next) return 1;

    if (ptr_head == ptr_tail) return 0;

    size_t payload_size = c_md_serialized_size(market_data);
    size_t data_head = offset_array[buffer->ptr_head];
    size_t data_tail = buffer->data_tail;

    if (data_tail >= data_head) {
        size_t space_end = buffer->data_capacity - data_tail;
        size_t space_start = data_head;
        if (payload_size <= space_end) {
            return 0;
        }
        // Will not wrapped write, if the end space is insufficient, we just check the start space
        else if (payload_size <= space_start) {
            return 0;
        }
        else {
            return 1;
        }
    }
    else {
        size_t space_middle = data_head - data_tail;
        if (payload_size <= space_middle) {
            return 0;
        }
        else {
            return 1;
        }
    }
}

static inline int c_md_ring_buffer_is_empty(md_ring_buffer* buffer) {
    if (!buffer) return -1;

    size_t ptr_head = buffer->ptr_head;
    size_t ptr_tail = buffer->ptr_tail;

    if (ptr_head == ptr_tail) return 1;
    else return 0;
}

static inline size_t c_md_ring_buffer_size(md_ring_buffer* buffer) {
    if (!buffer) return 0;

    size_t ptr_head = buffer->ptr_head;
    size_t ptr_tail = buffer->ptr_tail;
    size_t capacity = buffer->ptr_capacity;

    if (ptr_tail >= ptr_head) {
        return ptr_tail - ptr_head;
    }
    else {
        return (capacity - ptr_head) + ptr_tail;
    }
}

static inline int c_md_ring_buffer_put(md_ring_buffer* buffer, market_data_t* market_data) {
    if (!buffer || !market_data) return -1;

    size_t* offset_array = (size_t*) (buffer->buffer + buffer->ptr_offset);
    size_t ptr_head = buffer->ptr_head;
    size_t ptr_tail = buffer->ptr_tail;
    size_t ptr_next = (ptr_tail + 1) % buffer->ptr_capacity;

    if (ptr_head == ptr_next) {
        return -2;
    }

    size_t serialized_size = c_md_serialized_size(market_data);
    size_t data_head = offset_array[ptr_head];
    size_t data_tail = buffer->data_tail;

    size_t write_offset = 0;
    if (data_tail >= data_head) {
        size_t space_end = buffer->data_capacity - data_tail;
        if (serialized_size <= space_end) {
            write_offset = data_tail;
            buffer->data_tail += serialized_size;
        }
        else {
            // Wrap around
            if (serialized_size <= data_head) {
                write_offset = 0;
                buffer->data_tail = serialized_size;
            }
            else {
                return -3;
            }
        }
    }
    else {
        size_t space_middle = data_head - data_tail;
        if (serialized_size <= space_middle) {
            write_offset = data_tail;
            buffer->data_tail += serialized_size;
        }
        else {
            return -3;
        }
    }

    char* data_ptr = buffer->buffer + buffer->data_offset + write_offset;
    c_md_serialize(market_data, data_ptr);

    offset_array[ptr_tail] = write_offset;
    buffer->ptr_tail = ptr_next;

    return 0;
}

static inline const char* c_md_ring_buffer_get(md_ring_buffer* buffer, size_t index) {
    if (!buffer) return NULL;
    if (index >= buffer->ptr_capacity) return NULL;
    size_t size = c_md_ring_buffer_size(buffer);
    if (index >= size) return NULL;

    size_t ptr_head = buffer->ptr_head;
    size_t ptr_capacity = buffer->ptr_capacity;
    size_t ptr_idx = (ptr_head + index) % ptr_capacity;

    size_t* offset_array = (size_t*) (buffer->buffer + buffer->ptr_offset);
    size_t data_offset = offset_array[ptr_idx];

    if (data_offset >= buffer->data_capacity) {
        return NULL;
    }

    char* data_ptr = buffer->buffer + buffer->data_offset + data_offset;
    return data_ptr;
}

static inline int c_md_ring_buffer_listen(md_ring_buffer* buffer, int block, double timeout, const char** out) {
    if (!buffer || !out) return -1;

    const uint32_t spin_per_check = 1000;
    time_t start_time = 0;
    time_t current_time;
    double elapsed = 0.0;
    uint32_t spin_count = 0;
    uint32_t sleep_us = 0;
    const int use_timeout = timeout > 0.0;
    size_t idx = buffer->ptr_head;

    if (!block && idx == buffer->ptr_tail) {
        return -2; /* empty and non-blocking */
    }

    time(&start_time);

    for (;;) {
        if (idx != buffer->ptr_tail) {
            size_t* offset_array = (size_t*) (buffer->buffer + buffer->ptr_offset);
            size_t data_offset = offset_array[idx];
            if (data_offset >= buffer->data_capacity) return -3; /* corrupt offset */

            const char* data_ptr = (buffer->buffer + buffer->data_offset + data_offset);
            buffer->ptr_head = (idx + 1) % buffer->ptr_capacity;
            *out = data_ptr;
            return 0;
        }

        if ((spin_count % spin_per_check) == 0) {
            time(&current_time);
            elapsed = difftime(current_time, start_time);

            if (use_timeout && elapsed >= timeout) {
                return -4; /* timeout */
            }

            if (elapsed < 0.1) {
                sleep_us = 0;      /* <100 ms: pure spin */
            }
            else if (elapsed < 1.0) {
                sleep_us = 1;      /* 100-1000 ms */
            }
            else if (elapsed < 3.0) {
                sleep_us = 10;     /* 1-3 s */
            }
            else if (elapsed < 15.0) {
                sleep_us = 100;    /* 3-15 s */
            }
            else {
                sleep_us = 1000;   /* >15 s */
            }
        }

        if (sleep_us > 0) {
            c_usleep(sleep_us);
        }
        spin_count += 1;
    }
}

// ========== ConcurrentBuffer API Functions ==========

static inline md_concurrent_buffer* c_md_concurrent_buffer_new(size_t n_workers, size_t capacity, shm_allocator_ctx* shm_allocator, int with_lock) {
    size_t size = sizeof(md_concurrent_buffer)
        + n_workers * sizeof(md_concurrent_buffer_worker_t)
        + capacity * sizeof(market_data_t*);

    if (size == 0) return NULL;

    if (shm_allocator) {
        pthread_mutex_t* lock = with_lock ? &shm_allocator->shm_allocator->lock : NULL;
        md_concurrent_buffer* buffer = (md_concurrent_buffer*) c_shm_request(shm_allocator, size, 1, lock);
        if (!buffer) return NULL;
        buffer->shm_allocator = shm_allocator->shm_allocator;
        buffer->n_workers = n_workers;
        buffer->workers = (md_concurrent_buffer_worker_t*) (buffer + 1);
        buffer->buffer = (market_data_t**) (buffer->workers + n_workers);
        buffer->capacity = capacity;
        buffer->tail = 0;
        return buffer;
    }
    else {
        return NULL; /* only support shm allocator for now */
    }
}

static inline int c_md_concurrent_buffer_free(md_concurrent_buffer* buffer, int with_lock) {
    if (!buffer) return -1;

    shm_allocator_t* shm_allocator = buffer->shm_allocator;

    if (shm_allocator) {
        pthread_mutex_t* lock = with_lock ? &shm_allocator->lock : NULL;
        c_shm_free((void*) buffer, lock);
    }
    else {
        return -1;
    }
    return 0;
}

static inline int c_md_concurrent_buffer_enable_worker(md_concurrent_buffer* buffer, size_t worker_id) {
    if (!buffer) return -1;
    if (worker_id >= buffer->n_workers) return -1;

    buffer->workers[worker_id].enabled = 1;
    buffer->workers[worker_id].ptr_head = 0;
    return 0;
}

static inline int c_md_concurrent_buffer_disable_worker(md_concurrent_buffer* buffer, size_t worker_id) {
    if (!buffer) return -1;
    if (worker_id >= buffer->n_workers) return -1;

    buffer->workers[worker_id].enabled = 0;
    buffer->workers[worker_id].ptr_head = 0;
    return 0;
}

static inline int c_md_concurrent_buffer_is_full(md_concurrent_buffer* buffer, market_data_t* market_data) {
    if (!buffer || !market_data) return -1;

    size_t next_tail = (buffer->tail + 1) % buffer->capacity;

    for (size_t i = 0; i < buffer->n_workers; i++) {
        md_concurrent_buffer_worker_t* worker = buffer->workers + i;
        if (!worker->enabled) continue;

        if (worker->ptr_head == next_tail) {
            return 1;
        }
    }
    return 0;
}

static inline int c_md_concurrent_buffer_is_empty(md_concurrent_buffer* buffer, size_t worker_id) {
    if (!buffer) return -1;
    if (worker_id >= buffer->n_workers) return -1;

    md_concurrent_buffer_worker_t* worker = buffer->workers + worker_id;
    if (!worker->enabled) return -1;

    if (worker->ptr_head == buffer->tail) return 1;
    else return 0;
}

static inline int c_md_concurrent_buffer_put(md_concurrent_buffer* buffer, market_data_t* market_data) {
    if (!buffer || !market_data) return -1;

    if (!market_data->meta_info.shm_allocator) {
        market_data_t* md_shm = c_md_send_to_shm(market_data, NULL, NULL, 0);
        if (!md_shm) return -1;
        market_data = md_shm;
    }

    size_t next_tail = (buffer->tail + 1) % buffer->capacity;

    for (size_t i = 0; i < buffer->n_workers; i++) {
        md_concurrent_buffer_worker_t* worker = buffer->workers + i;
        if (!worker->enabled) continue;

        if (worker->ptr_head == next_tail) {
            return -1;
        }
    }

    buffer->buffer[buffer->tail] = market_data;
    buffer->tail = next_tail;

    return 0;
}

#endif /* C_MARKET_DATA_BUFFER_H */

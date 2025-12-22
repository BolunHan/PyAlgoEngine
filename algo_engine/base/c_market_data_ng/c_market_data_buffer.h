#ifndef C_MARKET_DATA_BUFFER_H
#define C_MARKET_DATA_BUFFER_H

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <pthread.h>
#include <sched.h>
#include "c_shm_allocator.h"
#include "c_heap_allocator.h"
#include "c_intern_string.h"
#include "c_market_data.h"

// ========== MarketData Buffer Structs ==========

/* Unified return codes for all buffer APIs */
#define MD_BUF_OK             0   /* success */
#define MD_BUF_ERR_INVALID   -1   /* invalid args/buffer/state */
#define MD_BUF_ERR_NOT_SHM   -2   /* market_data not in SHM (concurrent buffer) */
#define MD_BUF_ERR_FULL      -3   /* buffer full / insufficient space */
#define MD_BUF_ERR_EMPTY     -4   /* buffer empty (non-blocking listen) */
#define MD_BUF_ERR_TIMEOUT   -5   /* timeout when blocking */
#define MD_BUF_ERR_CORRUPT   -6   /* corrupt offset or data */
#define MD_BUF_OOR           -7   /* generic (index, worker, ptr, buffer) out of range */
#define MD_BUF_DISABLED      -8   /* generic (index, worker, ptr, buffer) disabled */

typedef struct md_block_buffer {
    shm_allocator* shm_allocator;
    heap_allocator* heap_allocator;
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
    shm_allocator* shm_allocator;
    heap_allocator* heap_allocator;
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
    shm_allocator* shm_allocator;
    md_concurrent_buffer_worker_t* workers;
    size_t n_workers;
    md_variant** buffer;
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

static inline size_t c_md_total_buffer_size(md_variant** md_array, size_t n_md) {
    md_variant* md;
    size_t total_size = 0;

    for (size_t i = 0; i < n_md; i++) {
        md = md_array[i];
        total_size += c_md_serialized_size(md);
    }
    return total_size;
}

static inline md_variant* c_md_send_to_shm(md_variant* market_data, shm_allocator_ctx* shm_allocator, istr_map* shm_pool, int with_lock) {
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
    md_data_type dtype = market_data->meta_info.dtype;
    md_variant* shm_md = c_md_new(dtype, shm_allocator, NULL, with_lock);
    size_t size = c_md_get_size(dtype);
    if (!shm_md) return NULL;

    memcpy((void*) shm_md, (void*) market_data, size);
    shm_md->meta_info.ticker = interned_ticker;

    // Step 4: Handle order book pointers if md_tick_data
    if (dtype == DTYPE_TICK) {
        md_tick_data* src_tick = &market_data->tick_data_full;
        md_tick_data* dst_tick = &shm_md->tick_data_full;

        // Bid order book
        if (src_tick->bid) {
            size_t ob_capacity = src_tick->bid->capacity;
            size_t ob_size = sizeof(md_orderbook) + (ob_capacity * sizeof(md_orderbook_entry));
            md_orderbook* ob_shm = c_md_orderbook_new(ob_capacity, shm_allocator, NULL, with_lock);
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
            size_t ob_size = sizeof(md_orderbook) + (ob_capacity * sizeof(md_orderbook_entry));
            md_orderbook* ob_shm = c_md_orderbook_new(ob_capacity, shm_allocator, NULL, with_lock);
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

static inline md_block_buffer* c_md_block_buffer_new(size_t ptr_capacity, size_t data_capacity, shm_allocator_ctx* shm_allocator, heap_allocator* heap_allocator, int with_lock) {
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
    if (!buffer) return MD_BUF_ERR_INVALID;

    shm_allocator* shm_allocator = buffer->shm_allocator;
    heap_allocator* heap_allocator = buffer->heap_allocator;

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
    return MD_BUF_OK;
}

static inline md_block_buffer* c_md_block_buffer_extend(md_block_buffer* buffer, size_t new_ptr_capacity, size_t new_data_capacity, shm_allocator_ctx* shm_allocator, heap_allocator* heap_allocator, int with_lock) {
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

static inline int c_md_block_buffer_put(md_block_buffer* buffer, md_variant* market_data) {
    if (!buffer || !market_data) return MD_BUF_ERR_INVALID;

    if (buffer->ptr_tail >= buffer->ptr_capacity) {
        return MD_BUF_ERR_FULL;
    }

    size_t serialized_size = c_md_serialized_size(market_data);
    if (buffer->data_tail + serialized_size > buffer->data_capacity) {
        return MD_BUF_ERR_FULL;
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

    return MD_BUF_OK;
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
    if (!buffer) return MD_BUF_ERR_INVALID;
    if (buffer->sorted) return MD_BUF_OK;
    if (buffer->ptr_tail <= 1) {
        buffer->sorted = 1;
        return MD_BUF_OK;
    }

    size_t* offset_array = (size_t*) (buffer->buffer + buffer->ptr_offset);
    char* data_base = buffer->buffer + buffer->data_offset;
    md_variant** ptr_array = (md_variant**) malloc(buffer->ptr_tail * sizeof(md_variant*));
    if (!ptr_array) return MD_BUF_ERR_INVALID;

    for (size_t i = 0; i < buffer->ptr_tail; i++) {
        ptr_array[i] = (md_variant*) (data_base + offset_array[i]);
    }

    qsort(ptr_array, buffer->ptr_tail, sizeof(md_variant*), c_md_compare_serialized);

    for (size_t i = 0; i < buffer->ptr_tail; i++) {
        offset_array[i] = (size_t) ((char*) ptr_array[i] - data_base);
    }

    buffer->sorted = 1;
    free(ptr_array);
    return MD_BUF_OK;
}

static inline int c_md_block_buffer_clear(md_block_buffer* buffer) {
    if (!buffer) return MD_BUF_ERR_INVALID;

    buffer->ptr_tail = 0;
    buffer->data_tail = 0;
    buffer->current_timestamp = 0.0;
    buffer->sorted = 1;
    return MD_BUF_OK;
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

static inline md_ring_buffer* c_md_ring_buffer_new(size_t ptr_capacity, size_t data_capacity, shm_allocator_ctx* shm_allocator, heap_allocator* heap_allocator, int with_lock) {
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
    if (!buffer) return MD_BUF_ERR_INVALID;

    shm_allocator* shm_allocator = buffer->shm_allocator;
    heap_allocator* heap_allocator = buffer->heap_allocator;

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
    return MD_BUF_OK;
}

static inline int c_md_ring_buffer_is_full(md_ring_buffer* buffer, md_variant* market_data) {
    if (!buffer) return MD_BUF_ERR_INVALID;

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
    if (!buffer) return MD_BUF_ERR_INVALID;

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

static inline int c_md_ring_buffer_put(md_ring_buffer* buffer, md_variant* market_data, int block, double timeout) {
    if (!buffer || !market_data) return MD_BUF_ERR_INVALID;

    const uint32_t spin_per_check = 1000;
    time_t start_time = 0;
    time_t current_time;
    double elapsed = 0.0;
    uint32_t spin_count = 0;
    uint32_t sleep_us = 0;
    const int use_timeout = timeout > 0.0;

    size_t serialized_size = c_md_serialized_size(market_data);

    time(&start_time);

    for (;;) {
        size_t* offset_array = (size_t*) (buffer->buffer + buffer->ptr_offset);
        size_t ptr_head = buffer->ptr_head;
        size_t ptr_tail = buffer->ptr_tail;
        size_t ptr_next = (ptr_tail + 1) % buffer->ptr_capacity;

        /* Check pointer slot availability */
        if (ptr_head == ptr_next) {
            if (!block) return MD_BUF_ERR_FULL; /* full, non-blocking */
        }
        else {
            /* Check data space availability and compute write_offset */
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
                    /* Wrap around */
                    if (serialized_size <= data_head) {
                        write_offset = 0;
                        buffer->data_tail = serialized_size;
                    }
                    else {
                        /* insufficient space; treat as full */
                        write_offset = (size_t) -1;
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
                    write_offset = (size_t) -1;
                }
            }

            if (write_offset != (size_t) -1) {
                /* Commit write */
                char* data_ptr = buffer->buffer + buffer->data_offset + write_offset;
                c_md_serialize(market_data, data_ptr);
                offset_array[ptr_tail] = write_offset;
                buffer->ptr_tail = ptr_next;
                return MD_BUF_OK;
            }

            /* No data space available */
            if (!block) return MD_BUF_ERR_FULL;
        }

        /* Blocking wait with timeout/backoff */
        if ((spin_count % spin_per_check) == 0) {
            time(&current_time);
            elapsed = difftime(current_time, start_time);

            if (use_timeout && elapsed >= timeout) {
                return MD_BUF_ERR_TIMEOUT; /* timeout */
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
        else {
            sched_yield();
        }
        spin_count += 1;
    }
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
    if (!buffer || !out) return MD_BUF_ERR_INVALID;

    const uint32_t spin_per_check = 1000;
    time_t start_time = 0;
    time_t current_time;
    double elapsed = 0.0;
    uint32_t spin_count = 0;
    uint32_t sleep_us = 0;
    const int use_timeout = timeout > 0.0;
    size_t idx = buffer->ptr_head;

    if (!block && idx == buffer->ptr_tail) {
        return MD_BUF_ERR_EMPTY; /* empty and non-blocking */
    }

    time(&start_time);

    for (;;) {
        if (idx != buffer->ptr_tail) {
            size_t* offset_array = (size_t*) (buffer->buffer + buffer->ptr_offset);
            size_t data_offset = offset_array[idx];
            if (data_offset >= buffer->data_capacity) return MD_BUF_ERR_CORRUPT; /* corrupt offset */

            const char* data_ptr = (buffer->buffer + buffer->data_offset + data_offset);
            buffer->ptr_head = (idx + 1) % buffer->ptr_capacity;
            *out = data_ptr;
            return MD_BUF_OK;
        }

        if ((spin_count % spin_per_check) == 0) {
            time(&current_time);
            elapsed = difftime(current_time, start_time);

            if (use_timeout && elapsed >= timeout) {
                return MD_BUF_ERR_TIMEOUT; /* timeout */
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
        else {
            sched_yield();
        }
        spin_count += 1;
    }
}

// ========== ConcurrentBuffer API Functions ==========

/* Concurrent buffer uses the same unified return codes */

static inline md_concurrent_buffer* c_md_concurrent_buffer_new(size_t n_workers, size_t capacity, shm_allocator_ctx* shm_allocator, int with_lock) {
    size_t size = sizeof(md_concurrent_buffer)
        + n_workers * sizeof(md_concurrent_buffer_worker_t)
        + capacity * sizeof(md_variant*);

    if (size == 0) return NULL;

    if (!shm_allocator) return NULL; /* only support shm allocator for now */

    pthread_mutex_t* lock = with_lock ? &shm_allocator->shm_allocator->lock : NULL;
    md_concurrent_buffer* buffer = (md_concurrent_buffer*) c_shm_request(shm_allocator, size, 1, lock);
    if (!buffer) return NULL;
    buffer->shm_allocator = shm_allocator->shm_allocator;
    buffer->n_workers = n_workers;
    buffer->workers = (md_concurrent_buffer_worker_t*) (buffer + 1);

    for (size_t i = 0; i < n_workers; i++) {
        buffer->workers[i].enabled = 1;
        // buffer->workers[i].ptr_head = 0;
    }

    buffer->buffer = (md_variant**) (buffer->workers + n_workers);
    buffer->capacity = capacity;
    buffer->tail = 0;
    return buffer;
}

static inline int c_md_concurrent_buffer_free(md_concurrent_buffer* buffer, int with_lock) {
    if (!buffer) return MD_BUF_ERR_INVALID;

    shm_allocator* shm_allocator = buffer->shm_allocator;

    if (shm_allocator) {
        pthread_mutex_t* lock = with_lock ? &shm_allocator->lock : NULL;
        c_shm_free((void*) buffer, lock);
    }
    else {
        return MD_BUF_ERR_INVALID;
    }
    return MD_BUF_OK;
}

static inline int c_md_concurrent_buffer_enable_worker(md_concurrent_buffer* buffer, size_t worker_id) {
    if (!buffer) return MD_BUF_ERR_INVALID;
    if (worker_id >= buffer->n_workers) return MD_BUF_OOR;

    buffer->workers[worker_id].enabled = 1;
    buffer->workers[worker_id].ptr_head = 0;
    return 0;
}

static inline int c_md_concurrent_buffer_disable_worker(md_concurrent_buffer* buffer, size_t worker_id) {
    if (!buffer) return MD_BUF_ERR_INVALID;
    if (worker_id >= buffer->n_workers) return MD_BUF_OOR;

    buffer->workers[worker_id].enabled = 0;
    buffer->workers[worker_id].ptr_head = (buffer->tail + buffer->capacity - 1) % buffer->capacity;
    return 0;
}

static inline int c_md_concurrent_buffer_is_full(md_concurrent_buffer* buffer) {
    if (!buffer) return MD_BUF_ERR_INVALID;

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
    if (!buffer) return MD_BUF_ERR_INVALID;
    if (worker_id >= buffer->n_workers) return MD_BUF_OOR;

    md_concurrent_buffer_worker_t* worker = buffer->workers + worker_id;
    if (!worker->enabled) return MD_BUF_DISABLED;
    if (worker->ptr_head == buffer->tail) return 1;
    else return 0;
}

static inline int c_md_concurrent_buffer_put(md_concurrent_buffer* buffer, md_variant* market_data, int block, double timeout) {
    /* Returns: MD_BUF_OK, MD_BUF_ERR_INVALID, MD_BUF_ERR_NOT_SHM, MD_BUF_ERR_FULL, MD_BUF_ERR_TIMEOUT */
    if (!buffer || !market_data) return MD_BUF_ERR_INVALID;

    if (!market_data->meta_info.shm_allocator) return MD_BUF_ERR_NOT_SHM;

    const uint32_t spin_per_check = 1000;
    time_t start_time = 0;
    time_t current_time;
    double elapsed = 0.0;
    uint32_t spin_count = 0;
    uint32_t sleep_us = 0;
    const int use_timeout = timeout > 0.0;

    size_t next_tail = (buffer->tail + 1) % buffer->capacity;

    // Fast check: if full and non-blocking, exit
    for (size_t i = 0; i < buffer->n_workers; i++) {
        md_concurrent_buffer_worker_t* worker = buffer->workers + i;
        if (!worker->enabled) continue;
        if (worker->ptr_head == next_tail) {
            if (!block) return MD_BUF_ERR_FULL; /* full, non-blocking */
            time(&start_time);
            break;
        }
    }

    // Blocking wait until a slot is available or timeout
    // Single-producer assumption: caller provides external synchronization
    while (block) {
        int is_full = 0;
        for (size_t i = 0; i < buffer->n_workers; i++) {
            md_concurrent_buffer_worker_t* worker = buffer->workers + i;
            if (!worker->enabled) continue;
            if (worker->ptr_head == next_tail) {
                is_full = 1;
                break;
            }
        }

        if (!is_full) break; /* space available */

        if ((spin_count % spin_per_check) == 0) {
            time(&current_time);
            elapsed = difftime(current_time, start_time);

            if (use_timeout && elapsed >= timeout) {
                return MD_BUF_ERR_TIMEOUT; /* timeout */
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
        else {
            sched_yield();
        }
        spin_count += 1;
    }

    // Commit write
    buffer->buffer[buffer->tail] = market_data;
    buffer->tail = next_tail;

    return MD_BUF_OK;
}

static inline int c_md_concurrent_buffer_listen(md_concurrent_buffer* buffer, size_t worker_id, int block, double timeout, md_variant** out) {
    /* Returns: MD_BUF_OK, MD_BUF_ERR_INVALID, -2 (worker OOR), -3 (worker disabled), MD_BUF_ERR_EMPTY, MD_BUF_ERR_TIMEOUT */
    if (!buffer || !out) return MD_BUF_ERR_INVALID;
    if (worker_id >= buffer->n_workers) return MD_BUF_OOR; /* worker out of range */

    md_concurrent_buffer_worker_t* worker = buffer->workers + worker_id;
    if (!worker->enabled) return MD_BUF_DISABLED; /* worker disabled */

    const uint32_t spin_per_check = 1000;
    time_t start_time = 0;
    time_t current_time;
    double elapsed = 0.0;
    uint32_t spin_count = 0;
    uint32_t sleep_us = 0;
    const int use_timeout = timeout > 0.0;
    size_t idx = worker->ptr_head;

    if (!block && idx == buffer->tail) {
        return MD_BUF_ERR_EMPTY; /* empty and non-blocking */
    }

    time(&start_time);

    for (;;) {
        if (idx != buffer->tail) {
            md_variant* md = buffer->buffer[idx];
            worker->ptr_head = (idx + 1) % buffer->capacity;
            *out = md;
            return MD_BUF_OK;
        }

        if ((spin_count % spin_per_check) == 0) {
            time(&current_time);
            elapsed = difftime(current_time, start_time);

            if (use_timeout && elapsed >= timeout) {
                return MD_BUF_ERR_TIMEOUT; /* timeout */
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
        else {
            sched_yield();
        }
        spin_count += 1;
    }
}

#endif /* C_MARKET_DATA_BUFFER_H */

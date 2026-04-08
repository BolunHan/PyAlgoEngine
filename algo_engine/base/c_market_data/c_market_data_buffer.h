#ifndef C_MARKET_DATA_BUFFER_H
#define C_MARKET_DATA_BUFFER_H

#include <math.h>
#include <pthread.h>
#include <sched.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "c_intern_string.h"
#include "c_market_data.h"

#ifndef MD_BUF_PTR_DEFAULT_CAP
#define MD_BUF_PTR_DEFAULT_CAP 16
#endif

#ifndef MD_BUF_DATA_DEFAULT_CAP
#define MD_BUF_DATA_DEFAULT_CAP 1024
#endif

// ========== MarketData Buffer Structs ==========

typedef struct md_ptr_array {
    size_t capacity;
    size_t idx_head;
    size_t idx_tail;
    size_t* offsets;  // Array of offsets to the actual data in the buffer.
} md_ptr_array;

typedef struct md_data_array {
    size_t capacity;
    size_t occupied;
    char* buf;  // Contiguous buffer storing serialized market data.
} md_data_array;

typedef struct md_block_buffer {
    md_ptr_array ptr_array;
    md_data_array data_array;
    double current_timestamp;
    bool sorted;
} md_block_buffer;

typedef struct md_ring_buffer {
    md_ptr_array ptr_array;
    md_data_array data_array;
} md_ring_buffer;

typedef struct md_concurrent_buffer_worker_t {
    size_t ptr_head;
    bool enabled;
} md_concurrent_buffer_worker_t;

typedef struct md_concurrent_buffer {
    md_concurrent_buffer_worker_t* workers;
    size_t n_workers;
    md_variant** buffer;
    size_t capacity;
    size_t tail;
} md_concurrent_buffer;

// ========== Forward Declarations (Public API) ==========

static inline int c_md_compare_serialized(const void* a, const void* b);
static inline size_t c_md_total_buffer_size(md_variant** md_array, size_t n_md);
static inline md_variant* c_md_send_to_shm(md_variant* market_data, allocator_protocol* shm_allocator, istr_map* shm_pool);

static inline md_block_buffer* c_md_block_buffer_new(size_t ptr_capacity, size_t data_capacity, allocator_protocol* allocator);
static inline void c_md_block_buffer_free(md_block_buffer* buffer);
static inline int c_md_block_buffer_extend(md_block_buffer* buffer, size_t new_ptr_capacity, size_t new_data_capacity);
static inline int c_md_block_buffer_put(md_block_buffer* buffer, md_variant* market_data);
static inline int c_md_block_buffer_get(md_block_buffer* buffer, size_t idx, const char** out);
static inline int c_md_block_buffer_sort(md_block_buffer* buffer);
static inline int c_md_block_buffer_clear(md_block_buffer* buffer);
static inline size_t c_md_block_buffer_serialized_size(md_block_buffer* buffer);
static inline int c_md_block_buffer_serialize(md_block_buffer* buffer, char* out);
static inline md_block_buffer* c_md_block_buffer_deserialize(const char* blob, allocator_protocol* allocator);

static inline md_ring_buffer* c_md_ring_buffer_new(size_t ptr_capacity, size_t data_capacity, allocator_protocol* allocator);
static inline void c_md_ring_buffer_free(md_ring_buffer* buffer);
static inline int c_md_ring_buffer_is_full(md_ring_buffer* buffer, md_variant* market_data);
static inline int c_md_ring_buffer_is_empty(md_ring_buffer* buffer);
static inline size_t c_md_ring_buffer_size(md_ring_buffer* buffer);
static inline int c_md_ring_buffer_put(md_ring_buffer* buffer, md_variant* market_data, bool block, double timeout);
static inline int c_md_ring_buffer_get(md_ring_buffer* buffer, size_t index, const char** out);
static inline int c_md_ring_buffer_listen(md_ring_buffer* buffer, bool block, double timeout, const char** out);

static inline md_concurrent_buffer* c_md_concurrent_buffer_new(size_t n_workers, size_t capacity, allocator_protocol* shm_allocator);
static inline void c_md_concurrent_buffer_free(md_concurrent_buffer* buffer);
static inline int c_md_concurrent_buffer_enable_worker(md_concurrent_buffer* buffer, size_t worker_id);
static inline int c_md_concurrent_buffer_disable_worker(md_concurrent_buffer* buffer, size_t worker_id);
static inline int c_md_concurrent_buffer_is_full(md_concurrent_buffer* buffer);
static inline int c_md_concurrent_buffer_is_empty(md_concurrent_buffer* buffer, size_t worker_id);
static inline int c_md_concurrent_buffer_put(md_concurrent_buffer* buffer, md_variant* market_data, bool block, double timeout);
static inline int c_md_concurrent_buffer_listen(md_concurrent_buffer* buffer, size_t worker_id, bool block, double timeout, md_variant** out);

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

static inline md_variant* c_md_send_to_shm(md_variant* market_data, allocator_protocol* shm_allocator, istr_map* shm_pool) {
    if (!market_data || !shm_allocator || !shm_pool) return NULL;

    // Step 1: Intern ticker string
    const char* interned_ticker;
    if (market_data->meta_info.ticker) {
        if (shm_allocator->with_lock) {
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
    allocator_protocol* original_allocator = c_md_protocol_from_ptr(market_data);
    if (original_allocator->shm_allocator == shm_allocator->shm_allocator) {
        market_data->meta_info.ticker = interned_ticker;
        return market_data;
    }

    // Step 3: Initialize new market_data in SHM
    md_data_type dtype = market_data->meta_info.dtype;
    md_variant* payload = c_md_new(dtype, shm_allocator);
    size_t size = c_md_get_size(dtype);
    if (!payload) return NULL;

    memcpy((void*) payload, (void*) market_data, size);
    payload->meta_info.ticker = interned_ticker;

    // Step 4: Handle order book pointers if md_tick_data
    if (dtype == DTYPE_TICK) {
        md_tick_data* src_tick = &market_data->tick_data_full;
        md_tick_data* dst_tick = &payload->tick_data_full;

        // Bid order book
        if (src_tick->bid) {
            size_t ob_capacity = src_tick->bid->capacity;
            size_t ob_size = sizeof(md_orderbook) + (ob_capacity * sizeof(md_orderbook_entry));
            md_orderbook* ob_shm = c_md_orderbook_new(ob_capacity, shm_allocator);
            if (!ob_shm) {
                c_md_free(payload);
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
            md_orderbook* ob_shm = c_md_orderbook_new(ob_capacity, shm_allocator);
            if (!ob_shm) {
                if (dst_tick->bid) {
                    c_md_free(dst_tick->bid);
                }
                c_md_free(payload);
                return NULL;
            }
            memcpy((void*) ob_shm, (void*) src_tick->ask, ob_size);
            dst_tick->ask = ob_shm;
        }
        else {
            dst_tick->ask = NULL;
        }
    }

    return payload;
}

// ========== BlockBuffer API Functions ==========

static inline md_block_buffer* c_md_block_buffer_new(size_t ptr_capacity, size_t data_capacity, allocator_protocol* allocator) {
    md_block_buffer* buffer = (md_block_buffer*) c_md_alloc(sizeof(md_block_buffer), allocator);
    if (!buffer) return NULL;

    if (ptr_capacity) {
        buffer->ptr_array.capacity = ptr_capacity;
        buffer->ptr_array.offsets = (size_t*) c_md_alloc(ptr_capacity * sizeof(size_t), allocator);
        if (!buffer->ptr_array.offsets) goto oom;
    }

    if (data_capacity) {
        buffer->data_array.capacity = data_capacity;
        buffer->data_array.buf = (char*) c_md_alloc(data_capacity, allocator);
        if (!buffer->data_array.buf) goto oom;
    }

    buffer->sorted = 1;
    buffer->current_timestamp = NAN;
    return buffer;

oom:
    c_md_block_buffer_free(buffer);
    return NULL;
}

static inline void c_md_block_buffer_free(md_block_buffer* buffer) {
    if (!buffer) return;
    if (buffer->ptr_array.offsets) c_md_free(buffer->ptr_array.offsets);
    if (buffer->data_array.buf) c_md_free(buffer->data_array.buf);
    c_md_free(buffer);
}

static inline int c_md_block_buffer_extend(md_block_buffer* buffer, size_t new_ptr_capacity, size_t new_data_capacity) {
    if (!buffer) return MD_ERR_INVALID_INPUT;

    allocator_protocol* allocator = c_md_protocol_from_ptr(buffer);

    if (new_ptr_capacity > buffer->ptr_array.capacity) {
        size_t* new_offsets = (size_t*) c_md_alloc(new_ptr_capacity * sizeof(size_t), allocator);
        if (!new_offsets) return MD_ERR_OOM;
        if (buffer->ptr_array.offsets) {
            memcpy(new_offsets, buffer->ptr_array.offsets, buffer->ptr_array.capacity * sizeof(size_t));
            c_md_free(buffer->ptr_array.offsets);
        }
        buffer->ptr_array.offsets = new_offsets;
        buffer->ptr_array.capacity = new_ptr_capacity;
    }

    if (new_data_capacity > buffer->data_array.capacity) {
        char* new_data_buf = (char*) c_md_alloc(new_data_capacity, allocator);
        if (!new_data_buf) return MD_ERR_OOM;
        if (buffer->data_array.buf) {
            memcpy(new_data_buf, buffer->data_array.buf, buffer->data_array.capacity);
            c_md_free(buffer->data_array.buf);
        }
        buffer->data_array.buf = new_data_buf;
        buffer->data_array.capacity = new_data_capacity;
    }

    return MD_OK;
}

static inline int c_md_block_buffer_put(md_block_buffer* buffer, md_variant* market_data) {
    if (!buffer || !market_data) return MD_ERR_INVALID_INPUT;

    size_t idx_tail = buffer->ptr_array.idx_tail;
    size_t ptr_capacity = buffer->ptr_array.capacity;
    if (idx_tail >= ptr_capacity) {
        int ret_code = c_md_block_buffer_extend(
            buffer,
            ptr_capacity ? ptr_capacity * 2 : MD_BUF_PTR_DEFAULT_CAP,
            0
        );
        if (ret_code != MD_OK) return ret_code;
    }

    size_t serialized_size = c_md_serialized_size(market_data);
    size_t data_tail = buffer->data_array.occupied;
    size_t data_capacity = buffer->data_array.capacity;
    if (data_tail + serialized_size > data_capacity) {
        int ret_code = c_md_block_buffer_extend(
            buffer,
            0,
            data_capacity ? data_capacity * 2 : MD_BUF_DATA_DEFAULT_CAP
        );
        if (ret_code != MD_OK) return ret_code;
    }

    char* data_ptr = buffer->data_array.buf + data_tail;
    c_md_serialize(market_data, data_ptr);

    buffer->ptr_array.offsets[idx_tail] = data_tail;
    buffer->ptr_array.idx_tail++;
    buffer->data_array.occupied += serialized_size;

    double ts = market_data->meta_info.timestamp;
    if (idx_tail > 0) {
        if (ts < buffer->current_timestamp) buffer->sorted = 0;
        else buffer->current_timestamp = ts;
    }
    else {
        buffer->sorted = 1;
        buffer->current_timestamp = ts;
    }

    return MD_OK;
}

static inline int c_md_block_buffer_get(md_block_buffer* buffer, size_t idx, const char** out) {
    if (!buffer) return MD_ERR_INVALID_INPUT;
    size_t ptr_tail = buffer->ptr_array.idx_tail;
    if (idx >= ptr_tail) return MD_ERR_OOR;

    size_t data_offset = buffer->ptr_array.offsets[idx];
    if (out) *out = (const char*) buffer->data_array.buf + data_offset;
    return MD_OK;
}

static inline int c_md_block_buffer_sort(md_block_buffer* buffer) {
    if (!buffer) return MD_ERR_INVALID_INPUT;
    if (buffer->sorted) return MD_OK;

    size_t ptr_tail = buffer->ptr_array.idx_tail;
    if (ptr_tail <= 1) {
        buffer->sorted = 1;
        return MD_OK;
    }

    size_t* offset_array = buffer->ptr_array.offsets;
    char* data_array = buffer->data_array.buf;
    md_variant** ptr_array = (md_variant**) malloc(ptr_tail * sizeof(md_variant*));
    if (!ptr_array) return MD_ERR_OOM;

    for (size_t i = 0; i < ptr_tail; i++) {
        ptr_array[i] = (md_variant*) (data_array + offset_array[i]);
    }

    qsort(ptr_array, ptr_tail, sizeof(md_variant*), c_md_compare_serialized);

    for (size_t i = 0; i < ptr_tail; i++) {
        offset_array[i] = (size_t) ((char*) ptr_array[i] - data_array);
    }

    buffer->sorted = 1;
    free(ptr_array);
    return MD_OK;
}

static inline int c_md_block_buffer_clear(md_block_buffer* buffer) {
    if (!buffer) return MD_ERR_INVALID_INPUT;

    buffer->ptr_array.idx_tail = 0;
    if (buffer->ptr_array.offsets) memset(buffer->ptr_array.offsets, 0, buffer->ptr_array.capacity * sizeof(size_t));
    buffer->data_array.occupied = 0;
    if (buffer->data_array.buf) memset(buffer->data_array.buf, 0, buffer->data_array.capacity);
    buffer->current_timestamp = NAN;
    buffer->sorted = 1;
    return MD_OK;
}

static inline size_t c_md_block_buffer_serialized_size(md_block_buffer* buffer) {
    if (!buffer) return 0;
    size_t ttl_size = sizeof(md_block_buffer);
    ttl_size += buffer->ptr_array.idx_tail * sizeof(size_t);
    ttl_size += buffer->data_array.occupied;
    return ttl_size;
}

static inline int c_md_block_buffer_serialize(md_block_buffer* buffer, char* out) {
    if (!buffer || !out) return 0;

    char* ptr = out;
    memcpy(ptr, buffer, sizeof(md_block_buffer));
    ptr += sizeof(md_block_buffer);

    memcpy(ptr, buffer->ptr_array.offsets, buffer->ptr_array.idx_tail * sizeof(size_t));
    ptr += buffer->ptr_array.idx_tail * sizeof(size_t);

    memcpy(ptr, buffer->data_array.buf, buffer->data_array.occupied);
    ptr += buffer->data_array.occupied;

    md_block_buffer* out_buf = (md_block_buffer*) out;
    out_buf->ptr_array.offsets = NULL;
    out_buf->ptr_array.capacity = buffer->ptr_array.idx_tail;
    out_buf->data_array.buf = NULL;
    out_buf->data_array.capacity = buffer->data_array.occupied;

    return MD_OK;
}

static inline md_block_buffer* c_md_block_buffer_deserialize(const char* blob, allocator_protocol* allocator) {
    if (!blob) return NULL;

    md_block_buffer* in_buf = (md_block_buffer*) blob;
    md_block_buffer* buffer = c_md_block_buffer_new(
        in_buf->ptr_array.idx_tail,
        in_buf->data_array.occupied,
        allocator
    );
    if (!buffer) return NULL;

    buffer->ptr_array.idx_tail = in_buf->ptr_array.idx_tail;
    buffer->data_array.occupied = in_buf->data_array.occupied;
    buffer->current_timestamp = in_buf->current_timestamp;
    buffer->sorted = in_buf->sorted;
    const char* ptr = blob + sizeof(md_block_buffer);

    memcpy(buffer->ptr_array.offsets, ptr, in_buf->ptr_array.idx_tail * sizeof(size_t));
    ptr += in_buf->ptr_array.idx_tail * sizeof(size_t);

    memcpy(buffer->data_array.buf, ptr, in_buf->data_array.occupied);
    ptr += in_buf->data_array.occupied;

    return buffer;
}

// ========== RingBuffer API Functions ==========

static inline md_ring_buffer* c_md_ring_buffer_new(size_t ptr_capacity, size_t data_capacity, allocator_protocol* allocator) {
    md_ring_buffer* buffer = (md_ring_buffer*) c_md_alloc(sizeof(md_ring_buffer), allocator);
    if (!buffer) return NULL;

    buffer->ptr_array.capacity = ptr_capacity;
    if (ptr_capacity) {
        buffer->ptr_array.offsets = (size_t*) c_md_alloc(ptr_capacity * sizeof(size_t), allocator);
        if (!buffer->ptr_array.offsets) goto oom;
    }

    buffer->data_array.capacity = data_capacity;
    if (data_capacity) {
        buffer->data_array.buf = (char*) c_md_alloc(data_capacity, allocator);
        if (!buffer->data_array.buf) goto oom;
    }

    return buffer;

oom:
    c_md_ring_buffer_free(buffer);
    return NULL;
}

static inline void c_md_ring_buffer_free(md_ring_buffer* buffer) {
    if (!buffer) return;
    if (buffer->ptr_array.offsets) c_md_free(buffer->ptr_array.offsets);
    if (buffer->data_array.buf) c_md_free(buffer->data_array.buf);
    c_md_free(buffer);
}

static inline int c_md_ring_buffer_is_full(md_ring_buffer* buffer, md_variant* market_data) {
    if (!buffer) return MD_ERR_INVALID_INPUT;

    size_t ptr_head = buffer->ptr_array.idx_head;
    size_t ptr_tail = buffer->ptr_array.idx_tail;
    size_t ptr_capacity = buffer->ptr_array.capacity;
    size_t ptr_next = (ptr_tail + 1) % ptr_capacity;

    if (ptr_head == ptr_next) return 1;

    if (ptr_head == ptr_tail) return 0;

    size_t payload_size = c_md_serialized_size(market_data);
    size_t data_head = buffer->ptr_array.offsets[ptr_head];
    size_t data_tail = buffer->data_array.occupied;

    if (data_tail >= data_head) {
        size_t space_end = buffer->data_array.capacity - data_tail;
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
    if (!buffer) return MD_ERR_INVALID_INPUT;

    size_t ptr_head = buffer->ptr_array.idx_head;
    size_t ptr_tail = buffer->ptr_array.idx_tail;

    if (ptr_head == ptr_tail) return 1;
    else return 0;
}

static inline size_t c_md_ring_buffer_size(md_ring_buffer* buffer) {
    if (!buffer) return 0;

    size_t ptr_head = buffer->ptr_array.idx_head;
    size_t ptr_tail = buffer->ptr_array.idx_tail;
    size_t capacity = buffer->ptr_array.capacity;

    if (ptr_tail >= ptr_head) {
        return ptr_tail - ptr_head;
    }
    else {
        return (capacity - ptr_head) + ptr_tail;
    }
}

static inline int c_md_ring_buffer_put(md_ring_buffer* buffer, md_variant* market_data, bool block, double timeout) {
    if (!buffer || !market_data) return MD_ERR_INVALID_INPUT;

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
        size_t* offset_array = buffer->ptr_array.offsets;
        size_t ptr_head = buffer->ptr_array.idx_head;
        size_t ptr_tail = buffer->ptr_array.idx_tail;
        size_t ptr_next = (ptr_tail + 1) % buffer->ptr_array.capacity;

        /* Check pointer slot availability */
        if (ptr_head == ptr_next) {
            if (!block) return MD_ERR_BUF_FULL; /* full, non-blocking */
        }
        else {
            /* Check data space availability and compute write_offset */
            size_t data_head = offset_array[ptr_head];
            size_t data_tail = buffer->data_array.occupied;
            size_t write_offset = 0;

            if (data_tail >= data_head) {
                size_t space_end = buffer->data_array.capacity - data_tail;
                if (serialized_size <= space_end) {
                    write_offset = data_tail;
                    buffer->data_array.occupied += serialized_size;
                }
                else {
                    /* Wrap around */
                    if (serialized_size <= data_head) {
                        write_offset = 0;
                        buffer->data_array.occupied = serialized_size;
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
                    buffer->data_array.occupied += serialized_size;
                }
                else {
                    write_offset = (size_t) -1;
                }
            }

            if (write_offset != (size_t) -1) {
                /* Commit write */
                char* data_ptr = buffer->data_array.buf + write_offset;
                c_md_serialize(market_data, data_ptr);
                offset_array[ptr_tail] = write_offset;
                buffer->ptr_array.idx_tail = ptr_next;
                return MD_OK;
            }

            /* No data space available */
            if (!block) return MD_ERR_BUF_FULL;
        }

        /* Blocking wait with timeout/backoff */
        if ((spin_count % spin_per_check) == 0) {
            time(&current_time);
            elapsed = difftime(current_time, start_time);

            if (use_timeout && elapsed >= timeout) {
                return MD_ERR_TIMEOUT; /* timeout */
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

static inline int c_md_ring_buffer_get(md_ring_buffer* buffer, size_t index, const char** out) {
    if (!buffer) return MD_ERR_INVALID_INPUT;
    if (index >= buffer->ptr_array.capacity) return MD_ERR_OOR;
    size_t size = c_md_ring_buffer_size(buffer);
    if (index >= size) return MD_ERR_OOR;

    size_t ptr_head = buffer->ptr_array.idx_head;
    size_t ptr_capacity = buffer->ptr_array.capacity;
    size_t ptr_idx = (ptr_head + index) % ptr_capacity;

    size_t* offset_array = buffer->ptr_array.offsets;
    size_t data_offset = offset_array[ptr_idx];

    if (data_offset >= buffer->data_array.capacity) return MD_ERR_BUF_CORRUPTED;
    if (out) *out = buffer->data_array.buf + data_offset;
    return MD_OK;
}

static inline int c_md_ring_buffer_listen(md_ring_buffer* buffer, bool block, double timeout, const char** out) {
    if (!buffer || !out) return MD_ERR_INVALID_INPUT;

    const uint32_t spin_per_check = 1000;
    time_t start_time = 0;
    time_t current_time;
    double elapsed = 0.0;
    uint32_t spin_count = 0;
    uint32_t sleep_us = 0;
    const int use_timeout = timeout > 0.0;
    size_t idx = buffer->ptr_array.idx_head;

    if (!block && idx == buffer->ptr_array.idx_tail) {
        return MD_ERR_BUF_EMPTY; /* empty and non-blocking */
    }

    time(&start_time);

    for (;;) {
        if (idx != buffer->ptr_array.idx_tail) {
            size_t* offset_array = buffer->ptr_array.offsets;
            size_t data_offset = offset_array[idx];
            if (data_offset >= buffer->data_array.capacity) return MD_ERR_BUF_CORRUPTED; /* corrupt offset */
            const char* data_ptr = buffer->data_array.buf + data_offset;
            buffer->ptr_array.idx_head = (idx + 1) % buffer->ptr_array.capacity;
            if (out) *out = data_ptr;
            return MD_OK;
        }

        if ((spin_count % spin_per_check) == 0) {
            time(&current_time);
            elapsed = difftime(current_time, start_time);

            if (use_timeout && elapsed >= timeout) {
                return MD_ERR_TIMEOUT; /* timeout */
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

static inline md_concurrent_buffer* c_md_concurrent_buffer_new(size_t n_workers, size_t capacity, allocator_protocol* shm_allocator) {
    size_t size = sizeof(md_concurrent_buffer)
        + n_workers * sizeof(md_concurrent_buffer_worker_t)
        + capacity * sizeof(md_variant*);

    if (size == 0) return NULL;

    if (!shm_allocator || !shm_allocator->with_shm) return NULL; /* only support shm allocator!*/

    md_concurrent_buffer* buffer = (md_concurrent_buffer*) c_md_alloc(size, shm_allocator);
    if (!buffer) return NULL;
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

static inline void c_md_concurrent_buffer_free(md_concurrent_buffer* buffer) {
    if (!buffer) return;
    c_md_free((void*) buffer);
}

static inline int c_md_concurrent_buffer_enable_worker(md_concurrent_buffer* buffer, size_t worker_id) {
    if (!buffer) return MD_ERR_INVALID_INPUT;
    if (worker_id >= buffer->n_workers) return MD_ERR_OOR;

    buffer->workers[worker_id].enabled = 1;
    buffer->workers[worker_id].ptr_head = 0;
    return 0;
}

static inline int c_md_concurrent_buffer_disable_worker(md_concurrent_buffer* buffer, size_t worker_id) {
    if (!buffer) return MD_ERR_INVALID_INPUT;
    if (worker_id >= buffer->n_workers) return MD_ERR_OOR;

    buffer->workers[worker_id].enabled = 0;
    buffer->workers[worker_id].ptr_head = (buffer->tail + buffer->capacity - 1) % buffer->capacity;
    return 0;
}

static inline int c_md_concurrent_buffer_is_full(md_concurrent_buffer* buffer) {
    if (!buffer) return MD_ERR_INVALID_INPUT;

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
    if (!buffer) return MD_ERR_INVALID_INPUT;
    if (worker_id >= buffer->n_workers) return MD_ERR_OOR;

    md_concurrent_buffer_worker_t* worker = buffer->workers + worker_id;
    if (!worker->enabled) return MD_ERR_DISABLED;
    if (worker->ptr_head == buffer->tail) return 1;
    else return 0;
}

static inline int c_md_concurrent_buffer_put(md_concurrent_buffer* buffer, md_variant* market_data, bool block, double timeout) {
    /* Returns: MD_OK, MD_ERR_INVALID_INPUT, MD_ERR_INVALID_ALLOCATOR, MD_ERR_BUF_FULL, MD_ERR_TIMEOUT */
    if (!buffer || !market_data) return MD_ERR_INVALID_INPUT;

    allocator_protocol* market_data_allocator = c_md_protocol_from_ptr(market_data);
    if (!market_data_allocator->with_shm) return MD_ERR_INVALID_ALLOCATOR;

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
            if (!block) return MD_ERR_BUF_FULL; /* full, non-blocking */
            time(&start_time);
            break;
        }
    }

    // Blocking wait until a slot is available or timeout
    // Single-producer assumption: caller provides external synchronization
    while (block) {
        bool is_full = 0;
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
                return MD_ERR_TIMEOUT; /* timeout */
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

    return MD_OK;
}

static inline int c_md_concurrent_buffer_listen(md_concurrent_buffer* buffer, size_t worker_id, bool block, double timeout, md_variant** out) {
    /* Returns: MD_OK, MD_ERR_INVALID_INPUT, -2 (worker OOR), -3 (worker disabled), MD_ERR_BUF_EMPTY, MD_ERR_TIMEOUT */
    if (!buffer || !out) return MD_ERR_INVALID_INPUT;
    if (worker_id >= buffer->n_workers) return MD_ERR_OOR; /* worker out of range */

    md_concurrent_buffer_worker_t* worker = buffer->workers + worker_id;
    if (!worker->enabled) return MD_ERR_DISABLED; /* worker disabled */

    const uint32_t spin_per_check = 1000;
    time_t start_time = 0;
    time_t current_time;
    double elapsed = 0.0;
    uint32_t spin_count = 0;
    uint32_t sleep_us = 0;
    const int use_timeout = timeout > 0.0;
    size_t idx = worker->ptr_head;

    if (!block && idx == buffer->tail) {
        return MD_ERR_BUF_EMPTY; /* empty and non-blocking */
    }

    time(&start_time);

    for (;;) {
        if (idx != buffer->tail) {
            md_variant* md = buffer->buffer[idx];
            worker->ptr_head = (idx + 1) % buffer->capacity;
            *out = md;
            return MD_OK;
        }

        if ((spin_count % spin_per_check) == 0) {
            time(&current_time);
            elapsed = difftime(current_time, start_time);

            if (use_timeout && elapsed >= timeout) {
                return MD_ERR_TIMEOUT; /* timeout */
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

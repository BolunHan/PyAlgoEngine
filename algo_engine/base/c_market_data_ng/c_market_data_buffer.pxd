# cython: language_level=3
from cpython.object cimport PyObject

from .c_market_data cimport market_data_t
from ..c_heap_allocator cimport heap_allocator_t, C_ALLOCATOR as HEAP_ALLOCATOR
from ..c_shm_allocator cimport shm_allocator_ctx, shm_allocator_t, C_ALLOCATOR as SHM_ALLOCATOR
from ..c_intern_string cimport C_POOL as SHM_POOL, C_INTRA_POOL as HEAP_POOL, istr_map, c_istr, c_istr_synced


cdef extern from "c_market_data_buffer.h":
    ctypedef struct md_block_buffer:
        shm_allocator_t* shm_allocator
        heap_allocator_t* heap_allocator
        int sorted
        size_t ptr_capacity
        size_t ptr_offset
        size_t ptr_tail
        size_t data_capacity
        size_t data_offset
        size_t data_tail
        double current_timestamp
        char buffer[]

    ctypedef struct md_ring_buffer:
        shm_allocator_t* shm_allocator
        heap_allocator_t* heap_allocator
        size_t ptr_capacity
        size_t ptr_offset
        size_t ptr_head
        size_t ptr_tail
        size_t data_capacity
        size_t data_offset
        size_t data_tail
        char buffer[]

    ctypedef struct md_concurrent_buffer_worker_t:
        size_t ptr_head
        int enabled

    ctypedef struct md_concurrent_buffer:
        shm_allocator_t* shm_allocator
        md_concurrent_buffer_worker_t* workers
        size_t n_workers
        market_data_t** buffer
        size_t capacity
        size_t tail

    size_t c_md_total_buffer_size(market_data_t** md_array, size_t n_md)
    market_data_t* c_md_send_to_shm(market_data_t* market_data, shm_allocator_ctx* shm_allocator, istr_map* shm_pool, int with_lock)

    md_block_buffer* c_md_block_buffer_new(size_t ptr_capacity, size_t data_capacity, shm_allocator_ctx* shm_allocator, heap_allocator_t* heap_allocator, int with_lock)
    int c_md_block_buffer_free(md_block_buffer* buffer, int with_lock)
    md_block_buffer* c_md_block_buffer_extend(md_block_buffer* buffer, size_t new_ptr_capacity, size_t new_data_capacity, shm_allocator_ctx* shm_allocator, heap_allocator_t* heap_allocator, int with_lock)
    int c_md_block_buffer_put(md_block_buffer* buffer, market_data_t* market_data)
    const char* c_md_block_buffer_get(md_block_buffer* buffer, size_t index)
    int c_md_block_buffer_sort(md_block_buffer* buffer)
    int c_md_block_buffer_clear(md_block_buffer* buffer)
    size_t c_md_block_buffer_serialized_size(md_block_buffer* buffer)
    size_t c_md_block_buffer_serialize(md_block_buffer* buffer, char* out_buffer)

    md_ring_buffer* c_md_ring_buffer_new(size_t ptr_capacity, size_t data_capacity, shm_allocator_ctx* shm_allocator, heap_allocator_t* heap_allocator, int with_lock)
    int c_md_ring_buffer_free(md_ring_buffer* buffer, int with_lock)
    int c_md_ring_buffer_is_full(md_ring_buffer* buffer, market_data_t* market_data)
    int c_md_ring_buffer_is_empty(md_ring_buffer* buffer)
    size_t c_md_ring_buffer_put(md_ring_buffer* buffer, market_data_t* market_data)
    market_data_t* c_md_ring_buffer_get(md_ring_buffer* buffer)
    int c_md_ring_buffer_listen(md_ring_buffer* buffer, int block, double timeout, market_data_t** out)

    md_concurrent_buffer* c_md_concurrent_buffer_new(size_t n_workers, size_t capacity, shm_allocator_ctx* shm_allocator, int with_lock)
    int c_md_concurrent_buffer_free(md_concurrent_buffer* buffer, int with_lock)
    int c_md_concurrent_buffer_enable_worker(md_concurrent_buffer* buffer, size_t worker_id)
    int c_md_concurrent_buffer_disable_worker(md_concurrent_buffer* buffer, size_t worker_id)
    int c_md_concurrent_buffer_is_full(md_concurrent_buffer* buffer, market_data_t* market_data)
    int c_md_concurrent_buffer_is_empty(md_concurrent_buffer* buffer, size_t worker_id)
    int c_md_concurrent_buffer_put(md_concurrent_buffer* buffer, market_data_t* market_data)


cdef class MarketDataBufferCache:
    cdef market_data_t** c_array
    cdef PyObject** py_array

    cdef readonly object parent
    cdef readonly size_t capacity
    cdef readonly size_t size

    cdef int c_write_block_buffer(self, MarketDataBuffer buffer)


cdef class MarketDataBuffer:
    cdef md_block_buffer* header
    cdef size_t md_array_size
    cdef bint owner
    cdef size_t iter_idx

    cdef void c_sort(self)

    cdef void c_put(self, market_data_t* market_data)

    cdef market_data_t* c_get(self, ssize_t idx)

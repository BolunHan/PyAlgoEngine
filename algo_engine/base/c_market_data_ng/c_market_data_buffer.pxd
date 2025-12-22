# cython: language_level=3
from cpython.object cimport PyObject

from .c_market_data cimport md_variant
from ..c_heap_allocator cimport heap_allocator, C_ALLOCATOR as HEAP_ALLOCATOR
from ..c_shm_allocator cimport shm_allocator_ctx, shm_allocator, C_ALLOCATOR as SHM_ALLOCATOR
from ..c_intern_string cimport C_POOL as SHM_POOL, C_INTRA_POOL as HEAP_POOL, istr_map, c_istr, c_istr_synced


cdef extern from "c_market_data_buffer.h":
    const int MD_BUF_OK
    const int MD_BUF_ERR_INVALID
    const int MD_BUF_ERR_NOT_SHM
    const int MD_BUF_ERR_FULL
    const int MD_BUF_ERR_EMPTY
    const int MD_BUF_ERR_TIMEOUT
    const int MD_BUF_ERR_CORRUPT
    const int MD_BUF_OOR
    const int MD_BUF_DISABLED

    ctypedef struct md_block_buffer:
        shm_allocator* shm_allocator
        heap_allocator* heap_allocator
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
        shm_allocator* shm_allocator
        heap_allocator* heap_allocator
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
        shm_allocator* shm_allocator
        md_concurrent_buffer_worker_t* workers
        size_t n_workers
        md_variant** buffer
        size_t capacity
        size_t tail

    int c_md_compare_serialized(const void* a, const void* b)
    size_t c_md_total_buffer_size(md_variant** md_array, size_t n_md)
    md_variant* c_md_send_to_shm(md_variant* market_data, shm_allocator_ctx* shm_allocator, istr_map* shm_pool, int with_lock)

    md_block_buffer* c_md_block_buffer_new(size_t ptr_capacity, size_t data_capacity, shm_allocator_ctx* shm_allocator, heap_allocator* heap_allocator, int with_lock)
    int c_md_block_buffer_free(md_block_buffer* buffer, int with_lock)
    md_block_buffer* c_md_block_buffer_extend(md_block_buffer* buffer, size_t new_ptr_capacity, size_t new_data_capacity, shm_allocator_ctx* shm_allocator, heap_allocator* heap_allocator, int with_lock)
    int c_md_block_buffer_put(md_block_buffer* buffer, md_variant* market_data)
    const char* c_md_block_buffer_get(md_block_buffer* buffer, size_t index)
    int c_md_block_buffer_sort(md_block_buffer* buffer)
    int c_md_block_buffer_clear(md_block_buffer* buffer)
    size_t c_md_block_buffer_serialized_size(md_block_buffer* buffer)
    size_t c_md_block_buffer_serialize(md_block_buffer* buffer, char* out_buffer)

    md_ring_buffer* c_md_ring_buffer_new(size_t ptr_capacity, size_t data_capacity, shm_allocator_ctx* shm_allocator, heap_allocator* heap_allocator, int with_lock)
    int c_md_ring_buffer_free(md_ring_buffer* buffer, int with_lock)
    int c_md_ring_buffer_is_full(md_ring_buffer* buffer, md_variant* market_data)
    int c_md_ring_buffer_is_empty(md_ring_buffer* buffer)
    size_t c_md_ring_buffer_size(md_ring_buffer* buffer)
    int c_md_ring_buffer_put(md_ring_buffer* buffer, md_variant* market_data, int block, double timeout)
    const char* c_md_ring_buffer_get(md_ring_buffer* buffer, size_t index)
    int c_md_ring_buffer_listen(md_ring_buffer* buffer, int block, double timeout, const char** out)

    md_concurrent_buffer* c_md_concurrent_buffer_new(size_t n_workers, size_t capacity, shm_allocator_ctx* shm_allocator, int with_lock)
    int c_md_concurrent_buffer_free(md_concurrent_buffer* buffer, int with_lock)
    int c_md_concurrent_buffer_enable_worker(md_concurrent_buffer* buffer, size_t worker_id)
    int c_md_concurrent_buffer_disable_worker(md_concurrent_buffer* buffer, size_t worker_id)
    int c_md_concurrent_buffer_is_full(md_concurrent_buffer* buffer)
    int c_md_concurrent_buffer_is_empty(md_concurrent_buffer* buffer, size_t worker_id)
    int c_md_concurrent_buffer_put(md_concurrent_buffer* buffer, md_variant* market_data, int block, double timeout)
    int c_md_concurrent_buffer_listen(md_concurrent_buffer* buffer, size_t worker_id, int block, double timeout, md_variant** out)


cdef class MarketDataBufferCache:
    cdef md_variant** c_array
    cdef PyObject** py_array

    cdef readonly object parent
    cdef readonly size_t capacity
    cdef readonly size_t size

    cdef int c_write_block_buffer(self, MarketDataBuffer buffer)


cdef class MarketDataBuffer:
    cdef md_block_buffer* header
    cdef bint owner
    cdef size_t iter_idx

    cdef void c_sort(self)

    cdef void c_put(self, md_variant* market_data)

    cdef md_variant* c_get(self, ssize_t idx)

    cdef void c_clear(self)


cdef class MarketDataRingBuffer:
    cdef md_ring_buffer* header
    cdef bint owner
    cdef size_t iter_idx

    cdef bint c_is_empty(self)

    cdef bint c_is_full(self, md_variant* market_data)

    cdef void c_put(self, md_variant* market_data, bint block=?, double timeout=?)

    cdef md_variant* c_get(self, ssize_t idx)

    cdef md_variant* c_listen(self, bint block=?, double timeout=?)


cdef class MarketDataConcurrentBuffer:
    cdef md_concurrent_buffer* header
    cdef bint owner
    cdef size_t iter_idx

    cdef bint c_is_worker_empty(self, size_t worker_id)

    cdef bint c_is_empty(self)

    cdef bint c_is_full(self)

    cdef void c_put(self, md_variant* market_data, bint block=?, double timeout=?)

    cdef md_variant* c_listen(self, size_t worker_id, bint block=?, double timeout=?)

    cdef void c_disable_worker(self, size_t worker_id)

    cdef void c_enable_worker(self, size_t worker_id)

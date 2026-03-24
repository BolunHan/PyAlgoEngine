from libcpp cimport bool as c_bool
from cpython.object cimport PyObject

from .c_allocator_protocol cimport allocator_protocol
from .c_market_data cimport md_variant
from ..c_intern_string cimport istr_map


cdef extern from "c_market_data_buffer.h":
    size_t MD_BUF_PTR_DEFAULT_CAP
    size_t MD_BUF_DATA_DEFAULT_CAP

    ctypedef struct md_ptr_array:
        size_t capacity
        size_t idx_head
        size_t idx_tail
        size_t* offsets

    ctypedef struct md_data_array:
        size_t capacity
        size_t occupied
        char* buf

    ctypedef struct md_block_buffer:
        md_ptr_array ptr_array
        md_data_array data_array
        double current_timestamp
        c_bool sorted

    ctypedef struct md_ring_buffer:
        md_ptr_array ptr_array
        md_data_array data_array

    ctypedef struct md_concurrent_buffer_worker_t:
        size_t ptr_head
        c_bool enabled

    ctypedef struct md_concurrent_buffer:
        md_concurrent_buffer_worker_t* workers
        size_t n_workers
        md_variant** buffer
        size_t capacity
        size_t tail

    int c_md_compare_serialized(const void* a, const void* b) noexcept nogil
    size_t c_md_total_buffer_size(md_variant** md_array, size_t n_md) noexcept nogil
    md_variant* c_md_send_to_shm(md_variant* market_data, allocator_protocol* shm_allocator, istr_map* shm_pool) noexcept nogil

    md_block_buffer* c_md_block_buffer_new(size_t ptr_capacity, size_t data_capacity, allocator_protocol* allocator) noexcept nogil
    void c_md_block_buffer_free(md_block_buffer* buffer) noexcept nogil
    int c_md_block_buffer_extend(md_block_buffer* buffer, size_t new_ptr_capacity, size_t new_data_capacity) noexcept nogil
    int c_md_block_buffer_put(md_block_buffer* buffer, md_variant* market_data) noexcept nogil
    int c_md_block_buffer_get(md_block_buffer* buffer, size_t idx, const char** out) noexcept nogil
    int c_md_block_buffer_sort(md_block_buffer* buffer) noexcept nogil
    int c_md_block_buffer_clear(md_block_buffer* buffer) noexcept nogil
    size_t c_md_block_buffer_serialized_size(md_block_buffer* buffer) noexcept nogil
    int c_md_block_buffer_serialize(md_block_buffer* buffer, char* out) noexcept nogil
    md_block_buffer* c_md_block_buffer_deserialize(const char* blob, allocator_protocol* allocator) noexcept nogil

    md_ring_buffer* c_md_ring_buffer_new(size_t ptr_capacity, size_t data_capacity, allocator_protocol* allocator) noexcept nogil
    void c_md_ring_buffer_free(md_ring_buffer* buffer) noexcept nogil
    int c_md_ring_buffer_is_full(md_ring_buffer* buffer, md_variant* market_data) noexcept nogil
    int c_md_ring_buffer_is_empty(md_ring_buffer* buffer) noexcept nogil
    size_t c_md_ring_buffer_size(md_ring_buffer* buffer) noexcept nogil
    int c_md_ring_buffer_put(md_ring_buffer* buffer, md_variant* market_data, c_bool block, double timeout) noexcept nogil
    int c_md_ring_buffer_get(md_ring_buffer* buffer, size_t index, const char** out) noexcept nogil
    int c_md_ring_buffer_listen(md_ring_buffer* buffer, c_bool block, double timeout, const char** out) noexcept nogil

    md_concurrent_buffer* c_md_concurrent_buffer_new(size_t n_workers, size_t capacity, allocator_protocol* shm_allocator) noexcept nogil
    void c_md_concurrent_buffer_free(md_concurrent_buffer* buffer) noexcept nogil
    int c_md_concurrent_buffer_enable_worker(md_concurrent_buffer* buffer, size_t worker_id) noexcept nogil
    int c_md_concurrent_buffer_disable_worker(md_concurrent_buffer* buffer, size_t worker_id) noexcept nogil
    int c_md_concurrent_buffer_is_full(md_concurrent_buffer* buffer) noexcept nogil
    int c_md_concurrent_buffer_is_empty(md_concurrent_buffer* buffer, size_t worker_id) noexcept nogil
    int c_md_concurrent_buffer_put(md_concurrent_buffer* buffer, md_variant* market_data, c_bool block, double timeout) noexcept nogil
    int c_md_concurrent_buffer_listen(md_concurrent_buffer* buffer, size_t worker_id, c_bool block, double timeout, md_variant** out) noexcept nogil


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
    cdef readonly object buf

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

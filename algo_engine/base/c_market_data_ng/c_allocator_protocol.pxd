from libcpp cimport bool as c_bool
from libc.stdint cimport uint8_t, uint64_t

from ..c_heap_allocator cimport heap_allocator, C_ALLOCATOR as HEAP_ALLOCATOR
from ..c_shm_allocator cimport shm_allocator, shm_allocator_ctx, C_ALLOCATOR as SHM_ALLOCATOR


cdef extern from "c_allocator_protocol.h":
    uint8_t MD_ALLOC_VIGILANT
    uint64_t MD_ALLOC_MAGIC

    ctypedef struct allocator_protocol:
        shm_allocator* shm_allocator
        shm_allocator_ctx* shm_allocator_ctx
        heap_allocator* heap_allocator
        c_bool with_lock
        c_bool with_shm
        c_bool with_freelist
        size_t size
        uint64_t magic
        char buf[]

    allocator_protocol* c_md_allocator_protocol_new(size_t size, shm_allocator_ctx* shm_allocator, heap_allocator* heap_allocator, c_bool with_lock) noexcept nogil
    void c_md_allocator_protocol_free(allocator_protocol* protocol) noexcept nogil

    allocator_protocol* c_md_protocol_from_ptr(const void* ptr) noexcept nogil
    void* c_md_alloc(size_t size, allocator_protocol* schematic) noexcept nogil
    void c_md_free(void* ptr) noexcept nogil
    char* c_md_strdup(const char* src, allocator_protocol* allocator) noexcept nogil
    void* c_md_realloc(void* src, size_t new_size, allocator_protocol* allocator) noexcept nogil


cdef bint MD_CFG_LOCKED
cdef bint MD_CFG_SHARED
cdef bint MD_CFG_FREELIST


cdef class EnvConfigContext:
    cdef dict overrides
    cdef dict originals

    cdef void c_activate(self)

    cdef void c_deactivate(self)


cdef EnvConfigContext MD_SHARED
cdef EnvConfigContext MD_LOCKED
cdef EnvConfigContext MD_FREELIST


cdef class AllocatorProtocol:
    cdef allocator_protocol* protocol
    cdef bint owner

    @staticmethod
    cdef AllocatorProtocol c_from_protocol(allocator_protocol* protocol, bint owner)


cdef allocator_protocol* MD_DEFAULT_ALLOCATOR
cdef allocator_protocol* MD_SHM_ALLOCATOR
cdef allocator_protocol* MD_HEAP_ALLOCATOR

from libc.stdint cimport uint64_t

from .c_shm_allocator cimport shm_allocator_ctx


cdef extern from "pthread.h":
    ctypedef struct pthread_mutex_t:
        pass


cdef extern from "c_intern_string.h":
    # === Constants ===
    size_t ISTR_INITIAL_CAPACITY

    # === Structs ===
    ctypedef struct istr_entry:
        char* internalized;
        uint64_t hash;
        istr_entry* next;

    ctypedef struct istr_map:
        shm_allocator_ctx* allocator
        size_t capacity
        size_t size
        istr_entry* first
        istr_entry* pool
        pthread_mutex_t* lock
        pthread_mutex_t  local_lock

    # === Functions ===
    uint64_t fnv1a_hash(const char* key, size_t key_length)
    istr_map* c_istr_map_new(size_t capacity, shm_allocator_ctx* allocator)
    void c_istr_map_free(istr_map* map)
    int c_istr_map_extend(istr_map* map, size_t new_capacity)
    const istr_entry* c_istr_map_lookup(istr_map* map, const char* key)
    const istr_entry* c_istr_map_lookup_synced(istr_map* map, const char* key)
    const char* c_istr(istr_map* map, const char* key)
    const char* c_istr_synced(istr_map* map, const char* key)


cdef class InternStringPool:
    cdef istr_map* pool
    cdef bint owner

    @staticmethod
    cdef InternStringPool c_from_header(const istr_map* header, bint owner=*)

    cdef const char* c_istr(self, const char* string, bint with_lock=*)

    cpdef InternString istr(self, str string, bint with_lock=*)


cdef istr_map* C_POOL
cdef InternStringPool POOL
cdef istr_map* C_INTRA_POOL
cdef InternStringPool INTRA_POOL


cdef class InternString:
    cdef const istr_map* pool
    cdef const char* internalized
    cdef uint64_t hash

    @staticmethod
    cdef InternString c_from_entry(const istr_entry* entry, istr_map* pool)
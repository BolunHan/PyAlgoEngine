from libc.stdint cimport uint64_t

from .c_allocator_protocol cimport allocator_protocol


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
        pthread_mutex_t lock
        istr_entry* pool
        size_t capacity
        size_t size
        istr_entry* first

    # === Functions ===
    uint64_t fnv1a_hash(const char* key, size_t key_length)
    istr_map* c_istr_map_new(size_t capacity, allocator_protocol* allocator)
    void c_istr_map_free(istr_map* imap)
    int c_istr_map_extend(istr_map* imap, size_t new_capacity)
    int c_istr_map_extend_synced(istr_map* imap, size_t new_capacity)
    const istr_entry* c_istr_map_lookup(const istr_map* imap, const char* key)
    const istr_entry* c_istr_map_lookup_synced(const istr_map* imap, const char* key)
    const char* c_istr(istr_map* imap, const char* key, const istr_entry** out_entry)
    const char* c_istr_synced(istr_map* imap, const char* key, const istr_entry** out_entry)


cdef class InternString:
    cdef readonly InternStringPool pool
    cdef const char* internalized
    cdef uint64_t hash

    @staticmethod
    cdef InternString c_from_entry(const istr_entry* entry, InternStringPool pool)


cdef class InternStringPool:
    cdef istr_map* pool
    cdef bint owner

    @staticmethod
    cdef InternStringPool c_from_header(const istr_map* header, bint owner=*)

    cpdef InternString istr(self, str string)


cdef istr_map* C_POOL
cdef InternStringPool POOL
cdef istr_map* C_INTRA_POOL
cdef InternStringPool INTRA_POOL

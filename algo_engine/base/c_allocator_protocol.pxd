from libcpp cimport bool as c_bool

from cbase.allocator_protocol cimport AllocatorConfigContext, allocator_protocol, heap_allocator, shm_allocator_ctx

cdef c_bool MD_CFG_LOCKED
cdef c_bool MD_CFG_SHARED
cdef c_bool MD_CFG_FREELIST


cdef class MDConfigContext(AllocatorConfigContext):
    pass


cdef heap_allocator* HEAP_ALLOCATOR
cdef shm_allocator_ctx* SHM_ALLOCATOR

cdef allocator_protocol* MD_DEFAULT_ALLOCATOR
cdef allocator_protocol* MD_SHM_ALLOCATOR
cdef allocator_protocol* MD_HEAP_ALLOCATOR

cdef MDConfigContext MD_SHARED
cdef MDConfigContext MD_LOCKED
cdef MDConfigContext MD_LOCKFREE
cdef MDConfigContext MD_FREELIST

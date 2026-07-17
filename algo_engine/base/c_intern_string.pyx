from libc.stdint cimport uintptr_t

from .c_allocator_protocol cimport MD_SHM_ALLOCATOR, MD_HEAP_ALLOCATOR

cdef istr_map* C_POOL = c_istr_map_new(0, MD_SHM_ALLOCATOR)
globals()['C_POOL'] = <uintptr_t> C_POOL
cdef InternStringPool POOL = InternStringPool.c_from_header(C_POOL, True)
globals()['POOL'] = POOL

cdef istr_map* C_INTRA_POOL = c_istr_map_new(0, MD_HEAP_ALLOCATOR)
globals()['C_INTRA_POOL'] = <uintptr_t> C_INTRA_POOL
cdef InternStringPool INTRA_POOL = InternStringPool.c_from_header(C_INTRA_POOL, True)
globals()['INTRA_POOL'] = INTRA_POOL

# Re-export cdef classes at Python level so `from ... import ...` works
globals()['InternString'] = InternString
globals()['InternStringPool'] = InternStringPool

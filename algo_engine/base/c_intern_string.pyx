from cpython.unicode cimport PyUnicode_AsUTF8, PyUnicode_FromString
from libc.stdint cimport uintptr_t

from .c_allocator_protocol cimport MD_DEFAULT_ALLOCATOR, MD_SHM_ALLOCATOR, MD_HEAP_ALLOCATOR


cdef class InternString:
    @staticmethod
    cdef InternString c_from_entry(const istr_entry* entry, InternStringPool parent_pool):
        cdef InternString instance = InternString.__new__(InternString)
        instance.pool = parent_pool
        instance.internalized = <const char*> entry.internalized
        instance.hash = entry.hash
        return instance

    def __gt__(self, object other):
        if isinstance(other, InternString):
            return self.string.__gt__(other.string)
        return self.string.__gt__(other)

    def __eq__(self, object other):
        if isinstance(other, InternString):
            return self.string.__eq__(other.string)
        return self.string.__eq__(other)

    def __hash__(self):
        return self.hash_value

    def __repr__(self):
        if self.internalized:
            return f'<{self.__class__.__name__}>({PyUnicode_FromString(self.internalized)})>'
        return f'<{self.__class__.__name__}>(uninitialized)>'

    property string:
        def __get__(self):
            if self.internalized:
                return PyUnicode_FromString(self.internalized)
            raise RuntimeError('Not initialize')

    property hash_value:
        def __get__(self):
            if self.hash:
                return self.hash

    property address:
        def __get__(self):
            if self.internalized:
                return f'{<uintptr_t> self.internalized:#0x}'
            return None


cdef class InternStringPool:
    def __init__(self):
        self.pool = c_istr_map_new(0, MD_DEFAULT_ALLOCATOR)
        self.owner = True

    def __dealloc__(self):
        if not self.owner:
            return

        if self.pool:
            c_istr_map_free(self.pool)

    @staticmethod
    cdef InternStringPool c_from_header(const istr_map* header, bint owner=False):
        cdef InternStringPool instance = InternStringPool.__new__(InternStringPool)
        instance.pool = <istr_map*> header
        instance.owner = owner
        return instance

    def __len__(self):
        if not self.pool:
            raise RuntimeError(f'<{self.__class__.__name__}> Not properly initialized! Missing istr_map header!')
        return self.pool.size

    def __getitem__(self, str key):
        cdef const char* utf8_string = PyUnicode_AsUTF8(key)
        cdef const istr_entry* entry = c_istr_map_lookup_synced(self.pool, utf8_string)
        if not entry:
            raise KeyError(f'{key} not interned')
        return InternString.c_from_entry(entry, self)

    cpdef InternString istr(self, str string):
        cdef const char* utf8_string = PyUnicode_AsUTF8(string)
        cdef const istr_entry* out_entry = NULL
        cdef const char* istr = c_istr_synced(self.pool, utf8_string, &out_entry)
        if not istr:
            raise MemoryError('Failed to intern string.')
        return InternString.c_from_entry(out_entry, self)

    def internalized(self):
        cdef istr_entry* entry = self.pool.first
        while entry:
            yield InternString.c_from_entry(entry, self)
            entry = entry.next

    property size:
        def __get__(self):
            if self.pool:
                return self.pool.size
            raise RuntimeError('Not initialize')

    property address:
        def __get__(self):
            if self.pool:
                return f'{<uintptr_t> self.pool:#0x}'
            return None


cdef istr_map* C_POOL = c_istr_map_new(0, MD_SHM_ALLOCATOR)
globals()['C_POOL'] = <uintptr_t> C_POOL
cdef InternStringPool POOL = InternStringPool.c_from_header(C_POOL, True)
globals()['POOL'] = POOL

cdef istr_map* C_INTRA_POOL = c_istr_map_new(0, MD_HEAP_ALLOCATOR)
globals()['C_INTRA_POOL'] = <uintptr_t> C_INTRA_POOL
cdef InternStringPool INTRA_POOL = InternStringPool.c_from_header(C_INTRA_POOL, True)
globals()['INTRA_POOL'] = INTRA_POOL
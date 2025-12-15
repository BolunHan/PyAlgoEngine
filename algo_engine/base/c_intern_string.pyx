from cpython.unicode cimport PyUnicode_AsUTF8, PyUnicode_FromString
from libc.stdint cimport uintptr_t

from .c_shm_allocator cimport C_ALLOCATOR


cdef class InternStringPool:
    def __cinit__(self, bint alloc=True):
        if not alloc:
            return

        if POOL is None:
            global POOL
            self.pool = c_istr_map_new(0, C_ALLOCATOR)
            self.owner = True
            POOL = self
            return

        raise RuntimeError('Must not initialize')

    def __dealloc__(self):
        if not self.owner:
            return

        if self.pool:
            c_istr_map_free(self.pool)

    @staticmethod
    cdef InternStringPool c_from_header(const istr_map* header, bint owner=False):
        cdef InternStringPool instance = InternStringPool.__new__(InternStringPool, False)
        instance.pool = <istr_map*> header
        instance.owner = owner
        return instance

    cdef const char* c_istr(self, const char* string, bint with_lock=False):
        cdef const char* istr

        if with_lock:
            istr = c_istr_synced(self.pool, string)
        else:
            istr = c_istr(self.pool, string)

        if not istr:
            raise MemoryError('Failed to intern string.')
        return istr

    def __len__(self):
        if self.pool:
            return self.pool.size
        return 0

    def __getitem__(self, str key):
        cdef const char* utf8_string = PyUnicode_AsUTF8(key)
        cdef const istr_entry* entry = c_istr_map_lookup_synced(self.pool, utf8_string)
        if not entry:
            raise KeyError(f'{key} not interned')
        return InternString.c_from_entry(entry, self.pool)

    cpdef InternString istr(self, str string, bint with_lock=False):
        cdef const char* utf8_string = PyUnicode_AsUTF8(string)
        cdef const char* istr

        if with_lock:
            istr = c_istr_synced(self.pool, utf8_string)
        else:
            istr = c_istr(self.pool, utf8_string)

        if not istr:
            raise MemoryError('Failed to intern string.')
        return InternString.c_from_entry(c_istr_map_lookup_synced(self.pool, utf8_string), self.pool)

    def internalized(self):
        cdef istr_entry* entry = self.pool.first
        while entry:
            yield InternString.c_from_entry(entry, self.pool)
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


cdef istr_map* C_POOL = c_istr_map_new(0, C_ALLOCATOR)
globals()['C_POOL'] = <uintptr_t> C_POOL
cdef InternStringPool POOL = InternStringPool.c_from_header(C_POOL, True)
globals()['POOL'] = POOL

cdef istr_map* C_INTRA_POOL = c_istr_map_new(0, NULL)
globals()['C_INTRA_POOL'] = <uintptr_t> C_INTRA_POOL
cdef InternStringPool INTRA_POOL = InternStringPool.c_from_header(C_INTRA_POOL, True)
globals()['INTRA_POOL'] = INTRA_POOL


cdef class InternString:
    @staticmethod
    cdef InternString c_from_entry(const istr_entry* entry, istr_map* pool):
        cdef InternString instance = InternString.__new__(InternString)
        instance.pool = pool
        instance.internalized = entry.internalized
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

    property intern_pool:
        def __get__(self):
            if self.pool:
                return InternStringPool.c_from_header(self.pool, False)
            raise RuntimeError('Not initialize')

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

from libc.errno cimport errno
from libc.stdint cimport uintptr_t


cdef class HeapMemoryPage:
    def __cinit__(self, uintptr_t page_addr=0):
        self.page = <heap_page*> page_addr if page_addr else NULL

    @staticmethod
    cdef inline HeapMemoryPage c_from_header(heap_page* header):
        cdef HeapMemoryPage instance = HeapMemoryPage.__new__(HeapMemoryPage, 0)
        instance.page = header
        return instance

    def __repr__(self):
        if self.page:
            return f"<{self.__class__.__name__}>(capacity={self.capacity:,}, occupied={self.occupied:,})"
        return f"<{self.__class__.__name__}>(uninitialized)"

    @classmethod
    def from_buffer(cls, uintptr_t buffer_addr) -> HeapMemoryPage:
        cdef HeapMemoryPage instance = cls.__new__(cls, 0)
        instance.page = <heap_page*> (<char*> buffer_addr - sizeof(heap_page))
        return instance

    def allocated(self):
        if not self.page:
            return
        cdef heap_memory_block* block = self.page.allocated
        while block:
            yield HeapMemoryBlock.c_from_header(block, False)
            block = block.next_allocated

    cpdef void reclaim(self):
        if not self.page:
            raise RuntimeError(f"Uninitialized <{self.__class__.__name__}>")
        c_heap_page_reclaim(self.page.allocator, self.page)

    property capacity:
        def __get__(self):
            if self.page:
                return <size_t> self.page.capacity
            return 0

    property occupied:
        def __get__(self):
            if self.page:
                return <size_t> self.page.occupied
            return 0

    property allocator:
        def __get__(self):
            if not self.page:
                return None
            return HeapAllocator.c_from_header(self.page.allocator, False)


cdef class HeapMemoryBlock:
    def __cinit__(self, uintptr_t block=0, bint owner=False):
        self.block = <heap_memory_block*> block if block else NULL
        self.owner = owner

    def __dealloc__(self):
        if not self.owner:
            return

        if self.block:
            c_heap_free(self.block.buffer, &self.block.parent_page.allocator.lock)

    @staticmethod
    cdef inline HeapMemoryBlock c_from_header(heap_memory_block* header, bint owner=False):
        cdef HeapMemoryBlock instance = HeapMemoryBlock.__new__(HeapMemoryBlock, 0, owner)
        instance.block = header
        return instance

    def __repr__(self):
        if self.block:
            return f"<{self.__class__.__name__}>(size={self.block.size}, capacity={self.block.capacity})"
        return f"<{self.__class__.__name__}>(uninitialized)"

    property size:
        def __get__(self):
            if not self.block:
                return -1
            return <size_t> self.block.size

    property capacity:
        def __get__(self):
            if not self.block:
                return -1
            return <size_t> self.block.capacity

    property next_free:
        def __get__(self):
            if self.block and self.block.next_free:
                return HeapMemoryBlock.c_from_header(self.block.next_free, False)
            return None

    property next_allocated:
        def __get__(self):
            if self.block and self.block.next_allocated:
                return HeapMemoryBlock.c_from_header(self.block.next_allocated, False)
            return None

    property parent_page:
        def __get__(self):
            if not self.block:
                return None
            return HeapMemoryPage.c_from_header(self.block.parent_page)

    property buffer:
        def __get__(self):
            if not self.block:
                return None
            cdef size_t size = self.block.size
            return <char[:size]> self.block.buffer

    property address:
        def __get__(self):
            if not self.block:
                return None
            return f"{<uintptr_t> self.block.buffer:#0x}"


cdef class HeapAllocator:
    def __cinit__(self, bint owner=True):
        self.owner = owner
        if not owner:
            return

        global ALLOCATOR
        if ALLOCATOR is not None:
            raise RuntimeError(f'Global allocator already initialized, Use {ALLOCATOR} instead!')

        ALLOCATOR = self
        self.allocator = c_heap_allocator_new()
        if not self.allocator:
            raise OSError(errno, "Initialize heap allocator failed")

    def __dealloc__(self):
        if not self.owner:
            return

        if self.allocator:
            c_heap_allocator_free(self.allocator)

    @staticmethod
    cdef HeapAllocator c_from_header(heap_allocator* header, bint owner=False):
        cdef HeapAllocator instance = HeapAllocator.__new__(HeapAllocator, False)
        instance.allocator = header
        instance.owner = owner
        return instance

    cdef inline heap_page* c_extend(self, size_t capacity=0, pthread_mutex_t* lock=NULL):
        if not self.allocator:
            raise RuntimeError(f'Uninitialized <{self.__class__.__name__}>')
        return c_heap_allocator_extend(self.allocator, capacity, lock)

    cdef inline void* c_calloc(self, size_t size, pthread_mutex_t* lock=NULL):
        if not self.allocator:
            raise RuntimeError(f'Uninitialized <{self.__class__.__name__}>')
        return c_heap_calloc(self.allocator, size, lock)

    cdef inline void* c_request(self, size_t size, int scan_all_pages=1, pthread_mutex_t* lock=NULL):
        if not self.allocator:
            raise RuntimeError(f'Uninitialized <{self.__class__.__name__}>')
        return c_heap_request(self.allocator, size, scan_all_pages, lock)

    cdef inline void c_free(self, void* ptr, pthread_mutex_t* lock=NULL):
        if not self.allocator:
            raise RuntimeError(f'Uninitialized <{self.__class__.__name__}>')
        c_heap_free(ptr, lock)

    cpdef HeapMemoryPage extend(self, size_t capacity=0, bint with_lock=True):
        if not self.allocator:
            raise RuntimeError(f'Uninitialized <{self.__class__.__name__}>')
        cdef pthread_mutex_t* lock = &self.allocator.lock if with_lock else NULL
        cdef heap_page* page = c_heap_allocator_extend(self.allocator, capacity, lock)
        if not page:
            raise OSError(errno, f'<{self.__class__.__name__}> failed to extend new page')
        return HeapMemoryPage.c_from_header(page)

    cpdef HeapMemoryBlock calloc(self, size_t size, bint with_lock=True):
        if not self.allocator:
            raise RuntimeError(f"Uninitialized <{self.__class__.__name__}>")
        cdef pthread_mutex_t* lock = &self.allocator.lock if with_lock else NULL
        cdef void* p = c_heap_calloc(self.allocator, size, lock)
        if not p:
            raise OSError(errno, f"<{self.__class__.__name__}> failed to calloc new buffer")
        cdef heap_memory_block* block = <heap_memory_block*> (<char*> p - sizeof(heap_memory_block))
        return HeapMemoryBlock.c_from_header(block, True)

    cpdef HeapMemoryBlock request(self, size_t size, bint with_lock=True, bint scan_all_pages=True):
        if not self.allocator:
            raise RuntimeError(f'Uninitialized <{self.__class__.__name__}>')
        cdef pthread_mutex_t* lock = &self.allocator.lock if with_lock else NULL
        cdef void* p = c_heap_request(self.allocator, size, <int> scan_all_pages, lock)
        if not p:
            raise OSError(errno, f'<{self.__class__.__name__}> failed to request new buffer')
        cdef heap_memory_block* block = <heap_memory_block*> (<char*> p - sizeof(heap_memory_block))
        return HeapMemoryBlock.c_from_header(block, True)

    cpdef void free(self, HeapMemoryBlock buffer, bint with_lock=True):
        if not self.allocator:
            raise RuntimeError(f'Uninitialized <{self.__class__.__name__}>')
        if not buffer or not buffer.block:
            return
        cdef pthread_mutex_t* lock = &buffer.block.parent_page.allocator.lock if with_lock else NULL
        c_heap_free(<void*> buffer.block.buffer, lock)
        buffer.owner = False
        buffer.block = NULL

    cpdef void reclaim(self, bint with_lock=True):
        if not self.allocator:
            raise RuntimeError(f'Uninitialized <{self.__class__.__name__}>')
        cdef pthread_mutex_t* lock = &self.allocator.lock if with_lock else NULL
        c_heap_reclaim(self.allocator, lock)

    def pages(self):
        if not self.allocator:
            raise RuntimeError(f'Uninitialized <{self.__class__.__name__}>')
        cdef heap_page* page = self.allocator.active_page
        while page:
            yield HeapMemoryPage.c_from_header(page)
            page = page.prev

    def allocated(self):
        if not self.allocator:
            raise RuntimeError(f'Uninitialized <{self.__class__.__name__}>')
        cdef heap_page* page = self.allocator.active_page
        cdef heap_memory_block* block
        while page:
            block = page.allocated
            while block:
                yield HeapMemoryBlock.c_from_header(block, False)
                block = block.next_allocated
            page = page.prev

    def free_list(self):
        if not self.allocator:
            raise RuntimeError(f'Uninitialized <{self.__class__.__name__}>')
        cdef heap_memory_block* block = self.allocator.free_list
        while block:
            yield HeapMemoryBlock.c_from_header(block, False)
            block = block.next_free

    property mapped_pages:
        def __get__(self):
            if not self.allocator:
                return 0
            return <size_t> self.allocator.mapped_pages

    property active_page:
        def __get__(self):
            if not self.allocator or not self.allocator.active_page:
                return None
            return HeapMemoryPage.c_from_header(self.allocator.active_page)


cdef HeapAllocator ALLOCATOR = HeapAllocator(True)
cdef heap_allocator* C_ALLOCATOR = ALLOCATOR.allocator

globals()['ALLOCATOR'] = ALLOCATOR
globals()['DEFAULT_AUTOPAGE_CAPACITY'] = DEFAULT_AUTOPAGE_CAPACITY
globals()['MAX_AUTOPAGE_CAPACITY'] = MAX_AUTOPAGE_CAPACITY
globals()['DEFAULT_AUTOPAGE_ALIGNMENT'] = DEFAULT_AUTOPAGE_ALIGNMENT

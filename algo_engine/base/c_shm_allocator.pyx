import os

from cpython.unicode cimport PyUnicode_FromString
from libc.errno cimport errno


cdef class SharedMemoryPage:
    def __cinit__(self, uintptr_t page_addr=0):
        self.ctx = <shm_page_ctx*> page_addr if page_addr else NULL

    @staticmethod
    cdef inline SharedMemoryPage c_from_header(shm_page_ctx* header):
        cdef SharedMemoryPage instance = SharedMemoryPage.__new__(SharedMemoryPage, 0)
        instance.ctx = header
        return instance

    def __repr__(self):
        if self.ctx and self.ctx.shm_page:
            return f"<{self.__class__.__name__}>(name={self.name}, capacity={self.capacity:,}, occupied={self.occupied:,})>"
        return f"<{self.__class__.__name__}>(uninitialized)"

    @classmethod
    def from_buffer(cls, uintptr_t buffer_addr) -> SharedMemoryPage:
        cdef SharedMemoryPage instance = cls.__new__(cls, 0)
        instance.ctx = <shm_page_ctx*> (<char*> buffer_addr - sizeof(shm_page_ctx))
        return instance

    def allocated(self):
        if not self.ctx or not self.ctx.shm_page:
            return

        cdef shm_memory_block* block = self.ctx.shm_page.allocated
        while block:
            yield SharedMemoryBlock.c_from_header(block, False)
            block = block.next_allocated

    cpdef void reclaim(self):
        if not self.ctx:
            raise RuntimeError(f"Uninitialized <{self.__class__.__name__}>")
        c_shm_page_reclaim(self.ctx.shm_page.allocator, self.ctx)

    property name:
        def __get__(self) -> str:
            if not self.ctx or not self.ctx.shm_page:
                return None
            return PyUnicode_FromString(self.ctx.shm_page.shm_name)

    property capacity:
        def __get__(self) -> size_t:
            if self.ctx and self.ctx.shm_page:
                return <size_t> self.ctx.shm_page.capacity
            return 0

    property occupied:
        def __get__(self) -> size_t:
            if self.ctx and self.ctx.shm_page:
                return <size_t> self.ctx.shm_page.occupied
            return 0


cdef class SharedMemoryBlock:
    def __cinit__(self, uintptr_t block=0, bint owner=False):
        self.block = <shm_memory_block*> block if block else NULL
        self.owner = owner

    def __dealloc__(self):
        if not self.owner:
            return

        if self.block:
            c_shm_free(self.block.buffer, &self.block.parent_page.allocator.lock)

    @staticmethod
    cdef inline SharedMemoryBlock c_from_header(shm_memory_block* header, bint owner=False):
        cdef SharedMemoryBlock instance = SharedMemoryBlock.__new__(SharedMemoryBlock, 0, owner)
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
                return SharedMemoryBlock.c_from_header(self.block.next_free, False)
            return None

    property next_allocated:
        def __get__(self):
            if self.block and self.block.next_allocated:
                return SharedMemoryBlock.c_from_header(self.block.next_allocated, False)
            return None

    property buffer:
        def __get__(self):
            if not self.block:
                return None
            cdef size_t size = self.block.size
            if size:
                return <char[:size]> self.block.buffer
            return None

    property address:
        def __get__(self):
            if self.block:
                return f'{<uintptr_t> self.block.buffer:#0x}'
            return None


cdef class SharedMemoryAllocator:
    def __cinit__(self, size_t region_size=0, bint owner=True):
        if not region_size:
            return

        global ALLOCATOR
        if ALLOCATOR is not None:
            raise RuntimeError(f'Global allocator already initialized, Use {ALLOCATOR} instead!')
        ALLOCATOR = self
        self.ctx = c_shm_allocator_new(region_size)
        if not self.ctx:
            raise OSError(errno, "Initialize SHM allocator failed")
        self.owner = owner

    def __dealloc__(self):
        if not self.owner:
            return

        if self.ctx:
            c_shm_allocator_free(self.ctx)

    @staticmethod
    cdef SharedMemoryAllocator c_from_header(shm_allocator_ctx* header, bint owner=False):
        cdef SharedMemoryAllocator instance = SharedMemoryAllocator.__new__(SharedMemoryAllocator, 0, False)
        instance.ctx = header
        instance.owner = owner
        return instance

    cdef inline shm_page_ctx* c_extend(self, size_t capacity=0, pthread_mutex_t* lock=NULL):
        if not self.ctx:
            raise RuntimeError(f'Uninitialized <{self.__class__.__name__}>')
        return c_shm_allocator_extend(self.ctx, capacity, lock)

    cdef inline void* c_calloc(self, size_t size, pthread_mutex_t* lock=NULL):
        if not self.ctx:
            raise RuntimeError(f'Uninitialized <{self.__class__.__name__}>')
        return c_shm_calloc(self.ctx, size, lock)

    cdef inline void* c_request(self, size_t size, pthread_mutex_t* lock=NULL):
        if not self.ctx:
            raise RuntimeError(f'Uninitialized <{self.__class__.__name__}>')
        return c_shm_request(self.ctx, size, 1, lock)

    cdef inline void c_free(self, void* ptr, pthread_mutex_t* lock=NULL):
        if not self.ctx:
            raise RuntimeError(f'Uninitialized <{self.__class__.__name__}>')
        c_shm_free(ptr, lock)

    def __repr__(self):
        if self.ctx and self.ctx.shm_allocator:
            return f"<{self.__class__.__name__}>(name={self.name}, pid={self.pid}, mapped_addr={<uintptr_t> self.ctx.shm_allocator.region}>"
        return f"<{self.__class__.__name__}>(uninitialized)"

    @classmethod
    def get_pid(cls, str shm_name):
        cdef bytes shm_name_bytes = shm_name.encode('utf-8')
        return c_shm_pid(<char*> shm_name_bytes)

    cpdef SharedMemoryPage extend(self, size_t capacity=0, bint with_lock=True):
        if not self.ctx:
            raise RuntimeError(f'Uninitialized <{self.__class__.__name__}>')
        cdef pthread_mutex_t* lock = &self.ctx.shm_allocator.lock if with_lock else NULL
        cdef shm_page_ctx* page = c_shm_allocator_extend(self.ctx, capacity, lock)
        if not page:
            raise OSError(errno, f'<{self.__class__.__name__}> failed to extend new page')
        return SharedMemoryPage.c_from_header(page)

    cpdef SharedMemoryBlock calloc(self, size_t size, bint with_lock=True):
        if not self.ctx:
            raise RuntimeError(f'Uninitialized <{self.__class__.__name__}>')
        cdef pthread_mutex_t* lock = &self.ctx.shm_allocator.lock if with_lock else NULL
        cdef void* p = c_shm_calloc(self.ctx, size, lock)
        if not p:
            raise OSError(errno, f'<{self.__class__.__name__}> failed to calloc new buffer')

        cdef shm_memory_block* block = <shm_memory_block*> (<char*> p - sizeof(shm_memory_block))
        return SharedMemoryBlock.c_from_header(block, True)

    cpdef SharedMemoryBlock request(self, size_t size, bint scan_all_pages=True, bint with_lock=True):
        if not self.ctx:
            raise RuntimeError(f'Uninitialized <{self.__class__.__name__}>')
        cdef pthread_mutex_t* lock = &self.ctx.shm_allocator.lock if with_lock else NULL
        cdef void* p = c_shm_request(self.ctx, size, scan_all_pages, lock)
        if not p:
            raise OSError(errno, f'<{self.__class__.__name__}> failed to request new buffer')

        cdef shm_memory_block* block = <shm_memory_block*> (<char*> p - sizeof(shm_memory_block))
        return SharedMemoryBlock.c_from_header(block, True)

    cpdef void free(self, SharedMemoryBlock buffer, bint with_lock=True):
        if not self.ctx:
            raise RuntimeError(f'Uninitialized <{self.__class__.__name__}>')
        cdef pthread_mutex_t* lock = &self.ctx.shm_allocator.lock if with_lock else NULL
        c_shm_free(<void*> buffer.block.buffer, lock)
        buffer.owner = False
        buffer.block = NULL

    cpdef void reclaim(self, bint with_lock=True):
        if not self.ctx:
            raise RuntimeError(f'Uninitialized <{self.__class__.__name__}>')
        cdef pthread_mutex_t* lock = &self.ctx.shm_allocator.lock if with_lock else NULL
        c_shm_reclaim(self.ctx, lock)

    cpdef list dangling(self):
        cdef list out = []
        cdef list entries = os.listdir('/dev/shm')
        cdef str prefix = PyUnicode_FromString(SHM_ALLOCATOR_PREFIX)

        if prefix.startswith('/'):
            prefix = prefix[1:]

        cdef str shm_name, candidate, pid_str
        cdef pid_t p
        for shm_name in entries:
            if not shm_name.startswith(prefix):
                continue
            candidate = '/' + shm_name
            pid_str = shm_name.removeprefix(prefix).split('_')[1]
            p = <pid_t> int(pid_str)
            try:
                os.kill(p, 0)
            except ProcessLookupError:
                out.append(candidate)
            except PermissionError:
                continue
            except OSError:
                continue
        return out

    cpdef list dangling_pages(self):
        cdef list out = []
        cdef list entries = os.listdir('/dev/shm')
        cdef str prefix = PyUnicode_FromString(SHM_PAGE_PREFIX)

        if prefix.startswith('/'):
            prefix = prefix[1:]

        cdef str shm_name, candidate, pid_str
        cdef pid_t p
        for shm_name in entries:
            if not shm_name.startswith(prefix):
                continue
            candidate = '/' + shm_name
            pid_str = shm_name.removeprefix(prefix).split('_')[1]
            p = <pid_t> int(pid_str)
            try:
                os.kill(p, 0)
            except ProcessLookupError:
                out.append(candidate)
            except PermissionError:
                continue
            except OSError:
                continue
        return out

    cpdef void cleanup_dangling(self):
        c_shm_clear_dangling()

    def pages(self):
        if not self.ctx:
            raise RuntimeError(f'Uninitialized <{self.__class__.__name__}>')
        cdef shm_page_ctx* page = self.ctx.active_page
        while page:
            yield SharedMemoryPage.c_from_header(page)
            page = page.prev

    def allocated(self):
        if not self.ctx:
            raise RuntimeError(f'Uninitialized <{self.__class__.__name__}>')
        cdef shm_page_ctx* page = self.ctx.active_page
        cdef shm_memory_block* block
        while page:
            block = page.shm_page.allocated
            while block:
                yield SharedMemoryBlock.c_from_header(block, False)
                block = block.next_allocated
            page = page.prev

    def free_list(self):
        if not self.ctx:
            raise RuntimeError(f'Uninitialized <{self.__class__.__name__}>')
        cdef shm_memory_block* block = self.ctx.shm_allocator.free_list
        while block:
            yield SharedMemoryBlock.c_from_header(block, False)
            block = block.next_free

    property name:
        def __get__(self) -> str:
            if not self.ctx or not self.ctx.shm_allocator:
                return None
            return PyUnicode_FromString(self.ctx.shm_allocator.shm_name)

    property pid:
        def __get__(self):
            if not self.ctx or not self.ctx.shm_allocator:
                return -1
            return <pid_t> self.ctx.shm_allocator.pid

    property region:
        def __get__(self):
            if not self.ctx or not self.ctx.shm_allocator:
                return -1
            return <uintptr_t> self.ctx.shm_allocator.region

    property region_addr:
        def __get__(self):
            if not self.ctx or not self.ctx.shm_allocator:
                return None
            return f'{<uintptr_t> self.ctx.shm_allocator.region:#0x}'

    property region_size:
        def __get__(self):
            if not self.ctx or not self.ctx.shm_allocator:
                return -1
            return <size_t> self.ctx.shm_allocator.region_size

    property mapped_size:
        def __get__(self):
            if not self.ctx or not self.ctx.shm_allocator:
                return -1
            return <size_t> self.ctx.shm_allocator.mapped_size

    property mapped_pages:
        def __get__(self):
            if not self.ctx or not self.ctx.shm_allocator:
                return -1
            return <size_t> self.ctx.shm_allocator.mapped_pages

    property active_page:
        def __get__(self):
            if not self.ctx:
                return None
            return SharedMemoryPage.c_from_header(self.ctx.active_page)


cdef SharedMemoryAllocator ALLOCATOR = SharedMemoryAllocator(SHM_ALLOCATOR_DEFAULT_REGION_SIZE, True)
cdef shm_allocator_ctx* C_ALLOCATOR = ALLOCATOR.ctx


globals()['ALLOCATOR'] = ALLOCATOR
globals()['DEFAULT_AUTOPAGE_CAPACITY'] = DEFAULT_AUTOPAGE_CAPACITY
globals()['MAX_AUTOPAGE_CAPACITY'] = MAX_AUTOPAGE_CAPACITY
globals()['DEFAULT_AUTOPAGE_ALIGNMENT'] = DEFAULT_AUTOPAGE_ALIGNMENT
globals()['SHM_ALLOCATOR_PREFIX'] = PyUnicode_FromString(SHM_ALLOCATOR_PREFIX)
globals()['SHM_PAGE_PREFIX'] = PyUnicode_FromString(SHM_PAGE_PREFIX)
globals()['SHM_NAME_LEN'] = SHM_NAME_LEN
globals()['SHM_ALLOCATOR_DEFAULT_REGION_SIZE'] = SHM_ALLOCATOR_DEFAULT_REGION_SIZE


def cleanup():
    global ALLOCATOR, C_ALLOCATOR
    globals()['ALLOCATOR'] = ALLOCATOR = None
    C_ALLOCATOR = NULL
    c_shm_clear_dangling()


import atexit
atexit.register(cleanup)



cdef extern from "pthread.h":
    ctypedef struct pthread_mutex_t:
        pass

    int pthread_mutex_init(pthread_mutex_t* mutex, void* attr)
    int pthread_mutex_lock(pthread_mutex_t* mutex)
    int pthread_mutex_unlock(pthread_mutex_t* mutex)
    int pthread_mutex_destroy(pthread_mutex_t* mutex)


cdef extern from "c_heap_allocator.h":
    const size_t DEFAULT_AUTOPAGE_CAPACITY
    const size_t MAX_AUTOPAGE_CAPACITY
    const size_t DEFAULT_AUTOPAGE_ALIGNMENT

    ctypedef struct heap_memory_block:
        size_t capacity
        size_t size
        heap_memory_block* next_free
        heap_memory_block* next_allocated
        heap_page* parent_page
        char buffer[]

    ctypedef struct heap_page:
        size_t capacity
        size_t occupied
        heap_allocator* allocator
        heap_page* prev
        heap_memory_block* allocated
        char buffer[]

    ctypedef struct heap_allocator:
        pthread_mutex_t lock
        size_t mapped_pages
        heap_memory_block* free_list
        heap_page* active_page

    size_t c_page_roundup(size_t size)
    size_t c_block_roundup(size_t size)
    void c_heap_page_reclaim(heap_allocator* allocator, heap_page* page)

    heap_page* c_heap_allocator_extend(heap_allocator* allocator, size_t capacity, pthread_mutex_t* lock)
    heap_allocator* c_heap_allocator_new()
    void c_heap_allocator_free(heap_allocator* allocator)
    void* c_heap_calloc(heap_allocator* allocator, size_t size, pthread_mutex_t* lock)
    void* c_heap_request(heap_allocator* allocator, size_t size, int scan_all_pages, pthread_mutex_t* lock)
    void c_heap_free(void* ptr, pthread_mutex_t* lock)
    void c_heap_reclaim(heap_allocator* allocator, pthread_mutex_t* lock)


cdef class HeapMemoryPage:
    cdef heap_page* page

    @staticmethod
    cdef inline HeapMemoryPage c_from_header(heap_page* header)

    cpdef void reclaim(self)


cdef class HeapMemoryBlock:
    cdef heap_memory_block* block
    cdef readonly bint owner

    @staticmethod
    cdef inline HeapMemoryBlock c_from_header(heap_memory_block* header, bint owner=*)


cdef class HeapAllocator:
    cdef heap_allocator* allocator
    cdef readonly bint owner

    @staticmethod
    cdef HeapAllocator c_from_header(heap_allocator* header, bint owner=*)

    cdef inline heap_page* c_extend(self, size_t capacity=*, pthread_mutex_t* lock=*)

    cdef inline void* c_calloc(self, size_t size, pthread_mutex_t* lock=*)

    cdef inline void* c_request(self, size_t size, int scan_all_pages=*, pthread_mutex_t* lock=*)

    cdef inline void c_free(self, void* ptr, pthread_mutex_t* lock=*)

    cpdef HeapMemoryPage extend(self, size_t capacity=*, bint with_lock=*)

    cpdef HeapMemoryBlock calloc(self, size_t size, bint with_lock=*)

    cpdef HeapMemoryBlock request(self, size_t size, bint with_lock=*, bint scan_all_pages=*)

    cpdef void free(self, HeapMemoryBlock buffer, bint with_lock=*)

    cpdef void reclaim(self, bint with_lock=*)


cdef HeapAllocator ALLOCATOR
cdef heap_allocator* C_ALLOCATOR

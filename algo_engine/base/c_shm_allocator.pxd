from libc.stdint cimport uintptr_t
from posix.unistd cimport pid_t


cdef extern from "pthread.h":
    ctypedef struct pthread_mutex_t:
        pass  # Opaque type, details handled by C

    int pthread_mutex_init(pthread_mutex_t* mutex, void* attr)
    int pthread_mutex_lock(pthread_mutex_t* mutex)
    int pthread_mutex_unlock(pthread_mutex_t* mutex)
    int pthread_mutex_destroy(pthread_mutex_t* mutex)


cdef extern from "c_shm_allocator.h":
    const size_t DEFAULT_AUTOPAGE_CAPACITY
    const size_t MAX_AUTOPAGE_CAPACITY
    const size_t DEFAULT_AUTOPAGE_ALIGNMENT
    const char* SHM_ALLOCATOR_PREFIX
    const char* SHM_PAGE_PREFIX
    const size_t SHM_NAME_LEN
    const size_t SHM_ALLOCATOR_DEFAULT_REGION_SIZE
    const size_t c_shm_page_overhead
    const size_t c_shm_block_overhead

    ctypedef struct shm_page:
        size_t capacity
        size_t occupied
        size_t offset
        shm_allocator* allocator;
        shm_memory_block* allocated
        char shm_name[SHM_NAME_LEN]
        char prev_name[SHM_NAME_LEN]

    ctypedef struct shm_page_ctx:
        shm_page* shm_page
        int shm_fd
        char* buffer
        shm_page_ctx* prev

    ctypedef struct shm_memory_block:
        size_t capacity
        size_t size
        shm_memory_block* next_free
        shm_memory_block* next_allocated
        shm_page* parent_page
        char buffer[]

    ctypedef struct shm_allocator:
        char shm_name[SHM_NAME_LEN]
        size_t pid
        pthread_mutex_t lock
        uintptr_t region
        size_t region_size
        size_t mapped_size
        char active_page[SHM_NAME_LEN]
        size_t mapped_pages
        shm_memory_block* free_list

    ctypedef struct shm_allocator_ctx:
        shm_allocator* shm_allocator
        int shm_fd
        shm_page_ctx* active_page

    size_t c_page_roundup(size_t size)
    size_t c_block_roundup(size_t size)
    void c_shm_allocator_name(const void* region, char* out)
    void c_shm_page_name(shm_allocator* allocator, char* out)
    int c_shm_scan(const char* prefix, char* out)
    shm_page_ctx* c_shm_page_new(size_t page_capacity)
    int c_shm_page_map(shm_allocator* allocator, shm_page_ctx* page_ctx)
    void c_shm_page_reclaim(shm_allocator* allocator, shm_page_ctx* page_ctx)

    shm_page_ctx* c_shm_allocator_extend(shm_allocator_ctx* ctx, size_t capacity, pthread_mutex_t* lock)
    shm_allocator_ctx* c_shm_allocator_new(size_t region_size)
    void c_shm_allocator_free(shm_allocator_ctx* ctx)
    void* c_shm_calloc(shm_allocator_ctx* ctx, size_t size, pthread_mutex_t* lock)
    void* c_shm_request(shm_allocator_ctx* ctx, size_t size, int scan_all_pages, pthread_mutex_t* lock)
    void c_shm_free(void* ptr, pthread_mutex_t* lock)
    void c_shm_reclaim(shm_allocator_ctx* ctx, pthread_mutex_t* lock)
    int c_shm_scan_allocator(char* shm_name)
    int c_shm_scan_page(char* shm_name)
    pid_t c_shm_pid(char* shm_name)
    shm_allocator* c_shm_allocator_dangling(char* shm_name)
    void c_shm_clear_dangling()


cdef class SharedMemoryPage:
    cdef shm_page_ctx* ctx

    @staticmethod
    cdef inline SharedMemoryPage c_from_header(shm_page_ctx* header)

    cpdef void reclaim(self)


cdef class SharedMemoryBlock:
    cdef shm_memory_block* block
    cdef readonly bint owner

    @staticmethod
    cdef inline SharedMemoryBlock c_from_header(shm_memory_block* header, bint owner=*)


cdef class SharedMemoryAllocator:
    cdef shm_allocator_ctx* ctx
    cdef readonly bint owner

    @staticmethod
    cdef SharedMemoryAllocator c_from_header(shm_allocator_ctx* header, bint owner=*)

    cdef inline shm_page_ctx* c_extend(self, size_t capacity=*, pthread_mutex_t* lock=*)

    cdef inline void* c_calloc(self, size_t size, pthread_mutex_t* lock=*)

    cdef inline void* c_request(self, size_t size, pthread_mutex_t* lock=*)

    cdef inline void c_free(self, void* ptr, pthread_mutex_t* lock=NULL)

    cpdef SharedMemoryPage extend(self, size_t capacity=*, bint with_lock=*)

    cpdef SharedMemoryBlock calloc(self, size_t size, bint with_lock=*)

    cpdef SharedMemoryBlock request(self, size_t size, bint scan_all_pages=*, bint with_lock=*)

    cpdef void free(self, SharedMemoryBlock buffer, bint with_lock=*)

    cpdef void reclaim(self, bint with_lock=*)

    cpdef list dangling(self)

    cpdef list dangling_pages(self)

    cpdef void cleanup_dangling(self)


cdef SharedMemoryAllocator ALLOCATOR
cdef shm_allocator_ctx* C_ALLOCATOR
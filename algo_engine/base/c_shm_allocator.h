#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#ifndef C_SHM_ALLOCATOR_H
#define C_SHM_ALLOCATOR_H

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <pthread.h>
#include <signal.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

// Configuration
#ifndef DEFAULT_AUTOPAGE_CAPACITY
#define DEFAULT_AUTOPAGE_CAPACITY (64 * 1024) /* 64 KiB */
#endif

#ifndef MAX_AUTOPAGE_CAPACITY
#define MAX_AUTOPAGE_CAPACITY (16 * 1024 * 1024) /* 16 MiB */
#endif

#ifndef DEFAULT_AUTOPAGE_ALIGNMENT
#define DEFAULT_AUTOPAGE_ALIGNMENT (4 * 1024) /* 4 KiB */
#endif

#ifndef SHM_ALLOCATOR_PREFIX
#define SHM_ALLOCATOR_PREFIX "/c_shm_allocator"
#endif

#ifndef SHM_PAGE_PREFIX
#define SHM_PAGE_PREFIX "/c_shm_page"
#endif

#ifndef SHM_NAME_LEN
#define SHM_NAME_LEN 256
#endif

#ifndef SHM_ALLOCATOR_DEFAULT_REGION_SIZE
#define SHM_ALLOCATOR_DEFAULT_REGION_SIZE (128ULL << 30)  /* 128 GiB */
#endif

typedef struct shm_page {
    size_t capacity;  // total capacity, excluding metadata
    size_t occupied;  // bytes occupied, excluding metadata
    size_t offset;
    struct shm_allocator* allocator;
    struct shm_memory_block* allocated;
    char shm_name[SHM_NAME_LEN];
    char prev_name[SHM_NAME_LEN];
} shm_page;

typedef struct shm_page_ctx {
    shm_page* shm_page;
    int shm_fd;
    char* buffer;
    struct shm_page_ctx* prev;
} shm_page_ctx;

typedef struct shm_memory_block {
    size_t capacity;
    size_t size;
    struct shm_memory_block* next_free;
    struct shm_memory_block* next_allocated;
    shm_page* parent_page;
    char buffer[];
} shm_memory_block;

typedef struct shm_allocator {
    char shm_name[SHM_NAME_LEN];
    size_t pid;
    pthread_mutex_t lock;
    uintptr_t region;
    size_t region_size;
    size_t mapped_size;
    char active_page[SHM_NAME_LEN];
    size_t mapped_pages;
    shm_memory_block* free_list;
} shm_allocator;

typedef struct shm_allocator_ctx {
    shm_allocator* shm_allocator;
    int shm_fd;
    shm_page_ctx* active_page;
} shm_allocator_ctx;

// ========== Forward Declarations (public API) ==========
/**
 * @brief Round up size to DEFAULT_AUTOPAGE_ALIGNMENT.
 * @param size Requested size in bytes.
 * @return Rounded-up size.
 */
static size_t c_page_roundup(size_t size);

/**
 * @brief Round up size to pointer alignment.
 * @param size Requested size in bytes.
 * @return Rounded-up size.
 */
static size_t c_block_roundup(size_t size);

/**
 * @brief Scan /dev/shm for an entry matching prefix.
 * @param prefix Prefix to match (may include leading '/').
 * @param out Output buffer to receive the found name (with leading '/').
 * @return 0 on success, -1 with errno set otherwise.
 */
static inline int c_shm_scan(const char* prefix, char* out);

/**
 * @brief Allocate a new shared-memory page metadata + fd (not yet mapped).
 * @param allocator Allocator metadata (shared) or NULL.
 * @param page_capacity Total bytes including metadata.
 * @return Page context or NULL on failure.
 */
static inline shm_page_ctx* c_shm_page_new(shm_allocator* allocator, size_t page_capacity);

/**
 * @brief Map a page into the allocator's reserved region at the next offset.
 * @param allocator Allocator metadata (shared).
 * @param page_ctx Page context to map.
 * @return 0 on success, -1 on error.
 */
static inline int c_shm_page_map(shm_allocator* allocator, shm_page_ctx* page_ctx);

/**
 * @brief Reclaim freed blocks of a given page back to unallocated state, with best effort.
 * @param allocator Allocator metadata (shared).
 * @param page_ctx Page context to reclaim from.
 */
static inline void c_shm_page_reclaim(shm_allocator* allocator, shm_page_ctx* page_ctx);

/**
 * @brief Extend allocator with a new page, optionally locking.
 * @param ctx Allocator context.
 * @param capacity Payload bytes requested (0 for auto sizing).
 * @param lock Optional mutex to lock during extend.
 * @return New active page context or NULL on error.
 */
static inline shm_page_ctx* c_shm_allocator_extend(shm_allocator_ctx* ctx, size_t capacity, pthread_mutex_t* lock);

/**
 * @brief Create a new allocator and reserve address space.
 * @param region_size Virtual region size to reserve (0 for default).
 * @return Allocator context or NULL on failure.
 */
static inline shm_allocator_ctx* c_shm_allocator_new(size_t region_size);

/**
 * @brief Tear down allocator, unmapping pages and unlinking SHM objects.
 * @param ctx Allocator context; NULL-safe.
 */
static inline void c_shm_allocator_free(shm_allocator_ctx* ctx);

/**
 * @brief Allocate zeroed memory from shared pages.
 * @param ctx Allocator context.
 * @param size Requested size.
 * @param lock Optional mutex controlling allocation.
 * @return Pointer to zeroed memory or NULL on error.
 */
static inline void* c_shm_calloc(shm_allocator_ctx* ctx, size_t size, pthread_mutex_t* lock);

/**
 * @brief Allocate memory with optional free-list reuse and page scanning.
 * @param ctx Allocator context.
 * @param size Requested size.
 * @param scan_all_pages Non-zero scans older pages before extending.
 * @param lock Optional mutex controlling allocation.
 * @return Pointer to allocated memory or NULL on error.
 */
static inline void* c_shm_request(shm_allocator_ctx* ctx, size_t size, int scan_all_pages, pthread_mutex_t* lock);

/**
 * @brief Return a previously allocated block to the allocator free list.
 * @param ptr User pointer obtained from request/calloc.
 * @param lock Optional mutex controlling free.
 */
static inline void c_shm_free(void* ptr, pthread_mutex_t* lock);

/**
 * @brief Best-effort reclaim of freed blocks across all pages owned by ctx.
 * @param ctx Allocator context.
 * @param lock Optional mutex controlling reclaim.
 */
static inline void c_shm_reclaim(shm_allocator_ctx* ctx, pthread_mutex_t* lock);

/**
 * @brief Scan for an allocator SHM name.
 * @param shm_name Output buffer for found name.
 * @return 0 on success, -1 on failure.
 */
static inline int c_shm_scan_allocator(char* shm_name);

/**
 * @brief Scan for a page SHM name.
 * @param shm_name Output buffer for found name.
 * @return 0 on success, -1 on failure.
 */
static inline int c_shm_scan_page(char* shm_name);

/**
 * @brief Extract pid embedded in allocator/page SHM name.
 * @param shm_name SHM object name (with or without leading '/').
 * @return pid on success, -1 with errno=EINVAL on parse failure.
 */
static inline pid_t c_shm_pid(char* shm_name);

/**
 * @brief Find and map a dangling allocator (whose creator pid is gone).
 * @param shm_name Output buffer to receive the dangling allocator name.
 * @return Mapped allocator pointer or NULL with errno set.
 */
static inline shm_allocator* c_shm_allocator_dangling(char* shm_name);

/**
 * @brief Unlink all dangling allocator/page SHM objects.
 */
static inline void c_shm_clear_dangling();

// ========== Utility Functions ==========

#ifndef C_COMMON_ROUNDUP_UTILS_DEFINED
#define C_COMMON_ROUNDUP_UTILS_DEFINED
static inline size_t c_page_roundup(size_t size) {
    return (size + DEFAULT_AUTOPAGE_ALIGNMENT - 1) & ~(DEFAULT_AUTOPAGE_ALIGNMENT - 1);
}

static inline size_t c_block_roundup(size_t size) {
    return (size + sizeof(void*) - 1) & ~(sizeof(void*) - 1);
}
#endif /* C_COMMON_ROUNDUP_UTILS_DEFINED */

#ifndef C_SHM_OVERHEAD_CONSTANTS_DEFINED
#define C_SHM_OVERHEAD_CONSTANTS_DEFINED
static const size_t c_shm_page_overhead = (sizeof(shm_page) + sizeof(void*) - 1) & ~(sizeof(void*) - 1);
static const size_t c_shm_block_overhead = (sizeof(shm_memory_block) + sizeof(void*) - 1) & ~(sizeof(void*) - 1);
#endif /* C_SHM_OVERHEAD_CONSTANTS_DEFINED */

static inline void c_shm_allocator_name(const void* region, char* out) {
    pid_t pid = getpid();
    // The shm name should be in format of {prefix}_{pid}_{region_hex}
    snprintf(out, SHM_NAME_LEN, "%s_%d_%lx", SHM_ALLOCATOR_PREFIX, (int) pid, (uintptr_t) region);

}

static inline void c_shm_page_name(shm_allocator* allocator, char* out) {
    pid_t pid = getpid();
    size_t page_idx = allocator->mapped_pages;
    // The shm name should be in format of {prefix}_{pid}_{6 digit page_idx}
    snprintf(out, SHM_NAME_LEN, "%s_%d_%06zu", SHM_PAGE_PREFIX, (int) pid, page_idx);
}

static inline int c_shm_scan(const char* prefix, char* out) {
    if (!prefix || !out) {
        errno = EINVAL;
        return -1;
    }

    if (prefix[0] == '/') {
        prefix += 1;
    }

    DIR* dir = opendir("/dev/shm");
    if (!dir) {
        return -1;
    }

    size_t prefix_len = strlen(prefix);
    struct dirent* ent = NULL;
    int found = 0;

    while ((ent = readdir(dir)) != NULL) {
        if (ent->d_name[0] == '.') {
            continue;
        }
        if (strncmp(ent->d_name, prefix, prefix_len) != 0) {
            continue;
        }

        size_t name_len = strnlen(ent->d_name, SHM_NAME_LEN - 1);
        if (name_len + 2 <= SHM_NAME_LEN) {
            out[0] = '/';
            memcpy(out + 1, ent->d_name, name_len);
            out[1 + name_len] = '\0';
            found = 1;
            break;
        }
    }

    closedir(dir);

    if (!found) {
        errno = ENOENT;
        return -1;
    }

    return 0;
}

static inline shm_page_ctx* c_shm_page_new(shm_allocator* allocator, size_t page_capacity) {
    shm_page_ctx* ctx = (shm_page_ctx*) calloc(1, sizeof(shm_page_ctx));
    if (!ctx) {
        return NULL;
    }

    shm_page* meta = (shm_page*) calloc(1, c_shm_page_overhead);
    if (!meta) {
        free(ctx);
        return NULL;
    }
    ctx->shm_page = meta;

    meta->capacity = page_capacity;
    meta->occupied = c_shm_page_overhead;
    meta->offset = 0;

    c_shm_page_name(allocator, meta->shm_name);

    int fd = shm_open(meta->shm_name, O_CREAT | O_RDWR | O_EXCL, 0600);
    if (fd == -1) {
        free(meta);
        free(ctx);
        return NULL;
    }

    if (ftruncate(fd, page_capacity) != 0) {
        close(fd);
        shm_unlink(meta->shm_name);
        free(meta);
        free(ctx);
        return NULL;
    }

    ctx->shm_page = meta;
    ctx->shm_fd = fd;
    ctx->buffer = NULL;
    ctx->prev = NULL;

    return ctx;
}

static inline int c_shm_page_map(shm_allocator* allocator, shm_page_ctx* page_ctx) {
    if (!allocator || !page_ctx || !page_ctx->shm_page || page_ctx->shm_fd < 0) {
        errno = EINVAL;
        return -1;
    }

    size_t page_capacity = page_ctx->shm_page->capacity;
    size_t offset = allocator->mapped_size;

    if (offset + page_capacity > allocator->region_size) {
        errno = ENOMEM;
        return -1;
    }

    /* compute target address from integer region base */
    void* target_addr = (void*) ((char*) (uintptr_t) allocator->region + offset);
    void* mapped = mmap(target_addr, page_capacity, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED, page_ctx->shm_fd, 0);
    if (mapped == MAP_FAILED) {
        return -1;
    }

    // Step 0: Point to metadata at start of page
    shm_page* page_meta = (shm_page*) mapped;
    memcpy(page_meta, page_ctx->shm_page, c_shm_page_overhead);
    free(page_ctx->shm_page);
    page_ctx->shm_page = page_meta;
    page_ctx->buffer = (char*) mapped;  // occupied already included the overhead, so the buffer should start at mapped

    // Step 1: Set correct offset
    page_meta->offset = offset;
    page_meta->allocator = allocator;

    // Step 2: Set prev_name = current active_page (enables backward traversal)
    if (allocator->mapped_pages == 0) {
        page_meta->prev_name[0] = '\0';
    }
    else {
        strncpy(page_meta->prev_name, allocator->active_page, SHM_NAME_LEN - 1);
        page_meta->prev_name[SHM_NAME_LEN - 1] = '\0';
    }

    // Step 4: Update allocator state
    allocator->mapped_size += page_capacity;
    strncpy(allocator->active_page, page_meta->shm_name, SHM_NAME_LEN - 1);
    allocator->active_page[SHM_NAME_LEN - 1] = '\0';
    allocator->mapped_pages++;

    return 0;
}

static inline void c_shm_page_reclaim(shm_allocator* allocator, shm_page_ctx* page_ctx) {
    if (!allocator || !page_ctx || !page_ctx->shm_page) {
        errno = EINVAL;
        return;
    }

    shm_page* page = page_ctx->shm_page;

    // Best-effort: walk from newest; stop at first live block.
    shm_memory_block** prevp = &page->allocated;
    while (*prevp) {
        shm_memory_block* block = *prevp;
        if (block->size != 0) {
            break; // first live block reached; older blocks retained
        }

        // Remove from page allocation list
        *prevp = block->next_allocated;
        block->next_allocated = NULL;

        // Remove from allocator free list (single occurrence)
        shm_memory_block** free_prev = &allocator->free_list;
        while (*free_prev && *free_prev != block) {
            free_prev = &(*free_prev)->next_free;
        }
        if (*free_prev == block) {
            *free_prev = block->next_free;
        }
        block->next_free = NULL;

        // Return occupied size back to page
        size_t cap_total = block->capacity + c_shm_block_overhead;
        if (page->occupied >= cap_total) {
            page->occupied -= cap_total;
        }
    }
}

// ========== Public API Functions ==========

static inline shm_page_ctx* c_shm_allocator_extend(shm_allocator_ctx* ctx, size_t capacity, pthread_mutex_t* lock) {
    if (!ctx || !ctx->shm_allocator) {
        errno = EINVAL;
        return NULL;
    }

    shm_allocator* allocator = ctx->shm_allocator;
    uint8_t locked = 0;

    if (lock) {
        int ret = pthread_mutex_lock(lock);
        if (ret != 0) {
            errno = ret;
            return NULL;
        }
        locked = 1;
    }

    // Step 1: Determine new page capacity
    if (capacity == 0) {
        if (!ctx->active_page) {
            capacity = DEFAULT_AUTOPAGE_CAPACITY;
        }
        else {
            size_t prev_cap = ctx->active_page->shm_page->capacity;
            capacity = prev_cap * 2;
            if (capacity < DEFAULT_AUTOPAGE_CAPACITY) {
                capacity = DEFAULT_AUTOPAGE_CAPACITY;
            }
            else if (capacity > MAX_AUTOPAGE_CAPACITY) {
                capacity = MAX_AUTOPAGE_CAPACITY;
            }
        }
    }

    size_t aligned_capacity = c_page_roundup(capacity);

    if (aligned_capacity + allocator->mapped_size > allocator->region_size) {
        if (locked) pthread_mutex_unlock(lock);
        errno = ENOMEM;
        return NULL;
    }

    // Step 2: Create new page
    shm_page_ctx* page_ctx = c_shm_page_new(allocator, aligned_capacity);
    if (!page_ctx) {
        if (locked) pthread_mutex_unlock(lock);
        return NULL;
    }

    // Step 3: Map new page into allocator region
    if (c_shm_page_map(allocator, page_ctx) != 0) {
        shm_unlink(page_ctx->shm_page->shm_name);
        close(page_ctx->shm_fd);
        free(page_ctx->shm_page);  // temp heap metadata
        free(page_ctx);
        if (locked) pthread_mutex_unlock(lock);
        return NULL;
    }

    // Step 4: Update context active_page
    page_ctx->prev = ctx->active_page;
    ctx->active_page = page_ctx;

    // Unlock if we locked here
    if (locked) pthread_mutex_unlock(lock);
    return page_ctx;
}

static inline shm_allocator_ctx* c_shm_allocator_new(size_t region_size) {
    shm_allocator_ctx* ctx = (shm_allocator_ctx*) calloc(1, sizeof(shm_allocator_ctx));
    if (!ctx) {
        return NULL;
    }

    if (region_size == 0) {
        region_size = SHM_ALLOCATOR_DEFAULT_REGION_SIZE; // 128 GiB
    }

    // Step 1: Reserve virtual address space for the 128 GiB page region
    void* virtual_region = mmap(
        NULL, region_size,
        PROT_NONE,
        MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE,
        -1, 0
    );
    if (virtual_region == MAP_FAILED) {
        free(ctx);
        return NULL;
    }

    // Step 2: Create SHM object for allocator metadata
    char meta_shm_name[SHM_NAME_LEN];
    c_shm_allocator_name(virtual_region, meta_shm_name);

    int meta_fd = shm_open(meta_shm_name, O_CREAT | O_RDWR | O_EXCL, 0600);
    if (meta_fd == -1) {
        munmap(virtual_region, region_size);
        free(ctx);
        return NULL;
    }

    if (ftruncate(meta_fd, sizeof(shm_allocator)) != 0) {
        close(meta_fd);
        shm_unlink(meta_shm_name);
        munmap(virtual_region, region_size);
        free(ctx);
        return NULL;
    }

    // Step 3: Map allocator metadata into memory
    shm_allocator* meta = (shm_allocator*) mmap(NULL, sizeof(shm_allocator), PROT_READ | PROT_WRITE, MAP_SHARED, meta_fd, 0);
    if (meta == MAP_FAILED) {
        close(meta_fd);
        shm_unlink(meta_shm_name);
        munmap(virtual_region, region_size);
        free(ctx);
        return NULL;
    }

    // Step 4: Initialize metadata IN SHARED MEMORY
    // Must zero first (mmap doesn't guarantee zeroed memory for ftruncate'd SHM)
    memset(meta, 0, sizeof(shm_allocator));

    size_t name_len = strnlen(meta_shm_name, SHM_NAME_LEN - 1);
    memcpy(meta->shm_name, meta_shm_name, name_len);
    meta->shm_name[name_len] = '\0';
    meta->pid = (size_t) getpid();
    meta->region = (uintptr_t) virtual_region;
    meta->region_size = region_size;
    meta->mapped_size = 0;
    meta->mapped_pages = 0;
    meta->active_page[0] = '\0';

    // Step 5: Initialize mutex in shared memory with PTHREAD_PROCESS_SHARED
    pthread_mutexattr_t mattr;
    if (pthread_mutexattr_init(&mattr) != 0) {
        goto cleanup;
    }
    if (pthread_mutexattr_setpshared(&mattr, PTHREAD_PROCESS_SHARED) != 0) {
        pthread_mutexattr_destroy(&mattr);
        goto cleanup;
    }
    if (pthread_mutex_init(&meta->lock, &mattr) != 0) {
        pthread_mutexattr_destroy(&mattr);
        goto cleanup;
    }
    pthread_mutexattr_destroy(&mattr);

    // Step 6: Fill context
    ctx->shm_allocator = meta;
    ctx->shm_fd = meta_fd;
    ctx->active_page = NULL;

    return ctx;

cleanup:
    munmap(virtual_region, region_size);
    munmap(meta, sizeof(shm_allocator));
    close(meta_fd);
    shm_unlink(meta_shm_name);
    free(ctx);
    return NULL;
}

static inline void c_shm_allocator_free(shm_allocator_ctx* ctx) {
    if (!ctx) {
        return;
    }

    char shm_name[SHM_NAME_LEN];

    // Step 1: Unmap all pages
    shm_page_ctx* page_ctx = ctx->active_page;
    while (page_ctx) {
        shm_page_ctx* prev = page_ctx->prev;
        shm_page* page_meta = page_ctx->shm_page;

        strncpy(shm_name, page_meta->shm_name, sizeof(shm_name) - 1);
        shm_name[sizeof(shm_name) - 1] = '\0';

        if (page_meta) {
            munmap((void*) page_meta, page_meta->capacity);
            shm_unlink(shm_name);
        }

        if (page_ctx->shm_fd >= 0) {
            close(page_ctx->shm_fd);
        }

        free(page_ctx);
        page_ctx = prev;
    }

    // Step 2: Unmap allocator region
    shm_allocator* allocator = ctx->shm_allocator;
    if (allocator) {
        // Step 2.1: Save shm_name before unmapping
        strncpy(shm_name, allocator->shm_name, sizeof(shm_name) - 1);
        shm_name[sizeof(shm_name) - 1] = '\0';

        // Step 2.2: Destroy mutex
        pthread_mutex_destroy(&allocator->lock);

        // Step 2.3: Unmap region
        if (allocator->region) {
            munmap((void*) allocator->region, allocator->region_size);
        }

        // Step 2.4: Unmap allocator shm
        munmap((void*) allocator, sizeof(shm_allocator));

        // Step 2.5: Unlink allocator SHM
        shm_unlink(shm_name);
    }

    // Step 3: Close allocator SHM fd
    if (ctx->shm_fd >= 0) {
        close(ctx->shm_fd);
    }

    // Step 4: Free context
    free(ctx);
}

static inline void* c_shm_calloc(shm_allocator_ctx* ctx, size_t size, pthread_mutex_t* lock) {
    if (!ctx || !ctx->shm_allocator || size == 0) {
        errno = EINVAL;
        return NULL;
    }

    size_t cap_net = c_block_roundup(size);
    // the overhead is already aligned due to struct padding
    size_t cap_total = cap_net + c_shm_block_overhead;
    shm_allocator* allocator = ctx->shm_allocator;

    // Step 1: Lock allocator
    // Locking strategy:
    // - If lock is NULL, caller is responsible for locking, calloc and extend will not be locked
    // - If lock is allocator's builtin lock, calloc will be locked, extend will not be locked to void double lock
    // - If lock is a different lock, calloc will use the provided lock, extend will also use the builtin lock

    uint8_t locked = 0;
    pthread_mutex_t* builtin_lock = &allocator->lock;
    pthread_mutex_t* child_lock = &allocator->lock;
    if (lock) {
        if (lock == builtin_lock) {
            child_lock = NULL; // avoid double locking
        }
        int ret = pthread_mutex_lock(lock);
        if (ret != 0) {
            errno = ret;
            return NULL;
        }
        locked = 1;
    }
    else {
        child_lock = NULL; // caller is responsible for locking
    }

    // Step 2: Extend the allocator if there is no active_page or insufficient space
    shm_page_ctx* page_ctx = ctx->active_page;
    if (!page_ctx) {
        // Step 2.1: Roundup to nearest DEFAULT_AUTOPAGE_CAPACITY * 2^n, for the first page
        size_t target_cap = DEFAULT_AUTOPAGE_CAPACITY;
        while (target_cap < cap_total + c_shm_page_overhead) {
            target_cap *= 2;
        }

        page_ctx = c_shm_allocator_extend(ctx, target_cap, child_lock);
        if (!page_ctx) {
            if (locked) pthread_mutex_unlock(lock);
            return NULL;
        }
    }

    shm_page* page_meta = page_ctx->shm_page;

    // Step 2: Extend the allocator if there is no active_page or insufficient space
    if (page_meta->occupied + cap_total > page_meta->capacity) {
        // Step 2.1: Determine new page capacity
        size_t target_cap = page_meta->capacity;

        if (target_cap < DEFAULT_AUTOPAGE_CAPACITY) {
            target_cap = DEFAULT_AUTOPAGE_CAPACITY;
        }
        else if (target_cap < MAX_AUTOPAGE_CAPACITY) {
            target_cap *= 2;
        }

        // Step 2.1: Roundup to nearest DEFAULT_AUTOPAGE_CAPACITY * 2^n, for the first page
        while (target_cap < cap_total + c_shm_page_overhead) {
            target_cap *= 2;
        }
        page_ctx = c_shm_allocator_extend(ctx, target_cap, child_lock);
        if (!page_ctx) {
            if (locked) pthread_mutex_unlock(lock);
            return NULL;
        }

        page_meta = page_ctx->shm_page;
    }

    // Step 3: Allocate memory from active page
    size_t offset = page_meta->occupied;
    shm_memory_block* block = (shm_memory_block*) (page_ctx->buffer + offset);
    block->capacity = cap_net;
    block->size = size;
    block->next_free = NULL;

    // Step 4: Link fresh block to its page (calloc path is always fresh)
    block->parent_page = page_meta;
    block->next_allocated = page_meta->allocated;
    page_meta->allocated = block;

    page_meta->occupied += cap_total;

    // Step 5: Unlock allocator
    if (locked) pthread_mutex_unlock(lock);

    memset(block + 1, 0, cap_net);
    return (void*) block->buffer;
}

static inline void* c_shm_request(shm_allocator_ctx* ctx, size_t size, int scan_all_pages, pthread_mutex_t* lock) {
    if (!ctx || !ctx->shm_allocator || size == 0) {
        errno = EINVAL;
        return NULL;
    }

    size_t cap_net = c_block_roundup(size);
    size_t cap_total = cap_net + c_shm_block_overhead;
    shm_allocator* allocator = ctx->shm_allocator;

    uint8_t locked = 0;
    pthread_mutex_t* builtin_lock = &allocator->lock;
    pthread_mutex_t* child_lock = &allocator->lock;
    if (lock) {
        if (lock == builtin_lock) {
            child_lock = NULL;
        }
        int ret = pthread_mutex_lock(lock);
        if (ret != 0) {
            errno = ret;
            return NULL;
        }
        locked = 1;
    }
    else {
        child_lock = NULL;
    }

    // Step 1: Try allocator-level free list reuse
    shm_memory_block** prevp = &allocator->free_list;
    shm_memory_block* free_blk = allocator->free_list;
    while (free_blk) {
        if (free_blk->capacity >= cap_net) {
            *prevp = free_blk->next_free;
            free_blk->next_free = NULL;
            free_blk->size = size;
            if (locked) pthread_mutex_unlock(lock);
            memset(free_blk + 1, 0, cap_net);
            return (void*) free_blk->buffer;
        }
        prevp = &free_blk->next_free;
        free_blk = free_blk->next_free;
    }

    // Step 2: Locate a page with enough space (optionally scan all pages)
    shm_page_ctx* target_ctx = NULL;
    if (scan_all_pages) {
        shm_page_ctx* iter = ctx->active_page;
        while (iter) {
            shm_page* meta = iter->shm_page;
            if (meta && meta->occupied + cap_total <= meta->capacity) {
                target_ctx = iter;
                break;
            }
            iter = iter->prev;
        }
    }
    else if (ctx->active_page) {
        shm_page* meta = ctx->active_page->shm_page;
        if (meta && meta->occupied + cap_total <= meta->capacity) {
            target_ctx = ctx->active_page;
        }
    }

    // Step 3: Extend allocator if no existing page fits
    if (!target_ctx) {
        shm_page_ctx* current = ctx->active_page;
        size_t target_cap;

        if (!current) {
            target_cap = DEFAULT_AUTOPAGE_CAPACITY;
            while (target_cap < cap_total + c_shm_page_overhead) {
                target_cap *= 2;
            }
        }
        else {
            size_t prev_cap = current->shm_page->capacity;
            size_t new_cap = prev_cap;

            if (new_cap < DEFAULT_AUTOPAGE_CAPACITY) {
                new_cap = DEFAULT_AUTOPAGE_CAPACITY;
            }
            else if (new_cap < MAX_AUTOPAGE_CAPACITY) {
                new_cap *= 2;
            }

            while (new_cap < cap_total + c_shm_page_overhead) {
                new_cap *= 2;
            }
            target_cap = new_cap;
        }

        target_ctx = c_shm_allocator_extend(ctx, target_cap, child_lock);
        if (!target_ctx) {
            if (locked) pthread_mutex_unlock(lock);
            return NULL;
        }
    }

    // Step 4: Allocate from the selected page
    shm_page* page_meta = target_ctx->shm_page;
    size_t offset = page_meta->occupied;
    shm_memory_block* block = (shm_memory_block*) (target_ctx->buffer + offset);
    block->capacity = cap_net;
    block->size = size;
    block->next_free = NULL;

    // Step 5: Link block to its page for potential recycling scans
    block->parent_page = page_meta;
    block->next_allocated = page_meta->allocated;
    page_meta->allocated = block;

    page_meta->occupied += cap_total;

    if (locked) pthread_mutex_unlock(lock);

    memset(block + 1, 0, cap_net);
    return (void*) block->buffer;
}

static inline void c_shm_free(void* ptr, pthread_mutex_t* lock) {
    if (!ptr) {
        errno = EINVAL;
        return;
    }

    shm_memory_block* block = (shm_memory_block*) ((char*) ptr - c_shm_block_overhead);
    shm_page* page = block->parent_page;
    if (!page || !page->allocator) {
        errno = EINVAL;
        return;
    }

    shm_allocator* allocator = page->allocator;

    uint8_t locked = 0;
    if (lock) {
        int ret = pthread_mutex_lock(lock);
        if (ret != 0) {
            errno = ret;
            return;
        }
        locked = 1;
    }

    block->size = 0;
    block->next_free = allocator->free_list;
    allocator->free_list = block;

    if (locked) pthread_mutex_unlock(lock);
}

static inline void c_shm_reclaim(shm_allocator_ctx* ctx, pthread_mutex_t* lock) {
    if (!ctx || !ctx->shm_allocator) {
        errno = EINVAL;
        return;
    }

    uint8_t locked = 0;
    if (lock) {
        int ret = pthread_mutex_lock(lock);
        if (ret != 0) {
            errno = ret;
            return;
        }
        locked = 1;
    }

    shm_allocator* allocator = ctx->shm_allocator;
    shm_page_ctx* page_ctx = ctx->active_page;
    while (page_ctx) {
        c_shm_page_reclaim(allocator, page_ctx);
        page_ctx = page_ctx->prev;
    }

    if (locked) pthread_mutex_unlock(lock);
}

// ========== Internal SHM Management API ==========

static inline int c_shm_scan_allocator(char* shm_name) {
    return c_shm_scan(SHM_ALLOCATOR_PREFIX, shm_name);
}

static inline int c_shm_scan_page(char* shm_name) {
    return c_shm_scan(SHM_PAGE_PREFIX, shm_name);
}

static inline pid_t c_shm_pid(char* shm_name) {
    if (!shm_name) {
        errno = EINVAL;
        return -1;
    }

    const char* prefixes[] = { SHM_ALLOCATOR_PREFIX, SHM_PAGE_PREFIX };

    const char* base = shm_name;
    if (base[0] == '/') {
        base += 1;
    }

    const char* matched_prefix = NULL;
    for (size_t i = 0; i < sizeof(prefixes) / sizeof(prefixes[0]); ++i) {
        const char* p = prefixes[i];
        if (p[0] == '/') p += 1;
        size_t len = strlen(p);
        if (strncmp(base, p, len) == 0) {
            matched_prefix = p;
            base += len;
            break;
        }
    }

    if (!matched_prefix) {
        errno = EINVAL;
        return -1;
    }

    if (base[0] != '_') {
        errno = EINVAL;
        return -1;
    }
    base += 1;

    const char* us = strchr(base, '_');
    if (!us) {
        errno = EINVAL;
        return -1;
    }

    size_t pid_len = (size_t) (us - base);
    if (pid_len == 0 || pid_len >= 32) {
        errno = EINVAL;
        return -1;
    }

    char pidbuf[32];
    memcpy(pidbuf, base, pid_len);
    pidbuf[pid_len] = '\0';

    errno = 0;
    char* endptr = NULL;
    long v = strtol(pidbuf, &endptr, 10);
    if (errno != 0 || endptr == pidbuf || *endptr != '\0') {
        errno = EINVAL;
        return -1;
    }

    if (v <= 0 || v > INT_MAX) {
        errno = EINVAL;
        return -1;
    }

    return (pid_t) v;
}

static inline shm_allocator* c_shm_allocator_dangling(char* shm_name) {
    if (!shm_name) {
        errno = EINVAL;
        return NULL;
    }

    const char* fs_prefix = SHM_ALLOCATOR_PREFIX;
    if (fs_prefix[0] == '/') fs_prefix += 1;
    size_t prefix_len = strlen(fs_prefix);

    DIR* dir = opendir("/dev/shm");
    if (!dir) {
        return NULL;
    }

    struct dirent* ent = NULL;
    shm_allocator* mapped = NULL;
    char candidate[SHM_NAME_LEN];

    while ((ent = readdir(dir)) != NULL) {
        if (ent->d_name[0] == '.') continue;
        if (strncmp(ent->d_name, fs_prefix, prefix_len) != 0) continue;

        size_t name_len = strnlen(ent->d_name, SHM_NAME_LEN - 1);
        if (name_len + 2 > SHM_NAME_LEN) {
            continue; // would truncate; skip this entry
        }
        candidate[0] = '/';
        memcpy(candidate + 1, ent->d_name, name_len);
        candidate[1 + name_len] = '\0';

        pid_t pid = c_shm_pid(candidate);
        if (pid <= 0) continue;

        if (kill(pid, 0) == 0 || errno != ESRCH) {
            errno = 0; // either alive or another error; treat as not dangling
            continue;
        }

        int fd = shm_open(candidate, O_RDWR, 0);
        if (fd == -1) {
            continue;
        }

        struct stat st;
        if (fstat(fd, &st) != 0) {
            close(fd);
            continue;
        }

        void* map = mmap(NULL, (size_t) st.st_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        close(fd);
        if (map == MAP_FAILED) {
            continue;
        }

        strncpy(shm_name, candidate, SHM_NAME_LEN - 1);
        shm_name[SHM_NAME_LEN - 1] = '\0';
        mapped = (shm_allocator*) map;
        break;
    }

    closedir(dir);

    if (!mapped) {
        errno = ENOENT;
    }

    return mapped;
}

static inline void c_shm_clear_dangling() {
    DIR* dir = opendir("/dev/shm");
    if (!dir) {
        return;
    }

    const char* prefixes[] = { SHM_ALLOCATOR_PREFIX, SHM_PAGE_PREFIX };
    size_t prefix_count = sizeof(prefixes) / sizeof(prefixes[0]);

    struct dirent* ent = NULL;
    while ((ent = readdir(dir)) != NULL) {
        if (ent->d_name[0] == '.') continue;

        int matched = 0;
        for (size_t i = 0; i < prefix_count; ++i) {
            const char* p = prefixes[i];
            if (p[0] == '/') p += 1;
            size_t len = strlen(p);
            if (strncmp(ent->d_name, p, len) == 0) {
                matched = 1;
                break;
            }
        }
        if (!matched) continue;

        char candidate[SHM_NAME_LEN];
        size_t name_len = strnlen(ent->d_name, SHM_NAME_LEN - 1);
        if (name_len + 2 > SHM_NAME_LEN) {
            continue; // would truncate; skip this entry
        }
        candidate[0] = '/';
        memcpy(candidate + 1, ent->d_name, name_len);
        candidate[1 + name_len] = '\0';

        pid_t pid = c_shm_pid(candidate);
        if (pid <= 0) {
            continue;
        }

        errno = 0;
        if (kill(pid, 0) == -1 && errno == ESRCH) {
            shm_unlink(candidate); // best effort
        }
    }

    closedir(dir);
}

#endif // C_SHM_ALLOCATOR_H
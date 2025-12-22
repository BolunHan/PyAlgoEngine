#ifndef C_HEAP_ALLOCATOR_H
#define C_HEAP_ALLOCATOR_H

#include <errno.h>
#include <pthread.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

// ========== Configuration ==========

#ifndef DEFAULT_AUTOPAGE_CAPACITY
#define DEFAULT_AUTOPAGE_CAPACITY (64 * 1024) /* 64 KiB */
#endif

#ifndef MAX_AUTOPAGE_CAPACITY
#define MAX_AUTOPAGE_CAPACITY (16 * 1024 * 1024) /* 16 MiB */
#endif

#ifndef DEFAULT_AUTOPAGE_ALIGNMENT
#define DEFAULT_AUTOPAGE_ALIGNMENT (4 * 1024) /* 4 KiB */
#endif

// ========== Heap Allocator Structs ==========

typedef struct heap_memory_block {
    size_t capacity;
    size_t size;
    struct heap_memory_block* next_free;
    struct heap_memory_block* next_allocated;
    struct heap_page* parent_page;
    char buffer[];
} heap_memory_block;

typedef struct heap_page {
    size_t capacity;
    size_t occupied;
    struct heap_allocator* allocator;
    struct heap_page* prev;
    struct heap_memory_block* allocated;
    char buffer[];
} heap_page;

typedef struct heap_allocator {
    pthread_mutex_t lock;
    size_t mapped_pages;
    struct heap_memory_block* free_list;
    struct heap_page* active_page;
} heap_allocator;

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

static inline void c_heap_page_reclaim(heap_allocator* allocator, heap_page* page) {
    if (!allocator || !page) {
        errno = EINVAL;
        return;
    }

    heap_memory_block** prevp = &page->allocated;
    while (*prevp) {
        heap_memory_block* block = *prevp;
        if (block->size != 0) {
            break;
        }

        *prevp = block->next_allocated;
        block->next_allocated = NULL;

        heap_memory_block** free_prev = &allocator->free_list;
        while (*free_prev && *free_prev != block) {
            free_prev = &(*free_prev)->next_free;
        }
        if (*free_prev == block) {
            *free_prev = block->next_free;
        }
        block->next_free = NULL;

        size_t cap_total = block->capacity + sizeof(heap_memory_block);
        if (page->occupied >= cap_total) {
            page->occupied -= cap_total;
        }
    }
}

// ========== Public API Functions ==========

static inline heap_page* c_heap_allocator_extend(heap_allocator* allocator, size_t capacity, pthread_mutex_t* lock) {
    if (!allocator) {
        errno = EINVAL;
        return NULL;
    }

    uint8_t locked = 0;

    if (lock) {
        int ret = pthread_mutex_lock(lock);
        if (ret != 0) {
            errno = ret;
            return NULL;
        }
        locked = 1;
    }

    if (capacity == 0) {
        if (!allocator->active_page) {
            capacity = DEFAULT_AUTOPAGE_CAPACITY;
        }
        else {
            size_t prev_cap = allocator->active_page->capacity;
            capacity = prev_cap * 2;
            if (capacity < DEFAULT_AUTOPAGE_CAPACITY) {
                capacity = DEFAULT_AUTOPAGE_CAPACITY;
            }
            else if (capacity > MAX_AUTOPAGE_CAPACITY) {
                capacity = MAX_AUTOPAGE_CAPACITY;
            }
        }
    }

    size_t total_capacity = c_page_roundup(capacity);

    heap_page* page = (heap_page*) calloc(1, total_capacity);

    if (!page) {
        if (locked) pthread_mutex_unlock(lock);
        return NULL;
    }

    page->capacity = capacity;
    page->occupied = sizeof(heap_page);
    page->allocator = allocator;
    page->allocated = NULL;
    page->prev = allocator->active_page;
    allocator->active_page = page;
    allocator->mapped_pages++;

    if (locked) pthread_mutex_unlock(lock);
    return page;
}

static inline heap_allocator* c_heap_allocator_new() {
    heap_allocator* allocator = (heap_allocator*) calloc(1, sizeof(heap_allocator));
    if (!allocator) {
        return NULL;
    }

    if (pthread_mutex_init(&allocator->lock, NULL) != 0) {
        free(allocator);
        return NULL;
    }

    allocator->mapped_pages = 0;
    allocator->free_list = NULL;
    allocator->active_page = NULL;

    return allocator;
}

static inline void c_heap_allocator_free(heap_allocator* allocator) {
    if (!allocator) {
        return;
    }

    heap_page* page = allocator->active_page;
    while (page) {
        heap_page* prev = page->prev;
        free(page);
        page = prev;
    }

    pthread_mutex_destroy(&allocator->lock);
    free(allocator);
}

static inline void* c_heap_calloc(heap_allocator* allocator, size_t size, pthread_mutex_t* lock) {
    if (!allocator || size == 0) {
        errno = EINVAL;
        return NULL;
    }

    size_t cap_net = c_block_roundup(size);
    size_t overhead = sizeof(heap_memory_block);
    size_t cap_total = cap_net + overhead;

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

    heap_page* page = allocator->active_page;
    if (!page) {
        size_t target_cap = DEFAULT_AUTOPAGE_CAPACITY;
        while (target_cap < cap_total + sizeof(heap_page)) {
            target_cap *= 2;
        }

        page = c_heap_allocator_extend(allocator, target_cap, child_lock);
        if (!page) {
            if (locked) pthread_mutex_unlock(lock);
            return NULL;
        }
    }

    if (page->occupied + cap_total > page->capacity) {
        size_t target_cap = page->capacity;

        if (target_cap < DEFAULT_AUTOPAGE_CAPACITY) {
            target_cap = DEFAULT_AUTOPAGE_CAPACITY;
        }
        else if (target_cap < MAX_AUTOPAGE_CAPACITY) {
            target_cap *= 2;
        }

        while (target_cap < cap_total + sizeof(heap_page)) {
            target_cap *= 2;
        }
        page = c_heap_allocator_extend(allocator, target_cap, child_lock);
        if (!page) {
            if (locked) pthread_mutex_unlock(lock);
            return NULL;
        }
    }

    size_t offset = page->occupied;
    heap_memory_block* block = (heap_memory_block*) ((char*) page + offset);
    block->capacity = cap_net;
    block->size = size;
    block->next_free = NULL;

    block->parent_page = page;
    block->next_allocated = page->allocated;
    page->allocated = block;
    page->occupied += cap_total;

    if (locked) pthread_mutex_unlock(lock);

    memset(block + 1, 0, cap_net);
    return (void*) block->buffer;
}

static inline void* c_heap_request(heap_allocator* allocator, size_t size, int scan_all_pages, pthread_mutex_t* lock) {
    if (!allocator || size == 0) {
        errno = EINVAL;
        return NULL;
    }

    size_t cap_net = c_block_roundup(size);
    size_t overhead = sizeof(heap_memory_block);
    size_t cap_total = cap_net + overhead;

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

    heap_memory_block** prevp = &allocator->free_list;
    heap_memory_block* free_blk = allocator->free_list;
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

    heap_page* target_page = NULL;
    if (scan_all_pages) {
        heap_page* iter = allocator->active_page;
        while (iter) {
            if (iter->occupied + cap_total <= iter->capacity) {
                target_page = iter;
                break;
            }
            iter = iter->prev;
        }
    }
    else if (allocator->active_page) {
        heap_page* meta = allocator->active_page;
        if (meta && meta->occupied + cap_total <= meta->capacity) {
            target_page = allocator->active_page;
        }
    }

    if (!target_page) {
        heap_page* current = allocator->active_page;
        size_t target_cap;

        if (!current) {
            target_cap = DEFAULT_AUTOPAGE_CAPACITY;
            while (target_cap < cap_total + sizeof(heap_page)) {
                target_cap *= 2;
            }
        }
        else {
            size_t prev_cap = current->capacity;
            size_t new_cap = prev_cap;

            if (new_cap < DEFAULT_AUTOPAGE_CAPACITY) {
                new_cap = DEFAULT_AUTOPAGE_CAPACITY;
            }
            else if (new_cap < MAX_AUTOPAGE_CAPACITY) {
                new_cap *= 2;
            }

            while (new_cap < cap_total + sizeof(heap_page)) {
                new_cap *= 2;
            }
            target_cap = new_cap;
        }

        target_page = c_heap_allocator_extend(allocator, target_cap, child_lock);
        if (!target_page) {
            if (locked) pthread_mutex_unlock(lock);
            return NULL;
        }
    }

    size_t offset = target_page->occupied;
    heap_memory_block* block = (heap_memory_block*) ((char*) target_page + offset);
    block->capacity = cap_net;
    block->size = size;
    block->next_free = NULL;

    block->parent_page = target_page;
    block->next_allocated = target_page->allocated;
    target_page->allocated = block;

    target_page->occupied += cap_total;

    if (locked) pthread_mutex_unlock(lock);

    memset(block + 1, 0, cap_net);
    return (void*) block->buffer;
}

static inline void c_heap_free(void* ptr, pthread_mutex_t* lock) {
    if (!ptr) {
        errno = EINVAL;
        return;
    }

    heap_memory_block* block = (heap_memory_block*) ((char*) ptr - sizeof(heap_memory_block));
    heap_page* page = block->parent_page;
    if (!page || !page->allocator) {
        errno = EINVAL;
        return;
    }

    heap_allocator* allocator = page->allocator;

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

static inline void c_heap_reclaim(heap_allocator* allocator, pthread_mutex_t* lock) {
    if (!allocator) {
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

    heap_page* page = allocator->active_page;
    while (page) {
        c_heap_page_reclaim(allocator, page);
        page = page->prev;
    }

    if (locked) pthread_mutex_unlock(lock);
}

#endif /* C_HEAP_ALLOCATOR_H */

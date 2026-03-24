#ifndef C_AE_ALLOCATOR_PROTOCOL_H
#define C_AE_ALLOCATOR_PROTOCOL_H

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "../c_heap_allocator.h"
#include "../c_shm_allocator.h"

// ========== Constants ==========

#ifndef MD_ALLOC_VIGILANT
#define MD_ALLOC_VIGILANT 1
#endif

#ifndef MD_ALLOC_MAGIC
#define MD_ALLOC_MAGIC 0xCFBBBBFCULL
#endif

// ========== Structs ==========

typedef struct allocator_protocol {
    shm_allocator* shm_allocator;
    shm_allocator_ctx* shm_allocator_ctx;
    heap_allocator* heap_allocator;
    bool with_shm;
    bool with_lock;
    size_t size;
#if MD_ALLOC_VIGILANT > 0
    uint64_t magic;
#endif
    char buf[];
} allocator_protocol;

// ========== Forward Declaration ==========

static inline allocator_protocol* c_ae_allocator_protocol_request(size_t size, shm_allocator_ctx* shm_allocator, heap_allocator* heap_allocator, bool with_lock);
static inline void c_ae_allocator_protocol_recycle(allocator_protocol* protocol, bool with_lock);
static inline void* c_md_alloc(size_t size, allocator_protocol* schematic);
static inline void c_md_free(void* ptr);
static inline char* c_md_strdup(const char* src, allocator_protocol* allocator);
static inline void* c_md_realloc(void* src, size_t new_size, allocator_protocol* allocator);

// ========== Utilities Functions ==========

static inline allocator_protocol* c_ae_allocator_protocol_request(size_t size, shm_allocator_ctx* shm_allocator, heap_allocator* heap_allocator, bool with_lock) {
    if (size == 0) return NULL;
    size_t ttl_size = sizeof(allocator_protocol) + size;
    allocator_protocol* protocol;

    if (shm_allocator) {
        pthread_mutex_t* lock = with_lock ? &shm_allocator->shm_allocator->lock : NULL;
        protocol = (allocator_protocol*) c_shm_request(shm_allocator, ttl_size, 0, lock);
        if (!protocol) return NULL;
        protocol->shm_allocator = shm_allocator->shm_allocator;
        protocol->shm_allocator_ctx = shm_allocator;
        protocol->with_shm = 1;
    }
    else if (heap_allocator) {
        pthread_mutex_t* lock = with_lock ? &heap_allocator->lock : NULL;
        protocol = (allocator_protocol*) c_heap_request(heap_allocator, ttl_size, 0, lock);
        if (!protocol) return NULL;
        protocol->heap_allocator = heap_allocator;
    }
    else {
        protocol = (allocator_protocol*) calloc(1, ttl_size);
        if (!protocol) return NULL;
    }

    protocol->with_lock = with_lock;
    protocol->size = size;

#if MD_ALLOC_VIGILANT > 0
    protocol->magic = MD_ALLOC_MAGIC;
#endif

    return protocol;
}

static inline void c_ae_allocator_protocol_recycle(allocator_protocol* protocol, bool with_lock) {
    if (!protocol) return;

#if MD_ALLOC_VIGILANT > 0
    // Invalidate magic to catch double free or invalid free attempts
    protocol->magic = 0;
#endif

    shm_allocator* shm_allocator = protocol->shm_allocator;
    heap_allocator* heap_allocator = protocol->heap_allocator;

    if (shm_allocator) {
        pthread_mutex_t* lock = with_lock ? &shm_allocator->lock : NULL;
        c_shm_free((void*) protocol, lock);
    }
    else if (heap_allocator) {
        pthread_mutex_t* lock = with_lock ? &heap_allocator->lock : NULL;
        c_heap_free((void*) protocol, lock);
    }
    else {
        free((void*) protocol);
    }
}

static inline allocator_protocol* c_md_protocol_from_ptr(const void* ptr) {
    if (!ptr) return NULL;
    allocator_protocol* protocol = (allocator_protocol*) ((char*) ptr - offsetof(allocator_protocol, buf));
#if MD_ALLOC_VIGILANT > 0
    if (protocol->magic != MD_ALLOC_MAGIC) {
        fprintf(stderr, "[MD_ALLOC_VIGILANT] ERROR: Magic mismatch! Attempting to c_md_protocol_from_ptr a non-qk-allocated pointer!\n");
        fprintf(stderr, "[MD_ALLOC_VIGILANT] Expected magic: 0x%llx, Got: 0x%llx\n", (unsigned long long) MD_ALLOC_MAGIC, (unsigned long long) protocol->magic);
        fprintf(stderr, "[MD_ALLOC_VIGILANT] This is likely a regular malloc / calloc'd pointer or already freed!\n");
        fflush(stderr);
        abort();
    }
#endif
    return protocol;
}

// ========== Public APIs ==========

static inline void* c_md_alloc(size_t size, allocator_protocol* schematic) {
    allocator_protocol* clone;

    if (!schematic) {
        clone = (allocator_protocol*) calloc(1, sizeof(allocator_protocol) + size);
        if (!clone) return NULL;
        clone->size = size;

#if MD_ALLOC_VIGILANT > 0
        clone->magic = MD_ALLOC_MAGIC;
#endif

        return (void*) clone->buf;
    }

    if (schematic->with_shm) {
        shm_allocator_ctx* ctx = schematic->shm_allocator_ctx;
        shm_allocator* allocator = schematic->shm_allocator;
        pthread_mutex_t* lock = schematic->with_lock ? &allocator->lock : NULL;
        clone = (allocator_protocol*) c_shm_request(ctx, sizeof(allocator_protocol) + size, 0, lock);
        if (!clone) return NULL;
        clone->shm_allocator = allocator;
        clone->shm_allocator_ctx = ctx;
        clone->with_shm = 1;
    }
    else if (schematic->heap_allocator) {
        pthread_mutex_t* lock = schematic->with_lock ? &schematic->heap_allocator->lock : NULL;
        heap_allocator* allocator = schematic->heap_allocator;
        clone = (allocator_protocol*) c_heap_request(allocator, sizeof(allocator_protocol) + size, 0, lock);
        if (!clone) return NULL;
        clone->heap_allocator = allocator;
    }
    else {
        clone = (allocator_protocol*) calloc(1, sizeof(allocator_protocol) + size);
        if (!clone) return NULL;
    }

    clone->with_lock = schematic->with_lock;
    clone->size = size;

#if MD_ALLOC_VIGILANT > 0
    clone->magic = MD_ALLOC_MAGIC;
#endif

    return (void*) clone->buf;
}

static inline void c_md_free(void* ptr) {
    if (!ptr) return;
    allocator_protocol* protocol = c_md_protocol_from_ptr(ptr);

#if MD_ALLOC_VIGILANT > 0
    if (protocol->magic != MD_ALLOC_MAGIC) {
        fprintf(stderr, "[MD_ALLOC_VIGILANT] ERROR: Magic mismatch! Attempting to c_md_free a non-qk-allocated pointer!\n");
        fprintf(stderr, "[MD_ALLOC_VIGILANT] Expected magic: 0x%llx, Got: 0x%llx\n", (unsigned long long) MD_ALLOC_MAGIC, (unsigned long long) protocol->magic);
        fprintf(stderr, "[MD_ALLOC_VIGILANT] This is likely a regular malloc / calloc'd pointer or already freed!\n");
        fflush(stderr);
        abort();
    }
#endif

    c_ae_allocator_protocol_recycle(protocol, protocol->with_lock);
}

static inline char* c_md_strdup(const char* src, allocator_protocol* allocator) {
    if (!src) return NULL;
    size_t len = strlen(src);
    char* trg = (char*) c_md_alloc(len + 1, allocator);
    if (!trg) return NULL;
    memcpy(trg, src, len);
    return trg;
}

static inline void* c_md_realloc(void* src, size_t new_size, allocator_protocol* allocator) {
    if (!src) return c_md_alloc(new_size, allocator);
    if (new_size == 0) {
        c_md_free(src);
        return NULL;
    }

    allocator_protocol* protocol = c_md_protocol_from_ptr(src);
    size_t copy_size = protocol->size < new_size ? protocol->size : new_size;
    void* new_ptr = c_md_alloc(new_size, allocator);
    if (!new_ptr) return NULL;
    memcpy(new_ptr, src, copy_size);
    c_md_free(src);
    return new_ptr;
}

#endif /* C_AE_ALLOCATOR_PROTOCOL_H */
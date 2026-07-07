#ifndef C_INTERN_STRING_H
#define C_INTERN_STRING_H

#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <cbase/allocator_protocol/c_allocator_protocol.h>

// ========== Constants ==========

#define FNV_OFFSET_BASIS 14695981039346656037ULL
#define FNV_PRIME 1099511628211ULL

#ifndef ISTR_INITIAL_CAPACITY
#define ISTR_INITIAL_CAPACITY 4096
#endif

// ========== Intern String Pool Structs ==========

typedef struct istr_entry {
    const char*        internalized;
    uint64_t           hash;
    struct istr_entry* next;
} istr_entry;

typedef struct istr_map {
    pthread_mutex_t lock;
    istr_entry*     pool;
    size_t          capacity;
    size_t          size;
    istr_entry*     first;
} istr_map;

// ========== Forward Declarations ==========

/*
 * @brief FNV-1a hash function for strings.
 * @param key Input string.
 * @param key_length Length of the input string.
 * @return 64-bit hash value.
 */
static inline uint64_t fnv1a_hash(const char* key, size_t key_length);

/*
 * @brief Round up n to the next power of two.
 * @param n Input value.
 * @return Smallest power of two >= n, minimum 1.
 */
static inline size_t c_next_pow2(size_t n);

/*
 * @brief Create a new interned string map.
 * @param capacity Initial capacity of the map.
 * @param allocator Shared memory allocator context (optional).
 * @return Pointer to the newly created istr_map.
 */
static inline istr_map* c_istr_map_new(size_t capacity, allocator_protocol* allocator);

/*
 * @brief Free an interned string map and its contents.
 * @param map Pointer to the istr_map to free.
 */
static inline void c_istr_map_free(istr_map* map);

/*
 * @brief Extend the capacity of an interned string map.
 * @param map Pointer to the istr_map to extend.
 * @param new_capacity New capacity (0 to auto double).
 * @return 0 on success, -1 on failure.
 */
static inline int c_istr_map_extend(istr_map* map, size_t new_capacity);

/**
 * @brief Extend the capacity of an interned string map with locking.
 * @param map Pointer to the istr_map to extend.
 * @param new_capacity New capacity (0 to auto double).
 * @return 0 on success, -1 on failure.
 */
static inline int c_istr_map_extend_synced(istr_map* map, size_t new_capacity);

/*
 * @brief Lookup an interned string in the map without locking.
 * @param map Pointer to the istr_map.
 * @param key Input string to lookup.
 * @return Pointer to the istr_entry if found, or NULL if not found.
 */
static inline const istr_entry* c_istr_map_lookup(const istr_map* map, const char* key);

/*
 * @brief Lookup an interned string in the map with locking.
 * @param map Pointer to the istr_map.
 * @param key Input string to lookup.
 * @return Pointer to the istr_entry if found, or NULL if not found.
 */
static inline const istr_entry* c_istr_map_lookup_synced(const istr_map* map, const char* key);

/*
 * @brief Intern a string, returning a pointer to the internalized copy, lock free.
 * @param map Pointer to the istr_map.
 * @param key Input string to intern.
 * @return Pointer to the interned string, or NULL on failure.
 */
static inline const char* c_istr(istr_map* map, const char* key, const istr_entry** out_entry);

/*
 * @brief Intern a string, returning a pointer to the internalized copy, with locking.
 * @param map Pointer to the istr_map.
 * @param key Input string to intern.
 * @return Pointer to the interned string, or NULL on failure.
 */
static inline const char* c_istr_synced(istr_map* map, const char* key, const istr_entry** out_entry);

// ========== Utility Functions ==========

static inline uint64_t fnv1a_hash(const char* key, size_t key_length) {
    uint64_t hash = FNV_OFFSET_BASIS;
    for (size_t i = 0; i < key_length; ++i) {
        hash ^= (uint8_t) key[i];
        hash *= FNV_PRIME;
    }
    return hash;
}

static inline size_t c_next_pow2(size_t n) {
    if (n == 0) return 1;
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
#if __SIZEOF_SIZE_T__ >= 2
    n |= n >> 8;
#endif
#if __SIZEOF_SIZE_T__ >= 4
    n |= n >> 16;
#endif
#if __SIZEOF_SIZE_T__ >= 8
    n |= n >> 32;
#endif
#if __SIZEOF_SIZE_T__ >= 16
    n |= n >> 64;
#endif
    n++;
    return n;
}

// ========== Public APIs ==========

static inline istr_map* c_istr_map_new(size_t capacity, allocator_protocol* allocator) {
    capacity = (capacity == 0) ? ISTR_INITIAL_CAPACITY : c_next_pow2(capacity);

    istr_map* map = c_ap_alloc(sizeof(istr_map), allocator);
    if (!map) return NULL;

    istr_entry* pool = c_ap_alloc(capacity * sizeof(istr_entry), allocator);
    if (!pool) {
        c_ap_free(map);
        return NULL;
    }

    if (pthread_mutex_init(&map->lock, NULL) != 0) {
        c_ap_free(pool);
        c_ap_free(map);
        return NULL;
    }

    map->capacity = capacity;
    map->pool = pool;

    return map;
}

static inline void c_istr_map_free(istr_map* map) {
    if (!map) return;

    // Free interned strings before releasing the pool
    istr_entry* it = map->first;
    while (it) {
        if (it->internalized) {
            c_ap_free((void*) it->internalized);
        }
        it = it->next;
    }

    pthread_mutex_destroy(&map->lock);

    c_ap_free((void*) map->pool);
    c_ap_free((void*) map);
}

static inline int c_istr_map_extend(istr_map* map, size_t new_capacity) {
    if (!map) return -1;

    // Step 1: Determine new capacity
    if (!new_capacity) {
        if (map->capacity >= SIZE_MAX / 2) return -1;
        else if (map->capacity == 0) {
            new_capacity = ISTR_INITIAL_CAPACITY;
        }
        else {
            new_capacity = map->capacity * 2;
        }
    }
    else {
        new_capacity = c_next_pow2(new_capacity);
        if (new_capacity <= map->capacity) return -1;
    }

    allocator_protocol* allocator = c_ap_protocol_from_ptr(map);
    istr_entry*         new_pool;

    // Step 2: Allocate new pool
    new_pool = (istr_entry*) c_ap_alloc(new_capacity * sizeof(istr_entry), allocator);
    if (!new_pool) return -1;

    // Step 3: Rehash existing entries into new pool
    istr_entry* current = map->first;
    map->first = NULL;
    size_t mask = new_capacity - 1;

    while (current) {
        istr_entry* next = current->next;
        uint64_t    idx = current->hash & mask;
        istr_entry* entry = new_pool + idx;

        // Linear probing for collision resolution
        // There always should be at least one free slot
        while (entry->internalized) {
            if (++idx == new_capacity) idx = 0;
            entry = new_pool + idx;
        }

        entry->internalized = current->internalized;
        entry->hash = current->hash;
        entry->next = map->first;
        map->first = entry;

        current = next;
    }

    // Step 4: Free old pool and update map
    c_ap_free((void*) map->pool);

    map->pool = new_pool;
    map->capacity = new_capacity;
    return 0;
}

static inline int c_istr_map_extend_synced(istr_map* map, size_t new_capacity) {
    if (!map) return -1;
    pthread_mutex_t* lock = &map->lock;
    pthread_mutex_lock(lock);
    int result = c_istr_map_extend(map, new_capacity);
    pthread_mutex_unlock(lock);
    return result;
}

static inline const istr_entry* c_istr_map_lookup(const istr_map* map, const char* key) {
    if (!map || !map->capacity || !key) return NULL;

    // Step 1: Compute hash and index
    size_t      key_length = strlen(key);
    uint64_t    hash = fnv1a_hash(key, key_length);
    size_t      capacity = map->capacity;
    uint64_t    idx = hash & (capacity - 1);
    istr_entry* pool = map->pool;
    istr_entry* entry = pool + idx;

    // Step 2: Search for existing entry
    while (entry->internalized) {
        if (entry->hash == hash && strcmp(entry->internalized, key) == 0) {
            return entry;
        }
        if (++idx == capacity) idx = 0;
        entry = pool + idx;
    }

    return NULL;
}

static inline const istr_entry* c_istr_map_lookup_synced(const istr_map* map, const char* key) {
    pthread_mutex_t* lock = (pthread_mutex_t*) &map->lock;
    pthread_mutex_lock(lock);
    const istr_entry* out = c_istr_map_lookup(map, key);
    pthread_mutex_unlock(lock);
    return out;
}

static inline const char* c_istr(istr_map* map, const char* key, const istr_entry** out_entry) {
    if (!map || !map->capacity || !key) return NULL;

    // Step 1: Compute hash and index
    size_t   key_length = strlen(key);
    uint64_t hash = fnv1a_hash(key, key_length);

    // Step 2: Check if key already exists
    size_t      capacity = map->capacity;
    uint64_t    idx = hash & (capacity - 1);
    istr_entry* pool = map->pool;
    istr_entry* entry = pool + idx;

    while (entry->internalized) {
        if (entry->hash == hash && strcmp(entry->internalized, key) == 0) {
            if (out_entry) *out_entry = entry;
            return entry->internalized;
        }
        if (++idx == capacity) idx = 0;
        entry = pool + idx;
    }

    // Step 3: Check capacity and extend if necessary
    if (map->size >= capacity / 2) {
        if (c_istr_map_extend(map, 0) != 0) return NULL;
        return c_istr(map, key, out_entry);
    }

    // Step 4: Duplicate key
    char*               interned_copy = NULL;
    size_t              total_size = key_length + 1;
    allocator_protocol* allocator = c_ap_protocol_from_ptr(map);

    interned_copy = (char*) c_ap_alloc(total_size, allocator);
    if (!interned_copy) return NULL;
    memcpy(interned_copy, key, total_size);

    // Step 5: Insert new entry
    entry->internalized = interned_copy;
    entry->hash = hash;
    entry->next = map->first;
    map->first = entry;
    map->size += 1;
    if (out_entry) *out_entry = entry;
    return interned_copy;
}

static inline const char* c_istr_synced(istr_map* map, const char* key, const istr_entry** out_entry) {
    pthread_mutex_t* lock = &map->lock;
    pthread_mutex_lock(lock);
    const char* out = c_istr(map, key, out_entry);
    pthread_mutex_unlock(lock);
    return out;
}

#endif  // C_INTERN_STRING_H
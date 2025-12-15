#ifndef C_INTERN_STRING_H
#define C_INTERN_STRING_H

#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "c_shm_allocator.h"

// ========== Constants ==========

#define FNV_OFFSET_BASIS 14695981039346656037ULL
#define FNV_PRIME 1099511628211ULL

#ifndef ISTR_INITIAL_CAPACITY
#define ISTR_INITIAL_CAPACITY 4096
#endif

// ========== Intern String Pool Structs ==========

typedef struct istr_entry {
    char* internalized;
    uint64_t hash;
    struct istr_entry* next;
} istr_entry;

typedef struct istr_map {
    shm_allocator_ctx* allocator;
    size_t capacity;
    size_t size;
    istr_entry* first;
    istr_entry* pool;
    pthread_mutex_t* lock;      // points to allocator lock if present, else local_lock
    pthread_mutex_t  local_lock;
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
* @brief Create a new interned string map.
* @param capacity Initial capacity of the map.
* @param allocator Shared memory allocator context (optional).
* @return Pointer to the newly created istr_map.
*/
static inline istr_map* c_istr_map_new(size_t capacity, shm_allocator_ctx* allocator);

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

/*
* @brief Lookup an interned string in the map without locking.
* @param map Pointer to the istr_map.
* @param key Input string to lookup.
* @return Pointer to the istr_entry if found, or NULL if not found.
*/
static inline const istr_entry* c_istr_map_lookup(istr_map* map, const char* key);

/*
* @brief Lookup an interned string in the map with locking.
* @param map Pointer to the istr_map.
* @param key Input string to lookup.
* @return Pointer to the istr_entry if found, or NULL if not found.
*/
static inline const istr_entry* c_istr_map_lookup_synced(istr_map* map, const char* key);

/*
* @brief Intern a string, returning a pointer to the internalized copy, lock free.
* @param map Pointer to the istr_map.
* @param key Input string to intern.
* @return Pointer to the interned string, or NULL on failure.
*/
static inline const char* c_istr(istr_map* map, const char* key);

/*
* @brief Intern a string, returning a pointer to the internalized copy, with locking.
* @param map Pointer to the istr_map.
* @param key Input string to intern.
* @return Pointer to the interned string, or NULL on failure.
*/
static inline const char* c_istr_synced(istr_map* map, const char* key);

// ========== Utility Functions ==========

static inline uint64_t fnv1a_hash(const char* key, size_t key_length) {
    uint64_t hash = FNV_OFFSET_BASIS;
    for (size_t i = 0; i < key_length; ++i) {
        hash ^= (uint8_t) key[i];
        hash *= FNV_PRIME;
    }
    return hash;
}

// ========== Public APIs ==========

static inline istr_map* c_istr_map_new(size_t capacity, shm_allocator_ctx* allocator) {
    if (capacity == 0) capacity = ISTR_INITIAL_CAPACITY;

    istr_map* map;
    istr_entry* pool;
    if (allocator) {
        map = (istr_map*) c_shm_request(allocator, sizeof(istr_map), 0, &allocator->shm_allocator->lock);
        if (!map) return NULL;

        pool = (istr_entry*) c_shm_request(allocator, capacity * sizeof(istr_entry), 0, &allocator->shm_allocator->lock);
        if (!pool) {
            c_shm_free((void*) map, &allocator->shm_allocator->lock);
            return NULL;
        }
    }
    else {
        map = (istr_map*) calloc(1, sizeof(istr_map));
        if (!map) return NULL;

        pool = (istr_entry*) calloc(capacity, sizeof(istr_entry));
        if (!pool) {
            free(map);
            return NULL;
        }
    }

    map->allocator = allocator;
    map->capacity = capacity;
    map->size = 0;
    map->first = NULL;
    map->pool = pool;
    if (allocator) {
        map->lock = &allocator->shm_allocator->lock;
    }
    else {
        map->lock = &map->local_lock;
        if (pthread_mutex_init(&map->local_lock, NULL) != 0) {
            free(pool);
            free(map);
            return NULL;
        }
    }
    return map;
}

static inline void c_istr_map_free(istr_map* map) {
    if (!map) return;

    shm_allocator_ctx* allocator = map->allocator;

    // Free interned strings before releasing the pool
    istr_entry* it = map->first;
    while (it) {
        if (it->internalized) {
            if (allocator) {
                c_shm_free((void*) it->internalized, &allocator->shm_allocator->lock);
            }
            else {
                free(it->internalized);
            }
        }
        it = it->next;
    }

    if (!allocator) {
        pthread_mutex_destroy(&map->local_lock);
    }

    if (allocator) {
        c_shm_free((void*) map->pool, &allocator->shm_allocator->lock);
        c_shm_free((void*) map, &allocator->shm_allocator->lock);
    }
    else {
        free(map->pool);
        free(map);
    }
}

static inline int c_istr_map_extend(istr_map* map, size_t new_capacity) {
    if (!map) return -1;
    pthread_mutex_t* lock = map->lock;
    pthread_mutex_lock(lock);

    // Step 1: Determine new capacity
    if (!new_capacity) {
        if (map->capacity >= SIZE_MAX / 2) goto fail_and_return;
        else if (map->capacity == 0) {
            new_capacity = ISTR_INITIAL_CAPACITY;
        }
        else {
            new_capacity = map->capacity * 2;
        }
    }
    else if (new_capacity <= map->capacity) goto fail_and_return;

    shm_allocator_ctx* allocator = map->allocator;
    istr_entry* new_pool;

    // Step 2: Allocate new pool
    if (allocator) {
        new_pool = (istr_entry*) c_shm_request(allocator, new_capacity * sizeof(istr_entry), 0, NULL);
        if (!new_pool) goto fail_and_return;
    }
    else {
        new_pool = (istr_entry*) calloc(new_capacity, sizeof(istr_entry));
        if (!new_pool) goto fail_and_return;
    }

    // Step 3: Rehash existing entries into new pool
    istr_entry* current = map->first;
    map->first = NULL;

    while (current) {
        istr_entry* next = current->next;
        uint64_t idx = current->hash % new_capacity;
        istr_entry* entry = new_pool + idx;

        // Linear probing for collision resolution
        // There always should be at least one free slot
        while (entry->internalized) {
            entry++;
            if (entry == new_pool + new_capacity) entry = new_pool;
        }

        entry->internalized = current->internalized;
        entry->hash = current->hash;
        entry->next = map->first;
        map->first = entry;

        // Update current pointer
        current = next;
    }

    // Step 4: Free old pool and update map
    if (allocator) {
        c_shm_free((void*) map->pool, &allocator->shm_allocator->lock);
    }
    else {
        free(map->pool);
    }

    map->pool = new_pool;
    map->capacity = new_capacity;
    pthread_mutex_unlock(lock);
    return 0;

fail_and_return:
    pthread_mutex_unlock(lock);
    return -1;
}

static inline const istr_entry* c_istr_map_lookup(istr_map* map, const char* key) {
    if (!map || !map->capacity || !key) return NULL;

    // Step 1: Compute hash and index
    size_t key_length = strlen(key);
    uint64_t hash = fnv1a_hash(key, key_length);
    size_t capacity = map->capacity;
    uint64_t idx = hash % capacity;
    istr_entry* pool = map->pool;
    istr_entry* entry = pool + idx;

    // Step 2: Search for existing entry
    while (entry->internalized) {
        if (strcmp(entry->internalized, key) == 0) {
            return entry;
        }
        idx++;
        if (idx == capacity) idx = 0;
        entry = pool + idx;
    }

    return NULL;
}

static inline const istr_entry* c_istr_map_lookup_synced(istr_map* map, const char* key) {
    if (!map || !map->capacity || !key) return NULL;

    pthread_mutex_t* lock = map->lock;
    pthread_mutex_lock(lock);

    // Step 1: Compute hash and index
    size_t key_length = strlen(key);
    uint64_t hash = fnv1a_hash(key, key_length);
    size_t capacity = map->capacity;
    uint64_t idx = hash % capacity;
    istr_entry* pool = map->pool;
    istr_entry* entry = pool + idx;

    // Step 2: Search for existing entry
    while (entry->internalized) {
        if (strcmp(entry->internalized, key) == 0) {
            pthread_mutex_unlock(lock);
            return entry;
        }
        idx++;
        if (idx == capacity) idx = 0;
        entry = pool + idx;
    }

    pthread_mutex_unlock(lock);
    return NULL;
}

static inline const char* c_istr(istr_map* map, const char* key) {
    if (!map || !map->capacity || !key) return NULL;

    // Step 1: Compute hash and index
    size_t key_length = strlen(key);
    uint64_t hash = fnv1a_hash(key, key_length);

    // Step 2: Check if key already exists
    size_t capacity = map->capacity;
    uint64_t idx = hash % capacity;
    istr_entry* pool = map->pool;
    istr_entry* entry = pool + idx;

    while (entry->internalized) {
        if (strcmp(entry->internalized, key) == 0) {
            return entry->internalized;
        }
        idx++;
        if (idx == capacity) idx = 0;
        entry = pool + idx;
    }

    // Step 3: Check capacity and extend if necessary
    if (map->size >= capacity / 2) {
        if (c_istr_map_extend(map, 0) != 0) return NULL;
        return c_istr(map, key);
    }

    // Step 4: Duplicate key
    char* interned_copy = NULL;
    size_t total_size = key_length + 1;
    shm_allocator_ctx* allocator = map->allocator;

    if (allocator) {
        interned_copy = (char*) c_shm_request(allocator, total_size, 1, &allocator->shm_allocator->lock);
        if (!interned_copy) return NULL;
    }
    else {
        interned_copy = (char*) calloc(1, total_size);
        if (!interned_copy) return NULL;
    }
    memcpy(interned_copy, key, total_size);

    // Step 5: Insert new entry
    entry->internalized = interned_copy;
    entry->hash = hash;
    entry->next = map->first;
    map->first = entry;
    map->size += 1;
    return interned_copy;
}

static inline const char* c_istr_synced(istr_map* map, const char* key) {
    if (!map || !map->capacity || !key) return NULL;

    // Step 1: Compute hash and index
    size_t key_length = strlen(key);
    uint64_t hash = fnv1a_hash(key, key_length);

    char* interned_copy = NULL;
    pthread_mutex_t* lock = map->lock;
    pthread_mutex_lock(lock);

    // Step 2: Check if key already exists
    size_t capacity = map->capacity;
    uint64_t idx = hash % capacity;
    istr_entry* pool = map->pool;
    istr_entry* entry = pool + idx;

    while (entry->internalized) {
        if (strcmp(entry->internalized, key) == 0) {
            interned_copy = entry->internalized;
            goto unlock_and_return;
        }
        idx++;
        if (idx == capacity) idx = 0;
        entry = pool + idx;
    }

    // Step 3: Check capacity and extend if necessary
    if (map->size >= capacity / 2) {
        pthread_mutex_unlock(lock);
        if (c_istr_map_extend(map, 0) != 0) return NULL;
        return c_istr_synced(map, key);
    }

    // Step 4: Duplicate key
    size_t total_size = key_length + 1;
    shm_allocator_ctx* allocator = map->allocator;

    if (allocator) {
        interned_copy = (char*) c_shm_request(allocator, total_size, 1, NULL);
        if (!interned_copy) goto unlock_and_return;
    }
    else {
        interned_copy = (char*) calloc(1, total_size);
        if (!interned_copy) goto unlock_and_return;
    }
    memcpy(interned_copy, key, total_size);

    // Step 5: Insert new entry, no need for reprobing as we hold the lock
    entry->internalized = interned_copy;
    entry->hash = hash;
    entry->next = map->first;
    map->first = entry;
    map->size += 1;

unlock_and_return:
    pthread_mutex_unlock(lock);
    return interned_copy;
}

#endif // C_INTERN_STRING_H
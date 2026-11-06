#ifndef C_MARKET_ENGINE_H
#define C_MARKET_ENGINE_H

#include <math.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include <algo_engine/base/c_market_data/c_market_data.h>
#include <algo_engine/exchange_profile/c_ex_profile_base.h>
#include <cbase/allocator_protocol/c_allocator_protocol.h>
#include <cbase/bytemap/c_bytemap.h>
#include <cbase/bytemap/xxh3.h>

// ========== Enums ==========

// clang-format off

/** @enum mds_ret_code
 *  @brief 0 is success, positives are success-with-status, negatives are errors. */
typedef enum mds_ret_code {
    MDS_OK_SUBSCRIBED     =  1,  // The feed registered a ticker for the first time.
    MDS_OK                =  0,
    MDS_ERR_INVALID_INPUT = -1,
    MDS_ERR_UNKNOWN_DTYPE = -2,
    MDS_ERR_INTERNAL_DATA = -3,
    MDS_ERR_OOM           = -4,
    MDS_ERR_NOT_FOUND     = -5,
} mds_ret_code;

// clang-format on

// ========== Structs ==========

// clang-format off

typedef unsigned char mds_monitor_id[16];

/**
 * @struct mds_monitor
 * @brief The generic monitor base every monitor derives from.
 *
 * These are the monitor fields that are not factor-specific: identity, status
 * and the generic market-data filter. A derived monitor (Quark's fct_monitor)
 * embeds this struct at offset 0, so a `mds_monitor*` can be recovered from the
 * derived pointer with a cast and the monitor can be registered in the MDS
 * registry without a wrapper.
 *
 * `name` is NOT owned: the owner keeps the string alive for as long as the
 * monitor may be registered.
 */
typedef struct mds_monitor {
    int16_t        priority;    // Dispatch order hint (lower first).
    const char*    name;        // OWNED - strdup'ed by init, freed by dealloc.
    mds_monitor_id monitor_id;  // 16-byte identity used by the registries.
    md_filter_flag md_filter;   // Generic market-data filter.
    bool           enabled;     // false skips the monitor entirely.
    bool           is_ready;    // false means the monitor has no output yet.
} mds_monitor;

/**
 * @struct mds_subscription
 * @brief Per-ticker state the service stores in its subscription bytemap.
 *
 * The ticker is the map key, so the struct carries no name of its own. The
 * whole struct is the map value: capacity and growth are the bytemap's
 * defaults (capacity 0 asks for them), never a second capacity knob here.
 * `last_update` is the market data's own session datetime, never a converted
 * double.
 */
typedef struct mds_subscription {
    session_datetime_t last_update;            // Session datetime of the latest observation.
    double             last_price;             // Price resolved through c_md_get_price.
    size_t             n_feeds;                // Number of observations routed to the ticker.
    bool               is_subscribed_td;       // A transaction (DTYPE_TRANSACTION) was seen.
    bool               is_subscribed_od;       // An order (DTYPE_ORDER) was seen.
    bool               is_subscribed_tk_lite;  // A lite tick (DTYPE_TICK_LITE) was seen.
    bool               is_subscribed_tk;       // A full tick (DTYPE_TICK) was seen.
    bool               is_subscribed_cs;       // A bar (DTYPE_BAR) was seen.
} mds_subscription;

typedef struct mds_monitor_manager mds_monitor_manager;

/**
 * @brief Callback invoked when an observation reaches a monitor manager.
 *
 * The manager is the base struct itself. A derived manager embeds
 * `mds_monitor_manager` as its FIRST member, so the callback recovers the
 * derived pointer with a plain cast. `user_data` is the value registered with
 * the callback and is always the last parameter.
 */
typedef void (*mds_monitor_manager_callback_func)(mds_monitor_manager* manager, const md_variant* market_data, void* user_data);

/**
 * @struct mds_monitor_manager_callback_ctx
 * @brief One registered feed callback (a linked-list node).
 *
 * Nodes are small, localized registration state: they are `calloc`/`free`d,
 * never allocated through the allocator protocol. `id` is the opaque handle
 * `c_mds_monitor_manager_unregister_callback` takes.
 */
typedef struct mds_monitor_manager_callback_ctx {
    mds_monitor_manager_callback_func        fn;
    void*                                    user_data;
    uintptr_t                                id;
    struct mds_monitor_manager_callback_ctx* next;
} mds_monitor_manager_callback_ctx;

/**
 * @struct mds_monitor_manager
 * @brief The generic monitor manager base every manager derives from.
 *
 * It owns the monitor registry (16-byte monitor id -> borrowed `mds_monitor*`)
 * and the observer callback list, and carries the generic enabled flag. The
 * Python face binds a map view to this registry; dispatch itself still runs
 * through the callbacks, which is what lets a derived manager (Quark) run its
 * own context/worker pipeline.
 *
 * Callbacks are bound only through the register/unregister pair -- never
 * through `_new`/`_init`, which take the allocator only.
 */
struct mds_monitor_manager {
    bytemap*                          monitors;   // OWNED - 16-byte id -> mds_monitor*.
    mds_monitor_manager_callback_ctx* callbacks;  // OWNED - calloc'ed observer nodes.
    _Atomic bool                      enabled;    // Atomic: start/stop from the main process.
};

/**
 * @struct mds
 * @brief The core market data service state.
 *
 * `mds` is the single C-side state of the service: the subscription bytemap,
 * the session datetime of the latest observation (the market data's own
 * `md_meta.dt`, never a converted double), the borrowed monitor manager, and
 * the monitor registry the Python face maps.
 */
typedef struct mds {
    mds_monitor_manager* monitor_manager;  // NOT owned - set with c_mds_set_manager.
    bytemap*             subscriptions;    // OWNED - ticker -> mds_subscription value.
    bytemap*             monitors;         // OWNED - 16-byte monitor id -> mds_monitor*.
    session_datetime_t   ts;               // Zeroed date = no observation yet.
} mds;

// clang-format on

// ========== Forward Declarations ==========

static inline mds_monitor*         c_mds_monitor_new(const char* name, int16_t priority, md_filter_flag md_filter, allocator_protocol* allocator);
static inline int                  c_mds_monitor_init(mds_monitor* monitor, const char* name, int16_t priority, md_filter_flag md_filter, allocator_protocol* allocator);
static inline void                 c_mds_monitor_dealloc(mds_monitor* monitor);
static inline void                 c_mds_monitor_free(mds_monitor* monitor);
static inline void                 c_mds_monitor_gen_uid(mds_monitor* monitor, const char* salt, size_t salt_len);
static inline bool                 c_mds_monitor_filter_md(const mds_monitor* monitor, const md_variant* market_data);
static inline void                 c_mds_monitor_set_enabled(mds_monitor* monitor, bool enabled);
static inline bool                 c_mds_monitor_is_enabled(const mds_monitor* monitor);

static inline mds_monitor_manager* c_mds_monitor_manager_new(allocator_protocol* allocator);
static inline int                  c_mds_monitor_manager_init(mds_monitor_manager* manager, allocator_protocol* allocator);
static inline void                 c_mds_monitor_manager_dealloc(mds_monitor_manager* manager);
static inline void                 c_mds_monitor_manager_free(mds_monitor_manager* manager);
static inline int                  c_mds_monitor_manager_register_callback(mds_monitor_manager* manager, mds_monitor_manager_callback_func callback, void* user_data, uintptr_t* out_id);
static inline int                  c_mds_monitor_manager_unregister_callback(mds_monitor_manager* manager, uintptr_t callback_id);
static inline void                 c_mds_monitor_manager_feed(mds_monitor_manager* manager, const md_variant* market_data);
static inline int                  c_mds_monitor_manager_register_monitor(mds_monitor_manager* manager, mds_monitor* monitor);
static inline int                  c_mds_monitor_manager_unregister_monitor(mds_monitor_manager* manager, mds_monitor* monitor);
static inline mds_monitor*         c_mds_monitor_manager_get_monitor(const mds_monitor_manager* manager, const mds_monitor_id monitor_id);
static inline int                  c_mds_monitor_manager_clear_monitors(mds_monitor_manager* manager);
static inline void                 c_mds_monitor_manager_set_enabled(mds_monitor_manager* manager, bool enabled);
static inline bool                 c_mds_monitor_manager_is_enabled(const mds_monitor_manager* manager);

static inline mds*                 c_mds_new(allocator_protocol* allocator);
static inline int                  c_mds_init(mds* service, allocator_protocol* allocator);
static inline void                 c_mds_dealloc(mds* service);
static inline void                 c_mds_free(mds* service);
static inline int                  c_mds_clear(mds* service);
static inline int                  c_mds_set_manager(mds* service, mds_monitor_manager* manager);
static inline mds_monitor_manager* c_mds_get_manager(const mds* service);
static inline int                  c_mds_register_monitor(mds* service, mds_monitor* monitor);
static inline int                  c_mds_unregister_monitor(mds* service, mds_monitor* monitor);
static inline mds_monitor*         c_mds_get_monitor(const mds* service, const mds_monitor_id monitor_id);
static inline size_t               c_mds_n_monitors(const mds* service);
static inline int                  c_mds_update_subscription(mds* service, const md_variant* market_data);
static inline int                  c_mds_on_market_data(mds* service, const md_variant* market_data);
static inline int                  c_mds_on_internal_data(mds* service, const md_internal* internal_data);
static inline double               c_mds_get_market_price(const mds* service, const char* ticker);

// ========== Public APIs (Monitor Base) ==========

/**
 * @brief Allocate a monitor base, copy its name and set the generic fields.
 * @param name The monitor name (copied; NULL leaves the name NULL).
 * @param priority The dispatch priority.
 * @param md_filter The generic market-data filter.
 * @param allocator Allocator for the block (NULL falls back to the process default).
 * @return The new monitor, or NULL when the block or the name cannot be allocated.
 */
static inline mds_monitor* c_mds_monitor_new(const char* name, int16_t priority, md_filter_flag md_filter, allocator_protocol* allocator) {
    mds_monitor* monitor = (mds_monitor*) c_ap_alloc(sizeof(mds_monitor), allocator);
    if (!monitor) return NULL;

    if (c_mds_monitor_init(monitor, name, priority, md_filter, allocator) != MDS_OK) {
        c_ap_free(monitor);
        return NULL;
    }

    return monitor;
}

/**
 * @brief Initialize an allocator-backed, zeroed monitor base.
 *
 * The name is duplicated into the monitor, parented to the monitor block, and
 * released with the block by `c_mds_monitor_free` (or explicitly by
 * `c_mds_monitor_dealloc`). A fresh monitor is enabled and ready. The monitor
 * must be an allocator block start: a derived monitor embeds it at offset 0,
 * which is the derived block's start.
 *
 * @param monitor The zeroed monitor block to initialize.
 * @param name The monitor name (copied; NULL leaves the name NULL).
 * @param priority The dispatch priority.
 * @param md_filter The generic market-data filter.
 * @param allocator Allocator for the name copy; NULL derives it from the monitor block (a block start).
 * @return MDS_OK, MDS_ERR_INVALID_INPUT when monitor is NULL, or MDS_ERR_OOM.
 */
static inline int c_mds_monitor_init(mds_monitor* monitor, const char* name, int16_t priority, md_filter_flag md_filter, allocator_protocol* allocator) {
    if (!monitor) return MDS_ERR_INVALID_INPUT;
    if (!allocator) allocator = c_ap_protocol_from_ptr(monitor);

    memset(monitor, 0, sizeof(mds_monitor));

    if (name) {
        monitor->name = c_ap_strdup(name, allocator);
        if (!monitor->name) return MDS_ERR_OOM;
        c_ap_acquire_ownership((const void*) monitor->name, (const void*) monitor);
    }

    monitor->priority = priority;
    monitor->md_filter = md_filter;
    monitor->enabled = true;
    monitor->is_ready = true;
    return MDS_OK;
}

/**
 * @brief Tear down a caller-allocated monitor base, releasing the name and leaving it zeroed.
 * @param monitor The monitor (NULL-safe).
 */
static inline void c_mds_monitor_dealloc(mds_monitor* monitor) {
    if (!monitor) return;

    if (monitor->name) c_ap_free((void*) monitor->name);
    memset(monitor, 0, sizeof(mds_monitor));
}

/**
 * @brief Free a monitor allocated by `c_mds_monitor_new`.
 * @param monitor The monitor (NULL-safe).
 */
static inline void c_mds_monitor_free(mds_monitor* monitor) {
    if (!monitor) return;

    c_mds_monitor_dealloc(monitor);
    c_ap_free_owned(monitor);
}

/**
 * @brief Generate a stable 16-byte monitor id from the block address and a salt.
 *
 * The salt is any caller-chosen string (Quark passes fct_prefix; the Python
 * proxy passes the monitor name). The result is laid out as a UUID (version 4 /
 * RFC variant).
 *
 * @param monitor The monitor whose id is written (NULL-safe).
 * @param salt Optional salt bytes.
 * @param salt_len Length of salt.
 */
static inline void c_mds_monitor_gen_uid(mds_monitor* monitor, const char* salt, size_t salt_len) {
    if (!monitor) return;

    XXH3_state_t state;
    XXH3_128bits_reset(&state);

    uintptr_t address = (uintptr_t) monitor;
    XXH3_128bits_update(&state, &address, sizeof(address));
    if (salt && salt_len) XXH3_128bits_update(&state, salt, salt_len);

    XXH128_hash_t hash = XXH3_128bits_digest(&state);
    memcpy(monitor->monitor_id, &hash.low64, 8);
    memcpy(monitor->monitor_id + 8, &hash.high64, 8);

    monitor->monitor_id[6] = (monitor->monitor_id[6] & 0x0F) | 0x40;
    monitor->monitor_id[8] = (monitor->monitor_id[8] & 0x3F) | 0x80;
}

/**
 * @brief Apply the generic market-data filter.
 * @param monitor The monitor (NULL-safe).
 * @param market_data The observation (NULL-safe).
 * @return true when the observation passes the monitor's md_filter.
 */
static inline bool c_mds_monitor_filter_md(const mds_monitor* monitor, const md_variant* market_data) {
    if (!monitor || !market_data) return false;

    return c_md_filter(market_data, monitor->md_filter);
}

/**
 * @brief Set the monitor's enabled flag.
 * @param monitor The monitor (NULL-safe).
 * @param enabled The new flag.
 */
static inline void c_mds_monitor_set_enabled(mds_monitor* monitor, bool enabled) {
    if (!monitor) return;

    monitor->enabled = enabled;
}

/**
 * @brief Read the monitor's enabled flag.
 * @param monitor The monitor (NULL-safe).
 * @return The flag, or false when monitor is NULL.
 */
static inline bool c_mds_monitor_is_enabled(const mds_monitor* monitor) {
    return monitor ? monitor->enabled : false;
}

// ========== Public APIs (Monitor Manager Base) ==========

/**
 * @brief Allocate a monitor manager base.
 *
 * The constructor takes only the allocator: callbacks and monitors are
 * registered afterwards. A fresh manager is enabled and has an empty registry.
 *
 * @param allocator Allocator for the block and its children.
 * @return The new manager, or NULL on allocation failure.
 */
static inline mds_monitor_manager* c_mds_monitor_manager_new(allocator_protocol* allocator) {
    mds_monitor_manager* manager = (mds_monitor_manager*) c_ap_alloc(sizeof(mds_monitor_manager), allocator);
    if (!manager) return NULL;

    int ret_code = c_mds_monitor_manager_init(manager, allocator);
    if (ret_code != MDS_OK) {
        c_ap_free(manager);
        return NULL;
    }

    return manager;
}

/**
 * @brief Initialize a caller-allocated, zeroed monitor manager base.
 * @param manager The zeroed manager to initialize.
 * @param allocator Allocator for the children; derived from manager when NULL.
 * @return MDS_OK, MDS_ERR_INVALID_INPUT, or MDS_ERR_OOM.
 */
static inline int c_mds_monitor_manager_init(mds_monitor_manager* manager, allocator_protocol* allocator) {
    if (!manager) return MDS_ERR_INVALID_INPUT;
    if (!allocator) allocator = c_ap_protocol_from_ptr(manager);

    manager->monitors = c_bytemap_ex_new(0, sizeof(mds_monitor*), allocator);
    if (!manager->monitors) return MDS_ERR_OOM;

    atomic_init(&manager->enabled, true);
    return MDS_OK;
}

/**
 * @brief Tear down a caller-allocated manager base, leaving it zeroed.
 *
 * The monitor registry and every callback node are released; the monitors
 * themselves are borrowed and never freed here.
 *
 * @param manager The manager (NULL-safe).
 */
static inline void c_mds_monitor_manager_dealloc(mds_monitor_manager* manager) {
    if (!manager) return;

    mds_monitor_manager_callback_ctx* callback = manager->callbacks;
    while (callback) {
        mds_monitor_manager_callback_ctx* next = callback->next;
        free(callback);
        callback = next;
    }

    if (manager->monitors) {
        c_bytemap_ex_free(manager->monitors);
        manager->monitors = NULL;
    }

    memset(manager, 0, sizeof(mds_monitor_manager));
}

/**
 * @brief Free a manager allocated by `c_mds_monitor_manager_new`.
 * @param manager The manager (NULL-safe).
 */
static inline void c_mds_monitor_manager_free(mds_monitor_manager* manager) {
    if (!manager) return;

    c_mds_monitor_manager_dealloc(manager);
    c_ap_free(manager);
}

/**
 * @brief Register a feed callback (observer).
 * @param manager The manager (NULL-safe).
 * @param callback The callback to invoke (NULL is refused).
 * @param user_data Opaque value handed to the callback.
 * @param out_id Optional: receives the id used to unregister.
 * @return MDS_OK, MDS_ERR_INVALID_INPUT, or MDS_ERR_OOM.
 */
static inline int c_mds_monitor_manager_register_callback(mds_monitor_manager* manager, mds_monitor_manager_callback_func callback, void* user_data, uintptr_t* out_id) {
    if (out_id) *out_id = 0;
    if (!manager || !callback) return MDS_ERR_INVALID_INPUT;

    mds_monitor_manager_callback_ctx* node = (mds_monitor_manager_callback_ctx*) calloc(1, sizeof(mds_monitor_manager_callback_ctx));
    if (!node) return MDS_ERR_OOM;

    node->fn = callback;
    node->user_data = user_data;
    node->id = (uintptr_t) node;

    if (!manager->callbacks) {
        manager->callbacks = node;
    }
    else {
        mds_monitor_manager_callback_ctx* tail = manager->callbacks;
        while (tail->next) tail = tail->next;
        tail->next = node;
    }

    if (out_id) *out_id = node->id;
    return MDS_OK;
}

/**
 * @brief Unregister a feed callback by the id registration returned.
 * @param manager The manager (NULL-safe).
 * @param callback_id The id from registration (0 is refused).
 * @return MDS_OK, MDS_ERR_INVALID_INPUT, or MDS_ERR_NOT_FOUND.
 */
static inline int c_mds_monitor_manager_unregister_callback(mds_monitor_manager* manager, uintptr_t callback_id) {
    if (!manager || !callback_id) return MDS_ERR_INVALID_INPUT;

    mds_monitor_manager_callback_ctx* previous = NULL;
    mds_monitor_manager_callback_ctx* current = manager->callbacks;

    while (current) {
        if (current->id == callback_id) {
            if (previous) previous->next = current->next;
            else manager->callbacks = current->next;
            free(current);
            return MDS_OK;
        }

        previous = current;
        current = current->next;
    }

    return MDS_ERR_NOT_FOUND;
}

/**
 * @brief Hand one observation to every registered callback.
 *
 * The walk snapshots `next` before each call, so a callback may unregister
 * itself without corrupting the traversal. A disabled manager drops the feed.
 *
 * @param manager The manager (NULL-safe).
 * @param market_data The observation to deliver (NULL-safe).
 */
static inline void c_mds_monitor_manager_feed(mds_monitor_manager* manager, const md_variant* market_data) {
    if (!manager || !market_data) return;
    if (!atomic_load_explicit(&manager->enabled, memory_order_acquire)) return;

    mds_monitor_manager_callback_ctx* callback = manager->callbacks;
    while (callback) {
        mds_monitor_manager_callback_ctx* next = callback->next;
        if (callback->fn) callback->fn(manager, market_data, callback->user_data);
        callback = next;
    }
}

/**
 * @brief Register a monitor in the manager's registry (borrowed pointer).
 *
 * The registry key is the monitor's 16-byte id. When the id is all zero it is
 * generated from the monitor address and its name.
 *
 * @param manager The manager (NULL-safe).
 * @param monitor The monitor to register (NULL-safe).
 * @return MDS_OK, MDS_ERR_INVALID_INPUT, or MDS_ERR_OOM.
 */
static inline int c_mds_monitor_manager_register_monitor(mds_monitor_manager* manager, mds_monitor* monitor) {
    if (!manager || !monitor || !manager->monitors) return MDS_ERR_INVALID_INPUT;

    static const mds_monitor_id zero_id = {0};
    if (memcmp(monitor->monitor_id, zero_id, sizeof(mds_monitor_id)) == 0) {
        c_mds_monitor_gen_uid(monitor, monitor->name, monitor->name ? strlen(monitor->name) : 0);
    }

    int ret_code = c_bytemap_ex_set(
        manager->monitors,
        (const char*) monitor->monitor_id,
        sizeof(mds_monitor_id),
        (const char*) &monitor,
        sizeof(mds_monitor*),
        0,
        NULL
    );

    if (ret_code != BYTEMAP_OK) return (ret_code == BYTEMAP_ERR_OOM || ret_code == BYTEMAP_ERR_FULL) ? MDS_ERR_OOM : MDS_ERR_INVALID_INPUT;
    return MDS_OK;
}

/**
 * @brief Remove a monitor from the manager's registry.
 * @param manager The manager (NULL-safe).
 * @param monitor The monitor to remove (NULL-safe).
 * @return MDS_OK, MDS_ERR_INVALID_INPUT, or MDS_ERR_NOT_FOUND.
 */
static inline int c_mds_monitor_manager_unregister_monitor(mds_monitor_manager* manager, mds_monitor* monitor) {
    if (!manager || !monitor || !manager->monitors) return MDS_ERR_INVALID_INPUT;

    int ret_code = c_bytemap_ex_pop(
        manager->monitors,
        (const char*) monitor->monitor_id,
        sizeof(mds_monitor_id),
        0,
        NULL,
        NULL
    );

    if (ret_code == BYTEMAP_OK) return MDS_OK;
    if (ret_code == BYTEMAP_ERR_NOT_FOUND) return MDS_ERR_NOT_FOUND;
    return MDS_ERR_INVALID_INPUT;
}

/**
 * @brief Look up a registered monitor by id.
 * @param manager The manager (NULL-safe).
 * @param monitor_id The 16-byte monitor id.
 * @return The borrowed monitor, or NULL when not registered.
 */
static inline mds_monitor* c_mds_monitor_manager_get_monitor(const mds_monitor_manager* manager, const mds_monitor_id monitor_id) {
    if (!manager || !manager->monitors) return NULL;

    char* stored = NULL;

    int   ret_code = c_bytemap_ex_get_ptr(manager->monitors, (const char*) monitor_id, sizeof(mds_monitor_id), &stored, NULL);
    if (ret_code != BYTEMAP_OK) return NULL;

    return *(mds_monitor**) stored;
}

/**
 * @brief Remove every monitor from the registry (they are not freed).
 * @param manager The manager (NULL-safe).
 * @return MDS_OK, or MDS_ERR_INVALID_INPUT.
 */
static inline int c_mds_monitor_manager_clear_monitors(mds_monitor_manager* manager) {
    if (!manager || !manager->monitors) return MDS_ERR_INVALID_INPUT;

    c_bytemap_ex_clear(manager->monitors);
    return MDS_OK;
}

/**
 * @brief Set the manager's enabled flag.
 * @param manager The manager (NULL-safe).
 * @param enabled The new flag.
 */
static inline void c_mds_monitor_manager_set_enabled(mds_monitor_manager* manager, bool enabled) {
    if (!manager) return;

    atomic_store_explicit(&manager->enabled, enabled, memory_order_release);
}

/**
 * @brief Read the manager's enabled flag.
 * @param manager The manager (NULL-safe).
 * @return The flag, or false.
 */
static inline bool c_mds_monitor_manager_is_enabled(const mds_monitor_manager* manager) {
    return manager ? atomic_load_explicit(&manager->enabled, memory_order_acquire) : false;
}

// ========== Public APIs (Market Data Service) ==========

/**
 * @brief Allocate and initialize the core mds state.
 * @param allocator Allocator for the service and its children.
 * @return The new service, or NULL on allocation failure.
 */
static inline mds* c_mds_new(allocator_protocol* allocator) {
    mds* service = (mds*) c_ap_alloc(sizeof(mds), allocator);
    if (!service) return NULL;

    int ret_code = c_mds_init(service, allocator);
    if (ret_code != MDS_OK) {
        c_ap_free(service);
        return NULL;
    }

    return service;
}

/**
 * @brief Initialize a caller-allocated, zeroed mds state.
 * @param service The zeroed service to initialize.
 * @param allocator Allocator for the children; derived from service when NULL.
 * @return MDS_OK, MDS_ERR_INVALID_INPUT, or MDS_ERR_OOM.
 */
static inline int c_mds_init(mds* service, allocator_protocol* allocator) {
    if (!service) return MDS_ERR_INVALID_INPUT;
    if (!allocator) allocator = c_ap_protocol_from_ptr(service);

    service->subscriptions = c_bytemap_ex_new(0, sizeof(mds_subscription), allocator);
    if (!service->subscriptions) return MDS_ERR_OOM;

    service->monitors = c_bytemap_ex_new(0, sizeof(mds_monitor*), allocator);
    if (!service->monitors) {
        c_bytemap_ex_free(service->subscriptions);
        service->subscriptions = NULL;
        return MDS_ERR_OOM;
    }

    memset(&service->ts, 0, sizeof(session_datetime_t));
    return MDS_OK;
}

/**
 * @brief Tear down a caller-allocated mds state, leaving it zeroed.
 *
 * Both registries are released; the borrowed monitor manager and the monitors
 * are not freed here.
 *
 * @param service The service (NULL-safe).
 */
static inline void c_mds_dealloc(mds* service) {
    if (!service) return;

    c_mds_clear(service);

    if (service->subscriptions) {
        c_bytemap_ex_free(service->subscriptions);
        service->subscriptions = NULL;
    }

    if (service->monitors) {
        c_bytemap_ex_free(service->monitors);
        service->monitors = NULL;
    }

    memset(service, 0, sizeof(mds));
}

/**
 * @brief Free an mds state allocated by `c_mds_new`.
 * @param service The service (NULL-safe).
 */
static inline void c_mds_free(mds* service) {
    if (!service) return;

    c_mds_dealloc(service);
    c_ap_free(service);
}

/**
 * @brief Reset the service to its state right after `_init`.
 *
 * Both registries are cleared and the session datetime returns to zero. The
 * maps keep their capacity, and the borrowed monitor manager is left in place.
 *
 * @param service The service (NULL-safe).
 * @return MDS_OK, or MDS_ERR_INVALID_INPUT.
 */
static inline int c_mds_clear(mds* service) {
    if (!service) return MDS_ERR_INVALID_INPUT;

    if (service->subscriptions) c_bytemap_ex_clear(service->subscriptions);
    if (service->monitors) c_bytemap_ex_clear(service->monitors);

    memset(&service->ts, 0, sizeof(session_datetime_t));
    return MDS_OK;
}

/**
 * @brief Replace the borrowed monitor manager.
 * @param service The service (NULL-safe).
 * @param manager The new manager base (NULL disables dispatch).
 * @return MDS_OK, or MDS_ERR_INVALID_INPUT.
 */
static inline int c_mds_set_manager(mds* service, mds_monitor_manager* manager) {
    if (!service) return MDS_ERR_INVALID_INPUT;

    service->monitor_manager = manager;
    return MDS_OK;
}

/**
 * @brief Read the borrowed monitor manager.
 * @param service The service (NULL-safe).
 * @return The current manager base, or NULL.
 */
static inline mds_monitor_manager* c_mds_get_manager(const mds* service) {
    return service ? service->monitor_manager : NULL;
}

/**
 * @brief Register a monitor in the service registry (borrowed pointer).
 * @param service The service (NULL-safe).
 * @param monitor The monitor (NULL-safe).
 * @return MDS_OK, MDS_ERR_INVALID_INPUT, or MDS_ERR_OOM.
 */
static inline int c_mds_register_monitor(mds* service, mds_monitor* monitor) {
    if (!service || !monitor || !service->monitors) return MDS_ERR_INVALID_INPUT;

    static const mds_monitor_id zero_id = {0};
    if (memcmp(monitor->monitor_id, zero_id, sizeof(mds_monitor_id)) == 0) {
        c_mds_monitor_gen_uid(monitor, monitor->name, monitor->name ? strlen(monitor->name) : 0);
    }

    int ret_code = c_bytemap_ex_set(
        service->monitors,
        (const char*) monitor->monitor_id,
        sizeof(mds_monitor_id),
        (const char*) &monitor,
        sizeof(mds_monitor*),
        0,
        NULL
    );

    if (ret_code != BYTEMAP_OK) return (ret_code == BYTEMAP_ERR_OOM || ret_code == BYTEMAP_ERR_FULL) ? MDS_ERR_OOM : MDS_ERR_INVALID_INPUT;
    return MDS_OK;
}

/**
 * @brief Remove a monitor from the service registry.
 * @param service The service (NULL-safe).
 * @param monitor The monitor (NULL-safe).
 * @return MDS_OK, MDS_ERR_INVALID_INPUT, or MDS_ERR_NOT_FOUND.
 */
static inline int c_mds_unregister_monitor(mds* service, mds_monitor* monitor) {
    if (!service || !monitor || !service->monitors) return MDS_ERR_INVALID_INPUT;

    int ret_code = c_bytemap_ex_pop(
        service->monitors,
        (const char*) monitor->monitor_id,
        sizeof(mds_monitor_id),
        0,
        NULL,
        NULL
    );

    if (ret_code == BYTEMAP_OK) return MDS_OK;
    if (ret_code == BYTEMAP_ERR_NOT_FOUND) return MDS_ERR_NOT_FOUND;
    return MDS_ERR_INVALID_INPUT;
}

/**
 * @brief Look up a registered monitor by id.
 * @param service The service (NULL-safe).
 * @param monitor_id The 16-byte monitor id.
 * @return The borrowed monitor, or NULL.
 */
static inline mds_monitor* c_mds_get_monitor(const mds* service, const mds_monitor_id monitor_id) {
    if (!service || !service->monitors) return NULL;

    char* stored = NULL;

    int   ret_code = c_bytemap_ex_get_ptr(service->monitors, (const char*) monitor_id, sizeof(mds_monitor_id), &stored, NULL);
    if (ret_code != BYTEMAP_OK) return NULL;

    return *(mds_monitor**) stored;
}

/**
 * @brief Number of monitors in the service registry.
 * @param service The service (NULL-safe).
 * @return The count, or 0.
 */
static inline size_t c_mds_n_monitors(const mds* service) {
    return service && service->monitors ? c_bytemap_ex_len(service->monitors) : 0;
}

/**
 * @brief Register a ticker on first sight and update its subscription state.
 *
 * Existence is the bytemap's: one `c_bytemap_ex_get_ptr` decides subscribed vs
 * first sight from its return code. A hit is updated in place; a miss inserts
 * the zeroed default with `c_bytemap_ex_set_default` and fills the same slot.
 *
 * @param service The service (NULL-safe).
 * @param market_data The observation (NULL-safe).
 * @return MDS_OK, MDS_OK_SUBSCRIBED, MDS_ERR_INVALID_INPUT, MDS_ERR_UNKNOWN_DTYPE or MDS_ERR_OOM.
 */
static inline int c_mds_update_subscription(mds* service, const md_variant* market_data) {
    if (!service || !market_data || !service->subscriptions) return MDS_ERR_INVALID_INPUT;

    const md_meta* meta = &market_data->meta_info;
    const char*    ticker = meta->ticker;
    if (!ticker) return MDS_ERR_INVALID_INPUT;

    md_data_type dtype = meta->dtype;
    switch (dtype) {
        case DTYPE_TRANSACTION:
        case DTYPE_ORDER:
        case DTYPE_TICK_LITE:
        case DTYPE_TICK:
        case DTYPE_BAR:
            break;
        default:
            return MDS_ERR_UNKNOWN_DTYPE;
    }

    size_t ticker_len = strlen(ticker);
    char*  stored = NULL;

    int    ret_code = c_bytemap_ex_get_ptr(service->subscriptions, ticker, ticker_len, &stored, NULL);
    if (ret_code != BYTEMAP_OK && ret_code != BYTEMAP_ERR_NOT_FOUND) return MDS_ERR_INVALID_INPUT;

    bool is_new = ret_code == BYTEMAP_ERR_NOT_FOUND;

    if (is_new) {
        mds_subscription default_subscription;
        memset(&default_subscription, 0, sizeof(mds_subscription));

        ret_code = c_bytemap_ex_set_default(
            service->subscriptions,
            ticker,
            ticker_len,
            (const char*) &default_subscription,
            sizeof(mds_subscription),
            0,
            &stored,
            NULL
        );

        if (ret_code != BYTEMAP_OK) {
            return (ret_code == BYTEMAP_ERR_OOM || ret_code == BYTEMAP_ERR_FULL) ? MDS_ERR_OOM : MDS_ERR_INVALID_INPUT;
        }
    }

    mds_subscription* subscription = (mds_subscription*) stored;

    subscription->last_update = meta->dt;
    subscription->last_price = c_md_get_price(market_data);
    subscription->n_feeds += 1;

    switch (dtype) {
        case DTYPE_TRANSACTION:
            subscription->is_subscribed_td = true;
            break;
        case DTYPE_ORDER:
            subscription->is_subscribed_od = true;
            break;
        case DTYPE_TICK_LITE:
            subscription->is_subscribed_tk_lite = true;
            break;
        case DTYPE_TICK:
            subscription->is_subscribed_tk = true;
            break;
        case DTYPE_BAR:
            subscription->is_subscribed_cs = true;
            break;
        default:
            break;
    }

    return is_new ? MDS_OK_SUBSCRIBED : MDS_OK;
}

/**
 * @brief Route one market data observation through the service.
 *
 * The hot path: the subscription update runs inline, with the bytemap's
 * lookup deciding subscribed vs first sight; the session datetime is copied
 * from the observation, and the borrowed manager's callbacks run.
 *
 * @param service The service (NULL-safe).
 * @param market_data The observation (NULL-safe).
 * @return MDS_OK, MDS_OK_SUBSCRIBED, MDS_ERR_INVALID_INPUT, MDS_ERR_INTERNAL_DATA, MDS_ERR_UNKNOWN_DTYPE or MDS_ERR_OOM.
 */
static inline int c_mds_on_market_data(mds* service, const md_variant* market_data) {
    if (!service || !market_data) return MDS_ERR_INVALID_INPUT;

    const md_meta* meta = &market_data->meta_info;
    md_data_type   dtype = meta->dtype;
    const char*    ticker = meta->ticker;

    if (dtype == DTYPE_INTERNAL) return MDS_ERR_INTERNAL_DATA;
    if (!ticker || !service->subscriptions) return MDS_ERR_INVALID_INPUT;

    switch (dtype) {
        case DTYPE_TRANSACTION:
        case DTYPE_ORDER:
        case DTYPE_TICK_LITE:
        case DTYPE_TICK:
        case DTYPE_BAR:
            break;
        default:
            return MDS_ERR_UNKNOWN_DTYPE;
    }

    // Step 1: Inlined subscription update.
    size_t ticker_len = strlen(ticker);
    char*  stored = NULL;

    int    ret_code = c_bytemap_ex_get_ptr(service->subscriptions, ticker, ticker_len, &stored, NULL);
    if (ret_code != BYTEMAP_OK && ret_code != BYTEMAP_ERR_NOT_FOUND) return MDS_ERR_INVALID_INPUT;

    bool is_new = ret_code == BYTEMAP_ERR_NOT_FOUND;

    if (is_new) {
        mds_subscription default_subscription;
        memset(&default_subscription, 0, sizeof(mds_subscription));

        ret_code = c_bytemap_ex_set_default(
            service->subscriptions,
            ticker,
            ticker_len,
            (const char*) &default_subscription,
            sizeof(mds_subscription),
            0,
            &stored,
            NULL
        );

        if (ret_code != BYTEMAP_OK) {
            return (ret_code == BYTEMAP_ERR_OOM || ret_code == BYTEMAP_ERR_FULL) ? MDS_ERR_OOM : MDS_ERR_INVALID_INPUT;
        }
    }

    mds_subscription* subscription = (mds_subscription*) stored;

    subscription->last_update = meta->dt;
    subscription->last_price = c_md_get_price(market_data);
    subscription->n_feeds += 1;

    switch (dtype) {
        case DTYPE_TRANSACTION:
            subscription->is_subscribed_td = true;
            break;
        case DTYPE_ORDER:
            subscription->is_subscribed_od = true;
            break;
        case DTYPE_TICK_LITE:
            subscription->is_subscribed_tk_lite = true;
            break;
        case DTYPE_TICK:
            subscription->is_subscribed_tk = true;
            break;
        case DTYPE_BAR:
            subscription->is_subscribed_cs = true;
            break;
        default:
            break;
    }

    // Step 2: Stamp the session datetime and dispatch.
    service->ts = meta->dt;
    c_mds_monitor_manager_feed(service->monitor_manager, market_data);

    return is_new ? MDS_OK_SUBSCRIBED : MDS_OK;
}

/**
 * @brief Route one internal observation through the service.
 *
 * Internal data does not enter the subscription map; it is handed to the
 * monitor manager exactly like market data, preserving the service's original
 * contract.
 *
 * @param service The service (NULL-safe).
 * @param internal_data The internal observation (NULL-safe).
 * @return MDS_OK, or MDS_ERR_INVALID_INPUT.
 */
static inline int c_mds_on_internal_data(mds* service, const md_internal* internal_data) {
    if (!service || !internal_data) return MDS_ERR_INVALID_INPUT;

    c_mds_monitor_manager_feed(service->monitor_manager, (const md_variant*) internal_data);
    return MDS_OK;
}

/**
 * @brief Read the latest price of a ticker.
 * @param service The service (NULL-safe).
 * @param ticker The ticker symbol (NULL-safe).
 * @return The latest price, or NaN when the ticker is unknown or unfed.
 */
static inline double c_mds_get_market_price(const mds* service, const char* ticker) {
    if (!service || !ticker || !service->subscriptions) return NAN;

    char* stored = NULL;

    int   ret_code = c_bytemap_ex_get_ptr(service->subscriptions, ticker, strlen(ticker), &stored, NULL);
    if (ret_code != BYTEMAP_OK) return NAN;

    const mds_subscription* subscription = (const mds_subscription*) stored;
    return subscription->last_price;
}

#endif /* C_MARKET_ENGINE_H */

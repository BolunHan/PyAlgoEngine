#ifndef C_DMA_PROTOCOL_H
#define C_DMA_PROTOCOL_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cbase/allocator_protocol/c_allocator_protocol.h>
#include <cbase/bytemap/c_bytemap.h>
#include <cbase/intern_string/c_intern_string.h>
#include <event_engine/capi/c_engine.h>
#include <event_engine/capi/c_event.h>
#include <event_engine/capi/c_topic.h>

#include <algo_engine/base/c_market_data/c_market_data.h>
#include <algo_engine/base/c_market_data/c_market_data_buffer.h>
#include <algo_engine/engine/c_event_engine.h>

// ========== Constants ==========

#ifndef DMA_PROTOCOL_DEFAULT_BLOCK
#define DMA_PROTOCOL_DEFAULT_BLOCK 1
#endif

#ifndef DMA_PROTOCOL_DEFAULT_MAX_SPIN
#define DMA_PROTOCOL_DEFAULT_MAX_SPIN DEFAULT_MQ_SPIN_LIMIT
#endif

#ifndef DMA_PROTOCOL_DEFAULT_TIMEOUT
#define DMA_PROTOCOL_DEFAULT_TIMEOUT 0.0
#endif

// ========== Enums ==========

// clang-format off

/** @enum dma_protocol_ret_code
 *  @brief 0 is success, negatives are errors. */
typedef enum dma_protocol_ret_code {
    DMA_PROTOCOL_OK                =  0,
    DMA_PROTOCOL_ERR_INVALID_INPUT = -1,
    DMA_PROTOCOL_ERR_OOM           = -2,
    DMA_PROTOCOL_ERR_NO_SHM        = -3,
    DMA_PROTOCOL_ERR_NO_TOPIC      = -4,
    DMA_PROTOCOL_ERR_PUBLISH       = -5,
    DMA_PROTOCOL_ERR_ENGINE        = -6,
    DMA_PROTOCOL_ERR_NOT_FOUND     = -7,
    DMA_PROTOCOL_ERR_BUSY          = -8,
} dma_protocol_ret_code;

// clang-format on

// ========== Structs ==========

typedef struct dma_protocol_ctx dma_protocol_ctx;

/**
 * @brief Listener invoked with the shared-memory market data.
 *
 * The market data is const: the transport does not copy it per listener, so
 * the listener must not modify it. It is valid only for the duration of the
 * call; a listener that keeps it past the call must copy it.
 *
 * The callback is registered directly with the event engine as a
 * `EVT_CALLBACK_WITH_ARGS_USERDATA` handler, so the listened topic must carry
 * only transport payloads.
 */
typedef void (*dma_protocol_callback_func)(const md_variant* market_data, void* user_data);

/**
 * @struct dma_protocol_ctx
 * @brief Market-data transport over the event engine.
 *
 * The engine, the topic set, the shared-memory allocator and the shared-memory
 * intern pool are borrowed and must outlive the protocol. The protocol must be
 * its own allocator block and must outlive any queued payload.
 */
typedef struct dma_protocol_ctx {
    evt_engine*         ENGINE;         // NOT owned - the event engine.
    evt_topic_set*      TOPICS;         // NOT owned - the topic registry.
    allocator_protocol* SHM_ALLOCATOR;  // NOT owned - the shared-memory allocator.
    istr_map*           SHM_POOL;       // NOT owned - the shared-memory intern pool.
    bool                block;          // Publish blocks when the queue is full.
    size_t              max_spin;       // Spin iterations before blocking.
    double              timeout;        // Blocking timeout in seconds; <= 0 waits forever.
} dma_protocol_ctx;

// ========== Static Payload ==========

/**
 * @brief The single payload reused by every send in this translation unit.
 *
 * The event engine queues the pointer and hands it back to `fn_dealloc` after
 * dispatch; the payload itself is never freed. `_PAYLOAD` is the owning
 * market-data/report payload and its cleanup releases the md it carries;
 * `_ORDER_PAYLOAD` carries a borrowed order pointer and has no cleanup.
 */
static evt_message_payload      _PAYLOAD;        // Owning market-data/report payload; cleanup releases it.
static evt_message_payload      _ORDER_PAYLOAD;  // Borrowed in-process order payload; no cleanup.

// ========== Forward Declarations ==========

static inline dma_protocol_ctx* c_dma_protocol_new(evt_engine* engine, evt_topic_set* topics, allocator_protocol* shm_allocator, istr_map* shm_pool, allocator_protocol* allocator);
static inline int               c_dma_protocol_init(dma_protocol_ctx* protocol, evt_engine* engine, evt_topic_set* topics, allocator_protocol* shm_allocator, istr_map* shm_pool);
static inline void              c_dma_protocol_dealloc(dma_protocol_ctx* protocol);
static inline void              c_dma_protocol_free(dma_protocol_ctx* protocol);

static inline void              c_dma_protocol_cleanup(evt_message_payload* payload);
static inline evt_topic*        c_dma_protocol_callback_topic(dma_protocol_ctx* protocol, const md_variant* market_data);
static inline evt_topic*        c_dma_protocol_handler_topic(dma_protocol_ctx* protocol, evt_topic* pattern, const md_trade_instruction* order);

static inline int               c_dma_protocol_config(dma_protocol_ctx* protocol, bool block, size_t max_spin, double timeout);
static inline int               c_dma_protocol_send(dma_protocol_ctx* protocol, md_variant* market_data);
static inline int               c_dma_protocol_send_ex(dma_protocol_ctx* protocol, md_variant* market_data, evt_topic* topic);
static inline int               c_dma_protocol_register_callback(dma_protocol_ctx* protocol, evt_topic* topic, dma_protocol_callback_func fn, void* user_data);
static inline int               c_dma_protocol_unregister_callback(dma_protocol_ctx* protocol, evt_topic* topic, dma_protocol_callback_func fn, void* user_data);
static inline void              c_dma_protocol_launch_order(md_trade_instruction* order, uint32_t seq_id, void* user_data);
static inline void              c_dma_protocol_cancel_order(const md_trade_instruction* order, uint32_t seq_id, void* user_data);

// ========== Utility Functions ==========

/**
 * @brief Release one dispatched payload and leave the static payload reusable.
 *
 * Installed as `evt_message_payload.fn_dealloc`. The engine calls it after
 * dispatch, or when a publish fails. The payload's market data belongs to the
 * transport, so it is always released.
 *
 * @param payload The engine payload (NULL-safe).
 */
static inline void              c_dma_protocol_cleanup(evt_message_payload* payload) {
    if (!payload) return;

    if (payload->args) c_ap_free(payload->args);

    payload->args       = NULL;
    payload->topic      = NULL;
    payload->seq_id     = 0;
    payload->fn_dealloc = NULL;
}

/**
 * @brief Resolve the publish topic from the payload dtype.
 *
 * A trade instruction publishes on `on_order`, an exchange report on
 * `on_report`, and every other dtype (including order data) on
 * `realtime.<ticker>.<dtype>` through `c_md_topic_push`.
 *
 * @param protocol The protocol (NULL-safe).
 * @param market_data The payload (NULL-safe).
 * @return The exact topic, or NULL when none applies.
 */
static inline evt_topic* c_dma_protocol_callback_topic(dma_protocol_ctx* protocol, const md_variant* market_data) {
    if (!protocol || !protocol->TOPICS || !market_data) return NULL;

    switch (market_data->meta_info.dtype) {
        case DTYPE_INSTRUCTION:
            return protocol->TOPICS->on_order;
        case DTYPE_REPORT:
            return protocol->TOPICS->on_report;
        default:
            return c_md_topic_push(protocol->TOPICS, market_data);
    }
}

/**
 * @brief Resolve the outgoing launch/cancel topic for an order.
 *
 * The pattern is one of the topic set's launch_order/cancel_order templates;
 * the exact topic comes from that pattern and its evt_launch/evt_cancel ticker
 * cache. The protocol keeps no topic state of its own.
 *
 * @param protocol The protocol (NULL-safe).
 * @param pattern The set's `launch_order` or `cancel_order` pattern.
 * @param order The order (NULL-safe; its ticker is read).
 * @return The registered exact topic, or NULL.
 */
static inline evt_topic* c_dma_protocol_handler_topic(dma_protocol_ctx* protocol, evt_topic* pattern, const md_trade_instruction* order) {
    if (!protocol || !protocol->TOPICS || !pattern || !order) return NULL;

    if (pattern == protocol->TOPICS->cancel_order) return c_md_topic_dma_cancel(protocol->TOPICS, order);

    return c_md_topic_dma_launch(protocol->TOPICS, order);
}

// ========== Lifecycle ==========

/**
 * @brief Allocate and initialize a protocol; NULL on allocation failure.
 *
 * @param engine The event engine (borrowed).
 * @param topics The topic set (borrowed).
 * @param shm_allocator The shared-memory allocator (borrowed).
 * @param shm_pool The shared-memory intern pool (borrowed).
 * @param allocator Allocator for the protocol block.
 * @return The new protocol, or NULL.
 */
static inline dma_protocol_ctx* c_dma_protocol_new(evt_engine* engine, evt_topic_set* topics, allocator_protocol* shm_allocator, istr_map* shm_pool, allocator_protocol* allocator) {
    dma_protocol_ctx* protocol = (dma_protocol_ctx*) c_ap_alloc(sizeof(dma_protocol_ctx), allocator);
    if (!protocol) return NULL;

    if (c_dma_protocol_init(protocol, engine, topics, shm_allocator, shm_pool) != DMA_PROTOCOL_OK) {
        c_ap_free(protocol);
        return NULL;
    }
    return protocol;
}

/**
 * @brief Initialize a caller-allocated protocol.
 *
 * The caller owns the block. The block must be zeroed on entry; the borrowed
 * engine, topic set, allocator and pool must outlive the protocol.
 *
 * @param protocol The block to initialize (must be a block start).
 * @param engine The event engine (borrowed).
 * @param topics The topic set (borrowed).
 * @param shm_allocator The shared-memory allocator (borrowed).
 * @param shm_pool The shared-memory intern pool (borrowed).
 * @return DMA_PROTOCOL_OK, or DMA_PROTOCOL_ERR_INVALID_INPUT.
 */
static inline int c_dma_protocol_init(dma_protocol_ctx* protocol, evt_engine* engine, evt_topic_set* topics, allocator_protocol* shm_allocator, istr_map* shm_pool) {
    if (!protocol) return DMA_PROTOCOL_ERR_INVALID_INPUT;

    memset(protocol, 0, sizeof(dma_protocol_ctx));

    protocol->ENGINE        = engine;
    protocol->TOPICS        = topics;
    protocol->SHM_ALLOCATOR = shm_allocator;
    protocol->SHM_POOL      = shm_pool;
    protocol->block         = DMA_PROTOCOL_DEFAULT_BLOCK;
    protocol->max_spin      = DMA_PROTOCOL_DEFAULT_MAX_SPIN;
    protocol->timeout       = DMA_PROTOCOL_DEFAULT_TIMEOUT;
    return DMA_PROTOCOL_OK;
}

/**
 * @brief Tear down a caller-allocated protocol and leave it zeroed.
 *
 * Unregister every callback with `c_dma_protocol_unregister_callback` before
 * this call; the protocol keeps no registration list. Drain the engine queue
 * before releasing it so no queued payload outlives the protocol.
 *
 * @param protocol The protocol (NULL-safe).
 */
static inline void c_dma_protocol_dealloc(dma_protocol_ctx* protocol) {
    if (!protocol) return;

    memset(protocol, 0, sizeof(dma_protocol_ctx));
}

/**
 * @brief Free a protocol created by `c_dma_protocol_new`.
 * @param protocol The protocol (NULL-safe).
 */
static inline void c_dma_protocol_free(dma_protocol_ctx* protocol) {
    if (!protocol) return;
    c_dma_protocol_dealloc(protocol);
    c_ap_free(protocol);
}

// ========== Public APIs ==========

/**
 * @brief Configure how `send` publishes.
 *
 * @param protocol The protocol (NULL-safe).
 * @param block Block when the engine queue is full.
 * @param max_spin Spin iterations before falling back to blocking.
 * @param timeout Blocking timeout in seconds; <= 0 waits forever.
 * @return DMA_PROTOCOL_OK, or DMA_PROTOCOL_ERR_INVALID_INPUT.
 */
static inline int c_dma_protocol_config(dma_protocol_ctx* protocol, bool block, size_t max_spin, double timeout) {
    if (!protocol) return DMA_PROTOCOL_ERR_INVALID_INPUT;

    protocol->block    = block;
    protocol->max_spin = max_spin;
    protocol->timeout  = timeout;
    return DMA_PROTOCOL_OK;
}

/**
 * @brief Publish market data on its dtype's topic.
 *
 * A trade instruction publishes on `on_order`, an exchange report on
 * `on_report`, and every other dtype (including order data) on
 * `realtime.<ticker>.<dtype>`.
 *
 * The transport owns the payload's market data: a source already in its
 * shared-memory region is transferred and released after dispatch, while a
 * source outside the region is copied and the copy is released. The caller
 * must not release an in-region source after this call.
 *
 * @param protocol The protocol (NULL-safe).
 * @param market_data The market data to publish.
 * @return DMA_PROTOCOL_OK, or a `dma_protocol_ret_code` error.
 */
static inline int c_dma_protocol_send(dma_protocol_ctx* protocol, md_variant* market_data) {
    return c_dma_protocol_send_ex(protocol, market_data, NULL);
}

/**
 * @brief Publish market data on an explicit exact topic.
 *
 * The transport reuses one static payload: a second send before the previous
 * message is dispatched is refused as `DMA_PROTOCOL_ERR_BUSY`.
 *
 * @param protocol The protocol (NULL-safe).
 * @param market_data The market data to publish.
 * @param topic Optional exact topic; NULL resolves from the market data.
 * @return DMA_PROTOCOL_OK, DMA_PROTOCOL_ERR_INVALID_INPUT,
 *         DMA_PROTOCOL_ERR_BUSY, DMA_PROTOCOL_ERR_NO_TOPIC,
 *         DMA_PROTOCOL_ERR_NO_SHM, or DMA_PROTOCOL_ERR_PUBLISH.
 */
static inline int c_dma_protocol_send_ex(dma_protocol_ctx* protocol, md_variant* market_data, evt_topic* topic) {
    if (!protocol || !market_data || !protocol->ENGINE || !protocol->ENGINE->mq || !protocol->TOPICS || !protocol->SHM_ALLOCATOR || !protocol->SHM_POOL) {
        return DMA_PROTOCOL_ERR_INVALID_INPUT;
    }

    if (_PAYLOAD.args) return DMA_PROTOCOL_ERR_BUSY;

    evt_topic* target = topic ? topic : c_dma_protocol_callback_topic(protocol, market_data);
    if (!target || !target->is_exact) return DMA_PROTOCOL_ERR_NO_TOPIC;

    const md_variant* payload_market_data = c_md_send_to_shm(market_data, protocol->SHM_ALLOCATOR, protocol->SHM_POOL);
    if (!payload_market_data) return DMA_PROTOCOL_ERR_NO_SHM;

    _PAYLOAD.args       = (void*) payload_market_data;
    _PAYLOAD.topic      = target;
    _PAYLOAD.seq_id     = 0;
    _PAYLOAD.fn_dealloc = c_dma_protocol_cleanup;

    int ret_code = c_evt_engine_publish(protocol->ENGINE, &_PAYLOAD, protocol->block, protocol->max_spin, protocol->timeout);
    if (ret_code != EVT_RET_OK) {
        // The engine runs the cleanup on a queue failure; an early rejection
        // leaves it armed, so release the payload here instead of leaking it.
        if (_PAYLOAD.fn_dealloc) c_dma_protocol_cleanup(&_PAYLOAD);
        return DMA_PROTOCOL_ERR_PUBLISH;
    }

    return DMA_PROTOCOL_OK;
}

/**
 * @brief Register a callback directly with the event engine.
 *
 * The engine stores the function as an `EVT_CALLBACK_WITH_ARGS_USERDATA`
 * handler and passes `payload->args` as the `const md_variant*`. The listened
 * topic must therefore carry transport payloads only.
 *
 * @param protocol The protocol (NULL-safe).
 * @param topic The exact or generic topic to listen on.
 * @param fn The market-data callback (NULL is refused).
 * @param user_data Opaque value handed to every call.
 * @return DMA_PROTOCOL_OK, DMA_PROTOCOL_ERR_INVALID_INPUT,
 *         DMA_PROTOCOL_ERR_OOM, or DMA_PROTOCOL_ERR_ENGINE.
 */
static inline int c_dma_protocol_register_callback(dma_protocol_ctx* protocol, evt_topic* topic, dma_protocol_callback_func fn, void* user_data) {
    if (!protocol || !protocol->ENGINE || !topic || !fn) return DMA_PROTOCOL_ERR_INVALID_INPUT;

    int ret_code = c_evt_engine_register_handler(protocol->ENGINE, topic, (const void*) fn, EVT_CALLBACK_WITH_ARGS_USERDATA, user_data, true);

    if (ret_code == EVT_RET_OK || ret_code == EVT_RET_ERR_DUPLICATE) return DMA_PROTOCOL_OK;
    if (ret_code == EVT_RET_ERR_OOM) return DMA_PROTOCOL_ERR_OOM;
    return DMA_PROTOCOL_ERR_ENGINE;
}

/**
 * @brief Unregister a callback from the event engine.
 *
 * Matches the callback type, the function and the user data, so callbacks
 * sharing a function are independently removable.
 *
 * @param protocol The protocol (NULL-safe).
 * @param topic The registered topic.
 * @param fn The registered callback.
 * @param user_data The value registered with the callback.
 * @return DMA_PROTOCOL_OK, DMA_PROTOCOL_ERR_INVALID_INPUT,
 *         DMA_PROTOCOL_ERR_NOT_FOUND, or DMA_PROTOCOL_ERR_ENGINE.
 */
static inline int c_dma_protocol_unregister_callback(dma_protocol_ctx* protocol, evt_topic* topic, dma_protocol_callback_func fn, void* user_data) {
    if (!protocol || !protocol->ENGINE || !topic || !fn) return DMA_PROTOCOL_ERR_INVALID_INPUT;

    bytemap* map = topic->is_exact ? protocol->ENGINE->exact_topic_hooks : protocol->ENGINE->generic_topic_hooks;
    if (!map) return DMA_PROTOCOL_ERR_ENGINE;

    evt_hook* hook = NULL;
    if (c_bytemap_get(map, topic->key, topic->key_len, (void**) &hook) != BYTEMAP_OK || !hook) return DMA_PROTOCOL_ERR_NOT_FOUND;

    for (size_t i = 0; i < hook->n_callbacks; i++) {
        evt_callback* callback = &hook->callbacks[i];
        if (callback->type != EVT_CALLBACK_WITH_ARGS_USERDATA) continue;
        if ((const void*) callback->fn.bare != (const void*) fn) continue;
        if (callback->user_data != user_data) continue;

        int ret_code = c_evt_hook_pop_callback(hook, i);
        if (ret_code != EVT_RET_OK) return DMA_PROTOCOL_ERR_ENGINE;

        if (hook->n_callbacks == 0) {
            c_bytemap_pop(map, topic->key, topic->key_len, NULL);
            c_evt_hook_free(hook);
        }
        return DMA_PROTOCOL_OK;
    }

    return DMA_PROTOCOL_ERR_NOT_FOUND;
}

/**
 * @brief DMA launch handler: publish an order on `launch_order.<ticker>`.
 *
 * Signature-compatible with `dma_order_launch_func`; `user_data` is the
 * `dma_protocol_ctx*`. The order is dispatched in process by pointer; its
 * ownership stays with the DMA, so there is no copy, no shared-memory transfer
 * and no payload cleanup. Failures are reported to stderr because the handler
 * contract returns void.
 *
 * @param order The order to publish.
 * @param seq_id The access sequence id, carried as the event payload seq_id.
 * @param user_data The `dma_protocol_ctx*`.
 */
static inline void c_dma_protocol_launch_order(md_trade_instruction* order, uint32_t seq_id, void* user_data) {
    dma_protocol_ctx* protocol = (dma_protocol_ctx*) user_data;
    evt_topic*        target   = NULL;
    int               ret_code = DMA_PROTOCOL_ERR_INVALID_INPUT;

    if (protocol && protocol->ENGINE && protocol->ENGINE->mq && order && order->meta_info.ticker) {
        target = c_dma_protocol_handler_topic(protocol, protocol->TOPICS->launch_order, order);

        if (target) {
            _ORDER_PAYLOAD.args       = (void*) order;
            _ORDER_PAYLOAD.topic      = target;
            _ORDER_PAYLOAD.seq_id     = seq_id;
            _ORDER_PAYLOAD.fn_dealloc = NULL;

            int mq_ret = protocol->block
                             ? c_mq_put_hybrid(protocol->ENGINE->mq, &_ORDER_PAYLOAD, protocol->max_spin, protocol->timeout)
                             : c_mq_put(protocol->ENGINE->mq, &_ORDER_PAYLOAD);

            ret_code = mq_ret == EVT_RET_OK ? DMA_PROTOCOL_OK : DMA_PROTOCOL_ERR_PUBLISH;
        }
        else {
            ret_code = DMA_PROTOCOL_ERR_NO_TOPIC;
        }
    }

    if (ret_code != DMA_PROTOCOL_OK) {
        fprintf(stderr, "[DMA] ERROR: launch order publish failed, code %d, ticker <%s>\n", ret_code, order && order->meta_info.ticker ? order->meta_info.ticker : "<unknown>");
    }
}

/**
 * @brief DMA cancel handler: publish an order on `cancel_order.<ticker>`.
 *
 * Signature-compatible with `dma_order_cancel_func`; `user_data` is the
 * `dma_protocol_ctx*`. The order is dispatched in process by pointer; its
 * ownership stays with the DMA, so there is no copy, no shared-memory transfer
 * and no payload cleanup. Failures are reported to stderr because the handler
 * contract returns void.
 *
 * @param order The cancel instruction.
 * @param seq_id The access sequence id, carried as the event payload seq_id.
 * @param user_data The `dma_protocol_ctx*`.
 */
static inline void c_dma_protocol_cancel_order(const md_trade_instruction* order, uint32_t seq_id, void* user_data) {
    dma_protocol_ctx* protocol = (dma_protocol_ctx*) user_data;
    evt_topic*        target   = NULL;
    int               ret_code = DMA_PROTOCOL_ERR_INVALID_INPUT;

    if (protocol && protocol->ENGINE && protocol->ENGINE->mq && order && order->meta_info.ticker) {
        target = c_dma_protocol_handler_topic(protocol, protocol->TOPICS->cancel_order, order);

        if (target) {
            _ORDER_PAYLOAD.args       = (void*) order;
            _ORDER_PAYLOAD.topic      = target;
            _ORDER_PAYLOAD.seq_id     = seq_id;
            _ORDER_PAYLOAD.fn_dealloc = NULL;

            int mq_ret = protocol->block
                             ? c_mq_put_hybrid(protocol->ENGINE->mq, &_ORDER_PAYLOAD, protocol->max_spin, protocol->timeout)
                             : c_mq_put(protocol->ENGINE->mq, &_ORDER_PAYLOAD);

            ret_code = mq_ret == EVT_RET_OK ? DMA_PROTOCOL_OK : DMA_PROTOCOL_ERR_PUBLISH;
        }
        else {
            ret_code = DMA_PROTOCOL_ERR_NO_TOPIC;
        }
    }

    if (ret_code != DMA_PROTOCOL_OK) {
        fprintf(stderr, "[DMA] ERROR: cancel order publish failed, code %d, ticker <%s>\n", ret_code, order && order->meta_info.ticker ? order->meta_info.ticker : "<unknown>");
    }
}

#endif  // C_DMA_PROTOCOL_H

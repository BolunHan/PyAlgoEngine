from cpython.exc cimport PyErr_Clear
from cpython.object cimport PyObject
from libc.stdint cimport uintptr_t

from cbase.bytemap cimport c_bytemap_len

from algo_engine.base.c_market_data.c_market_data cimport (
    c_get_long_id,
    md_trade_report,
    md_variant,
)
from algo_engine.base.c_market_data.c_trade_utils cimport TradeReport, report_from_header

from algo_engine.backtest.c_simmatch cimport (
    SimMatch,
    c_smm_match_deregister_listener,
    c_smm_match_register_listener,
    smm_match_ctx,
    smm_match_event,
    smm_order_entry,
    smm_ret_code,
)


cdef dict _listeners = {}


cdef void c_listener(smm_match_ctx* ctx, smm_match_event event,
                     const smm_order_entry* entry,
                     const md_trade_report* report,
                     void* user_data) noexcept:
    cdef object py_cb = <object> user_data
    cdef object key
    cdef TradeReport py_report
    try:
        if not entry:
            return
        key = c_get_long_id(&entry.order.order_id)
        if report:
            py_report = <TradeReport> report_from_header(<const md_variant*> report, False)
            py_report = <TradeReport> py_report.__copy__()
        else:
            py_report = None
        py_cb(event, key, py_report)
    except:
        PyErr_Clear()


def register_listener(SimMatch sim, callback):
    cdef uintptr_t callback_id
    cdef int ret_code = c_smm_match_register_listener(
        sim.header, c_listener, <void*> <PyObject*> callback, &callback_id)
    if ret_code != smm_ret_code.SMM_OK:
        raise RuntimeError(f'Failed to register listener, error code {ret_code}')
    _listeners[callback_id] = callback
    return callback_id


def deregister_listener(SimMatch sim, callback_id):
    _listeners.pop(callback_id, None)
    cdef int ret_code = c_smm_match_deregister_listener(sim.header, callback_id)
    if ret_code != smm_ret_code.SMM_OK:
        raise RuntimeError(f'Failed to deregister listener, error code {ret_code}')


def adopt(SimMatch sim):
    return SimMatch.c_from_header(sim.header, sim.ticker, False)


def registry_state(SimMatch sim):
    cdef smm_match_ctx* ctx = sim.header
    if not ctx:
        raise RuntimeError(f'Uninitialized <{sim.__class__.__name__}>')
    return (
        c_bytemap_len(ctx.working),
        c_bytemap_len(ctx.history),
    )


def working_order_ids(SimMatch sim):
    cdef smm_match_ctx* ctx = sim.header
    if not ctx:
        raise RuntimeError(f'Uninitialized <{sim.__class__.__name__}>')
    cdef list order_ids = []
    cdef smm_order_entry* entry = ctx.newest
    while entry:
        order_ids.append(c_get_long_id(&entry.order.order_id))
        entry = entry.older
    return order_ids

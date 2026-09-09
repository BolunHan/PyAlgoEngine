from cpython.exc cimport PyErr_Clear
from cpython.object cimport PyObject
from libc.stdint cimport uintptr_t

from algo_engine.base.c_market_data.c_market_data cimport (
    c_get_long_id,
    md_trade_report,
    md_variant,
)
from algo_engine.base.c_market_data.c_trade_utils cimport TradeReport, report_from_header

from algo_engine.backtest.c_simmatch_ex cimport (
    SMM_EVENT_CANCEL,
    SMM_EVENT_MATCH,
    SMM_OK,
    SimMatchEx,
    c_smm_match_deregister_listener,
    c_smm_match_register_listener,
    smm_match_ctx,
    smm_match_event,
    smm_order_entry,
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
        key = c_get_long_id(&entry.order_id)
        if report:
            py_report = <TradeReport> report_from_header(<const md_variant*> report, False)
            py_report = <TradeReport> py_report.__copy__()
        else:
            py_report = None
        py_cb(event, key, py_report)
    except:
        PyErr_Clear()


def register_listener(SimMatchEx sim, callback):
    cdef uintptr_t callback_id
    cdef int ret_code = c_smm_match_register_listener(
        sim.ctx, c_listener, <void*> <PyObject*> callback, &callback_id)
    if ret_code != SMM_OK:
        raise RuntimeError(f'Failed to register listener, error code {ret_code}')
    _listeners[callback_id] = callback
    return callback_id


def deregister_listener(SimMatchEx sim, callback_id):
    _listeners.pop(callback_id, None)
    cdef int ret_code = c_smm_match_deregister_listener(sim.ctx, callback_id)
    if ret_code != SMM_OK:
        raise RuntimeError(f'Failed to deregister listener, error code {ret_code}')

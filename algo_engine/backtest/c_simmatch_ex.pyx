import random as py_random

from cpython.exc cimport PyErr_Clear
from cpython.mem cimport PyMem_Free, PyMem_Malloc
from cpython.object cimport PyObject
from libc.math cimport NAN, isnan
from libc.stdint cimport int8_t, uint64_t, uintptr_t
from libc.string cimport memset

from event_engine.capi cimport EventEngine
from event_engine.capi.c_event cimport EventHook
from event_engine.capi.c_topic cimport Topic

from algo_engine.base.c_market_data.c_market_data cimport (
    MarketData,
    c_get_long_id,
    long_md_id,
    md_order_state,
    md_order_type,
    md_variant,
)
from algo_engine.base.c_market_data.c_trade_utils cimport (
    TradeInstruction,
    TradeReport,
    report_from_header,
)
from algo_engine.exchange_profile.c_exchange_profile cimport PROFILE

from ..engine.c_event_engine cimport EVENT_ENGINE, TOPIC, TopicSet

from . import LOGGER

from .c_simmatch_ex cimport (
    SMM_ERR_INVALID_SIDE,
    SMM_ERR_NO_VALID_PRICE,
    SMM_EVENT_CANCEL,
    SMM_EVENT_MATCH,
    SMM_OK,
    c_smm_match_best_price,
    c_smm_match_cancel,
    c_smm_match_clear,
    c_smm_match_dealloc,
    c_smm_match_deregister_listener,
    c_smm_match_eod,
    c_smm_match_free,
    c_smm_match_launch,
    c_smm_match_new,
    c_smm_match_process,
    c_smm_match_register,
    c_smm_match_register_listener,
    c_smm_match_set_config,
    c_smm_match_set_seed,
    c_smm_match_unregister,
    c_smm_match_worst_price,
    smm_engine_binding,
    smm_match_ctx,
    smm_match_event,
)
from algo_engine.exchange_profile.c_exchange_profile cimport EX_PROFILE_IMPORT

if EX_PROFILE_IMPORT() != 0:
    raise ImportError('exchange_profile globals not resolved — is the Windows build up to date?')


cdef void c_simmatch_ex_callback(smm_match_ctx* ctx, smm_match_event event,
                                 const smm_order_entry* entry,
                                 const md_trade_report* report,
                                 void* user_data) noexcept:
    cdef SimMatchEx sim = <SimMatchEx> user_data
    cdef object key
    cdef object order
    cdef TradeReport py_report
    try:
        if not entry or not entry.py_order:
            return
        order = <object> entry.py_order
        key = c_get_long_id(&entry.order_id)

        if event == SMM_EVENT_PLACED:
            sim.working[key] = order
            if not ctx.engine:
                sim.on_order(order=order)
        elif event == SMM_EVENT_MATCH:
            py_report = <TradeReport> report_from_header(<const md_variant*> report, False)
            py_report = <TradeReport> py_report.__copy__()
            (<TradeInstruction> order).trades[py_report.trade_id] = py_report
            if (<TradeInstruction> order).order_state_int == md_order_state.STATE_FILLED:
                sim.working.pop(key, None)
                sim.history[key] = order
            if not ctx.engine:
                sim.on_report(report=py_report)
                sim.on_order(order=order)
        else:
            sim.working.pop(key, None)
            sim.history[key] = order
            if not ctx.engine:
                sim.on_order(order=order)
    except:
        PyErr_Clear()


cdef class SimMatchEx:
    def __cinit__(self, str ticker, object event_engine=None, object topic_set=None,
                  object seed=None, **kwargs):
        self.owner = True

        cdef double fee_rate = float(kwargs.get('fee_rate', 0.))
        cdef bint instant_fill = bool(kwargs.get('instant_fill', False))
        cdef double lag_ts = float(kwargs.get('lag_ts', 0.))
        cdef uint64_t lag_n_transaction = int(kwargs.get('lag_n_transaction', 0))
        cdef double hit_prob = float(kwargs.get('hit_prob', 1.))
        cdef double slippery_rate = float(kwargs.get('slippery_rate', 0.0001))
        cdef bint incremental_order_volume = bool(kwargs.get('incremental_order_volume', False))

        self.ctx = c_smm_match_new(fee_rate, instant_fill, lag_ts, lag_n_transaction,
                                   hit_prob, slippery_rate, incremental_order_volume, NULL)
        if not self.ctx:
            raise MemoryError(f'Failed to allocate {self.__class__.__name__} context')

        cdef uintptr_t callback_id
        cdef int ret_code = c_smm_match_register_listener(
            self.ctx, c_simmatch_ex_callback, <void*> <PyObject*> self, &callback_id)
        if ret_code != SMM_OK:
            raise RuntimeError(f'Failed to register callback for {self.__class__.__name__}, error code {ret_code}')
        self._callback_id = callback_id

        self.ticker = ticker
        self.event_engine = event_engine if event_engine is not None else EVENT_ENGINE
        self.topic_set = topic_set if topic_set is not None else TOPIC
        self.working = {}
        self.history = {}

        cdef uint64_t py_seed
        if seed is not None:
            py_seed = <uint64_t> int(seed)
        else:
            py_seed = <uint64_t> py_random.getrandbits(64)
        self._seed = py_seed
        c_smm_match_set_seed(self.ctx, py_seed)
        self.random = py_random.Random(self._seed)

    def __dealloc__(self):
        if self._callback_id and self.ctx:
            c_smm_match_deregister_listener(self.ctx, self._callback_id)
        if self.ctx:
            if self.owner:
                c_smm_match_free(self.ctx)
            else:
                c_smm_match_dealloc(self.ctx)

    def __call__(self, **kwargs):
        cdef object order = kwargs.pop('order', None)
        cdef object market_data = kwargs.pop('market_data', None)

        if order is not None:
            if (<TradeInstruction> order).order_type_int == md_order_type.ORDER_LIMIT:
                self.launch_order(order=order)
            elif (<TradeInstruction> order).order_type_int == md_order_type.ORDER_CANCEL:
                self.cancel_order(order=order)
            else:
                raise ValueError(f'Invalid order {order}')

        if market_data is not None:
            self.c_process((<MarketData> market_data).header)

    cdef inline int c_process(self, const md_variant* market_data):
        return c_smm_match_process(self.ctx, market_data)

    @staticmethod
    def best_price(*price, side):
        cdef int8_t sign = <int8_t> side.sign
        cdef Py_ssize_t n = len(price)
        cdef double* prices = <double*> PyMem_Malloc(n * sizeof(double))
        if not prices:
            raise MemoryError(f'Failed to allocate {n} price slots')
        cdef Py_ssize_t i
        cdef object p
        cdef double out = 0.
        cdef int ret_code
        try:
            for i in range(n):
                p = price[i]
                if p is None:
                    prices[i] = NAN
                else:
                    prices[i] = <double> p
            ret_code = c_smm_match_best_price(prices, n, sign, &out)
            if ret_code == SMM_ERR_NO_VALID_PRICE:
                raise ValueError('No valid prices provided')
            elif ret_code == SMM_ERR_INVALID_SIDE:
                raise ValueError(f'Invalid side {side}!')
            return out
        finally:
            PyMem_Free(prices)

    @staticmethod
    def worst_price(*price, side):
        cdef int8_t sign = <int8_t> side.sign
        cdef Py_ssize_t n = len(price)
        cdef double* prices = <double*> PyMem_Malloc(n * sizeof(double))
        if not prices:
            raise MemoryError(f'Failed to allocate {n} price slots')
        cdef Py_ssize_t i
        cdef object p
        cdef double out = 0.
        cdef int ret_code
        try:
            for i in range(n):
                p = price[i]
                if p is None:
                    prices[i] = NAN
                else:
                    prices[i] = <double> p
            ret_code = c_smm_match_worst_price(prices, n, sign, &out)
            if ret_code == SMM_ERR_NO_VALID_PRICE:
                raise ValueError('No valid prices provided')
            elif ret_code == SMM_ERR_INVALID_SIDE:
                raise ValueError(f'Invalid side {side}!')
            return out
        finally:
            PyMem_Free(prices)

    def register(self, topic_set=None, event_engine=None):
        cdef EventEngine engine
        cdef TopicSet ts
        cdef Topic launch_topic
        cdef Topic cancel_topic
        cdef Topic realtime_topic
        cdef EventHook launch_hook
        cdef EventHook cancel_hook
        cdef EventHook realtime_hook
        cdef smm_engine_binding binding
        cdef int ret_code

        if topic_set is not None:
            self.topic_set = topic_set

        if event_engine is not None:
            self.event_engine = event_engine

        if isinstance(self.event_engine, EventEngine):
            engine = <EventEngine> self.event_engine
            ts = <TopicSet> self.topic_set
            launch_topic = ts.launch_order(ticker=self.ticker)
            cancel_topic = ts.cancel_order(ticker=self.ticker)
            realtime_topic = ts.realtime(ticker=self.ticker)

            launch_hook = EventHook.__new__(EventHook, launch_topic, engine.logger)
            cancel_hook = EventHook.__new__(EventHook, cancel_topic, engine.logger)
            realtime_hook = EventHook.__new__(EventHook, realtime_topic, engine.logger)

            memset(&binding, 0, sizeof(smm_engine_binding))
            binding.mq = engine.mq
            binding.exact_topic_hooks = engine.exact_topic_hooks
            binding.generic_topic_hooks = engine.generic_topic_hooks

            binding.launch_topic = launch_topic.header
            binding.launch_hook = launch_hook.header
            binding.launch_hook_obj = <PyObject*> launch_hook
            binding.cancel_topic = cancel_topic.header
            binding.cancel_hook = cancel_hook.header
            binding.cancel_hook_obj = <PyObject*> cancel_hook
            binding.realtime_topic = realtime_topic.header
            binding.realtime_hook = realtime_hook.header
            binding.realtime_hook_obj = <PyObject*> realtime_hook

            binding.on_order_topic = ts.on_order.header
            binding.on_order_topic_obj = <PyObject*> ts.on_order
            binding.on_report_topic = ts.on_report.header
            binding.on_report_topic_obj = <PyObject*> ts.on_report

            if self.ctx.engine:
                c_smm_match_unregister(self.ctx)

            ret_code = c_smm_match_register(self.ctx, &binding)
            if ret_code != SMM_OK:
                raise RuntimeError(
                    f'Failed to register {self.__class__.__name__} on the event engine, error code {ret_code}')
        else:
            self.event_engine.register_handler(
                topic=self.topic_set.launch_order(ticker=self.ticker), handler=self.launch_order)
            self.event_engine.register_handler(
                topic=self.topic_set.cancel_order(ticker=self.ticker), handler=self.cancel_order)
            self.event_engine.register_handler(
                topic=self.topic_set.realtime(ticker=self.ticker), handler=self)

    def unregister(self):
        cdef EventEngine engine
        if isinstance(self.event_engine, EventEngine):
            c_smm_match_unregister(self.ctx)
        else:
            self.event_engine.unregister_handler(
                topic=self.topic_set.launch_order(ticker=self.ticker), handler=self.launch_order)
            self.event_engine.unregister_handler(
                topic=self.topic_set.cancel_order(ticker=self.ticker), handler=self.cancel_order)
            self.event_engine.unregister_handler(
                topic=self.topic_set.realtime(ticker=self.ticker), handler=self)

    def launch_order(self, TradeInstruction order, **kwargs):
        cdef object order_id = order.order_id
        if order_id in self.working or order_id in self.history:
            raise ValueError(f'Invalid instruction {order}, OrderId already in working or history')

        if isnan(order.limit_price) and order.order_type_int == md_order_type.ORDER_LIMIT:
            LOGGER.warning(f'order {order} does not have a valid limit price!')

        order.transaction_count_at_placement = self.ctx.last_transaction_count

        cdef int ret_code = c_smm_match_launch(self.ctx, <md_variant*> order.header, <PyObject*> order)
        if ret_code != SMM_OK:
            raise ValueError(f'Invalid instruction {order}, OrderId already in working or history')

    def cancel_order(self, TradeInstruction order=None, object order_id=None, **kwargs):
        if order is None and order_id is None:
            raise ValueError('Must assign a order or order_id to cancel order')

        if order_id is None:
            order_id = order.order_id

        cdef object order_obj = self.working.get(order_id)
        if order_obj is None:
            LOGGER.info(f'[{self.market_time:%Y-%m-%d %H:%M:%S}] failed to cancel {order_id} order!')
            return

        cdef int ret_code = c_smm_match_cancel(
            self.ctx, &(<TradeInstruction> order_obj).header.trade_instruction.order_id)
        if ret_code != SMM_OK:
            LOGGER.info(f'[{self.market_time:%Y-%m-%d %H:%M:%S}] failed to cancel {order_id} order!')
            return

        LOGGER.info(f'[{self.market_time:%Y-%m-%d %H:%M:%S}] Sim-canceled '
                    f'{order_obj.side.name} {order_obj.ticker} order!')

    def eod(self):
        c_smm_match_eod(self.ctx)

    def clear(self):
        c_smm_match_clear(self.ctx)
        self.working.clear()
        self.history.clear()
        self.random = py_random.Random(self._seed)

    def on_order(self, order, **kwargs):
        self.event_engine.put(topic=self.topic_set.on_order, order=order)

    def on_report(self, report, **kwargs):
        self.event_engine.put(topic=self.topic_set.on_report, report=report, **kwargs)

    property timestamp:
        def __get__(self):
            return self.ctx.timestamp

        def __set__(self, double value):
            self.ctx.timestamp = value

    property last_price:
        def __get__(self):
            cdef double price = self.ctx.last_price
            if isnan(price):
                return None
            return price

        def __set__(self, object value):
            if value is None:
                self.ctx.last_price = NAN
            else:
                self.ctx.last_price = <double> value

    property last_transaction_count:
        def __get__(self):
            return self.ctx.last_transaction_count

        def __set__(self, uint64_t value):
            self.ctx.last_transaction_count = value

    property seed:
        def __get__(self):
            return self._seed

        def __set__(self, object value):
            if value is None:
                self._seed = <uint64_t> py_random.getrandbits(64)
            else:
                self._seed = <uint64_t> int(value)
            c_smm_match_set_seed(self.ctx, self._seed)
            self.random = py_random.Random(self._seed)

    property matching_config:
        def __get__(self):
            return {
                'fee_rate': self.ctx.fee_rate,
                'instant_fill': self.ctx.instant_fill,
                'lag': {
                    'ts': self.ctx.lag_ts,
                    'n_transaction': self.ctx.lag_n_transaction
                },
                'hit': {
                    'prob': self.ctx.hit_prob,
                    'slippery': self.ctx.slippery_rate
                },
                'incremental_order_volume': self.ctx.incremental_order_volume
            }

    property market_time:
        def __get__(self):
            return PROFILE.c_timestamp_to_datetime(self.ctx.timestamp)

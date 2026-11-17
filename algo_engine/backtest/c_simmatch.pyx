import random as py_random

from cpython.exc cimport PyErr_Clear
from cpython.mem cimport PyMem_Free, PyMem_Malloc
from cpython.object cimport PyObject
from event_engine.capi cimport EventEngineEx
from libc.math cimport NAN, isfinite, isnan
from libc.string cimport memcpy

from algo_engine.base.c_allocator_protocol cimport MD_DEFAULT_ALLOCATOR
from algo_engine.base.c_market_data.c_market_data cimport MarketData, c_get_long_id, md_order_state, md_order_type, md_trade_instruction
from algo_engine.base.c_market_data.c_trade_utils cimport TradeInstruction, TradeReport, instruction_from_header, report_from_header
from algo_engine.engine.c_event_engine cimport EVENT_ENGINE, TOPIC_SET, TopicSetEx
from algo_engine.engine.c_dma_protocol cimport DMAProtocol, dma_protocol_ret_code
from algo_engine.exchange_profile.c_exchange_profile cimport PROFILE

from . import LOGGER


cdef class SimMatch(CCPType):
    def __cinit__(self, str ticker, EventEngineEx event_engine=None, TopicSetEx topic_set=None, uint64_t seed=0, **kwargs):
        cdef double fee_rate = float(kwargs.get('fee_rate', 0.))
        cdef bint instant_fill = bool(kwargs.get('instant_fill', False))
        cdef double lag_ts = float(kwargs.get('lag_ts', 0.))
        cdef uint64_t lag_n_transaction = int(kwargs.get('lag_n_transaction', 0))
        cdef double hit_prob = float(kwargs.get('hit_prob', 1.))
        cdef double slippery_rate = float(kwargs.get('slippery_rate', 0.0001))
        cdef bint incremental_order_volume = bool(kwargs.get('incremental_order_volume', False))

        if event_engine is None:
            event_engine = EVENT_ENGINE
        if topic_set is None:
            topic_set = TOPIC_SET

        self.header = c_smm_match_new(fee_rate, instant_fill, lag_ts, lag_n_transaction,
                                      hit_prob, slippery_rate, incremental_order_volume,
                                      MD_DEFAULT_ALLOCATOR)
        if not self.header:
            raise MemoryError(f'Failed to allocate {self.__class__.__name__} context')

        self.owner = True
        try:
            self.ccp_bind(&self.header)
        except BaseException:
            c_smm_match_free(self.header)
            self.header = NULL
            raise

        cdef uintptr_t callback_id
        cdef int ret_code = c_smm_match_register_listener(
            self.header, SimMatch.c_simmatch_callback, <void*> <PyObject*> self, &callback_id)
        if ret_code != smm_ret_code.SMM_OK:
            c_smm_match_free(self.header)
            self.header = NULL
            raise RuntimeError(f'Failed to register callback for {self.__class__.__name__}, error code {ret_code}')
        self.callback_id = callback_id

        self.ticker = ticker
        self.event_engine = event_engine
        self.topic_set = topic_set
        self.working = {}
        self.history = {}

        cdef uint64_t py_seed = seed
        if py_seed == 0:
            py_seed = <uint64_t> py_random.getrandbits(64)
        self.seed = py_seed
        cdef int seed_code = c_smm_match_set_seed(self.header, py_seed)
        if seed_code != smm_ret_code.SMM_OK:
            raise RuntimeError(f'Failed to seed <{self.__class__.__name__}>, code {seed_code}')
        self.random = py_random.Random(self.seed)

    def __dealloc__(self):
        cdef DMAProtocol protocol = self.protocol
        self.protocol = None
        if protocol is not None:
            try:
                self.c_unregister_protocol(protocol)
            except BaseException:
                PyErr_Clear()

        if self.header and self.callback_id:
            c_smm_match_deregister_listener(self.header, self.callback_id)
            self.callback_id = 0

        if not self.owner:
            return

        if self.header:
            c_smm_match_free(self.header)
            self.header = NULL

    @staticmethod
    cdef inline void c_simmatch_callback(smm_match_ctx* ctx, smm_match_event event, const smm_order_entry* entry, const md_trade_report* report, void* user_data) noexcept:
        cdef SimMatch sim = <SimMatch> user_data
        cdef object key
        cdef TradeInstruction order_obj
        cdef TradeInstruction borrowed
        cdef TradeInstruction order
        cdef TradeReport py_report
        try:
            if not entry or not entry.order:
                return

            key = c_get_long_id(&entry.order_id)
            order_obj = sim.working.get(key)
            if order_obj is None:
                borrowed = instruction_from_header(<const md_variant*> entry.order, False)
                order = <TradeInstruction> borrowed.__copy__()
            else:
                order = order_obj
                memcpy(<void*> (<md_variant*> order.header), <const void*> entry.order, sizeof(md_trade_instruction))

            if event == smm_match_event.SMM_EVENT_PLACED:
                sim.working[key] = order
                sim.on_order(order=order)
            elif event == smm_match_event.SMM_EVENT_MATCH:
                py_report = <TradeReport> report_from_header(<const md_variant*> report, False)
                py_report = <TradeReport> py_report.__copy__()
                order.trades[py_report.trade_id] = py_report
                if order.order_state_int == md_order_state.STATE_FILLED:
                    sim.working.pop(key, None)
                    sim.history[key] = order
                sim.on_report(report=py_report)
                sim.on_order(order=order)
            else:
                sim.working.pop(key, None)
                sim.history[key] = order
                sim.on_order(order=order)
        except:
            PyErr_Clear()

    @staticmethod
    cdef inline double c_price_extreme(tuple prices, int8_t sign, bint select_best, object side) except *:
        cdef size_t n = <size_t> len(prices)
        if n == 0:
            raise ValueError('No valid prices provided')

        cdef double* values = <double*> PyMem_Malloc(n * sizeof(double))
        if not values:
            raise MemoryError(f'Failed to allocate {n} price slots')

        cdef size_t i
        cdef object value
        cdef double out
        cdef int ret_code
        cdef str which
        try:
            for i in range(n):
                value = prices[i]
                if value is None:
                    values[i] = NAN
                else:
                    values[i] = <double> value

            if select_best:
                ret_code = c_smm_match_best_price(values, n, sign, &out)
            else:
                ret_code = c_smm_match_worst_price(values, n, sign, &out)

            if ret_code == smm_ret_code.SMM_ERR_NO_VALID_PRICE:
                raise ValueError('No valid prices provided')
            if ret_code == smm_ret_code.SMM_ERR_INVALID_SIDE:
                raise ValueError(f'Invalid side {side}')
            if ret_code != smm_ret_code.SMM_OK:
                which = 'best' if select_best else 'worst'
                raise RuntimeError(f'Failed to select the {which} price, code {ret_code}')
            return out
        finally:
            PyMem_Free(values)

    cdef inline int c_process(self, const md_variant* market_data) except -1:
        cdef int ret_code = c_smm_match_process(self.header, market_data)
        if ret_code == smm_ret_code.SMM_OK:
            return ret_code
        if ret_code == smm_ret_code.SMM_ERR_INVALID_ARG:
            raise ValueError(f'Invalid market data for <{self.__class__.__name__}>')
        raise RuntimeError(f'Failed to process market data for <{self.__class__.__name__}>, code {ret_code}')

    cdef void c_unregister_protocol(self, DMAProtocol protocol) except *:
        cdef TopicSetEx ts = self.topic_set
        cdef int ret_code = protocol.unregister_callback(self._on_realtime, ts.realtime(ticker=self.ticker))
        if ret_code == dma_protocol_ret_code.DMA_PROTOCOL_OK or ret_code == dma_protocol_ret_code.DMA_PROTOCOL_ERR_NOT_FOUND:
            return
        raise RuntimeError(f'Failed to unregister the realtime callback, code {ret_code}')

    def __repr__(self):
        if not self.header:
            return f'<{self.__class__.__name__}: unbound>'
        return f'<{self.__class__.__name__}(ticker={self.ticker!r})>'

    def __call__(self, **kwargs):
        if not self.header:
            raise RuntimeError(f'Uninitialized <{self.__class__.__name__}>')

        cdef TradeInstruction order = kwargs.pop('order', None)
        cdef MarketData market_data = kwargs.pop('market_data', None)

        if order is not None:
            if order.order_type_int == md_order_type.ORDER_LIMIT:
                self.launch_order(order=order)
            elif order.order_type_int == md_order_type.ORDER_CANCEL:
                self.cancel_order(order=order)
            else:
                raise ValueError(f'Invalid order {order}')

        if market_data is not None:
            self.c_process(market_data.header)

    def _on_realtime(self, MarketData market_data):
        if not self.header or not market_data.header:
            return
        self.c_process(market_data.header)

    @staticmethod
    def best_price(*price, side):
        return SimMatch.c_price_extreme(price, <int8_t> side.sign, True, side)

    @staticmethod
    def worst_price(*price, side):
        return SimMatch.c_price_extreme(price, <int8_t> side.sign, False, side)

    def register(self, TopicSetEx topic_set=None, EventEngineEx event_engine=None):
        if not self.header:
            raise RuntimeError(f'Uninitialized <{self.__class__.__name__}>')

        cdef TopicSetEx ts
        cdef DMAProtocol protocol = None
        cdef bint launch_bound = False
        cdef bint cancel_bound = False

        self.unregister()

        if topic_set is not None:
            self.topic_set = topic_set

        if event_engine is not None:
            self.event_engine = event_engine

        ts = self.topic_set
        self.registered = True
        try:
            self.event_engine.register_handler(
                topic=ts.launch_order(ticker=self.ticker), handler=self.launch_order)
            launch_bound = True
            self.event_engine.register_handler(
                topic=ts.cancel_order(ticker=self.ticker), handler=self.cancel_order)
            cancel_bound = True

            protocol = DMAProtocol(self.event_engine, ts)
            self.protocol = protocol
            protocol.config(False)
            protocol.register_callback(self._on_realtime, ts.realtime(ticker=self.ticker))
        except BaseException:
            if protocol is not None:
                try:
                    self.c_unregister_protocol(protocol)
                except BaseException:
                    PyErr_Clear()
                self.protocol = None
            if launch_bound:
                try:
                    self.event_engine.unregister_handler(
                        topic=ts.launch_order(ticker=self.ticker), handler=self.launch_order)
                except BaseException:
                    PyErr_Clear()
            if cancel_bound:
                try:
                    self.event_engine.unregister_handler(
                        topic=ts.cancel_order(ticker=self.ticker), handler=self.cancel_order)
                except BaseException:
                    PyErr_Clear()
            self.registered = False
            raise

    def unregister(self):
        cdef DMAProtocol protocol = self.protocol
        cdef TopicSetEx ts = self.topic_set

        if not self.registered:
            return

        if protocol is not None:
            self.c_unregister_protocol(protocol)
            self.protocol = None

        self.event_engine.unregister_handler(
            topic=ts.launch_order(ticker=self.ticker), handler=self.launch_order)
        self.event_engine.unregister_handler(
            topic=ts.cancel_order(ticker=self.ticker), handler=self.cancel_order)

        self.registered = False

    def launch_order(self, TradeInstruction order, **kwargs):
        if not self.header:
            raise RuntimeError(f'Uninitialized <{self.__class__.__name__}>')

        cdef object order_id = order.order_id
        if order_id in self.working or order_id in self.history:
            raise ValueError(f'Invalid instruction {order}, OrderId already in working or history')

        if not order.header:
            raise ValueError(f'Invalid instruction {order}, order buffer is not bound')

        if order.order_type_int == md_order_type.ORDER_LIMIT and not isfinite(order.limit_price):
            LOGGER.warning(f'order {order} does not have a valid limit price!')

        cdef TradeInstruction working_order = order.__copy__()
        working_order.trades = {}
        working_order.transaction_count_at_placement = self.header.last_transaction_count
        self.working[order_id] = working_order

        cdef int ret_code = c_smm_match_launch(self.header, &order.header.trade_instruction)
        if ret_code == smm_ret_code.SMM_OK:
            return

        self.working.pop(order_id, None)
        if ret_code == smm_ret_code.SMM_ERR_DUPLICATE:
            raise ValueError(f'Invalid instruction {order}, OrderId already in the working registry')
        if ret_code == smm_ret_code.SMM_ERR_INVALID_ARG:
            raise ValueError(f'Invalid instruction {order}, launch rejected the argument')
        if ret_code == smm_ret_code.SMM_ERR_OOM:
            raise MemoryError(f'Failed to launch {order}, the working registry is out of memory')
        raise RuntimeError(f'Failed to launch {order}, code {ret_code}')

    def cancel_order(self, TradeInstruction order=None, object order_id=None, **kwargs):
        if not self.header:
            raise RuntimeError(f'Uninitialized <{self.__class__.__name__}>')

        if order is None and order_id is None:
            raise ValueError('Must assign a order or order_id to cancel order')

        if order_id is None:
            order_id = order.order_id

        cdef TradeInstruction working_order = self.working.get(order_id)
        if working_order is None:
            LOGGER.info(f'[{self.market_time:%Y-%m-%d %H:%M:%S}] failed to cancel {order_id} order!')
            return

        cdef int ret_code = c_smm_match_cancel(
            self.header, &working_order.header.trade_instruction.order_id)
        if ret_code == smm_ret_code.SMM_ERR_NOT_FOUND:
            LOGGER.info(f'[{self.market_time:%Y-%m-%d %H:%M:%S}] failed to cancel {order_id} order!')
            return
        if ret_code != smm_ret_code.SMM_OK:
            raise RuntimeError(f'Failed to cancel {order_id}, code {ret_code}')

        LOGGER.info(f'[{self.market_time:%Y-%m-%d %H:%M:%S}] Sim-canceled '
                    f'{working_order.side.name} {working_order.ticker} order!')

    def eod(self):
        if not self.header:
            raise RuntimeError(f'Uninitialized <{self.__class__.__name__}>')
        c_smm_match_eod(self.header)

    def clear(self):
        if not self.header:
            raise RuntimeError(f'Uninitialized <{self.__class__.__name__}>')

        c_smm_match_clear(self.header)
        self.working.clear()
        self.history.clear()
        self.random = py_random.Random(self.seed)

    def on_order(self, TradeInstruction order, **kwargs):
        self.event_engine.put(topic=self.topic_set.on_order, order=order)

    def on_report(self, TradeReport report, **kwargs):
        self.event_engine.put(topic=self.topic_set.on_report, report=report, **kwargs)

    property timestamp:
        def __get__(self):
            if not self.header:
                raise RuntimeError(f'Uninitialized <{self.__class__.__name__}>')
            return self.header.timestamp

        def __set__(self, double value):
            if not self.header:
                raise RuntimeError(f'Uninitialized <{self.__class__.__name__}>')
            self.header.timestamp = value

    property last_price:
        def __get__(self):
            if not self.header:
                raise RuntimeError(f'Uninitialized <{self.__class__.__name__}>')
            cdef double price = self.header.last_price
            if isnan(price):
                return None
            return price

        def __set__(self, object value):
            if not self.header:
                raise RuntimeError(f'Uninitialized <{self.__class__.__name__}>')
            if value is None:
                self.header.last_price = NAN
            else:
                self.header.last_price = <double> value

    property last_transaction_count:
        def __get__(self):
            if not self.header:
                raise RuntimeError(f'Uninitialized <{self.__class__.__name__}>')
            return self.header.last_transaction_count

        def __set__(self, uint64_t value):
            if not self.header:
                raise RuntimeError(f'Uninitialized <{self.__class__.__name__}>')
            self.header.last_transaction_count = value

    property matching_config:
        def __get__(self):
            if not self.header:
                raise RuntimeError(f'Uninitialized <{self.__class__.__name__}>')
            return {
                'fee_rate': self.header.fee_rate,
                'instant_fill': self.header.instant_fill,
                'lag': {
                    'ts': self.header.lag_ts,
                    'n_transaction': self.header.lag_n_transaction
                },
                'hit': {
                    'prob': self.header.hit_prob,
                    'slippery': self.header.slippery_rate
                },
                'incremental_order_volume': self.header.incremental_order_volume
            }

    property market_time:
        def __get__(self):
            if not self.header:
                raise RuntimeError(f'Uninitialized <{self.__class__.__name__}>')
            return PROFILE.c_timestamp_to_datetime(self.header.timestamp)

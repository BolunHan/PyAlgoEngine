import random as py_random

from cpython.exc cimport PyErr_Clear
from cpython.mem cimport PyMem_Free, PyMem_Malloc
from cpython.object cimport PyObject
from libc.math cimport NAN, isfinite, isnan

from cbase.bytemap cimport BoundByteMap, bytemap, bytemap_entry, c_bytemap_gen_seq_id
from event_engine.capi cimport EventEngineEx

from algo_engine.base.c_allocator_protocol cimport MD_DEFAULT_ALLOCATOR
from algo_engine.base.c_market_data.c_market_data cimport MarketData, c_get_long_id, c_set_long_id, long_md_id, md_order_type, md_trade_report, md_variant
from algo_engine.base.c_market_data.c_trade_utils cimport TradeInstruction, TradeReport, instruction_from_header, report_from_header
from algo_engine.engine.c_event_engine cimport EVENT_ENGINE, TOPIC_SET, TopicSetEx
from algo_engine.engine.c_dma_protocol cimport DMAProtocol, dma_protocol_ret_code
from algo_engine.exchange_profile.c_exchange_profile cimport PROFILE

from . import LOGGER


cdef class OrderMapView(BoundByteMap):
    @staticmethod
    cdef OrderMapView c_from_header(bytemap* header, bint owner=False):
        if owner:
            raise PermissionError('Must not assign the order registry ownership to <OrderMapView>')

        cdef OrderMapView instance = OrderMapView.__new__(OrderMapView)
        instance.owner = False
        instance.seq_id = c_bytemap_gen_seq_id(<void*> instance)
        instance.c_bind(header)
        return instance

    cdef const char* c_serialize_key(self, object obj, size_t* key_len):
        raise PermissionError(f'Must not assign an order to <{self.__class__.__name__}>')

    cdef object c_deserialize_key(self, const char* key, size_t key_len):
        return c_get_long_id(<const long_md_id*> key)

    cdef const char* c_serialize_value(self, object obj, size_t* value_len):
        raise PermissionError(f'Must not assign an order to <{self.__class__.__name__}>')

    @staticmethod
    cdef inline TradeInstruction c_compose(const smm_order_entry* entry):
        cdef TradeInstruction order = <TradeInstruction> instruction_from_header(<const md_variant*> entry, False)
        if not entry.reports:
            return order

        # Mirror the native report copies into the order's trades record.
        cdef bytemap_entry* node = entry.reports.first
        cdef md_trade_report* report
        cdef TradeReport py_report
        while node:
            report = (<md_trade_report**> node.value)[0]
            if report:
                py_report = <TradeReport> report_from_header(<const md_variant*> report, False)
                order.trades[py_report.trade_id] = py_report
            node = node.next
        return order

    cdef object c_deserialize_value(self, const char* value, size_t value_len):
        cdef smm_order_entry* entry = (<smm_order_entry**> value)[0]
        if not entry:
            return None
        return OrderMapView.c_compose(entry)


cdef class SimMatch(CCPType):
    def __init__(self, str ticker, EventEngineEx event_engine=None, TopicSetEx topic_set=None, uint64_t seed=0, **kwargs):
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

        self.header = c_smm_match_new(fee_rate, instant_fill, lag_ts, lag_n_transaction, hit_prob, slippery_rate, incremental_order_volume, MD_DEFAULT_ALLOCATOR)
        if not self.header:
            raise MemoryError(f'Failed to allocate {self.__class__.__name__} context')

        self.owner = True
        self.ccp_bind(&self.header)

        cdef int ret_code = c_smm_match_register_listener(self.header, SimMatch.c_simmatch_callback, <void*> <PyObject*> self, &self.callback_id)
        if ret_code != smm_ret_code.SMM_OK:
            c_smm_match_free(self.header)
            self.header = NULL
            raise RuntimeError(f'Failed to register callback for {self.__class__.__name__}, error code {ret_code}')

        self.ticker = ticker
        self.event_engine = event_engine
        self.topic_set = topic_set
        self.working = OrderMapView.c_from_header(self.header.working, False)
        self.history = OrderMapView.c_from_header(self.header.history, False)

        if seed == 0:
            seed = <uint64_t> py_random.getrandbits(64)
        self.seed = seed
        cdef int seed_code = c_smm_match_set_seed(self.header, seed)
        if seed_code != smm_ret_code.SMM_OK:
            raise RuntimeError(f'Failed to seed <{self.__class__.__name__}>, code {seed_code}')
        self.random = py_random.Random(self.seed)

    @staticmethod
    cdef SimMatch c_from_header(smm_match_ctx* header, str ticker, bint owner=False):
        if not header:
            raise ValueError('SimMatch requires a matcher context')

        cdef SimMatch instance = SimMatch.__new__(SimMatch)
        instance.header = header
        instance.owner = owner
        instance.ccp_bind(&instance.header)
        instance.ticker = ticker
        instance.event_engine = EVENT_ENGINE
        instance.topic_set = TOPIC_SET
        instance.seed = header.seed
        instance.random = py_random.Random(header.seed)
        instance.working = OrderMapView.c_from_header(header.working, False)
        instance.history = OrderMapView.c_from_header(header.history, False)

        cdef int ret_code = c_smm_match_register_listener(header, SimMatch.c_simmatch_callback, <void*> <PyObject*> instance, &instance.callback_id)
        if ret_code != smm_ret_code.SMM_OK:
            raise RuntimeError(f'Failed to register callback for <{SimMatch.__name__}>, error code {ret_code}')
        return instance

    def __dealloc__(self):
        if self.protocol is not None:
            self.c_unregister_protocol(self.protocol)

        if self.header and self.callback_id:
            c_smm_match_deregister_listener(self.header, self.callback_id)

        if not self.owner:
            return

        if self.header:
            c_smm_match_free(self.header)

    @staticmethod
    cdef inline void c_simmatch_callback(smm_match_ctx* ctx, smm_match_event event, const smm_order_entry* entry, const md_trade_report* report, void* user_data) noexcept:
        cdef SimMatch sim = <SimMatch> <object> <PyObject*> user_data
        cdef object key
        cdef TradeInstruction order
        cdef TradeReport py_report
        try:
            if not entry:
                return

            key = c_get_long_id(&entry.order.order_id)
            order = sim.working.get(key)
            if order is None:
                order = sim.history.get(key)
            if order is None:
                order = OrderMapView.c_compose(entry)

            if event == smm_match_event.SMM_EVENT_MATCH:
                py_report = <TradeReport> report_from_header(<const md_variant*> report, False)
                sim.on_report(report=py_report)
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

    cpdef void _on_realtime(self, MarketData market_data):
        if not self.header or not market_data.header:
            return
        self.c_process(market_data.header)

    @staticmethod
    def best_price(*price, side):
        return SimMatch.c_price_extreme(price, <int8_t> side.sign, True, side)

    @staticmethod
    def worst_price(*price, side):
        return SimMatch.c_price_extreme(price, <int8_t> side.sign, False, side)

    cpdef void register(self, TopicSetEx topic_set=None, EventEngineEx event_engine=None):
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

    cpdef void unregister(self):
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

        if not order.header:
            raise ValueError(f'Invalid instruction {order}, order buffer is not bound')

        if order.order_type_int == md_order_type.ORDER_LIMIT and not isfinite(order.limit_price):
            LOGGER.warning(f'order {order} does not have a valid limit price!')

        cdef int ret_code = c_smm_match_launch(self.header, &order.header.trade_instruction)
        if ret_code == smm_ret_code.SMM_OK:
            return

        if ret_code == smm_ret_code.SMM_ERR_DUPLICATE:
            raise ValueError(f'Invalid instruction {order}, OrderId already in working or history')
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

        cdef long_md_id key
        if order is not None:
            order_id = order.order_id
            key = order.header.trade_instruction.order_id
        else:
            c_set_long_id(&key, order_id)

        cdef int ret_code = c_smm_match_cancel(self.header, &key)
        if ret_code == smm_ret_code.SMM_OK:
            LOGGER.info(f'[{self.market_time:%Y-%m-%d %H:%M:%S}] Sim-canceled {self.ticker} order {order_id}!')
            return
        if ret_code == smm_ret_code.SMM_ERR_NOT_FOUND:
            LOGGER.info(f'[{self.market_time:%Y-%m-%d %H:%M:%S}] failed to cancel {order_id} order!')
            return
        raise RuntimeError(f'Failed to cancel {order_id}, code {ret_code}')

    cpdef void eod(self):
        if not self.header:
            raise RuntimeError(f'Uninitialized <{self.__class__.__name__}>')
        c_smm_match_eod(self.header)

    cpdef void clear(self):
        if not self.header:
            raise RuntimeError(f'Uninitialized <{self.__class__.__name__}>')

        c_smm_match_clear(self.header)
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

    property last_bid_price:
        def __get__(self):
            if not self.header:
                raise RuntimeError(f'Uninitialized <{self.__class__.__name__}>')
            cdef double price = self.header.last_bid_price
            if isnan(price):
                return None
            return price

        def __set__(self, object value):
            if not self.header:
                raise RuntimeError(f'Uninitialized <{self.__class__.__name__}>')
            if value is None:
                self.header.last_bid_price = NAN
            else:
                self.header.last_bid_price = <double> value

    property last_ask_price:
        def __get__(self):
            if not self.header:
                raise RuntimeError(f'Uninitialized <{self.__class__.__name__}>')
            cdef double price = self.header.last_ask_price
            if isnan(price):
                return None
            return price

        def __set__(self, object value):
            if not self.header:
                raise RuntimeError(f'Uninitialized <{self.__class__.__name__}>')
            if value is None:
                self.header.last_ask_price = NAN
            else:
                self.header.last_ask_price = <double> value

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

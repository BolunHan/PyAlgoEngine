# cython: language_level=3
from cpython.buffer cimport PyBUF_SIMPLE, PyObject_GetBuffer, PyBuffer_Release
from cpython.bytes cimport PyBytes_FromStringAndSize
from libc.stdlib cimport malloc, free, qsort
from libc.string cimport memcpy, memset
from libc.stdint cimport uint8_t, INT8_MAX, uint32_t, uint64_t, uintptr_t
from libc.time cimport time, time_t, difftime

from .c_market_data cimport compare_md_ptr, _MarketDataVirtualBase, InternalData, DataType, _MetaInfo, _InternalBuffer, _MarketDataBuffer, _TransactionDataBuffer, _OrderDataBuffer, _TickDataLiteBuffer, _TickDataBuffer, _CandlestickBuffer, TICKER_SIZE, Side, platform_usleep
from .c_transaction cimport TransactionData, OrderData, TransactionHelper
from .c_tick cimport TickData, TickDataLite, OrderBook
from .c_candlestick cimport BarData


cdef class MarketDataBuffer:
    def __cinit__(self, object buffer, bint skip_initialize=False, int capacity=0):
        # step 1: obtain a buffer view and pointer from the given python buffer object
        PyObject_GetBuffer(buffer, &self._view, PyBUF_SIMPLE)
        self._view_obtained = True
        self._buffer = <char*> self._view.buf

        if skip_initialize:
            return

        # step 2: prepare the header
        self._header = <_BufferHeader*> self._buffer
        cdef size_t header_size = sizeof(_BufferHeader)

        # step 2.1: --- pointer array ---
        self._header.ptr_offset = header_size
        self._ptr_array = <uint64_t*> (<void*> self._header + self._header.ptr_offset)
        cdef uint64_t total_size = self._view.len
        cdef size_t pointer_size = sizeof(uint64_t)
        cdef size_t estimated_entry_size = _MarketDataVirtualBase.c_max_size()
        self._estimated_entry_size = estimated_entry_size
        cdef size_t minimal_size_required = header_size + pointer_size * max(1, capacity) + estimated_entry_size
        if minimal_size_required > total_size:
            PyBuffer_Release(&self._view)
            raise MemoryError(f'Buffer is too small. Minimal size required: {minimal_size_required}')
        cdef uint32_t ptr_capacity = capacity if capacity > 0 else (total_size - header_size) // (pointer_size + estimated_entry_size)
        self._header.ptr_capacity = ptr_capacity
        self._ptr_capacity = ptr_capacity
        cdef size_t ptr_array_size = pointer_size * ptr_capacity

        # step 2.2:--- data array ---
        cdef uint64_t data_capacity = total_size - header_size - ptr_array_size
        self._header.data_offset = header_size + ptr_array_size
        self._data_array = <char*> self._header + self._header.data_offset
        self._header.data_capacity = data_capacity
        self._data_capacity = data_capacity

        # step 3: cleanup headers
        self._header.ptr_tail = 0
        self._header.data_tail = 0

        # step 4: optional cleanups, the reset of the buffer need not be clean but just in case
        memset(<void*> self._ptr_array, 0, ptr_array_size)
        memset(<void*> self._data_array, 0, data_capacity)

    def __dealloc__(self):
        """
        Release the buffer view when the object is deallocated.
        """
        if self._view_obtained:
            PyBuffer_Release(&self._view)
            self._view_obtained = False

    @staticmethod
    cdef size_t c_buffer_size(uint32_t n_internal_data=0, uint32_t n_transaction_data=0, uint32_t n_order_data=0, uint32_t n_tick_data_lite=0, uint32_t n_tick_data=0, uint32_t n_bar_data=0):
        cdef size_t header_size = sizeof(_BufferHeader)
        cdef size_t capacity = n_internal_data + n_transaction_data + n_order_data + n_tick_data_lite + n_tick_data + n_bar_data
        cdef size_t offset_size = sizeof(uint64_t) * capacity
        cdef size_t buffer_size = header_size + offset_size
        buffer_size += n_internal_data * sizeof(_InternalBuffer)
        buffer_size += n_transaction_data * sizeof(_TransactionDataBuffer)
        buffer_size += n_order_data * sizeof(_OrderDataBuffer)
        buffer_size += n_tick_data_lite * sizeof(_TickDataLiteBuffer)
        buffer_size += n_tick_data * sizeof(_TickDataBuffer)
        buffer_size += n_bar_data * sizeof(_CandlestickBuffer)
        return buffer_size

    @staticmethod
    cdef MarketDataBuffer c_from_buffer(object buffer):
        # Create instance without initialization
        cdef MarketDataBuffer instance = MarketDataBuffer.__new__(MarketDataBuffer, buffer, True)

        # restore from buffer
        instance._header = <_BufferHeader*> instance._buffer
        instance._ptr_capacity = instance._header.ptr_capacity
        instance._ptr_array = <uint64_t*> (<void*> instance._header + instance._header.ptr_offset)
        instance._estimated_entry_size = _MarketDataVirtualBase.c_max_size()
        instance._data_array = <char*> instance._header + instance._header.data_offset

        # reset data capacity based on the actual data size
        cdef uint64_t total_size = instance._view.len
        cdef size_t header_size = sizeof(_BufferHeader)
        cdef size_t pointer_size = sizeof(uint64_t)
        cdef size_t ptr_array_size = pointer_size * instance._ptr_capacity
        cdef uint64_t data_capacity = total_size - header_size - ptr_array_size
        instance._data_capacity = data_capacity
        instance._header.data_capacity = data_capacity

        return instance

    cdef bytes c_to_bytes(self):
        if self._buffer == NULL:
            return b''

        # Calculate total size (header + pointer array + used data)
        cdef uint64_t total_size = self._header.data_offset + self._header.data_tail
        cdef uint64_t data_capacity = self._header.data_capacity

        self._header.data_capacity = self._header.data_tail
        data = PyBytes_FromStringAndSize(self._buffer, total_size)
        self._header.data_capacity = data_capacity

        return data

    @staticmethod
    cdef c_from_bytes(bytes data, object buffer=None):
        """
        Create a MarketDataBuffer from bytes data, storing it in the provided buffer.

        Parameters:
        -----------
        data : bytes
            The serialized MarketDataBuffer data
        buffer : object
            A Python object supporting the buffer protocol where the data will be stored
        """
        if buffer is None:
            buffer = bytearray(data)  # create mutable buffer from bytes

        cdef MarketDataBuffer instance = MarketDataBuffer.__new__(MarketDataBuffer, buffer, False)

        # Copy data to buffer
        cdef size_t data_size = len(data)
        cdef size_t total_size = instance._view.len
        if data_size > total_size:
            PyBuffer_Release(&instance._view)
            instance._view_obtained = False
            raise ValueError(f"Buffer size {total_size} is too small for data size {data_size}")

        memcpy(instance._buffer, <const char*> data, data_size)

        # restore from buffer
        instance._header = <_BufferHeader*> instance._buffer
        instance._ptr_capacity = instance._header.ptr_capacity
        instance._ptr_array = <uint64_t*> (<void*> instance._header + instance._header.ptr_offset)
        instance._estimated_entry_size = _MarketDataVirtualBase.c_max_size()
        instance._data_array = <char*> instance._header + instance._header.data_offset

        # reset data capacity based on the actual data size
        cdef size_t header_size = sizeof(_BufferHeader)
        cdef size_t pointer_size = sizeof(uint64_t)
        cdef size_t ptr_array_size = pointer_size * instance._ptr_capacity
        cdef uint64_t data_capacity = total_size - header_size - ptr_array_size
        instance._data_capacity = data_capacity
        instance._header.data_capacity = data_capacity

        return instance

    cdef void c_put(self, _MarketDataBuffer* market_data_ptr):
        cdef uint8_t entry_dtype = market_data_ptr.MetaInfo.dtype
        cdef double entry_timestamp = market_data_ptr.MetaInfo.timestamp
        cdef size_t entry_size = _MarketDataVirtualBase.c_get_size(entry_dtype)

        # Check if we have enough space in the pointer array
        if self._header.ptr_tail >= self._header.ptr_capacity:
            raise MemoryError(f"Not enough space in pointer array for new entry, pointer capacity {self._header.ptr_capacity}")

        # Check if we have enough space in the data section
        if self._header.data_tail + entry_size > self._header.data_capacity:
            raise MemoryError(f"Not enough space in buffer for new entry, requested {entry_size}, remaining {self._header.data_capacity - self._header.data_tail}")

        # Copy data directly from market_data._data to our buffer at the current tail offset
        memcpy(self._data_array + self._header.data_tail, <const char*> market_data_ptr, entry_size)

        # Add pointer (offset) to the new entry
        self._ptr_array[self._header.ptr_tail] = self._header.data_tail
        self._header.ptr_tail += 1
        self._header.data_tail += entry_size

        # Update current_timestamp if needed
        if entry_timestamp >= self._header.current_timestamp:
            self._header.current_timestamp = entry_timestamp
        # New entry has earlier timestamp, need to resort
        else:
            self._header.sorted = 0

    cdef object c_get(self, uint32_t idx):
        if idx >= self._header.ptr_tail:
            raise IndexError(f'{self.__class__.__name__} index {idx} out of range {self._header.ptr_tail}')

        cdef uint64_t data_offset = self._ptr_array[idx]
        cdef _MetaInfo* ptr = <_MetaInfo*> (<void*> self._data_array + data_offset)
        cdef uint8_t dtype = ptr.dtype
        cdef size_t length = _MarketDataVirtualBase.c_get_size(dtype)
        cdef InternalData internal_data
        cdef TransactionData transaction_data
        cdef OrderData order_data
        cdef TickDataLite tick_data_lite
        cdef TickData tick_data
        cdef BarData bar_data

        if dtype == DataType.DTYPE_INTERNAL:
            internal_data = InternalData.__new__(InternalData)
            memcpy(<char*> internal_data._data_ptr, <const char*> self._data_array + data_offset, length)
            return internal_data
        elif dtype == DataType.DTYPE_TRANSACTION:
            transaction_data = TransactionData.__new__(TransactionData)
            memcpy(<char*> transaction_data._data_ptr, <const char*> self._data_array + data_offset, length)
            return transaction_data
        elif dtype == DataType.DTYPE_ORDER:
            order_data = OrderData.__new__(OrderData)
            memcpy(<char*> order_data._data_ptr, <const char*> self._data_array + data_offset, length)
            return order_data
        elif dtype == DataType.DTYPE_TICK_LITE:
            tick_data_lite = TickDataLite.__new__(TickDataLite)
            memcpy(<char*> tick_data_lite._data_ptr, <const char*> self._data_array + data_offset, length)
            return tick_data_lite
        elif dtype == DataType.DTYPE_TICK:
            tick_data = TickData.__new__(TickData)
            memcpy(<char*> tick_data._data_ptr, <const char*> self._data_array + data_offset, length)
            tick_data._init_order_book()
            return tick_data
        elif dtype == DataType.DTYPE_BAR:
            bar_data = BarData.__new__(BarData)
            memcpy(<char*> bar_data._data_ptr, <const char*> self._data_array + data_offset, length)
            return bar_data
        else:
            raise ValueError(f'Unknown data type {dtype}')

    @staticmethod
    cdef void _set_internal_fields(void* buffer, uint32_t code):
        cdef _InternalBuffer* data = <_InternalBuffer*> buffer

        data.code = code

    @staticmethod
    cdef void _set_transaction_fields(void* buffer, double price, double volume, uint8_t side, double multiplier=1.0, double notional=0.0, object transaction_id=None, object buy_id=None, object sell_id=None):
        cdef _TransactionDataBuffer* data = <_TransactionDataBuffer*> buffer

        data.price = price
        data.volume = volume
        data.side = side
        data.multiplier = multiplier

        if notional == 0.0:
            data.notional = price * volume * multiplier
        else:
            data.notional = notional

        TransactionHelper.set_id(id_ptr=&data.transaction_id, id_value=transaction_id)
        TransactionHelper.set_id(id_ptr=&data.buy_id, id_value=buy_id)
        TransactionHelper.set_id(id_ptr=&data.sell_id, id_value=sell_id)

    @staticmethod
    cdef void _set_order_fields(void* buffer, double price, double volume, uint8_t side, object order_id=None, uint8_t order_type=0):
        cdef _OrderDataBuffer* data = <_OrderDataBuffer*> buffer

        data.price = price
        data.volume = volume
        data.side = side
        data.order_type = order_type

        TransactionHelper.set_id(id_ptr=&data.order_id, id_value=order_id)

    @staticmethod
    cdef void _set_tick_lite_fields(void* buffer, double last_price, double bid_price, double bid_volume, double ask_price, double ask_volume, double total_traded_volume=0.0, double total_traded_notional=0.0, uint32_t total_trade_count=0):
        cdef _TickDataLiteBuffer* data = <_TickDataLiteBuffer*> buffer

        data.last_price = last_price
        data.bid_price = bid_price
        data.bid_volume = bid_volume
        data.ask_price = ask_price
        data.ask_volume = ask_volume
        data.total_traded_volume = total_traded_volume
        data.total_traded_notional = total_traded_notional
        data.total_trade_count = total_trade_count

    @staticmethod
    cdef void _set_tick_fields(void* buffer):
        raise NotImplementedError()

    @staticmethod
    cdef void _set_bar_fields(void* buffer, double high_price, double low_price, double open_price, double close_price, double bar_span, double volume=0.0, double notional=0.0, uint32_t trade_count=0):
        cdef _CandlestickBuffer* data = <_CandlestickBuffer*> buffer

        # Initialize bar-specific fields
        data.high_price = high_price
        data.low_price = low_price
        data.open_price = open_price
        data.close_price = close_price
        data.volume = volume
        data.notional = notional
        data.trade_count = trade_count
        data.bar_span = bar_span

    cdef void c_sort(self):
        if self._header.ptr_tail <= 1 or self._header.sorted:
            return

        cdef uint32_t i
        cdef uint64_t* ptr_array
        cdef _MetaInfo** actual_ptrs

        # Step 1: Allocate memory for actual pointers
        actual_ptrs = <_MetaInfo**> malloc(self._header.ptr_tail * sizeof(_MetaInfo*))
        if actual_ptrs == NULL:
            raise MemoryError("Failed to allocate memory for actual pointers")

        # Step 2: Convert offsets to actual pointers
        for i in range(self._header.ptr_tail):
            actual_ptrs[i] = <_MetaInfo*> (<char*> self._data_array + self._ptr_array[i])

        # Step 3: Sort the actual pointers using qsort with compare_md_ptr
        qsort(actual_ptrs, self._header.ptr_tail, sizeof(_MetaInfo*), compare_md_ptr)

        # Step 4: Convert back to offsets
        for i in range(self._header.ptr_tail):
            self._ptr_array[i] = <uint64_t> (<char*> actual_ptrs[i] - self._data_array)

        # Step 5: Free the temporary array
        free(actual_ptrs)

        # Mark as sorted
        self._header.sorted = 1

    # --- python interface ---

    def __iter__(self):
        self.c_sort()
        self._idx = 0
        return self

    def __getitem__(self, idx: int):
        return self.c_get(idx)

    def __len__(self):
        return self._header.ptr_tail

    def __next__(self):
        if self._idx >= self._header.ptr_tail:
            raise StopIteration

        md = self.c_get(self._idx)
        self._idx += 1
        return md

    @classmethod
    def buffer_size(cls, uint32_t n_internal_data=0, uint32_t n_transaction_data=0, uint32_t n_order_data=0, uint32_t n_tick_data_lite=0, uint32_t n_tick_data=0, uint32_t n_bar_data=0):
        return MarketDataBuffer.c_buffer_size(n_internal_data=n_internal_data, n_transaction_data=n_transaction_data, n_order_data=n_order_data, n_tick_data_lite=n_tick_data_lite, n_tick_data=n_tick_data, n_bar_data=n_bar_data)

    def put(self, object market_data):
        cdef uintptr_t data_addr = market_data._data_addr
        return self.c_put(market_data_ptr=<_MarketDataBuffer*> data_addr)

    def update(self, ticker: str, double timestamp, uint8_t dtype, **kwargs):
        # Validate dtype if specified for the buffer
        if dtype == DataType.DTYPE_TICK:
            return self.c_put(market_data_ptr=TickData(ticker=ticker, timestamp=timestamp, **kwargs)._data_ptr)

        # Get size based on dtype
        cdef size_t entry_size = _MarketDataVirtualBase.c_get_size(dtype)

        # Check if we have enough space in the pointer array
        if self._header.ptr_tail >= self._header.ptr_capacity:
            raise MemoryError(f"Not enough space in pointer array for new entry, pointer capacity {self._header.ptr_capacity}")

        # Check if we have enough space in the data section
        if self._header.data_tail + entry_size > self._header.data_capacity:
            raise MemoryError(f"Not enough space in buffer for new entry, requested {entry_size}, remaining {self._header.data_capacity - self._header.data_tail}")

        cdef _MarketDataBuffer* data = <_MarketDataBuffer*> (<void*> self._data_array + self._header.data_tail)
        cdef bytes ticker_bytes = ticker.encode('utf-8')
        cdef size_t ticker_len = min(len(ticker_bytes), TICKER_SIZE - 1)
        memcpy(<void*> &data.MetaInfo.ticker, <const char*> ticker_bytes, ticker_len)
        data.MetaInfo.ticker[ticker_len] = 0
        data.MetaInfo.timestamp = timestamp
        data.MetaInfo.dtype = dtype

        # Set fields based on kwargs
        if dtype == DataType.DTYPE_INTERNAL:
            MarketDataBuffer._set_internal_fields(
                buffer=<void*> data,
                code=kwargs['code'],
            )
        elif dtype == DataType.DTYPE_TRANSACTION:
            MarketDataBuffer._set_transaction_fields(
                buffer=<void*> data,
                price=kwargs['price'],
                volume=kwargs['volume'],
                side=kwargs['side'],
                multiplier=kwargs.get('multiplier', 1.0),
                notional=kwargs.get('notional', 0.0),
                transaction_id=kwargs.get('transaction_id', None),
                buy_id=kwargs.get('buy_id', None),
                sell_id=kwargs.get('sell_id', None)
            )
        elif dtype == DataType.DTYPE_ORDER:
            MarketDataBuffer._set_order_fields(
                buffer=<void*> data,
                price=kwargs['price'],
                volume=kwargs['volume'],
                side=kwargs['side'],
                order_id=kwargs.get('order_id', None),
                order_type=kwargs.get('order_type', 0)
            )
        elif dtype == DataType.DTYPE_TICK_LITE:
            MarketDataBuffer._set_tick_lite_fields(
                buffer=<void*> data,
                last_price=kwargs['last_price'],
                bid_price=kwargs['bid_price'],
                bid_volume=kwargs['bid_volume'],
                ask_price=kwargs['ask_price'],
                ask_volume=kwargs['ask_volume'],
                total_traded_volume=kwargs.get('total_traded_volume', 0.0),
                total_traded_notional=kwargs.get('total_traded_notional', 0.0),
                total_trade_count=kwargs.get('total_trade_count', 0)
            )
        elif dtype == DataType.DTYPE_BAR:
            MarketDataBuffer._set_bar_fields(
                buffer=<void*> data,
                high_price=kwargs['high_price'],
                low_price=kwargs['low_price'],
                open_price=kwargs['open_price'],
                close_price=kwargs['close_price'],
                bar_span=kwargs['bar_span'],
                volume=kwargs.get('volume', 0.0),
                notional=kwargs.get('notional', 0.0),
                trade_count=kwargs.get('trade_count', 0)
            )
        else:
            raise ValueError(f'Unknown data type {dtype}')

        self._ptr_array[self._header.ptr_tail] = self._header.data_tail
        self._header.ptr_tail += 1
        self._header.data_tail += entry_size

        if timestamp >= self._header.current_timestamp:
            self._header.current_timestamp = timestamp
        else:
            self._header.sorted = 0

    def sort(self):
        return self.c_sort()

    def to_bytes(self) -> bytes():
        return self.c_to_bytes()

    @classmethod
    def from_bytes(cls, data: bytes, buffer = None) -> MarketDataBuffer:
        return MarketDataBuffer.c_from_bytes(data=data, buffer=buffer)

    @classmethod
    def from_buffer(cls, buffer = None) -> MarketDataBuffer:
        return MarketDataBuffer.c_from_buffer(buffer=buffer)

    @property
    def ptr_capacity(self) -> int:
        return self._ptr_capacity

    @property
    def ptr_tail(self) -> int:
        return self._header.ptr_tail

    @property
    def data_capacity(self) -> int:
        return self._data_capacity

    @property
    def data_tail(self) -> int:
        return self._header.data_tail


cdef class MarketDataRingBuffer:
    def __cinit__(self, object buffer, int capacity=0):
        # step 1: obtain a buffer view and pointer from the given python buffer object
        PyObject_GetBuffer(buffer, &self._view, PyBUF_SIMPLE)
        self._view_obtained = True
        self._buffer = <char*> self._view.buf

        # step 2: prepare the header
        self._header = <_RingBufferHeader*> self._buffer
        cdef size_t header_size = sizeof(_RingBufferHeader)

        # step 2.1: --- pointer array ---
        self._header.ptr_offset = header_size
        self._ptr_array = <uint64_t*> (<void*> self._header + self._header.ptr_offset)
        cdef uint64_t total_size = self._view.len
        cdef size_t pointer_size = sizeof(uint64_t)
        cdef size_t estimated_entry_size = _MarketDataVirtualBase.c_max_size()
        self._estimated_entry_size = estimated_entry_size
        cdef size_t minimal_size_required = header_size + pointer_size * max(1, capacity) + estimated_entry_size
        if minimal_size_required > total_size:
            PyBuffer_Release(&self._view)
            raise MemoryError(f'Buffer is too small. Minimal size required: {minimal_size_required}')
        cdef uint32_t ptr_capacity = capacity if capacity > 0 else (total_size - header_size) // (pointer_size + estimated_entry_size)
        self._header.ptr_capacity = ptr_capacity
        self._ptr_capacity = ptr_capacity
        cdef size_t ptr_array_size = pointer_size * ptr_capacity

        # step 2.2:--- data array ---
        cdef uint64_t data_capacity = total_size - header_size - ptr_array_size
        self._header.data_offset = header_size + ptr_array_size
        self._data_array = <char*> self._header + self._header.data_offset
        self._header.data_capacity = data_capacity
        self._data_capacity = data_capacity

        # step 3: cleanup headers
        self._header.ptr_head = 0
        self._header.ptr_tail = 0
        self._header.data_tail = 0

        # step 4: optional cleanups, the reset of the buffer need not be clean but just in case
        memset(<void*> self._ptr_array, 0, ptr_array_size)
        memset(<void*> self._data_array, 0, data_capacity)

    def __dealloc__(self):
        if self._view_obtained:
            PyBuffer_Release(&self._view)
            self._view_obtained = False

    cdef bint c_is_empty(self):
        return self._header.ptr_head == self._header.ptr_tail

    cdef uint32_t c_get_ptr_distance(self, uint32_t ptr_idx):
        cdef uint32_t ptr_capacity = self._ptr_capacity
        cdef uint32_t ptr_tail = self._header.ptr_tail
        cdef uint32_t ptr_distance = 0

        if ptr_idx <= ptr_tail:
            ptr_distance = ptr_tail - ptr_idx
        else:
            ptr_distance = ptr_capacity - ptr_idx + ptr_tail

        return ptr_distance

    cpdef bint c_is_full(self):
        cdef uint32_t ptr_tail = self._header.ptr_tail
        cdef uint32_t ptr_next = (ptr_tail + 1) % self._ptr_capacity
        cdef uint32_t ptr_head = self._header.ptr_head

        if ptr_head == ptr_next:
            return True

        if ptr_head == ptr_tail:
            return False

        cdef uint64_t data_head = self._ptr_array[ptr_head]
        cdef uint64_t data_tail = self._header.data_tail

        if data_head > data_tail:
            return data_head - data_tail < self._estimated_entry_size
        else:
            return data_head + self._data_capacity - data_tail < self._estimated_entry_size

    cdef void c_write(self, const char* data, uint64_t length):
        cdef uint32_t ptr_tail = self._header.ptr_tail
        cdef uint64_t data_tail = self._header.data_tail
        cdef uint64_t first_part = self._data_capacity - data_tail
        cdef uint64_t second_part
        cdef uint64_t data_tail_next

        # Contiguous writing
        if length < first_part:
            # print(f'[writer]\t(n_bytes={length}, data_pos={data_tail}, space_before_wrap={first_part}) writing data...')
            memcpy(<void*> self._data_array + data_tail, <const char*> data, length)
            data_tail_next = data_tail + length
        elif length == first_part:
            # print(f'[writer]\t(n_bytes={length}, data_pos={data_tail}, space_before_wrap={first_part}, space_after_wrap=0) writing data...')
            memcpy(<void*> self._data_array + data_tail, data, length)
            data_tail_next = 0
        # Wrapped writing
        else:
            second_part = length - first_part
            # print(f'[writer]\t(n_bytes={length}, data_pos={data_tail}, space_before_wrap={first_part}, space_after_wrap={second_part}) writing data...')
            memcpy(<void*> self._data_array + data_tail, <const char*> data, first_part)
            memcpy(<void*> self._data_array, <const char*> data + first_part, second_part)
            data_tail_next = second_part

        self._ptr_array[ptr_tail] = data_tail
        self._header.data_tail = data_tail_next
        self._header.ptr_tail = (ptr_tail + 1) % self._ptr_capacity

    cdef void c_read(self, uint64_t data_offset, uint64_t length, char* output):
        cdef uint64_t first_part = self._data_capacity - data_offset
        cdef uint64_t second_part

        # Contiguous reading
        if length <= first_part:
            memcpy(<char*> output, <const char*> self._data_array + data_offset, length)
        # Wrapped reading
        else:
            # Wrap-around case
            second_part = length - first_part
            memcpy(<char*> output, <const char*> self._data_array + data_offset, first_part)
            memcpy(<char*> output + first_part, <const char*> self._data_array, second_part)

    cdef bytes c_to_bytes(self, uint64_t data_offset, uint64_t length):
        cdef uint64_t first_part = self._data_capacity - data_offset
        cdef uint64_t second_part

        if length <= first_part:
            return PyBytes_FromStringAndSize(self._data_array + data_offset, length)

        second_part = length - first_part
        return PyBytes_FromStringAndSize(self._data_array + data_offset, first_part) + PyBytes_FromStringAndSize(self._data_array, second_part)

    cdef void c_put(self, _MarketDataBuffer* market_data_ptr):
        if self.c_is_full():
            raise MemoryError(f"Buffer is full")

        # Get a data pointer and dtype directly from market_data
        cdef uint8_t dtype = market_data_ptr.MetaInfo.dtype
        cdef uint64_t length = <uint64_t> _MarketDataVirtualBase.c_get_size(dtype)
        self.c_write(data=<const char*> market_data_ptr, length=length)

    cdef object c_get(self, uint32_t idx):
        cdef uint64_t data_offset = self._ptr_array[idx]
        cdef const char* market_data_ptr = <char*> self._data_array + data_offset
        cdef uint8_t dtype = market_data_ptr[0]
        cdef uint64_t length = <uint64_t> _MarketDataVirtualBase.c_get_size(dtype)
        # cdef MarketData instance
        cdef InternalData internal_data
        cdef TransactionData transaction_data
        cdef OrderData order_data
        cdef TickDataLite tick_data_lite
        cdef TickData tick_data
        cdef BarData bar_data

        # Create appropriate MarketData object
        if dtype == DataType.DTYPE_INTERNAL:
            internal_data = InternalData.__new__(InternalData)
            memcpy(<char*> internal_data._data_ptr, <const char*> self._data_array + data_offset, length)
            return internal_data
        elif dtype == DataType.DTYPE_TRANSACTION:
            transaction_data = TransactionData.__new__(TransactionData)
            self.c_read(data_offset=data_offset, length=length, output=<char*> transaction_data._data_ptr)
            return transaction_data
        elif dtype == DataType.DTYPE_ORDER:
            order_data = OrderData.__new__(OrderData)
            self.c_read(data_offset=data_offset, length=length, output=<char*> order_data._data_ptr)
            return order_data
        elif dtype == DataType.DTYPE_TICK_LITE:
            tick_data_lite = TickDataLite.__new__(TickDataLite)
            self.c_read(data_offset=data_offset, length=length, output=<char*> tick_data_lite._data_ptr)
            return tick_data_lite
        elif dtype == DataType.DTYPE_TICK:
            tick_data = TickData.__new__(TickData)
            self.c_read(data_offset=data_offset, length=length, output=<char*> tick_data._data_ptr)
            tick_data._init_order_book()
            return tick_data
        elif dtype == DataType.DTYPE_BAR:
            bar_data = BarData.__new__(BarData)
            self.c_read(data_offset=data_offset, length=length, output=<char*> bar_data._data_ptr)
            return bar_data
        else:
            raise ValueError(f'Unknown data type {dtype}')

    cdef object c_listen(self, bint block=True, double timeout=-1.0):
        cdef uint32_t spin_per_check = 1000
        cdef time_t start_time = 0
        cdef time_t current_time
        cdef double elapsed = 0.0
        cdef uint32_t spin_count = 0
        cdef uint32_t sleep_us = 0
        cdef bint use_timeout = timeout > 0
        cdef uint32_t idx = self._header.ptr_head

        if (not block) and idx == self._header.ptr_tail:
            raise BufferError(f'Buffer is empty')

        time(&start_time)

        while True:
            # Check for data - direct struct field access
            if idx != self._header.ptr_tail:
                # print(f'[listener]\t(worker_id={worker_id}, ptr_idx={idx}, ptr_next={idx + 1}) getting data...')
                md = self.c_get(idx=idx)
                self._header.ptr_head = (idx + 1) % self._ptr_capacity
                return md

            # Timeout check
            if spin_count % spin_per_check == 0:
                time(&current_time)
                elapsed = difftime(current_time, start_time)

                if use_timeout and elapsed >= timeout:
                    raise TimeoutError("Timeout while waiting for data")

                # Progressive backoff based on elapsed time
                if elapsed < 0.1:  # < 100 ms: pure spin
                    sleep_us = 0
                elif elapsed < 1.0:  # 100 - 1000 ms: 1us sleep
                    sleep_us = 1
                elif elapsed < 3.0:  # 1000 - 3000 ms: 10us sleep
                    sleep_us = 10
                elif elapsed < 15.0:  # 3000 ms - 15 s: 100us sleep
                    sleep_us = 100
                else:  # > 1s: 1ms sleep
                    sleep_us = 1000

            if sleep_us > 0:
                platform_usleep(sleep_us)
            spin_count += 1

    # --- python interface ---

    def __len__(self) -> int:
        return self.c_get_ptr_distance(ptr_idx=self._header.ptr_head)

    def __call__(self, timeout: float = -1.0):
        class _Listener:
            def __init__(_self, outer, worker_id, timeout):
                _self.outer = outer
                _self.timeout = timeout
                _self._running = True

            def __iter__(_self):
                while _self._running:
                    try:
                        md = _self.outer.listen(block=True, timeout=_self.timeout)
                        yield md
                    except TimeoutError:
                        break  # Exit on timeout

            def __enter__(_self):
                return _self.__iter__()

            def __exit__(_self, exc_type, exc_val, exc_tb):
                _self._running = False
                return False

        return _Listener(self, timeout)

    def is_full(self) -> bool:
        return self.c_is_full()

    def is_empty(self) -> bool:
        return self.c_is_empty()

    def read(self, idx: int) -> bytes:
        cdef uint64_t data_offset = self._ptr_array[idx]
        cdef const char* market_data_ptr = <char*> self._data_array + data_offset
        cdef uint8_t dtype = market_data_ptr[0]
        cdef uint64_t length = <uint64_t> _MarketDataVirtualBase.c_get_size(dtype)
        return self.c_to_bytes(data_offset=data_offset, length=length)

    def put(self, object market_data):
        cdef uintptr_t data_addr = market_data._data_addr
        return self.c_put(market_data_ptr=<_MarketDataBuffer*> data_addr)

    def get(self, idx: int):
        return self.c_get(idx=idx)

    def listen(self, block: bool = True, timeout: float = -1.0):
        return self.c_listen(block=block, timeout=timeout)

    def collect_info(self) -> dict:
        cdef uint64_t data_head = self._ptr_array[self._header.ptr_head]
        cdef uint64_t data_tail = self._header.data_tail
        cdef uint64_t data_capacity = self._data_capacity

        return dict(
            ptr_capacity=self._header.ptr_capacity,
            ptr_offset=self._header.ptr_offset,
            ptr_head=self._header.ptr_head,
            ptr_tail=self._header.ptr_tail,
            ptr_free_slots=self._ptr_capacity - self.c_get_ptr_distance(self._header.ptr_head),
            data_capacity=data_capacity,
            data_offset=self._header.data_offset,
            data_head=data_head,
            data_tail=data_tail,
            data_free_buffer=data_head + data_capacity - data_tail if data_tail > data_head else data_head - data_tail
        )


cdef class MarketDataConcurrentBuffer:
    def __cinit__(self, object buffer, uint32_t n_workers, int capacity=0):
        if n_workers > INT8_MAX:
            raise ValueError("Maximum number of workers is 255")

        # step 1: obtain a buffer view and pointer from the given python buffer object
        PyObject_GetBuffer(buffer, &self._view, PyBUF_SIMPLE)
        self._view_obtained = True
        self._buffer = <char*> self._view.buf

        # step 2: prepare the header
        self._header = <_ConcurrentBufferHeader*> self._buffer
        self._header.n_workers = n_workers
        self.n_workers = n_workers
        cdef size_t header_size = sizeof(_ConcurrentBufferHeader)

        # step 2.1: --- worker header ---
        self._header.worker_header_offset = header_size
        self._worker_header_array = <uint32_t*> (<void*> self._header + self._header.worker_header_offset)
        cdef size_t worker_header_size = sizeof(_WorkerHeader) * n_workers

        # step 2.2: --- pointer array ---
        self._header.ptr_offset = header_size + worker_header_size
        self._ptr_array = <uint64_t*> (<void*> self._header + self._header.ptr_offset)
        cdef uint64_t total_size = self._view.len
        cdef size_t pointer_size = sizeof(uint64_t)
        cdef size_t estimated_entry_size = _MarketDataVirtualBase.c_max_size()
        self._estimated_entry_size = estimated_entry_size
        cdef size_t minimal_size_required = header_size + worker_header_size + pointer_size * max(1, capacity) + estimated_entry_size
        if minimal_size_required > total_size:
            PyBuffer_Release(&self._view)
            raise MemoryError(f'Buffer is too small. Minimal size required: {minimal_size_required}')
        cdef uint32_t ptr_capacity = capacity if capacity > 0 else (total_size - header_size - worker_header_size) // (pointer_size + estimated_entry_size)
        self._header.ptr_capacity = ptr_capacity
        self._ptr_capacity = ptr_capacity
        cdef size_t ptr_array_size = pointer_size * ptr_capacity

        # step 2.3:--- data array ---
        cdef uint64_t data_capacity = total_size - header_size - worker_header_size - ptr_array_size
        self._header.data_offset = header_size + worker_header_size + ptr_array_size
        self._data_array = <char*> self._header + self._header.data_offset
        self._header.data_capacity = data_capacity
        self._data_capacity = data_capacity

        # step 3: cleanup headers
        self._header.ptr_tail = 0
        self._header.data_tail = 0
        memset(<void*> self._worker_header_array, 0, worker_header_size)

        # step 4: optional cleanups, the reset of the buffer need not be clean but just in case
        memset(<void*> self._ptr_array, 0, ptr_array_size)
        memset(<void*> self._data_array, 0, data_capacity)

    def __dealloc__(self):
        # since the buffer should be provided from python calls, there should be no memory release requirements
        if self._view_obtained:
            PyBuffer_Release(&self._view)
            self._view_obtained = False

    cdef uint32_t c_get_worker_head(self, uint32_t worker_id) except -1:
        if worker_id >= self.n_workers:
            return -1
        return self._worker_header_array[worker_id]

    cdef uint32_t c_get_ptr_distance(self, uint32_t ptr_idx):
        cdef uint32_t ptr_capacity = self._ptr_capacity
        cdef uint32_t ptr_tail = self._header.ptr_tail
        cdef uint32_t ptr_distance = 0

        if ptr_idx <= ptr_tail:
            ptr_distance = ptr_tail - ptr_idx
        else:
            ptr_distance = ptr_capacity - ptr_idx + ptr_tail

        return ptr_distance

    cdef uint32_t c_get_ptr_head(self):
        cdef uint32_t ptr_head = self._worker_header_array[0]
        cdef uint32_t ptr_distance = 0
        cdef uint32_t ptr_tail = self._header.ptr_tail
        cdef uint32_t ptr_capacity = self._ptr_capacity
        cdef uint32_t worker_head
        cdef uint32_t worker_distance

        for i in range(self.n_workers):
            worker_head = self._worker_header_array[i]

            if worker_head <= ptr_tail:
                worker_distance = ptr_tail - worker_head
            else:
                worker_distance = ptr_capacity - worker_head + ptr_tail

            if worker_distance >= ptr_distance:
                ptr_head = worker_head
                ptr_distance = worker_distance

        return ptr_head

    cdef uint64_t c_get_data_head(self):
        cdef uint32_t ptr_head = self.c_get_ptr_head()
        return self._ptr_array[ptr_head]

    cdef bint c_is_worker_empty(self, uint32_t worker_id) except -1:
        if worker_id >= self.n_workers:
            raise -1
        return self._worker_header_array[worker_id] == self._header.ptr_tail

    cdef bint c_is_empty(self):
        cdef size_t worker_id

        for worker_id in range(self.n_workers):
            if self._worker_header_array[worker_id] < self._header.ptr_tail:
                return False
        return True

    cdef bint c_is_full(self):
        cdef uint32_t ptr_head = self._worker_header_array[0]
        cdef uint32_t ptr_distance = 0
        cdef uint32_t ptr_tail = self._header.ptr_tail
        cdef uint32_t ptr_capacity = self._ptr_capacity
        cdef uint32_t worker_head
        cdef uint32_t worker_distance
        cdef uint32_t ptr_next = (ptr_tail + 1) % ptr_capacity

        # step 1: check if the pointer array is full
        for i in range(self.n_workers):
            worker_head = self._worker_header_array[i]
            # print(f'[checker]\t(worker_id={i}, worker_head={worker_head} ptr_next={ptr_next}) checking pointer array')

            if worker_head == ptr_next:
                # print(f'[checker]\t(worker_id={i}, worker_head={worker_head} ptr_tail={self._header.ptr_tail}) checker report ptr array is full')
                return True

            if worker_head <= ptr_tail:
                worker_distance = ptr_tail - worker_head
            else:
                worker_distance = ptr_capacity - worker_head + ptr_tail

            if worker_distance >= ptr_distance:
                ptr_head = worker_head
                ptr_distance = worker_distance

        # step 2: check if the data array is full
        # print(f'[checker]\t(ptr_head={ptr_head}, ptr_tail={ptr_tail}, ptr_capacity={ptr_capacity}) report {ptr_capacity - self.c_get_ptr_distance(ptr_head)} available slot in pointer array')
        if ptr_head == ptr_tail:
            return False

        cdef uint64_t data_head = self._ptr_array[ptr_head]
        cdef uint64_t data_tail = self._header.data_tail

        # print(f'[checker]\t(data_head={data_head}, data_tail={data_tail}, data_capacity={self._data_capacity}, est_entry_size={self._estimated_entry_size}) checking data array')
        if data_head > data_tail:
            return data_head - data_tail < self._estimated_entry_size
        else:
            return data_head + self._data_capacity - data_tail < self._estimated_entry_size

    cdef void c_write(self, const char* data, uint64_t length):
        cdef uint32_t ptr_tail = self._header.ptr_tail
        cdef uint64_t data_tail = self._header.data_tail
        cdef uint64_t first_part = self._data_capacity - data_tail
        cdef uint64_t second_part
        cdef uint64_t data_tail_next

        # Contiguous writing
        if length < first_part:
            # print(f'[writer]\t(n_bytes={length}, data_pos={data_tail}, space_before_wrap={first_part}) writing data...')
            memcpy(<void*> self._data_array + data_tail, <const char*> data, length)
            data_tail_next = data_tail + length
        elif length == first_part:
            # print(f'[writer]\t(n_bytes={length}, data_pos={data_tail}, space_before_wrap={first_part}, space_after_wrap=0) writing data...')
            memcpy(<void*> self._data_array + data_tail, data, length)
            data_tail_next = 0
        # Wrapped writing
        else:
            second_part = length - first_part
            # print(f'[writer]\t(n_bytes={length}, data_pos={data_tail}, space_before_wrap={first_part}, space_after_wrap={second_part}) writing data...')
            memcpy(<void*> self._data_array + data_tail, <const char*> data, first_part)
            memcpy(<void*> self._data_array, <const char*> data + first_part, second_part)
            data_tail_next = second_part

        # print(f'[writer]\t(data={PyBytes_FromStringAndSize(data, length)}')
        self._ptr_array[ptr_tail] = data_tail
        self._header.data_tail = data_tail_next
        self._header.ptr_tail = (ptr_tail + 1) % self._ptr_capacity
        # print(f'[writer]\t(ptr_tail={ptr_tail}, data_tail={data_tail}, ptr_next={self._header.ptr_tail}, data_next={data_tail_next}, dtype={<uint8_t> data[0]}, data_length={length}) updating pointer array...')

    cdef void c_read(self, uint64_t data_offset, uint64_t length, char* output):
        cdef uint64_t first_part = self._data_capacity - data_offset
        cdef uint64_t second_part

        if length <= first_part:
            # Single contiguous block
            # print(f'[reader]\t(n_bytes={length}, data_pos={data_offset}, space_before_wrap={first_part}) reading data...')
            memcpy(<char*> output, <const char*> self._data_array + data_offset, length)
        else:
            # Wrap-around case
            second_part = length - first_part
            # print(f'[reader]\t(n_bytes={length}, data_pos={data_offset}, space_before_wrap={first_part}, space_after_wrap={second_part}) reading data...')
            memcpy(<char*> output, <const char*> self._data_array + data_offset, first_part)
            memcpy(<char*> output + first_part, <const char*> self._data_array, second_part)

    cdef bytes c_to_bytes(self, uint64_t data_offset, uint64_t length):
        cdef uint64_t first_part = self._data_capacity - data_offset
        cdef uint64_t second_part

        if length <= first_part:
            return PyBytes_FromStringAndSize(self._data_array + data_offset, length)

        second_part = length - first_part
        return PyBytes_FromStringAndSize(self._data_array + data_offset, first_part) + PyBytes_FromStringAndSize(self._data_array, second_part)

    cdef void c_put(self, _MarketDataBuffer* market_data_ptr):
        if self.c_is_full():
            raise MemoryError(f"Buffer is full")

        # Get a data pointer and dtype directly from market_data
        cdef uint8_t dtype = market_data_ptr.MetaInfo.dtype
        cdef uint64_t length = <uint64_t> _MarketDataVirtualBase.c_get_size(dtype)
        self.c_write(data=<const char*> market_data_ptr, length=length)

    cdef object c_get(self, uint32_t idx):
        cdef uint64_t data_offset = self._ptr_array[idx]
        # print(f'[getter]\t(ptr_idx={idx}, data_offset={data_offset}) parsing pointer...')
        cdef const char* market_data_ptr = <char*> self._data_array + data_offset
        cdef uint8_t dtype = market_data_ptr[0]
        cdef uint64_t length = <uint64_t> _MarketDataVirtualBase.c_get_size(dtype)
        # cdef MarketData instance
        cdef InternalData internal_data
        cdef TransactionData transaction_data
        cdef OrderData order_data
        cdef TickDataLite tick_data_lite
        cdef TickData tick_data
        cdef BarData bar_data
        # print(f'[getter]\t(ptr_idx={idx}, data_offset={data_offset}, dtype={dtype}, data_length={length}) parsing data...')

        # Create appropriate MarketData object
        if dtype == DataType.DTYPE_INTERNAL:
            internal_data = InternalData.__new__(InternalData)
            memcpy(<char*> internal_data._data_ptr, <const char*> self._data_array + data_offset, length)
            return internal_data
        elif dtype == DataType.DTYPE_TRANSACTION:
            transaction_data = TransactionData.__new__(TransactionData)
            self.c_read(data_offset=data_offset, length=length, output=<char*> transaction_data._data_ptr)
            return transaction_data
        elif dtype == DataType.DTYPE_ORDER:
            order_data = OrderData.__new__(OrderData)
            self.c_read(data_offset=data_offset, length=length, output=<char*> order_data._data_ptr)
            return order_data
        elif dtype == DataType.DTYPE_TICK_LITE:
            tick_data_lite = TickDataLite.__new__(TickDataLite)
            self.c_read(data_offset=data_offset, length=length, output=<char*> tick_data_lite._data_ptr)
            return tick_data_lite
        elif dtype == DataType.DTYPE_TICK:
            tick_data = TickData.__new__(TickData)
            self.c_read(data_offset=data_offset, length=length, output=<char*> tick_data._data_ptr)
            tick_data._init_order_book()
            return tick_data
        elif dtype == DataType.DTYPE_BAR:
            bar_data = BarData.__new__(BarData)
            self.c_read(data_offset=data_offset, length=length, output=<char*> bar_data._data_ptr)
            return bar_data
        else:
            # print(f'[reader]\t(data={self.c_to_bytes(data_offset=data_offset, length=length)}')
            raise ValueError(f'Unknown data type {dtype}')

    cdef object c_listen(self, uint32_t worker_id, bint block=True, double timeout=-1.0):
        cdef uint32_t spin_per_check = 1000
        cdef time_t start_time = 0
        cdef time_t current_time
        cdef double elapsed = 0.0
        cdef uint32_t spin_count = 0
        cdef uint32_t sleep_us = 0
        cdef bint use_timeout = timeout > 0
        cdef uint32_t idx = self._worker_header_array[worker_id]

        if worker_id >= self.n_workers:
            raise IndexError(f'worker_id exceeds total workers {self.n_workers}')

        if (not block) and idx == self._header.ptr_tail:
            raise BufferError(f'Buffer is empty for worker {worker_id}')

        time(&start_time)

        while True:
            # Check for data - direct struct field access
            if idx != self._header.ptr_tail:
                # print(f'[listener]\t(worker_id={worker_id}, ptr_idx={idx}, ptr_next={idx + 1}) getting data...')
                md = self.c_get(idx=idx)
                self._worker_header_array[worker_id] = (idx + 1) % self._ptr_capacity
                return md

            # Timeout check
            if spin_count % spin_per_check == 0:
                time(&current_time)
                elapsed = difftime(current_time, start_time)

                if use_timeout and elapsed >= timeout:
                    raise TimeoutError("Timeout while waiting for data")

                # Progressive backoff based on elapsed time
                if elapsed < 0.1:  # < 100 ms: pure spin
                    sleep_us = 0
                elif elapsed < 1.0:  # 100 - 1000 ms: 1us sleep
                    sleep_us = 1
                elif elapsed < 3.0:  # 1000 - 3000 ms: 10us sleep
                    sleep_us = 10
                elif elapsed < 15.0:  # 3000 ms - 15 s: 100us sleep
                    sleep_us = 100
                else:  # > 1s: 1ms sleep
                    sleep_us = 1000

            if sleep_us > 0:
                platform_usleep(sleep_us)
            spin_count += 1

    # --- python interface ---

    def __len__(self) -> int:
        cdef uint32_t ptr_head = self.c_get_ptr_head()
        return self.c_get_ptr_distance(ptr_idx=ptr_head)

    def __call__(self, worker_id: int, timeout: float = -1.0):
        class _Listener:
            def __init__(_self, outer, worker_id, timeout):
                _self.outer = outer
                _self.worker_id = worker_id
                _self.timeout = timeout
                _self._running = True

            def __iter__(_self):
                while _self._running:
                    try:
                        md = _self.outer.listen(worker_id=_self.worker_id, block=True, timeout=_self.timeout)
                        yield md
                    except TimeoutError:
                        break  # Exit on timeout

            def __enter__(_self):
                return _self.__iter__()

            def __exit__(_self, exc_type, exc_val, exc_tb):
                _self._running = False
                return False  # Don't suppress exceptions

        return _Listener(self, worker_id, timeout)

    def ptr_head(self, worker_id: int) -> int:
        return self._worker_header_array[worker_id]

    def data_head(self, worker_id: int) -> int:
        return self._ptr_array[self._worker_header_array[worker_id]]

    def is_full(self) -> bool:
        return self.c_is_full()

    def is_empty(self) -> bool:
        return self.c_is_empty()

    def read(self, idx: int) -> bytes:
        cdef uint64_t data_offset = self._ptr_array[idx]
        cdef const char* market_data_ptr = <char*> self._data_array + data_offset
        cdef uint8_t dtype = market_data_ptr[0]
        cdef uint64_t length = <uint64_t> _MarketDataVirtualBase.c_get_size(dtype)
        return self.c_to_bytes(data_offset=data_offset, length=length)

    def put(self, object market_data):
        cdef uintptr_t data_addr = market_data._data_addr
        # return self.c_put(market_data_ptr=(<_MarketDataVirtualBase> market_data)._data_ptr)
        return self.c_put(market_data_ptr=<_MarketDataBuffer*> data_addr)

    def get(self, idx: int):
        return self.c_get(idx=idx)

    def listen(self, worker_id: int, block: bool = True, timeout: float = -1.0):
        return self.c_listen(worker_id=worker_id, block=block, timeout=timeout)

    def collect_header_info(self) -> dict:
        """
        uint8_t n_workers              # number of workers
        uint32_t worker_header_offset  # Offset to find the worker header section
        uint32_t ptr_capacity          # Maximum number of pointers that can be stored ~4.2GB
        uint32_t ptr_offset            # Offset to find the pointer array
        uint32_t ptr_tail              # Index where the pointer of the next element will be added
        uint64_t data_capacity         # Maximum of data that can be stored, ~18.4EB
        uint64_t data_offset           # Offset to find the data buffer
        uint64_t data_tail             # Offset where the data of the next element will be added
        """
        return dict(
            n_workers=self._header.n_workers,
            worker_header_offset=self._header.worker_header_offset,
            ptr_capacity=self._header.ptr_capacity,
            ptr_offset=self._header.ptr_offset,
            ptr_tail=self._header.ptr_tail,
            data_capacity=self._header.data_capacity,
            data_offset=self._header.data_offset,
            data_tail=self._header.data_tail,
        )

    @property
    def ptr_capacity(self) -> int:
        return self._ptr_capacity

    @property
    def data_capacity(self) -> int:
        return self._data_capacity

    @property
    def ptr_tail(self) -> int:
        return self._header.ptr_tail

    @property
    def data_tail(self) -> int:
        return self._header.data_tail

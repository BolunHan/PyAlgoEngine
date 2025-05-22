# cython: language_level=3
from cpython.buffer cimport PyBUF_SIMPLE, PyObject_GetBuffer, PyBuffer_Release
from cpython.bytes cimport PyBytes_FromStringAndSize
from libc.math cimport NAN
from libc.stdlib cimport malloc, free, qsort
from libc.string cimport memcpy, memset
from libc.stdint cimport uint8_t, uint32_t, uint64_t
from libc.time cimport time, time_t, difftime

from .market_data cimport compare_md_ptr, MarketData, DataType, _MetaInfo, _MarketDataBuffer, _TransactionDataBuffer, _OrderDataBuffer, _TickDataLiteBuffer, _TickDataBuffer, _CandlestickBuffer, TICKER_SIZE, MAX_WORKERS, platform_usleep
from .transaction cimport TransactionData, OrderData
from .tick cimport TickData, TickDataLite
from .candlestick cimport BarData


cdef class MarketDataBuffer:
    """
    A buffer for storing and managing multiple market data entries.
    Supports iteration, sorting, and data management.

    The buffer layout is:
    - Header section: Stores metadata about the buffer
    - Pointer array section: Stores relative pointers (offsets) to market data entries
    - Data section: Stores the actual market data entries
    """

    def __cinit__(self):
        """
        Initialize the MarketDataBuffer without allocating memory.
        """
        self._buffer = NULL
        self._view_obtained = False
        self._header = NULL
        self._offsets = NULL
        self._data = NULL

    def __init__(self, buffer, uint8_t dtype=0, uint64_t capacity=0):
        """
        Initialize the MarketDataBuffer with a memory buffer.

        Parameters:
        -----------
        buffer : object
            A Python object supporting the buffer protocol (e.g., memoryview, bytearray, RawArray)
        dtype : uint8_t, optional
            Data type for validation. If 0, mixed data types are allowed.
        capacity : uint64_t, optional
            Maximum number of entries (calculated from buffer size if 0)
        """
        # Get buffer view
        cdef Py_buffer view
        PyObject_GetBuffer(buffer, &view, PyBUF_SIMPLE)

        cdef uint64_t estimated_entry_size
        cdef uint64_t pointer_size
        cdef uint64_t pointer_offset
        cdef uint64_t data_offset
        cdef uint64_t total_size = view.len
        cdef uint64_t header_size = sizeof(_BufferHeader)

        self._view = view
        self._view_obtained = True
        self._buffer = <char *> view.buf

        if capacity <= 0:
            # Estimate max entries (accounting for header and pointer array)
            # Each pointer is an uint64_t (8 bytes)
            # Formula: capacity = (total_size - header_size) / (8 + estimated_entry_size)
            if dtype == 0:
                # For mixed types, use the smallest possible entry size to estimate max entries
                estimated_entry_size = MarketData.min_size()
            else:
                # For specific dtype, use the exact entry size
                estimated_entry_size = MarketData.get_size(dtype)
            # Calculate based on buffer size
            capacity = (total_size - header_size) // (sizeof(uint64_t) + estimated_entry_size)

            if capacity <= 0:
                raise ValueError("Buffer too small to store any entries")

        # Calculate pointer array size (in bytes)
        pointer_size = capacity * sizeof(uint64_t)

        # Calculate offsets
        pointer_offset = header_size
        data_offset = pointer_offset + pointer_size

        # Initialize header
        self._header = <_BufferHeader *> self._buffer
        self._header.dtype = dtype
        self._header.sorted = 1  # with no entry yet, the buffer is already sorted
        self._header.count = 0  # No entries yet
        self._header.current_index = 0
        self._header.pointer_offset = pointer_offset
        self._header.capacity = capacity  # Maximum number of pointers
        self._header.data_offset = data_offset
        self._header.tail_offset = 0  # No data yet
        self._header.max_offset = total_size - data_offset  # Maximum data size
        self._header.current_timestamp = 0.0  # No timestamp yet

        # Set pointers to sections
        self._offsets = <uint64_t *> (self._buffer + pointer_offset)
        self._data = self._buffer + data_offset

        memset(self._offsets, 0, capacity * sizeof(uint64_t))

    def __dealloc__(self):
        """
        Release the buffer view when the object is deallocated.
        """
        if self._view_obtained:
            PyBuffer_Release(&self._view)
            self._view_obtained = False

    @classmethod
    def buffer_size(cls, n_transaction_data: int = 0, n_order_data: int = 0, n_tick_data_lite: int = 0, n_tick_data: int = 0, n_bar_data: int = 0) -> int:
        header_size = sizeof(_BufferHeader)
        capacity = n_transaction_data + n_order_data + n_tick_data_lite + n_tick_data + n_bar_data
        offset_size = sizeof(uint64_t) * capacity
        buffer_size = header_size + offset_size
        buffer_size += n_transaction_data * sizeof(_TransactionDataBuffer)
        buffer_size += n_order_data * sizeof(_OrderDataBuffer)
        buffer_size += n_tick_data_lite * sizeof(_TickDataLiteBuffer)
        buffer_size += n_tick_data * sizeof(_TickDataBuffer)
        buffer_size += n_bar_data * sizeof(_CandlestickBuffer)
        return buffer_size

    @classmethod
    def from_buffer(cls, buffer):
        """
        Create a MarketDataBuffer from an existing buffer without reinitializing it.

        Parameters:
        -----------
        buffer : object
            A Python object supporting the buffer protocol (e.g., memoryview, bytearray, RawArray)
            that already contains MarketDataBuffer data
        """
        # Create instance without initialization
        cdef MarketDataBuffer instance = cls.__new__(cls)

        # Get buffer view
        cdef Py_buffer view
        PyObject_GetBuffer(buffer, &view, PyBUF_SIMPLE)
        cdef size_t view_size = view.len

        instance._view = view
        instance._view_obtained = True

        # Set buffer pointer
        instance._buffer = <char *> view.buf

        # Set header pointer
        instance._header = <_BufferHeader *> instance._buffer

        # Validate buffer
        if view_size < sizeof(_BufferHeader):
            PyBuffer_Release(&instance._view)
            instance._view_obtained = False
            raise ValueError("Buffer is too small to contain a valid header")

        # Set pointers to sections based on header info
        instance._offsets = <uint64_t *> (instance._buffer + instance._header.pointer_offset)
        instance._data = instance._buffer + instance._header.data_offset

        return instance

    @classmethod
    def from_bytes(cls, bytes data, buffer=None):
        """
        Create a MarketDataBuffer from bytes data, storing it in the provided buffer.

        Parameters:
        -----------
        data : bytes
            The serialized MarketDataBuffer data
        buffer : object
            A Python object supporting the buffer protocol where the data will be stored
        """
        # Create instance without initialization
        cdef MarketDataBuffer instance = cls.__new__(cls)

        # Determine buffer source
        if buffer is None:
            buffer = bytearray(data)  # create mutable buffer from bytes

        # Get buffer view
        cdef Py_buffer view
        PyObject_GetBuffer(buffer, &view, PyBUF_SIMPLE)
        cdef size_t view_size = view.len
        instance._view = view
        instance._view_obtained = True

        # Set buffer pointer
        instance._buffer = <char *> view.buf

        # Copy data to buffer
        cdef size_t data_size = len(data)
        if data_size > view_size:
            PyBuffer_Release(&instance._view)
            instance._view_obtained = False
            raise ValueError(f"Buffer size {view_size} is too small for data size {data_size}")

        memcpy(instance._buffer, <const char *> data, data_size)

        # Set header pointer
        instance._header = <_BufferHeader *> instance._buffer

        # Set pointers to sections based on header info
        instance._offsets = <uint64_t *> (instance._buffer + instance._header.pointer_offset)
        instance._data = instance._buffer + instance._header.data_offset

        return instance

    cpdef void push(self, MarketData market_data):
        """
        Add market data to the buffer.

        Parameters:
        -----------
        market_data : MarketData
            The market data to add to the buffer
        """
        # Get data pointer and dtype directly from market_data
        cdef _MarketDataBuffer * data_ptr = market_data._data
        cdef uint8_t entry_dtype = data_ptr.MetaInfo.dtype
        cdef double entry_timestamp = data_ptr.MetaInfo.timestamp
        cdef size_t entry_size = MarketData.get_size(entry_dtype)

        # Validate dtype if specified
        if self._header.dtype != 0 and entry_dtype != self._header.dtype:
            raise TypeError(f"Expected dtype {self._header.dtype}, but found {entry_dtype}")

        # Check if we have enough space in the data section
        if self._header.tail_offset + entry_size > self._header.max_offset:
            raise MemoryError("Not enough space in buffer for new entry")

        # Check if we have enough space in the pointer array
        if self._header.count >= self._header.capacity:
            raise MemoryError("Not enough space in pointer array for new entry")

        # Copy data directly from market_data._data to our buffer at the current tail offset
        memcpy(self._data + self._header.tail_offset, data_ptr, entry_size)

        # Add pointer (offset) to the new entry
        self._offsets[self._header.count] = self._header.tail_offset

        # Update count and tail offset
        self._header.count += 1
        self._header.tail_offset += entry_size

        # Update current_timestamp if needed
        if entry_timestamp >= self._header.current_timestamp:
            self._header.current_timestamp = entry_timestamp
        # New entry has earlier timestamp, need to resort
        else:
            self._header.sorted = 0

    def update(self, uint8_t dtype, **kwargs):
        """
        Create and add market data to the buffer without initializing a MarketData object.

        Parameters:
        -----------
        dtype : uint8_t
            The data type of the market data
        **kwargs : dict
            Keyword arguments for initializing the market data fields
        """
        # Validate dtype if specified for the buffer
        if self._header.dtype != 0 and dtype != self._header.dtype:
            raise TypeError(f"Expected dtype {self._header.dtype}, but found {dtype}")
        elif dtype == DataType.DTYPE_TICK:
            return self.push(market_data=TickData(**kwargs))

        # Get size based on dtype
        cdef size_t entry_size = MarketData.get_size(dtype)

        # Check if we have enough space in the data section
        if self._header.tail_offset + entry_size > self._header.max_offset:
            raise MemoryError("Not enough space in buffer for new entry")

        # Check if we have enough space in the pointer array
        if self._header.count >= self._header.capacity:
            raise MemoryError("Not enough space in pointer array for new entry")

        # Get pointer to the location where we'll store the new entry
        cdef char * entry_ptr = self._data + self._header.tail_offset

        # Initialize with zeros
        memset(entry_ptr, 0, entry_size)

        # Set dtype
        (<_MetaInfo *> entry_ptr).dtype = dtype

        # Set fields based on kwargs
        self._set_fields(entry_ptr, dtype, kwargs)

        # Get timestamp from the entry
        cdef double entry_timestamp = (<_MetaInfo *> entry_ptr).timestamp

        # Add pointer (offset) to the new entry
        self._offsets[self._header.count] = self._header.tail_offset

        # Update count and tail offset
        self._header.count += 1
        self._header.tail_offset += entry_size

        # Update current_timestamp if needed
        if entry_timestamp >= self._header.current_timestamp:
            self._header.current_timestamp = entry_timestamp
        # New entry has earlier timestamp, need to resort
        else:
            self._header.sorted = 0

    cdef void _set_fields(self, char * buffer, uint8_t dtype, dict kwargs):
        """
        Set fields in the buffer based on kwargs.
        """
        # Set common fields (MetaInfo)
        cdef bytes ticker_bytes = kwargs['ticker'].encode('utf-8')
        cdef int ticker_len = min(len(ticker_bytes), TICKER_SIZE - 1)

        memcpy(&(<_MetaInfo *> buffer).ticker, <char *> ticker_bytes, ticker_len)
        (<_MetaInfo *> buffer).timestamp = kwargs['timestamp']

        # Set specific fields based on dtype
        if dtype == DataType.DTYPE_TRANSACTION:
            MarketDataBuffer._set_transaction_fields(buffer, kwargs)
        elif dtype == DataType.DTYPE_ORDER:
            MarketDataBuffer._set_order_fields(buffer, kwargs)
        elif dtype == DataType.DTYPE_TICK_LITE:
            MarketDataBuffer._set_tick_lite_fields(buffer, kwargs)
        elif dtype == DataType.DTYPE_TICK:
            MarketDataBuffer._set_tick_fields(buffer, kwargs)
        elif dtype == DataType.DTYPE_BAR:
            MarketDataBuffer._set_bar_fields(buffer, kwargs)

    @staticmethod
    cdef void _set_transaction_fields(char * buffer, dict kwargs):
        """
        Set fields for TransactionData.
        """
        cdef _TransactionDataBuffer * data = <_TransactionDataBuffer *> buffer

        # the optional fields
        data.price = price = kwargs['price']
        data.volume = volume = kwargs['volume']
        data.side = kwargs['side']
        data.multiplier = multiplier = kwargs.get('multiplier', 1.)
        data.notional = kwargs.get('notional', price * volume * multiplier)
        TransactionData._set_id(&data.transaction_id, kwargs.get('transaction_id'))
        TransactionData._set_id(&data.buy_id, kwargs.get('buy_id'))
        TransactionData._set_id(&data.sell_id, kwargs.get('sell_id'))

    @staticmethod
    cdef void _set_order_fields(char * buffer, dict kwargs):
        """
        Set fields for OrderData.
        """
        cdef _OrderDataBuffer * data = <_OrderDataBuffer *> buffer

        data.price = kwargs['price']
        data.volume = kwargs['volume']
        data.side = kwargs['side']
        data.order_type = kwargs.get('order_type', 0)
        TransactionData._set_id(&data.order_id, kwargs.get('order_id'))

    @staticmethod
    cdef void _set_tick_lite_fields(char * buffer, dict kwargs):
        """
        Set fields for TickDataLite.
        """
        cdef _TickDataLiteBuffer * data = <_TickDataLiteBuffer *> buffer

        data.last_price = kwargs['last_price']
        data.bid_price = kwargs['bid_price']
        data.bid_volume = kwargs['bid_volume']
        data.ask_price = kwargs['ask_price']
        data.ask_volume = kwargs['ask_volume']
        data.total_traded_volume = kwargs.get('total_traded_volume', 0.)
        data.total_traded_notional = kwargs.get('total_traded_notional', 0.)
        data.total_trade_count = kwargs.get('total_trade_count', 0)

    @staticmethod
    cdef void _set_tick_fields(char * buffer, dict kwargs):
        """
        Set fields for TickData.
        """
        # First set the TickDataLite fields (which are part of TickData)
        cdef _TickDataBuffer * data = <_TickDataBuffer *> buffer

        data.lite.last_price = kwargs['last_price']
        data.lite.bid_price = kwargs.get('bid_price_1', NAN)
        data.lite.bid_volume = kwargs.get('bid_volume_1', NAN)
        data.lite.ask_price = kwargs.get('ask_price_1', NAN)
        data.lite.ask_volume = kwargs.get('ask_volume_1', NAN)
        data.lite.total_traded_volume = kwargs.get('total_traded_volume', 0.)
        data.lite.total_traded_notional = kwargs.get('total_traded_notional', 0.)
        data.lite.total_trade_count = kwargs.get('total_trade_count', 0)

        tick_data = TickData.from_buffer(buffer)
        tick_data._owner = True
        tick_data.parse(kwargs)

    @staticmethod
    cdef void _set_bar_fields(char * buffer, dict kwargs):
        """
        Set fields for BarData (CandlestickBuffer).
        """
        cdef _CandlestickBuffer * data = <_CandlestickBuffer *> buffer

        data.high_price = kwargs['high_price']
        data.low_price = kwargs['low_price']
        data.open_price = kwargs['open_price']
        data.close_price = kwargs['close_price']
        data.bar_span = kwargs['bar_span']  # during update mode, the bar_span is now required

        data.volume = kwargs.get('volume', 0.)
        data.notional = kwargs.get('notional', 0.)
        data.trade_count = kwargs.get('trade_count', 0)

    cpdef void sort(self, bint inplace=True):
        """
        Sort the buffer by timestamp using the array of pointers.
        Works with any dtype situation.

        Parameters:
        -----------
        inplace : bool, optional
            If True, sort the pointer array in place (not thread-safe).
            If False, create a copy of the pointer array, sort it, and then copy back (thread-safe).
        """
        if self._header.count <= 1 or self._header.sorted:
            return

        cdef uint32_t i
        cdef uint64_t *ptr_array
        cdef _MetaInfo ** actual_ptrs

        if inplace:
            # Sort in place (not thread-safe)
            # 1. Allocate memory for actual pointers
            actual_ptrs = <_MetaInfo **> malloc(self._header.count * sizeof(_MetaInfo *))
            if actual_ptrs == NULL:
                raise MemoryError("Failed to allocate memory for actual pointers")

            # 2. Convert offsets to actual pointers
            for i in range(self._header.count):
                actual_ptrs[i] = <_MetaInfo *> (self._data + self._offsets[i])

            # 3. Sort the actual pointers using qsort with compare_md_ptr
            qsort(actual_ptrs, self._header.count, sizeof(_MetaInfo *), compare_md_ptr)

            # 4. Convert back to offsets
            for i in range(self._header.count):
                self._offsets[i] = <uint64_t> (<char *> actual_ptrs[i] - self._data)

            # 5. Free the temporary array
            free(actual_ptrs)
        else:
            # Thread-safe sort (create a copy)
            # 1. Allocate memory for a copy of the pointer array
            ptr_array = <uint64_t *> malloc(self._header.count * sizeof(uint64_t))
            if ptr_array == NULL:
                raise MemoryError("Failed to allocate memory for pointer array copy")

            # 2. Copy the pointer array
            memcpy(ptr_array, self._offsets, self._header.count * sizeof(uint64_t))

            # 3. Allocate memory for actual pointers
            actual_ptrs = <_MetaInfo **> malloc(self._header.count * sizeof(_MetaInfo *))
            if actual_ptrs == NULL:
                free(ptr_array)
                raise MemoryError("Failed to allocate memory for actual pointers")

            # 4. Convert offsets to actual pointers
            for i in range(self._header.count):
                actual_ptrs[i] = <_MetaInfo *> (self._data + ptr_array[i])

            # 5. Sort the actual pointers using qsort with compare_md_ptr
            qsort(actual_ptrs, self._header.count, sizeof(_MetaInfo *), compare_md_ptr)

            # 6. Convert back to offsets
            for i in range(self._header.count):
                ptr_array[i] = <uint64_t> (<char *> actual_ptrs[i] - self._data)

            # 7. Copy the sorted pointer array back
            memcpy(self._offsets, ptr_array, self._header.count * sizeof(uint64_t))

            # 8. Free the temporary arrays
            free(actual_ptrs)
            free(ptr_array)

        # Mark as sorted
        self._header.sorted = 1

    def __iter__(self):
        """
        Return an iterator over the market data entries.
        """
        self.sort()
        self._header.current_index = 0
        return self

    def __getitem__(self, idx: int):
        if idx >= self._header.count:
            raise IndexError(f'{self.__class__.__name__} index out of range')

        cdef uint64_t offset = self._offsets[idx]
        cdef _MetaInfo * ptr = <_MetaInfo *> (self._data + offset)
        cdef uint8_t dtype = ptr.dtype
        cdef size_t entry_size = MarketData.get_size(dtype)
        cdef bytes data = PyBytes_FromStringAndSize(<char *> ptr, entry_size)

        # Create appropriate object based on dtype
        if dtype == DataType.DTYPE_TRANSACTION:
            return TransactionData.from_bytes(data)
        elif dtype == DataType.DTYPE_ORDER:
            return OrderData.from_bytes(data)
        elif dtype == DataType.DTYPE_TICK_LITE:
            return TickDataLite.from_bytes(data)
        elif dtype == DataType.DTYPE_TICK:
            return TickData.from_bytes(data)
        elif dtype == DataType.DTYPE_BAR:
            return BarData.from_bytes(data)
        else:
            return MarketData.from_bytes(data)

    def __next__(self):
        """
        Get the next market data entry.
        """
        if self._header.current_index >= self._header.count:
            raise StopIteration

        # Get offset for current index
        cdef uint64_t offset = self._offsets[self._header.current_index]

        # Get pointer to entry
        cdef _MetaInfo * ptr = <_MetaInfo *> (self._data + offset)

        # Get dtype
        cdef uint8_t dtype = ptr.dtype

        # Get entry size
        cdef size_t entry_size = MarketData.get_size(dtype)

        # Create bytes object from buffer
        cdef bytes data = PyBytes_FromStringAndSize(<char *> ptr, entry_size)

        # Increment current index
        self._header.current_index += 1

        # Create appropriate object based on dtype
        if dtype == DataType.DTYPE_TRANSACTION:
            return TransactionData.from_bytes(data)
        elif dtype == DataType.DTYPE_ORDER:
            return OrderData.from_bytes(data)
        elif dtype == DataType.DTYPE_TICK_LITE:
            return TickDataLite.from_bytes(data)
        elif dtype == DataType.DTYPE_TICK:
            return TickData.from_bytes(data)
        elif dtype == DataType.DTYPE_BAR:
            return BarData.from_bytes(data)
        else:
            return MarketData.from_bytes(data)

    def __len__(self):
        """
        Get the number of entries in the buffer.
        """
        return self._header.count

    cpdef bytes to_bytes(self):
        """
        Convert the buffer to bytes.
        """
        if self._buffer == NULL:
            return b''

        # Calculate total size (header + pointer array + used data)
        cdef uint64_t total_size = self._header.data_offset + self._header.tail_offset
        cdef uint64_t max_offset = self._header.max_offset

        self._header.max_offset = self._header.tail_offset

        # Return the buffer up to the used size
        data = PyBytes_FromStringAndSize(self._buffer, total_size)

        self._header.max_offset = max_offset

        return data

cdef class MarketDataRingBuffer:
    """
    A ring buffer implementation for MarketData with variable-sized entries.
    Provides FIFO operations and full/empty state checks.
    """

    def __cinit__(self):
        self._buffer = NULL
        self._view_obtained = False
        self._header = NULL
        self._offsets = NULL
        self._data = NULL

    def __init__(self, buffer, uint8_t dtype=0, uint64_t capacity=0):
        """
        Initialize the MarketDataRingBuffer with a memory buffer.

        Parameters:
        -----------
        buffer : object
            A Python object supporting the buffer protocol
        dtype : uint8_t, optional
            Data type for validation (0 for mixed types)
        capacity : uint64_t, optional
            Maximum number of entries (calculated from buffer size if 0)
        """
        cdef Py_buffer view
        PyObject_GetBuffer(buffer, &view, PyBUF_SIMPLE)

        cdef uint64_t estimated_entry_size
        cdef uint64_t pointer_size
        cdef uint64_t pointer_offset
        cdef uint64_t data_offset
        cdef uint64_t total_size = view.len
        cdef uint64_t header_size = sizeof(_RingBufferHeader)

        self._view = view
        self._view_obtained = True
        self._buffer = <char *> view.buf

        if capacity <= 0:
            if dtype == 0:
                estimated_entry_size = MarketData.min_size()
            else:
                estimated_entry_size = MarketData.get_size(dtype)

            # Calculate maximum possible capacity
            capacity = (total_size - header_size) // (sizeof(uint64_t) + estimated_entry_size)

            if capacity <= 0:
                raise ValueError("Buffer too small to store any entries")

        # Calculate pointer array size (in bytes)
        pointer_size = capacity * sizeof(uint64_t)

        # Calculate offsets
        pointer_offset = header_size
        data_offset = pointer_offset + pointer_size

        self._header = <_RingBufferHeader *> self._buffer
        memset(self._header, 0, sizeof(_RingBufferHeader))

        self._header.buffer_header.dtype = dtype
        self._header.buffer_header.sorted = 1
        self._header.buffer_header.count = 0
        self._header.buffer_header.current_index = 0
        self._header.buffer_header.pointer_offset = pointer_offset
        self._header.buffer_header.capacity = capacity
        self._header.buffer_header.data_offset = header_size + (capacity * sizeof(uint64_t))
        self._header.buffer_header.tail_offset = 0
        self._header.buffer_header.max_offset = total_size - data_offset  # Maximum data size
        self._header.buffer_header.current_timestamp = 0.0

        # Initialize ring buffer specific fields
        self._header.head = 0
        self._header.tail = 0

        # Set section pointers
        self._offsets = <uint64_t *> (self._buffer + pointer_offset)
        self._data = self._buffer + data_offset

        # Initialize all pointers to 0
        memset(self._offsets, 0, capacity * sizeof(uint64_t))

    def __len__(self) -> int:
        return self._header.tail - self._header.head

    def __dealloc__(self):
        if self._view_obtained:
            PyBuffer_Release(&self._view)
            self._view_obtained = False

    cdef uint64_t data_head(self):
        """
        Get the current head position in the data section.

        Returns:
        --------
        uint64_t
            The offset from start of data section of the oldest entry
        """
        cdef uint64_t head_idx = self._header.head % self._header.buffer_header.capacity
        return self._offsets[head_idx]

    cdef uint64_t data_tail(self):
        """
        Get the end position of the most recently added entry in the data section.
        Returns 0 if buffer is empty.
        """
        # reset the ring buffer
        if self.is_empty():
            return 0

        cdef uint64_t tail_idx = (self._header.tail - 1) % self._header.buffer_header.capacity
        cdef uint64_t offset = self._offsets[tail_idx]
        cdef _MetaInfo * ptr = <_MetaInfo *> (self._data + offset)
        cdef size_t entry_size = MarketData.get_size(ptr.dtype)
        cdef uint64_t end_pos = offset + entry_size

        if end_pos > self._header.buffer_header.max_offset:
            end_pos = entry_size - (self._header.buffer_header.max_offset - offset)

        return end_pos

    cpdef bint is_empty(self):
        """Return True if the buffer is empty."""
        return self._header.head == self._header.tail

    cpdef bint is_full(self):
        """Return True if buffer cannot accept any new entries."""
        # 1. First check if pointer array is full
        if self._header.tail - self._header.head >= self._header.buffer_header.capacity:
            return True

        # 2. Check data section space (treat as true ring buffer)
        cdef uint64_t data_head_pos = self.data_head()
        # cdef uint64_t data_tail_pos = self.data_tail()
        # since the tail_offset is well managed, can use this value for fast locating
        cdef uint64_t data_tail_pos = self._header.buffer_header.tail_offset
        cdef uint64_t max_offset = self._header.buffer_header.max_offset
        cdef uint64_t free_space = data_head_pos - data_tail_pos

        if self.is_empty():
            # Case 3: Empty buffer - all space available
            return False
        elif data_tail_pos >= data_head_pos:
            # Case 1: Normal case - free space at end and beginning
            # Calculate contiguous space at end plus space at beginning
            return free_space + max_offset < MarketData.max_size()
        else:
            # Case 2: Wrapped case - free space between tail and head
            return free_space < MarketData.max_size()

    @staticmethod
    cdef bytes read(char * data, uint64_t offset, size_t size, uint64_t max_offset):
        """
        Read data from buffer, handling wrap-around if needed.

        Parameters:
        -----------
        offset : uint64_t
            Starting offset in data section
        size : size_t
            Number of bytes to read

        Returns:
        --------
        bytes
            The read data
        """
        cdef char * temp_buf = <char *> malloc(size)
        cdef size_t first_part = max_offset - offset

        if temp_buf == NULL:
            raise MemoryError("Failed to allocate temporary buffer")

        try:
            if size <= first_part:
                # Contiguous read
                memcpy(temp_buf, data + offset, size)
            else:
                # Wrapped read
                memcpy(temp_buf, data + offset, first_part)
                memcpy(temp_buf + first_part, data, size - first_part)

            return PyBytes_FromStringAndSize(temp_buf, size)
        finally:
            free(temp_buf)

    @staticmethod
    cdef void write(char * data, uint64_t offset, char * src, size_t size, uint64_t max_offset):
        """
        Write data to buffer, handling wrap-around if needed.

        Parameters:
        -----------
        offset : uint64_t
            Starting offset in data section
        src : char *
            Source data to write
        size : size_t
            Number of bytes to write
        """
        cdef size_t first_part = max_offset - offset

        if size <= first_part:
            # Contiguous write
            memcpy(data + offset, src, size)
        else:
            # Wrapped write
            memcpy(data + offset, src, first_part)
            memcpy(data, src + first_part, size - first_part)

    cpdef void put(self, MarketData market_data):
        """
        Add market data to the buffer.
        """
        if self.is_full():
            raise MemoryError("Buffer is full")

        # Get data pointer and dtype directly from market_data
        cdef _MarketDataBuffer * data_ptr = market_data._data
        cdef uint8_t entry_dtype = data_ptr.MetaInfo.dtype
        cdef size_t entry_size = MarketData.get_size(entry_dtype)

        # Validate dtype if specified
        if self._header.buffer_header.dtype != 0 and entry_dtype != self._header.buffer_header.dtype:
            raise TypeError(f"Expected dtype {self._header.buffer_header.dtype}, but found {entry_dtype}")

        # Check if write would exceed buffer capacity
        if entry_size > self._header.buffer_header.max_offset:
            raise MemoryError("Market data too large for buffer.")

        # Check available space
        cdef uint64_t write_offset = self._header.buffer_header.tail_offset

        # Write the data
        MarketDataRingBuffer.write(data=self._data, offset=write_offset, src=<char *> data_ptr, size=entry_size, max_offset=self._header.buffer_header.max_offset)

        # Store the offset
        self._offsets[self._header.tail % self._header.buffer_header.capacity] = write_offset

        # Update tail position (no modulo here)
        self._header.tail += 1
        self._header.buffer_header.tail_offset += entry_size
        if self._header.buffer_header.tail_offset >= self._header.buffer_header.max_offset:
            self._header.buffer_header.tail_offset -= self._header.buffer_header.max_offset

        # Update count
        self._header.buffer_header.count += 1

        # Update current_timestamp if needed
        if data_ptr.MetaInfo.timestamp > self._header.buffer_header.current_timestamp:
            self._header.buffer_header.current_timestamp = data_ptr.MetaInfo.timestamp

    cpdef MarketData get(self, uint64_t idx):
        cdef uint64_t offset = self._offsets[idx % self._header.buffer_header.capacity]
        cdef char * data_ptr = self._data + offset
        cdef uint8_t dtype = data_ptr[0]
        cdef size_t entry_size = MarketData.get_size(dtype)
        cdef bytes data = MarketDataRingBuffer.read(data=self._data, offset=offset, size=entry_size, max_offset=self._header.buffer_header.max_offset)

        # print(f'getting buffer, '
        #       f'ttl={self._header.buffer_header.count}, '
        #       f'max_size={self._header.buffer_header.capacity}, '
        #       f'current_size {self.tail - self.head}, '
        #       f'head={self.head}, '
        #       f'tail={self.tail=}, '
        #       f'idx={idx} '
        #       f'offset={offset} '
        #       f'max_offset={self._header.buffer_header.max_offset} '
        #       f'data_head={self.data_head()}, '
        #       f'data_tail={self.data_tail()}.')

        # Create appropriate MarketData object
        if dtype == DataType.DTYPE_TRANSACTION:
            return TransactionData.from_bytes(data)
        elif dtype == DataType.DTYPE_ORDER:
            return OrderData.from_bytes(data)
        elif dtype == DataType.DTYPE_TICK_LITE:
            return TickDataLite.from_bytes(data)
        elif dtype == DataType.DTYPE_TICK:
            return TickData.from_bytes(data)
        elif dtype == DataType.DTYPE_BAR:
            return BarData.from_bytes(data)
        else:
            return MarketData.from_bytes(data)

    cpdef MarketData listen(self):
        """
        Get the oldest MarketData from the buffer (FIFO order).
        """
        if self.is_empty():
            raise IndexError("Buffer is empty")

        md = self.get(idx=self.head)

        self._header.head += 1

        return md

    @property
    def head(self) -> int:
        return self._header.head

    @property
    def tail(self) -> int:
        return self._header.tail

    @property
    def count(self) -> int:
        return self._header.buffer_header.count

cdef class MarketDataConcurrentBuffer:
    def __cinit__(self):
        self._buffer = NULL
        self._view_obtained = False
        self._header = NULL
        self._offsets = NULL
        self._data = NULL

    def __init__(self, buffer, uint32_t n_workers, uint8_t dtype=0, uint64_t capacity=0):
        """
        Initialize the MarketDataConcurrentBuffer with a memory buffer.

        Parameters:
        -----------
        buffer : object
            A Python object supporting the buffer protocol
        n_workers : uint32_t
            Number of worker threads that will access this buffer
        dtype : uint8_t, optional
            Data type for validation (0 for mixed types)
        capacity : uint64_t, optional
            Maximum number of entries (calculated from buffer size if 0)
        """
        if n_workers > 128:
            raise ValueError("Maximum number of workers is 128")

        cdef Py_buffer view
        PyObject_GetBuffer(buffer, &view, PyBUF_SIMPLE)

        cdef uint64_t estimated_entry_size
        cdef uint64_t pointer_size
        cdef uint64_t pointer_offset
        cdef uint64_t data_offset
        cdef uint64_t total_size = view.len
        cdef uint64_t header_size = sizeof(_ConcurrentBufferHeader)

        self._view = view
        self._view_obtained = True
        self._buffer = <char *> view.buf

        if capacity <= 0:
            if dtype == 0:
                estimated_entry_size = MarketData.min_size()
            else:
                estimated_entry_size = MarketData.get_size(dtype)

            # Calculate maximum possible capacity
            capacity = (total_size - header_size) // (sizeof(uint64_t) + estimated_entry_size)

            if capacity <= 0:
                print('Buffer too small to store any entries')
                PyBuffer_Release(&view)
                raise ValueError("Buffer too small to store any entries")

        # Calculate pointer array size (in bytes)
        pointer_size = capacity * sizeof(uint64_t)

        # Calculate offsets
        pointer_offset = header_size
        data_offset = pointer_offset + pointer_size

        self._header = <_ConcurrentBufferHeader *> self._buffer
        memset(self._header, 0, sizeof(_ConcurrentBufferHeader))

        # Initialize base header
        self._header.buffer_header.dtype = dtype
        self._header.buffer_header.sorted = 1
        self._header.buffer_header.count = 0
        self._header.buffer_header.current_index = 0
        self._header.buffer_header.pointer_offset = pointer_offset
        self._header.buffer_header.capacity = capacity
        self._header.buffer_header.data_offset = data_offset
        self._header.buffer_header.tail_offset = 0
        self._header.buffer_header.max_offset = total_size - data_offset
        self._header.buffer_header.current_timestamp = 0.0

        # Initialize concurrent buffer specific fields
        self._header.n_workers = n_workers
        self._header.tail = 0
        memset(self._header.heads, 0, MAX_WORKERS * sizeof(uint64_t))
        # memset(self._header.notify, 0, 128 * sizeof(uint8_t))

        # Set section pointers
        self._offsets = <uint64_t *> (self._buffer + pointer_offset)
        self._data = self._buffer + data_offset

        # Initialize all pointers to 0
        memset(self._offsets, 0, capacity * sizeof(uint64_t))

    def __len__(self) -> int:
        return self._header.tail - self.min_head()

    def __dealloc__(self):
        if self._view_obtained:
            PyBuffer_Release(&self._view)
            self._view_obtained = False

    cpdef uint64_t get_head(self, uint32_t worker_id):
        if worker_id >= self._header.n_workers:
            raise IndexError("worker_id exceeds number of workers")
        return self._header.heads[worker_id]

    cpdef uint64_t min_head(self):
        """Get the minimum head position across all workers."""
        cdef uint64_t min_head = self._header.heads[0]
        cdef uint32_t i

        for i in range(1, self._header.n_workers):
            if self._header.heads[i] < min_head:
                min_head = self._header.heads[i]
        return min_head

    cdef uint64_t data_head(self):
        """Get the current head position in the data section."""
        cdef uint64_t min_head = self.min_head()
        cdef uint64_t head_idx = min_head % self._header.buffer_header.capacity
        return self._offsets[head_idx]

    cdef uint64_t data_tail(self):
        """
        Get the end position of the most recently added entry in the data section.
        Returns 0 if buffer is empty.
        """
        # reset the ring buffer
        if self.is_empty_all():
            return 0

        cdef uint64_t tail_idx = (self._header.tail - 1) % self._header.buffer_header.capacity
        cdef uint64_t offset = self._offsets[tail_idx]
        cdef _MetaInfo * ptr = <_MetaInfo *> (self._data + offset)
        cdef size_t entry_size = MarketData.get_size(ptr.dtype)
        cdef uint64_t end_pos = offset + entry_size

        if end_pos > self._header.buffer_header.max_offset:
            end_pos = entry_size - (self._header.buffer_header.max_offset - offset)

        return end_pos

    cpdef bint is_empty(self, uint32_t worker_id):
        """Return True if the buffer is empty for the given worker."""
        if worker_id >= self._header.n_workers:
            raise IndexError("worker_id exceeds number of workers")
        return self._header.heads[worker_id] == self._header.tail

    cpdef bint is_empty_all(self):
        """Return True if the buffer is empty for the given worker."""
        cdef uint32_t i

        for i in range(self._header.n_workers):
            if not self.is_empty(worker_id=i):
                return False
        return True

    cpdef bint is_full(self):
        """Return True if buffer cannot accept any new entries."""
        # Same logic as parent class
        if self._header.tail - self.min_head() >= self._header.buffer_header.capacity:
            return True

        cdef uint64_t data_head_pos = self.data_head()
        cdef uint64_t data_tail_pos = self._header.buffer_header.tail_offset
        cdef uint64_t max_offset = self._header.buffer_header.max_offset
        cdef uint64_t free_space = data_head_pos - data_tail_pos

        if self.min_head() == self._header.tail:
            return False
        elif data_tail_pos >= data_head_pos:
            return free_space + max_offset < MarketData.max_size()
        else:
            return free_space < MarketData.max_size()

    cpdef void put(self, MarketData market_data):
        if self.is_full():
            raise MemoryError("Buffer is full")

        # Get data pointer and dtype directly from market_data
        cdef _MarketDataBuffer * data_ptr = market_data._data
        cdef uint8_t entry_dtype = data_ptr.MetaInfo.dtype
        cdef size_t entry_size = MarketData.get_size(entry_dtype)

        # Validate dtype if specified
        if self._header.buffer_header.dtype != 0 and entry_dtype != self._header.buffer_header.dtype:
            raise TypeError(f"Expected dtype {self._header.buffer_header.dtype}, but found {entry_dtype}")

        # Check if write would exceed buffer capacity
        if entry_size > self._header.buffer_header.max_offset:
            raise MemoryError("Market data too large for buffer.")

        # Check available space
        cdef uint64_t write_offset = self._header.buffer_header.tail_offset

        # Write the data
        MarketDataRingBuffer.write(data=self._data, offset=write_offset, src=<char *> data_ptr, size=entry_size, max_offset=self._header.buffer_header.max_offset)

        # Store the offset
        self._offsets[self._header.tail % self._header.buffer_header.capacity] = write_offset

        # Update tail position (no modulo here)
        self._header.tail += 1
        self._header.buffer_header.tail_offset += entry_size
        if self._header.buffer_header.tail_offset >= self._header.buffer_header.max_offset:
            self._header.buffer_header.tail_offset -= self._header.buffer_header.max_offset

        # Update count
        self._header.buffer_header.count += 1

        # Update current_timestamp if needed
        if data_ptr.MetaInfo.timestamp > self._header.buffer_header.current_timestamp:
            self._header.buffer_header.current_timestamp = data_ptr.MetaInfo.timestamp

    cpdef MarketData get(self, uint64_t idx):
        cdef uint64_t offset = self._offsets[idx % self._header.buffer_header.capacity]
        cdef char * data_ptr = self._data + offset
        cdef uint8_t dtype = data_ptr[0]
        cdef size_t entry_size = MarketData.get_size(dtype)
        cdef bytes data = MarketDataRingBuffer.read(data=self._data, offset=offset, size=entry_size, max_offset=self._header.buffer_header.max_offset)

        # Create appropriate MarketData object
        if dtype == DataType.DTYPE_TRANSACTION:
            return TransactionData.from_bytes(data)
        elif dtype == DataType.DTYPE_ORDER:
            return OrderData.from_bytes(data)
        elif dtype == DataType.DTYPE_TICK_LITE:
            return TickDataLite.from_bytes(data)
        elif dtype == DataType.DTYPE_TICK:
            return TickData.from_bytes(data)
        elif dtype == DataType.DTYPE_BAR:
            return BarData.from_bytes(data)
        else:
            return MarketData.from_bytes(data)

    cpdef MarketData listen(self, uint32_t worker_id, bint block=True, double timeout=-1.0):
        """
        Ultra-low-latency wait with progressive backoff strategy.
        """
        cdef uint32_t spin_per_check = 1000
        cdef time_t start_time = 0
        cdef time_t current_time
        cdef double elapsed = 0.0
        cdef uint32_t spin_count = 0
        cdef uint32_t sleep_us = 0
        cdef bint use_timeout = timeout > 0
        cdef _ConcurrentBufferHeader * header = self._header  # Local pointer for direct access
        cdef uint64_t idx = self._header.heads[worker_id]

        if worker_id >= self._header.n_workers:
            raise IndexError("worker_id exceeds number of workers")

        if (not block) and self.is_empty(worker_id):
            raise BufferError("Buffer is empty for this worker")

        time(&start_time)

        while True:
            # Check for data - direct struct field access
            if idx != header.tail:
                self._header.heads[worker_id] += 1
                return self.get(idx=idx)

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

    @property
    def head(self) -> list[int]:
        """Return a Python list containing all head indices from the buffer header."""
        return [self._header.heads[i] for i in range(self._header.n_workers)]

    @property
    def tail(self) -> int:
        return self._header.tail

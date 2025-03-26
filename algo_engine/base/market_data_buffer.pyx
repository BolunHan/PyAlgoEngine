# cython: language_level=3
from cpython.buffer cimport PyBUF_SIMPLE, PyObject_GetBuffer, PyBuffer_Release
from cpython.bytes cimport PyBytes_FromStringAndSize
from libc.stdlib cimport malloc, free, qsort
from libc.string cimport memcpy, memset
from libc.stdint cimport uint8_t, uint32_t, uint64_t

from .market_data cimport compare_md_ptr, MarketData, DataType, _MetaInfo, _MarketDataBuffer, _TransactionDataBuffer, _OrderDataBuffer, _TickDataLiteBuffer, _TickDataBuffer, _CandlestickBuffer, TICKER_SIZE


# Define buffer header structure
cdef struct _BufferHeader:
    uint8_t dtype  # Data type (0 for mixed)
    bint sorted  # Sorted flag
    uint32_t count  # Number of entries
    uint32_t current_index  # Current index for iteration
    uint64_t pointer_offset  # Offset to pointer array
    uint64_t max_entries  # Maximum number of pointers that can be stored
    uint64_t data_offset  # Offset to data section
    uint64_t tail_offset  # Current offset at the end of used data
    uint64_t max_offset  # Maximum offset (size of data section)
    double current_timestamp  # Current maximum timestamp

cdef class MarketDataBuffer:
    """
    A buffer for storing and managing multiple market data entries.
    Supports iteration, sorting, and data management.

    The buffer layout is:
    - Header section: Stores metadata about the buffer
    - Pointer array section: Stores relative pointers (offsets) to market data entries
    - Data section: Stores the actual market data entries
    """
    cdef char * _buffer  # Pointer to the buffer
    cdef Py_buffer _view  # Buffer view
    cdef bint _view_obtained  # Flag to track if buffer view was obtained
    cdef _BufferHeader * _header  # Pointer to the header section
    cdef uint64_t * _pointers  # Pointer to the pointer array section
    cdef char * _data  # Pointer to the data section

    def __cinit__(self):
        """
        Initialize the MarketDataBuffer without allocating memory.
        """
        self._buffer = NULL
        self._view_obtained = False
        self._header = NULL
        self._pointers = NULL
        self._data = NULL

    def __init__(self, buffer, uint8_t dtype=0, uint64_t max_size=0):
        """
        Initialize the MarketDataBuffer with a memory buffer.

        Parameters:
        -----------
        buffer : object
            A Python object supporting the buffer protocol (e.g., memoryview, bytearray, RawArray)
        dtype : uint8_t, optional
            Data type for validation. If 0, mixed data types are allowed.
        max_size : uint64_t, optional
            Maximum size for data section. If 0, calculated based on buffer size.
        """
        # Get buffer view
        cdef Py_buffer view

        # Determine pointer array size based on dtype and buffer size
        cdef uint64_t estimated_entry_size
        cdef uint64_t max_entries

        cdef uint64_t pointer_size

        # Calculate offsets
        cdef uint64_t pointer_offset
        cdef uint64_t data_offset

        PyObject_GetBuffer(buffer, &view, PyBUF_SIMPLE)

        # Calculate sizes and offsets
        cdef uint64_t total_size = view.len
        cdef uint64_t header_size = sizeof(_BufferHeader)

        self._view = view
        self._view_obtained = True

        # Set buffer pointer
        self._buffer = <char *> view.buf

        if max_size > 0:
            # Use provided max_size
            max_entries = max_size
        else:
            # Estimate max entries (accounting for header and pointer array)
            # Each pointer is an uint64_t (8 bytes)
            # Formula: max_entries = (total_size - header_size) / (8 + estimated_entry_size)
            if dtype == 0:
                # For mixed types, use the smallest possible entry size to estimate max entries
                estimated_entry_size = min(MarketData.get_size(DataType.DTYPE_TICK_LITE), MarketData.get_size(DataType.DTYPE_ORDER), MarketData.get_size(DataType.DTYPE_BAR))
            else:
                # For specific dtype, use the exact entry size
                estimated_entry_size = MarketData.get_size(dtype)
            # Calculate based on buffer size
            max_entries = (total_size - header_size) // (sizeof(uint64_t) + estimated_entry_size)

        # Ensure at least some entries can be stored
        if max_entries <= 0:
            raise ValueError(f"Buffer size {total_size} is too small to store any entries")

        # Calculate pointer array size (in bytes)
        pointer_size = max_entries * sizeof(uint64_t)

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
        self._header.max_entries = max_entries  # Maximum number of pointers
        self._header.data_offset = data_offset
        self._header.tail_offset = 0  # No data yet
        self._header.max_offset = total_size - data_offset  # Maximum data size
        self._header.current_timestamp = 0.0  # No timestamp yet

        # Set pointers to sections
        self._pointers = <uint64_t *> (self._buffer + pointer_offset)
        self._data = self._buffer + data_offset

    def __dealloc__(self):
        """
        Release the buffer view when the object is deallocated.
        """
        if self._view_obtained:
            PyBuffer_Release(&self._view)
            self._view_obtained = False

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
        cdef size_t view_size = view.len

        PyObject_GetBuffer(buffer, &view, PyBUF_SIMPLE)
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
        instance._pointers = <uint64_t *> (instance._buffer + instance._header.pointer_offset)
        instance._data = instance._buffer + instance._header.data_offset

        return instance

    @classmethod
    def from_bytes(cls, bytes data, buffer):
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

        # Get buffer view
        cdef Py_buffer view
        cdef size_t view_size = view.len
        PyObject_GetBuffer(buffer, &view, PyBUF_SIMPLE)
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
        instance._pointers = <uint64_t *> (instance._buffer + instance._header.pointer_offset)
        instance._data = instance._buffer + instance._header.data_offset

        return instance

    cpdef push(self, MarketData market_data):
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
        if self._header.count >= self._header.max_entries:
            raise MemoryError("Not enough space in pointer array for new entry")

        # Copy data directly from market_data._data to our buffer at the current tail offset
        memcpy(self._data + self._header.tail_offset, data_ptr, entry_size)

        # Add pointer (offset) to the new entry
        self._pointers[self._header.count] = self._header.tail_offset

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

        # Get size based on dtype
        cdef size_t entry_size = MarketData.get_size(dtype)

        # Check if we have enough space in the data section
        if self._header.tail_offset + entry_size > self._header.max_offset:
            raise MemoryError("Not enough space in buffer for new entry")

        # Check if we have enough space in the pointer array
        if self._header.count >= self._header.max_entries:
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
        self._pointers[self._header.count] = self._header.tail_offset

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

        if 'timestamp' in kwargs:
            (<_MetaInfo *> buffer).timestamp = kwargs['timestamp']

        # Set specific fields based on dtype
        if dtype == DataType.DTYPE_TRANSACTION:
            self._set_transaction_fields(buffer, kwargs)
        elif dtype == DataType.DTYPE_ORDER:
            self._set_order_fields(buffer, kwargs)
        elif dtype == DataType.DTYPE_TICK_LITE:
            self._set_tick_lite_fields(buffer, kwargs)
        elif dtype == DataType.DTYPE_TICK:
            self._set_tick_fields(buffer, kwargs)
        elif dtype == DataType.DTYPE_BAR:
            self._set_bar_fields(buffer, kwargs)

    cdef void _set_transaction_fields(self, char * buffer, dict kwargs):
        """
        Set fields for TransactionData.
        """
        cdef _TransactionDataBuffer * data = <_TransactionDataBuffer *> buffer

        if 'price' in kwargs:
            data.price = kwargs['price']

        if 'volume' in kwargs:
            data.volume = kwargs['volume']

        if 'side' in kwargs:
            data.side = kwargs['side']

        if 'multiplier' in kwargs:
            data.multiplier = kwargs['multiplier']

        if 'notional' in kwargs:
            data.notional = kwargs['notional']

        # Note: transaction_id, buy_id, sell_id are not set here
        # as they require more complex handling

    cdef void _set_order_fields(self, char * buffer, dict kwargs):
        """
        Set fields for OrderData.
        """
        cdef _OrderDataBuffer * data = <_OrderDataBuffer *> buffer

        if 'price' in kwargs:
            data.price = kwargs['price']

        if 'volume' in kwargs:
            data.volume = kwargs['volume']

        if 'side' in kwargs:
            data.side = kwargs['side']

        if 'order_type' in kwargs:
            data.order_type = kwargs['order_type']

        # Note: order_id is not set here as it requires more complex handling

    cdef void _set_tick_lite_fields(self, char * buffer, dict kwargs):
        """
        Set fields for TickDataLite.
        """
        cdef _TickDataLiteBuffer * data = <_TickDataLiteBuffer *> buffer

        if 'bid_price' in kwargs:
            data.bid_price = kwargs['bid_price']

        if 'bid_volume' in kwargs:
            data.bid_volume = kwargs['bid_volume']

        if 'ask_price' in kwargs:
            data.ask_price = kwargs['ask_price']

        if 'ask_volume' in kwargs:
            data.ask_volume = kwargs['ask_volume']

        if 'last_price' in kwargs:
            data.last_price = kwargs['last_price']

        if 'total_traded_volume' in kwargs:
            data.total_traded_volume = kwargs['total_traded_volume']

        if 'total_traded_notional' in kwargs:
            data.total_traded_notional = kwargs['total_traded_notional']

        if 'total_trade_count' in kwargs:
            data.total_trade_count = kwargs['total_trade_count']

    cdef void _set_tick_fields(self, char * buffer, dict kwargs):
        """
        Set fields for TickData.
        """
        # First set the TickDataLite fields (which are part of TickData)
        self._set_tick_lite_fields(buffer, kwargs)

        # TickData also has bid and ask order books, but these are complex structures
        # and would require more detailed handling which is beyond the scope of this implementation

    cdef void _set_bar_fields(self, char * buffer, dict kwargs):
        """
        Set fields for BarData (CandlestickBuffer).
        """
        cdef _CandlestickBuffer * data = <_CandlestickBuffer *> buffer

        if 'bar_span' in kwargs:
            data.bar_span = kwargs['bar_span']

        if 'high_price' in kwargs:
            data.high_price = kwargs['high_price']

        if 'low_price' in kwargs:
            data.low_price = kwargs['low_price']

        if 'open_price' in kwargs:
            data.open_price = kwargs['open_price']

        if 'close_price' in kwargs:
            data.close_price = kwargs['close_price']

        if 'volume' in kwargs:
            data.volume = kwargs['volume']

        if 'notional' in kwargs:
            data.notional = kwargs['notional']

        if 'trade_count' in kwargs:
            data.trade_count = kwargs['trade_count']

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
                actual_ptrs[i] = <_MetaInfo *> (self._data + self._pointers[i])

            # 3. Sort the actual pointers using qsort with compare_md_ptr
            qsort(actual_ptrs, self._header.count, sizeof(_MetaInfo *), compare_md_ptr)

            # 4. Convert back to offsets
            for i in range(self._header.count):
                self._pointers[i] = <uint64_t> (<char *> actual_ptrs[i] - self._data)

            # 5. Free the temporary array
            free(actual_ptrs)
        else:
            # Thread-safe sort (create a copy)
            # 1. Allocate memory for a copy of the pointer array
            ptr_array = <uint64_t *> malloc(self._header.count * sizeof(uint64_t))
            if ptr_array == NULL:
                raise MemoryError("Failed to allocate memory for pointer array copy")

            # 2. Copy the pointer array
            memcpy(ptr_array, self._pointers, self._header.count * sizeof(uint64_t))

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
            memcpy(self._pointers, ptr_array, self._header.count * sizeof(uint64_t))

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

    def __next__(self):
        """
        Get the next market data entry.
        """
        if self._header.current_index >= self._header.count:
            raise StopIteration

        # Get offset for current index
        cdef uint64_t offset = self._pointers[self._header.current_index]

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

        # Import here to avoid circular imports
        from transaction import TransactionData as PyTransactionData, OrderData as PyOrderData
        from tick import TickDataLite as PyTickDataLite, TickData as PyTickData
        from candlestick import BarData as PyBarData

        # Create appropriate object based on dtype
        if dtype == DataType.DTYPE_TRANSACTION:
            return PyTransactionData.from_bytes(data)
        elif dtype == DataType.DTYPE_ORDER:
            return PyOrderData.from_bytes(data)
        elif dtype == DataType.DTYPE_TICK_LITE:
            return PyTickDataLite.from_bytes(data)
        elif dtype == DataType.DTYPE_TICK:
            return PyTickData.from_bytes(data)
        elif dtype == DataType.DTYPE_BAR:
            return PyBarData.from_bytes(data)
        else:
            return MarketData.from_bytes(data)

    def __len__(self):
        """
        Get the number of entries in the buffer.
        """
        return self._header.count

    def to_bytes(self):
        """
        Convert the buffer to bytes.
        """
        if self._buffer == NULL:
            return b''

        # Calculate total size (header + pointer array + used data)
        cdef uint64_t total_size = self._header.data_offset + self._header.tail_offset

        # Return the buffer up to the used size
        return PyBytes_FromStringAndSize(self._buffer, total_size)

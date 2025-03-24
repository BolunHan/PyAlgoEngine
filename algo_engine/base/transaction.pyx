# cython: language_level=3
from market_data cimport MarketData, _MarketDataBuffer, _TransactionDataBuffer, UnionID, TransactionHelper, DataType, ID_SIZE

from libc.string cimport memcpy, memset
from cpython.buffer cimport PyBuffer_FillInfo
from cpython.bytes cimport PyBytes_FromStringAndSize
from cpython.mem cimport PyMem_Malloc

cdef class TransactionData(MarketData):
    """
    Represents transaction data for a specific market.
    """
    def __cinit__(self):
        """
        Allocate memory for the transaction data structure but don't initialize it.
        """
        self._dtype = DataType.DTYPE_TRANSACTION
        self._owner = True

    def __init__(self, str ticker, double timestamp, double price, double volume, int side=0, double multiplier=1.0, double notional=0.0, object transaction_id=None, object buy_id=None, object sell_id=None):
        """
        Initialize the transaction data with values.
        """
        if not self._owner:
            raise MemoryError(f"Can not initialize a view of {self.__class__.__name__}.")

        self._data = <_MarketDataBuffer*> PyMem_Malloc(sizeof(_TransactionDataBuffer))
        memset(self._data, 0, sizeof(_TransactionDataBuffer))

        # Initialize base class fields
        MarketData.__init__(self, ticker, timestamp)

        # Set data type for TransactionData
        self._data.MetaInfo.dtype = DataType.DTYPE_TRANSACTION

        # Initialize transaction-specific fields
        self._data.TransactionData.price = price
        self._data.TransactionData.volume = volume
        self._data.TransactionData.side = side
        self._data.TransactionData.multiplier = multiplier

        # Calculate notional if not provided
        if notional == 0.0:
            self._data.TransactionData.notional = price * volume * multiplier
        else:
            self._data.TransactionData.notional = notional

        # Initialize IDs
        self._set_id(id_ptr=&self._data.TransactionData.transaction_id, id_value=transaction_id)
        self._set_id(id_ptr=&self._data.TransactionData.buy_id, id_value=buy_id)
        self._set_id(id_ptr=&self._data.TransactionData.sell_id, id_value=sell_id)

    cdef void _set_id(self, UnionID* id_ptr, object id_value):
        """
        Set an ID field in the transaction data.
        """
        cdef bytes id_bytes
        cdef int id_len

        if id_value is None:
            id_ptr.id_type = 0  # None type
        elif isinstance(id_value, int):
            id_ptr.id_int.id_type = 1  # Int type
            # Convert int to string and store in data
            id_bytes = id_value.to_bytes(ID_SIZE - 1, byteorder='little', signed=True)
            id_len = min(len(id_bytes), ID_SIZE - 1)
            memset(id_ptr.id_int.data, 0, ID_SIZE)
            memcpy(id_ptr.id_int.data, <char*>id_bytes, id_len)
        elif isinstance(id_value, str):
            id_ptr.id_str.id_type = 2  # Str type
            id_bytes = id_value.encode('utf-8')
            id_len = min(len(id_bytes), ID_SIZE - 1)
            memset(id_ptr.id_str.data, 0, ID_SIZE)
            memcpy(id_ptr.id_str.data, <char*>id_bytes, id_len)

    cdef object _get_id(self, UnionID* id_ptr):
        """
        Get an ID field from the transaction data.
        """
        if id_ptr.id_type == 0:
            return None
        elif id_ptr.id_type == 1:  # Int type
            return int.from_bytes(id_ptr.id_int.data[:ID_SIZE - 1], byteorder='little', signed=True)
        elif id_ptr.id_type == 2:  # Str type
            return id_ptr.id_str.data.decode('utf-8').rstrip('\0')

        return None

    @classmethod
    def from_buffer(cls, const unsigned char[:] buffer):
        """
        Create a new instance from a buffer.
        """
        cdef TransactionData instance = cls.__new__(cls)

        # Point to the buffer data
        instance._data = <_MarketDataBuffer*>&buffer[0]
        instance._owner = False

        return instance

    @classmethod
    def from_bytes(cls, bytes data):
        """
        Create a new instance from bytes.
        Creates a copy of the data, so the instance owns the memory.
        """
        cdef TransactionData instance = cls.__new__(cls)
        cdef const char* data_ptr = <const char*>data

        instance._owner = True
        instance._data = <_MarketDataBuffer*>PyMem_Malloc(sizeof(_TransactionDataBuffer))

        if instance._data == NULL:
            raise MemoryError("Failed to allocate memory for TransactionData")

        memcpy(instance._data, data_ptr, sizeof(_TransactionDataBuffer))

        return instance

    def to_bytes(self):
        """
        Convert the transaction data to bytes.
        """
        if self._data == NULL:
            raise ValueError("Cannot convert uninitialized data to bytes")

        return PyBytes_FromStringAndSize(<char*>self._data, sizeof(_TransactionDataBuffer))

    def __getbuffer__(self, Py_buffer* buffer, int flags):
        """
        Implement the buffer protocol for read-only access.
        """
        if self._data == NULL:
            raise ValueError("Cannot get buffer from uninitialized data")
        PyBuffer_FillInfo(buffer, self, <_TransactionDataBuffer*>self._data, sizeof(_TransactionDataBuffer), 1, flags)

    def __copy__(self):
        """
        Create a copy of this instance.
        """
        if self._data == NULL:
            raise ValueError("Cannot copy uninitialized data")

        cdef TransactionData new_td = TransactionData.__new__(TransactionData)
        # Allocate memory specifically for a TransactionData buffer
        new_td._data = <_MarketDataBuffer*>PyMem_Malloc(sizeof(_TransactionDataBuffer))
        if new_td._data == NULL:
            raise MemoryError("Failed to allocate memory for copy")

        new_td._owner = True
        memcpy(new_td._data, self._data, sizeof(_TransactionDataBuffer))

        return new_td

    @property
    def price(self) -> float:
        """
        Get the transaction price.
        """
        if self._data == NULL:
            raise ValueError("Data not initialized")
        return self._data.TransactionData.price

    @property
    def volume(self) -> float:
        """
        Get the transaction volume.
        """
        if self._data == NULL:
            raise ValueError("Data not initialized")
        return self._data.TransactionData.volume

    @property
    def side(self) -> int:
        """
        Get the transaction side.
        """
        if self._data == NULL:
            raise ValueError("Data not initialized")
        return self._data.TransactionData.side

    @property
    def multiplier(self) -> float:
        """
        Get the transaction multiplier.
        """
        if self._data == NULL:
            raise ValueError("Data not initialized")
        return self._data.TransactionData.multiplier

    @property
    def notional(self) -> float:
        """
        Get the transaction notional value.
        """
        if self._data == NULL:
            raise ValueError("Data not initialized")
        return self._data.TransactionData.notional

    @property
    def transaction_id(self) -> object:
        """
        Get the transaction ID.
        """
        if self._data == NULL:
            raise ValueError("Data not initialized")
        return self._get_id(&self._data.TransactionData.transaction_id)

    @property
    def buy_id(self) -> object:
        """
        Get the buy ID.
        """
        if self._data == NULL:
            raise ValueError("Data not initialized")
        return self._get_id(&self._data.TransactionData.buy_id)

    @property
    def sell_id(self) -> object:
        """
        Get the sell ID.
        """
        if self._data == NULL:
            raise ValueError("Data not initialized")
        return self._get_id(&self._data.TransactionData.sell_id)

    @property
    def market_price(self) -> float:
        """
        Alias for the transaction price.
        """
        return self.price

    @property
    def flow(self) -> float:
        """
        Calculate the flow of the transaction.
        """
        if self._data == NULL:
            raise ValueError("Data not initialized")
        cdef int sign = TransactionHelper.get_sign(self._data.TransactionData.side)
        return sign * self._data.TransactionData.volume

    def __repr__(self) -> str:
        """
        String representation of the transaction data.
        """
        if self._data == NULL:
            return "TransactionData(uninitialized)"
        side_name = TransactionHelper.get_side_name(self._data.TransactionData.side).decode('utf-8')
        return f"TransactionData(ticker='{self.ticker}', timestamp={self.timestamp}, price={self.price}, volume={self.volume}, side={side_name})"

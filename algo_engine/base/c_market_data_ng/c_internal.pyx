from ..c_shm_allocator cimport ALLOCATOR


cdef class InternalData:
    def __cinit__(self):
        self.header = <_MarketDataBuffer*> &self._data
        self._data_addr = <uintptr_t> self._data_ptr

    def __init__(self, *, ticker: str, double timestamp, uint32_t code, **kwargs):
        # Initialize base class fields
        cdef Py_ssize_t ticker_len
        cdef const char* ticker_ptr = PyUnicode_AsUTF8AndSize(ticker, &ticker_len)
        memcpy(<void*> &self._data.ticker, ticker_ptr, min(ticker_len, TICKER_SIZE - 1))
        self._data.timestamp = timestamp
        self._data.dtype = data_type_t.DTYPE_INTERNAL
        if kwargs: self.__dict__.update(kwargs)

        # Initialize internal-specific fields
        self._data.code = code

    def __repr__(self) -> str:
        return f"<InternalData>([{self.market_time:%Y-%m-%d %H:%M:%S}] {self.ticker}, code={self.code})"

    def __reduce__(self):
        return self.__class__.from_bytes, (self.to_bytes(),), self.__dict__

    def __setstate__(self, state):
        if state:
            self.__dict__.update(state)

    def __copy__(self):
        cdef TransactionData instance = TransactionData.__new__(TransactionData)
        memcpy(<void*> &instance._data, <const char*> &self._data, sizeof(_InternalBuffer))
        return instance

    @classmethod
    def buffer_size(cls):
        return sizeof(_InternalBuffer)

    cdef bytes c_to_bytes(self):
        return PyBytes_FromStringAndSize(<char*> &self._data, sizeof(self._data))

    @staticmethod
    cdef InternalData c_from_bytes(bytes data):
        cdef InternalData instance = InternalData.__new__(InternalData)
        memcpy(<void*> &instance._data, <const char*> data, sizeof(_InternalBuffer))
        return instance

    def to_bytes(self) -> bytes:
        return self.c_to_bytes()

    @classmethod
    def from_bytes(cls, bytes data):
        return InternalData.c_from_bytes(data)

    @property
    def ticker(self) -> str:
        return PyUnicode_FromString(&self._data.ticker[0])

    @property
    def timestamp(self) -> float:
        return self._data.timestamp

    @property
    def dtype(self) -> int:
        return self._data.dtype

    @property
    def topic(self) -> str:
        cdef str ticker_str = PyUnicode_FromString(&self._data.ticker[0])
        return f'{ticker_str}.{self.__class__.__name__}'

    @property
    def market_time(self) :
        return _MarketDataVirtualBase.c_to_dt(self._data.timestamp)

    @property
    def price(self) -> float:
        return 0.

    @property
    def code(self) -> int:
        return self._data.code

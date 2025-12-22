import enum
import uuid

from cpython.bytes cimport PyBytes_FromStringAndSize
from cpython.datetime cimport datetime
from cpython.unicode cimport PyUnicode_AsUTF8AndSize, PyUnicode_AsUTF8
from libc.stdint cimport UINT64_MAX, INT64_MIN, int64_t
from libc.string cimport memset, memcpy


class DataType(enum.IntEnum):
    DTYPE_UNKNOWN = md_data_type.DTYPE_UNKNOWN
    DTYPE_INTERNAL = md_data_type.DTYPE_INTERNAL
    DTYPE_MARKET_DATA = md_data_type.DTYPE_MARKET_DATA
    DTYPE_TRANSACTION = md_data_type.DTYPE_TRANSACTION
    DTYPE_ORDER = md_data_type.DTYPE_ORDER
    DTYPE_TICK_LITE = md_data_type.DTYPE_TICK_LITE
    DTYPE_TICK = md_data_type.DTYPE_TICK
    DTYPE_BAR = md_data_type.DTYPE_BAR
    DTYPE_REPORT = md_data_type.DTYPE_REPORT
    DTYPE_INSTRUCTION = md_data_type.DTYPE_INSTRUCTION


cdef bint MD_CFG_LOCKED = False
cdef bint MD_CFG_SHARED = True
cdef bint MD_CFG_FREELIST = True
cdef size_t MD_CFG_BOOK_SIZE = BOOK_SIZE


cdef class EnvConfigContext:
    def __cinit__(self, **kwargs):
        self.overrides = kwargs
        self.originals = {}

    cdef void c_activate(self):
        if 'locked' in self.overrides:
            global MD_CFG_LOCKED
            self.originals['locked'] = MD_CFG_LOCKED
            MD_CFG_LOCKED = self.overrides['locked']

        if 'shared' in self.overrides:
            global MD_CFG_SHARED
            self.originals['shared'] = MD_CFG_SHARED
            MD_CFG_SHARED = self.overrides['shared']

        if 'freelist' in self.overrides:
            global MD_CFG_FREELIST
            self.originals['freelist'] = MD_CFG_FREELIST
            MD_CFG_FREELIST = self.overrides['freelist']

        if 'book_size' in self.overrides:
            global MD_CFG_BOOK_SIZE
            self.originals['book_size'] = MD_CFG_BOOK_SIZE
            MD_CFG_BOOK_SIZE = self.overrides['book_size']

    cdef void c_deactivate(self):
        if 'locked' in self.originals:
            global MD_CFG_LOCKED
            MD_CFG_LOCKED = self.originals.pop('locked')

        if 'shared' in self.originals:
            global MD_CFG_SHARED
            MD_CFG_SHARED = self.originals.pop('shared')

        if 'freelist' in self.originals:
            global MD_CFG_FREELIST
            MD_CFG_FREELIST = self.originals.pop('freelist')

        if 'book_size' in self.originals:
            global MD_CFG_BOOK_SIZE
            MD_CFG_BOOK_SIZE = self.originals.pop('book_size')

    def __repr__(self):
        return f'{self.__class__.__name__}({self.overrides!r})'

    def __enter__(self):
        self.c_activate()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.c_deactivate()

    def __or__(self, EnvConfigContext other):
        if not isinstance(other, EnvConfigContext):
            return NotImplemented
        merged_overrides = self.overrides | other.overrides
        return EnvConfigContext(**merged_overrides)

    def __invert__(self):
        return EnvConfigContext.__new__(
            EnvConfigContext,
            **{k: not v if isinstance(v, bool) else v for k, v in self.overrides.items()}
        )

    def __call__(self, func):
        def wrapper(*args, **kwargs):
            self.c_activate()
            ret = func(*args, **kwargs)
            self.c_deactivate()
            return ret
        return wrapper


cdef EnvConfigContext MD_SHARED     = EnvConfigContext(shared=True)
cdef EnvConfigContext MD_LOCKED     = EnvConfigContext(locked=True)
cdef EnvConfigContext MD_FREELIST   = EnvConfigContext(freelist=True)
cdef EnvConfigContext MD_BOOK5      = EnvConfigContext(book_size=5)
cdef EnvConfigContext MD_BOOK10     = EnvConfigContext(book_size=10)
cdef EnvConfigContext MD_BOOK20     = EnvConfigContext(book_size=20)


globals()['MD_SHARED'] = MD_SHARED
globals()['MD_LOCKED'] = MD_LOCKED
globals()['MD_FREELIST'] = MD_FREELIST
globals()['MD_BOOK5'] = MD_BOOK5
globals()['MD_BOOK10'] = MD_BOOK10
globals()['MD_BOOK20'] = MD_BOOK20


cdef void c_set_id(md_id* id_ptr, object id_value):
    cdef bytes id_bytes
    cdef const char* id_chars
    cdef Py_ssize_t id_len

    memset(<void*> id_ptr.data, 0, ID_SIZE + 1)

    if id_value is None:
        # None type
        id_ptr.id_type = md_id_type.MID_UNKNOWN
    elif isinstance(id_value, int):
        if id_value >= 0:
            if id_value < UINT64_MAX and MID_ALLOW_INT64:
                # uint64_t type
                id_ptr.id_type = md_id_type.MID_UINT64
                (<uint64_t*> id_ptr.data)[0] = <uint64_t> id_value
                return
            elif id_value < UINT128_MAX and MID_ALLOW_INT128:
                # uint128_t type
                id_ptr.id_type = md_id_type.MID_UINT128
                # (<uint128_t*> id_ptr.data)[0] = <uint128_t> id_value
                c_write_uint128(<void*> id_ptr.data, <uint128_t> id_value)
                return
            else:
                raise ValueError(f'Integer ID {id_value} is too large to fit in the ID buffer.')
        else:
            if id_value > INT64_MIN and MID_ALLOW_INT64:
                id_ptr.id_type = md_id_type.MID_INT64
                (<int64_t*> id_ptr.data)[0] = <int64_t> id_value
                return
            elif id_value > INT128_MIN and MID_ALLOW_INT128:
                id_ptr.id_type = md_id_type.MID_INT128
                # (<int128_t*> id_ptr.data)[0] = <int128_t> id_value
                c_write_int128(<void*> id_ptr.data, <int128_t> id_value)
                return
            else:
                raise ValueError(f'Integer ID {id_value} is too small to fit in the ID buffer.')
    elif isinstance(id_value, str):
        # str type
        id_ptr.id_type = md_id_type.MID_STRING
        id_chars = PyUnicode_AsUTF8AndSize(id_value, &id_len)
        if id_len <= ID_SIZE:
            memcpy(<void*> id_ptr.data, id_chars, id_len)
            return
        else:
            raise ValueError(f'String ID {id_value} is too long to fit in the ID buffer.')
    elif isinstance(id_value, bytes):
        # bytes type
        id_ptr.id_type = md_id_type.MID_BYTE
        id_chars = <const char*> id_value
        id_len = len(id_value)
        if id_len <= ID_SIZE:
            memcpy(<void*> id_ptr.data, id_chars, id_len)
            return
        else:
            raise ValueError(f'Byte ID {id_value} is too long to fit in the ID buffer.')
    elif isinstance(id_value, uuid.UUID):
        # uuid type
        id_ptr.id_type = md_id_type.MID_UUID
        id_bytes = id_value.bytes_le
        if MID_ALLOW_INT128:
            memcpy(<void*> id_ptr.data, <const char*> id_bytes, 16)
            return
        else:
            raise ValueError(f'UUID ID {id_value} is too long to fit in the ID buffer.')


cdef object c_get_id(md_id* id_ptr):
    if id_ptr.id_type == md_id_type.MID_UNKNOWN:
        return None
    elif id_ptr.id_type == md_id_type.MID_UINT64:
        return (<uint64_t*> id_ptr.data)[0]
    elif id_ptr.id_type == md_id_type.MID_INT64:
        return (<int64_t*> id_ptr.data)[0]
    elif id_ptr.id_type == md_id_type.MID_UINT128:
        return c_read_uint128(<void*> id_ptr.data)
    elif id_ptr.id_type == md_id_type.MID_INT128:
        return c_read_int128(<void*> id_ptr.data)
    elif id_ptr.id_type == md_id_type.MID_STRING:
        return PyUnicode_FromString(&id_ptr.data[0])
    elif id_ptr.id_type == md_id_type.MID_BYTE:
        return PyBytes_FromStringAndSize(&id_ptr.data[0], ID_SIZE).rstrip(b'\0')
    elif id_ptr.id_type == md_id_type.MID_UUID:
        return uuid.UUID(bytes_le=id_ptr.data[:16])
    raise ValueError(f'Cannot decode the id buffer with type {id_ptr.id_type}.')


cdef void c_set_long_id(long_md_id* id_ptr, object id_value):
    cdef bytes id_bytes
    cdef const char* id_chars
    cdef Py_ssize_t id_len

    memset(<void*> id_ptr.data, 0, LONG_ID_SIZE + 1)

    if id_value is None:
        # None type
        id_ptr.id_type = md_id_type.MID_UNKNOWN
    elif isinstance(id_value, int):
        if id_value >= 0:
            if id_value < UINT64_MAX and MID_ALLOW_INT64:
                # uint64_t type
                id_ptr.id_type = md_id_type.MID_UINT64
                (<uint64_t*> id_ptr.data)[0] = <uint64_t> id_value
                return
            elif id_value < UINT128_MAX and MID_ALLOW_INT128:
                # uint128_t type
                id_ptr.id_type = md_id_type.MID_UINT128
                # (<uint128_t*> id_ptr.data)[0] = <uint128_t> id_value
                c_write_uint128(<void*> id_ptr.data, <uint128_t> id_value)
                return
            else:
                raise ValueError(f'Integer ID {id_value} is too large to fit in the ID buffer.')
        else:
            if id_value > INT64_MIN and MID_ALLOW_INT64:
                id_ptr.id_type = md_id_type.MID_INT64
                (<int64_t*> id_ptr.data)[0] = <int64_t> id_value
                return
            elif id_value > INT128_MIN and MID_ALLOW_INT128:
                id_ptr.id_type = md_id_type.MID_INT128
                # (<int128_t*> id_ptr.data)[0] = <int128_t> id_value
                c_write_int128(<void*> id_ptr.data, <int128_t> id_value)
                return
            else:
                raise ValueError(f'Integer ID {id_value} is too small to fit in the ID buffer.')
    elif isinstance(id_value, str):
        # str type
        id_ptr.id_type = md_id_type.MID_STRING
        id_chars = PyUnicode_AsUTF8AndSize(id_value, &id_len)
        if id_len <= LONG_ID_SIZE:
            memcpy(<void*> id_ptr.data, id_chars, id_len)
            return
        else:
            raise ValueError(f'String ID {id_value} is too long to fit in the ID buffer.')
    elif isinstance(id_value, bytes):
        # bytes type
        id_ptr.id_type = md_id_type.MID_BYTE
        id_chars = <const char*> id_value
        id_len = len(id_value)
        if id_len <= LONG_ID_SIZE:
            memcpy(<void*> id_ptr.data, id_chars, id_len)
            return
        else:
            raise ValueError(f'Byte ID {id_value} is too long to fit in the ID buffer.')
    elif isinstance(id_value, uuid.UUID):
        # uuid type
        id_ptr.id_type = md_id_type.MID_UUID
        id_bytes = id_value.bytes_le
        if MID_ALLOW_INT128:
            memcpy(<void*> id_ptr.data, <const char*> id_bytes, 16)
            return
        else:
            raise ValueError(f'UUID ID {id_value} is too long to fit in the ID buffer.')


cdef object c_get_long_id(long_md_id* id_ptr):
    if id_ptr.id_type == md_id_type.MID_UNKNOWN:
        return None
    elif id_ptr.id_type == md_id_type.MID_UINT64:
        return (<uint64_t*> id_ptr.data)[0]
    elif id_ptr.id_type == md_id_type.MID_INT64:
        return (<int64_t*> id_ptr.data)[0]
    elif id_ptr.id_type == md_id_type.MID_UINT128:
        return c_read_uint128(<void*> id_ptr.data)
    elif id_ptr.id_type == md_id_type.MID_INT128:
        return c_read_int128(<void*> id_ptr.data)
    elif id_ptr.id_type == md_id_type.MID_STRING:
        return PyUnicode_FromString(&id_ptr.data[0])
    elif id_ptr.id_type == md_id_type.MID_BYTE:
        return PyBytes_FromStringAndSize(&id_ptr.data[0], LONG_ID_SIZE).rstrip(b'\0')
    elif id_ptr.id_type == md_id_type.MID_UUID:
        return uuid.UUID(bytes_le=id_ptr.data[:16])
    raise ValueError(f'Cannot decode the id buffer with type {id_ptr.id_type}.')


cdef class MarketData:
    def __dealloc__(self):
        if not self.owner:
            return

        if self.header:
            c_recycle_buffer(self.header)

    def __reduce__(self):
        return MarketData.from_bytes, (self.to_bytes(),), self.__dict__

    def __setstate__(self, state):
        if state:
            self.__dict__.update(state)

    def __copy__(self):
        cdef object cls = self.__class__
        cdef MarketData instance = <MarketData> cls.__new__(cls)
        cdef md_data_type dtype = self.header.meta_info.dtype
        cdef md_variant* header = c_md_new(dtype, NULL, NULL, <int> MD_CFG_LOCKED)
        cdef size_t size = c_md_get_size(self.header.meta_info.dtype)
        memcpy(<void*> instance.header, <const char*> self.header, size)
        instance.__dict__.update(self.__dict__)
        return instance

    @staticmethod
    cdef inline object c_from_header(md_variant* market_data, bint owner=False):
        cdef md_data_type dtype = market_data.meta_info.dtype

        if dtype == md_data_type.DTYPE_INTERNAL:
            return internal_from_header(market_data, owner)
        elif dtype == md_data_type.DTYPE_TRANSACTION:
            return transaction_from_header(market_data, owner)
        elif dtype == md_data_type.DTYPE_ORDER:
            return order_from_header(market_data, owner)
        elif dtype == md_data_type.DTYPE_TICK_LITE:
            return tick_lite_from_header(market_data, owner)
        elif dtype == md_data_type.DTYPE_TICK:
            return tick_from_header(market_data, owner)
        elif dtype == md_data_type.DTYPE_BAR:
            return bar_from_header(market_data, owner)
        elif dtype == md_data_type.DTYPE_REPORT:
            return report_from_header(market_data, owner)
        elif dtype == md_data_type.DTYPE_INSTRUCTION:
            return instruction_from_header(market_data, owner)
        else:
            raise NotImplementedError()

    cdef inline size_t c_get_size(self):
        cdef md_data_type dtype = self.header.meta_info.dtype
        cdef size_t size = c_md_get_size(dtype)
        if not size:
            raise ValueError(f'Unknown data type {dtype}')
        return size

    cdef inline str c_dtype_name(self):
        cdef md_data_type dtype = self.header.meta_info.dtype
        cdef const char* dtype_name = c_md_dtype_name(dtype)
        if not dtype_name:
            raise ValueError(f'Unknown data type {dtype}')
        return PyUnicode_FromString(dtype_name)

    cdef inline void c_to_bytes(self, char* out):
        c_md_serialize(self.header, out)

    @staticmethod
    cdef inline md_variant* c_from_bytes(bytes data):
        return c_deserialize_buffer(<const char*> data)

    @staticmethod
    def buffer_size(md_data_type dtype):
        cdef size_t size = c_md_get_size(dtype)
        if not size:
            raise ValueError(f'Unknown data type {dtype}')
        return size

    def to_bytes(self) -> bytes:
        cdef size_t size = c_md_serialized_size(self.header)
        cdef bytes data = PyBytes_FromStringAndSize(NULL, size)
        c_md_serialize(self.header, <char*> data)
        return data

    @classmethod
    def from_bytes(cls, bytes data):
        cdef md_variant* header = MarketData.c_from_bytes(data)
        cdef MarketData instance = MarketData.c_from_header(header, owner=True)
        return instance

    @staticmethod
    def from_ptr(uintptr_t addr):
        cdef md_variant* header = <md_variant*> addr
        cdef MarketData instance = MarketData.c_from_header(header, owner=False)
        return instance

    property ticker:
        def __get__(self):
            if self.header.meta_info.ticker:
                return PyUnicode_FromString(self.header.meta_info.ticker)
            return None

        def __set__(self, str value):
            if value is None:
                self.header.meta_info.ticker = NULL
                return
            cdef const char* scr = PyUnicode_AsUTF8(value)
            cdef const char* istr = c_istr_synced(SHM_POOL if MD_CFG_SHARED else HEAP_POOL, scr)
            self.header.meta_info.ticker = istr

    property timestamp:
        def __get__(self):
            return self.header.meta_info.timestamp

    property dtype:
        def __get__(self):
            return DataType(self.header.meta_info.dtype)

    property topic:
        def __get__(self):
            return f'{self.ticker}.{self.c_dtype_name()}'

    property market_time:
        def __get__(self):
            return datetime.fromtimestamp(self.header.meta_info.timestamp, tz=C_PROFILE.time_zone)

    property market_price:
        def __get__(self):
            return c_md_get_price(self.header)

    property price:
        def __get__(self):
            return c_md_get_price(self.header)

    property address:
        def __get__(self):
            if self.header:
                return f'{<uintptr_t> self.header:#0x}'
            return None


cdef class FilterMode:
    # Class-level constants for flags
    NO_INTERNAL = FilterMode.__new__(FilterMode, md_filter_flag.NO_INTERNAL)
    NO_CANCEL = FilterMode.__new__(FilterMode, md_filter_flag.NO_CANCEL)
    NO_AUCTION = FilterMode.__new__(FilterMode, md_filter_flag.NO_AUCTION)
    NO_ORDER = FilterMode.__new__(FilterMode, md_filter_flag.NO_ORDER)
    NO_TRADE = FilterMode.__new__(FilterMode, md_filter_flag.NO_TRADE)
    NO_TICK = FilterMode.__new__(FilterMode, md_filter_flag.NO_TICK)

    def __cinit__(self, md_filter_flag value):
        self.value = value

    def __or__(self, FilterMode other):
        return FilterMode.__new__(FilterMode, self.value | other.value)

    def __and__(self, FilterMode other):
        return FilterMode.__new__(FilterMode, self.value & other.value)

    def __invert__(self):
        # Invert all bits except those beyond our known flags
        inverted_value = ~self.value & (
            md_filter_flag.NO_INTERNAL |
            md_filter_flag.NO_CANCEL |
            md_filter_flag.NO_AUCTION |
            md_filter_flag.NO_ORDER |
            md_filter_flag.NO_TRADE |
            md_filter_flag.NO_TICK
        )
        return FilterMode.__new__(FilterMode, inverted_value)

    def __contains__(self, FilterMode other):
        return (self.value & other.value) == other.value

    def __repr__(self):
        flags = []
        if md_filter_flag.NO_INTERNAL & self.value: flags.append("NO_INTERNAL")
        if md_filter_flag.NO_CANCEL & self.value: flags.append("NO_CANCEL")
        if md_filter_flag.NO_AUCTION & self.value: flags.append("NO_AUCTION")
        if md_filter_flag.NO_ORDER & self.value: flags.append("NO_ORDER")
        if md_filter_flag.NO_TRADE & self.value: flags.append("NO_TRADE")
        if md_filter_flag.NO_TICK & self.value: flags.append("NO_TICK")
        return f"<FilterMode {self.value:#0x}: {' | '.join(flags) or 'None'}>"

    @staticmethod
    cdef inline bint c_mask_data(md_variant* market_data, md_filter_flag filter_mode):
        cdef md_data_type dtype = market_data.meta_info.dtype
        cdef md_side side

        if md_filter_flag.NO_INTERNAL & filter_mode:
            if dtype == md_data_type.DTYPE_INTERNAL:
                return False

        if md_filter_flag.NO_CANCEL & filter_mode:
            if dtype == md_data_type.DTYPE_TRANSACTION:
                side = (<md_transaction_data*> market_data).side
                if c_md_side_offset(side) == md_offset.OFFSET_CANCEL:
                    return False

        cdef double timestamp = market_data.meta_info.timestamp

        if md_filter_flag.NO_AUCTION & filter_mode:
            if not C_PROFILE.c_timestamp_in_market_session(timestamp):
                return False

        if md_filter_flag.NO_ORDER & filter_mode:
            if dtype == md_data_type.DTYPE_ORDER:
                return False

        if md_filter_flag.NO_TRADE & filter_mode:
            if dtype == md_data_type.DTYPE_TRANSACTION:
                return False

        if md_filter_flag.NO_TICK & filter_mode:
            if dtype == md_data_type.DTYPE_TICK or dtype == md_data_type.DTYPE_TICK_LITE:
                return False

        return True

    @classmethod
    def all(cls):
        return FilterMode.__new__(
            FilterMode,
            md_filter_flag.NO_INTERNAL |
            md_filter_flag.NO_CANCEL |
            md_filter_flag.NO_AUCTION |
            md_filter_flag.NO_ORDER |
            md_filter_flag.NO_TRADE |
            md_filter_flag.NO_TICK
        )

    cpdef bint mask_data(self, MarketData market_data):
        return FilterMode.c_mask_data(market_data.header, self.value)


cdef class ConfigViewer:
    def __repr__(self):
        return (
            f"<ConfigViewer>("
            f"DEBUG={self.DEBUG}; "
            f"TICKER_SIZE={self.TICKER_SIZE}; "
            f"BOOK_SIZE={self.BOOK_SIZE}; "
            f"ID_SIZE={self.ID_SIZE}; "
            f"LONG_ID_SIZE={self.LONG_ID_SIZE}; "
            f"MAX_WORKERS={self.MAX_WORKERS}; "
            f"MD_CFG_LOCKED={self.MD_CFG_LOCKED}; "
            f"MD_CFG_SHARED={self.MD_CFG_SHARED}; "
            f"MD_CFG_FREELIST={self.MD_CFG_FREELIST}; "
            f"MD_CFG_BOOK_SIZE={self.MD_CFG_BOOK_SIZE}"
            f")"
        )

    property DEBUG:
        def __get__(self):
            return DEBUG

    property TICKER_SIZE:
        def __get__(self):
            return TICKER_SIZE

    property BOOK_SIZE:
        def __get__(self):
            return BOOK_SIZE

    property ID_SIZE:
        def __get__(self):
            return ID_SIZE

    property LONG_ID_SIZE:
        def __get__(self):
            return LONG_ID_SIZE

    property MAX_WORKERS:
        def __get__(self):
            return MAX_WORKERS

    property MD_CFG_LOCKED:
        def __get__(self):
            return MD_CFG_LOCKED

    property MD_CFG_SHARED:
        def __get__(self):
            return MD_CFG_SHARED

    property MD_CFG_FREELIST:
        def __get__(self):
            return MD_CFG_FREELIST

    property MD_CFG_BOOK_SIZE:
        def __get__(self):
            return MD_CFG_BOOK_SIZE


C_CONFIG = ConfigViewer.__new__(ConfigViewer)
globals()['CONFIG'] = C_CONFIG

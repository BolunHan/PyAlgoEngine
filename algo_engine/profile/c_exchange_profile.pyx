import enum
from libc.stdlib cimport free

from cpython.datetime cimport time as py_time
from cpython.object cimport Py_LT, Py_LE, Py_EQ, Py_NE, Py_GT, Py_GE


class SessionType(enum.IntEnum):
    NON_TRADING = session_type.SESSION_TYPE_NON_TRADING
    NORMINAL = session_type.SESSION_TYPE_NORMINAL
    SUSPENDED = session_type.SESSION_TYPE_SUSPENDED
    HALF_DAY = session_type.SESSION_TYPE_HALF_DAY
    CIRCUIT_BREAK = session_type.SESSION_TYPE_CIRCUIT_BREAK


class SessionPhase(enum.IntEnum):
    UNKNOWN = session_phase.SESSION_PHASE_UNKNOWN
    PREOPEN = session_phase.SESSION_PHASE_PREOPEN
    OPEN_AUCTION = session_phase.SESSION_PHASE_OPEN_AUCTION
    CONTINUOUS = session_phase.SESSION_PHASE_CONTINUOUS
    BREAK = session_phase.SESSION_PHASE_BREAK
    SUSPENDED = session_phase.SESSION_PHASE_SUSPENDED
    CLOSE_AUCTION = session_phase.SESSION_PHASE_CLOSE_AUCTION
    CLOSED = session_phase.SESSION_PHASE_CLOSED


class AuctionPhase(enum.IntEnum):
    ACTIVE = auction_phase.AUCTION_PHASE_ACTIVE
    NO_CANCEL = auction_phase.AUCTION_PHASE_NO_CANCEL
    FROZEN = auction_phase.AUCTION_PHASE_FROZEN
    UNCROSSING = auction_phase.AUCTION_PHASE_UNCROSSING
    DONE = auction_phase.AUCTION_PHASE_DONE


cdef class SessionTime:
    def __cinit__(
            self,
            uint8_t hour=0,
            uint8_t minute=0,
            uint8_t second=0,
            uint32_t nanosecond=0
    ):
        self.header = c_ex_profile_session_time_new(hour, minute, second, nanosecond)
        if not self.header:
            raise MemoryError(f"Failed to allocate memory for {self.__class__.__name__}")
        self.owner = True

    def __dealloc__(self):
        if not self.owner:
            return

        if self.header:
            free(<void*> self.header)

    @staticmethod
    cdef SessionTime c_from_header(session_time_t* header, bint owner):
        cdef SessionTime session_time = SessionTime.__new__(SessionTime, hour=header.hour, minute=header.minute, second=header.second, microsecond=header.nanosecond // 1000)
        session_time.header = header
        session_time.owner = owner
        return session_time

    # === Python Interfaces ===

    def __repr__(self):
        return f"<{self.__class__.__name__}>({self.to_pytime().__str__()})"

    def __richcmp__(self, object other, int op):
        cdef int cmp
        if isinstance(other, SessionTime):
             cmp = c_ex_profile_session_time_compare(self.header, (<SessionTime> other).header)
        elif isinstance(other, py_time):
            cmp = c_ex_profile_session_time_compare(self.header, (<SessionTime> SessionTime.from_pytime(other)).header)
        else:
            raise TypeError(f'Can not compare {self.__class__.__name__} with {other.__class__.__name__}')

        if op == Py_LT:
            return cmp < 0
        elif op == Py_LE:
            return cmp <= 0
        elif op == Py_EQ:
            return cmp == 0
        elif op == Py_NE:
            return cmp != 0
        elif op == Py_GT:
            return cmp > 0
        elif op == Py_GE:
            return cmp >= 0
        else:
            raise ValueError(f'Invalid comparison operator: {op}')

    @classmethod
    def from_pytime(cls, py_time t):
        cdef SessionTime instance = SessionTime.__new__(SessionTime, t.hour, t.minute, t.second, t.microsecond * 1000)
        return instance

    def to_pytime(self):
        return py_time(self.header.hour, self.header.minute, self.header.second, self.header.nanosecond // 1000)

    def isoformat(self, *args, **kwargs):
        return self.to_pytime().isoformat(*args, **kwargs)

    @classmethod
    def fromisoformat(cls, str time_str):
        return cls.from_pytime(py_time.fromisoformat(time_str))

    property hour:
        def __get__(self):
            return self.header.hour

    property minute:
        def __get__(self):
            return self.header.minute

    property second:
        def __get__(self):
            return self.header.second

    property microsecond:
        def __get__(self):
            return self.header.nanosecond // 1000

    property nanosecond:
        def __get__(self):
            return self.header.nanosecond

    property elapsed_seconds:
        def __get__(self):
            return self.header.elapsed_seconds

    property ts:
        def __get__(self):
            return c_ex_profile_time_to_ts(self.header.hour, self.header.minute, self.header.second, self.header.nanosecond)

    property session_phase:
        def __get__(self):
            return SessionPhase(self.header.phase)

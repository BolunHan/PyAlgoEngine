import enum
from libc.stdlib cimport free


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


cdef class SessionTime(py_time):
    def __init__(
            self,
            uint8_t hour=0,
            uint8_t minute=0,
            uint8_t second=0,
            uint32_t microsecond=0,
            *args,
            **kwargs
    ):
        super().__init__(hour, minute, second, microsecond, *args, **kwargs)
        self.header = c_ex_profile_session_time_new(hour, minute, second, microsecond * 1000)
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

    property session_phase:
        def __get__(self):
            return SessionPhase(self.header.phase)

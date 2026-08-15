from cpython.datetime cimport PyDateTime_GET_DAY, PyDateTime_GET_MONTH, PyDateTime_GET_YEAR, date as pydate
from libc.stdint cimport uintptr_t
from libc.stdlib cimport calloc
from libc.string cimport memset

# Windows: redirect EX_PROFILE to a pointer resolved at runtime via
# GetProcAddress (PE has no RTLD_GLOBAL-equivalent global symbol scope).
# Must precede the algo_engine cimport so the macro is defined before
# c_ex_profile_base.h's inline bodies are included. No-op on POSIX.
cdef extern from "test/exchange_profile/c_exchange_profile_linkage_shim.h":
    int _pyx_test_resolve_ex_profile(uintptr_t module_handle)

from algo_engine.exchange_profile cimport SessionDate, c_ex_profile_session_date_init, session_date_t

IF UNAME_SYSNAME == "Windows":
    # Resolve EX_PROFILE from the SAME copy of the .pyd that this process's
    # Python imports use: several copies may be loaded (installed + source
    # tree), so pass the exact module handle instead of matching by name.
    import ctypes
    import algo_engine.exchange_profile.c_exchange_profile as _c_ex  # ensure the .pyd is loaded
    if not _pyx_test_resolve_ex_profile(ctypes.CDLL(_c_ex.__file__)._handle):
        raise ImportError("EX_PROFILE not exported by c_exchange_profile — is the Windows build up to date?")


cpdef SessionDate pydate_to_cdate(pydate date):
    """Convert via the C interface: session_date_t init, then adopt the header."""
    cdef session_date_t * out = <session_date_t *> calloc(1, sizeof(session_date_t))
    c_ex_profile_session_date_init(out, PyDateTime_GET_YEAR(date), PyDateTime_GET_MONTH(date), PyDateTime_GET_DAY(date))
    return SessionDate.c_from_header(out, True)


cpdef int session_type_of(pydate date):
    """Resolve the session type through the shared C global (EX_PROFILE).

    This must observe the same EX_PROFILE the exchange_profile package mutates
    on activate()/deactivate() — a private copy of the global would keep
    returning the default profile's results.
    """
    cdef session_date_t out
    c_ex_profile_session_date_init(&out, PyDateTime_GET_YEAR(date), PyDateTime_GET_MONTH(date), PyDateTime_GET_DAY(date))
    return <int> out.stype


cpdef int session_date_sizeof():
    """C-side size of session_date_t, for the layout contract assertions."""
    return sizeof(session_date_t)


cpdef bytes session_date_raw_bytes(pydate date):
    """Raw memory image of a session_date_t initialized onto a 0xAA-poisoned stack buffer.

    With EX_PROFILE_SESSION_DATE_NO_PADDING every byte of the struct is a real
    field (no suffix padding), so no poison byte may survive and the image is
    fully determined by (year, month, day, stype).
    """
    cdef session_date_t out
    memset(<void*> &out, 0xAA, sizeof(out))
    c_ex_profile_session_date_init(&out, PyDateTime_GET_YEAR(date), PyDateTime_GET_MONTH(date), PyDateTime_GET_DAY(date))
    return bytes((<char*> &out)[:sizeof(out)])

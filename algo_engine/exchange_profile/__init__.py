import ctypes
import logging
import pathlib
import sysconfig

from ..base.telemetrics import LOGGER

LOGGER = LOGGER.getChild('ExchangeProfile')

# Promote the C extension's symbols (e.g. EX_PROFILE) to the dynamic linker's global scope before it is imported.
# CPython loads extensions with RTLD_LOCAL, which hides their symbols from other extensions that cimport this module's C interface and reference them at load time.
# Preloading the .so with RTLD_GLOBAL makes them resolvable by any extension imported after this package.
# No-op on Windows: PE DLLs have no global symbol scope — undefined symbols must be resolved at link time via import libraries.
_RTLD_GLOBAL = getattr(ctypes, 'RTLD_GLOBAL', 0)
if _RTLD_GLOBAL:
    try:
        ctypes.CDLL(
            str(pathlib.Path(__file__).parent / f"c_exchange_profile{sysconfig.get_config_var('EXT_SUFFIX')}"),
            mode=_RTLD_GLOBAL,
        )
    except Exception as _:
        pass  # graceful fallback: Python-level API remains fully functional


def set_logger(logger: logging.Logger):
    global LOGGER
    LOGGER = logger
    c_exchange_profile.LOGGER = logger


from .c_exchange_profile import (
    SessionDate, SessionDateEx, SessionDateRange, SessionTime, SessionTimeRange, SessionDateTime,
    CallAuction, SessionBreak, SessionType, SessionPhase, AuctionPhase,
    ExchangeProfile,
    local_utc_offset_seconds, unix_to_datetime
)

from .c_profile_dispatcher import PROFILE
from .c_profile_default import PROFILE_DEFAULT
from .c_profile_cn import PROFILE_CN

Profile = ExchangeProfile  # Alias for backward compatibility

__all__ = [
    'LOGGER',
    'SessionDate', 'SessionDateEx', 'SessionDateRange', 'SessionTime', 'SessionTimeRange', 'SessionDateTime',
    'CallAuction', 'SessionBreak', 'SessionType', 'SessionPhase', 'AuctionPhase',
    'ExchangeProfile', 'Profile', 'local_utc_offset_seconds', 'unix_to_datetime',

    'PROFILE',
    'PROFILE_CN',
    'PROFILE_DEFAULT'
]

import logging

from ..base.telemetrics import LOGGER

LOGGER = LOGGER.getChild('ExchangeProfile')


def set_logger(logger: logging.Logger):
    global LOGGER
    LOGGER = logger
    c_exchange_profile.LOGGER = logger


from .c_exchange_profile import (
    SessionDate, SessionDateRange, SessionTime, SessionTimeRange, SessionDateTime,
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
    'SessionDate', 'SessionDateRange', 'SessionTime', 'SessionTimeRange', 'SessionDateTime',
    'CallAuction', 'SessionBreak', 'SessionType', 'SessionPhase', 'AuctionPhase',
    'ExchangeProfile', 'Profile', 'local_utc_offset_seconds', 'unix_to_datetime',

    'PROFILE',
    'PROFILE_CN',
    'PROFILE_DEFAULT'
]

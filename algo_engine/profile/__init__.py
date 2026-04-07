import warnings

from .c_base import PROFILE, PROFILE_DEFAULT, ProfileDispatcher as Profile
from .c_cn import PROFILE_CN

warnings.warn(
    "`algo_engine.profile` is deprecated and will be removed in a future release. "
    "Use `algo_engine.exchange_profile` instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ['PROFILE', 'Profile', 'PROFILE_DEFAULT', 'PROFILE_CN']

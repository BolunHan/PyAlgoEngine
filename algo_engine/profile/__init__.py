import os

from .c_base import PROFILE, PROFILE_DEFAULT
from .c_cn import PROFILE_CN


def get_include() -> str:
    return os.path.dirname(__file__)


__all__ = ['get_include', 'PROFILE', 'PROFILE_DEFAULT', 'PROFILE_CN']

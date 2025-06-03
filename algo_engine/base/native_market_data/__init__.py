import os
from .market_utils_posix import *

if os.name == 'nt':
    from .market_utils_nt import *

from .trade_utils_native import *
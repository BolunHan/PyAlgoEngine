__version__ = "0.2.5"

import traceback

from . import Engine
from . import Strategies

Engine.LOGGER.info(f'AlgoEngine version {__version__}')

# import addon module
try:
    from . import EngineAddon

    Engine.LOGGER.info(f'AlgoEngine_Addons import successful, version {EngineAddon.__version__}')
except ImportError:
    Engine.LOGGER.warning(f'Install AlgoEngine_Addons to use Statistics module\n{traceback.format_exc()}')

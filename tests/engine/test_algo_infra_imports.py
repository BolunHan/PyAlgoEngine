import pathlib
import unittest

import algo_engine.engine.algo as algo_package

from algo_engine.engine import ALGO_ENGINE, ALGO_REGISTRY
from algo_engine.engine.algo import (
    Aggressive,
    AggressiveTimeout,
    AlgoTemplate,
    Passive,
    PassiveTimeout,
)


INFRA_PXD = pathlib.Path(__file__).resolve().parents[2] / 'algo_engine' / 'engine' / 'algo' / '__infra__.pxd'

ENGINE_ENTRIES = (
    'c_dma_algo_engine_new_order',
    'c_dma_algo_engine_free_orders',
    'c_dma_algo_engine_on_order',
    'c_dma_algo_engine_on_trade',
    'c_dma_algo_engine_on_md',
)


class TestAlgoInfraSurface(unittest.TestCase):
    """Contract: the algo package's C surface carries only the algo declarations.

    Expected behavior:
        - The aggregation file does not re-export the engine's C entry points.
        - Every algo wrapper stays importable from the package and the registry
          exposes exactly the four built-in kinds.

    Oracle: the aggregation file is read as text, and the exported names are
    compared against the package's declared surface.
    """

    def test_00_no_engine_entry_cimports(self) -> None:
        source = INFRA_PXD.read_text(encoding='utf-8')

        self.assertNotIn('..c_dma_algo', source)

        for symbol in ENGINE_ENTRIES:
            self.assertNotIn(symbol, source)

    def test_01_algo_surface_imports(self) -> None:
        self.assertTrue(issubclass(Passive, AlgoTemplate))
        self.assertTrue(issubclass(PassiveTimeout, Passive))
        self.assertTrue(issubclass(Aggressive, Passive))
        self.assertTrue(issubclass(AggressiveTimeout, Passive))
        self.assertEqual(set(ALGO_REGISTRY.names), {'passive', 'passive_timeout', 'aggressive', 'aggressive_timeout'})
        self.assertIs(ALGO_ENGINE.registry, ALGO_REGISTRY)

    def test_02_no_parent_reexports(self) -> None:
        for name in ('ALGO_ENGINE', 'ALGO_REGISTRY', 'AlgoEngine', 'AlgoLimitMode', 'AlgoRegistry'):
            self.assertFalse(hasattr(algo_package, name), f'{name} must stay out of the algo package')

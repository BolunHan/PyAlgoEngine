import pathlib
import py_compile
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]

CONSUMERS = (
    'algo_engine/strategy/strategy_engine.py',
    'algo_engine/backtest/__main__.py',
    'algo_engine/apps/backtest/tester.py',
)

RETIRED_SOURCES = (
    'algo_engine/engine/c_dma.pyx',
    'algo_engine/engine/c_dma.pyi',
    'algo_engine/strategy/__init__.py',
)

COMPILED = (
    'algo_engine/engine/__init__.py',
    'algo_engine/strategy/__init__.py',
    'algo_engine/strategy/strategy_engine.py',
    'algo_engine/backtest/__main__.py',
    'algo_engine/apps/backtest/tester.py',
    'tests/engine/test_c_dma.py',
)


class TestBalanceRetired(unittest.TestCase):
    """Contract: the Balance registry facade is gone; only the data-host balance remains."""

    def test_00_no_facade_in_the_retired_modules(self) -> None:
        for relative in RETIRED_SOURCES:
            text = (ROOT / relative).read_text(encoding='utf-8')

            self.assertNotIn('class Balance', text, relative)

    def test_01_consumers_route_through_the_algo_engine(self) -> None:
        for relative in CONSUMERS:
            text = (ROOT / relative).read_text(encoding='utf-8')

            self.assertIn('algo_engine.on_order', text, relative)
            self.assertIn('algo_engine.on_trade', text, relative)
            self.assertNotIn('.balance', text, relative)

    def test_02_the_facade_is_not_exported(self) -> None:
        import algo_engine.engine as engine_package
        import algo_engine.engine.c_balance as c_balance_module
        import algo_engine.engine.c_dma as c_dma_module
        import algo_engine.strategy as strategy_package

        self.assertFalse(hasattr(c_dma_module, 'Balance'))
        self.assertFalse(hasattr(strategy_package, 'BALANCE'))
        self.assertIsInstance(c_dma_module.BALANCE, c_balance_module.Balance)

        # The surviving Balance is the c_balance data host, not the facade.
        self.assertIs(engine_package.Balance, c_balance_module.Balance)

    def test_03_the_edited_sources_compile(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            for relative in COMPILED:
                py_compile.compile(
                    str(ROOT / relative),
                    cfile=str(pathlib.Path(tmp_dir) / f'{pathlib.Path(relative).stem}.pyc'),
                    doraise=True,
                )


if __name__ == '__main__':
    unittest.main()

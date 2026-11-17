import unittest

from algo_engine.base import TransactionSide
from algo_engine.engine import (
    ALGO_REGISTRY,
    Aggressive,
    AggressiveTimeout,
    AlgoStatus,
    Passive,
    PassiveTimeout,
    PositionManagementService,
)


TICKER = 'AAPL'
SIDE_LONG = TransactionSide.SIDE_LONG_OPEN


class TestAlgoFactoryDispatch(unittest.TestCase):
    """Contract: wrapping a tracked record resolves to its registered wrapper class.

    Expected behavior:
        - A record built by a kind wraps as that kind's concrete class.
        - The wrapped view is borrowed; releasing it never frees the record the
          owner still holds.
        - The registry name surface (`cast`, `in`, `names`) is unchanged.

    Oracle: each wrapped value is compared, by exact type, against the class
    that built the record; the borrowed view is released with the owner kept
    alive, then the owner is read back.
    """

    def test_00_dispatch_returns_the_registered_class(self) -> None:
        position = PositionManagementService()

        owners = [
            Passive(TICKER, SIDE_LONG, 100.),
            PassiveTimeout(TICKER, SIDE_LONG, 100., timeout=1.),
            Aggressive(TICKER, SIDE_LONG, 100.),
            AggressiveTimeout(TICKER, SIDE_LONG, 100., timeout=1.),
        ]

        for owner in owners:
            position.add_algo(owner)

        wrapped = {algo.algo_id: algo for algo in position.algos.values()}

        for owner in owners:
            self.assertIs(type(wrapped[owner.algo_id]), type(owner))

    def test_01_wrapped_view_is_borrowed(self) -> None:
        position = PositionManagementService()
        owner = Passive(TICKER, SIDE_LONG, 100.)
        position.add_algo(owner)

        wrapped = next(iter(position.algos.values()))
        self.assertIsNot(wrapped, owner)

        del wrapped

        self.assertEqual(owner.status, AlgoStatus.Idle)
        self.assertIsNotNone(owner.algo_id)

    def test_02_registry_surface_is_unchanged(self) -> None:
        self.assertIs(ALGO_REGISTRY.cast('passive'), Passive)
        self.assertIn('passive', ALGO_REGISTRY)
        self.assertEqual(
            set(ALGO_REGISTRY.names),
            {'passive', 'passive_timeout', 'aggressive', 'aggressive_timeout'},
        )
        self.assertEqual(
            ALGO_REGISTRY.reversed_registry,
            {
                'Passive': 'passive',
                'PassiveTimeout': 'passive_timeout',
                'Aggressive': 'aggressive',
                'AggressiveTimeout': 'aggressive_timeout',
            },
        )

    def test_03_add_algo_requires_a_class(self) -> None:
        with self.assertRaises(TypeError):
            ALGO_REGISTRY.add_algo('passive', 42)

        with self.assertRaises(TypeError):
            ALGO_REGISTRY.add_algo('passive', int)

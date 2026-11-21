import os
import pathlib
import random
import unittest
from collections import Counter
from datetime import date

from algo_engine.base.c_market_data.c_candlestick import DailyBar
from algo_engine.base.c_market_data.c_market_data_buffer import MarketDataBuffer
from md_gen import random_market_data


def shm_used_bytes():
    stat = os.statvfs('/dev/shm')
    return (stat.f_blocks - stat.f_bfree) * stat.f_frsize


class MarketDataBufferCacheTests(unittest.TestCase):
    def setUp(self):
        random.seed(0)

    def _generate_market_data(self, count: int):
        return [random_market_data() for _ in range(count)]

    def _signature(self, market_data):
        return market_data.to_bytes()

    def test_cache_context_flushes_entries(self):
        buffer = MarketDataBuffer(2, 256)
        samples = self._generate_market_data(6)

        with buffer.cache() as cache:
            self.assertIs(cache.parent, buffer)
            for idx, market_data in enumerate(samples, start=1):
                cache.put(market_data)
                self.assertEqual(len(cache), idx)

        self.assertEqual(len(buffer), len(samples))
        flushed = [buffer[i] for i in range(len(buffer))]
        self.assertEqual(
            [self._signature(md) for md in samples],
            [self._signature(md) for md in flushed],
        )

    def test_cache_flushes_even_on_exception(self):
        buffer = MarketDataBuffer(1, 128)
        samples = self._generate_market_data(3)

        with self.assertRaises(RuntimeError):
            with buffer.cache() as cache:
                cache.put(samples[0])
                raise RuntimeError("forced failure to test flush")

        self.assertEqual(len(buffer), 1)

    def test_buffer_auto_extends_capacity(self):
        initial_ptr = 2
        initial_data = 256
        buffer = MarketDataBuffer(initial_ptr, initial_data)
        samples = self._generate_market_data(1_000_000)

        for market_data in samples:
            buffer.put(market_data)

        self.assertEqual(len(buffer), len(samples))
        self.assertGreaterEqual(buffer.ptr_capacity, initial_ptr * 2)
        self.assertGreaterEqual(buffer.data_capacity, initial_data * 2)

    def test_cache_auto_extends_capacity(self):
        buffer = MarketDataBuffer(1, 64)
        samples = self._generate_market_data(1_000_000)

        with buffer.cache() as cache:
            for market_data in samples:
                cache.put(market_data)

            self.assertEqual(len(cache), len(samples))
            self.assertGreaterEqual(cache.capacity, 4)

        assert buffer.__len__() == 1_000_000

    def test_sorting_and_index_access(self):
        buffer = MarketDataBuffer(2, 256)
        samples = self._generate_market_data(1_000)

        # sample should already be in random order
        last_md = None
        for md in samples:
            if last_md is None:
                last_md = md
                continue

            if last_md.timestamp > md.timestamp:
                break
        else:
            samples = samples[::-1]

        with buffer.cache() as cache:
            for md in samples:
                cache.put(md)

        got = list(buffer)

        # The C buffer sorts with qsort, which is unstable on MSVC: entries
        # sharing a timestamp may come out in any order. Verify the contract
        # without assuming stability — order must be sorted, content must be
        # the same multiset.
        expected_ts = sorted(md.timestamp for md in samples)
        self.assertEqual([md.timestamp for md in got], expected_ts)
        self.assertEqual(
            Counter(self._signature(md) for md in got),
            Counter(self._signature(md) for md in samples),
        )

    def test_daily_bar_sorting_uses_date(self):
        """Date-only records sort by dt.date, not by their zero unix_ts."""
        buffer = MarketDataBuffer(2, 256)
        market_dates = [
            date(2024, 3, 15),
            date(2023, 12, 31),
            date(2024, 1, 2),
        ]
        bars = [
            DailyBar(
                ticker='DAY',
                market_date=market_date,
                high_price=1.0,
                low_price=1.0,
                open_price=1.0,
                close_price=1.0,
            )
            for market_date in market_dates
        ]

        with buffer.cache() as cache:
            for bar in bars:
                cache.put(bar)

        got = list(buffer)
        self.assertEqual([bar.market_date for bar in got], sorted(market_dates))
        self.assertEqual([bar.timestamp for bar in got], [0.0] * len(market_dates))

    def test_serialize_and_deserialize_roundtrip(self):
        buffer = MarketDataBuffer(8, 1024)
        samples = self._generate_market_data(1_00)

        with buffer.cache() as cache:
            for md in samples:
                cache.put(md)

        serialized = buffer.to_bytes()
        clone = MarketDataBuffer.from_bytes(serialized)

        self.assertEqual(len(clone), len(samples))
        expected = [self._signature(md) for md in samples]
        got = [self._signature(clone[i]) for i in range(len(clone))]
        self.assertEqual(expected, got)

    def test_access_does_not_leak_shared_memory(self):
        """Each access deserializes a copy; the returned view must own it.

        A borrowed view (``owner=False``) leaves one allocator block behind
        per access -- a replayed session (millions of entries) leaked ~1.4 GiB
        of /dev/shm per day and SIGBUSed the multi-day evaluation once the
        container's shared-memory cap was reached.
        """
        if not pathlib.Path('/dev/shm').is_dir():
            self.skipTest('/dev/shm is not available')

        buffer = MarketDataBuffer(4, 256)
        samples = self._generate_market_data(100_000)
        for md in samples:
            buffer.put(md)

        before = shm_used_bytes()
        count = 0
        for md in buffer:
            _ = md.ticker
            count += 1
        for idx in range(len(buffer)):
            _ = buffer[idx].ticker
            count += 1
        after = shm_used_bytes()

        self.assertEqual(count, 2 * len(samples))
        self.assertLess(
            after - before,
            4 << 20,
            f'{count:,} buffer accesses leaked {(after - before) / 2**20:.1f} MiB of /dev/shm',
        )


if __name__ == "__main__":
    unittest.main()

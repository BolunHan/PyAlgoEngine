import random
import time
import unittest

from algo_engine.base.c_market_data_ng.c_market_data_buffer import (
    MarketDataBuffer,
    MarketDataBufferCache,
)
from md_gen import random_market_data


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

        expected = sorted(samples, key=lambda md: md.timestamp)
        got = list(buffer)

        for md_0, md_1 in zip(expected, got):
            self.assertEqual(self._signature(md_0), self._signature(md_1))

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


if __name__ == "__main__":
    unittest.main()

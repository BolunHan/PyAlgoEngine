import random
import time
import unittest

from algo_engine.base.c_market_data_ng.c_market_data_buffer import BufferFull, BufferEmpty, PipeTimeoutError, MarketDataRingBuffer
from md_gen import random_market_data


class MarketDataRingBufferTests(unittest.TestCase):
    def setUp(self):
        random.seed(0)

    def _generate_market_data(self, count: int):
        return [random_market_data() for _ in range(count)]

    def _signature(self, market_data):
        return market_data.to_bytes()

    def test_put_get_round_trip(self):
        buffer = MarketDataRingBuffer(64, 1 << 18)
        samples = self._generate_market_data(16)

        for market_data in samples:
            buffer.put(market_data)

        self.assertEqual(len(buffer), len(samples))
        restored = [buffer[i] for i in range(len(buffer))]
        self.assertEqual(
            [self._signature(md) for md in samples],
            [self._signature(md) for md in restored],
        )

    def test_ring_wrap_keeps_recent_entries(self):
        got = []
        buffer = MarketDataRingBuffer(4, 1 << 18)
        size = 1000
        samples = self._generate_market_data(size)
        enabled = True

        def worker():
            while enabled:
                try:
                    md = buffer.listen(block=False)
                    got.append(md)
                except (BufferEmpty, PipeTimeoutError):
                    time.sleep(0.0)
                    continue

        import threading
        t = threading.Thread(target=worker)
        t.start()

        for i , market_data in enumerate(samples):
            while True:
                try:
                    print(f'[RingBuffer][put] sending {i} / {size} md...')
                    buffer.put(market_data, block=False)
                    break
                except BufferFull:
                    time.sleep(0.0)
                    continue

        time.sleep(1)
        enabled = False

        self.assertEqual(size, len(got))
        self.assertEqual(len(buffer), 0)
        expected = [self._signature(md) for md in samples]
        restored = [self._signature(md) for md in got]
        self.assertEqual(expected, restored)

    def test_negative_index_and_iteration_consistency(self):
        buffer = MarketDataRingBuffer(128, 1 << 20)
        samples = self._generate_market_data(50)

        for offset, market_data in enumerate(samples):
            buffer.put(market_data)

        self.assertEqual(self._signature(buffer[-1]), self._signature(samples[-1]))
        self.assertEqual(self._signature(buffer[-len(samples)]), self._signature(samples[0]))

        iter_once = [self._signature(md) for md in buffer]
        iter_twice = [self._signature(md) for md in buffer]
        direct = [self._signature(buffer[i]) for i in range(len(buffer))]

        self.assertEqual(iter_once, iter_twice)
        self.assertEqual(iter_once, direct)

    def test_listen_non_blocking_behavior(self):
        buffer = MarketDataRingBuffer(16, 1 << 18)

        with self.assertRaises(BufferEmpty):
            buffer.listen(block=False)

        samples = self._generate_market_data(6)
        for market_data in samples:
            buffer.put(market_data)

        size_before = len(buffer)
        first_signature = self._signature(buffer[0])
        listened = buffer.listen(block=False)
        self.assertEqual(self._signature(listened), first_signature)
        self.assertEqual(len(buffer), size_before - 1)

    def test_is_empty_and_drain_via_listen(self):
        buffer = MarketDataRingBuffer(16, 1 << 18)
        self.assertTrue(buffer.is_empty)

        samples = self._generate_market_data(8)
        for idx, market_data in enumerate(samples, start=1):
            buffer.put(market_data)
            self.assertEqual(len(buffer), idx)
            self.assertFalse(buffer.is_empty)

        while len(buffer):
            buffer.listen(block=False, timeout=0)

        self.assertTrue(buffer.is_empty)
        with self.assertRaises(BufferEmpty):
            buffer.listen(block=False, timeout=0)


if __name__ == "__main__":
    unittest.main()

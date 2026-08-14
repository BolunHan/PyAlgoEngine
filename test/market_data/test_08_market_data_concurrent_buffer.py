import os
import random
import time
import unittest

from algo_engine.base.c_market_data.c_market_data_buffer import BufferEmpty, BufferFull, MarketDataConcurrentBuffer, PipeTimeoutError
from md_gen import random_market_data

_GLOBAL_CONCURRENT_BUFFER = None


def _listen_worker(buffer, data, size, worker_id):
    """Module-level worker so multiprocessing spawn can pickle it on Windows."""
    for i in range(size):
        md = buffer.listen(worker_id=worker_id, block=True)
        if repr(md) != repr(data[i]):
            raise AssertionError(f"worker {worker_id}: sample {i} mismatch")


class MarketDataConcurrentBufferTests(unittest.TestCase):
    def setUp(self):
        random.seed(0)

    def _signature(self, market_data):
        return repr(market_data)

    def test_single_process(self):
        size = 1000
        data = [random_market_data() for _ in range(size)]
        buffer = MarketDataConcurrentBuffer(n_workers=1, capacity=16)
        collected = []

        from threading import Thread

        def worker():
            for i in range(size):
                while True:
                    try:
                        md = buffer.listen(worker_id=0, block=False)
                        collected.append(md)
                        print(f'[ConcurrentBuffer][listen] got {i} / {size} md...')
                        break
                    except (BufferEmpty, PipeTimeoutError) as e:
                        time.sleep(0.0)
                        continue
                if i == size - 1:
                    break

        t = Thread(target=worker)
        t.start()

        for i, md in enumerate(data):
            while True:
                try:
                    print(f'[ConcurrentBuffer][put] sending {i} / {size} md...')
                    buffer.put(md, block=False)
                    break
                except (BufferFull, PipeTimeoutError):
                    time.sleep(0.0)
                    continue

        time.sleep(1)
        t.join()

        self.assertEqual(
            [self._signature(md) for md in data],
            [self._signature(md) for md in collected],
        )

        self.assertEqual(
            [self._signature(md) for md in collected],
            [self._signature(md) for md in data],
        )

    def test_multi_thread(self):
        size = 1000
        n_workers = 4
        data = [random_market_data() for _ in range(size)]
        buffer = MarketDataConcurrentBuffer(n_workers=n_workers, capacity=16)
        collected = [[] for _ in range(n_workers)]
        workers = []

        from threading import Thread

        def worker(worker_id):
            for i in range(size):
                while True:
                    try:
                        md = buffer.listen(worker_id=worker_id, block=False)
                        collected[worker_id].append(md)
                        print(f'[ConcurrentBuffer][listen] got {i} / {size} md...')
                        break
                    except (BufferEmpty, PipeTimeoutError) as e:
                        time.sleep(0.0)
                        continue
                if i == size - 1:
                    break

        for w_id in range(n_workers):
            t = Thread(target=worker, args=(w_id,))
            workers.append(t)
            t.start()

        for i, md in enumerate(data):
            while True:
                try:
                    print(f'[ConcurrentBuffer][put] sending {i} / {size} md...')
                    buffer.put(md, block=False)
                    break
                except (BufferFull, PipeTimeoutError):
                    time.sleep(0.0)
                    continue

        time.sleep(1)
        for w_id in range(n_workers):
            t = workers[w_id]
            t.join()

            self.assertEqual(
                [self._signature(md) for md in data],
                [self._signature(md) for md in collected[w_id]],
            )

            self.assertEqual(
                [self._signature(md) for md in collected[w_id]],
                [self._signature(md) for md in data],
            )

    @unittest.skipIf(
        os.name == "nt",
        "Windows spawn cannot pickle MarketDataConcurrentBuffer (shm-backed cdef "
        "class without __reduce__/attach API); the multi-process path is covered "
        "on POSIX via fork",
    )
    def test_multi_processes(self):
        size = 1000
        n_workers = 4
        data = [random_market_data() for _ in range(size)]
        buffer = MarketDataConcurrentBuffer(n_workers=n_workers, capacity=16)
        workers = []

        from multiprocessing import Process

        for w_id in range(n_workers):
            t = Process(target=_listen_worker, args=(buffer, data, size, w_id))
            workers.append(t)
            t.start()

        for i, md in enumerate(data):
            print(f'[ConcurrentBuffer][put] sending {i} / {size} md...')
            buffer.put(md, block=True)

        time.sleep(1)
        for w_id, t in enumerate(workers):
            t.join()
            self.assertEqual(t.exitcode, 0, f"worker {w_id} failed with exitcode {t.exitcode}")


if __name__ == "__main__":
    unittest.main()

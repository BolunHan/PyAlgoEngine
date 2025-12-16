import time
import unittest
import uuid

from algo_engine.base import c_shm_allocator as c_shm
from algo_engine.base.c_shm_allocator import ALLOCATOR, SharedMemoryBlock


class TestCShmAllocator(unittest.TestCase):
    def test_00_constants_and_allocator_exists(self):
        self.assertTrue(hasattr(c_shm, 'SHM_ALLOCATOR_PREFIX'))
        self.assertTrue(hasattr(c_shm, 'SHM_PAGE_PREFIX'))
        self.assertTrue(hasattr(c_shm, 'SHM_ALLOCATOR_DEFAULT_REGION_SIZE'))

        self.assertIsInstance(c_shm.SHM_ALLOCATOR_PREFIX, str)
        self.assertIsInstance(c_shm.SHM_PAGE_PREFIX, str)
        self.assertIsInstance(c_shm.SHM_ALLOCATOR_DEFAULT_REGION_SIZE, int)
        self.assertGreater(c_shm.SHM_ALLOCATOR_DEFAULT_REGION_SIZE, 0)

        self.assertTrue(hasattr(c_shm, 'ALLOCATOR'))
        self.assertIsNotNone(c_shm.ALLOCATOR)
        # ensure repr doesn't crash
        _ = repr(c_shm.ALLOCATOR)

    def test_01_calloc_request_free_and_free_list(self):
        # calloc
        b1 = ALLOCATOR.calloc(1024)
        addr_1 = b1.address
        self.assertIsNotNone(b1)
        self.assertTrue(hasattr(b1, 'size'))
        self.assertGreaterEqual(b1.size, 1024)

        # request
        b2 = ALLOCATOR.request(512)
        addr_2 = b2.address
        self.assertIsNotNone(b2)
        self.assertGreaterEqual(b2.size, 512)
        self.assertIsNot(b1, b2)

        # free b1
        ALLOCATOR.free(b2)

        # after freeing, the free_list should contain at least one block with sufficient capacity
        self.assertIn(addr_2, [_.address for _ in ALLOCATOR.free_list()])

        # cleanup: try to reclaim
        occupied = ALLOCATOR.active_page.occupied
        ALLOCATOR.reclaim()
        self.assertNotIn(addr_2, [_.address for _ in ALLOCATOR.free_list()])
        self.assertLessEqual(ALLOCATOR.active_page.occupied, occupied)

        self.assertIsInstance(repr(b1), str)
        b3 = ALLOCATOR.request(256)
        self.assertEqual(b3.address, addr_2)

        ALLOCATOR.free(b1)
        occupied = ALLOCATOR.active_page.occupied
        ALLOCATOR.reclaim()
        self.assertEqual(ALLOCATOR.active_page.occupied, occupied)

    def test_02_extend_page_and_allocated_iteration(self):
        page = ALLOCATOR.extend(1 << 16)
        self.assertIsNotNone(page)
        self.assertTrue(hasattr(page, 'capacity'))
        self.assertGreaterEqual(page.capacity, (1 << 16))
        self.assertIsInstance(page.name, str)

        # allocate some blocks on this allocator
        blocks = []
        for i in range(4):
            print('==== Allocation iteration', i, '====')
            print('[DEBUG] allocated:', [b for _page in ALLOCATOR.pages() for b in _page.allocated()])
            print('[DEBUG] free_list:', [b for b in ALLOCATOR.free_list()])
            print('[DEBUG] requesting 128B buffer:')
            b = ALLOCATOR.request(128)
            print('[DEBUG] allocated:', [b for _page in ALLOCATOR.pages() for b in _page.allocated()])
            print('[DEBUG] free_list:', [b for b in ALLOCATOR.free_list()])
            print('==== Allocation iteration', i, ' Complete ====')
            blocks.append(b)

        allocated = [b for _page in ALLOCATOR.pages() for b in _page.allocated()]
        # allocated should contain at least the 4 we requested (global allocator may have other blocks)
        self.assertGreaterEqual(len([b for b in allocated if b.size >= 128]), 4)

        # test block properties
        b = blocks[0]
        self.assertTrue(hasattr(b, 'size'))
        self.assertTrue(hasattr(b, 'capacity'))
        # test next_allocated/next_free access doesn't crash
        _ = getattr(b, 'next_allocated', None)
        _ = getattr(b, 'next_free', None)

        # cleanup
        for bb in blocks:
            ALLOCATOR.free(bb)
        ALLOCATOR.reclaim()

    def test_03_dangling_detection_and_cleanup(self):
        # Step 1: use os to create a new python process, import ALLOCATOR and exit
        import subprocess
        script = (
            "import os\n"
            "from algo_engine.base import c_shm_allocator as c_shm\n"
            "allocator = c_shm.ALLOCATOR\n"
            "print(allocator)\n"
            "print(os.getpid())\n"
        )

        # run the script in python
        process = subprocess.Popen(['python3', '-c', script], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        self.assertEqual(process.returncode, 0, f'Child process failed: {stderr.decode()}')

        # Step 2: scan for dangling
        subprocess_pid = int(stdout.decode().split('\n')[-2])
        for shm_name in ALLOCATOR.dangling():
            pid = ALLOCATOR.get_pid(shm_name)
            self.assertNotEqual(pid, subprocess_pid)

        script = (
            "import os\n"
            "from algo_engine.base import c_shm_allocator as c_shm\n"
            "allocator = c_shm.ALLOCATOR\n"
            "print(allocator)\n"
            "print(os.getpid())\n"
            "os._exit(0)\n"
        )

        # run the script in python
        process = subprocess.Popen(['python3', '-c', script], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        self.assertEqual(process.returncode, 0, f'Child process failed: {stderr.decode()}')
        subprocess_pid = int(stdout.decode().split('\n')[-2])
        found = False

        for shm_name in ALLOCATOR.dangling():
            pid = ALLOCATOR.get_pid(shm_name)
            if pid == subprocess_pid:
                found = True
                break

        self.assertTrue(found, 'Dangling allocator from subprocess not found.')

        ALLOCATOR.cleanup_dangling()
        for shm_name in ALLOCATOR.dangling():
            pid = ALLOCATOR.get_pid(shm_name)
            self.assertNotEqual(pid, subprocess_pid)

    def test_04_multiprocessing_access(self):
        # Step 1: measure the addr diff for 2 calloc buffer
        ALLOCATOR.extend(1 << 16)
        b1 = ALLOCATOR.calloc(1024)
        b2 = ALLOCATOR.calloc(1024)
        addr_diff = int(b2.address, 16) - int(b1.address, 16)
        payload = uuid.uuid4().bytes

        # Step 2: spawn a new process to access the same allocator and verify addr diff
        from multiprocessing import Process
        def worker():
            time.sleep(1)
            addr_3 = int(b2.address, 16) + addr_diff
            b3_mirrored = SharedMemoryBlock(addr_3, False)
            # there is an overhead, needs to be calculated
            overhead = int(b3_mirrored.address, 16) - addr_3
            b3_mirrored = SharedMemoryBlock(addr_3 - overhead, False)
            if bytes(b3_mirrored.buffer[:16]) != payload:
                raise ValueError('Payload mismatch in multiprocessing access test.')
            print('sub-process access payload by reference successful.')

        process = Process(target=worker)
        process.start()

        # Init a new buffer and write payload
        b3 = ALLOCATOR.calloc(1024)
        b3.buffer[:16] = payload
        process.join()
        self.assertEqual(process.exitcode, 0, 'Multiprocessing access test failed.')

    def test_05_auto_extend(self):
        ALLOCATOR.extend(4 * 1024)
        current_page = ALLOCATOR.active_page
        buf_1 = ALLOCATOR.calloc(2048)
        # buf_1 should be allocated in current_page
        self.assertEqual(current_page.name, ALLOCATOR.active_page.name)
        available_buffer = current_page.capacity - current_page.occupied
        buf_2 = ALLOCATOR.calloc(available_buffer - 2)
        # buf_2 should trigger auto-extend, as the overhead is far larger than 2 bytes
        self.assertNotEqual(current_page.name, ALLOCATOR.active_page.name)


if __name__ == '__main__':
    unittest.main()

import unittest

from algo_engine.base import c_heap_allocator as c_heap
from algo_engine.base.c_heap_allocator import ALLOCATOR


class TestCHeapAllocator(unittest.TestCase):
    def setUp(self):
        self.allocator = ALLOCATOR
        if self.allocator is None:
            self.fail('Global heap allocator is not initialized')

    def test_constants_and_allocator_exists(self):
        self.assertTrue(hasattr(c_heap, 'DEFAULT_AUTOPAGE_CAPACITY'))
        self.assertTrue(hasattr(c_heap, 'MAX_AUTOPAGE_CAPACITY'))
        self.assertTrue(hasattr(c_heap, 'DEFAULT_AUTOPAGE_ALIGNMENT'))

        self.assertIsInstance(c_heap.DEFAULT_AUTOPAGE_CAPACITY, int)
        self.assertIsInstance(c_heap.MAX_AUTOPAGE_CAPACITY, int)
        self.assertIsInstance(c_heap.DEFAULT_AUTOPAGE_ALIGNMENT, int)

        self.assertGreater(c_heap.DEFAULT_AUTOPAGE_CAPACITY, 0)
        self.assertGreaterEqual(c_heap.MAX_AUTOPAGE_CAPACITY, c_heap.DEFAULT_AUTOPAGE_CAPACITY)
        self.assertGreater(c_heap.DEFAULT_AUTOPAGE_ALIGNMENT, 0)

        self.assertTrue(hasattr(c_heap, 'ALLOCATOR'))
        self.assertIsNotNone(self.allocator)
        _ = repr(self.allocator)

    def test_calloc_request_free_and_free_list(self):
        block_a = self.allocator.calloc(1024)
        self.assertGreaterEqual(block_a.size, 1024)
        self.assertIsNotNone(block_a.address)
        self.assertEqual(bytes(block_a.buffer[:16]), b"\x00" * 16)

        block_b = self.allocator.request(512)
        addr_b = block_b.address
        self.assertGreaterEqual(block_b.size, 512)
        self.assertIsNotNone(block_b.address)
        self.assertNotEqual(block_a.address, block_b.address)

        self.allocator.free(block_b)
        free_addresses = [blk.address for blk in self.allocator.free_list()]
        self.assertIn(addr_b, free_addresses)

        occupied_before = self.allocator.active_page.occupied
        self.allocator.reclaim()
        free_addresses_after = [blk.address for blk in self.allocator.free_list()]
        self.assertNotIn(addr_b, free_addresses_after)
        self.assertLessEqual(self.allocator.active_page.occupied, occupied_before)

        block_c = self.allocator.request(256, scan_all_pages=False)
        self.assertEqual(block_c.address, addr_b)

        self.allocator.free(block_a)
        self.allocator.free(block_c)
        self.allocator.reclaim()

    def test_extend_page_and_iteration_helpers(self):
        target_capacity = 1 << 15
        page = self.allocator.extend(target_capacity)
        self.assertGreaterEqual(page.capacity, target_capacity)
        self.assertIsNotNone(page.capacity)

        blocks = []
        for _ in range(3):
            blocks.append(self.allocator.request(256))

        # pages() iterator should return newest page first
        pages = list(self.allocator.pages())
        self.assertGreaterEqual(len(pages), 1)
        self.assertIsNotNone(pages[0].capacity)

        allocated_blocks = [blk for pg in self.allocator.pages() for blk in pg.allocated()]
        self.assertGreaterEqual(len(allocated_blocks), len(blocks))

        # Access optional attributes to ensure they do not raise
        sample_block = blocks[0]
        _ = getattr(sample_block, 'next_allocated', None)
        _ = getattr(sample_block, 'next_free', None)
        _ = getattr(sample_block, 'parent_page', None)

        for blk in blocks:
            self.allocator.free(blk)
        self.allocator.reclaim()


if __name__ == '__main__':
    unittest.main()


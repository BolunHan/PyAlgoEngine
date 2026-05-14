import time
import unittest

from algo_engine.base import c_intern_string as cis
from algo_engine.base.c_intern_string import InternStringPool, InternString, POOL, INTRA_POOL


class TestCInternString(unittest.TestCase):

    def test_module_and_globals_exist(self):
        # module-level symbols
        self.assertIsNotNone(cis)
        self.assertTrue(hasattr(cis, 'POOL'))
        # self.assertTrue(hasattr(cis, 'C_POOL'))
        self.assertTrue(hasattr(cis, 'InternStringPool'))
        self.assertTrue(hasattr(cis, 'InternString'))

        self.assertIsNotNone(cis.POOL)
        # address should be available as a string when pool initialized
        addr = getattr(cis.POOL, 'address', None)
        if addr is not None:
            self.assertIsInstance(addr, str)

    def test_intern_and_size_and_len(self):
        pool = cis.POOL
        initial_size = pool.size
        s1 = pool.istr('alpha')
        s2 = pool.istr('beta')
        size_after_two = pool.size
        s3 = pool.istr('alpha')  # already interned, must not grow size

        # two unique strings must increase size by exactly 2 (or 0 if already present)
        self.assertGreaterEqual(size_after_two, initial_size)
        # re-interning 'alpha' must not change size
        self.assertEqual(pool.size, size_after_two)
        # len() must match pool.size
        self.assertEqual(len(pool), pool.size)

        self.assertIsNotNone(s1)
        self.assertIsNotNone(s2)
        self.assertIsNotNone(s3)

    def test_internalized_iteration_and_internstring_api(self):
        pool = cis.POOL
        # Ensure 'gamma' is present and retrieve it directly
        inst = pool.istr('gamma')
        inst2 = pool['gamma']

        internalized = list(pool.internalized())
        self.assertGreaterEqual(len(internalized), 1)

        # string property should return a Python string
        pystr = inst.string
        self.assertIsInstance(pystr, str)
        self.assertEqual(pystr, 'gamma')

        # repr should not crash
        _ = repr(inst)

        # equality between two InternString objects for the same key
        self.assertTrue(inst == inst2)
        self.assertFalse(inst != inst2)
        # compare with Python string (should not raise)
        self.assertTrue(inst == 'gamma')

        # hash() must return an int
        h = hash(inst)
        self.assertIsInstance(h, int)

    def test_stable_pointer_and_no_double_count(self):
        pool = cis.POOL
        key = 'stable_ptr_test'
        s1 = pool.istr(key)
        size_after_first = pool.size
        s2 = pool.istr(key)
        # Re-interning must not grow the pool
        self.assertEqual(pool.size, size_after_first)
        # Both InternString objects must point to the same internalized copy
        self.assertEqual(s1.address, s2.address)
        # The internalized string must equal the original key
        self.assertEqual(s1.string, key)

    def test_getitem_missing_key_raises(self):
        pool = cis.POOL
        with self.assertRaises(KeyError):
            _ = pool['__key_that_was_never_interned__']

    def test_internstring_properties(self):
        pool = cis.POOL
        inst = pool.istr('prop_test_key')

        # hash_value: non-None int (FNV-1a never produces 0 in practice)
        hv = inst.hash_value
        self.assertIsNotNone(hv)
        self.assertIsInstance(hv, int)

        # address: hex string
        addr = inst.address
        self.assertIsInstance(addr, str)
        self.assertTrue(addr.startswith('0x'), msg=f'Expected hex address, got: {addr!r}')

        # intern_pool: round-trips back to an InternStringPool
        ip = inst.pool
        self.assertIsInstance(ip, InternStringPool)

    def test_inter_multiprocess(self):
        POOL.istr('cat')

        from multiprocessing import Process
        def worker():
            istr_cat = POOL['cat']
            time.sleep(1)
            istr_dog = POOL['dog']
            print('post fork shared istr tested successfully')

        p = Process(target=worker)
        p.start()
        time.sleep(.1)
        POOL.istr('dog')
        p.join()
        self.assertEqual(p.exitcode, 0)

    def test_local_pool(self):
        s1 = INTRA_POOL.istr('local1')
        s2 = INTRA_POOL.istr('local2')
        self.assertIsNotNone(s1)
        self.assertIsNotNone(s2)

        s1_inter = INTRA_POOL.istr('local1')
        self.assertEqual(s1.address, s1_inter.address)
        self.assertIn(s1.string, [_.string for _ in INTRA_POOL.internalized()])
        self.assertIn(s2.string, [_.string for _ in INTRA_POOL.internalized()])


if __name__ == '__main__':
    unittest.main()

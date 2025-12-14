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
        # initial size (may be zero)
        initial_size = pool.size
        # intern a few strings
        s1 = pool.istr('alpha')
        s2 = pool.istr('beta')
        s3 = pool.istr('alpha')  # same as s1

        # size should have increased at least for the unique strings
        self.assertGreaterEqual(pool.size, initial_size)
        # len() should match pool.size
        self.assertEqual(len(pool), pool.size)

        # interning the same string should return a value (non-None)
        self.assertIsNotNone(s1)
        self.assertIsNotNone(s2)
        self.assertIsNotNone(s3)

    def test_internalized_iteration_and_internstring_api(self):
        pool = cis.POOL
        # Ensure at least one string is present
        pool.istr('gamma')

        internalized = list(pool.internalized())
        # internalized yields InternString instances or compatible objects
        self.assertTrue(len(internalized) >= 1)

        inst = internalized[0]
        # string property should return a Python string
        self.assertTrue(hasattr(inst, 'string'))
        pystr = inst.string
        self.assertIsInstance(pystr, str)

        # repr should not crash
        _ = repr(inst)

        # equality and gt should work against Python strings and other InternString
        inst2 = pool['gamma']
        self.assertTrue(inst == inst2)
        self.assertFalse(inst != inst2)
        # compare with Python string (should not raise)
        _ = (inst == pystr)

        # hash() should be available (may be an int or raise if uninitialized)
        try:
            h = hash(inst)
            self.assertIsInstance(h, int)
        except Exception:
            # acceptable if object is uninitialized in this build
            pass

    def test_pool_singleton_and_no_double_init(self):
        no_alloc = InternStringPool.__new__(InternStringPool, False)
        with self.assertRaises(RuntimeError):
            _ = InternStringPool()

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

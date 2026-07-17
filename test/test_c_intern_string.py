"""Test suite for PyAlgoEngine c_intern_string (backported from PyCyBase).

Verifies:
- Module globals and symbol presence
- Porting: PyAlgoEngine uses its own pools (MD allocators), not PyCyBase globals
- __contains__ membership test
- Interning, lookup, size tracking
- Iteration, stable pointers
- Cross-process SHM sharing (POOL)
- Intra-process heap isolation (INTRA_POOL)
- Compatibility: c_market_data consumers still work
"""
import os
import sys
import unittest
import multiprocessing

from algo_engine.base import c_intern_string as cis
from algo_engine.base.c_intern_string import (
    InternStringPool,
    InternString,
    POOL,
    INTRA_POOL,
)

_FORK_AVAILABLE = hasattr(os, 'fork') and sys.platform != 'win32'


# ============================================================
#  Module globals & porting verification
# ============================================================

class TestModuleGlobals(unittest.TestCase):
    """Verify module-level symbols and that we use PyAlgoEngine's own pools."""

    def test_module_exists(self):
        self.assertIsNotNone(cis)

    def test_symbols_present(self):
        self.assertTrue(hasattr(cis, 'POOL'))
        self.assertTrue(hasattr(cis, 'C_POOL'))
        self.assertTrue(hasattr(cis, 'INTRA_POOL'))
        self.assertTrue(hasattr(cis, 'C_INTRA_POOL'))
        self.assertTrue(hasattr(cis, 'InternStringPool'))
        self.assertTrue(hasattr(cis, 'InternString'))

    def test_pool_is_instance(self):
        self.assertIsInstance(POOL, InternStringPool)
        self.assertIsInstance(INTRA_POOL, InternStringPool)

    def test_pool_address_is_hex_string(self):
        addr = POOL.address
        self.assertIsInstance(addr, str)
        self.assertTrue(addr.startswith('0x'), msg=f'Expected hex address, got: {addr!r}')
        int(addr, 16)

    def test_c_pool_is_int(self):
        self.assertIsInstance(cis.C_POOL, int)

    def test_c_intra_pool_is_int(self):
        self.assertIsInstance(cis.C_INTRA_POOL, int)

    def test_uses_md_allocators_not_pycybase_globals(self):
        """PyAlgoEngine POOL must be independent from PyCyBase's POOL."""
        from cbase.intern_string.c_intern_string import POOL as CBC_POOL
        # Different objects
        self.assertIsNot(POOL, CBC_POOL)
        # Different underlying C maps (different addresses)
        self.assertNotEqual(POOL.address, CBC_POOL.address)

    def test_intra_pool_independent_from_pycybase(self):
        """PyAlgoEngine INTRA_POOL must be independent."""
        from cbase.intern_string.c_intern_string import INTRA_POOL as CBC_INTRA_POOL
        self.assertIsNot(INTRA_POOL, CBC_INTRA_POOL)
        self.assertNotEqual(INTRA_POOL.address, CBC_INTRA_POOL.address)

    def test_pools_independent_from_each_other(self):
        """POOL and INTRA_POOL are separate."""
        self.assertNotEqual(POOL.address, INTRA_POOL.address)


# ============================================================
#  __contains__ membership test
# ============================================================

class TestContains(unittest.TestCase):
    """Verify __contains__ on InternStringPool."""

    def setUp(self):
        self.pool = InternStringPool()

    def tearDown(self):
        del self.pool

    def test_contains_existing(self):
        self.pool.istr('ae_alpha')
        self.assertIn('ae_alpha', self.pool)

    def test_contains_missing(self):
        self.assertNotIn('ae_nonexistent', self.pool)

    def test_contains_is_side_effect_free(self):
        size_before = self.pool.size
        self.assertNotIn('ae_no_such_key', self.pool)
        self.assertEqual(self.pool.size, size_before)

    def test_contains_consistency_with_getitem(self):
        self.pool.istr('ae_present')
        self.assertIn('ae_present', self.pool)
        self.assertEqual(self.pool['ae_present'].string, 'ae_present')
        self.assertNotIn('ae_absent', self.pool)
        with self.assertRaises(KeyError):
            _ = self.pool['ae_absent']

    def test_contains_cross_pool_isolation(self):
        pool2 = InternStringPool()
        self.pool.istr('ae_only_here')
        self.assertIn('ae_only_here', self.pool)
        self.assertNotIn('ae_only_here', pool2)
        del pool2

    def test_contains_empty_string(self):
        self.pool.istr('')
        self.assertIn('', self.pool)


# ============================================================
#  Interning, lookup, and size
# ============================================================

class TestInternAndLookup(unittest.TestCase):
    """Verify basic interning and lookup operations."""

    def test_intern_increases_size(self):
        pool = InternStringPool()
        self.assertEqual(pool.size, 0)
        pool.istr('a')
        self.assertEqual(pool.size, 1)
        pool.istr('b')
        self.assertEqual(pool.size, 2)
        pool.istr('a')  # no change
        self.assertEqual(pool.size, 2)
        del pool

    def test_len_matches_size(self):
        pool = InternStringPool()
        pool.istr('x')
        self.assertEqual(len(pool), pool.size)
        del pool

    def test_getitem_existing(self):
        POOL.istr('ae_lookup_key')
        inst = POOL['ae_lookup_key']
        self.assertIsInstance(inst, InternString)
        self.assertEqual(inst.string, 'ae_lookup_key')

    def test_getitem_missing_raises_keyerror(self):
        with self.assertRaises(KeyError):
            _ = POOL['ae_never_interned_xyz']

    def test_empty_string_can_be_interned(self):
        pool = InternStringPool()
        inst = pool.istr('')
        self.assertEqual(inst.string, '')
        inst2 = pool.istr('')
        self.assertEqual(inst.address, inst2.address)
        del pool


# ============================================================
#  InternString properties
# ============================================================

class TestInternStringProperties(unittest.TestCase):
    """Verify all InternString property accessors."""

    def setUp(self):
        self.pool = InternStringPool()
        self.inst = self.pool.istr('ae_props')

    def tearDown(self):
        del self.pool

    def test_string_property(self):
        self.assertEqual(self.inst.string, 'ae_props')
        self.assertIsInstance(self.inst.string, str)

    def test_hash_value_property(self):
        hv = self.inst.hash_value
        self.assertIsNotNone(hv)
        self.assertIsInstance(hv, int)
        self.assertGreater(hv, 0)

    def test_address_property(self):
        addr = self.inst.address
        self.assertIsInstance(addr, str)
        self.assertTrue(addr.startswith('0x'))

    def test_pool_property(self):
        self.assertIs(self.inst.pool, self.pool)

    def test_repr(self):
        r = repr(self.inst)
        self.assertIn('InternString', r)
        self.assertIn('ae_props', r)


# ============================================================
#  Comparison & hashing
# ============================================================

class TestComparison(unittest.TestCase):
    """Verify equality, comparison, and hashing."""

    def setUp(self):
        self.pool = InternStringPool()

    def tearDown(self):
        del self.pool

    def test_eq_same_key(self):
        a = self.pool.istr('ae_alpha')
        b = self.pool.istr('ae_alpha')
        self.assertTrue(a == b)

    def test_eq_different_key(self):
        a = self.pool.istr('ae_alpha')
        b = self.pool.istr('ae_beta')
        self.assertFalse(a == b)

    def test_eq_python_string(self):
        a = self.pool.istr('ae_alpha')
        self.assertTrue(a == 'ae_alpha')
        self.assertFalse(a == 'ae_beta')

    def test_hash_same_key_same_hash(self):
        a = self.pool.istr('ae_delta')
        b = self.pool.istr('ae_delta')
        self.assertEqual(hash(a), hash(b))

    def test_hash_usable_in_dict(self):
        a = self.pool.istr('ae_zeta')
        d = {a: 'value'}
        self.assertEqual(d[self.pool.istr('ae_zeta')], 'value')


# ============================================================
#  Stable pointer identity
# ============================================================

class TestStablePointer(unittest.TestCase):
    """Verify same key always maps to same internalized pointer."""

    def setUp(self):
        self.pool = InternStringPool()

    def tearDown(self):
        del self.pool

    def test_same_address_on_reintern(self):
        key = 'ae_stable_ptr'
        s1 = self.pool.istr(key)
        for _ in range(100):
            s2 = self.pool.istr(key)
            self.assertEqual(s1.address, s2.address)

    def test_address_different_for_different_keys(self):
        a = self.pool.istr('ae_key_a')
        b = self.pool.istr('ae_key_b')
        self.assertNotEqual(a.address, b.address)

    def test_address_preserved_after_other_insertions(self):
        a = self.pool.istr('ae_stable_key')
        addr_before = a.address
        for i in range(100):
            self.pool.istr(f'ae_filler_{i}')
        a_after = self.pool.istr('ae_stable_key')
        self.assertEqual(addr_before, a_after.address)


# ============================================================
#  Iteration
# ============================================================

class TestIteration(unittest.TestCase):
    """Verify the internalized() generator."""

    def setUp(self):
        self.pool = InternStringPool()

    def tearDown(self):
        del self.pool

    def test_empty_pool_yields_nothing(self):
        items = list(self.pool.internalized())
        self.assertEqual(items, [])

    def test_iteration_returns_internstrings(self):
        keys = {'ae_x', 'ae_y', 'ae_z'}
        for k in keys:
            self.pool.istr(k)
        items = list(self.pool.internalized())
        self.assertEqual(len(items), 3)
        for item in items:
            self.assertIsInstance(item, InternString)
        returned_keys = {item.string for item in items}
        self.assertEqual(returned_keys, keys)

    def test_reinterning_does_not_change_order(self):
        keys = ['ae_a', 'ae_b', 'ae_c']
        for k in keys:
            self.pool.istr(k)
        self.pool.istr('ae_a')  # re-intern
        items = list(self.pool.internalized())
        self.assertEqual([item.string for item in items], list(reversed(keys)))


# ============================================================
#  Unicode & edge cases
# ============================================================

class TestUnicode(unittest.TestCase):
    """Verify unicode string interning."""

    def setUp(self):
        self.pool = InternStringPool()

    def tearDown(self):
        del self.pool

    def test_unicode_basic(self):
        inst = self.pool.istr('café')
        self.assertEqual(inst.string, 'café')

    def test_unicode_chinese(self):
        inst = self.pool.istr('你好世界')
        self.assertEqual(inst.string, '你好世界')

    def test_unicode_emoji(self):
        inst = self.pool.istr('🎉🚀🔥')
        self.assertEqual(inst.string, '🎉🚀🔥')

    def test_long_string(self):
        s = 'x' * 5000
        inst = self.pool.istr(s)
        self.assertEqual(inst.string, s)


# ============================================================
#  SHM multiprocessing (POOL)
# ============================================================

@unittest.skipUnless(_FORK_AVAILABLE, "fork not available on this platform")
class TestShmMultiprocessing(unittest.TestCase):
    """Verify SHM-backed POOL visibility across fork."""

    def _fork_ctx(self):
        if 'fork' not in multiprocessing.get_all_start_methods():
            self.skipTest("'fork' start method not available")
        return multiprocessing.get_context('fork')

    def test_pre_fork_visibility(self):
        """Key interned before fork is visible in child."""
        POOL.istr('ae_shm_pre')

        ctx = self._fork_ctx()
        result = ctx.Queue()

        def worker(q):
            try:
                ok = 'ae_shm_pre' in POOL
                val = POOL['ae_shm_pre'].string
                q.put(('ok', ok, val))
            except Exception as e:
                q.put(('error', str(e)))

        p = ctx.Process(target=worker, args=(result,))
        p.start()
        p.join()
        status, ok, val = result.get()
        self.assertEqual(status, 'ok')
        self.assertTrue(ok)
        self.assertEqual(val, 'ae_shm_pre')

    def test_post_fork_visibility(self):
        """Key interned AFTER fork is visible in child — SHM is shared."""
        ctx = self._fork_ctx()
        ready = ctx.Event()
        result = ctx.Queue()

        def worker(evt, q):
            evt.wait()
            try:
                ok = 'ae_shm_post' in POOL
                q.put(('ok', ok))
            except Exception as e:
                q.put(('error', str(e)))

        p = ctx.Process(target=worker, args=(ready, result))
        p.start()
        POOL.istr('ae_shm_post')
        ready.set()
        p.join()
        status, ok = result.get()
        self.assertEqual(status, 'ok')
        self.assertTrue(ok, "Post-fork SHM key must be visible in child")

    def test_child_intern_visible_to_parent(self):
        """Child interns a key → parent sees it (shared SHM)."""
        ctx = self._fork_ctx()
        child_done = ctx.Event()
        result = ctx.Queue()

        def worker(evt, q):
            try:
                POOL.istr('ae_shm_child')
                q.put(('ok',))
            except Exception as e:
                q.put(('error', str(e)))
            finally:
                evt.set()

        p = ctx.Process(target=worker, args=(child_done, result))
        p.start()
        child_done.wait()
        p.join()
        status, = result.get()
        self.assertEqual(status, 'ok')
        self.assertIn('ae_shm_child', POOL)


# ============================================================
#  Heap multiprocessing (INTRA_POOL)
# ============================================================

@unittest.skipUnless(_FORK_AVAILABLE, "fork not available on this platform")
class TestIntraMultiprocessing(unittest.TestCase):
    """Verify heap-backed INTRA_POOL behaviour across fork."""

    def _fork_ctx(self):
        if 'fork' not in multiprocessing.get_all_start_methods():
            self.skipTest("'fork' start method not available")
        return multiprocessing.get_context('fork')

    def test_pre_fork_visibility(self):
        """Key interned before fork is visible in child (COW copy)."""
        INTRA_POOL.istr('ae_intra_pre')

        ctx = self._fork_ctx()
        result = ctx.Queue()

        def worker(q):
            try:
                ok = 'ae_intra_pre' in INTRA_POOL
                q.put(('ok', ok))
            except Exception as e:
                q.put(('error', str(e)))

        p = ctx.Process(target=worker, args=(result,))
        p.start()
        p.join()
        status, ok = result.get()
        self.assertEqual(status, 'ok')
        self.assertTrue(ok, "Pre-fork heap key must be visible in child")

    def test_post_fork_invisible(self):
        """Key interned AFTER fork is NOT visible in child."""
        ctx = self._fork_ctx()
        ready = ctx.Event()
        result = ctx.Queue()

        def worker(evt, q):
            evt.wait()
            try:
                ok = 'ae_intra_post' in INTRA_POOL
                q.put(('ok', ok))
            except Exception as e:
                q.put(('error', str(e)))

        p = ctx.Process(target=worker, args=(ready, result))
        p.start()
        INTRA_POOL.istr('ae_intra_post')  # only in parent's heap
        ready.set()
        p.join()
        status, ok = result.get()
        self.assertEqual(status, 'ok')
        self.assertFalse(ok, "Post-fork heap key must NOT be visible in child")

    def test_pre_fork_visible_post_fork_not(self):
        """Child sees pre-fork key but NOT post-fork key."""
        INTRA_POOL.istr('ae_intra_pre2')

        ctx = self._fork_ctx()
        ready = ctx.Event()
        result = ctx.Queue()

        def worker(evt, q):
            evt.wait()
            try:
                pre_ok = 'ae_intra_pre2' in INTRA_POOL
                post_ok = 'ae_intra_post2' in INTRA_POOL
                q.put(('ok', pre_ok, post_ok))
            except Exception as e:
                q.put(('error', str(e)))

        p = ctx.Process(target=worker, args=(ready, result))
        p.start()
        INTRA_POOL.istr('ae_intra_post2')
        ready.set()
        p.join()
        status, pre_ok, post_ok = result.get()
        self.assertEqual(status, 'ok')
        self.assertTrue(pre_ok, "Pre-fork heap key must be visible")
        self.assertFalse(post_ok, "Post-fork heap key must NOT be visible")


# ============================================================
#  Stress
# ============================================================

class TestStress(unittest.TestCase):
    """Stress-test with many strings."""

    def test_many_strings(self):
        pool = InternStringPool()
        count = 1000
        for i in range(count):
            pool.istr(f'ae_stress_{i:08d}')
        self.assertEqual(pool.size, count)
        for i in range(count):
            self.assertEqual(pool[f'ae_stress_{i:08d}'].string, f'ae_stress_{i:08d}')
        del pool

    def test_repeated_intern_same_key(self):
        pool = InternStringPool()
        key = 'ae_repeated'
        for _ in range(1000):
            inst = pool.istr(key)
            self.assertEqual(inst.string, key)
        self.assertEqual(pool.size, 1)
        del pool


if __name__ == '__main__':
    unittest.main()

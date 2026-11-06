import unittest

from algo_engine.base.c_allocator_protocol import MDConfigContext, RUNTIME_ALLOCATOR_CONFIG
from algo_engine.base.c_market_data import c_market_data as md

CONFIG = md.CONFIG
RUNTIME_MD_CONFIG = md.RUNTIME_MD_CONFIG


class TestEnvConfigAndViewer(unittest.TestCase):
    def test_config_is_nested_mappingproxy(self):
        """CONFIG is a read-only nested mapping proxy."""
        self.assertEqual(type(CONFIG).__name__, 'mappingproxy')

        # sections are mapping proxies too
        for section in ('market_data', 'market_data_buffer', 'exchange_profile'):
            self.assertIn(section, CONFIG)
            self.assertEqual(type(CONFIG[section]).__name__, 'mappingproxy')

    def test_config_compile_time_macros(self):
        """Compile-time macros are accessible via dict-style access."""
        md_section = CONFIG['market_data']
        self.assertEqual(md_section['BOOK_SIZE'], 10)
        self.assertEqual(md_section['ID_SIZE'], 16)
        self.assertEqual(md_section['LONG_ID_SIZE'], 128)
        self.assertEqual(md_section['MAX_WORKERS'], 128)
        self.assertFalse(md_section['DEBUG'])

        ep_section = CONFIG['exchange_profile']
        self.assertGreater(ep_section['SECONDS_PER_DAY'], 0)
        self.assertEqual(ep_section['EX_PROFILE_MIN_YEAR'], 1)

    def test_config_immutable(self):
        """CONFIG rejects item assignment."""
        with self.assertRaises(TypeError):
            CONFIG['new_key'] = 1

        with self.assertRaises(TypeError):
            CONFIG['market_data']['BOOK_SIZE'] = 99

    def test_env_config_context_context_manager_restores_values(self):
        original_locked = RUNTIME_ALLOCATOR_CONFIG.MD_CFG_LOCKED
        override = MDConfigContext(locked=not original_locked)

        with override:
            self.assertEqual(RUNTIME_ALLOCATOR_CONFIG.MD_CFG_LOCKED, (not original_locked))

        self.assertEqual(RUNTIME_ALLOCATOR_CONFIG.MD_CFG_LOCKED, original_locked)

    def test_env_config_context_or_and_invert(self):
        original_shared = RUNTIME_ALLOCATOR_CONFIG.MD_CFG_SHARED
        original_freelist = RUNTIME_ALLOCATOR_CONFIG.MD_CFG_FREELIST
        ctx_a = MDConfigContext(shared=False)
        ctx_b = MDConfigContext(freelist=False)

        with ctx_a:
            self.assertFalse(RUNTIME_ALLOCATOR_CONFIG.MD_CFG_SHARED)
        self.assertEqual(RUNTIME_ALLOCATOR_CONFIG.MD_CFG_SHARED, original_shared)

        with ctx_a | ctx_b:
            self.assertFalse(RUNTIME_ALLOCATOR_CONFIG.MD_CFG_SHARED)
            self.assertFalse(RUNTIME_ALLOCATOR_CONFIG.MD_CFG_FREELIST)
        self.assertEqual(RUNTIME_ALLOCATOR_CONFIG.MD_CFG_SHARED, original_shared)
        self.assertEqual(RUNTIME_ALLOCATOR_CONFIG.MD_CFG_FREELIST, original_freelist)

        with ~(ctx_a | ctx_b):
            self.assertTrue(RUNTIME_ALLOCATOR_CONFIG.MD_CFG_SHARED)
            self.assertTrue(RUNTIME_ALLOCATOR_CONFIG.MD_CFG_FREELIST)
        self.assertEqual(RUNTIME_ALLOCATOR_CONFIG.MD_CFG_SHARED, original_shared)
        self.assertEqual(RUNTIME_ALLOCATOR_CONFIG.MD_CFG_FREELIST, original_freelist)

    def test_env_config_context_callable_decorator(self):
        calls = []
        original_shared = RUNTIME_ALLOCATOR_CONFIG.MD_CFG_SHARED
        ctx = MDConfigContext(shared=not original_shared)

        @ctx
        def wrapped():
            calls.append(RUNTIME_ALLOCATOR_CONFIG.MD_CFG_SHARED)

        wrapped()

        self.assertEqual(calls, [not original_shared])
        self.assertEqual(RUNTIME_ALLOCATOR_CONFIG.MD_CFG_SHARED, original_shared)

    def test_runtime_config_reflects_globals(self):
        """RUNTIME_ALLOCATOR_CONFIG reads live cdef globals."""
        self.assertIsInstance(RUNTIME_ALLOCATOR_CONFIG.MD_CFG_LOCKED, bool)
        self.assertIsInstance(RUNTIME_ALLOCATOR_CONFIG.MD_CFG_SHARED, bool)
        self.assertIsInstance(RUNTIME_ALLOCATOR_CONFIG.MD_CFG_FREELIST, bool)
        self.assertIsInstance(RUNTIME_MD_CONFIG.MD_CFG_BOOK_SIZE, int)


if __name__ == "__main__":
    unittest.main()

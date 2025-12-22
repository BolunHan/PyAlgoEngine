import unittest

from algo_engine.base.c_market_data_ng import c_market_data as md

EnvConfigContext = md.EnvConfigContext
ConfigViewer = md.ConfigViewer
CONFIG = md.CONFIG


class TestEnvConfigAndViewer(unittest.TestCase):
    def test_env_config_context_context_manager_restores_values(self):
        original_locked = CONFIG.MD_CFG_LOCKED
        override = EnvConfigContext(locked=not original_locked)

        with override:
            self.assertEqual(CONFIG.MD_CFG_LOCKED, (not original_locked))

        self.assertEqual(CONFIG.MD_CFG_LOCKED, original_locked)

    def test_env_config_context_or_and_invert(self):
        original_shared = CONFIG.MD_CFG_SHARED
        original_freelist = CONFIG.MD_CFG_FREELIST
        ctx_a = EnvConfigContext(shared=False)
        ctx_b = EnvConfigContext(freelist=False)

        with ctx_a:
            self.assertFalse(CONFIG.MD_CFG_SHARED)
        self.assertEqual(CONFIG.MD_CFG_SHARED, original_shared)

        with ctx_a | ctx_b:
            self.assertFalse(CONFIG.MD_CFG_SHARED)
            self.assertFalse(CONFIG.MD_CFG_FREELIST)
        self.assertEqual(CONFIG.MD_CFG_SHARED, original_shared)
        self.assertEqual(CONFIG.MD_CFG_FREELIST, original_freelist)

        with ~(ctx_a | ctx_b):
            self.assertTrue(CONFIG.MD_CFG_SHARED)
            self.assertTrue(CONFIG.MD_CFG_FREELIST)
        self.assertEqual(CONFIG.MD_CFG_SHARED, original_shared)
        self.assertEqual(CONFIG.MD_CFG_FREELIST, original_freelist)

    def test_env_config_context_callable_decorator(self):
        calls = []
        original_shared = CONFIG.MD_CFG_SHARED
        ctx = EnvConfigContext(shared=not original_shared)

        @ctx
        def wrapped():
            calls.append(CONFIG.MD_CFG_SHARED)

        wrapped()

        self.assertEqual(calls, [not original_shared])
        self.assertEqual(CONFIG.MD_CFG_SHARED, original_shared)

    def test_config_viewer_reflects_globals(self):
        viewer = ConfigViewer.__new__(ConfigViewer)

        self.assertEqual(viewer.MD_CFG_LOCKED, CONFIG.MD_CFG_LOCKED)
        self.assertEqual(viewer.MD_CFG_SHARED, CONFIG.MD_CFG_SHARED)
        self.assertEqual(viewer.MD_CFG_FREELIST, CONFIG.MD_CFG_FREELIST)
        self.assertEqual(viewer.MD_CFG_BOOK_SIZE, CONFIG.MD_CFG_BOOK_SIZE)


if __name__ == "__main__":
    unittest.main()

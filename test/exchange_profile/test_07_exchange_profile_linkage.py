import datetime
import importlib.util
import pathlib
import sys
import sysconfig
import unittest

# Ensure the repo-tree package is imported (not an installed copy). On POSIX
# this also promotes the C symbols to the dynamic linker's global scope so the
# linkage extension can resolve EX_PROFILE at load time.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import algo_engine.exchange_profile  # noqa: F401
from algo_engine.exchange_profile.c_exchange_profile import SessionType
from algo_engine.exchange_profile.c_profile_cn import PROFILE_CN

SO_PATH = pathlib.Path(__file__).parent / f"c_exchange_profile_linkage{sysconfig.get_config_var('EXT_SUFFIX')}"


def _load_linkage_module():
    """Load the linkage extension by file path (built only with --with-tests)."""
    spec = importlib.util.spec_from_file_location("c_exchange_profile_linkage", str(SO_PATH))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@unittest.skipUnless(
    SO_PATH.exists(),
    "linkage extension not built — rebuild with: setup.py build_ext --inplace --with-tests",
)
class TestExchangeProfileLinkage(unittest.TestCase):
    """The extension consumes the exchange_profile C interface directly, so
    these tests prove cross-module C linkage: symbol resolution and, more
    importantly, sharing of the live EX_PROFILE global."""

    @classmethod
    def setUpClass(cls):
        cls.linkage = _load_linkage_module()

    def test_pydate_to_cdate_roundtrip(self):
        dt = datetime.date(2024, 2, 8)
        sd = self.linkage.pydate_to_cdate(dt)
        self.assertEqual(sd.to_pydate(), dt)

    def test_session_type_follows_active_profile(self):
        # The extension must observe the SAME EX_PROFILE global the package
        # mutates — a private copy would stay on the default profile.
        PROFILE_CN.activate()
        try:
            # 2024-02-12: CN Lunar New Year holiday, 2024-02-08: trading day
            self.assertEqual(self.linkage.session_type_of(datetime.date(2024, 2, 12)), int(SessionType.NON_TRADING))
            self.assertEqual(self.linkage.session_type_of(datetime.date(2024, 2, 8)), int(SessionType.NORMINAL))
        finally:
            PROFILE_CN.deactivate()

    def test_session_type_default_profile(self):
        # Without a CN profile active the same date resolves as a normal day.
        self.assertEqual(self.linkage.session_type_of(datetime.date(2024, 2, 12)), int(SessionType.NORMINAL))


if __name__ == "__main__":
    unittest.main()

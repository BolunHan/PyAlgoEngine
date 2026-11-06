import datetime
import importlib.util
import pathlib
import struct
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


@unittest.skipUnless(
    SO_PATH.exists(),
    "linkage extension not built — rebuild with: setup.py build_ext --inplace --with-tests",
)
class TestSessionDateLayoutContract(unittest.TestCase):
    """session_date_t tight-layout contract (EX_PROFILE_SESSION_DATE_NO_PADDING == 1).

    Expected behavior:
        - C-side sizeof(session_date_t) == 6.
        - Initializing a 0xAA-poisoned stack buffer writes every byte of the
          struct — no suffix padding survives.
        - The raw image is fully determined: it equals struct.pack('<HBBH', ...).

    Oracle: struct.pack little-endian. All supported targets (x86-64 Linux,
    x86-64/ARM64 Windows) are little-endian; on a known big-endian host
    c_ex_profile_base.h emits an informational [COMPILE] [DBG] diagnostic
    (no assert), so the little-endian image oracle only holds on little-endian
    builds. 2024-02-08 resolves to NORMINAL under both the default and the CN
    profile, so the expected image is profile-independent.
    """

    @classmethod
    def setUpClass(cls):
        cls.linkage = _load_linkage_module()

    def test_00_c_sizeof_is_six(self):
        self.assertEqual(self.linkage.session_date_sizeof(), 6)

    def test_01_no_padding_survives_poisoned_init(self):
        raw = self.linkage.session_date_raw_bytes(datetime.date(2024, 2, 8))
        self.assertEqual(len(raw), 6)
        self.assertNotIn(0xAA, raw)

    def test_02_raw_image_is_fully_determined(self):
        expected = struct.pack("<HBBH", 2024, 2, 8, int(SessionType.NORMINAL))
        self.assertEqual(self.linkage.session_date_raw_bytes(datetime.date(2024, 2, 8)), expected)


if __name__ == "__main__":
    unittest.main()

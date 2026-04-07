import pathlib
import sys
import timeit
import unittest
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

sys.path.insert(0, pathlib.Path(__file__).parents[2] / 'algo_engine')

from algo_engine.exchange_profile import PROFILE, PROFILE_CN

# ---------- Config ----------
TS = 1700000000  # fixed timestamp for consistency
N = 1_000_000  # number of iterations
TZ = ZoneInfo("Asia/Shanghai")
TZ_FIXED = timezone(offset=timedelta(hours=8), name="Asia/Shanghai")


# ---------- Functions (used by the benchmark) ----------
def naive():
    return datetime.fromtimestamp(TS)


def aware():
    return datetime.fromtimestamp(TS, TZ)


def aware_fixed():
    return datetime.fromtimestamp(TS, TZ_FIXED)


def profile_convert():
    return PROFILE.timestamp_to_datetime(TS)


def hybrid():
    return datetime.fromtimestamp(TS, PROFILE.time_zone)


class TestDTConversionPerformance(unittest.TestCase):
    """Performance benchmark for timestamp -> datetime conversions.

    This test case wraps the original script-style benchmark in a unittest so it
    can be executed together with the project's test suite. The test prints
    timing results but does not assert on them.
    """

    @classmethod
    def setUpClass(cls) -> None:
        # Activate CN profile for the benchmark (matches previous behavior).
        PROFILE_CN.activate()

    def test_conversion_performance(self) -> None:
        """Run the benchmark and print timing information.

        Note: this is a pure benchmark. It is intentionally left without
        assertions so it doesn't fail CI based on timing. Keep N large to
        preserve comparability with the original script.
        """
        t_naive = timeit.timeit("naive()", globals=globals(), number=N)
        t_aware = timeit.timeit("aware()", globals=globals(), number=N)
        t_aware_fixed = timeit.timeit("aware_fixed()", globals=globals(), number=N)
        t_profile_convert = timeit.timeit("profile_convert()", globals=globals(), number=N)
        t_hybrid = timeit.timeit("hybrid()", globals=globals(), number=N)

        print('Results:')
        print(f'{naive()=}')
        print(f'{aware()=}')
        print(f'{aware_fixed()=}')
        print(f'{profile_convert()=}')
        print(f'{hybrid()=}')

        print(f"Iterations: {N:,}")
        print(f"Naive           : {t_naive:.6f} sec ({t_naive / N * 1e9:.1f} ns/op)")
        print(f"With TZ         : {t_aware:.6f} sec ({t_aware / N * 1e9:.1f} ns/op)")
        print(f"Fixed offset    : {t_aware_fixed:.6f} sec ({t_aware_fixed / N * 1e9:.1f} ns/op)")
        print(f"PROFILE Convert : {t_profile_convert:.6f} sec ({t_profile_convert / N * 1e9:.1f} ns/op)")
        print(f"Hybrid          : {t_hybrid:.6f} sec ({t_hybrid / N * 1e9:.1f} ns/op)")


if __name__ == "__main__":
    unittest.main()

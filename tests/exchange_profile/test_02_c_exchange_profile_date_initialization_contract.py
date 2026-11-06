import copy
import pathlib
import pickle
import sys
import unittest
from datetime import date, datetime, timedelta

sys.path.insert(0, pathlib.Path(__file__).parents[2] / "algo_engine")

from algo_engine.exchange_profile.c_exchange_profile import (
    SessionDate,
    SessionDateRange,
    local_utc_offset_seconds,
)


TZ_OFFSET_SECONDS = local_utc_offset_seconds()


class TestSessionDateInitializationContract(unittest.TestCase):
    """Detect partially initialized SessionDate objects across all Python interfaces."""

    @classmethod
    def setUpClass(cls):
        cls.seed_pydate = date(2024, 11, 11)
        cls.seed = SessionDate.from_pydate(cls.seed_pydate)
        cls.seed_ts = datetime(2024, 11, 11, 12, 0, 0).timestamp() + TZ_OFFSET_SECONDS

    @staticmethod
    def _assert_bound_session_date(obj, source_name):
        """A bound SessionDate must not carry the '<SessionDate Unbound>' marker."""
        if not isinstance(obj, SessionDate):
            raise AssertionError(f"{source_name} did not return SessionDate, got {type(obj)}")

        text = repr(obj)
        if "Unbound" in text:
            raise AssertionError(f"{source_name} returned partially initialized SessionDate: {text}")

    def test_00_partial_instance_detector(self):
        partial = SessionDate.__new__(SessionDate, 2024, 11, 11)
        self.assertIsInstance(partial, SessionDate)
        self.assertIn("Unbound", repr(partial))

        with self.assertRaises(AssertionError):
            self._assert_bound_session_date(partial, "manual __new__")

    def test_01_overridden_and_new_constructors_return_bound_child(self):
        constructors = [
            ("SessionDate(...)", lambda: SessionDate(2024, 11, 11)),
            ("SessionDate.today()", SessionDate.today),
            ("SessionDate.from_unix", lambda: SessionDate.from_unix(self.seed_ts)),
            ("SessionDate.from_ordinal", lambda: SessionDate.from_ordinal(self.seed_pydate.toordinal())),
            ("SessionDate.from_pydate", lambda: SessionDate.from_pydate(self.seed_pydate)),
            ("SessionDate.fromisocalendar", lambda: SessionDate.fromisocalendar(2024, 46, 1)),
            ("SessionDate.fromisoformat", lambda: SessionDate.fromisoformat("2024-11-11")),
        ]

        for source_name, builder in constructors:
            with self.subTest(source_name=source_name):
                result = builder()
                self._assert_bound_session_date(result, source_name)

    def test_02_inherited_class_constructors_return_bound_child(self):
        inherited_class_constructors = [
            ("SessionDate.fromordinal", lambda: SessionDate.fromordinal(self.seed_pydate.toordinal())),
            ("SessionDate.fromtimestamp", lambda: SessionDate.fromtimestamp(self.seed_ts)),
        ]

        for source_name, builder in inherited_class_constructors:
            with self.subTest(source_name=source_name):
                result = builder()
                self._assert_bound_session_date(result, source_name)

    def test_03_instance_methods_returning_date_objects_are_bound_child(self):
        producers = [
            ("fork", lambda: self.seed.fork()),
            ("add_days", lambda: self.seed.add_days(1)),
            ("replace", lambda: self.seed.replace(day=12)),
            ("+ timedelta", lambda: self.seed + timedelta(days=1)),
            ("timedelta +", lambda: timedelta(days=1) + self.seed),
            ("- timedelta", lambda: self.seed - timedelta(days=1)),
        ]

        for source_name, producer in producers:
            with self.subTest(source_name=source_name):
                result = producer()
                self._assert_bound_session_date(result, source_name)

    def test_04_subtraction_interfaces_return_session_date_range_with_bound_dates(self):
        previous = SessionDate.from_pydate(date(2024, 11, 10))
        previous_pydate = date(2024, 11, 10)

        range_producers = [
            ("SessionDate - SessionDate", lambda: self.seed - previous),
            ("SessionDate - datetime.date", lambda: self.seed - previous_pydate),
            ("datetime.date - SessionDate", lambda: self.seed_pydate - previous),
        ]

        for source_name, producer in range_producers:
            with self.subTest(source_name=source_name):
                drange = producer()
                self.assertIsInstance(drange, SessionDateRange)
                self.assertGreaterEqual(drange.n_days, 1)
                self._assert_bound_session_date(drange[0], f"{source_name} first item")
                self._assert_bound_session_date(drange[-1], f"{source_name} last item")

    def test_05_copy_pickle_and_roundtrip_return_bound_child(self):
        copied = copy.copy(self.seed)
        self._assert_bound_session_date(copied, "copy.copy")

        deep_copied = copy.deepcopy(self.seed)
        self._assert_bound_session_date(deep_copied, "copy.deepcopy")

        loaded = pickle.loads(pickle.dumps(self.seed))
        self._assert_bound_session_date(loaded, "pickle roundtrip")

    def test_07_pickle_serialization_contract_for_bound_and_partial_instances(self):
        # __reduce__ should always restore a fully initialized SessionDate.
        bound_protocols = [0, 2, pickle.HIGHEST_PROTOCOL]
        for protocol in bound_protocols:
            with self.subTest(source="bound", protocol=protocol):
                payload = pickle.dumps(self.seed, protocol=protocol)
                loaded = pickle.loads(payload)
                self._assert_bound_session_date(loaded, f"pickle bound protocol={protocol}")
                self.assertEqual(loaded.to_pydate(), self.seed_pydate)

        partial = SessionDate.__new__(SessionDate, 2024, 11, 11)
        self.assertIn("Unbound", repr(partial))
        self.assertEqual(partial.addr, "0x0")

        for protocol in bound_protocols:
            with self.subTest(source="partial", protocol=protocol):
                payload = pickle.dumps(partial, protocol=protocol)
                loaded = pickle.loads(payload)
                self._assert_bound_session_date(loaded, f"pickle partial protocol={protocol}")
                self.assertEqual(loaded.to_pydate(), date(2024, 11, 11))

    def test_06_scalar_interfaces_are_stable_and_non_crashing(self):
        self.assertEqual(self.seed.year, 2024)
        self.assertEqual(self.seed.month, 11)
        self.assertEqual(self.seed.day, 11)

        self.assertIsInstance(self.seed.timestamp(), float)
        self.assertIsInstance(self.seed.unix_to_ordinal(self.seed_ts), int)
        self.assertIsInstance(self.seed.to_ordinal(), int)
        self.assertIsInstance(hash(self.seed), int)
        self.assertIsInstance(self.seed.session_type.name, str)

        self.assertIsInstance(self.seed.is_valid(), bool)
        self.assertIsInstance(self.seed.is_weekend(), bool)
        self.assertIsInstance(SessionDate.is_leap_year(2024), bool)
        self.assertIsInstance(SessionDate.days_in_month(2024, 11), int)

        self.assertEqual(self.seed.to_pydate(), self.seed_pydate)


if __name__ == "__main__":
    unittest.main()


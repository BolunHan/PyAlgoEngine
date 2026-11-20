"""Contract suite for the Cython and Python surface of the simulation matcher.

Stdlib only: the tests read ``c_simmatch.pxd`` / ``.pyx`` / ``.pyi`` and the
test toolkit, and check the dual-interface conventions — field mirroring, stub
symmetry, docstring coverage, ret-code mapping and the absence of duplicated
marshalling.
"""

import ast
import pathlib
import re
import unittest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
PYX_PATH = REPO_ROOT / 'algo_engine/backtest/c_simmatch.pyx'
PYI_PATH = REPO_ROOT / 'algo_engine/backtest/c_simmatch.pyi'
PXD_PATH = REPO_ROOT / 'algo_engine/backtest/c_simmatch.pxd'
TOOLKIT_PATH = REPO_ROOT / 'tests/backtest/c_simmatch_toolkit.pyx'


def read(path: pathlib.Path) -> str:
    return path.read_text(encoding='utf-8')


def method_body(text: str, name: str) -> str:
    match = re.search(rf'^    def {name}\(.*?\):\n((?:        .*\n)+)', text, re.M)
    if match is None:
        raise AssertionError(f'method not found: {name}')
    return match.group(1)


def public_pyx_methods(text: str) -> set[str]:
    """Names of the public def / cpdef methods (a cpdef may carry a C return type)."""
    names = set()
    for line in text.splitlines():
        match = re.match(r'^    (?:def|cpdef) (.*)\(', line)
        if match is None:
            continue
        name = match.group(1).split()[-1]
        if not name.startswith('_'):
            names.add(name)
    return names


class TestSimMatchSurface(unittest.TestCase):
    """The pxd, pyx, pyi and toolkit describe one consistent wrapper."""

    def setUp(self) -> None:
        self.pyx = read(PYX_PATH)
        self.pyi = read(PYI_PATH)
        self.pxd = read(PXD_PATH)
        self.toolkit = read(TOOLKIT_PATH)

    # === pxd / pyx parity ===

    def test_pxd_class_fields_are_used_in_the_pyx(self) -> None:
        """Every cdef field on the wrapper is reachable from the pyx."""
        class_block = self.pxd[self.pxd.index('cdef class SimMatch(CCPType):'):]
        fields = re.findall(r'^    cdef (?:public )?[^\n(]*?(\w+)$', class_block, re.M)
        self.assertTrue(fields, 'no cdef fields parsed from the pxd')

        for field in fields:
            self.assertRegex(self.pyx, rf'\bself\.{field}\b', f'pxd field unused in pyx: {field}')

    def test_pyx_public_methods_have_stubs(self) -> None:
        """Every public pyx method (and no other) has a pyi member."""
        pyx_methods = public_pyx_methods(self.pyx)
        pyi_properties = set(re.findall(r'^    @property\n    def (\w+)\(', self.pyi, re.M))
        pyi_methods = {
            name
            for name in re.findall(r'^    def (\w+)\(', self.pyi, re.M)
            if not name.startswith('_') and name not in pyi_properties
        }
        self.assertEqual(pyx_methods, pyi_methods)

    def test_pyx_properties_have_stubs(self) -> None:
        """The property set is identical in the pyx and the pyi."""
        pyx_properties = set(re.findall(r'^    property (\w+):', self.pyx, re.M))
        pyi_properties = set(re.findall(r'^    @property\n    def (\w+)\(', self.pyi, re.M))
        self.assertEqual(pyx_properties, {
            'timestamp',
            'last_price',
            'last_bid_price',
            'last_ask_price',
            'last_transaction_count',
            'matching_config',
            'market_time',
        })
        self.assertEqual(pyx_properties, pyi_properties)

    # === pyi contract ===

    def test_pyi_is_valid_python(self) -> None:
        compile(self.pyi, str(PYI_PATH), 'exec')

    def test_pyi_documents_every_member(self) -> None:
        """Every class, method and property getter carries a docstring."""
        tree = ast.parse(self.pyi)
        documented = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                self.assertTrue(ast.get_docstring(node), f'undocumented class: {node.name}')
                documented += 1
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if any(isinstance(decorator, ast.Attribute) and decorator.attr == 'setter' for decorator in node.decorator_list):
                    continue
                self.assertTrue(ast.get_docstring(node), f'undocumented member: {node.name}')
                documented += 1
        self.assertGreaterEqual(documented, 18)

    # === pyx hygiene ===

    def test_pyx_has_no_docstrings_or_return_annotations(self) -> None:
        """The pyx carries no docstrings and no Python return annotations."""
        self.assertNotIn('"""', self.pyx)
        self.assertIsNone(re.search(r'^\s*(?:c?def|cpdef) [^\n]*->', self.pyx, re.M))

    def test_pyx_uses_scoped_enum_members(self) -> None:
        """Enum members are reached through their type, never bare."""
        self.assertIsNone(re.search(r'(?<![\w.])SMM_[A-Z_0-9]+', self.pyx))

    def test_pyx_tests_pointers_by_truthiness(self) -> None:
        """NULL appears only as a value, never in a comparison."""
        self.assertNotIn('== NULL', self.pyx)
        self.assertNotIn('!= NULL', self.pyx)

    def test_pyx_uses_the_header_field(self) -> None:
        """The wrapper mirrors the C header in a field named header."""
        self.assertNotIn('.ctx', self.pyx)
        self.assertIn('self.header', self.pyx)
        self.assertIn('cdef smm_match_ctx* header', self.pxd)

    def test_static_adaptors_are_bound_to_the_wrapper(self) -> None:
        """The listener callback and price helper are SimMatch static methods."""
        self.assertIn('@staticmethod\n    cdef inline void c_simmatch_callback(', self.pyx)
        self.assertIn('@staticmethod\n    cdef inline void c_simmatch_callback(', self.pxd)
        self.assertIn('@staticmethod\n    cdef inline double c_price_extreme(', self.pyx)
        self.assertIn('@staticmethod\n    cdef inline double c_price_extreme(', self.pxd)
        self.assertIn('SimMatch.c_simmatch_callback', self.pyx)
        self.assertIn('SimMatch.c_price_extreme', self.pyx)

    def test_constructor_types_are_enforced(self) -> None:
        """The engine and topic-set types are statically enforced, in __init__."""
        self.assertIn(
            'def __init__(self, str ticker, EventEngineEx event_engine=None, TopicSetEx topic_set=None, uint64_t seed=0, **kwargs):',
            self.pyx,
        )
        self.assertNotIn('def __cinit__(', self.pyx)
        self.assertNotIn('isinstance(event_engine', self.pyx)
        self.assertNotIn('isinstance(topic_set', self.pyx)

    def test_wrapper_adopts_a_context_through_c_from_header(self) -> None:
        """The wrapper can be rebuilt from a native context pointer."""
        self.assertIn('cdef SimMatch c_from_header(smm_match_ctx* header, str ticker, bint owner=False):', self.pyx)
        self.assertIn('cdef SimMatch c_from_header(smm_match_ctx* header, str ticker, bint owner=?)', self.pxd)

    def test_order_registries_are_native_views(self) -> None:
        """working / history are bound bytemap views, never Python-side dicts."""
        self.assertIn('cdef class OrderMapView(BoundByteMap):', self.pyx)
        self.assertIn('cdef class OrderMapView(BoundByteMap):', self.pxd)
        self.assertIn('cdef readonly OrderMapView working', self.pxd)
        self.assertIn('cdef readonly OrderMapView history', self.pxd)

        self.assertNotIn('self.working = {}', self.pyx)
        self.assertNotIn('self.history = {}', self.pyx)
        self.assertNotIn('self.working[', self.pyx)
        self.assertNotIn('self.history[', self.pyx)
        self.assertNotIn('self.working.pop', self.pyx)
        self.assertNotIn('self.history.pop', self.pyx)

        self.assertIn('OrderMapView.c_from_header(self.header.working, False)', self.pyx)
        self.assertIn('OrderMapView.c_from_header(self.header.history, False)', self.pyx)
        self.assertIn('c_bytemap_gen_seq_id(<void*> instance)', self.pyx)

    def test_readonly_fields_honor_the_pxd(self) -> None:
        """The wrapper fields that carry runtime state are read-only and typed."""
        for field in ('ticker', 'event_engine', 'topic_set', 'protocol', 'working', 'history', 'random', 'seed'):
            self.assertRegex(self.pxd, rf'(?m)^    cdef readonly .* {field}$')
        self.assertIn('cdef readonly EventEngineEx event_engine', self.pxd)
        self.assertIn('cdef readonly TopicSetEx topic_set', self.pxd)

    def test_price_marshalling_is_not_duplicated(self) -> None:
        """One helper marshals the price tuple for both static methods."""
        self.assertIn('cdef inline double c_price_extreme(', self.pyx)
        self.assertEqual(self.pyx.count('c_smm_match_best_price('), 1)
        self.assertEqual(self.pyx.count('c_smm_match_worst_price('), 1)
        self.assertEqual(self.pyx.count('PyMem_Malloc('), 1)
        self.assertEqual(self.pyx.count('PyMem_Free('), 1)

        for name in ('best_price', 'worst_price'):
            body = method_body(self.pyx, name)
            self.assertIn('c_price_extreme', body)
            self.assertEqual(body.strip().count('\n'), 0, f'{name} does more than delegate')

    def test_fallible_c_calls_map_their_ret_code(self) -> None:
        """Each fallible C entry is followed by an inline ret-code mapping."""
        for name in ('c_smm_match_launch', 'c_smm_match_cancel', 'c_smm_match_process', 'c_smm_match_set_seed'):
            for match in re.finditer(rf'\b{name}\(', self.pyx):
                tail = self.pyx[match.end():match.end() + 600]
                self.assertIn('smm_ret_code.SMM_OK', tail, f'{name} ret code not interpreted at its call site')

    # === toolkit ===

    def test_toolkit_reads_the_header_and_scopes_enums(self) -> None:
        """The listener toolkit uses the wrapper contract and scoped constants."""
        self.assertIn('sim.header', self.toolkit)
        self.assertNotIn('sim.ctx', self.toolkit)
        self.assertIn('smm_ret_code.SMM_OK', self.toolkit)
        self.assertIn('def register_listener(', self.toolkit)
        self.assertIn('def deregister_listener(', self.toolkit)
        self.assertIn('def registry_state(', self.toolkit)
        self.assertIn('def working_order_ids(', self.toolkit)

    # === report map ===

    def test_pxd_declares_the_report_map(self) -> None:
        """The pxd mirrors the entry's report map and the view composer."""
        self.assertRegex(self.pxd, r'bytemap\*\s+reports')
        self.assertIn('cdef inline TradeInstruction c_compose(const smm_order_entry* entry)', self.pxd)
        self.assertIn('OrderMapView.c_compose(entry)', self.pyx)


if __name__ == '__main__':
    unittest.main()

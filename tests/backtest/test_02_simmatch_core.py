"""Contract suite for the C core of the simulation matcher.

Stdlib only: the tests read ``c_simmatch.h`` / ``c_simmatch.pxd`` and check the
house style, the header/pxd parity, the lifecycle and registry invariants, and
the matching guards.
"""

import pathlib
import re
import unittest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
HEADER_PATH = REPO_ROOT / 'algo_engine/backtest/c_simmatch.h'
PXD_PATH = REPO_ROOT / 'algo_engine/backtest/c_simmatch.pxd'


def read(path: pathlib.Path) -> str:
    return path.read_text(encoding='utf-8')


def function_body(text: str, name: str) -> str:
    match = re.search(rf'^static inline [^\n]*\b{name}\([^\n]*\{{$', text, re.M)
    if match is None:
        raise AssertionError(f'definition not found: {name}')

    brace = match.end() - 1
    depth = 0
    for index in range(brace, len(text)):
        if text[index] == '{':
            depth += 1
        elif text[index] == '}':
            depth -= 1
            if depth == 0:
                return text[brace:index + 1]
    raise AssertionError(f'unterminated function body: {name}')


def c_struct_fields(text: str, name: str) -> list[str]:
    block = re.search(rf'^struct {name} \{{$(.*?)^\}};', text, re.M | re.S)
    if block is None:
        raise AssertionError(f'struct not found: {name}')

    fields = []
    for line in block.group(1).splitlines():
        code = line.split('//')[0].strip()
        if not code or code.endswith(';') is False:
            continue
        fields.append(code[:-1].split()[-1])
    return fields


def pxd_struct_fields(text: str, name: str) -> list[str]:
    block = re.search(rf'^    ctypedef struct {name}:$(.*?)(?=^    ctypedef struct |^    # |^\Z)', text, re.M | re.S)
    if block is None:
        raise AssertionError(f'pxd struct not found: {name}')

    fields = []
    for line in block.group(1).splitlines():
        code = line.strip()
        if not code:
            continue
        fields.append(code.split()[-1])
    return fields


class TestSimMatchCore(unittest.TestCase):
    """The C header and its pxd mirror satisfy the house contract."""

    def setUp(self) -> None:
        self.header = read(HEADER_PATH)
        self.pxd = read(PXD_PATH)

    # === style and structure ===

    def test_header_guard_and_section_order(self) -> None:
        """The guard is C_SIMMATCH_H and the sections run in house order."""
        self.assertTrue(self.header.startswith('#ifndef C_SIMMATCH_H\n#define C_SIMMATCH_H\n'))
        self.assertTrue(self.header.rstrip().endswith('#endif /* C_SIMMATCH_H */'))

        sections = [
            '// ========== Constants ==========',
            '// ========== Enums ==========',
            '// ========== Structs ==========',
            '// ========== Forward Declarations ==========',
            '// ========== Utility Functions ==========',
            '// ========== Registry ==========',
            '// ========== Listener APIs ==========',
            '// ========== Public APIs ==========',
            '// ========== Matching Internals ==========',
        ]
        positions = [self.header.index(section) for section in sections]
        self.assertEqual(positions, sorted(positions))

    def test_field_comments_use_the_house_marker(self) -> None:
        """Field comments are ordinary trailing comments, not the malformed '// <'."""
        self.assertEqual(self.header.count('// <'), 0)

    def test_every_static_inline_definition_has_a_brief(self) -> None:
        """Each implementation carries a Doxygen @brief."""
        lines = self.header.splitlines()
        for index, line in enumerate(lines):
            if not re.match(r'^static inline .* \{$', line):
                continue
            window = []
            for back in range(index - 1, max(-1, index - 40), -1):
                if lines[back].startswith('static inline ') or lines[back].startswith('// =========='):
                    break
                window.append(lines[back])
            self.assertIn('@brief', '\n'.join(window), f'undocumented definition: {line}')

    def test_forward_declarations_are_aligned(self) -> None:
        """Every forward declaration starts the function name at one column."""
        columns = {
            line.index('c_smm_')
            for line in self.header.splitlines()
            if line.startswith('static inline ') and line.rstrip().endswith(';')
        }
        self.assertEqual(len(columns), 1, f'forward declarations misaligned: {sorted(columns)}')

    def test_struct_fields_are_aligned(self) -> None:
        """Every struct's field names share one column."""
        for name in ('smm_order_entry', 'smm_listener', 'smm_match_ctx'):
            block = re.search(rf'^struct {name} \{{$(.*?)^\}};', self.header, re.M | re.S)
            self.assertIsNotNone(block, f'struct not found: {name}')

            columns = set()
            for line in block.group(1).splitlines():
                code = line.split('//')[0].rstrip()
                if not code.strip() or code.strip().endswith(';') is False:
                    continue
                field = code.strip()[:-1].split()[-1]
                columns.add(code.rindex(field))
            self.assertEqual(len(columns), 1, f'{name} fields misaligned: {sorted(columns)}')

    # === constants ===

    def test_constants_split_fixed_and_tunable(self) -> None:
        """Fixed values are static const; the key size is derived; no capacity macro survives."""
        self.assertIn('static const uint8_t  SMM_UUID_VERSION_NIBBLE', self.header)
        self.assertIn('static const uint64_t SMM_RNG_MULTIPLIER', self.header)
        self.assertIn('static const double   SMM_RNG_DOUBLE_SCALE', self.header)

        self.assertIn('#define SMM_ID_KEY_SIZE (sizeof(md_id_type) + LONG_ID_SIZE)', self.header)
        self.assertNotIn('#ifndef SMM_ID_KEY_SIZE', self.header)
        self.assertNotIn('SMM_ORDER_RING_', self.header)

    # === header / pxd parity ===

    def test_struct_fields_mirror_the_pxd(self) -> None:
        """Field names and order are identical in the header and the pxd."""
        for name in ('smm_order_entry', 'smm_listener', 'smm_match_ctx'):
            self.assertEqual(
                c_struct_fields(self.header, name),
                pxd_struct_fields(self.pxd, name),
                f'{name} header/pxd mismatch',
            )

    def test_every_pxd_function_exists_in_the_header(self) -> None:
        """The pxd declares only functions the header defines."""
        header_functions = set(re.findall(r'\b(c_smm_[a-z0-9_]+)\s*\(', self.header))
        pxd_functions = set(re.findall(r'\b(c_smm_[a-z0-9_]+)\s*\(', self.pxd))
        self.assertTrue(pxd_functions <= header_functions, f'phantom pxd symbols: {sorted(pxd_functions - header_functions)}')

    def test_public_c_surface_is_pinned(self) -> None:
        """The public matcher and registry symbols are exactly the contract set."""
        public = set(re.findall(r'\b(c_smm_(?:match|check)_[a-z0-9_]+)\s*\(', self.header))
        self.assertEqual(public, {
            'c_smm_match_new',
            'c_smm_match_init',
            'c_smm_match_dealloc',
            'c_smm_match_free',
            'c_smm_match_clear',
            'c_smm_match_set_config',
            'c_smm_match_set_seed',
            'c_smm_match_register_listener',
            'c_smm_match_deregister_listener',
            'c_smm_match_fire',
            'c_smm_match_best_price',
            'c_smm_match_worst_price',
            'c_smm_match_launch',
            'c_smm_match_cancel',
            'c_smm_match_eod',
            'c_smm_match_process',
            'c_smm_check_bar',
            'c_smm_check_tick',
            'c_smm_check_tick_lite',
            'c_smm_check_order',
            'c_smm_check_trade',
        })

        registry = set(re.findall(r'\b(c_smm_(?:entry|entries|working|history)_[a-z0-9_]+)\s*\(', self.header))
        self.assertEqual(registry, {
            'c_smm_entry_new',
            'c_smm_entry_dealloc',
            'c_smm_entry_report_add',
            'c_smm_entries_clear',
            'c_smm_working_find',
            'c_smm_working_push',
            'c_smm_working_remove',
            'c_smm_working_touch',
            'c_smm_working_clear',
            'c_smm_history_find',
            'c_smm_history_insert',
            'c_smm_history_clear',
        })

    # === lifecycle and registry ===

    def test_lifecycle_contract(self) -> None:
        """_init takes a zeroed buf, _dealloc returns one, _new rolls back through _dealloc."""
        init = function_body(self.header, 'c_smm_match_init')
        self.assertNotIn('memset(ctx', init)
        self.assertIn('ctx->working = c_bytemap_new', init)
        self.assertIn('ctx->history = c_bytemap_new', init)
        self.assertNotIn('ws_report', self.header)

        dealloc = function_body(self.header, 'c_smm_match_dealloc')
        self.assertIn('c_smm_working_clear(ctx)', dealloc)
        self.assertIn('c_smm_history_clear(ctx)', dealloc)
        self.assertIn('c_bytemap_free(ctx->working)', dealloc)
        self.assertIn('c_bytemap_free(ctx->history)', dealloc)
        self.assertIn('free(listener)', dealloc)
        self.assertIn('memset(ctx, 0, sizeof(smm_match_ctx))', dealloc)

        new = function_body(self.header, 'c_smm_match_new')
        self.assertIn('c_smm_match_init(ctx)', new)
        self.assertIn('c_smm_match_dealloc(ctx)', new)
        self.assertIn('c_ap_free(ctx)', new)

        free = function_body(self.header, 'c_smm_match_free')
        self.assertIn('c_smm_match_dealloc(ctx)', free)

    def test_registry_invariants(self) -> None:
        """The registry is a bytemap of owned entries with a newest-first list."""
        self.assertIn('_Static_assert(offsetof(smm_order_entry, order) == 0, "a smm_order_entry is addressable as its instruction");', self.header)

        entry_new = function_body(self.header, 'c_smm_entry_new')
        self.assertIn('c_ap_alloc(sizeof(smm_order_entry)', entry_new)
        self.assertIn('entry->order = *order;', entry_new)

        find = function_body(self.header, 'c_smm_working_find')
        self.assertIn('c_bytemap_get(ctx->working', find)

        push = function_body(self.header, 'c_smm_working_push')
        self.assertIn('c_smm_entry_new(allocator, order)', push)
        self.assertIn('c_bytemap_set(ctx->working', push)
        self.assertIn('entry->older = ctx->newest;', push)
        self.assertIn('ctx->newest = entry;', push)

        remove = function_body(self.header, 'c_smm_working_remove')
        self.assertIn('c_bytemap_pop(ctx->working', remove)
        self.assertNotIn('c_ap_free', remove)
        self.assertNotIn('entry->older = NULL', remove)

        touch = function_body(self.header, 'c_smm_working_touch')
        self.assertIn('c_bytemap_set(ctx->working', touch)

        insert = function_body(self.header, 'c_smm_history_insert')
        self.assertIn('c_bytemap_set(ctx->history', insert)

        clear = function_body(self.header, 'c_smm_entries_clear')
        self.assertIn('c_smm_entry_dealloc(entry)', clear)
        self.assertIn('c_bytemap_clear(registry)', clear)

        self.assertIn('bytemap*         working;', self.header)
        self.assertIn('bytemap*         history;', self.header)
        self.assertIn('bytemap*             reports;', self.header)

    # === matching guards ===

    def test_order_copy_ownership(self) -> None:
        """The registry owns its order copies; launch takes a const instruction."""
        self.assertIn('c_smm_match_launch(smm_match_ctx* ctx, const md_trade_instruction* order)', self.header)
        self.assertIn('c_smm_working_push(smm_match_ctx* ctx, const md_trade_instruction* order)', self.header)

        launch = function_body(self.header, 'c_smm_match_launch')
        self.assertIn('entry->order.order_state', launch)
        self.assertIn('entry->order.ts_placed', launch)
        self.assertIsNone(re.search(r'(?<!entry->)order->order_state\s*=', launch))

        entry_new = function_body(self.header, 'c_smm_entry_new')
        self.assertIn('entry->order = *order;', entry_new)

        dealloc = function_body(self.header, 'c_smm_entry_dealloc')
        self.assertIn('c_ap_free(entry)', dealloc)
        self.assertIn('c_bytemap_free(entry->reports)', dealloc)

        report_add = function_body(self.header, 'c_smm_entry_report_add')
        self.assertIn('c_bytemap_ex_new(0u, sizeof(md_trade_report*)', report_add)
        self.assertIn('c_bytemap_set(entry->reports', report_add)

        self.assertIn("order id -> owned smm_order_entry*", self.header)

    def test_matching_guards(self) -> None:
        """The correctness guards the refactor landed stay in place."""
        has_limit = function_body(self.header, 'c_smm_has_limit_price')
        self.assertIn('isfinite(order->limit_price)', has_limit)

        lag = function_body(self.header, 'c_smm_lag_allowed')
        self.assertIn('ctx->last_transaction_count >= entry->transaction_count_at_launch', lag)
        self.assertIn(': 0u;', lag)

        try_match = function_body(self.header, 'c_smm_try_match')
        self.assertIn('if (sign == 0) return;', try_match)
        self.assertIn('if (!isfinite(match_price)) return;', try_match)
        self.assertIn('order->filled_volume >= order->volume', try_match)
        self.assertIn('c_smm_entry_report_add(entry, rpt)', try_match)
        self.assertIn('c_smm_working_remove(ctx, entry);', try_match)
        self.assertIn('c_smm_history_insert(ctx, entry)', try_match)
        self.assertIn('c_smm_working_touch(ctx, entry);', try_match)

        report_add = function_body(self.header, 'c_smm_entry_report_add')
        self.assertIn('c_bytemap_set(entry->reports', report_add)

        bar = function_body(self.header, 'c_smm_check_bar')
        self.assertIn('bar->volume > 0', bar)

        for name in ('c_smm_match_best_price', 'c_smm_match_worst_price'):
            price = function_body(self.header, name)
            self.assertIn('if (sign == 0) return SMM_ERR_INVALID_SIDE;', price)

    def test_hot_path_is_marked_and_allocation_free(self) -> None:
        """The per-event entry is named, documented and allocates nothing."""
        self.assertIn('// --- Hot path ---', self.header)
        self.assertIn('runs once per market-data event', self.header)

        process = function_body(self.header, 'c_smm_match_process')
        self.assertNotIn('c_ap_alloc', process)
        self.assertNotIn('memset', process)

    def test_scans_snapshot_the_next_link(self) -> None:
        """Every scan snapshots the older link before a fill can unlink the entry."""
        for name in ('c_smm_check_bar', 'c_smm_check_tick', 'c_smm_check_tick_lite', 'c_smm_check_order', 'c_smm_check_trade'):
            body = function_body(self.header, name)
            self.assertIn('smm_order_entry* next = entry->older;', body, f'{name} does not snapshot the next link')
            self.assertIn('entry = next;', body, f'{name} does not advance through the snapshot')


if __name__ == '__main__':
    unittest.main()

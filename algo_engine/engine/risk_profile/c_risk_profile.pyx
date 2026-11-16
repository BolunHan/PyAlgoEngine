import enum
import json
import os
import pathlib
from collections import defaultdict

from cpython.object cimport PyObject
from cpython.unicode cimport PyUnicode_AsUTF8, PyUnicode_AsUTF8AndSize, PyUnicode_FromString, PyUnicode_FromStringAndSize
from libc.math cimport isnan
from libc.string cimport memcpy

from cbase.allocator_protocol cimport c_ap_alloc, c_ap_free
from cbase.bytemap cimport bytemap_entry, c_bytemap_gen_seq_id

import pandas as pd

from algo_engine.base.c_allocator_protocol cimport MD_DEFAULT_ALLOCATOR
from algo_engine.base.c_market_data.c_trade_utils cimport TradeInstruction
from algo_engine.engine import LOGGER
from algo_engine.engine.c_dma cimport DMA
from algo_engine.engine.c_market_engine cimport MDS, c_mds_get_market_price


class RiskRule(enum.IntEnum):
    max_percentile         = dma_risk_rule.DMA_RULE_MAX_PERCENTILE
    max_trade_long         = dma_risk_rule.DMA_RULE_MAX_TRADE_LONG
    max_trade_short        = dma_risk_rule.DMA_RULE_MAX_TRADE_SHORT
    max_exposure_long      = dma_risk_rule.DMA_RULE_MAX_EXPOSURE_LONG
    max_exposure_short     = dma_risk_rule.DMA_RULE_MAX_EXPOSURE_SHORT
    max_notional_long      = dma_risk_rule.DMA_RULE_MAX_NOTIONAL_LONG
    max_notional_short     = dma_risk_rule.DMA_RULE_MAX_NOTIONAL_SHORT
    max_ttl_notional_long  = dma_risk_rule.DMA_RULE_MAX_TTL_NOTIONAL_LONG
    max_ttl_notional_short = dma_risk_rule.DMA_RULE_MAX_TTL_NOTIONAL_SHORT
    max_net_notional_long  = dma_risk_rule.DMA_RULE_MAX_NET_NOTIONAL_LONG
    max_net_notional_short = dma_risk_rule.DMA_RULE_MAX_NET_NOTIONAL_SHORT

    @property
    def is_global(self):
        return c_dma_risk_rule_is_global(self.value)


class RiskType(enum.IntEnum):
    invalid               = dma_risk_type.DMA_RISK_INVALID
    passed                = dma_risk_type.DMA_RISK_PASS
    incomplete            = dma_risk_type.DMA_RISK_INCOMPLETE
    price                 = dma_risk_type.DMA_RISK_PRICE
    trade_invalid         = dma_risk_type.DMA_RISK_TRADE_INVALID
    trade_long            = dma_risk_type.DMA_RISK_TRADE_LONG
    trade_short           = dma_risk_type.DMA_RISK_TRADE_SHORT
    exposure_long         = dma_risk_type.DMA_RISK_EXPOSURE_LONG
    exposure_short        = dma_risk_type.DMA_RISK_EXPOSURE_SHORT
    percentile_long       = dma_risk_type.DMA_RISK_PERCENTILE_LONG
    percentile_short      = dma_risk_type.DMA_RISK_PERCENTILE_SHORT
    notional_long         = dma_risk_type.DMA_RISK_NOTIONAL_LONG
    notional_short        = dma_risk_type.DMA_RISK_NOTIONAL_SHORT
    portfolio_net_long    = dma_risk_type.DMA_RISK_PORTFOLIO_NET_LONG
    portfolio_net_short   = dma_risk_type.DMA_RISK_PORTFOLIO_NET_SHORT
    portfolio_total_long  = dma_risk_type.DMA_RISK_PORTFOLIO_TOTAL_LONG
    portfolio_total_short = dma_risk_type.DMA_RISK_PORTFOLIO_TOTAL_SHORT
    ticker_blocked        = dma_risk_type.DMA_RISK_TICKER_BLOCKED
    ticker_unlisted       = dma_risk_type.DMA_RISK_TICKER_UNLISTED
    custom                = dma_risk_type.DMA_RISK_CUSTOM


cdef class RiskViolation(Exception):
    def __init__(self, object risk_type, code: int, str msg, *args, **kwargs):
        super().__init__(msg, *args)

        cdef object type_obj = RiskType(risk_type)
        cdef dma_risk_type type_value = <dma_risk_type> type_obj
        cdef dma_risk_violation* header = <dma_risk_violation*> c_ap_alloc(sizeof(dma_risk_violation), MD_DEFAULT_ALLOCATOR)
        if not header:
            raise MemoryError(f'Failed to allocate risk violation header for {self.__class__.__name__}.')

        cdef Py_ssize_t msg_len = 0
        cdef const char* msg_ptr = PyUnicode_AsUTF8AndSize(msg, &msg_len) if msg else NULL

        if msg_len >= DMA_RISK_MESSAGE_SIZE:
            msg_len = DMA_RISK_MESSAGE_SIZE - 1

        if msg_len > 0:
            memcpy(&header.msg[0], msg_ptr, msg_len)

        header.msg[msg_len] = 0
        header.code = code
        header.type = type_value

        self.header = <const dma_risk_violation*> header
        self.owner = True
        self.ccp_attach()

        for kwarg in kwargs:
            setattr(self, kwarg, kwargs[kwarg])

    def __dealloc__(self):
        self.ccp_detach()

        if not self.owner:
            return

        if self.header:
            c_ap_free(<void*> self.header)

    cdef inline void ccp_attach(self):
        CCPType.ccp_attach(<PyObject*> self, <const void**> &self.header, &self.ccp_ctx, <void*> NULL)

    cdef inline void ccp_detach(self):
        CCPType.ccp_detach(<PyObject*> self, &self.ccp_ctx)

    @staticmethod
    cdef RiskViolation c_from_header(const dma_risk_violation* header, bint owner=False):
        cdef RiskViolation instance = RiskViolation.__new__(RiskViolation)
        instance.header = header
        instance.owner = owner
        instance.ccp_attach()
        return instance

    @staticmethod
    cdef RiskViolation c_from_value(dma_risk_violation value):
        cdef RiskViolation instance = RiskViolation.__new__(RiskViolation)
        cdef dma_risk_violation* header = <dma_risk_violation*> c_ap_alloc(sizeof(dma_risk_violation), MD_DEFAULT_ALLOCATOR)
        if not header:
            raise MemoryError(f'Failed to allocate risk violation header for <{RiskViolation.__name__}>.')

        memcpy(header, &value, sizeof(dma_risk_violation))

        instance.header = <const dma_risk_violation*> header
        instance.owner = True
        instance.ccp_attach()
        return instance

    property code:
        def __get__(self):
            if not self.header:
                raise RuntimeError(f'<{self.__class__.__name__}> not initialized!')
            return self.header.code

    property type:
        def __get__(self):
            if not self.header:
                raise RuntimeError(f'<{self.__class__.__name__}> not initialized!')
            return RiskType(self.header.type)

    property msg:
        def __get__(self):
            if not self.header:
                raise RuntimeError(f'<{self.__class__.__name__}> not initialized!')
            return PyUnicode_FromString(&self.header.msg[0])


cdef class RiskLimits(CCPType):
    @staticmethod
    cdef RiskLimits c_from_header(dma_risk_limits* header):
        cdef RiskLimits instance = RiskLimits.__new__(RiskLimits)
        instance.header = header
        instance.ccp_bind(&instance.header)
        return instance

    def __repr__(self):
        return f'<{self.__class__.__name__}>{{id={id(self)}}}'

    # === Python Properties ===

    property max_percentile:
        def __get__(self):
            if not self.header:
                raise RuntimeError(f'<{self.__class__.__name__}> not initialized!')
            if not (self.header.enabled & c_dma_risk_rule_bit(dma_risk_rule.DMA_RULE_MAX_PERCENTILE)):
                return None
            return self.header.max_percentile

    property max_trade_long:
        def __get__(self):
            if not self.header:
                raise RuntimeError(f'<{self.__class__.__name__}> not initialized!')
            if not (self.header.enabled & c_dma_risk_rule_bit(dma_risk_rule.DMA_RULE_MAX_TRADE_LONG)):
                return None
            return self.header.max_trade_long

    property max_trade_short:
        def __get__(self):
            if not self.header:
                raise RuntimeError(f'<{self.__class__.__name__}> not initialized!')
            if not (self.header.enabled & c_dma_risk_rule_bit(dma_risk_rule.DMA_RULE_MAX_TRADE_SHORT)):
                return None
            return self.header.max_trade_short

    property max_exposure_long:
        def __get__(self):
            if not self.header:
                raise RuntimeError(f'<{self.__class__.__name__}> not initialized!')
            if not (self.header.enabled & c_dma_risk_rule_bit(dma_risk_rule.DMA_RULE_MAX_EXPOSURE_LONG)):
                return None
            return self.header.max_exposure_long

    property max_exposure_short:
        def __get__(self):
            if not self.header:
                raise RuntimeError(f'<{self.__class__.__name__}> not initialized!')
            if not (self.header.enabled & c_dma_risk_rule_bit(dma_risk_rule.DMA_RULE_MAX_EXPOSURE_SHORT)):
                return None
            return self.header.max_exposure_short

    property max_notional_long:
        def __get__(self):
            if not self.header:
                raise RuntimeError(f'<{self.__class__.__name__}> not initialized!')
            if not (self.header.enabled & c_dma_risk_rule_bit(dma_risk_rule.DMA_RULE_MAX_NOTIONAL_LONG)):
                return None
            return self.header.max_notional_long

    property max_notional_short:
        def __get__(self):
            if not self.header:
                raise RuntimeError(f'<{self.__class__.__name__}> not initialized!')
            if not (self.header.enabled & c_dma_risk_rule_bit(dma_risk_rule.DMA_RULE_MAX_NOTIONAL_SHORT)):
                return None
            return self.header.max_notional_short

    property enabled:
        def __get__(self):
            if not self.header:
                raise RuntimeError(f'<{self.__class__.__name__}> not initialized!')
            return self.header.enabled

    property rules:
        def __get__(self):
            if not self.header:
                raise RuntimeError(f'<{self.__class__.__name__}> not initialized!')

            cdef dma_risk_rule_ctx* rule = self.header.rules
            while rule:
                yield RiskRuleTemplate.c_from_header(rule, None)
                rule = rule.next


cdef class RiskLimitsMapView(BoundByteMap):
    @staticmethod
    cdef RiskLimitsMapView c_from_header(bytemap* header, bint owner=False):
        if owner:
            raise PermissionError('Must not assign the limit map ownership to <RiskLimitsMapView>')

        cdef RiskLimitsMapView instance = RiskLimitsMapView.__new__(RiskLimitsMapView)
        instance.owner = False
        instance.seq_id = c_bytemap_gen_seq_id(<void*> instance)
        instance.c_bind(header)
        return instance

    cdef const char* c_serialize_key(self, object obj, size_t* key_len):
        raise PermissionError(f'Must not assign a limit set to {self.__class__.__name__}')

    cdef object c_deserialize_key(self, const char* key, size_t key_len):
        return PyUnicode_FromStringAndSize(key, key_len)

    cdef const char* c_serialize_value(self, object obj, size_t* value_len):
        raise PermissionError(f'Must not assign a limit set to {self.__class__.__name__}')

    cdef object c_deserialize_value(self, const char* value, size_t value_len):
        cdef dma_risk_limits* limits = (<dma_risk_limits**> value)[0]
        if not limits:
            return None
        return RiskLimits.c_from_header(limits)


cdef class RiskProfile(CCPType):
    def __init__(self, **kwargs):
        self.registered_rules = {}

        self.header = c_dma_risk_profile_new(MD_DEFAULT_ALLOCATOR)
        if not self.header:
            raise MemoryError(f'Failed to allocate memory for <{self.__class__.__name__}>.')

        self.owner = True
        self.ccp_bind(&self.header)

        if kwargs:
            self.set_rules(kwargs)

    def __dealloc__(self):
        if not self.owner:
            return

        if self.header:
            c_dma_risk_profile_free(self.header)
            self.header = NULL

    cdef void __ccp_dealloc__(self):
        self.limits_map = None

        if self.registered_rules is not None:
            self.registered_rules.clear()

    def __repr__(self):
        return f'<{self.__class__.__name__}>{{id={id(self)}}}'

    def __call__(self, *order):
        if len(order) == 1:
            return self.check(order[0])
        else:
            return self.check_basket(*order)

    # === Cython Internal Binding ===

    cdef bint c_report_violation(self, const dma_risk_violation* violation):
        if violation.type == dma_risk_type.DMA_RISK_PASS:
            return True

        cdef str message = PyUnicode_FromString(violation.msg) if violation.msg[0] != 0 else ''
        LOGGER.error(f'<{PyUnicode_FromString(c_dma_risk_type_name(violation.type))}.{violation.code}>: {message}')
        return False

    cdef void c_set_limit(self, dma_risk_rule rule, str ticker, double value):
        cdef int ret_code = c_dma_risk_profile_set_limit(self.header, PyUnicode_AsUTF8(ticker) if ticker is not None else NULL, rule, value)

        if ret_code == dma_risk_ret_code.DMA_RISK_OK_RET:
            return

        cdef str rule_key = PyUnicode_FromString(c_dma_risk_rule_name(rule))

        if ret_code == dma_risk_ret_code.DMA_RISK_ERR_RULE_SCOPE:
            if not ticker:
                LOGGER.error(f'Invalid action: limit <{rule_key}> requires a valid ticker')
            else:
                LOGGER.error(f'Invalid action: can not set any ticker for limit <{rule_key}>')
        elif ret_code == dma_risk_ret_code.DMA_RISK_ERR_INVALID_INPUT:
            LOGGER.error(f'Invalid action: limit <{rule_key}> must be a non-negative number, got {value}')
        elif ret_code == dma_risk_ret_code.DMA_RISK_ERR_UNKNOWN_RULE:
            LOGGER.error(f'Invalid action: limit <{rule_key}> not found!')
        else:
            LOGGER.error(f'Failed to set limit <{rule_key}>={value}, err code: {ret_code}')

    cdef dict c_snapshot_rules(self):
        cdef dict rules = dict(entry=set(), whitelist=set(), blacklist=set(), custom={}, custom_global=[])

        cdef bytemap_entry* entry = self.header.entries.first
        cdef bytemap_entry* gate_entry
        cdef dma_risk_limits* limits
        cdef dma_risk_rule_ctx* rule_ctx
        cdef object rule, ticker
        cdef const char* ticker_ptr
        cdef double value

        for rule in RiskRule:
            rules[rule.name] = None if rule.is_global else {}

        while entry:
            ticker = PyUnicode_FromStringAndSize(entry.key, entry.key_length)
            ticker_ptr = PyUnicode_AsUTF8(ticker)
            limits = (<dma_risk_limits**> entry.value)[0]

            rules['entry'].add(ticker)

            for rule in RiskRule:
                if rule.is_global:
                    continue
                if c_dma_risk_profile_get_limit(self.header, ticker_ptr, <dma_risk_rule> rule, &value):
                    rules[rule.name][ticker] = value

            if limits and limits.rules:
                rules['custom'][ticker] = []

                rule_ctx = limits.rules

                while rule_ctx:
                    rules['custom'][ticker].append(rule_ctx.risk_rule_code)
                    rule_ctx = rule_ctx.next

            entry = entry.next

        for rule in RiskRule:
            if not rule.is_global:
                continue
            if c_dma_risk_profile_get_limit(self.header, NULL, <dma_risk_rule> rule, &value):
                rules[rule.name] = value

        rule_ctx = self.header.globals.rules

        while rule_ctx:
            rules['custom_global'].append(rule_ctx.risk_rule_code)
            rule_ctx = rule_ctx.next

        if self.header.whitelist:
            gate_entry = self.header.whitelist.first
            while gate_entry:
                rules['whitelist'].add(PyUnicode_FromStringAndSize(gate_entry.key, gate_entry.key_length))
                gate_entry = gate_entry.next

        if self.header.blacklist:
            gate_entry = self.header.blacklist.first
            while gate_entry:
                rules['blacklist'].add(PyUnicode_FromStringAndSize(gate_entry.key, gate_entry.key_length))
                gate_entry = gate_entry.next

        return rules

    # === Python Interfaces ===

    cpdef void set_rule(self, object key, double value, str ticker=None):
        cdef object rule_obj
        cdef dma_risk_rule rule

        try:
            rule_obj = RiskRule[key] if isinstance(key, str) else RiskRule(key)
        except (KeyError, ValueError):
            LOGGER.error(f'Invalid action: limit <{key}> not found!')
            return

        rule = <dma_risk_rule> rule_obj
        self.c_set_limit(rule, ticker, value)

    cpdef void set_rules(self, dict rules):
        cdef object value
        cdef dma_risk_rule rule
        cdef str key, ticker
        cdef double limit

        for key, value in rules.items():
            if key in ('entry', 'custom', 'custom_global'):
                continue

            if key == 'whitelist':
                for ticker in value:
                    self.set_whitelist(ticker, True)
                continue

            if key == 'blacklist':
                for ticker in value:
                    self.set_blacklist(ticker, True)
                continue

            try:
                rule = <dma_risk_rule> RiskRule[key]
            except KeyError:
                LOGGER.error(f'Invalid action: limit <{key}> not found!')
                continue

            if isinstance(value, dict):
                for ticker, limit in value.items():
                    self.c_set_limit(rule, ticker, limit)
            else:
                self.c_set_limit(rule, None, value)

    cpdef void set_whitelist(self, str ticker, bint enabled=True):
        cdef int ret_code

        if enabled:
            ret_code = c_dma_risk_profile_whitelist_add(self.header, PyUnicode_AsUTF8(ticker))
        else:
            ret_code = c_dma_risk_profile_whitelist_remove(self.header, PyUnicode_AsUTF8(ticker))

        if ret_code == dma_risk_ret_code.DMA_RISK_OK_RET:
            return

        if ret_code == dma_risk_ret_code.DMA_RISK_ERR_CONFLICT:
            LOGGER.error(f'Invalid action: <{ticker}> is blacklisted, drop it from the blacklist first')
        else:
            LOGGER.error(f'Failed to update the whitelist for <{ticker}>, err code: {ret_code}')

    cpdef void set_blacklist(self, str ticker, bint enabled=True):
        cdef int ret_code

        if enabled:
            ret_code = c_dma_risk_profile_blacklist_add(self.header, PyUnicode_AsUTF8(ticker))
        else:
            ret_code = c_dma_risk_profile_blacklist_remove(self.header, PyUnicode_AsUTF8(ticker))

        if ret_code == dma_risk_ret_code.DMA_RISK_OK_RET:
            return

        if ret_code == dma_risk_ret_code.DMA_RISK_ERR_CONFLICT:
            LOGGER.error(f'Invalid action: <{ticker}> is whitelisted, drop it from the whitelist first')
        else:
            LOGGER.error(f'Failed to update the blacklist for <{ticker}>, err code: {ret_code}')

    cpdef dict get(self, str ticker):
        cdef const char* ticker_ptr = PyUnicode_AsUTF8(ticker)
        cdef double price = c_mds_get_market_price(MDS.header, ticker_ptr)

        cdef dma_risk_check_state check_state
        c_dma_risk_profile_check_state_init(DMA.header, ticker_ptr, price, &check_state)

        cdef dict limit = dict(
            market_price=None if isnan(price) else price,
            working={'long': check_state.working_long, 'short': check_state.working_short},
            traded={'long': check_state.traded_long, 'short': check_state.traded_short},
            exposure={'long': check_state.exposure_long, 'short': check_state.exposure_short},
        )

        cdef object rule
        cdef double value
        for rule in RiskRule:
            if rule.is_global:
                if c_dma_risk_profile_get_limit(self.header, NULL, <dma_risk_rule> rule, &value):
                    limit[rule.name] = value
            elif c_dma_risk_profile_get_limit(self.header, ticker_ptr, <dma_risk_rule> rule, &value):
                limit[rule.name] = value

        return limit

    cpdef bint check(self, TradeInstruction order):
        cdef dma_risk_violation violation = c_dma_risk_profile_check_order(DMA.header, self.header, &order.header.trade_instruction)
        if violation.type == dma_risk_type.DMA_RISK_PASS:
            return True

        self.c_report_violation(&violation)
        raise RiskViolation.c_from_value(violation)

    cpdef bint check_order(self, str ticker, double volume, md_side side):
        cdef dma_risk_violation violation = c_dma_risk_profile_check_volume(DMA.header, self.header, PyUnicode_AsUTF8(ticker), side, volume)
        if violation.type == dma_risk_type.DMA_RISK_PASS:
            return True

        self.c_report_violation(&violation)
        raise RiskViolation.c_from_value(violation)

    def check_basket(self, *order):
        LOGGER.warning('risk control for basket order not implemented, check order individually')
        for item in order:
            self.check(item)

    cpdef void clear(self):
        c_dma_risk_profile_clear(self.header)
        self.registered_rules.clear()

    def to_json(self, fmt='str'):
        cdef dict json_dict = self.rules
        json_dict['entry'] = list(json_dict['entry'])
        json_dict['whitelist'] = list(json_dict['whitelist'])
        json_dict['blacklist'] = list(json_dict['blacklist'])

        if fmt == 'dict':
            return json_dict
        else:
            return json.dumps(json_dict)

    def from_json(self, json_str):
        if isinstance(json_str, (str, bytes)):
            json_dict = json.loads(json_str)
        elif isinstance(json_str, dict):
            json_dict = json_str
        else:
            raise TypeError(f'Invalid type {type(json_str)}, expect [str, bytes, dict]')

        self.set_rules(json_dict)
        return self

    def dump(self, file_path: str | pathlib.Path):
        file_path = pathlib.Path(file_path)
        dump_dir = file_path.parent

        os.makedirs(dump_dir, exist_ok=True)

        with open(file_path, 'w') as f:
            f.write(json.dumps(self.to_json(fmt='dict'), indent=4, sort_keys=True))

    def load(self, file_path: str | pathlib.Path):
        if not os.path.isfile(file_path):
            LOGGER.error(f'No such file {file_path}')
            return

        with open(file_path, 'r') as f:
            json_str = f.read()

        self.from_json(json_str)

    # === Python Properties ===

    property rules:
        def __get__(self):
            if not self.header:
                raise RuntimeError(f'<{self.__class__.__name__}> not initialized!')
            return self.c_snapshot_rules()

    property limits:
        def __get__(self):
            if not self.header:
                raise RuntimeError(f'<{self.__class__.__name__}> not initialized!')

            if self.limits_map is None:
                self.limits_map = RiskLimitsMapView.c_from_header(self.header.entries, False)

            return self.limits_map

    property info:
        def __get__(self):
            if not self.header:
                raise RuntimeError(f'<{self.__class__.__name__}> not initialized!')

            cdef object info_dict = defaultdict(dict)
            cdef dict rules = self.c_snapshot_rules()

            cdef str ticker
            cdef object rule
            for ticker in rules['entry']:
                for rule in RiskRule:
                    if rule.is_global:
                        continue
                    if ticker in rules[rule.name]:
                        info_dict[ticker][rule.name] = rules[rule.name][ticker]

            for rule in RiskRule:
                if not rule.is_global:
                    continue
                if rules[rule.name] is not None:
                    info_dict['global'][rule.name] = rules[rule.name]

            return pd.DataFrame(info_dict).T


cdef class RiskRuleTemplate(CCPType):
    def __init__(self, str name, int code):
        self.name = name
        self.code = code

    @staticmethod
    cdef dma_risk_type c_risk_rule_check_adaptor(
            const dma_risk_rule_ctx* rule,
            const dma* DMA,
            const dma_risk_profile* risk_profile,
            const char* ticker,
            md_side side,
            double volume,
            double price,
            void* user_data,
            dma_risk_violation* out
    ) noexcept:
        cdef RiskRuleTemplate self = <RiskRuleTemplate> <object> <PyObject*> user_data
        return self.c_check(rule, DMA, risk_profile, ticker, side, volume, price, out)

    @staticmethod
    cdef RiskRuleTemplate c_from_header(dma_risk_rule_ctx* risk_rule_ctx, RiskProfile parent):
        cdef RiskRuleTemplate instance = RiskRuleTemplate.__new__(RiskRuleTemplate)
        if not risk_rule_ctx:
            return None
        instance.risk_rule_ctx = risk_rule_ctx
        instance.profile = parent
        instance.name = PyUnicode_FromString(risk_rule_ctx.name) if risk_rule_ctx.name else None
        instance.code = risk_rule_ctx.risk_rule_code
        instance.ccp_bind(&instance.risk_rule_ctx)
        return instance

    # === Cython Internal Binding ===

    cdef dma_risk_type c_check(self, const dma_risk_rule_ctx* rule, const dma* DMA, const dma_risk_profile* risk_profile, const char* ticker, md_side side, double volume, double price, dma_risk_violation* out):
        return dma_risk_type.DMA_RISK_PASS

    cdef void c_register(self, RiskProfile risk_profile, str ticker):
        if self.profile is not None:
            raise RuntimeError(f'<{self}> is already registered.')

        if ticker is None:
            raise TypeError(f'<{self.__class__.__name__}> requires a ticker; use register_global for a global rule.')

        cdef dma_risk_rule_ctx* rule = c_dma_risk_profile_register_rule(
            risk_profile.header,
            PyUnicode_AsUTF8(ticker),
            RiskRuleTemplate.c_risk_rule_check_adaptor,
            self.code,
            PyUnicode_AsUTF8(self.name),
            <void*> <PyObject*> self
        )

        if not rule:
            raise RuntimeError(f'Failed to register <{self.__class__.__name__}>.')

        self.risk_rule_ctx = rule
        self.profile = risk_profile
        self.ccp_bind(&self.risk_rule_ctx)

        risk_profile.registered_rules[rule.rule_id] = self

    cdef void c_register_global(self, RiskProfile risk_profile):
        if self.profile is not None:
            raise RuntimeError(f'<{self}> is already registered.')

        cdef dma_risk_rule_ctx* rule = c_dma_risk_profile_register_global_rule(
            risk_profile.header,
            RiskRuleTemplate.c_risk_rule_check_adaptor,
            self.code,
            PyUnicode_AsUTF8(self.name),
            <void*> <PyObject*> self
        )

        if not rule:
            raise RuntimeError(f'Failed to register <{self.__class__.__name__}>.')

        self.risk_rule_ctx = rule
        self.profile = risk_profile
        self.ccp_bind(&self.risk_rule_ctx)

        risk_profile.registered_rules[rule.rule_id] = self

    cdef void c_unregister(self):
        if self.profile is None:
            raise RuntimeError(f'<{self}> is not registered.')

        cdef int ret_code
        cdef uintptr_t rule_id = self.rule_id

        if self.risk_rule_ctx.ticker:
            ret_code = c_dma_risk_profile_unregister_rule(self.profile.header, self.risk_rule_ctx.ticker, rule_id)
        else:
            ret_code = c_dma_risk_profile_unregister_global_rule(self.profile.header, rule_id)

        if ret_code != dma_risk_ret_code.DMA_RISK_OK_RET and ret_code != dma_risk_ret_code.DMA_RISK_ERR_NOT_FOUND:
            raise RuntimeError(f'Failed to unregister the rule, err code: {ret_code}')

        self.profile.registered_rules.pop(rule_id, None)
        self.risk_rule_ctx = NULL
        self.profile = None

    # === Python Interfaces ===

    def __repr__(self):
        cdef dma_risk_rule_ctx* rule = self.risk_rule_ctx
        cdef object scope

        if not rule:
            return f'<{self.__class__.__name__}>'

        scope = PyUnicode_FromString(rule.ticker) if rule.ticker else 'global'

        return (f'<{self.__class__.__name__} {scope}>('
                f'name={self.name} '
                f'code={rule.risk_rule_code} '
                f'id={rule.rule_id} '
                f'enabled={rule.enabled})')

    def register(self, RiskProfile risk_profile, str ticker=None):
        self.c_register(risk_profile, ticker)

    def register_global(self, RiskProfile risk_profile):
        self.c_register_global(risk_profile)

    def unregister(self):
        self.c_unregister()

    def enable(self):
        if self.profile is None:
            raise RuntimeError(f'<{self}> is not registered.')

        cdef int ret_code
        if self.risk_rule_ctx.ticker:
            ret_code = c_dma_risk_profile_enable_rule(self.profile.header, self.risk_rule_ctx.ticker, self.rule_id)
        else:
            ret_code = c_dma_risk_profile_enable_global_rule(self.profile.header, self.rule_id)

        if ret_code == dma_risk_ret_code.DMA_RISK_OK_RET:
            return

        if ret_code == dma_risk_ret_code.DMA_RISK_ERR_NOT_FOUND:
            raise RuntimeError(f'<{self}> is gone from its profile.')

        raise RuntimeError(f'Failed to enable the rule, err code: {ret_code}')

    def disable(self):
        if self.profile is None:
            raise RuntimeError(f'<{self}> is not registered.')

        cdef int ret_code
        if self.risk_rule_ctx.ticker:
            ret_code = c_dma_risk_profile_disable_rule(self.profile.header, self.risk_rule_ctx.ticker, self.rule_id)
        else:
            ret_code = c_dma_risk_profile_disable_global_rule(self.profile.header, self.rule_id)

        if ret_code == dma_risk_ret_code.DMA_RISK_OK_RET:
            return

        if ret_code == dma_risk_ret_code.DMA_RISK_ERR_NOT_FOUND:
            raise RuntimeError(f'<{self}> is gone from its profile.')

        raise RuntimeError(f'Failed to disable the rule, err code: {ret_code}')

    # === Python Properties ===

    property ticker:
        def __get__(self):
            if not self.risk_rule_ctx:
                raise RuntimeError(f'<{self.__class__.__name__}> not initialized!')
            return PyUnicode_FromString(self.risk_rule_ctx.ticker) if self.risk_rule_ctx.ticker else None

    property risk_rule_code:
        def __get__(self):
            if not self.risk_rule_ctx:
                raise RuntimeError(f'<{self.__class__.__name__}> not initialized!')
            return self.risk_rule_ctx.risk_rule_code

    property rule_id:
        def __get__(self):
            if not self.risk_rule_ctx:
                raise RuntimeError(f'<{self.__class__.__name__}> not initialized!')
            return self.risk_rule_ctx.rule_id

    property is_registered:
        def __get__(self):
            if not self.risk_rule_ctx:
                return False
            return self.profile is not None

    property enabled:
        def __get__(self):
            if not self.risk_rule_ctx:
                raise RuntimeError(f'<{self.__class__.__name__}> not initialized!')
            return self.risk_rule_ctx.enabled

from cpython.unicode cimport PyUnicode_AsUTF8, PyUnicode_FromString
from libc.stdint cimport uint32_t

from algo_engine.base.c_allocator_protocol cimport MD_DEFAULT_ALLOCATOR
from algo_engine.exchange_profile.c_exchange_profile cimport c_ex_profile_session_time_from_ts, c_ex_profile_time_to_ts

from .c_risk_profile cimport RiskProfile


cdef class RefusalWindowRule(RiskRuleTemplate):
    def __init__(self, py_time start=None, py_time end=None):
        if start is None and end is None:
            raise ValueError(f'<{self.__class__.__name__}> requires at least one bound.')

        self.start = start
        self.end = end
        RiskRuleTemplate.__init__(self, PyUnicode_FromString(REFUSAL_WINDOW_RULE_NAME), REFUSAL_WINDOW_RULE_CODE)

    # === Cython Internal Binding ===

    cdef void c_register(self, RiskProfile risk_profile, str ticker):
        if self.profile is not None:
            raise RuntimeError(f'<{self}> is already registered.')

        if ticker is None:
            raise TypeError(f'<{self.__class__.__name__}> requires a ticker; use register with the ticker.')

        cdef session_time_t start
        cdef session_time_t end
        cdef session_time_t* start_ptr = NULL
        cdef session_time_t* end_ptr = NULL
        cdef uint32_t nanosecond

        if self.start is not None:
            nanosecond = self.start.microsecond * 1000
            c_ex_profile_session_time_from_ts(c_ex_profile_time_to_ts(self.start.hour, self.start.minute, self.start.second, nanosecond), &start)
            start_ptr = &start

        if self.end is not None:
            nanosecond = self.end.microsecond * 1000
            c_ex_profile_session_time_from_ts(c_ex_profile_time_to_ts(self.end.hour, self.end.minute, self.end.second, nanosecond), &end)
            end_ptr = &end

        cdef refusal_window* window = c_refusal_window_new(start_ptr, end_ptr, MD_DEFAULT_ALLOCATOR)
        if not window:
            raise MemoryError(f'Failed to allocate memory for <{self.__class__.__name__}>.')

        cdef dma_risk_rule_ctx* rule = c_refusal_window_register(risk_profile.header, window, PyUnicode_AsUTF8(ticker))
        if not rule:
            c_refusal_window_free(window)
            raise RuntimeError(f'Failed to register <{self.__class__.__name__}>.')

        self.risk_rule_ctx = rule
        self.profile = risk_profile
        self.ccp_bind(&self.risk_rule_ctx)

        risk_profile.registered_rules[rule.rule_id] = self

    cdef void c_register_global(self, RiskProfile risk_profile):
        if self.profile is not None:
            raise RuntimeError(f'<{self}> is already registered.')

        cdef session_time_t start
        cdef session_time_t end
        cdef session_time_t* start_ptr = NULL
        cdef session_time_t* end_ptr = NULL
        cdef uint32_t nanosecond

        if self.start is not None:
            nanosecond = self.start.microsecond * 1000
            c_ex_profile_session_time_from_ts(c_ex_profile_time_to_ts(self.start.hour, self.start.minute, self.start.second, nanosecond), &start)
            start_ptr = &start

        if self.end is not None:
            nanosecond = self.end.microsecond * 1000
            c_ex_profile_session_time_from_ts(c_ex_profile_time_to_ts(self.end.hour, self.end.minute, self.end.second, nanosecond), &end)
            end_ptr = &end

        cdef refusal_window* window = c_refusal_window_new(start_ptr, end_ptr, MD_DEFAULT_ALLOCATOR)
        if not window:
            raise MemoryError(f'Failed to allocate memory for <{self.__class__.__name__}>.')

        cdef dma_risk_rule_ctx* rule = c_refusal_window_register_global(risk_profile.header, window)
        if not rule:
            c_refusal_window_free(window)
            raise RuntimeError(f'Failed to register <{self.__class__.__name__}>.')

        self.risk_rule_ctx = rule
        self.profile = risk_profile
        self.ccp_bind(&self.risk_rule_ctx)

        risk_profile.registered_rules[rule.rule_id] = self

    # === Python Interfaces ===

    def __repr__(self):
        cdef dma_risk_rule_ctx* rule = self.risk_rule_ctx
        if not rule:
            return f'<{self.__class__.__name__}>'

        cdef str scope = PyUnicode_FromString(rule.ticker) if rule.ticker else 'global'
        cdef str window

        if self.start is not None and self.end is not None:
            window = f'between {self.start} and {self.end}'
        elif self.start is not None:
            window = f'after {self.start}'
        else:
            window = f'before {self.end}'

        return (f'<{self.__class__.__name__} {scope}>('
                f'name={self.name} '
                f'code={rule.risk_rule_code} '
                f'id={rule.rule_id} '
                f'enabled={rule.enabled} '
                f'{window})')


cdef class RefusalWindowBeforeRule(RefusalWindowRule):
    def __init__(self, py_time until):
        RefusalWindowRule.__init__(self, None, until)


cdef class RefusalWindowAfterRule(RefusalWindowRule):
    def __init__(self, py_time since):
        RefusalWindowRule.__init__(self, since, None)


cdef class RefusalWindowBetweenRule(RefusalWindowRule):
    def __init__(self, py_time start, py_time end):
        RefusalWindowRule.__init__(self, start, end)

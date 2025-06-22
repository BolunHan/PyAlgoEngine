from cpython.datetime cimport timedelta, time_hour, time_minute, time_second, time_microsecond
from libc.math cimport fmod
from libc.stdlib cimport malloc, free

C_PROFILE = ProfileDispatcher()
C_PROFILE_DEFAULT = Profile()

PROFILE = C_PROFILE
PROFILE_DEFAULT = C_PROFILE_DEFAULT


cdef class ProfileDispatcher:
    def __cinit__(self):
        self.profile_id = C_PROFILE_DEFAULT.profile_id
        self.session_start = None
        self.session_end = None
        self.session_break = []
        self.time_zone = None

        self.session_break_num = 0
        self.session_break_start = NULL
        self.session_break_length = NULL

        self.c_trade_calendar = Profile.c_trade_calendar
        self.c_timestamp_in_market_session = Profile.c_timestamp_in_market_session
        self.c_time_in_market_session = Profile.c_time_in_market_session
        self.c_date_in_market_session = Profile.c_date_in_market_session

        self.c_refresh_cached_values()

    def __dealloc__(self):
        if self.session_break_start != NULL:
            free(self.session_break_start)

        if self.session_break_length != NULL:
            free(self.session_break_length)

    cdef void c_refresh_cached_values(self):
        # step 1: update session_start_ts, session_break_num, session_break_start and session_break_length
        cdef time session_start, session_end
        cdef bint break_adjusted = False

        if self.session_start is None:
            self.session_start_ts = 0.0
        else:
            session_start = <time> self.session_start
            self.session_start_ts = self.c_time_to_seconds(t=session_start, break_adjusted=break_adjusted)

        if self.session_end is None:
            self.session_end_ts = 86400.0
        else:
            session_end = <time> self.session_end
            self.session_end_ts = self.c_time_to_seconds(t=session_end, break_adjusted=break_adjusted)

        if self.session_break_start != NULL:
            free(self.session_break_start)

        if self.session_break_length != NULL:
            free(self.session_break_length)

        self.session_break_num = len(self.session_break)
        if self.session_break_num:
            self.session_break_start = <double*> malloc(self.session_break_num * sizeof(double))
            self.session_break_length = <double*> malloc(self.session_break_num * sizeof(double))

            if self.session_break_start == NULL or self.session_break_length == NULL:
                raise MemoryError("Unable to allocate memory")
        else:
            self.session_break_start = NULL
            self.session_break_length = NULL

        cdef double break_elapsed = 0.0
        cdef time break_start_time, break_end_time
        cdef double break_start_ts, break_end_ts, break_length
        cdef size_t i

        for i in range(self.session_break_num):
            break_start_time, break_end_time = self.session_break[i]
            break_start_ts = self.c_time_to_seconds(t=break_start_time, break_adjusted=break_adjusted)
            break_end_ts = self.c_time_to_seconds(t=break_end_time, break_adjusted=break_adjusted)
            break_length = break_end_ts - break_start_ts
            self.session_break_start[i] = break_start_ts
            self.session_break_length[i] = break_length
            break_elapsed += break_length

        # step 2: update the session_length, ts_offset
        self.tz_offset = datetime.now().astimezone().tzinfo.utcoffset(None).total_seconds()
        self.session_length = self.session_end_ts - self.session_start_ts - break_elapsed

    cdef inline double c_time_to_seconds(self, time t, bint break_adjusted):
        cdef double elapsed_seconds = (time_hour(t) * 60 + time_minute(t)) * 60 + time_second(t) + time_microsecond(t) / 1_000_000
        if break_adjusted:
            elapsed_seconds = self.c_break_adjusted(elapsed_seconds)
        return elapsed_seconds

    cdef inline double c_timestamp_to_seconds(self, double t, bint break_adjusted):
        cdef double elapsed_seconds = fmod(t + self.tz_offset, 86400.0)
        if break_adjusted:
            elapsed_seconds = self.c_break_adjusted(elapsed_seconds)
        return elapsed_seconds

    cdef inline double c_break_adjusted(self, double elapsed_seconds):
        cdef double break_elapsed = 0
        cdef double break_start
        cdef double break_length
        cdef size_t i

        for i in range(self.session_break_num):
            break_start = self.session_break_start[i]
            break_length = self.session_break_length[i]

            if elapsed_seconds > break_start + break_length:
                break_elapsed += break_length
            elif elapsed_seconds > break_start:
                break_elapsed += elapsed_seconds - break_start
            else:
                break

        cdef double session_ts = elapsed_seconds - self.session_start_ts - break_elapsed
        return session_ts

    cdef double c_trading_time_between(self, datetime start_time, datetime end_time):
        cdef date start_date = start_time.date()
        cdef date end_date = end_time.date()
        cdef size_t n_days = self.c_trading_days_between(start_date=start_date, end_date=end_date)
        cdef bint break_adjusted = True
        cdef double start_ts = self.c_time_to_seconds(t=start_time.time(), break_adjusted=break_adjusted)
        cdef double end_ts = self.c_time_to_seconds(t=end_time.time(), break_adjusted=break_adjusted)
        cdef double session_length = self.session_length
        cdef double seconds = end_ts - start_ts + session_length * n_days
        return seconds

    cdef date c_trading_days_before(self, date market_date, size_t days):
        # Calculate how many years we need to go back (250 trading days ≈ 1 year)
        cdef size_t years_back = (days // 250) + 1
        cdef date start_date = market_date - timedelta(days=365 * years_back)

        # Get the trade calendar
        cdef list trade_dates = self.c_trade_calendar(start_date=start_date, end_date=market_date)

        # Find the market_date in the calendar (it should be the last or near last)
        cdef int idx = len(trade_dates) - days

        if market_date == trade_dates[-1]:
            idx -= 1

        start_date = <date> trade_dates[0]
        if idx < 0:
            return self.c_trading_days_before(market_date=start_date, days=-idx)

        return trade_dates[idx]

    cdef date c_trading_days_after(self, date market_date, size_t days):
        # Calculate how many years we need to look ahead (250 trading days ≈ 1 year)
        cdef size_t years_ahead = (days // 250) + 1
        cdef date end_date = market_date + timedelta(days=365 * years_ahead)

        # Get the trade calendar
        cdef list trade_dates = self.c_trade_calendar(start_date=market_date, end_date=end_date)

        # Find the market_date in the calendar (it should be the first or near first)
        cdef size_t idx = days - 1

        if market_date == trade_dates[0]:
            idx += 1

        end_date = trade_dates[-1]
        cdef size_t n_days = len(trade_dates)
        if idx >= n_days:
            # Need to look further ahead
            return self.c_trading_days_after(market_date=end_date, days=idx - n_days + 1)

        return trade_dates[idx]

    cdef size_t c_trading_days_between(self, date start_date, date end_date):
        cdef size_t offset = 0

        if start_date < end_date:
            market_date_list = self.c_trade_calendar(start_date=start_date, end_date=end_date)
            offset = len(market_date_list)
            if market_date_list and market_date_list[-1] == end_date:
                offset -= 1

        return offset

    cdef date c_nearest_trading_date(self, date market_date, bint previous=True):
        cdef date start_date, end_date
        cdef list trade_calendar

        # If not a trading day, find nearest according to method
        if previous:
            start_date = market_date - timedelta(days=30)
            # Get previous trading day by looking back 1 month (safe upper bound)
            trade_calendar = self.c_trade_calendar(start_date=start_date, end_date=market_date)
            end_date = trade_calendar[-1]
            return end_date
        else:
            # Get next trading day by looking ahead 1 month (safe upper bound)
            end_date = market_date + timedelta(days=30)
            trade_calendar = self.trade_calendar(start_date=market_date, end_date=end_date)
            start_date = trade_calendar[0]
            return start_date

    # --- python interface ---

    def __repr__(self):
        return f'<Profile {self.profile_id}>({id(self)})'

    cpdef double time_to_seconds(self, time t, bint break_adjusted=True):
        assert self.c_time_in_market_session(t), f'{t} not in trading session'
        return self.c_time_to_seconds(t=t, break_adjusted=break_adjusted)

    cpdef double timestamp_to_seconds(self, double t, bint break_adjusted=True):
        assert self.c_timestamp_in_market_session(t), f'{t} not in trading session'
        return self.c_timestamp_to_seconds(t=t, break_adjusted=break_adjusted)

    cpdef double break_adjusted(self, double elapsed_seconds):
        return self.c_break_adjusted(elapsed_seconds=elapsed_seconds)

    cpdef double trading_time_between(self, object start_time, object end_time):
        cdef datetime st, et

        if isinstance(start_time, (float, int)):
            st = datetime.fromtimestamp(start_time, tz=self.time_zone)
            assert self.c_timestamp_in_market_session(<double> start_time), f'{start_time} not in trading session'
        else:
            st = <datetime> start_time
            assert self.c_time_in_market_session(<time> st.time()), f'{start_time} not in trading session'

        if isinstance(end_time, (float, int)):
            et = datetime.fromtimestamp(end_time, tz=self.time_zone)
            assert self.c_timestamp_in_market_session(<double> end_time), f'{end_time} not in trading session'
        else:
            et = <datetime> end_time
            assert self.c_time_in_market_session(<time> et.time()), f'{end_time} not in trading session'

        assert start_time <= end_time, f'The {end_time=} must not prior to the {start_time=}'
        cdef seconds = self.c_trading_time_between(start_time=st, end_time=et)
        return seconds

    cpdef bint is_market_session(self, object timestamp):
        cdef double ts
        cdef date d
        cdef datetime dt
        cdef time t

        if isinstance(timestamp, (float, int)):
            ts = <double> timestamp
            return self.c_timestamp_in_market_session(t=ts)

        if isinstance(timestamp, time):
            t = <time> timestamp
            return self.c_time_in_market_session(t=t)

        if isinstance(timestamp, datetime):
            dt = <datetime> timestamp
            d = dt.date()
            if self.c_date_in_market_session(d):
                t = dt.time()
                return self.c_time_in_market_session(t=t)
            else:
                return False

        raise TypeError(f'Invalid timestamp type {type(timestamp)}, except a datetime, time or numeric.')

    cpdef list trade_calendar(self, date start_date, date end_date):
        return self.c_trade_calendar(start_date=start_date, end_date=end_date)

    cpdef date trading_days_before(self, date market_date, int days):
        assert days > 0, "days must be positive"
        return self.c_trading_days_before(market_date=market_date, days=days)

    cpdef date trading_days_after(self, date market_date, int days):
        assert days > 0, "days must be positive"
        return self.c_trading_days_after(market_date=market_date, days=days)

    cpdef size_t trading_days_between(self, date start_date, date end_date):
        assert start_date <= end_date, f'The {end_date=} must not prior to the {start_date=}'
        return self.c_trading_days_between(start_date=start_date, end_date=end_date)

    cpdef date nearest_trading_date(self, date market_date, str method='previous'):
        cdef bint previous

        if method == 'previous':
            previous = True
        elif method == 'next':
            previous = False
        else:
            raise ValueError("method must be either 'previous' or 'next'")

        return self.c_nearest_trading_date(market_date=market_date, previous=previous)

    cpdef bint is_trading_day(self, date market_date):
        return self.c_date_in_market_session(market_date)

    @property
    def range_break(self) -> list[dict]:
        """
        an range break designed for plotly.
        """
        range_break = []

        if not self.session_break:
            return range_break

        # Convert session_break to range_break format
        for start, end in self.session_break:
            start_hour = start.hour + start.minute / 60
            end_hour = end.hour + end.minute / 60
            range_break.append(dict(bounds=[start_hour, end_hour], pattern="hour"))

        # Add the additional fixed non-trading periods
        if self.session_start is not None and self.session_start != time.min:
            range_break.append(
                dict(bounds=[0, time_hour(self.session_start) + time_minute(self.session_start) / 60], pattern="hour"),
            )

        if self.session_end is not None and self.session_end != time.max:
            range_break.append(
                dict(bounds=[time_hour(self.session_end) + time_minute(self.session_end) / 60, 24], pattern="hour"),
            )

        return range_break


cdef class Profile:
    def __cinit__(self):
        self.profile_id = 'non-stop'
        self.session_start = None
        self.session_end = None
        self.session_break = []
        self.time_zone = None

    @staticmethod
    cdef list c_trade_calendar(date start_date, date end_date):
        cdef size_t n_days = (end_date - start_date).days + 1
        cdef list calendar = []
        cdef size_t i

        for i in range(n_days):
            calendar.append(start_date + timedelta(days=i))

        return calendar

    @staticmethod
    cdef bint c_timestamp_in_market_session(double t):
        return True

    @staticmethod
    cdef bint c_time_in_market_session(time t):
        return True

    @staticmethod
    cdef bint c_date_in_market_session(date t):
        return True

    cdef void c_override_meta(self, ProfileDispatcher profile):
        profile.profile_id = self.profile_id
        profile.session_start = self.session_start
        profile.session_end = self.session_end
        profile.session_break.clear()
        profile.session_break.extend(self.session_break)

    cdef void c_override_func_ptr(self, ProfileDispatcher profile):
        profile.c_trade_calendar = Profile.c_trade_calendar
        profile.c_timestamp_in_market_session = Profile.c_timestamp_in_market_session
        profile.c_time_in_market_session = Profile.c_time_in_market_session
        profile.c_date_in_market_session = Profile.c_date_in_market_session

    # --- python interface ---

    cpdef ProfileDispatcher override_profile(self):
        cdef ProfileDispatcher profile = C_PROFILE
        self.c_override_meta(profile=profile)
        self.c_override_func_ptr(profile=profile)
        profile.c_refresh_cached_values()

        return C_PROFILE

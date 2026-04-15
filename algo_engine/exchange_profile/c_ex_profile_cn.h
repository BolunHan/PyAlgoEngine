#ifndef C_EX_PROFILE_CN_H
#define C_EX_PROFILE_CN_H

#include "c_ex_profile_base.h"

#ifndef EX_PROFILE_CN_PROFILE_MIN_YEAR
#define EX_PROFILE_CN_PROFILE_MIN_YEAR ((uint16_t) 1991)
#endif

#ifndef EX_PROFILE_CN_PROFILE_MAX_YEAR
#define EX_PROFILE_CN_PROFILE_MAX_YEAR ((uint16_t) 2030)
#endif

extern const session_date_t        EX_PROFILE_CN_HOLIDAYS_PRECOMPUTED[];
extern const session_date_t        EX_PROFILE_CN_HOLIDAYS_ESTIMATED[];
extern const session_date_t        EX_PROFILE_CN_CIRCUIT_BREAK_DATES[];

extern const size_t                EX_PROFILE_CN_HOLIDAYS_PRECOMPUTED_COUNT;
extern const size_t                EX_PROFILE_CN_HOLIDAYS_ESTIMATED_COUNT;
extern const size_t                EX_PROFILE_CN_CIRCUIT_BREAK_DATES_COUNT;

extern const session_time_range_t  EX_PROFILE_CN_OPENCALL_ACTIVE;
extern const session_time_range_t  EX_PROFILE_CN_OPENCALL_NO_CANCEL;
extern const session_time_range_t  EX_PROFILE_CN_OPENCALL_FROZEN;
extern const call_auction          EX_PROFILE_CN_OPENCALL_AUCTION;
extern const session_time_range_t  EX_PROFILE_CN_CLOSECALL_NO_CANCEL;
extern const call_auction          EX_PROFILE_CN_CLOSECALL_AUCTION;
extern const session_break         EX_PROFILE_CN_BREAK;

extern bool                        EX_PROFILE_CN_IS_ACTIVATED;
extern const session_date_range_t* EX_PROFILE_CN_TRADE_CALENDAR;
extern const exchange_profile      EX_PROFILE_CN;

extern bool                        c_ex_profile_cn_date_in_list(const session_date_t* date, const session_date_t* list, size_t n);
extern bool                        c_ex_profile_cn_is_holiday(const session_date_t* date);
extern bool                        c_ex_profile_cn_is_circuit_break(const session_date_t* date);
extern void                        c_ex_profile_cn_get_calendar(void);

#endif /* C_EX_PROFILE_CN_H */
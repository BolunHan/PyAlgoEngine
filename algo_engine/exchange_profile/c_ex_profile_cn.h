#ifndef C_EX_PROFILE_CN_H
#define C_EX_PROFILE_CN_H

#include <algo_engine/exchange_profile/c_ex_profile_base.h>

#ifndef EX_PROFILE_CN_PROFILE_MIN_YEAR
#define EX_PROFILE_CN_PROFILE_MIN_YEAR ((uint16_t) 1991)
#endif

#ifndef EX_PROFILE_CN_PROFILE_MAX_YEAR
#define EX_PROFILE_CN_PROFILE_MAX_YEAR ((uint16_t) 2030)
#endif

/* Data declarations follow the same Windows import/export split as the
 * profile globals in c_ex_profile_base.h: the c_ex_profile_base DLL
 * builds without EX_PROFILE_DLL_IMPORT (dllexport), consumer extensions
 * define it and link the import library. */
#if defined(_WIN32) || defined(_WIN64)
  #if defined(EX_PROFILE_DLL_IMPORT)
    #define EX_PROFILE_CN_DATA_DECL __declspec(dllimport) extern
  #else
    #define EX_PROFILE_CN_DATA_DECL __declspec(dllexport) extern
  #endif
#else
  #define EX_PROFILE_CN_DATA_DECL extern
#endif

EX_PROFILE_CN_DATA_DECL const session_date_t        EX_PROFILE_CN_HOLIDAYS_PRECOMPUTED[];
EX_PROFILE_CN_DATA_DECL const session_date_t        EX_PROFILE_CN_HOLIDAYS_ESTIMATED[];
EX_PROFILE_CN_DATA_DECL const session_date_t        EX_PROFILE_CN_CIRCUIT_BREAK_DATES[];

EX_PROFILE_CN_DATA_DECL const size_t                EX_PROFILE_CN_HOLIDAYS_PRECOMPUTED_COUNT;
EX_PROFILE_CN_DATA_DECL const size_t                EX_PROFILE_CN_HOLIDAYS_ESTIMATED_COUNT;
EX_PROFILE_CN_DATA_DECL const size_t                EX_PROFILE_CN_CIRCUIT_BREAK_DATES_COUNT;

EX_PROFILE_CN_DATA_DECL const session_time_range_t  EX_PROFILE_CN_OPENCALL_ACTIVE;
EX_PROFILE_CN_DATA_DECL const session_time_range_t  EX_PROFILE_CN_OPENCALL_NO_CANCEL;
EX_PROFILE_CN_DATA_DECL const session_time_range_t  EX_PROFILE_CN_OPENCALL_FROZEN;
EX_PROFILE_CN_DATA_DECL const call_auction          EX_PROFILE_CN_OPENCALL_AUCTION;
EX_PROFILE_CN_DATA_DECL const session_time_range_t  EX_PROFILE_CN_CLOSECALL_NO_CANCEL;
EX_PROFILE_CN_DATA_DECL const call_auction          EX_PROFILE_CN_CLOSECALL_AUCTION;
EX_PROFILE_CN_DATA_DECL const session_break         EX_PROFILE_CN_BREAK;

EX_PROFILE_CN_DATA_DECL bool                        EX_PROFILE_CN_IS_ACTIVATED;
EX_PROFILE_CN_DATA_DECL const session_date_range_t* EX_PROFILE_CN_TRADE_CALENDAR;
EX_PROFILE_CN_DATA_DECL const exchange_profile      EX_PROFILE_CN;

#undef EX_PROFILE_CN_DATA_DECL

extern bool                        c_ex_profile_cn_date_in_list(const session_date_t* date, const session_date_t* list, size_t n);
extern bool                        c_ex_profile_cn_is_holiday(const session_date_t* date);
extern bool                        c_ex_profile_cn_is_circuit_break(const session_date_t* date);
extern void                        c_ex_profile_cn_get_calendar(void);

#endif /* C_EX_PROFILE_CN_H */
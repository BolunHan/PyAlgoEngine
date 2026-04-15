#include "c_ex_profile_cn.h"

// ========== Pre-defined Calendar Events ==========

const session_date_t EX_PROFILE_CN_HOLIDAYS_PRECOMPUTED[] = {
    {.year = 1991, .month = 1, .day = 1, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 1991, .month = 2, .day = 15, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 1991, .month = 2, .day = 18, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 1991, .month = 5, .day = 1, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 1991, .month = 10, .day = 1, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 1991, .month = 10, .day = 2, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 1992, .month = 1, .day = 1, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 1992, .month = 2, .day = 4, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 1992, .month = 2, .day = 5, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 1992, .month = 2, .day = 6, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 1992, .month = 5, .day = 1, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 1992, .month = 10, .day = 1, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 1992, .month = 10, .day = 2, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 1993, .month = 1, .day = 1, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 1993, .month = 1, .day = 25, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 1993, .month = 1, .day = 26, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 1993, .month = 10, .day = 1, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 1994, .month = 2, .day = 7, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 1994, .month = 2, .day = 8, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 1994, .month = 2, .day = 9, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 1994, .month = 2, .day = 10, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 1994, .month = 2, .day = 11, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 1994, .month = 5, .day = 2, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 1994, .month = 10, .day = 3, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 1994, .month = 10, .day = 4, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 1995, .month = 1, .day = 2, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 1995, .month = 1, .day = 30, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 1995, .month = 1, .day = 31, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 1995, .month = 2, .day = 1, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 1995, .month = 2, .day = 2, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 1995, .month = 2, .day = 3, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 1995, .month = 5, .day = 1, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 1995, .month = 10, .day = 2, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 1995, .month = 10, .day = 3, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 1996, .month = 1, .day = 1, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 1996, .month = 2, .day = 19, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 1996, .month = 2, .day = 20, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 1996, .month = 2, .day = 21, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 1996, .month = 2, .day = 22, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 1996, .month = 2, .day = 23, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 1996, .month = 2, .day = 26, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 1996, .month = 2, .day = 27, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 1996, .month = 2, .day = 28, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 1996, .month = 2, .day = 29, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 1996, .month = 3, .day = 1, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 1996, .month = 5, .day = 1, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 1996, .month = 9, .day = 30, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 1996, .month = 10, .day = 1, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 1996, .month = 10, .day = 2, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 1997, .month = 1, .day = 1, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 1997, .month = 2, .day = 3, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 1997, .month = 2, .day = 4, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 1997, .month = 2, .day = 5, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 1997, .month = 2, .day = 6, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 1997, .month = 2, .day = 7, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 1997, .month = 2, .day = 10, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 1997, .month = 2, .day = 11, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 1997, .month = 2, .day = 12, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 1997, .month = 2, .day = 13, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 1997, .month = 2, .day = 14, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 1997, .month = 5, .day = 1, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 1997, .month = 5, .day = 2, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 1997, .month = 6, .day = 30, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 1997, .month = 7, .day = 1, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 1997, .month = 10, .day = 1, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 1997, .month = 10, .day = 2, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 1997, .month = 10, .day = 3, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 1998, .month = 1, .day = 1, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 1998, .month = 1, .day = 2, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 1998, .month = 1, .day = 26, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 1998, .month = 1, .day = 27, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 1998, .month = 1, .day = 28, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 1998, .month = 1, .day = 29, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 1998, .month = 1, .day = 30, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 1998, .month = 2, .day = 2, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 1998, .month = 2, .day = 3, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 1998, .month = 2, .day = 4, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 1998, .month = 2, .day = 5, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 1998, .month = 2, .day = 6, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 1998, .month = 5, .day = 1, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 1998, .month = 10, .day = 1, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 1998, .month = 10, .day = 2, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 1999, .month = 1, .day = 1, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 1999, .month = 2, .day = 10, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 1999, .month = 2, .day = 11, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 1999, .month = 2, .day = 12, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 1999, .month = 2, .day = 15, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 1999, .month = 2, .day = 16, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 1999, .month = 2, .day = 17, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 1999, .month = 2, .day = 18, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 1999, .month = 2, .day = 19, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 1999, .month = 2, .day = 22, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 1999, .month = 2, .day = 23, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 1999, .month = 2, .day = 24, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 1999, .month = 2, .day = 25, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 1999, .month = 2, .day = 26, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 1999, .month = 5, .day = 3, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 1999, .month = 10, .day = 1, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 1999, .month = 10, .day = 4, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 1999, .month = 10, .day = 5, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 1999, .month = 10, .day = 6, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 1999, .month = 10, .day = 7, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 1999, .month = 12, .day = 20, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 1999, .month = 12, .day = 31, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2000, .month = 1, .day = 3, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2000, .month = 1, .day = 31, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2000, .month = 2, .day = 1, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2000, .month = 2, .day = 2, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2000, .month = 2, .day = 3, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2000, .month = 2, .day = 4, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2000, .month = 2, .day = 7, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2000, .month = 2, .day = 8, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2000, .month = 2, .day = 9, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2000, .month = 2, .day = 10, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2000, .month = 2, .day = 11, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2000, .month = 5, .day = 1, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2000, .month = 5, .day = 2, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2000, .month = 5, .day = 3, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2000, .month = 5, .day = 4, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2000, .month = 5, .day = 5, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2000, .month = 10, .day = 2, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2000, .month = 10, .day = 3, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2000, .month = 10, .day = 4, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2000, .month = 10, .day = 5, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2000, .month = 10, .day = 6, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2001, .month = 1, .day = 1, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2001, .month = 1, .day = 22, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2001, .month = 1, .day = 23, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2001, .month = 1, .day = 24, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2001, .month = 1, .day = 25, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2001, .month = 1, .day = 26, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2001, .month = 1, .day = 29, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2001, .month = 1, .day = 30, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2001, .month = 1, .day = 31, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2001, .month = 2, .day = 1, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2001, .month = 2, .day = 2, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2001, .month = 5, .day = 1, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2001, .month = 5, .day = 2, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2001, .month = 5, .day = 3, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2001, .month = 5, .day = 4, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2001, .month = 5, .day = 7, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2001, .month = 10, .day = 1, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2001, .month = 10, .day = 2, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2001, .month = 10, .day = 3, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2001, .month = 10, .day = 4, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2001, .month = 10, .day = 5, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2002, .month = 1, .day = 1, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2002, .month = 1, .day = 2, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2002, .month = 1, .day = 3, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2002, .month = 2, .day = 11, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2002, .month = 2, .day = 12, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2002, .month = 2, .day = 13, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2002, .month = 2, .day = 14, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2002, .month = 2, .day = 15, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2002, .month = 2, .day = 18, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2002, .month = 2, .day = 19, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2002, .month = 2, .day = 20, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2002, .month = 2, .day = 21, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2002, .month = 2, .day = 22, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2002, .month = 5, .day = 1, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2002, .month = 5, .day = 2, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2002, .month = 5, .day = 3, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2002, .month = 5, .day = 6, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2002, .month = 5, .day = 7, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2002, .month = 9, .day = 30, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2002, .month = 10, .day = 1, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2002, .month = 10, .day = 2, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2002, .month = 10, .day = 3, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2002, .month = 10, .day = 4, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2002, .month = 10, .day = 7, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2003, .month = 1, .day = 1, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2003, .month = 1, .day = 30, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2003, .month = 1, .day = 31, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2003, .month = 2, .day = 3, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2003, .month = 2, .day = 4, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2003, .month = 2, .day = 5, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2003, .month = 2, .day = 6, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2003, .month = 2, .day = 7, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2003, .month = 5, .day = 1, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2003, .month = 5, .day = 2, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2003, .month = 5, .day = 5, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2003, .month = 5, .day = 6, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2003, .month = 5, .day = 7, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2003, .month = 5, .day = 8, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2003, .month = 5, .day = 9, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2003, .month = 10, .day = 1, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2003, .month = 10, .day = 2, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2003, .month = 10, .day = 3, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2003, .month = 10, .day = 6, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2003, .month = 10, .day = 7, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2004, .month = 1, .day = 1, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2004, .month = 1, .day = 19, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2004, .month = 1, .day = 20, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2004, .month = 1, .day = 21, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2004, .month = 1, .day = 22, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2004, .month = 1, .day = 23, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2004, .month = 1, .day = 26, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2004, .month = 1, .day = 27, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2004, .month = 1, .day = 28, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2004, .month = 5, .day = 3, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2004, .month = 5, .day = 4, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2004, .month = 5, .day = 5, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2004, .month = 5, .day = 6, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2004, .month = 5, .day = 7, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2004, .month = 10, .day = 1, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2004, .month = 10, .day = 4, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2004, .month = 10, .day = 5, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2004, .month = 10, .day = 6, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2004, .month = 10, .day = 7, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2005, .month = 1, .day = 3, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2005, .month = 2, .day = 7, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2005, .month = 2, .day = 8, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2005, .month = 2, .day = 9, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2005, .month = 2, .day = 10, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2005, .month = 2, .day = 11, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2005, .month = 2, .day = 14, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2005, .month = 2, .day = 15, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2005, .month = 5, .day = 2, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2005, .month = 5, .day = 3, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2005, .month = 5, .day = 4, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2005, .month = 5, .day = 5, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2005, .month = 5, .day = 6, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2005, .month = 10, .day = 3, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2005, .month = 10, .day = 4, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2005, .month = 10, .day = 5, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2005, .month = 10, .day = 6, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2005, .month = 10, .day = 7, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2006, .month = 1, .day = 2, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2006, .month = 1, .day = 3, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2006, .month = 1, .day = 26, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2006, .month = 1, .day = 27, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2006, .month = 1, .day = 30, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2006, .month = 1, .day = 31, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2006, .month = 2, .day = 1, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2006, .month = 2, .day = 2, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2006, .month = 2, .day = 3, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2006, .month = 5, .day = 1, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2006, .month = 5, .day = 2, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2006, .month = 5, .day = 3, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2006, .month = 5, .day = 4, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2006, .month = 5, .day = 5, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2006, .month = 10, .day = 2, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2006, .month = 10, .day = 3, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2006, .month = 10, .day = 4, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2006, .month = 10, .day = 5, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2006, .month = 10, .day = 6, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2007, .month = 1, .day = 1, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2007, .month = 1, .day = 2, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2007, .month = 1, .day = 3, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2007, .month = 2, .day = 19, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2007, .month = 2, .day = 20, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2007, .month = 2, .day = 21, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2007, .month = 2, .day = 22, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2007, .month = 2, .day = 23, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2007, .month = 5, .day = 1, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2007, .month = 5, .day = 2, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2007, .month = 5, .day = 3, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2007, .month = 5, .day = 4, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2007, .month = 5, .day = 7, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2007, .month = 10, .day = 1, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2007, .month = 10, .day = 2, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2007, .month = 10, .day = 3, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2007, .month = 10, .day = 4, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2007, .month = 10, .day = 5, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2007, .month = 12, .day = 31, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2008, .month = 1, .day = 1, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2008, .month = 2, .day = 6, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2008, .month = 2, .day = 7, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2008, .month = 2, .day = 8, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2008, .month = 2, .day = 11, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2008, .month = 2, .day = 12, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2008, .month = 4, .day = 4, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2008, .month = 5, .day = 1, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2008, .month = 5, .day = 2, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2008, .month = 6, .day = 9, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2008, .month = 9, .day = 15, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2008, .month = 9, .day = 29, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2008, .month = 9, .day = 30, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2008, .month = 10, .day = 1, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2008, .month = 10, .day = 2, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2008, .month = 10, .day = 3, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2009, .month = 1, .day = 1, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2009, .month = 1, .day = 2, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2009, .month = 1, .day = 26, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2009, .month = 1, .day = 27, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2009, .month = 1, .day = 28, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2009, .month = 1, .day = 29, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2009, .month = 1, .day = 30, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2009, .month = 4, .day = 6, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2009, .month = 5, .day = 1, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2009, .month = 5, .day = 28, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2009, .month = 5, .day = 29, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2009, .month = 10, .day = 1, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2009, .month = 10, .day = 2, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2009, .month = 10, .day = 5, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2009, .month = 10, .day = 6, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2009, .month = 10, .day = 7, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2009, .month = 10, .day = 8, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2010, .month = 1, .day = 1, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2010, .month = 2, .day = 15, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2010, .month = 2, .day = 16, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2010, .month = 2, .day = 17, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2010, .month = 2, .day = 18, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2010, .month = 2, .day = 19, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2010, .month = 4, .day = 5, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2010, .month = 5, .day = 3, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2010, .month = 6, .day = 14, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2010, .month = 6, .day = 15, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2010, .month = 6, .day = 16, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2010, .month = 9, .day = 22, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2010, .month = 9, .day = 23, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2010, .month = 9, .day = 24, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2010, .month = 10, .day = 1, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2010, .month = 10, .day = 4, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2010, .month = 10, .day = 5, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2010, .month = 10, .day = 6, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2010, .month = 10, .day = 7, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2011, .month = 1, .day = 3, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2011, .month = 2, .day = 2, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2011, .month = 2, .day = 3, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2011, .month = 2, .day = 4, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2011, .month = 2, .day = 7, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2011, .month = 2, .day = 8, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2011, .month = 4, .day = 4, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2011, .month = 4, .day = 5, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2011, .month = 5, .day = 2, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2011, .month = 6, .day = 6, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2011, .month = 9, .day = 12, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2011, .month = 10, .day = 3, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2011, .month = 10, .day = 4, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2011, .month = 10, .day = 5, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2011, .month = 10, .day = 6, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2011, .month = 10, .day = 7, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2012, .month = 1, .day = 2, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2012, .month = 1, .day = 3, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2012, .month = 1, .day = 23, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2012, .month = 1, .day = 24, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2012, .month = 1, .day = 25, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2012, .month = 1, .day = 26, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2012, .month = 1, .day = 27, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2012, .month = 4, .day = 2, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2012, .month = 4, .day = 3, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2012, .month = 4, .day = 4, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2012, .month = 4, .day = 30, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2012, .month = 5, .day = 1, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2012, .month = 6, .day = 22, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2012, .month = 10, .day = 1, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2012, .month = 10, .day = 2, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2012, .month = 10, .day = 3, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2012, .month = 10, .day = 4, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2012, .month = 10, .day = 5, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2013, .month = 1, .day = 1, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2013, .month = 1, .day = 2, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2013, .month = 1, .day = 3, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2013, .month = 2, .day = 11, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2013, .month = 2, .day = 12, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2013, .month = 2, .day = 13, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2013, .month = 2, .day = 14, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2013, .month = 2, .day = 15, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2013, .month = 4, .day = 4, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2013, .month = 4, .day = 5, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2013, .month = 4, .day = 29, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2013, .month = 4, .day = 30, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2013, .month = 5, .day = 1, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2013, .month = 6, .day = 10, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2013, .month = 6, .day = 11, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2013, .month = 6, .day = 12, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2013, .month = 9, .day = 19, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2013, .month = 9, .day = 20, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2013, .month = 10, .day = 1, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2013, .month = 10, .day = 2, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2013, .month = 10, .day = 3, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2013, .month = 10, .day = 4, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2013, .month = 10, .day = 7, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2014, .month = 1, .day = 1, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2014, .month = 1, .day = 31, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2014, .month = 2, .day = 3, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2014, .month = 2, .day = 4, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2014, .month = 2, .day = 5, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2014, .month = 2, .day = 6, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2014, .month = 4, .day = 7, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2014, .month = 5, .day = 1, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2014, .month = 5, .day = 2, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2014, .month = 6, .day = 2, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2014, .month = 9, .day = 8, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2014, .month = 10, .day = 1, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2014, .month = 10, .day = 2, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2014, .month = 10, .day = 3, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2014, .month = 10, .day = 6, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2014, .month = 10, .day = 7, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2015, .month = 1, .day = 1, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2015, .month = 1, .day = 2, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2015, .month = 2, .day = 18, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2015, .month = 2, .day = 19, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2015, .month = 2, .day = 20, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2015, .month = 2, .day = 23, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2015, .month = 2, .day = 24, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2015, .month = 4, .day = 6, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2015, .month = 5, .day = 1, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2015, .month = 6, .day = 22, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2015, .month = 9, .day = 3, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2015, .month = 9, .day = 4, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2015, .month = 10, .day = 1, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2015, .month = 10, .day = 2, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2015, .month = 10, .day = 5, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2015, .month = 10, .day = 6, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2015, .month = 10, .day = 7, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2016, .month = 1, .day = 1, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2016, .month = 2, .day = 8, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2016, .month = 2, .day = 9, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2016, .month = 2, .day = 10, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2016, .month = 2, .day = 11, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2016, .month = 2, .day = 12, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2016, .month = 4, .day = 4, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2016, .month = 5, .day = 2, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2016, .month = 6, .day = 9, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2016, .month = 6, .day = 10, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2016, .month = 9, .day = 15, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2016, .month = 9, .day = 16, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2016, .month = 10, .day = 3, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2016, .month = 10, .day = 4, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2016, .month = 10, .day = 5, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2016, .month = 10, .day = 6, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2016, .month = 10, .day = 7, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2017, .month = 1, .day = 2, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2017, .month = 1, .day = 27, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2017, .month = 1, .day = 30, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2017, .month = 1, .day = 31, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2017, .month = 2, .day = 1, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2017, .month = 2, .day = 2, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2017, .month = 4, .day = 3, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2017, .month = 4, .day = 4, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2017, .month = 5, .day = 1, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2017, .month = 5, .day = 29, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2017, .month = 5, .day = 30, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2017, .month = 10, .day = 2, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2017, .month = 10, .day = 3, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2017, .month = 10, .day = 4, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2017, .month = 10, .day = 5, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2017, .month = 10, .day = 6, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2018, .month = 1, .day = 1, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2018, .month = 2, .day = 15, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2018, .month = 2, .day = 16, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2018, .month = 2, .day = 19, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2018, .month = 2, .day = 20, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2018, .month = 2, .day = 21, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2018, .month = 4, .day = 5, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2018, .month = 4, .day = 6, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2018, .month = 4, .day = 30, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2018, .month = 5, .day = 1, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2018, .month = 6, .day = 18, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2018, .month = 9, .day = 24, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2018, .month = 10, .day = 1, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2018, .month = 10, .day = 2, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2018, .month = 10, .day = 3, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2018, .month = 10, .day = 4, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2018, .month = 10, .day = 5, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2018, .month = 12, .day = 31, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2019, .month = 1, .day = 1, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2019, .month = 2, .day = 4, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2019, .month = 2, .day = 5, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2019, .month = 2, .day = 6, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2019, .month = 2, .day = 7, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2019, .month = 2, .day = 8, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2019, .month = 4, .day = 5, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2019, .month = 5, .day = 1, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2019, .month = 5, .day = 2, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2019, .month = 5, .day = 3, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2019, .month = 6, .day = 7, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2019, .month = 9, .day = 13, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2019, .month = 10, .day = 1, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2019, .month = 10, .day = 2, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2019, .month = 10, .day = 3, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2019, .month = 10, .day = 4, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2019, .month = 10, .day = 7, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2020, .month = 1, .day = 1, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2020, .month = 1, .day = 24, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2020, .month = 1, .day = 27, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2020, .month = 1, .day = 28, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2020, .month = 1, .day = 29, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2020, .month = 1, .day = 30, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2020, .month = 1, .day = 31, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2020, .month = 4, .day = 6, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2020, .month = 5, .day = 1, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2020, .month = 5, .day = 4, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2020, .month = 5, .day = 5, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2020, .month = 6, .day = 25, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2020, .month = 6, .day = 26, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2020, .month = 10, .day = 1, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2020, .month = 10, .day = 2, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2020, .month = 10, .day = 5, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2020, .month = 10, .day = 6, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2020, .month = 10, .day = 7, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2020, .month = 10, .day = 8, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2021, .month = 1, .day = 1, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2021, .month = 2, .day = 11, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2021, .month = 2, .day = 12, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2021, .month = 2, .day = 15, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2021, .month = 2, .day = 16, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2021, .month = 2, .day = 17, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2021, .month = 4, .day = 5, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2021, .month = 5, .day = 3, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2021, .month = 5, .day = 4, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2021, .month = 5, .day = 5, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2021, .month = 6, .day = 14, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2021, .month = 9, .day = 20, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2021, .month = 9, .day = 21, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2021, .month = 10, .day = 1, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2021, .month = 10, .day = 4, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2021, .month = 10, .day = 5, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2021, .month = 10, .day = 6, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2021, .month = 10, .day = 7, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2022, .month = 1, .day = 3, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2022, .month = 1, .day = 31, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2022, .month = 2, .day = 1, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2022, .month = 2, .day = 2, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2022, .month = 2, .day = 3, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2022, .month = 2, .day = 4, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2022, .month = 4, .day = 4, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2022, .month = 4, .day = 5, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2022, .month = 5, .day = 2, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2022, .month = 5, .day = 3, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2022, .month = 5, .day = 4, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2022, .month = 6, .day = 3, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2022, .month = 9, .day = 12, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2022, .month = 10, .day = 3, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2022, .month = 10, .day = 4, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2022, .month = 10, .day = 5, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2022, .month = 10, .day = 6, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2022, .month = 10, .day = 7, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2023, .month = 1, .day = 2, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2023, .month = 1, .day = 23, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2023, .month = 1, .day = 24, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2023, .month = 1, .day = 25, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2023, .month = 1, .day = 26, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2023, .month = 1, .day = 27, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2023, .month = 4, .day = 5, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2023, .month = 5, .day = 1, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2023, .month = 5, .day = 2, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2023, .month = 5, .day = 3, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2023, .month = 6, .day = 22, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2023, .month = 6, .day = 23, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2023, .month = 9, .day = 29, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2023, .month = 10, .day = 2, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2023, .month = 10, .day = 3, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2023, .month = 10, .day = 4, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2023, .month = 10, .day = 5, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2023, .month = 10, .day = 6, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2024, .month = 1, .day = 1, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2024, .month = 2, .day = 9, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2024, .month = 2, .day = 12, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2024, .month = 2, .day = 13, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2024, .month = 2, .day = 14, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2024, .month = 2, .day = 15, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2024, .month = 2, .day = 16, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2024, .month = 4, .day = 4, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2024, .month = 4, .day = 5, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2024, .month = 5, .day = 1, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2024, .month = 5, .day = 2, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2024, .month = 5, .day = 3, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2024, .month = 6, .day = 10, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2024, .month = 9, .day = 16, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2024, .month = 9, .day = 17, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2024, .month = 10, .day = 1, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2024, .month = 10, .day = 2, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2024, .month = 10, .day = 3, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2024, .month = 10, .day = 4, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2024, .month = 10, .day = 7, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2025, .month = 1, .day = 1, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2025, .month = 1, .day = 28, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2025, .month = 1, .day = 29, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2025, .month = 1, .day = 30, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2025, .month = 1, .day = 31, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2025, .month = 2, .day = 3, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2025, .month = 2, .day = 4, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2025, .month = 4, .day = 4, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2025, .month = 5, .day = 1, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2025, .month = 5, .day = 2, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2025, .month = 5, .day = 5, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2025, .month = 6, .day = 2, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2025, .month = 10, .day = 1, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2025, .month = 10, .day = 2, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2025, .month = 10, .day = 3, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2025, .month = 10, .day = 6, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2025, .month = 10, .day = 7, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2025, .month = 10, .day = 8, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2026, .month = 1, .day = 1, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2026, .month = 1, .day = 2, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2026, .month = 2, .day = 16, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2026, .month = 2, .day = 17, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2026, .month = 2, .day = 18, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2026, .month = 2, .day = 19, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2026, .month = 2, .day = 20, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2026, .month = 2, .day = 23, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2026, .month = 4, .day = 6, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2026, .month = 5, .day = 1, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2026, .month = 5, .day = 4, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2026, .month = 5, .day = 5, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2026, .month = 6, .day = 19, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2026, .month = 9, .day = 25, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2026, .month = 10, .day = 1, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2026, .month = 10, .day = 2, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2026, .month = 10, .day = 5, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2026, .month = 10, .day = 6, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2026, .month = 10, .day = 7, .stype = SESSION_TYPE_NON_TRADING}
};

const session_date_t EX_PROFILE_CN_HOLIDAYS_ESTIMATED[] = {
    {.year = 2027, .month = 1, .day = 1, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2027, .month = 2, .day = 8, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2027, .month = 2, .day = 9, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2027, .month = 2, .day = 10, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2027, .month = 2, .day = 11, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2027, .month = 2, .day = 12, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2027, .month = 4, .day = 5, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2027, .month = 5, .day = 3, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2027, .month = 5, .day = 4, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2027, .month = 5, .day = 5, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2027, .month = 6, .day = 9, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2027, .month = 6, .day = 10, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2027, .month = 6, .day = 11, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2027, .month = 9, .day = 15, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2027, .month = 9, .day = 16, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2027, .month = 9, .day = 17, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2027, .month = 10, .day = 1, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2027, .month = 10, .day = 4, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2027, .month = 10, .day = 5, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2027, .month = 10, .day = 6, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2027, .month = 10, .day = 7, .stype = SESSION_TYPE_NON_TRADING},

    {.year = 2028, .month = 1, .day = 26, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2028, .month = 1, .day = 27, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2028, .month = 1, .day = 28, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2028, .month = 1, .day = 31, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2028, .month = 2, .day = 1, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2028, .month = 4, .day = 3, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2028, .month = 4, .day = 4, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2028, .month = 4, .day = 5, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2028, .month = 5, .day = 1, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2028, .month = 5, .day = 2, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2028, .month = 5, .day = 3, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2028, .month = 5, .day = 4, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2028, .month = 5, .day = 5, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2028, .month = 5, .day = 29, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2028, .month = 10, .day = 2, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2028, .month = 10, .day = 3, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2028, .month = 10, .day = 4, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2028, .month = 10, .day = 2, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2028, .month = 10, .day = 3, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2028, .month = 10, .day = 4, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2028, .month = 10, .day = 5, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2028, .month = 10, .day = 6, .stype = SESSION_TYPE_NON_TRADING},

    {.year = 2029, .month = 1, .day = 1, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2029, .month = 1, .day = 2, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2029, .month = 1, .day = 3, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2029, .month = 2, .day = 13, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2029, .month = 2, .day = 14, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2029, .month = 2, .day = 15, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2029, .month = 2, .day = 16, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2029, .month = 2, .day = 19, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2029, .month = 4, .day = 3, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2029, .month = 4, .day = 4, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2029, .month = 4, .day = 5, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2029, .month = 5, .day = 1, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2029, .month = 5, .day = 2, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2029, .month = 5, .day = 3, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2029, .month = 6, .day = 15, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2029, .month = 9, .day = 21, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2029, .month = 10, .day = 1, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2029, .month = 10, .day = 2, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2029, .month = 10, .day = 3, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2029, .month = 10, .day = 4, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2029, .month = 10, .day = 5, .stype = SESSION_TYPE_NON_TRADING},

    {.year = 2030, .month = 1, .day = 1, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2030, .month = 1, .day = 2, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2030, .month = 1, .day = 3, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2030, .month = 2, .day = 4, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2030, .month = 2, .day = 5, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2030, .month = 2, .day = 6, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2030, .month = 2, .day = 7, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2030, .month = 2, .day = 8, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2030, .month = 4, .day = 5, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2030, .month = 5, .day = 1, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2030, .month = 5, .day = 2, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2030, .month = 5, .day = 3, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2030, .month = 6, .day = 5, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2030, .month = 6, .day = 6, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2030, .month = 6, .day = 7, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2030, .month = 9, .day = 12, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2030, .month = 9, .day = 13, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2030, .month = 10, .day = 1, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2030, .month = 10, .day = 2, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2030, .month = 10, .day = 3, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2030, .month = 10, .day = 4, .stype = SESSION_TYPE_NON_TRADING},
    {.year = 2030, .month = 10, .day = 7, .stype = SESSION_TYPE_NON_TRADING}
};

const session_date_t EX_PROFILE_CN_CIRCUIT_BREAK_DATES[] = {
    {.year = 2016, .month = 1, .day = 4, .stype = SESSION_TYPE_CIRCUIT_BREAK},
    {.year = 2016, .month = 1, .day = 7, .stype = SESSION_TYPE_CIRCUIT_BREAK},
};

const size_t EX_PROFILE_CN_HOLIDAYS_PRECOMPUTED_COUNT = sizeof(EX_PROFILE_CN_HOLIDAYS_PRECOMPUTED) / sizeof(session_date_t);
const size_t EX_PROFILE_CN_HOLIDAYS_ESTIMATED_COUNT = sizeof(EX_PROFILE_CN_HOLIDAYS_ESTIMATED) / sizeof(session_date_t);
const size_t EX_PROFILE_CN_CIRCUIT_BREAK_DATES_COUNT = sizeof(EX_PROFILE_CN_CIRCUIT_BREAK_DATES) / sizeof(session_date_t);

// ========== Timestamp Constants ==========

static const double        EX_PROFILE_CN_TS_OPENCALL_START = HMS_TO_TS(9u, 15u, 0u);      // 09:15:00
static const double        EX_PROFILE_CN_TS_OPENCALL_NO_CANCEL = HMS_TO_TS(9u, 20u, 0u);  // 09:20:00
static const double        EX_PROFILE_CN_TS_OPENCALL_UNCROSS = HMS_TO_TS(9u, 25u, 0u);    // 09:25:00 (uncross)
static const double        EX_PROFILE_CN_TS_SESSION_START = HMS_TO_TS(9u, 30u, 0u);       // 09:30:00
static const double        EX_PROFILE_CN_TS_BREAK_START = HMS_TO_TS(11u, 30u, 0u);        // 11:30:00
static const double        EX_PROFILE_CN_TS_BREAK_END = HMS_TO_TS(13u, 0u, 0u);           // 13:00:00
static const double        EX_PROFILE_CN_TS_CLOSECALL_START = HMS_TO_TS(14u, 57u, 0u);    // 14:57:00
static const double        EX_PROFILE_CN_TS_SESSION_END = HMS_TO_TS(15u, 0u, 0u);         // 15:00:00

const session_time_range_t EX_PROFILE_CN_OPENCALL_ACTIVE = {
    .start = {.elapsed_seconds = -15.0 * SECONDS_PER_MINUTE, .hour = 9u, .minute = 15u, .second = 0u, .nanosecond = 0u, .phase = SESSION_PHASE_OPEN_AUCTION},
    .end = {.elapsed_seconds = -10.0 * SECONDS_PER_MINUTE, .hour = 9u, .minute = 20u, .second = 0u, .nanosecond = 0u, .phase = SESSION_PHASE_OPEN_AUCTION},
    .elapsed_seconds = 5 * SECONDS_PER_MINUTE,
};

const session_time_range_t EX_PROFILE_CN_OPENCALL_NO_CANCEL = {
    .start = {.elapsed_seconds = -10.0 * SECONDS_PER_MINUTE, .hour = 9u, .minute = 20u, .second = 0u, .nanosecond = 0u, .phase = SESSION_PHASE_OPEN_AUCTION},
    .end = {.elapsed_seconds = -5.0 * SECONDS_PER_MINUTE, .hour = 9u, .minute = 25u, .second = 0u, .nanosecond = 0u, .phase = SESSION_PHASE_OPEN_AUCTION},
    .elapsed_seconds = 5 * SECONDS_PER_MINUTE,
};

const session_time_range_t EX_PROFILE_CN_OPENCALL_FROZEN = {
    .start = {.elapsed_seconds = -5.0 * SECONDS_PER_MINUTE, .hour = 9u, .minute = 25u, .second = 0u, .nanosecond = 0u, .phase = SESSION_PHASE_OPEN_AUCTION},
    .end = {.elapsed_seconds = 0, .hour = 9u, .minute = 30u, .second = 0u, .nanosecond = 0u, .phase = SESSION_PHASE_OPEN_AUCTION},
    .elapsed_seconds = 5 * SECONDS_PER_MINUTE,
};

const call_auction EX_PROFILE_CN_OPENCALL_AUCTION = {
    .auction_start = {.elapsed_seconds = -15.0 * SECONDS_PER_MINUTE, .hour = 9u, .minute = 15u, .second = 0u, .nanosecond = 0u, .phase = SESSION_PHASE_OPEN_AUCTION},
    .active = &EX_PROFILE_CN_OPENCALL_ACTIVE,
    .no_cancel = &EX_PROFILE_CN_OPENCALL_NO_CANCEL,
    .frozen = &EX_PROFILE_CN_OPENCALL_FROZEN,
    .uncross = {.elapsed_seconds = -5.0 * SECONDS_PER_MINUTE, .hour = 9u, .minute = 25u, .second = 0u, .nanosecond = 0u, .phase = SESSION_PHASE_OPEN_AUCTION},
    .auction_end = {.elapsed_seconds = 0.0, .hour = 9u, .minute = 30u, .second = 0u, .nanosecond = 0u, .phase = SESSION_PHASE_OPEN_AUCTION},
};

const session_time_range_t EX_PROFILE_CN_CLOSECALL_NO_CANCEL = {
    .start = {.elapsed_seconds = 14220.0, .hour = 14u, .minute = 57u, .second = 0u, .nanosecond = 0u, .phase = SESSION_PHASE_CLOSE_AUCTION},
    .end = {.elapsed_seconds = 14400.0, .hour = 15u, .minute = 0u, .second = 0u, .nanosecond = 0u, .phase = SESSION_PHASE_CLOSE_AUCTION},
    .elapsed_seconds = 180.0,
};

const call_auction EX_PROFILE_CN_CLOSECALL_AUCTION = {
    .auction_start = {.elapsed_seconds = 14220.0, .hour = 14u, .minute = 57u, .second = 0u, .nanosecond = 0u, .phase = SESSION_PHASE_CLOSE_AUCTION},
    .active = NULL,
    .no_cancel = &EX_PROFILE_CN_CLOSECALL_NO_CANCEL,
    .frozen = NULL,
    .uncross = {.elapsed_seconds = 14400.0, .hour = 15u, .minute = 0u, .second = 0u, .nanosecond = 0u, .phase = SESSION_PHASE_CLOSE_AUCTION},
    .auction_end = {.elapsed_seconds = 14400.0, .hour = 15u, .minute = 0u, .second = 0u, .nanosecond = 0u, .phase = SESSION_PHASE_CLOSE_AUCTION},
};

const session_break EX_PROFILE_CN_BREAK = {
    .break_start = {.elapsed_seconds = 7200.0, .hour = 11u, .minute = 30u, .second = 0u, .nanosecond = 0u, .phase = SESSION_PHASE_BREAK},
    .break_end = {.elapsed_seconds = 7200.0, .hour = 13u, .minute = 0u, .second = 0u, .nanosecond = 0u, .phase = SESSION_PHASE_BREAK},
    .break_start_ts = EX_PROFILE_CN_TS_BREAK_START,  // 11:30:00
    .break_end_ts = EX_PROFILE_CN_TS_BREAK_END,      // 13:00:00
    .break_length_seconds = EX_PROFILE_CN_TS_BREAK_END - EX_PROFILE_CN_TS_BREAK_START,
    .next = NULL,
};

// ========== Forward Declaration ==========

static void                  c_ex_profile_on_activate_cn(const exchange_profile* profile);
static void                  c_ex_profile_on_deactivate_cn(const exchange_profile* profile);
static session_date_range_t* c_ex_profile_trade_calendar_cn(const session_date_t* start_date, const session_date_t* end_date);
static auction_phase         c_ex_profile_resolve_auction_phase_cn(double ts);
static session_phase         c_ex_profile_resolve_session_phase_cn(double ts);
static session_type          c_ex_profile_resolve_session_type_cn(uint16_t year, uint8_t month, uint8_t day);

// ========== Utilities Functions ==========

bool c_ex_profile_cn_date_in_list(const session_date_t* date, const session_date_t* list, size_t n) {
    if (!date || !list) return false;
    for (size_t i = 0; i < n; ++i)
        if (c_ex_profile_date_compare(date, list + i) == 0) return true;
    return false;
}

bool c_ex_profile_cn_is_holiday(const session_date_t* date) {
    if (c_ex_profile_cn_date_in_list(date, EX_PROFILE_CN_HOLIDAYS_PRECOMPUTED, EX_PROFILE_CN_HOLIDAYS_PRECOMPUTED_COUNT)) return true;
    if (c_ex_profile_cn_date_in_list(date, EX_PROFILE_CN_HOLIDAYS_ESTIMATED, EX_PROFILE_CN_HOLIDAYS_ESTIMATED_COUNT)) return true;
    return false;
}

bool c_ex_profile_cn_is_circuit_break(const session_date_t* date) {
    return c_ex_profile_cn_date_in_list(date, EX_PROFILE_CN_CIRCUIT_BREAK_DATES, EX_PROFILE_CN_CIRCUIT_BREAK_DATES_COUNT);
}

void c_ex_profile_cn_get_calendar(void) {
    if (EX_PROFILE_CN_TRADE_CALENDAR) return;
    session_date_t        start = {.year = EX_PROFILE_CN_PROFILE_MIN_YEAR, .month = 1u, .day = 1u, .stype = SESSION_TYPE_NON_TRADING};
    session_date_t        end = {.year = EX_PROFILE_CN_PROFILE_MAX_YEAR, .month = 12u, .day = 31u, .stype = SESSION_TYPE_NON_TRADING};
    session_date_range_t* built = c_ex_profile_trade_calendar_cn(&start, &end);
    if (!built) return;
    EX_PROFILE_CN_TRADE_CALENDAR = (const session_date_range_t*) built;
}

// ========== Public APIs ==========

static void c_ex_profile_on_activate_cn(const exchange_profile* profile) {
    if (!profile) return;
    (void) profile;

    if (EX_TRADE_CALENDAR_CACHE) free((void*) EX_TRADE_CALENDAR_CACHE);
    if (!EX_PROFILE_CN_TRADE_CALENDAR) c_ex_profile_cn_get_calendar();
    EX_TRADE_CALENDAR_CACHE = EX_PROFILE_CN_TRADE_CALENDAR;
    EX_PROFILE_CN_IS_ACTIVATED = true;
}

static void c_ex_profile_on_deactivate_cn(const exchange_profile* profile) {
    if (!profile) return;
    (void) profile;

    // To prevent EX_PROFILE_CN_TRADE_CALENDAR being freed by the default on_deactivate
    // We must unset the EX_TRADE_CALENDAR_CACHE
    EX_TRADE_CALENDAR_CACHE = NULL;
    if (EX_TRADE_CALENDAR_CACHE == EX_PROFILE_CN_TRADE_CALENDAR) EX_TRADE_CALENDAR_CACHE = NULL;
    EX_PROFILE_CN_IS_ACTIVATED = false;
}

static session_date_range_t* c_ex_profile_trade_calendar_cn(const session_date_t* start_date, const session_date_t* end_date) {
    if (!start_date || !end_date) return NULL;
    if (!c_ex_profile_date_is_valid(start_date) || !c_ex_profile_date_is_valid(end_date)) return NULL;

    // Normalize order.
    if (c_ex_profile_date_compare(start_date, end_date) == 1) {
        const session_date_t* tmp = start_date;
        start_date = end_date;
        end_date = tmp;
    }

    session_date_range_t* calendar = c_ex_profile_date_range(start_date, end_date);
    if (!calendar || calendar->n_days == 0u) return NULL;

    // Mark stype, then compact in-place with two cursors (read/write).
    const size_t n_total = calendar->n_days;
    size_t       write_i = 0u;
    for (size_t read_i = 0u; read_i < n_total; ++read_i) {
        session_date_t* d = calendar->dates + read_i;

        if (c_ex_profile_is_weekend(d) || c_ex_profile_cn_is_holiday(d)) continue;

        session_type stype = c_ex_profile_cn_is_circuit_break(d) ? SESSION_TYPE_CIRCUIT_BREAK : SESSION_TYPE_NORMINAL;
        if (write_i != read_i) calendar->dates[write_i] = *d;
        calendar->dates[write_i].stype = stype;
        write_i += 1u;
    }

    if (write_i == 0u) {
        free(calendar);
        return NULL;
    }

    calendar->n_days = write_i;
    calendar->start = calendar->dates[0u];
    calendar->end = calendar->dates[write_i - 1u];
    return calendar;
}

static auction_phase c_ex_profile_resolve_auction_phase_cn(double ts) {
    // OPEN auction: 09:15:00 ~ 09:30:00
    // 09:15:00 ~ 09:20:00: AUCTION_PHASE_ACTIVE
    // 09:20:00 ~ 09:25:00: AUCTION_PHASE_NO_CANCEL
    // 09:25:00: uncrossing
    // 09:25:00 ~ 09:30:00: AUCTION_PHASE_FROZEN
    // 09:30:00: AUCTION_PHASE_DONE

    // CLOSE auction:
    // 14:57:00 ~ 15:00:00: AUCTION_PHASE_NO_CANCEL
    // 15:00:00: uncrossing and AUCTION_PHASE_DONE

    if (ts < EX_PROFILE_CN_TS_OPENCALL_START) return AUCTION_PHASE_DONE;
    if (ts < EX_PROFILE_CN_TS_OPENCALL_NO_CANCEL) return AUCTION_PHASE_ACTIVE;
    if (ts < EX_PROFILE_CN_TS_OPENCALL_UNCROSS) return AUCTION_PHASE_NO_CANCEL;
    if (ts == EX_PROFILE_CN_TS_OPENCALL_UNCROSS) return AUCTION_PHASE_UNCROSSING;
    if (ts < EX_PROFILE_CN_TS_SESSION_START) return AUCTION_PHASE_FROZEN;
    if (ts < EX_PROFILE_CN_TS_CLOSECALL_START) return AUCTION_PHASE_DONE;
    if (ts < EX_PROFILE_CN_TS_SESSION_END) return AUCTION_PHASE_NO_CANCEL;
    if (ts == EX_PROFILE_CN_TS_SESSION_END) return AUCTION_PHASE_UNCROSSING;
    return AUCTION_PHASE_DONE;
}

static session_phase c_ex_profile_resolve_session_phase_cn(double ts) {
    // prier to 09:15:00: SESSION_PHASE_PREOPEN
    // 09:15:00 ~ 09:30:00: SESSION_PHASE_OPEN_AUCTION
    // 09:30:00 ~ 11:30:00: SESSION_PHASE_CONTINUOUS
    // 11:30:00 ~ 13:00:00: SESSION_PHASE_BREAK
    // 13:00:00 ~ 14:57:00: SESSION_PHASE_CONTINUOUS
    // 14:57:00 ~ 15:00:00: SESSION_PHASE_CLOSE_AUCTION
    // after 15:00:00: SESSION_PHASE_CLOSED

    if (ts < EX_PROFILE_CN_TS_OPENCALL_START) return SESSION_PHASE_PREOPEN;
    if (ts < EX_PROFILE_CN_TS_SESSION_START) return SESSION_PHASE_OPEN_AUCTION;
    if (ts <= EX_PROFILE_CN_TS_BREAK_START) return SESSION_PHASE_CONTINUOUS;
    if (ts < EX_PROFILE_CN_TS_BREAK_END) return SESSION_PHASE_BREAK;
    if (ts <= EX_PROFILE_CN_TS_CLOSECALL_START) return SESSION_PHASE_CONTINUOUS;
    if (ts < EX_PROFILE_CN_TS_SESSION_END) return SESSION_PHASE_CLOSE_AUCTION;
    return SESSION_PHASE_CLOSED;
}

static session_type c_ex_profile_resolve_session_type_cn(uint16_t year, uint8_t month, uint8_t day) {
    if (year < EX_PROFILE_CN_PROFILE_MIN_YEAR || year > EX_PROFILE_CN_PROFILE_MAX_YEAR) return SESSION_TYPE_NON_TRADING;
    session_date_t target = {.year = year, .month = month, .day = day, .stype = SESSION_TYPE_NORMINAL};
    size_t         idx = c_ex_profile_session_date_index(&target, EX_PROFILE_CN_TRADE_CALENDAR);
    if (idx == (size_t) -1) return SESSION_TYPE_NON_TRADING;
    const session_date_t* date = EX_PROFILE_CN_TRADE_CALENDAR->dates + idx;
    if (c_ex_profile_date_compare(date, &target) == 0) return date->stype;
    return SESSION_TYPE_NON_TRADING;
}

bool                        EX_PROFILE_CN_IS_ACTIVATED = false;

const session_date_range_t* EX_PROFILE_CN_TRADE_CALENDAR = NULL;

const exchange_profile      EX_PROFILE_CN = {
    .profile_id = "CN_STOCK",

    .session_start = {
        .elapsed_seconds = 0.0,
        .hour = 9u,
        .minute = 30u,
        .second = 0u,
        .nanosecond = 0u,
        .phase = SESSION_PHASE_CONTINUOUS,
    },
    .session_end = {
        .elapsed_seconds = 14400.0,
        .hour = 15u,
        .minute = 0u,
        .second = 0u,
        .nanosecond = 0u,
        .phase = SESSION_PHASE_CLOSED,
    },
    .session_start_ts = EX_PROFILE_CN_TS_SESSION_START,  // 09:30:00
    .session_end_ts = EX_PROFILE_CN_TS_SESSION_END,      // 15:00:00
    .session_length_seconds = 14400.0,                   // 4h continuous trading excluding lunch break

    .open_call_auction = &EX_PROFILE_CN_OPENCALL_AUCTION,
    .close_call_auction = &EX_PROFILE_CN_CLOSECALL_AUCTION,
    .session_breaks = &EX_PROFILE_CN_BREAK,

    .time_zone = "Asia/Shanghai",
    .tz_offset_seconds = 28800.0,

    .on_activate = c_ex_profile_on_activate_cn,
    .on_deactivate = c_ex_profile_on_deactivate_cn,
    .trade_calendar = c_ex_profile_trade_calendar_cn,
    .resolve_auction_phase = c_ex_profile_resolve_auction_phase_cn,
    .resolve_session_phase = c_ex_profile_resolve_session_phase_cn,
    .resolve_session_type = c_ex_profile_resolve_session_type_cn,
};
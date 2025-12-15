#ifndef C_MARKET_DATA_CONFIG_H
#define C_MARKET_DATA_CONFIG_H

#ifndef DEBUG
#define DEBUG 0
#endif

#ifndef TICKER_SIZE
#define TICKER_SIZE 32
#endif

#ifndef BOOK_SIZE
#define BOOK_SIZE 10
#endif

#ifndef ID_SIZE
#define ID_SIZE 16
#endif

#ifndef LONG_ID_SIZE
#define LONG_ID_SIZE 128
#endif

#ifndef MAX_WORKERS
#define MAX_WORKERS 128
#endif

static const int MID_ALLOW_INT64 = ID_SIZE >= 7;
static const int MID_ALLOW_INT128 = ID_SIZE >= 15;
static const int LONG_MID_ALLOW_INT64 = LONG_ID_SIZE >= 7;
static const int LONG_MID_ALLOW_INT128 = LONG_ID_SIZE >= 15;

#endif // C_MARKET_DATA_CONFIG_H

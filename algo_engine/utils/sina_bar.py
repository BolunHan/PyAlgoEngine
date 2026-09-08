"""
Sina Finance data fetcher: minute bars and real-time snapshots for A-share stocks
and indexes, plus monthly index constituent weights from CSIndex (中证指数官网).

All functions accept Wind-style (``600519.SH``), bare (``600519``), or prefixed
(``sh600519``) tickers; see each docstring for details and limitations.
"""
import datetime
import io
import requests
import pandas as pd

from typing import NamedTuple
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from algo_engine.base import BarData, TickData

# Retrying session: read/connect timeouts and transient 5xx are retried with backoff
# (CSIndex's oss-ch CDN is occasionally slow, e.g. ReadTimeout on the weight file).
_SESSION = requests.Session()
_SESSION.mount(
    "https://",
    HTTPAdapter(max_retries=Retry(
        total=3, connect=3, read=3,
        backoff_factor=1.0,
        status_forcelist=(500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
    )),
)

_BAR_URL = "https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData"
_HQ_URL = "https://hq.sinajs.cn/list="
_HQ_HEADERS = {"Referer": "https://finance.sina.com.cn"}  # required since ~2021, returns 403 without it
_CSINDEX_WEIGHT_URL = "https://oss-ch.csindex.com.cn/static/html/csindex/public/uploads/file/autofile/closeweight/{code}closeweight.xls"
_MAX_DATALEN = 1023  # Sina free API cap; scale=5 covers ~1 month of trading days
_MAX_SYMBOLS_PER_REQUEST = 80  # hq.sinajs.cn batch limit
_BEIJING_TZ = datetime.timezone(datetime.timedelta(hours=8))
_VALID_SCALES = (5, 15, 30, 60, 240)


def _parse_ticker(ticker: str, exchange: str | None = None) -> tuple[str, str]:
    """Return ``(code, exchange_suffix)`` for a Wind-style, bare, or prefixed ticker."""
    t = ticker.lower()
    if "." in t:  # Wind-style: "600519.SH"
        code, suffix = t.split(".", 1)
        if code.isdigit() and suffix in ("sh", "sz", "bj"):
            return code, suffix
        raise ValueError(f"Cannot parse Wind-style ticker {ticker!r}; expected e.g. '600519.SH'.")
    if t.startswith(("sh", "sz", "bj")):
        return t[2:], t[:2]
    if exchange:
        return ticker, exchange.lower()
    if ticker[0] in "569":
        return ticker, "sh"
    if ticker[0] in "0123" or ticker.startswith("399"):
        return ticker, "sz"
    if ticker[0] in "48" or ticker.startswith("92"):
        return ticker, "bj"
    raise ValueError(f"Cannot infer exchange prefix for ticker {ticker!r}; pass exchange='sh'/'sz'/'bj'.")


def normalize_symbol(ticker: str, exchange: str | None = None) -> str:
    """
    Prepend the exchange prefix that Sina's API requires (e.g. ``000016`` -> ``sh000016``).

    Args:
        ticker (str): Raw ticker, Wind-style (``600519.SH``, ``000016.SH``, ``000001.SZ``,
            ``430047.BJ``), or one already prefixed (``sh600519``).
        exchange (str | None): Explicit exchange (``sh``/``sz``/``bj``). Auto-detected when
            ``None``; required to disambiguate ``000xxx`` which is a Shanghai index
            (e.g. ``000016`` 上证50) or a Shenzhen stock (e.g. ``000001`` 平安银行).

    Returns:
        str: Symbol with exchange prefix.
    """
    code, suffix = _parse_ticker(ticker, exchange)
    return f"{suffix}{code}"


def get_minute_bars(
        ticker: str,
        market_date: datetime.date,
        *,
        exchange: str | None = None,
        scale: int = 5
) -> list[BarData]:
    """
    Fetch minute-level bar data for a given ticker and market date from Sina Finance.

    Args:
        ticker (str): The stock/index ticker symbol, e.g. ``600519``, ``600519.SH``,
            ``000016``, or ``sh600519``. Pass ``exchange`` for ambiguous ``000xxx``.
        market_date (datetime.date): The market date for which to fetch the bar data.
        exchange (str | None): Explicit exchange (``sh``/``sz``/``bj``) for the ticker.
        scale (int): Bar span in minutes, one of 5/15/30/60/240. Default 5.

    Returns:
        list[BarData]: Minute-level bars for ``market_date``. One bar per 5-minute slot
            (48 per trading day), in chronological order.

    Raises:
        ValueError: If the symbol/scale is invalid, or the date is older than the API's
            lookback window (datalen=1023, roughly one month of trading days).
    """
    if scale not in _VALID_SCALES:
        raise ValueError(f"scale must be one of {_VALID_SCALES}, got {scale}")
    symbol = normalize_symbol(ticker, exchange)

    # Size the fetch to cover the requested date: ~48 bars per trading day, 5/7 of
    # calendar days are trading days. The API returns the most recent N bars, cutting
    # from the start of the oldest day in the window, so one extra full day (48 bars)
    # of buffer guarantees the requested date is not truncated. Clamp to the API cap.
    span_days = (datetime.date.today() - market_date).days
    datalen = min(_MAX_DATALEN, max(48, span_days * 48 * 5 // 7 + 96))

    response = _SESSION.get(
        _BAR_URL,
        params={"symbol": symbol, "scale": scale, "ma": "no", "datalen": datalen},
        timeout=10,
    )
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, list) or not data:
        raise ValueError(f"Sina returned no data for {symbol} (scale={scale}): check the symbol and exchange prefix.")

    bars = []
    for item in data:
        bar_time = datetime.datetime.strptime(item['day'], '%Y-%m-%d %H:%M:%S').replace(tzinfo=_BEIJING_TZ)
        if bar_time.date() == market_date:
            bars.append(
                BarData(
                    ticker=ticker,
                    timestamp=bar_time.timestamp(),
                    open_price=float(item['open']),
                    high_price=float(item['high']),
                    low_price=float(item['low']),
                    close_price=float(item['close']),
                    volume=float(item['volume']),
                    bar_span=scale * 60,
                )
            )

    if not bars:
        raise ValueError(
            f"No {scale}-minute bars for {symbol} on {market_date}: Sina free API only covers "
            f"~{len(data) // 48} trading days back (oldest bar: {data[0]['day']})."
        )
    return bars


def get_realtime_ticks(tickers: str | list[str]) -> list[TickData]:
    """
    Fetch real-time market snapshots from Sina's quote API (hq.sinajs.cn).
    This is the endpoint trading clients poll every ~3 seconds for live quotes.

    Args:
        tickers (str | list[str]): One ticker or a list of tickers, Wind-style
            (``600519.SH``, ``000016.SH``) or bare (``600519``, ``000016``). At most
            ``_MAX_SYMBOLS_PER_REQUEST`` symbols are fetched per HTTP request.

    Returns:
        list[TickData]: One snapshot per requested ticker, in input order. Bid/ask
            books carry up to 5 price levels (zero levels skipped; indexes have none).

    Raises:
        ValueError: If a symbol is invalid or Sina returns no data for it.
    """
    if isinstance(tickers, str):
        tickers = [tickers]
    symbols = [normalize_symbol(t) for t in tickers]
    symbol_to_ticker = dict(zip(symbols, tickers))

    ticks: list[TickData] = []
    for i in range(0, len(symbols), _MAX_SYMBOLS_PER_REQUEST):
        batch = symbols[i:i + _MAX_SYMBOLS_PER_REQUEST]
        response = _SESSION.get(_HQ_URL + ",".join(batch), headers=_HQ_HEADERS, timeout=10)
        response.raise_for_status()
        for line in response.content.decode('gbk').splitlines():
            if not line.startswith("var hq_str_"):
                continue
            symbol = line[11:line.find('=')]
            fields = line.split('"', 2)[1].split(",")
            if len(fields) < 32 or not fields[0]:
                raise ValueError(f"Sina returned no data for {symbol}: check the symbol.")
            tick_time = datetime.datetime.strptime(
                f"{fields[30]} {fields[31]}", '%Y-%m-%d %H:%M:%S'
            ).replace(tzinfo=_BEIJING_TZ)

            # fields: 0 name, 1 open, 2 prev_close, 3 last, 4 high, 5 low, 6/7 bid1/ask1,
            # 8 volume, 9 amount, 10-19 bid1-5 (volume, price) pairs, 20-29 ask1-5 pairs,
            # 30 date, 31 time, 32 status
            book = {}
            total_bid_volume = 0.0
            total_ask_volume = 0.0
            for level in range(5):
                bid_volume, bid_price = float(fields[10 + level * 2]), float(fields[11 + level * 2])
                ask_volume, ask_price = float(fields[20 + level * 2]), float(fields[21 + level * 2])
                if bid_price > 0:
                    book[f'bid_volume_{level + 1}'] = bid_volume
                    book[f'bid_price_{level + 1}'] = bid_price
                    total_bid_volume += bid_volume
                if ask_price > 0:
                    book[f'ask_volume_{level + 1}'] = ask_volume
                    book[f'ask_price_{level + 1}'] = ask_price
                    total_ask_volume += ask_volume

            ticks.append(
                TickData(
                    ticker=symbol_to_ticker[symbol],
                    timestamp=tick_time.timestamp(),
                    last_price=float(fields[3]),
                    open_price=float(fields[1]),
                    prev_close=float(fields[2]),
                    total_traded_volume=float(fields[8]),
                    total_traded_notional=float(fields[9]),
                    total_bid_volume=total_bid_volume,
                    total_ask_volume=total_ask_volume,
                    **book,
                )
            )
    return ticks


class IndexWeight(NamedTuple):
    """One index constituent weight entry."""
    code: str
    name: str
    weight: float  # percentage, e.g. 3.2 means 3.2%


def _find_column(df: pd.DataFrame, *keywords: str) -> str:
    """Locate a column whose name contains all keywords (and no 'Eng' when excluded)."""
    for col in df.columns:
        text = str(col).lower()
        if all(k in text for k in keywords) and "eng" not in text:
            return col
    raise ValueError(f"Cannot locate column {keywords} in {list(df.columns)}")


def get_index_weights(ticker: str, exchange: str | None = None) -> tuple[datetime.date, list[IndexWeight]]:
    """
    Fetch the latest constituent weights for a CSIndex-published index from CSIndex
    (中证指数官网), e.g. ``000016.SH`` 上证50, ``000300.SH`` 沪深300, ``000905.SH`` 中证500.

    The file is the latest monthly close-weight snapshot — CSIndex publishes no
    arbitrary-date selection (the query string in the download URL is only a cache
    buster). SZ-exchange indices (``399xxx``) are not CSIndex-managed and 404.

    Args:
        ticker (str): The index ticker, e.g. ``000016.SH``, ``000016``, ``sh000016``.
        exchange (str | None): Explicit exchange (``sh``/``sz``/``bj``) for the ticker.

    Returns:
        tuple[datetime.date, list[IndexWeight]]: The weights' as-of date (the monthly
            close date) and constituents sorted by weight descending.

    Raises:
        ValueError: If CSIndex has no close-weight file for the ticker.
    """
    code, _ = _parse_ticker(ticker, exchange)
    response = _SESSION.get(_CSINDEX_WEIGHT_URL.format(code=code), timeout=60)
    if response.status_code == 404:
        raise ValueError(f"No close-weight file for {ticker}: CSIndex only publishes CSI/SSE indices (e.g. 000016.SH).")
    response.raise_for_status()

    df = pd.read_excel(io.BytesIO(response.content), engine="xlrd")
    date_col = _find_column(df, "date")
    code_col = _find_column(df, "constituent", "code")
    name_col = _find_column(df, "constituent", "name")
    weight_col = _find_column(df, "weight")

    as_of_raw = df[date_col].iloc[0]
    if isinstance(as_of_raw, datetime.datetime):
        as_of = as_of_raw.date()
    else:
        as_of = datetime.datetime.strptime(str(as_of_raw)[:8], '%Y%m%d').date()

    weights = [
        IndexWeight(code=str(row[code_col]), name=str(row[name_col]), weight=float(row[weight_col]))
        for _, row in df.iterrows()
    ]
    weights.sort(key=lambda w: w.weight, reverse=True)
    return as_of, weights


if __name__ == '__main__':
    print(__doc__)
    print("""
Usage guide
===========
    from algo_engine.utils.sina_bar import get_minute_bars, get_realtime_ticks, get_index_weights
    import datetime

    # 5-minute bars of a given date (scale: 5/15/30/60/240)
    bars = get_minute_bars('000016.SH', datetime.date(2026, 8, 17))
    bars = get_minute_bars('600519.SH', datetime.date(2026, 8, 17), scale=60)

    # real-time market snapshots (poll every ~3s for live quotes); list or single ticker
    ticks = get_realtime_ticks(['600519.SH', '000016.SH', '399001.SZ'])

    # latest monthly index constituent weights -> (as_of_date, [IndexWeight(code, name, weight%)])
    as_of, weights = get_index_weights('000016.SH')

Tickers
-------
    Wind-style '600519.SH' / bare '600519' (auto-detected sh/sz/bj) / prefixed 'sh600519'.
    Ambiguous '000xxx' defaults to the Shenzhen stock; pass exchange='sh' for the
    Shanghai index, e.g. get_minute_bars('000016', d, exchange='sh') -> 上证50.

Sina API history limitation
---------------------------
    The free bar API (datalen capped at 1023) keeps only ~21 trading days (~1 month)
    of minute bars; older dates raise ValueError. Tick API is real-time snapshot only,
    no history. Index weights come from CSIndex (not Sina) and are the latest monthly
    close snapshot - no arbitrary-date selection.

Live demo
=========""")
    d = datetime.date(2026, 8, 17)

    bars = get_minute_bars('000016.SH', d)
    print(f"{len(bars)} bars for 000016.SH on {d} (sina):")
    for bar in bars[:3]:
        print(bar)

    as_of, weights = get_index_weights('000016.SH')
    print(f"\n{len(weights)} constituents of 000016.SH as of {as_of} (top 5):")
    for w in weights[:5]:
        print(f"  {w.code} {w.name}: {w.weight}%")

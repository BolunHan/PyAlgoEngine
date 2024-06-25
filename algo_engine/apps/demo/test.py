import datetime
import time

from algo_engine.apps.backtest import WebApp, LOGGER
from algo_engine.base import Progress
from algo_engine.profile import PROFILE_CN
from algo_engine.utils import fake_data


def main():
    PROFILE_CN.override_profile()
    ticker = '000016.SH'
    market_date = datetime.date.today()

    data_set = fake_data(market_date=market_date)
    web_app = WebApp(start_date=market_date, end_date=market_date)
    LOGGER.info(f'{len(data_set)} fake data generated for {ticker} {market_date}.')

    web_app.register(ticker=ticker)
    web_app.serve(blocking=False)

    LOGGER.info(f'web app started at {web_app.url}')

    for ts, row in Progress(list(data_set.iterrows())):
        web_app.update(
            timestamp=ts,
            ticker=ticker,
            open_price=row['open_price'],
            close_price=row['close_price'],
            high_price=row['high_price'],
            low_price=row['low_price'],
        )
        time.sleep(0.05)


if __name__ == '__main__':
    main()

    # sys.exit(-1)

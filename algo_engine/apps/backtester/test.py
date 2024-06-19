import datetime
import sys
import time
from threading import Thread

from algo_engine.apps.backtester import DOC_MANAGER, start_app, LOGGER
from algo_engine.base import Progress
from algo_engine.profile import PROFILE_CN
from algo_engine.utils import fake_data


def main():
    PROFILE_CN.override_profile()
    ticker = '000016.SH'
    market_date = datetime.date.today()

    data_set = fake_data(market_date=market_date)
    LOGGER.info(f'{len(data_set)} fake data generated for {ticker} {market_date}.')

    doc_server = DOC_MANAGER.register(ticker=ticker, bokeh_update_interval=0)
    Thread(target=start_app, daemon=True).start()

    LOGGER.info(f'web application started at http://localhost:{DOC_MANAGER.port}')
    LOGGER.info(f'bokeh server started at http://localhost:5006{doc_server.url}')

    for ts, row in Progress(list(data_set.iterrows())):
        doc_server.update(
            timestamp=ts,
            open_price=row['open_price'],
            close_price=row['close_price'],
            high_price=row['high_price'],
            low_price=row['low_price'],
        )
        time.sleep(0.2)


if __name__ == '__main__':
    main()

    # sys.exit(-1)

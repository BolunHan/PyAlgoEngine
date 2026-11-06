from algo_engine.engine cimport MDS


def market_price(bytes ticker):
    return MDS.c_get_market_price(ticker)

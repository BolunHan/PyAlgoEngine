import copy
import datetime
import gc
import pathlib

from algo_engine.base import TickData
from algo_engine.base.c_market_data.c_market_data_buffer import MarketDataBuffer


def reconstruct():
    ts = datetime.datetime(2024, 11, 11, 9, 30, 0).timestamp()
    td_0 = TickData(
        ticker='600010.SH',
        timestamp=ts,
        last_price=123.45,
    )

    td_1 = TickData(
        ticker='000001.SZ',
        timestamp=ts,
        last_price=543.21,
    )

    td_2 = TickData(
        ticker='601888.SH',
        timestamp=ts,
        last_price=321.09,
    )

    assert td_0.ticker == '600010.SH'
    assert td_1.ticker == '000001.SZ'
    assert td_2.ticker == '601888.SH'

    blob_0 = td_0.to_bytes()
    blob_1 = td_1.to_bytes()
    blob_2 = td_2.to_bytes()

    del td_0
    del td_1
    del td_2

    gc.collect()

    td_reconstruct_0 = TickData.from_bytes(blob_0)
    td_reconstruct_1 = TickData.from_bytes(blob_1)
    td_reconstruct_2 = TickData.from_bytes(blob_2)

    assert td_reconstruct_0.ticker == '600010.SH'
    assert td_reconstruct_1.ticker == '000001.SZ'
    assert td_reconstruct_2.ticker == '601888.SH'


def test_02_reconstruct_from_buf():
    cwd = pathlib.Path(__file__).parent

    blob_0 = cwd / 'artifacts' / '20241111_ALL_000001.SZ.bin'
    blob_1 = cwd / 'artifacts' / '20241111_ALL_600010.SH.bin'
    blob_2 = cwd / 'artifacts' / '20241111_ALL_601888.SH.bin'

    buf_0 = MarketDataBuffer.from_bytes(blob_0.read_bytes())
    buf_1 = MarketDataBuffer.from_bytes(blob_1.read_bytes())
    buf_2 = MarketDataBuffer.from_bytes(blob_2.read_bytes())

    assert all(md.ticker == '000001.SZ' for md in buf_0)
    assert all(md.ticker == '600010.SH' for md in buf_1)
    assert all(md.ticker == '601888.SH' for md in buf_2)

    flatten_0 = list(buf_0)
    flatten_1 = list(buf_1)
    flatten_2 = list(buf_2)

    del buf_0
    del buf_1
    del buf_2

    gc.collect()

    assert all(md.ticker == '000001.SZ' for md in flatten_0)
    assert all(md.ticker == '600010.SH' for md in flatten_1)
    assert all(md.ticker == '601888.SH' for md in flatten_2)

    del flatten_0
    del flatten_1
    del flatten_2

    gc.collect()

    loaded_0 = dummy_load_data(blob_0)
    loaded_1 = dummy_load_data(blob_1)
    loaded_2 = dummy_load_data(blob_2)

    assert all(md.ticker == '000001.SZ' for md in loaded_0)
    assert all(md.ticker == '600010.SH' for md in loaded_1)
    assert all(md.ticker == '601888.SH' for md in loaded_2)


def dummy_load_data(blob_path: pathlib.Path):
    buf = MarketDataBuffer.from_bytes(blob_path.read_bytes())
    # This is the failing cases: with copy, it failed, with deepcopy, it works.
    # The only difference is that deepcopy re-lookup an istr from the interned string pool.
    data = list(copy.deepcopy(_) for _ in buf)
    # data = list(copy.copy(_) for _ in buf)
    return data


if __name__ == '__main__':
    test_02_reconstruct_from_buf()

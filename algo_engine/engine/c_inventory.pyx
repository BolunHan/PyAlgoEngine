import enum
import json
import os
import pathlib

from cpython.unicode cimport PyUnicode_AsUTF8, PyUnicode_FromString, PyUnicode_FromStringAndSize
from libc.stdint cimport int8_t

from cbase.bytemap cimport bytemap_entry

import pandas as pd

from algo_engine.base import TransactionDirection
from algo_engine.base.c_allocator_protocol cimport MD_DEFAULT_ALLOCATOR
from algo_engine.base.c_market_data.c_market_data cimport c_md_side_sign, md_side
from algo_engine.exchange_profile.c_exchange_profile cimport SessionDate, c_ex_profile_session_date_init_from_pydate

from . import LOGGER


class InventorySecurityType(enum.IntEnum):
    Commodity = dma_security_type.DMA_SECURITY_COMMODITY
    CurrencySwap = dma_security_type.DMA_SECURITY_CURRENCY_SWAP
    Crypto = dma_security_type.DMA_SECURITY_CRYPTO
    IndexFuture = dma_security_type.DMA_SECURITY_INDEX_FUTURE
    Stock = dma_security_type.DMA_SECURITY_STOCK


cdef class InventoryCashDividend(CCPType):
    def __init__(self, py_date record_date, py_date ex_date, double dividend_per_share):
        self.header = c_dma_inventory_cash_dividend_new(MD_DEFAULT_ALLOCATOR)
        if not self.header:
            raise MemoryError(f'Failed to allocate memory for <{self.__class__.__name__}>.')

        self.owner = True

        c_ex_profile_session_date_init_from_pydate(record_date, &self.header.record_date)
        c_ex_profile_session_date_init_from_pydate(ex_date, &self.header.ex_date)
        self.header.dividend_per_share = dividend_per_share

        self.ccp_bind(&self.header)

    def __dealloc__(self):
        if not self.owner:
            return

        if self.header:
            c_dma_inventory_cash_dividend_free(self.header)
            self.header = NULL

    @staticmethod
    cdef InventoryCashDividend c_from_header(dma_inventory_cash_dividend* header, bint owner):
        cdef InventoryCashDividend instance = InventoryCashDividend.__new__(InventoryCashDividend)
        instance.header = header
        instance.owner = owner

        if header:
            instance.ccp_bind(&instance.header)

        return instance

    property record_date:
        def __get__(self):
            if not self.header:
                raise RuntimeError(f'<{self.__class__.__name__}> not initialized!')
            return SessionDate(self.header.record_date.year, self.header.record_date.month, self.header.record_date.day)

    property ex_date:
        def __get__(self):
            if not self.header:
                raise RuntimeError(f'<{self.__class__.__name__}> not initialized!')
            return SessionDate(self.header.ex_date.year, self.header.ex_date.month, self.header.ex_date.day)

    property dividend_per_share:
        def __get__(self):
            if not self.header:
                raise RuntimeError(f'<{self.__class__.__name__}> not initialized!')
            return self.header.dividend_per_share

    property applied:
        def __get__(self):
            if not self.header:
                raise RuntimeError(f'<{self.__class__.__name__}> not initialized!')
            return self.header.applied


cdef class InventoryStockDividend(CCPType):
    def __init__(self, py_date record_date, py_date ex_date, double dividend_per_share):
        self.header = c_dma_inventory_stock_dividend_new(MD_DEFAULT_ALLOCATOR)
        if not self.header:
            raise MemoryError(f'Failed to allocate memory for <{self.__class__.__name__}>.')

        self.owner = True

        c_ex_profile_session_date_init_from_pydate(record_date, &self.header.record_date)
        c_ex_profile_session_date_init_from_pydate(ex_date, &self.header.ex_date)
        self.header.dividend_per_share = dividend_per_share

        self.ccp_bind(&self.header)

    def __dealloc__(self):
        if not self.owner:
            return

        if self.header:
            c_dma_inventory_stock_dividend_free(self.header)
            self.header = NULL

    @staticmethod
    cdef InventoryStockDividend c_from_header(dma_inventory_stock_dividend* header, bint owner):
        cdef InventoryStockDividend instance = InventoryStockDividend.__new__(InventoryStockDividend)
        instance.header = header
        instance.owner = owner

        if header:
            instance.ccp_bind(&instance.header)

        return instance

    property record_date:
        def __get__(self):
            if not self.header:
                raise RuntimeError(f'<{self.__class__.__name__}> not initialized!')
            return SessionDate(self.header.record_date.year, self.header.record_date.month, self.header.record_date.day)

    property ex_date:
        def __get__(self):
            if not self.header:
                raise RuntimeError(f'<{self.__class__.__name__}> not initialized!')
            return SessionDate(self.header.ex_date.year, self.header.ex_date.month, self.header.ex_date.day)

    property dividend_per_share:
        def __get__(self):
            if not self.header:
                raise RuntimeError(f'<{self.__class__.__name__}> not initialized!')
            return self.header.dividend_per_share

    property applied:
        def __get__(self):
            if not self.header:
                raise RuntimeError(f'<{self.__class__.__name__}> not initialized!')
            return self.header.applied


cdef class InventoryStockSplit(CCPType):
    def __init__(self, py_date record_date, py_date ex_date, double multiplier):
        self.header = c_dma_inventory_stock_split_new(MD_DEFAULT_ALLOCATOR)
        if not self.header:
            raise MemoryError(f'Failed to allocate memory for <{self.__class__.__name__}>.')

        self.owner = True

        c_ex_profile_session_date_init_from_pydate(record_date, &self.header.record_date)
        c_ex_profile_session_date_init_from_pydate(ex_date, &self.header.ex_date)
        self.header.multiplier = multiplier

        self.ccp_bind(&self.header)

    def __dealloc__(self):
        if not self.owner:
            return

        if self.header:
            c_dma_inventory_stock_split_free(self.header)
            self.header = NULL

    @staticmethod
    cdef InventoryStockSplit c_from_header(dma_inventory_stock_split* header, bint owner):
        cdef InventoryStockSplit instance = InventoryStockSplit.__new__(InventoryStockSplit)
        instance.header = header
        instance.owner = owner

        if header:
            instance.ccp_bind(&instance.header)

        return instance

    property record_date:
        def __get__(self):
            if not self.header:
                raise RuntimeError(f'<{self.__class__.__name__}> not initialized!')
            return SessionDate(self.header.record_date.year, self.header.record_date.month, self.header.record_date.day)

    property ex_date:
        def __get__(self):
            if not self.header:
                raise RuntimeError(f'<{self.__class__.__name__}> not initialized!')
            return SessionDate(self.header.ex_date.year, self.header.ex_date.month, self.header.ex_date.day)

    property multiplier:
        def __get__(self):
            if not self.header:
                raise RuntimeError(f'<{self.__class__.__name__}> not initialized!')
            return self.header.multiplier

    property applied:
        def __get__(self):
            if not self.header:
                raise RuntimeError(f'<{self.__class__.__name__}> not initialized!')
            return self.header.applied


cdef class InventoryStockConversion(CCPType):
    def __init__(self, py_date record_date, py_date ex_date, str convert_from, str convert_to, double multiplier):
        cdef int ret_code

        self.header = c_dma_inventory_stock_conversion_new(MD_DEFAULT_ALLOCATOR)
        if not self.header:
            raise MemoryError(f'Failed to allocate memory for <{self.__class__.__name__}>.')

        self.owner = True

        ret_code = c_dma_inventory_stock_conversion_set_tickers(self.header, PyUnicode_AsUTF8(convert_from), PyUnicode_AsUTF8(convert_to))
        if ret_code != inventory_ret_code.INVENTORY_OK:
            raise RuntimeError(f'Failed to set the conversion tickers, err code: {ret_code}')

        c_ex_profile_session_date_init_from_pydate(record_date, &self.header.record_date)
        c_ex_profile_session_date_init_from_pydate(ex_date, &self.header.ex_date)
        self.header.multiplier = multiplier

        self.ccp_bind(&self.header)

    def __dealloc__(self):
        if not self.owner:
            return

        if self.header:
            c_dma_inventory_stock_conversion_free(self.header)
            self.header = NULL

    @staticmethod
    cdef InventoryStockConversion c_from_header(dma_inventory_stock_conversion* header, bint owner):
        cdef InventoryStockConversion instance = InventoryStockConversion.__new__(InventoryStockConversion)
        instance.header = header
        instance.owner = owner

        if header:
            instance.ccp_bind(&instance.header)

        return instance

    property convert_from:
        def __get__(self):
            if not self.header:
                raise RuntimeError(f'<{self.__class__.__name__}> not initialized!')
            if not self.header.convert_from:
                return None
            return PyUnicode_FromString(self.header.convert_from)

    property convert_to:
        def __get__(self):
            if not self.header:
                raise RuntimeError(f'<{self.__class__.__name__}> not initialized!')
            if not self.header.convert_to:
                return None
            return PyUnicode_FromString(self.header.convert_to)

    property record_date:
        def __get__(self):
            if not self.header:
                raise RuntimeError(f'<{self.__class__.__name__}> not initialized!')
            return SessionDate(self.header.record_date.year, self.header.record_date.month, self.header.record_date.day)

    property ex_date:
        def __get__(self):
            if not self.header:
                raise RuntimeError(f'<{self.__class__.__name__}> not initialized!')
            return SessionDate(self.header.ex_date.year, self.header.ex_date.month, self.header.ex_date.day)

    property multiplier:
        def __get__(self):
            if not self.header:
                raise RuntimeError(f'<{self.__class__.__name__}> not initialized!')
            return self.header.multiplier

    property applied:
        def __get__(self):
            if not self.header:
                raise RuntimeError(f'<{self.__class__.__name__}> not initialized!')
            return self.header.applied


cdef class InventoryEntry(CCPType):
    def __init__(
            self,
            str ticker,
            double volume,
            double price,
            dma_security_type security_type,
            md_direction direction=md_direction.DIRECTION_LONG,
            **kwargs
    ):
        if volume < 0:
            LOGGER.warning('volume of Inventory.Entry normally should be positive!')

        if c_md_side_sign(<md_side> direction) == 0:
            raise ValueError('Inventory supports only DIRECTION_LONG and DIRECTION_SHORT.')

        self.ticker = ticker
        self.c_direction = direction

        self.header = c_dma_inventory_entry_new(MD_DEFAULT_ALLOCATOR)
        if not self.header:
            raise MemoryError(f'Failed to allocate memory for <{self.__class__.__name__}>.')

        self.owner = True

        self.header.volume = volume
        self.header.price = price
        self.header.notional = kwargs.pop('notional', volume * price)
        self.header.fee = kwargs.pop('fee', 0.)
        self.header.recalled = kwargs.pop('recalled', 0.)
        self.header.security_type = security_type

        self.ccp_bind(&self.header)

    def __dealloc__(self):
        if not self.owner:
            return

        if self.header:
            c_dma_inventory_entry_free(self.header)
            self.header = NULL

    @staticmethod
    cdef InventoryEntry c_from_header(dma_inventory_entry* header, str ticker, md_direction direction, bint owner):
        cdef InventoryEntry instance = InventoryEntry.__new__(InventoryEntry)
        instance.header = header
        instance.ticker = ticker
        instance.c_direction = direction
        instance.owner = owner

        if header:
            instance.ccp_bind(&instance.header)

        return instance

    def __repr__(self):
        if not self.header:
            return f'<Inventory.Entry>(released)'

        cdef str side = 'Long' if self.c_direction == md_direction.DIRECTION_LONG else 'Short'
        return f'<Inventory.Entry>(ticker={self.ticker}, side={side}, volume={self.header.volume:,}, fee={self.header.fee:.2f})'

    def __add__(self, other):
        if isinstance(other, InventoryEntry):
            return self.merge(other)

        raise TypeError(f'Can only merge type {self.__class__.__name__}')

    def __bool__(self):
        if not self.header:
            return False

        return self.header.volume != 0

    # === Cython Internal Binding ===

    def to_json(self, fmt='str'):
        json_dict = dict(
            ticker=self.ticker,
            volume=self.header.volume,
            price=self.header.price,
            security_type=self.security_type.value,
            direction=<int> self.c_direction,
            notional=self.header.notional,
            fee=self.header.fee,
            recalled=self.header.recalled
        )

        if fmt == 'dict':
            return json_dict
        else:
            return json.dumps(json_dict)

    @classmethod
    def from_json(cls, json_str):
        if isinstance(json_str, (str, bytes)):
            json_dict = json.loads(json_str)
        elif isinstance(json_str, dict):
            json_dict = json_str
        else:
            raise TypeError(f'Invalid type {type(json_str)}, expect [str, bytes, dict]')

        entry = cls(
            ticker=json_dict['ticker'],
            volume=json_dict['volume'],
            price=json_dict['price'],
            security_type=InventorySecurityType(json_dict['security_type']),
            direction=json_dict['direction'],
            notional=json_dict['notional'],
            fee=json_dict.get('fee', 0.),
            recalled=json_dict.get('recalled', 0.),
        )

        return entry

    def merge(self, InventoryEntry entry, bint inplace=False, **kwargs):
        if entry.ticker != self.ticker:
            raise ValueError(f'<ticker> not match! Expect {self.ticker}, got {entry.ticker}')

        if entry.c_direction != self.c_direction:
            raise ValueError(f'<direction> not match! Expect {<int> self.c_direction}, got {<int> entry.c_direction}')

        if entry.security_type != self.security_type:
            raise ValueError(f'<security_type> not match! Expect {self.security_type}, got {entry.security_type}')

        cdef double volume = kwargs.pop('volume', self.header.volume + entry.header.volume)
        cdef double notional = kwargs.pop('notional', self.header.notional + entry.header.notional)
        cdef double price = kwargs.pop('price', (self.header.price * self.header.volume + entry.header.price * entry.header.volume) / (self.header.volume + entry.header.volume))
        cdef double fee = kwargs.pop('fee', self.header.fee + entry.header.fee)
        cdef double recalled = kwargs.pop('recalled', self.header.recalled + entry.header.recalled)

        if inplace:
            self.header.volume = volume
            self.header.notional = notional
            self.header.price = price
            self.header.fee = fee
            self.header.recalled = recalled

            return self

        return InventoryEntry(
            ticker=self.ticker,
            volume=volume,
            price=price,
            security_type=self.security_type,
            direction=self.c_direction,
            notional=notional,
            fee=fee,
            recalled=recalled
        )

    def apply_cash_dividend(self, dividend: InventoryCashDividend):
        self.header.price -= dividend.dividend_per_share
        self.header.notional = self.header.price * self.header.volume

    def apply_stock_dividend(self, dividend: InventoryStockDividend):
        cdef double factor = 1. + dividend.dividend_per_share

        if factor <= 0:
            raise ValueError('Stock dividend factor must be positive.')

        self.header.volume *= factor
        self.header.price /= factor

    def apply_conversion(self, stock_conversion: InventoryStockConversion):
        cdef double multiplier = stock_conversion.multiplier

        if multiplier <= 0:
            raise ValueError('Conversion multiplier must be positive.')

        self.header.volume *= multiplier
        self.header.price /= multiplier
        self.ticker = stock_conversion.convert_to

    def apply_split(self, stock_split: InventoryStockSplit):
        cdef double multiplier = stock_split.multiplier

        if multiplier <= 0:
            raise ValueError('Split multiplier must be positive.')

        self.header.volume *= multiplier
        self.header.price /= multiplier

    # === Python Properties ===

    property direction:
        def __get__(self):
            return TransactionDirection(<int> self.c_direction)

    property volume:
        def __get__(self):
            if not self.header:
                raise RuntimeError(f'<{self.__class__.__name__}> not initialized!')
            return self.header.volume

        def __set__(self, double value):
            if not self.header:
                raise RuntimeError(f'<{self.__class__.__name__}> not initialized!')
            self.header.volume = value

    property price:
        def __get__(self):
            if not self.header:
                raise RuntimeError(f'<{self.__class__.__name__}> not initialized!')
            return self.header.price

        def __set__(self, double value):
            if not self.header:
                raise RuntimeError(f'<{self.__class__.__name__}> not initialized!')
            self.header.price = value

    property notional:
        def __get__(self):
            if not self.header:
                raise RuntimeError(f'<{self.__class__.__name__}> not initialized!')
            return self.header.notional

        def __set__(self, double value):
            if not self.header:
                raise RuntimeError(f'<{self.__class__.__name__}> not initialized!')
            self.header.notional = value

    property fee:
        def __get__(self):
            if not self.header:
                raise RuntimeError(f'<{self.__class__.__name__}> not initialized!')
            return self.header.fee

        def __set__(self, double value):
            if not self.header:
                raise RuntimeError(f'<{self.__class__.__name__}> not initialized!')
            self.header.fee = value

    property recalled:
        def __get__(self):
            if not self.header:
                raise RuntimeError(f'<{self.__class__.__name__}> not initialized!')
            return self.header.recalled

        def __set__(self, double value):
            if not self.header:
                raise RuntimeError(f'<{self.__class__.__name__}> not initialized!')
            self.header.recalled = value

    property security_type:
        def __get__(self):
            if not self.header:
                raise RuntimeError(f'<{self.__class__.__name__}> not initialized!')
            return InventorySecurityType(<int> self.header.security_type)

    property available:
        def __get__(self):
            if not self.header:
                raise RuntimeError(f'<{self.__class__.__name__}> not initialized!')
            return c_dma_inventory_entry_available(self.header)


cdef class Inventory(CCPType):
    SecurityType = InventorySecurityType
    Entry = InventoryEntry
    CashDividend = InventoryCashDividend
    StockDividend = InventoryStockDividend
    StockSplit = InventoryStockSplit
    StockConversion = InventoryStockConversion

    def __init__(self):
        self.header = c_dma_inventory_new(MD_DEFAULT_ALLOCATOR)
        if not self.header:
            raise MemoryError(f'Failed to allocate memory for <{self.__class__.__name__}>.')

        self.owner = True
        self.ccp_bind(&self.header)

    def __dealloc__(self):
        if not self.owner:
            return

        if self.header:
            c_dma_inventory_free(self.header)
            self.header = NULL

    cdef double c_side_volume(self, str ticker, bint is_long, bint* found):
        cdef const dma_inventory_entry* lot = NULL
        cdef double volume = 0.
        cdef int ret_code

        if is_long:
            ret_code = c_dma_inventory_get_inv_long(self.header, PyUnicode_AsUTF8(ticker), &lot)
        else:
            ret_code = c_dma_inventory_get_inv_short(self.header, PyUnicode_AsUTF8(ticker), &lot)

        if found:
            found[0] = ret_code == inventory_ret_code.INVENTORY_OK

        if ret_code != inventory_ret_code.INVENTORY_OK:
            return 0.

        while lot:
            volume += lot.volume
            lot = lot.next

        return volume

    def __repr__(self):
        return f'<{self.__class__.__name__}>{{id={id(self)}}}'

    def __call__(self, str ticker):
        return dict(
            Long=self.available_volume(ticker=ticker, direction=md_direction.DIRECTION_LONG),
            Short=self.available_volume(ticker=ticker, direction=md_direction.DIRECTION_SHORT)
        )

    # === Python Interfaces ===

    cpdef void recall(self, str ticker, double volume, md_direction direction=md_direction.DIRECTION_LONG):
        cdef int ret_code
        cdef int8_t side_sign = c_md_side_sign(<md_side> direction)

        if side_sign > 0:
            ret_code = c_dma_inventory_recall_inv_long(self.header, PyUnicode_AsUTF8(ticker), volume)
        elif side_sign < 0:
            ret_code = c_dma_inventory_recall_inv_short(self.header, PyUnicode_AsUTF8(ticker), volume)
        else:
            raise ValueError('Inventory supports only DIRECTION_LONG and DIRECTION_SHORT.')

        if ret_code == inventory_ret_code.INVENTORY_ERR_NOT_FOUND:
            LOGGER.info(f'No inventory for {ticker}, recall action ignored.')
            return

        if ret_code != inventory_ret_code.INVENTORY_OK:
            raise RuntimeError(f'Failed to recall inventory, err code: {ret_code}')

    cpdef void add_inv(self, InventoryEntry entry):
        cdef int ret_code
        cdef int8_t side_sign = c_md_side_sign(<md_side> entry.c_direction)

        if side_sign > 0:
            ret_code = c_dma_inventory_add_inv_long(
                self.header,
                PyUnicode_AsUTF8(entry.ticker),
                entry.header.price,
                entry.header.volume,
                entry.header.notional,
                entry.header.fee,
                entry.header.security_type
            )
        elif side_sign < 0:
            ret_code = c_dma_inventory_add_inv_short(
                self.header,
                PyUnicode_AsUTF8(entry.ticker),
                entry.header.price,
                entry.header.volume,
                entry.header.notional,
                entry.header.fee,
                entry.header.security_type
            )
        else:
            raise ValueError('Inventory supports only DIRECTION_LONG and DIRECTION_SHORT.')

        if ret_code != inventory_ret_code.INVENTORY_OK:
            raise RuntimeError(f'Failed to add inventory, err code: {ret_code}')

    cpdef InventoryEntry get_inv(self, str ticker, md_direction direction=md_direction.DIRECTION_LONG):
        cdef const dma_inventory_entry* lot = NULL
        cdef int8_t side_sign = c_md_side_sign(<md_side> direction)
        cdef int ret_code

        if side_sign > 0:
            ret_code = c_dma_inventory_get_inv_long(self.header, PyUnicode_AsUTF8(ticker), &lot)
        elif side_sign < 0:
            ret_code = c_dma_inventory_get_inv_short(self.header, PyUnicode_AsUTF8(ticker), &lot)
        else:
            raise ValueError('Inventory supports only DIRECTION_LONG and DIRECTION_SHORT.')

        if ret_code != inventory_ret_code.INVENTORY_OK:
            return None

        return InventoryEntry.c_from_header(<dma_inventory_entry*> lot, ticker, direction, False)

    cpdef void use_inv(self, str ticker, double volume, md_direction direction=md_direction.DIRECTION_LONG):
        cdef int ret_code
        cdef int8_t side_sign = c_md_side_sign(<md_side> direction)

        if side_sign > 0:
            ret_code = c_dma_inventory_use_inv_long(self.header, PyUnicode_AsUTF8(ticker), volume)
        elif side_sign < 0:
            ret_code = c_dma_inventory_use_inv_short(self.header, PyUnicode_AsUTF8(ticker), volume)
        else:
            raise ValueError('Inventory supports only DIRECTION_LONG and DIRECTION_SHORT.')

        if ret_code != inventory_ret_code.INVENTORY_OK:
            raise RuntimeError(f'Failed to record used inventory, err code: {ret_code}')

    cpdef double available_volume(self, str ticker, md_direction direction=md_direction.DIRECTION_LONG):
        cdef double out_long = 0.
        cdef double out_short = 0.
        cdef int8_t side_sign = c_md_side_sign(<md_side> direction)

        if side_sign == 0:
            raise ValueError('Inventory supports only DIRECTION_LONG and DIRECTION_SHORT.')

        cdef int ret_code = c_dma_inventory_available(self.header, PyUnicode_AsUTF8(ticker), &out_long, &out_short)

        if ret_code != inventory_ret_code.INVENTORY_OK:
            raise RuntimeError(f'Failed to read the available inventory, err code: {ret_code}')

        return out_long if side_sign > 0 else out_short

    cpdef void cash_dividend(self, py_date record_date, py_date ex_date, double dividend_per_share):
        cdef session_date_t record
        cdef session_date_t ex
        c_ex_profile_session_date_init_from_pydate(record_date, &record)
        c_ex_profile_session_date_init_from_pydate(ex_date, &ex)

        cdef int ret_code = c_dma_inventory_cash_dividend(self.header, record, ex, dividend_per_share)

        if ret_code != inventory_ret_code.INVENTORY_OK:
            raise RuntimeError(f'Failed to apply the cash dividend, err code: {ret_code}')

    cpdef void stock_dividend(self, py_date record_date, py_date ex_date, double dividend_per_share):
        cdef session_date_t record
        cdef session_date_t ex
        c_ex_profile_session_date_init_from_pydate(record_date, &record)
        c_ex_profile_session_date_init_from_pydate(ex_date, &ex)

        cdef int ret_code = c_dma_inventory_stock_dividend(self.header, record, ex, dividend_per_share)

        if ret_code != inventory_ret_code.INVENTORY_OK:
            raise RuntimeError(f'Failed to apply the stock dividend, err code: {ret_code}')

    cpdef void stock_split(self, py_date record_date, py_date ex_date, double multiplier):
        cdef session_date_t record
        cdef session_date_t ex
        c_ex_profile_session_date_init_from_pydate(record_date, &record)
        c_ex_profile_session_date_init_from_pydate(ex_date, &ex)

        cdef int ret_code = c_dma_inventory_stock_split(self.header, record, ex, multiplier)

        if ret_code != inventory_ret_code.INVENTORY_OK:
            raise RuntimeError(f'Failed to apply the split, err code: {ret_code}')

    cpdef void stock_conversion(self, py_date record_date, py_date ex_date, str convert_from, str convert_to, double multiplier):
        cdef session_date_t record
        cdef session_date_t ex
        c_ex_profile_session_date_init_from_pydate(record_date, &record)
        c_ex_profile_session_date_init_from_pydate(ex_date, &ex)

        cdef int ret_code = c_dma_inventory_stock_conversion(self.header, record, ex, PyUnicode_AsUTF8(convert_from), PyUnicode_AsUTF8(convert_to), multiplier)

        if ret_code != inventory_ret_code.INVENTORY_OK:
            raise RuntimeError(f'Failed to apply the conversion, err code: {ret_code}')

    cpdef void clear(self):
        c_dma_inventory_clear(self.header)

    def to_json(self, fmt='str'):
        cdef dict json_dict = {}
        cdef bytemap_entry* entry = self.header.nodes.first
        cdef dma_inventory_node* node
        cdef dma_inventory_entry* lot
        cdef str ticker, key

        while entry:
            ticker = PyUnicode_FromStringAndSize(entry.key, entry.key_length)
            node = (<dma_inventory_node**> entry.value)[0]

            if not node:
                entry = entry.next
                continue

            if node.long_inv:
                key = f'{ticker}.DIRECTION_LONG'
                json_dict[key] = {'used': node.long_used, 'inv': []}
                lot = node.long_inv

                while lot:
                    json_dict[key]['inv'].append({
                        'ticker': ticker,
                        'volume': lot.volume,
                        'price': lot.price,
                        'security_type': <int> lot.security_type,
                        'direction': <int> md_direction.DIRECTION_LONG,
                        'notional': lot.notional,
                        'fee': lot.fee,
                        'recalled': lot.recalled,
                    })
                    lot = lot.next

            if node.short_inv:
                key = f'{ticker}.DIRECTION_SHORT'
                json_dict[key] = {'used': node.short_used, 'inv': []}
                lot = node.short_inv

                while lot:
                    json_dict[key]['inv'].append({
                        'ticker': ticker,
                        'volume': lot.volume,
                        'price': lot.price,
                        'security_type': <int> lot.security_type,
                        'direction': <int> md_direction.DIRECTION_SHORT,
                        'notional': lot.notional,
                        'fee': lot.fee,
                        'recalled': lot.recalled,
                    })
                    lot = lot.next

            entry = entry.next

        if fmt == 'dict':
            return json_dict
        else:
            return json.dumps(json_dict)

    def from_json(self, json_str, with_used=False):
        if isinstance(json_str, (str, bytes)):
            json_dict = json.loads(json_str)
        elif isinstance(json_str, dict):
            json_dict = json_str
        else:
            raise TypeError(f'Invalid type {type(json_str)}, expect [str, bytes, dict]')

        for name, payload in json_dict.items():
            ticker, _, direction_name = name.rpartition('.')
            direction = TransactionDirection[direction_name]

            for entry_json in payload['inv']:
                self.add_inv(InventoryEntry.from_json(entry_json))

            if with_used:
                self.use_inv(ticker=ticker, volume=payload.get('used', 0.), direction=direction)

        return self

    def dump(self, file_path):
        file_path = pathlib.Path(file_path)
        dump_dir = file_path.parent

        os.makedirs(dump_dir, exist_ok=True)

        with open(file_path, 'w') as f:
            f.write(json.dumps(self.to_json(fmt='dict'), indent=4, sort_keys=True))

    def load(self, file_path, with_used=False):
        if not os.path.isfile(file_path):
            LOGGER.error(f'No such file {file_path}')
            return

        with open(file_path, 'r') as f:
            json_str = f.read()

        self.clear()
        self.from_json(json_str, with_used=with_used)

    def to_csv(self, file_path):
        inv_dict = {'inv_l': {}, 'inv_s': {}}
        cdef bint found
        cdef double volume

        for ticker in self.tickers:
            volume = self.c_side_volume(ticker, True, &found)
            if found:
                inv_dict['inv_l'][ticker] = volume

            volume = self.c_side_volume(ticker, False, &found)
            if found:
                inv_dict['inv_s'][ticker] = volume

        inv_df = pd.DataFrame(inv_dict)
        inv_df.to_csv(file_path)

    # === Python Properties ===

    property tickers:
        def __get__(self):
            cdef bytemap_entry* entry = self.header.nodes.first
            cdef set tickers = set()

            while entry:
                tickers.add(PyUnicode_FromStringAndSize(entry.key, entry.key_length))
                entry = entry.next

            return tickers

    property info:
        def __get__(self):
            cdef dict info_dict = {'inv_l': {}, 'inv_s': {}}
            cdef bint found
            cdef double volume
            cdef object ticker

            for ticker in self.tickers:
                volume = self.c_side_volume(ticker, True, &found)
                if found:
                    info_dict['inv_l'][ticker] = volume

                volume = self.c_side_volume(ticker, False, &found)
                if found:
                    info_dict['inv_s'][ticker] = volume

            return pd.DataFrame(info_dict)


INVENTORY = Inventory()

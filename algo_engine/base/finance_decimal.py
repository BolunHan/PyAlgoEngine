import math
import numbers
import operator

__all__ = ['FinancialDecimal']

from typing import Self

TICK_SIZE = 100


class FinancialDecimal(float):
    __slots__ = ('_k', '_tick')

    # We're immutable, so use __new__ not __init__
    def __new__(cls, value: numbers.Real | str = 0., /, k: int = None, tick: int = None):
        if tick is not None and type(tick) is not int:
            raise TypeError(f'tick of {cls} must be a integer.')
        elif tick is None:
            tick = TICK_SIZE
        elif tick <= 0:
            raise ValueError(f'tick of {cls} must be a positive integer.')

        if k is not None and type(k) is not int:
            raise TypeError(f'k of {cls} must be a integer.')
        elif k is None:
            if type(value) is int:
                k = value * tick
            elif isinstance(value, numbers.Real):
                k = round(value * tick)
            elif isinstance(value, (str, bytes)):
                k = round(float(value) * tick)
            else:
                raise TypeError(f'value of the {cls} must be a float-convertable.')

        self = super(FinancialDecimal, cls).__new__(cls, k / tick)

        self._k = k
        self._tick = tick
        return self

    def __repr__(self):
        return f'{self.__class__.__name__}({self._k}, {self._tick})'

    def __str__(self):
        if self._tick == 1:
            return str(self._k)
        else:
            digits = math.ceil(math.log10(self._tick))
            return format(float(self), f'.{digits}f')

    def __reduce__(self):
        return self.__class__, (self._k, self._tick)

    def __copy__(self):
        # for a float, the copy method should return a different object / memory location. In fact, this method should not even be implemented!
        # however the Fraction, as an immutable, use the same object / memory location, as clone.
        # this method uses float-style implementation
        return self.__class__(self._k, self._tick)

    def __deepcopy__(self, memo):
        return self.__class__(self._k, self._tick)

    @classmethod
    def from_float(cls, f: float, tick=None):
        if isinstance(f, numbers.Integral):
            return cls(int(f), tick=tick)
        elif not isinstance(f, float):
            raise TypeError(f"{cls.__name__}.from_float() only takes floats, not {f!r} ({type(f).__name__})")

        return cls(f, tick=tick)

    def as_integer_ratio(self) -> tuple[int, int]:
        return self._k, self._tick

    def __add__(self, other: int | float | numbers.Real):
        if isinstance(other, int):
            return self.__class__(k=self._k + other * self._tick, tick=self._tick)
        elif isinstance(other, self.__class__) and self._tick == other._tick:
            return self.__class__(k=self._k + other._k, tick=self._tick)
        elif isinstance(other, float):
            return self.__class__(k=round(self._k + float.__mul__(other, self._tick)), tick=self._tick)
        else:
            return float.__add__(self, other)

    def __radd__(self, other: int | float | numbers.Real):
        if isinstance(other, numbers.Real):
            return self.__add__(other)
        else:
            return other.__add__(float(self))

    def __sub__(self, other: int | float | numbers.Real):
        if isinstance(other, int):
            return self.__class__(k=self._k - other * self._tick, tick=self._tick)
        elif isinstance(other, self.__class__) and self._tick == other._tick:
            return self.__class__(k=self._k - other._k, tick=self._tick)
        elif isinstance(other, float):
            return self.__class__(k=round(self._k - float.__mul__(other, self._tick)), tick=self._tick)
        else:
            return float.__sub__(self, other)

    def __rsub__(self, other: int | float | numbers.Real):
        if isinstance(other, numbers.Real):
            return FinancialDecimal(other, tick=self._tick).__sub__(self)
        else:
            return float.__sub__(other, self)

    def __mul__(self, other: int | float | numbers.Real):
        if isinstance(other, int):
            return self.__class__(k=self._k * other, tick=self._tick)
        elif isinstance(other, float):
            return self.__class__(k=round(self._k * float(other)), tick=self._tick)
        else:
            return float.__mul__(self, other)

    def __rmul__(self, other: int | float | numbers.Real):
        if isinstance(other, numbers.Real):
            return self.__mul__(other)
        else:
            return other.__mul__(float(self))

    def __truediv__(self, other: int | float | numbers.Real):
        if isinstance(other, numbers.Real):
            return self.__class__(k=round(self._k / float(other)), tick=self._tick)
        else:
            float.__truediv__(self, other)

    def __rtruediv__(self, other: int | float | Self):
        if isinstance(other, numbers.Real):
            return self.__class__(k=round(float(other) / float(self) * self._tick), tick=self._tick)
        else:
            float.__truediv__(other, self)

    def __floordiv__(self, other: int | float | Self):
        if isinstance(other, (int, self.__class__)):
            return (self._k * other.denominator) // (self._tick * other.numerator)
        else:
            return float.__floordiv__(self, other)

    def __rdivmod__(self, other):
        return divmod(other, float(self))

    def __pow__(self, other: int | float | Self, __mod: None = None):
        if isinstance(other, numbers.Real):
            return self.__class__(k=round(float(self) ** float(other) * self._tick), tick=self._tick)
        else:
            return float.__pow__(self, other, __mod)

    def __pos__(self):
        return FinancialDecimal(k=self._k, tick=self._tick)

    def __neg__(self):
        return FinancialDecimal(k=-self._k, tick=self._tick)

    def __abs__(self):
        return FinancialDecimal(k=abs(self._k), tick=self._tick)

    def _richcmp(self, other, op):
        # convert other to a Rational instance where reasonable.
        if isinstance(other, (numbers.Rational, self.__class__)):
            return op(self._k * other.denominator, self._tick * other.numerator)
        if isinstance(other, float):
            if math.isnan(other) or math.isinf(other):
                return op(0.0, other)
            else:
                return op(self, FinancialDecimal(other, tick=self._tick))
        else:
            return NotImplemented

    def __eq__(self, other):
        return self._richcmp(other, operator.eq)

    def __lt__(self, other):
        """a < b"""
        return self._richcmp(other, operator.lt)

    def __gt__(self, other):
        """a > b"""
        return self._richcmp(other, operator.gt)

    def __le__(self, other):
        """a <= b"""
        return self._richcmp(other, operator.le)

    def __ge__(self, other):
        """a >= b"""
        return self._richcmp(other, operator.ge)

    def __bool__(self):
        return bool(self._k)

    @property
    def k(self):
        return self._k

    @property
    def numerator(self):
        return self._k

    @property
    def tick(self):
        return self._tick

    @property
    def denominator(self):
        return self._tick


def main():
    fd_0 = FinancialDecimal(math.pi, tick=10000)
    print(f'fd_0 => {fd_0}')
    print(f'fd_0:.3f => {fd_0:.3f}')

    # i = 2
    for i in [2, math.e]:
        print(f'fd_0 + {i} => {fd_0 + i}')
        print(f'{i} + fd_0 => {i + fd_0}')

        print(f'fd_0 - {i} => {fd_0 - i}')
        print(f'{i} - fd_0 => {i - fd_0}')

        print(f'fd_0 * {i} => {fd_0 * i}')
        print(f'{i} * fd_0 => {i * fd_0}')

        print(f'fd_0 / {i} => {fd_0 / i}')
        print(f'{i} / fd_0 => {i / fd_0}')

        print(f'fd_0 ** {i} => {fd_0 ** i}')
        print(f'{i} ** fd_0 => {i ** fd_0}')

        print(f'fd_0 // {i} => {fd_0 // i}')
        print(f'{i} // fd_0 => {i // fd_0}')

        print(f'fd_0 % {i} => {fd_0 % i}')
        print(f'{i} % fd_0 => {i % fd_0}')

        print(f'fd_0 __divmod__ {i} => {fd_0.__divmod__(i)}')
        print(f'{i} __divmod__ fd_0 => {divmod(i, fd_0)}')

    print(f'fd_0 == math.pi => {fd_0 == math.pi}')
    print(f'fd_0 == 2 => {fd_0 == 2}')
    print(f'fd_0 == math.e => {fd_0 == math.e}')

    print(f'fd_0 >= math.pi => {fd_0 >= math.pi}')
    print(f'fd_0 >= 2 => {fd_0 >= 2}')
    print(f'fd_0 >= math.e => {fd_0 >= math.e}')

    print(f'fd_0 <= math.pi => {fd_0 <= math.pi}')
    print(f'fd_0 <= 2 => {fd_0 <= 2}')
    print(f'fd_0 <= math.e => {fd_0 <= math.e}')

    print(f'math.pi == fd_0  => {math.pi == fd_0}')
    print(f'2 == fd_0=> {2 == fd_0}')
    print(f'math.e == fd_0 => {math.e == fd_0}')


if __name__ == '__main__':
    main()

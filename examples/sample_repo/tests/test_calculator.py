import pytest

from calculator import add, multiply, safe_divide, subtract


def test_add():
    assert add(1, 2) == 3
    assert add(-1, 1) == 0
    assert add(0, 0) == 0


def test_subtract():
    assert subtract(5, 3) == 2
    assert subtract(0, 4) == -4


def test_multiply():
    assert multiply(3, 4) == 12
    assert multiply(0, 5) == 0


def test_safe_divide():
    assert safe_divide(6, 2) == 3
    assert safe_divide(1, 0) == 0.0

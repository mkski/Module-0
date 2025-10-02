"""Collection of the core mathematical operators used throughout the code base."""

import math
from typing import Callable


def is_close(x: float, y: float) -> bool:
    """Check if two numbers are close.

    "Close" is defined as being within 1e-2 of each other.
    """
    return abs(x - y) < 1e-2


def mul(x: float, y: float) -> float:
    """Multiply two numbers."""
    return x * y


def id(x: float) -> float:
    """Identity function."""
    return x


def add(x: float, y: float) -> float:
    """Add two numbers."""
    return x + y


def neg(x: float) -> float:
    """Negate a number."""
    return -x


def lt(x: float, y: float) -> float:
    """Check if x is less than y. Returns 1.0 if true, 0.0 otherwise."""
    return 1.0 if x < y else 0.0


def eq(x: float, y: float) -> float:
    """Check if x is equal to y. Returns 1.0 if true, 0.0 otherwise."""
    return 1.0 if x == y else 0.0


def max(x: float, y: float) -> float:
    """Return the maximum of x and y."""
    return x if x > y else y


def exp(x: float) -> float:
    """Exponentiation function."""
    return math.e**x


def sigmoid(x: float) -> float:
    """Sigmoid function."""
    if x >= 0:
        return 1.0 / (1.0 + exp(-x))
    else:
        return exp(x) / (1.0 + exp(x))


def relu(x: float) -> float:
    """ReLU function."""
    return x if x > 0 else 0.0


def relu_back(x: float, d: float) -> float:
    """Backward pass for ReLU function."""
    return d if x > 0 else 0.0


def log(x: float) -> float:
    """Natural logarithm function."""
    return math.log(x)


def log_back(x: float, d: float) -> float:
    """Backward pass for natural logarithm function."""
    return (1.0 / x) * d


def inv(x: float) -> float:
    """Inverse function."""
    return 1.0 / x


def inv_back(x: float, d: float) -> float:
    """Backward pass for inverse function."""
    return -1.0 / (x * x) * d


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


# TODO: Implement for Task 0.3.

OneArgFn = Callable[[float], float]
TwoArgFn = Callable[[float, float], float]


def map(fn: OneArgFn, nums: list[float]) -> list[float]:
    """Map a function over a list."""
    return [fn(x) for x in nums]


def zipWith(fn: TwoArgFn, nums1: list[float], nums: list[float]):
    """Zip two lists together with a function."""
    return [fn(x, y) for x, y in zip(nums1, nums)]


def reduce(fn: TwoArgFn, nums: list[float]) -> float:
    """Reduce a list with a binary function."""
    if len(nums) == 0:
        return 0.0
    result = nums[0]
    for n in nums[1:]:
        result = fn(result, n)
    return result


def negList(nums: list[float]) -> list[float]:
    """Negate a list of numbers."""
    return map(neg, nums)


def addLists(nums1: list[float], nums2: list[float]) -> list[float]:
    """Add two lists of numbers."""
    return zipWith(add, nums1, nums2)


def sum(nums: list[float]) -> float:
    """Sum a list of numbers."""
    return reduce(add, nums)


def prod(nums: list[float]) -> float:
    """Product of a list of numbers."""
    if len(nums) == 0:
        return 1.0
    return reduce(mul, nums)

"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, List, Any, Iterable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# mul - Multiplies two numbers
# id - Returns the input unchanged
# add - Adds two numbers
# neg - Negates a number
# lt - Checks if one number is less than another
# eq - Checks if two numbers are equal
# max - Returns the larger of two numbers
# is_close - Checks if two numbers are close in value
# sigmoid - Calculates the sigmoid function
# relu - Applies the ReLU activation function
# log - Calculates the natural logarithm
# exp - Calculates the exponential function
# inv - Calculates the reciprocal
# log_back - Computes the derivative of log times a second arg
# inv_back - Computes the derivative of reciprocal times a second arg
# relu_back - Computes the derivative of ReLU times a second arg
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# TODO: Implement for Task 0.1.
def add(x: float, y: float) -> float:
    """Add two floating-point numbers."""
    return x + y


def mul(x: float, y: float) -> float:
    """Multiply two floating-point numbers."""
    return x * y


def id(input: float) -> float:
    """Returns the input unchanged"""
    return input


def neg(input: float) -> float:
    """Returns the negation of the input"""
    return -input


def lt(a: float, b: float) -> float:
    """Returns 1.0 if a is less than b, otherwise 0.0"""
    return 1.0 if a < b else 0.0


def eq(a: float, b: float) -> float:
    """Returns 1.0 if a is equal to b, otherwise 0.0"""
    return 1.0 if a == b else 0.0


def max(a: float, b: float) -> float:
    """Returns the larger of two numbers"""
    return a if a > b else b


def is_close(x: float, y: float) -> float:
    """Returns 1.0 if two numbers are close within a tolerance, otherwise returns 0.0"""
    return abs(x - y) < 1e-2


def sigmoid(input: float) -> float:
    """Calculates sigmoid of input using sigmoid function"""
    if input >= 0:
        return 1.0 / (1.0 + math.exp(-input))
    else:
        return math.exp(input) / (1.0 + math.exp(input))


def relu(input: float) -> float:
    """Returns the ReLU of the input"""
    return input if input > 0 else 0.0


def log(input: float) -> float:
    """Calculates natural logarithm of input"""
    return math.log(input)


def exp(input: float) -> float:
    """Calculates exponential of input"""
    return math.exp(input)


def inv(input: float) -> float:
    """Calculates reciprocal of input"""
    return 1.0 / input


def log_back(input: float, d_output: float) -> float:
    """Returns the derivative of log(x) * d_output"""
    return d_output / input


def inv_back(input: float, d_output: float) -> float:
    """Returns the derivative of reciprocal * d_output"""
    epsilon = 1e-2
    if abs(input) < epsilon:
        return 0.0
    return -d_output / (input * input)


def relu_back(input: float, d_output: float) -> float:
    """Returns the derivative of ReLU * d_output"""
    return d_output if input > 0 else 0.0


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


def map(fn: Callable[[Any], Any], lst: List[Any]) -> List[Any]:
    """Applies a given function to each element of a list.

    Args:
    ----
        fn: A function that takes one argument and returns a value.
        lst: A list of elements to which the function will be applied.

    Returns:
    -------
        A new list with the function applied to each element of the original list.

    """
    return [fn(x) for x in lst]


def zipWith(
    fn: Callable[[Any, Any], Any], lst1: List[Any], lst2: List[Any]
) -> List[Any]:
    """Combines two lists element-wise using a given function.

    Args:
    ----
        fn: A function that takes two arguments and returns a value.
        lst1: The first list to combine.
        lst2: The second list to combine.

    Returns:
    -------
        A new list where each element is the result of applying the function to corresponding elements from the two input lists.

    """
    return [fn(x, y) for x, y in zip(lst1, lst2)]


def reduce(fn: Callable[[Any, Any], Any], start: Any, lst: Iterable[Any]) -> Any:
    """Reduces a list to a single value by applying a function cumulatively, starting from an initial value.

    Args:
    ----
        fn: A function that takes two arguments and returns a value.
        start: The initial value to start the reduction.
        lst: The list to reduce.

    Returns:
    -------
        A single value obtained by applying the function cumulatively to the list elements, starting from the initial value.

    """
    result = start
    for x in lst:
        result = fn(result, x)
    return result


def negList(lst: List[float]) -> List[float]:
    """Negates all elements in a list.

    Args:
    ----
        lst: A list of float values.

    Returns:
    -------
        A new list with each element negated.

    """
    return map(lambda x: -x, lst)


def addLists(lst1: List[float], lst2: List[float]) -> List[float]:
    """Adds corresponding elements from two lists.

    Args:
    ----
        lst1: The first list of float values.
        lst2: The second list of float values.

    Returns:
    -------
        A new list where each element is the sum of corresponding elements from the input lists.

    """
    return zipWith(lambda x, y: x + y, lst1, lst2)


def sum(lst: List[float]) -> float:
    """Computes the sum of all elements in a list.

    Args:
    ----
        lst: A list of float values.

    Returns:
    -------
        The sum of all elements in the list.

    """
    return reduce(lambda x, y: x + y, 0.0, lst)


def prod(lst: Iterable[float]) -> float:
    """Computes the product of all elements in a list.

    Args:
    ----
        lst: A list of float values.

    Returns:
    -------
        The product of all elements in the list.

    """
    # return reduce(lambda x, y: x * y, lst, 1.0)
    return reduce(mul, 1.0, lst)

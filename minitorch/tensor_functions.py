"""Implementation of the autodifferentiation Functions for Tensor."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import numpy as np

import minitorch

from . import operators
from .autodiff import Context
from .tensor_ops import SimpleBackend, TensorBackend

if TYPE_CHECKING:
    from typing import Any, List, Tuple

    from .tensor import Tensor
    from .tensor_data import UserIndex, UserShape


def wrap_tuple(x: Any) -> tuple:  # type: ignore
    """Turn a possible value into a tuple"""
    if isinstance(x, tuple):
        return x
    return (x,)


# Constructors
class Function:
    @classmethod
    def _backward(cls, ctx: Context, grad_out: Tensor) -> Tuple[Tensor, ...]:
        return wrap_tuple(cls.backward(ctx, grad_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: Tensor) -> Tensor:
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: Tensor) -> Tensor:
        """Call the forward function and track history"""
        raw_vals = []
        need_grad = False
        for v in vals:
            if v.requires_grad():
                need_grad = True
            raw_vals.append(v.detach())

        # Create the context.
        ctx = Context(not need_grad)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        # assert isinstance(c, Tensor), "Expected return type Tensor got %s" % (
        #     type(c)
        # )

        # Create a new variable from the result with a new history.
        back = None
        if need_grad:
            back = minitorch.History(cls, ctx, vals)
        return minitorch.Tensor(c._tensor, back, backend=c.backend)


class Neg(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Computes the element-wise negation of the input tensor.

        Args:
        ----
        ctx: Context
            The context object to save any information for the backward pass.
        t1: Tensor
            The input tensor to be negated.

        Returns:
        -------
        Tensor
            A new tensor where each element is the negation of the corresponding
            element in the input tensor.

        """
        return t1.f.neg_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Computes the gradient of the negation operation for the backward pass.

        Args:
        ----
        ctx: Context
            The context object that holds information from the forward pass.
        grad_output: Tensor
            The gradient of the output with respect to some loss.

        Returns:
        -------
        Tensor
            The negated gradient of the output tensor.

        """
        return grad_output.f.neg_map(grad_output)


class Inv(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Computes the element-wise inverse (1/x) of the input tensor.

        Args:
        ----
        ctx: Context
            The context object to save any information for the backward pass.
        t1: Tensor
            The input tensor for which the element-wise inverse is to be computed.

        Returns:
        -------
        Tensor
            A new tensor where each element is the inverse of the corresponding
            element in the input tensor.

        """
        ctx.save_for_backward(t1)
        return t1.f.inv_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Computes the gradient of the inverse operation for the backward pass.

        Args:
        ----
        ctx: Context
            The context object that holds information from the forward pass.
        grad_output: Tensor
            The gradient of the output with respect to some loss.

        Returns:
        -------
        Tensor
            The gradient of the input tensor after applying the inverse function.

        """
        (t1,) = ctx.saved_values
        return grad_output.f.inv_back_zip(t1, grad_output)


class Add(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Computes the element-wise addition of two input tensors.

        Args:
        ----
        ctx: Context
            The context object to save any information for the backward pass.
        t1: Tensor
            The first input tensor for the addition operation.
        t2: Tensor
            The second input tensor for the addition operation.

        Returns:
        -------
        Tensor
            A new tensor where each element is the sum of the corresponding
            elements in the two input tensors.

        """
        return t1.f.add_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Computes the gradient of the addition operation for the backward pass.

        Args:
        ----
        ctx: Context
            The context object that holds information from the forward pass.
        grad_output: Tensor
            The gradient of the output with respect to some loss.

        Returns:
        -------
        Tuple[Tensor, Tensor]
            A tuple containing the gradient of the output with respect to each
            of the input tensors (t1 and t2). Both gradients are equal to the
            grad_output since the derivative of x + y with respect to x or y is 1.

        """
        return grad_output, grad_output


class All(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, dim: Tensor) -> Tensor:
        """Return 1 if all are true"""
        if dim is not None:
            return a.f.mul_reduce(a, int(dim.item()))
        else:
            return a.f.mul_reduce(a.contiguous().view(int(operators.list(a.shape))), 0)


# TODO: Implement for Task 2.3.
class Mul(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Computes the element-wise multiplication of two input tensors.

        Args:
        ----
        ctx: Context
            The context object to save information for the backward pass.
        t1: Tensor
            The first input tensor.
        t2: Tensor
            The second input tensor.

        Returns:
        -------
        Tensor
            A tensor where each element is the product of the corresponding elements in t1 and t2.

        """
        ctx.save_for_backward(t1, t2)
        return t1.f.mul_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Computes the gradient of the multiplication operation for the backward pass.

        Args:
        ----
        ctx: Context
            The context object containing saved information from the forward pass.
        grad_output: Tensor
            The gradient of the output tensor with respect to some loss.

        Returns:
        -------
        Tuple[Tensor, Tensor]
            Gradients of the output with respect to each input tensor.

        """
        t1, t2 = ctx.saved_values
        grad_t1 = t2 * grad_output
        grad_t2 = t1 * grad_output
        return grad_t1, grad_t2


class Sigmoid(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Computes the element-wise sigmoid function on the input tensor.

        Args:
        ----
        ctx: Context
            The context object to save information for the backward pass.
        t1: Tensor
            The input tensor.

        Returns:
        -------
        Tensor
            A tensor where each element is the sigmoid of the corresponding element in t1.

        """
        out = t1.f.sigmoid_map(t1)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Computes the gradient of the sigmoid function for the backward pass.

        Args:
        ----
        ctx: Context
            The context object containing saved information from the forward pass.
        grad_output: Tensor
            The gradient of the output tensor with respect to some loss.

        Returns:
        -------
        Tensor
            The gradient of the input tensor after applying the sigmoid function.

        """
        (out,) = ctx.saved_values
        one_tensor = out._ensure_tensor(1)
        sigmoid_derivative = out.f.mul_zip(
            out, out.f.add_zip(one_tensor, out.f.neg_map(out))
        )
        return grad_output.f.mul_zip(sigmoid_derivative, grad_output)


class ReLU(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Computes the element-wise ReLU (Rectified Linear Unit) function on the input tensor.

        Args:
        ----
        ctx: Context
            The context object to save information for the backward pass.
        t1: Tensor
            The input tensor.

        Returns:
        -------
        Tensor
            A tensor where each element is the ReLU of the corresponding element in t1.

        """
        ctx.save_for_backward(t1)
        return t1.f.relu_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Computes the gradient of the ReLU function for the backward pass.

        Args:
        ----
        ctx: Context
            The context object containing saved information from the forward pass.
        grad_output: Tensor
            The gradient of the output tensor with respect to some loss.

        Returns:
        -------
        Tensor
            The gradient of the input tensor after applying the ReLU function.

        """
        (t1,) = ctx.saved_values
        return grad_output.f.relu_back_zip(t1, grad_output)


class Log(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Computes the element-wise natural logarithm of the input tensor.

        Args:
        ----
        ctx: Context
            The context object to save information for the backward pass.
        t1: Tensor
            The input tensor.

        Returns:
        -------
        Tensor
            A tensor where each element is the logarithm of the corresponding element in t1.

        """
        ctx.save_for_backward(t1)
        return t1.f.log_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Computes the gradient of the logarithm function for the backward pass.

        Args:
        ----
        ctx: Context
            The context object containing saved information from the forward pass.
        grad_output: Tensor
            The gradient of the output tensor with respect to some loss.

        Returns:
        -------
        Tensor
            The gradient of the input tensor after applying the logarithm function.

        """
        (t1,) = ctx.saved_values
        return grad_output / t1


class Exp(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Computes the element-wise exponential of the input tensor.

        Args:
        ----
        ctx: Context
            The context object to save information for the backward pass.
        t1: Tensor
            The input tensor.

        Returns:
        -------
        Tensor
            A tensor where each element is the exponential of the corresponding element in t1.

        """
        out = t1.f.exp_map(t1)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Computes the gradient of the exponential function for the backward pass.

        Args:
        ----
        ctx: Context
            The context object containing saved information from the forward pass.
        grad_output: Tensor
            The gradient of the output tensor with respect to some loss.

        Returns:
        -------
        Tensor
            The gradient of the input tensor after applying the exponential function.

        """
        (out,) = ctx.saved_values
        return grad_output * out


class Sum(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, dim: Tensor) -> Tensor:
        """Computes the sum of elements along the specified dimension of the input tensor.

        Args:
        ----
        ctx: Context
            The context object to save information for the backward pass.
        a: Tensor
            The input tensor to be summed.
        dim: Tensor
            The dimension along which to perform the summation.

        Returns:
        -------
        Tensor
            A tensor containing the sum along the specified dimension.

        """
        ctx.save_for_backward(a.shape, dim)
        return a.f.add_reduce(a, int(dim.item()))

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Computes the gradient of the summation operation for the backward pass.

        Args:
        ----
        ctx: Context
            The context object containing saved information from the forward pass.
        grad_output: Tensor
            The gradient of the output tensor with respect to some loss.

        Returns:
        -------
        Tuple[Tensor, float]
            The gradient of the input tensor and a zero since the dimension is non-differentiable.

        """
        (_, _) = ctx.saved_values
        return grad_output, 0.0


class LT(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Computes the element-wise comparison (less than) between two input tensors.

        Args:
        ----
        ctx: Context
            The context object to save information for the backward pass.
        t1: Tensor
            The first input tensor.
        t2: Tensor
            The second input tensor.

        Returns:
        -------
        Tensor
            A tensor where each element is 1 if the corresponding element in t1 is less than that in t2, otherwise 0.

        """
        return t1.f.lt_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Returns zero gradients since the less-than operation is non-differentiable.

        Args:
        ----
        ctx: Context
            The context object containing saved information from the forward pass.
        grad_output: Tensor
            The gradient of the output tensor with respect to some loss.

        Returns:
        -------
        Tuple[Tensor, Tensor]
            Two zero tensors with the same shape as grad_output.

        """
        zero_tensor1 = grad_output.zeros(grad_output.shape)
        zero_tensor2 = grad_output.zeros(grad_output.shape)
        return zero_tensor1, zero_tensor2


class EQ(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Computes the element-wise comparison (equality) between two input tensors.

        Args:
        ----
        ctx: Context
            The context object to save information for the backward pass.
        t1: Tensor
            The first input tensor.
        t2: Tensor
            The second input tensor.

        Returns:
        -------
        Tensor
            A tensor where each element is 1 if the corresponding elements in t1 and t2 are equal, otherwise 0.

        """
        return t1.f.eq_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Returns zero gradients since the equality operation is non-differentiable.

        Args:
        ----
        ctx: Context
            The context object containing saved information from the forward pass.
        grad_output: Tensor
            The gradient of the output tensor with respect to some loss.

        Returns:
        -------
        Tuple[Tensor, Tensor]
            Two zero tensors with the same shape as grad_output.

        """
        zero_tensor1 = grad_output.zeros(grad_output.shape)
        zero_tensor2 = grad_output.zeros(grad_output.shape)
        return zero_tensor1, zero_tensor2


class IsClose(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Computes element-wise comparison to check if elements in t1 are close to those in t2.

        Args:
        ----
        ctx: Context
            The context object to save information for the backward pass.
        t1: Tensor
            The first input tensor.
        t2: Tensor
            The second input tensor.

        Returns:
        -------
        Tensor
            A tensor where each element is 1 if the corresponding elements in t1 and t2 are close, otherwise 0.

        """
        return t1.f.is_close_zip(t1, t2)


class Permute(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, order: Tensor) -> Tensor:
        """Permutes the dimensions of the input tensor according to the specified order.

        Args:
        ----
        ctx: Context
            The context object to save information for the backward pass.
        a: Tensor
            The input tensor to be permuted.
        order: Tensor
            A tensor representing the new order of dimensions for the permutation.

        Returns:
        -------
        Tensor
            A new tensor with its dimensions permuted according to the specified order.

        """
        # Extract the order as integers from the tensor storage
        int_order = [int(i) for i in order._tensor._storage]
        ctx.save_for_backward(int_order)
        new_tensor_data = a._tensor.permute(*int_order)
        return a._new(new_tensor_data)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Computes the gradient of the permute operation for the backward pass.

        Args:
        ----
        ctx: Context
            The context object containing saved information from the forward pass.
        grad_output: Tensor
            The gradient of the output tensor with respect to some loss.

        Returns:
        -------
        Tuple[Tensor, float]
            The gradient of the input tensor after reversing the permutation and
            a zero value for the non-differentiable 'order' parameter.

        """
        (order,) = ctx.saved_values

        # Calculate the inverse permutation
        order_map = {v: i for i, v in enumerate(order)}
        inv_order = [order_map[i] for i in range(len(order))]

        # Apply the inverse permutation to the gradient
        grad_input = grad_output._new(grad_output._tensor.permute(*inv_order))

        # Return a zero value for the non-differentiable parameter 'order'
        return grad_input, 0.0


class View(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, shape: Tensor) -> Tensor:
        """Reshapes the input tensor to the specified shape, ensuring the tensor is contiguous.

        Args:
        ----
        ctx: Context
            The context object to save information for the backward pass.
        a: Tensor
            The input tensor to be reshaped.
        shape: Tensor
            A tensor representing the new shape of the tensor.

        Returns:
        -------
        Tensor
            A new tensor with the specified shape.

        """
        ctx.save_for_backward(a.shape)
        assert a._tensor.is_contiguous(), "Must be contiguous to view"
        shape2 = [int(shape[i]) for i in range(shape.size)]
        return minitorch.Tensor.make(
            a._tensor._storage, tuple(shape2), backend=a.backend
        )

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Matrix Multiply backward (module 3)"""
        (original,) = ctx.saved_values
        return (
            minitorch.Tensor.make(
                grad_output._tensor._storage, original, backend=grad_output.backend
            ),
            0.0,
        )


class Copy(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor) -> Tensor:
        """Id function makes contiguous"""
        return a.f.id_map(a)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Undo"""
        return grad_output


class MatMul(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Matrix Multiply Forward (module 3)"""
        ctx.save_for_backward(t1, t2)
        return t1.f.matrix_multiply(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Matrix Multiply backward (module 3)"""
        t1, t2 = ctx.saved_values

        def transpose(a: Tensor) -> Tensor:
            order = list(range(a.dims))
            order[-2], order[-1] = order[-1], order[-2]
            return a._new(a._tensor.permute(*order))

        return (
            grad_output.f.matrix_multiply(grad_output, transpose(t2)),
            grad_output.f.matrix_multiply(transpose(t1), grad_output),
        )


# Helpers for Constructing tensors
def zeros(shape: UserShape, backend: TensorBackend = SimpleBackend) -> Tensor:
    """Produce a zero tensor of size `shape`.

    Args:
    ----
        shape : shape of tensor
        backend : tensor backend

    Returns:
    -------
        new tensor

    """
    return minitorch.Tensor.make(
        [0.0] * int(operators.prod(list(shape))), shape, backend=backend
    )


def rand(
    shape: UserShape,
    backend: TensorBackend = SimpleBackend,
    requires_grad: bool = False,
) -> Tensor:
    """Produce a random tensor of size `shape`.

    Args:
    ----
        shape : shape of tensor
        backend : tensor backend
        requires_grad : turn on autodifferentiation

    Returns:
    -------
        :class:`Tensor` : new tensor

    """
    vals = [random.random() for _ in range(int(operators.prod(list(shape))))]
    tensor = minitorch.Tensor.make(vals, shape, backend=backend)
    tensor.requires_grad_(requires_grad)
    return tensor


def _tensor(
    ls: Any,
    shape: UserShape,
    backend: TensorBackend = SimpleBackend,
    requires_grad: bool = False,
) -> Tensor:
    """Produce a tensor with data ls and shape `shape`.

    Args:
    ----
        ls: data for tensor
        shape: shape of tensor
        backend: tensor backend
        requires_grad: turn on autodifferentiation

    Returns:
    -------
        new tensor

    """
    tensor = minitorch.Tensor.make(ls, shape, backend=backend)
    tensor.requires_grad_(requires_grad)
    return tensor


def tensor(
    ls: Any, backend: TensorBackend = SimpleBackend, requires_grad: bool = False
) -> Tensor:
    """Produce a tensor with data and shape from ls

    Args:
    ----
        ls: data for tensor
        backend : tensor backend
        requires_grad : turn on autodifferentiation

    Returns:
    -------
        :class:`Tensor` : new tensor

    """

    def shape(ls: Any) -> List[int]:
        if isinstance(ls, (list, tuple)):
            return [len(ls)] + shape(ls[0])
        else:
            return []

    def flatten(ls: Any) -> List[float]:
        if isinstance(ls, (list, tuple)):
            return [y for x in ls for y in flatten(x)]
        else:
            return [ls]

    cur = flatten(ls)
    shape2 = shape(ls)
    return _tensor(cur, tuple(shape2), backend=backend, requires_grad=requires_grad)


# Gradient check for tensors


def grad_central_difference(
    f: Any, *vals: Tensor, arg: int = 0, epsilon: float = 1e-6, ind: UserIndex
) -> float:
    """Computes the central difference approximation of the gradient of a function with respect to one argument."""
    x = vals[arg]
    up = zeros(x.shape)
    up[ind] = epsilon
    vals1 = [x if j != arg else x + up for j, x in enumerate(vals)]
    vals2 = [x if j != arg else x - up for j, x in enumerate(vals)]
    delta: Tensor = f(*vals1).sum() - f(*vals2).sum()

    return delta[0] / (2.0 * epsilon)


def grad_check(f: Any, *vals: Tensor) -> None:
    """Check whether autodiff matches central difference."""
    for x in vals:
        x.requires_grad_(True)
        x.zero_grad_()
    random.seed(10)
    out = f(*vals)
    out.sum().backward()
    err_msg = """

Gradient check error for function %s.

Input %s

Received derivative %f for argument %d and index %s,
but was expecting derivative %f from central difference.

"""

    for i, x in enumerate(vals):
        ind = x._tensor.sample()
        check = grad_central_difference(f, *vals, arg=i, ind=ind)
        assert x.grad is not None
        np.testing.assert_allclose(
            x.grad[ind],
            check,
            1e-2,
            1e-2,
            err_msg=err_msg % (f, vals, x.grad[ind], i, ind, check),
        )

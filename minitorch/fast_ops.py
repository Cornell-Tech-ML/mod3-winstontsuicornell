from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar, Any

import numpy as np
from numba import prange
from numba import njit as _njit

from .tensor_data import (
    MAX_DIMS,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps

if TYPE_CHECKING:
    from typing import Callable, Optional

    from .tensor import Tensor
    from .tensor_data import Index, Shape, Storage, Strides

# TIP: Use `NUMBA_DISABLE_JIT=1 pytest tests/ -m task3_1` to run these tests without JIT.

# This code will JIT compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.
Fn = TypeVar("Fn")


def njit(fn: Fn, **kwargs: Any) -> Fn:
    return _njit(inline="always", **kwargs)(fn)  # type: ignore


to_index = njit(to_index)
index_to_position = njit(index_to_position)
broadcast_index = njit(broadcast_index)


class FastOps(TensorOps):
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """See `tensor_ops.py`"""
        # This line JIT compiles your tensor_map
        f = tensor_map(njit(fn))

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)
            f(*out.tuple(), *a.tuple())
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        """See `tensor_ops.py`"""
        f = tensor_zip(njit(fn))

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            f(*out.tuple(), *a.tuple(), *b.tuple())
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        """See `tensor_ops.py`"""
        f = tensor_reduce(njit(fn))

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = 1

            # Other values when not sum.
            out = a.zeros(tuple(out_shape))
            out._tensor._storage[:] = start

            f(*out.tuple(), *a.tuple(), dim)
            return out

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        """Batched tensor matrix multiply ::

            for n:
              for i:
                for j:
                  for k:
                    out[n, i, j] += a[n, i, k] * b[n, k, j]

        Where n indicates an optional broadcasted batched dimension.

        Should work for tensor shapes of 3 dims ::

            assert a.shape[-1] == b.shape[-2]

        Args:
        ----
            a : tensor data a
            b : tensor data b

        Returns:
        -------
            New tensor data

        """
        # Make these always be a 3 dimensional multiply
        both_2d = 0
        if len(a.shape) == 2:
            a = a.contiguous().view(1, a.shape[0], a.shape[1])
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d += 1
        both_2d = both_2d == 2

        ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        assert a.shape[-1] == b.shape[-2]
        out = a.zeros(tuple(ls))

        tensor_matrix_multiply(*out.tuple(), *a.tuple(), *b.tuple())

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out


# Implementations


def tensor_map(
    fn: Callable[[float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        # out_index = np.zeros(len(out_shape), dtype=np.int32)  # Buffer for output index
        # for i in prange(len(out)):  # Parallel loop
        #     to_index(i, out_shape, out_index)  # Convert flat index to multidimensional index
        #     in_pos = index_to_position(out_index, in_strides)  # Get position in input storage
        #     out[i] = fn(in_storage[in_pos])  # Apply function and store result in output
        


        out_size = len(out)
        out_len = len(out_shape)
        in_len = len(in_shape)
        # out_index = np.full(out_len, -1)
        # in_index = np.full(in_len, -1)
        if ((len(in_shape) == len(out_shape)) and (in_shape == out_shape).all() and (in_strides == out_strides).all()):
            for i in prange(len(out)):
                out[i] = fn(in_storage[i])
        else:
            for i in prange(out_size):
                out_index = np.empty(out_len, np.int32)
                in_index = np.empty(in_len, np.int32)
                # Gaining the out_index
                to_index(i, out_shape, out_index)
                # Gain the in_index from the out_index
                broadcast_index(out_index, out_shape, in_shape, in_index)
                # Finding the in_storage position from the in_index
                in_pos = index_to_position(in_index, in_strides)
                # Finding the out_storage position from the out_index
                # Throwing it in storage
                out[i] = fn(in_storage[in_pos])
    # return njit(parallel=True)(_map)
    return _map




def tensor_zip(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """NUMBA higher-order tensor zip function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * When `out`, `a`, `b` are stride-aligned, avoid indexing

    Args:
    ----
        fn: function maps two floats to float to apply.

    Returns:
    -------
        Tensor zip function.

    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        # TODO: Implement for Task 3.1.
        # out_index = np.zeros(len(out_shape), dtype=np.int32)  # Buffer for output index
        # for i in prange(len(out)):  # Parallel loop
        #     to_index(i, out_shape, out_index)  # Convert flat index to multidimensional index
        #     a_pos = index_to_position(out_index, a_strides)  # Get position in a_storage
        #     b_pos = index_to_position(out_index, b_strides)  # Get position in b_storage
        #     out[i] = fn(a_storage[a_pos], b_storage[b_pos])  # Apply function and store result
        out_size = len(out)
        out_len = len(out_shape)
        a_len = len(a_shape)
        b_len = len(b_shape)
        if ((a_len == b_len) and (a_strides == b_strides).all() and (a_shape == b_shape).all() and (b_strides == out_strides).all()):
            for i in prange(out_size):
                out[i] = fn(a_storage[i], b_storage[i])
        else:
            for i in prange(out_size):
                out_index = np.empty(out_len, np.int32)
                a_index = np.empty(a_len, np.int32)
                b_index = np.empty(b_len, np.int32)
                # Gaining the out_index
                to_index(i, out_shape, out_index)
                # Gain the a_index from the out_index
                broadcast_index(out_index, out_shape, a_shape, a_index)
                # Gain the b_index from the out_index
                broadcast_index(out_index, out_shape, b_shape, b_index)
                # Finding the a_storage position from the a_index
                a_pos = index_to_position(a_index, a_strides)
                # Finding the b_storage position from the b_index
                b_pos = index_to_position(b_index, b_strides)
                # Finding the out_storage position from the out_index
                #out_pos = index_to_position(out_index, out_strides)
                # Throwing it in storage
                out[i] = fn(a_storage[a_pos], b_storage[b_pos])


    # return njit(parallel=True)(_zip)  # type: ignore
    return _zip



def tensor_reduce(
    fn: Callable[[float, float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """NUMBA higher-order tensor reduce function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * Inner-loop should not call any functions or write non-local variables

    Args:
    ----
        fn: reduction function mapping two floats to float.

    Returns:
    -------
        Tensor reduce function

    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
    ) -> None:
        # TODO: Implement for Task 3.1.
        out_index = np.zeros(len(out_shape), dtype=np.int32)  # Buffer for output index
        in_index = np.zeros(len(a_shape), dtype=np.int32)  # Buffer for input index
        for i in prange(len(out)):  # Parallel loop
            to_index(i, out_shape, out_index)  # Convert flat index to multidimensional index
            reduced_value = out[i]  # Start with the initial value in the output tensor
            for j in range(a_shape[reduce_dim]):  # Iterate along the reduction dimension
                in_index[:] = out_index  # Copy current index
                in_index[reduce_dim] = j  # Update index for the reduction dimension
                in_pos = index_to_position(in_index, a_strides)  # Get position in input storage
                reduced_value = fn(reduced_value, a_storage[in_pos])  # Apply reduction
            out[i] = reduced_value  # Write reduced result to output tensor

        # reduce_size = a_shape[reduce_dim]
        # for i in prange(len(out)):
        #     out_index: Index = np.zeros(MAX_DIMS, np.int32)
        #     to_index(i, out_shape, out_index)
        #     o = index_to_position(out_index, out_strides)
        #     for s in range(reduce_size):
        #         out_index[reduce_dim] = s
        #         j = index_to_position(out_index, a_strides)
        #         out[o] = fn(out[o], a_storage[j])
        

    # return njit(parallel=True)(_reduce)
    return _reduce


def _tensor_matrix_multiply(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
    """NUMBA tensor matrix multiply function.

    Args:
        out (Storage): storage for `out` tensor
        out_shape (Shape): shape for `out` tensor
        out_strides (Strides): strides for `out` tensor
        a_storage (Storage): storage for `a` tensor
        a_shape (Shape): shape for `a` tensor
        a_strides (Strides): strides for `a` tensor
        b_storage (Storage): storage for `b` tensor
        b_shape (Shape): shape for `b` tensor
        b_strides (Strides): strides for `b` tensor

    Returns:
        None: Fills in `out`.

    """
    # batch_size, out_rows, out_cols = out_shape
    # a_rows, a_cols = a_shape[-2], a_shape[-1]
    # b_rows, b_cols = b_shape[-2], b_shape[-1]

    # # Ensure matrix dimensions match for multiplication
    # assert a_cols == b_rows

    # for n in prange(batch_size):  # Parallelize over batches
    #     for i in range(out_rows):  # Iterate over rows of the output
    #         for j in range(out_cols):  # Iterate over columns of the output
    #             out_index = n * out_strides[0] + i * out_strides[1] + j * out_strides[2]
    #             sum_value = 0.0  # Local variable to accumulate the dot product

    #             for k in range(a_cols):  # Iterate over the inner dimension
    #                 a_index = n * a_strides[0] + i * a_strides[1] + k * a_strides[2]
    #                 b_index = n * b_strides[0] + k * b_strides[1] + j * b_strides[2]

    #                 sum_value += a_storage[a_index] * b_storage[b_index]

    #             out[out_index] = sum_value  # Write the result to the output tensor
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0
    for x in prange(out_shape[0]):
        for y in prange(out_shape[1]):
            for z in prange(out_shape[2]):
                val = 0.0
                posA = x * a_batch_stride + y * a_strides[1]
                posB = x * b_batch_stride + z * b_strides[2]
                for a in range(a_shape[2]):
                    val += a_storage[posA] * b_storage[posB]
                    posA += a_strides[2]
                    posB += b_strides[1]
                outPos = x * out_strides[0] + y * out_strides[1] + z * out_strides[2]
                out[outPos] = val



tensor_matrix_multiply = njit(_tensor_matrix_multiply, parallel=True)
assert tensor_matrix_multiply is not None

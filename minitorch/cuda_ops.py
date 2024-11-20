# type: ignore
# Currently pyright doesn't support numba.cuda

from typing import Callable, Optional, TypeVar, Any

import numba
from numba import cuda
from numba.cuda import jit as _jit
from .tensor import Tensor
from .tensor_data import (
    MAX_DIMS,
    Shape,
    Storage,
    Strides,
    TensorData,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps

FakeCUDAKernel = Any

# This code will CUDA compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.

Fn = TypeVar("Fn")


def device_jit(fn: Fn, **kwargs) -> Fn:
    return _jit(device=True, **kwargs)(fn)  # type: ignore


def jit(fn, **kwargs) -> FakeCUDAKernel:
    return _jit(**kwargs)(fn)  # type: ignore


to_index = device_jit(to_index)
index_to_position = device_jit(index_to_position)
broadcast_index = device_jit(broadcast_index)

THREADS_PER_BLOCK = 32


class CudaOps(TensorOps):
    cuda = True

    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """See `tensor_ops.py`"""
        cufn: Callable[[float], float] = device_jit(fn)
        f = tensor_map(cufn)

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)

            # Instantiate and run the cuda kernel.
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
            f[blockspergrid, threadsperblock](*out.tuple(), out.size, *a.tuple())  # type: ignore
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        cufn: Callable[[float, float], float] = device_jit(fn)
        f = tensor_zip(cufn)

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + (threadsperblock - 1)) // threadsperblock
            f[blockspergrid, threadsperblock](  # type: ignore
                *out.tuple(), out.size, *a.tuple(), *b.tuple()
            )
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        cufn: Callable[[float, float], float] = device_jit(fn)
        f = tensor_reduce(cufn)

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = (a.shape[dim] - 1) // 1024 + 1
            out_a = a.zeros(tuple(out_shape))

            threadsperblock = 1024
            blockspergrid = out_a.size
            f[blockspergrid, threadsperblock](  # type: ignore
                *out_a.tuple(), out_a.size, *a.tuple(), dim, start
            )

            return out_a

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
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

        # One block per batch, extra rows, extra col
        blockspergrid = (
            (out.shape[1] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            (out.shape[2] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            out.shape[0],
        )
        threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1)

        tensor_matrix_multiply[blockspergrid, threadsperblock](
            *out.tuple(), out.size, *a.tuple(), *b.tuple()
        )

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out


# Implement


def tensor_map(
    fn: Callable[[float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """CUDA higher-order tensor map function. ::

      fn_map = tensor_map(fn)
      fn_map(out, ... )

    Args:
    ----
        fn: function mappings floats-to-floats to apply.

    Returns:
    -------
        Tensor map function.

    """

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        in_index = cuda.local.array(MAX_DIMS, numba.int32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        # TODO: Implement for Task 3.3.
        #raise NotImplementedError("Need to implement for Task 3.3")
        if i < out_size:
            # Convert linear index to multidimensional indices
            to_index(i, out_shape, out_index)
            out_pos = index_to_position(out_index, out_strides)

            # Broadcast input tensor indices
            broadcast_index(out_index, out_shape, in_shape, in_index)
            in_pos = index_to_position(in_index, in_strides)

            # Apply the function
            out[out_pos] = fn(in_storage[in_pos])

    return cuda.jit()(_map)  # type: ignore


def tensor_zip(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """CUDA higher-order tensor zipWith (or map2) function ::

      fn_zip = tensor_zip(fn)
      fn_zip(out, ...)

    Args:
    ----
        fn: function mappings two floats to float to apply.

    Returns:
    -------
        Tensor zip function.

    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        a_index = cuda.local.array(MAX_DIMS, numba.int32)
        b_index = cuda.local.array(MAX_DIMS, numba.int32)

        # Compute the global thread ID
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

        if i < out_size:  # Ensure thread is within bounds
            # Convert linear index `i` to multi-dimensional index for `out`
            to_index(i, out_shape, out_index)

            # Compute the output tensor position
            out_pos = index_to_position(out_index, out_strides)

            # Broadcast the `out_index` to match `a` and `b` shapes
            broadcast_index(out_index, out_shape, a_shape, a_index)
            a_pos = index_to_position(a_index, a_strides)

            broadcast_index(out_index, out_shape, b_shape, b_index)
            b_pos = index_to_position(b_index, b_strides)

            # Apply the function `fn` to corresponding elements of `a` and `b`
            out[out_pos] = fn(a_storage[a_pos], b_storage[b_pos])

    return cuda.jit()(_zip)  # type: ignore


def _sum_practice(out: Storage, a: Storage, size: int) -> None:
    """Practice sum kernel to prepare for reduce.

    Given an array of length $n$ and out of size $next{blockDIM}$
    it should sum up each blockDim values into an out cell.

    $[a_1, a_2, ..., a_{100}]$

    |

    $[a_1 +...+ a_{31}, a_{32} + ... + a_{64}, ... ,]$

    Note: Each block must do the sum using shared memory!

    Args:
    ----
        out (Storage): storage for `out` tensor.
        a (Storage): storage for `a` tensor.
        size (int):  length of a.

    """
    BLOCK_DIM = 32

    # Shared memory for intermediate results
    cache = cuda.shared.array(BLOCK_DIM, numba.float64)
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    pos = cuda.threadIdx.x

    # Load input into shared memory
    cache[pos] = a[i] if i < size else 0.0
    cuda.syncthreads()

    # Perform reduction in shared memory
    stride = 1
    while stride < BLOCK_DIM:
        if pos % (2 * stride) == 0 and (pos + stride) < BLOCK_DIM:
            cache[pos] += cache[pos + stride]
        stride *= 2
        cuda.syncthreads()

    # Write the block result to output
    if pos == 0:
        out[cuda.blockIdx.x] = cache[0]


jit_sum_practice = cuda.jit()(_sum_practice)


def sum_practice(a: Tensor) -> TensorData:
    (size,) = a.shape
    threadsperblock = THREADS_PER_BLOCK
    blockspergrid = (size // THREADS_PER_BLOCK) + 1
    out = TensorData([0.0 for i in range(2)], (2,))
    out.to_cuda_()
    jit_sum_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, size
    )
    return out


def tensor_reduce(
    fn: Callable[[float, float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """CUDA higher-order tensor reduce function.

    Args:
    ----
        fn: reduction function maps two floats to float.

    Returns:
    -------
        Tensor reduce function.

    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
        reduce_value: float,
        start: float,
    ) -> None:
        BLOCK_DIM = 1024

        # Shared memory for intermediate results
        cache = cuda.shared.array(BLOCK_DIM, numba.float64)
        cache_pos = cuda.threadIdx.x
        block_pos = cuda.blockIdx.x

        # Calculate the output index
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        to_index(block_pos, out_shape, out_index)
        out_pos = index_to_position(out_index, out_strides)

        # Initialize shared memory with start value
        cache[cache_pos] = start

        # Reduction over the specified dimension
        for i in range(cache_pos, a_shape[reduce_dim], BLOCK_DIM):
            in_index = out_index.copy()
            in_index[reduce_dim] = i
            in_pos = index_to_position(in_index, a_strides)
            cache[cache_pos] = fn(cache[cache_pos], a_storage[in_pos])

        cuda.syncthreads()

        # Perform reduction in shared memory
        stride = 1
        while stride < BLOCK_DIM:
            if cache_pos % (2 * stride) == 0 and (cache_pos + stride) < BLOCK_DIM:
                cache[cache_pos] = fn(cache[cache_pos], cache[cache_pos + stride])
            stride *= 2
            cuda.syncthreads()

        # Write the final reduced value
        if cache_pos == 0:
            out[out_pos] = cache[0]

    return cuda.jit(_reduce)  # type: ignore


def _mm_practice(out: Storage, a: Storage, b: Storage, size: int) -> None:
    """Practice square MM kernel to prepare for matmul.

    Given a storage `out` and two storage `a` and `b`. Where we know
    both are shape [size, size] with strides [size, 1].

    Size is always < 32.

    Requirements:

    * All data must be first moved to shared memory.
    * Only read each cell in `a` and `b` once.
    * Only write to global memory once per kernel.

    Compute

    ```
     for i:
         for j:
              for k:
                  out[i, j] += a[i, k] * b[k, j]
    ```

    Args:
    ----
        out (Storage): storage for `out` tensor.
        a (Storage): storage for `a` tensor.
        b (Storage): storage for `b` tensor.
        size (int): size of the square

    """
    BLOCK_DIM = 32
    # # TODO: Implement for Task 3.3.
    # raise NotImplementedError("Need to implement for Task 3.3")
    # Allocate shared memory
    a_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    b_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)

    # Thread and block indices
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    row = cuda.blockIdx.y * cuda.blockDim.y + ty
    col = cuda.blockIdx.x * cuda.blockDim.x + tx

    # Initialize the output value
    temp = 0.0

    # Loop over tiles of the input matrices
    for tile in range((size + BLOCK_DIM - 1) // BLOCK_DIM):
        # Load a tile of `a` and `b` into shared memory
        if row < size and tile * BLOCK_DIM + tx < size:
            a_shared[ty, tx] = a[row * size + tile * BLOCK_DIM + tx]
        else:
            a_shared[ty, tx] = 0.0

        if col < size and tile * BLOCK_DIM + ty < size:
            b_shared[ty, tx] = b[(tile * BLOCK_DIM + ty) * size + col]
        else:
            b_shared[ty, tx] = 0.0

        # Synchronize to ensure all threads have loaded their tiles
        cuda.syncthreads()

        # Perform the computation for this tile
        for k in range(BLOCK_DIM):
            temp += a_shared[ty, k] * b_shared[k, tx]

        # Synchronize to ensure no thread overwrites shared memory before others are done
        cuda.syncthreads()

    # Write the result to the output matrix
    if row < size and col < size:
        out[row * size + col] = temp


jit_mm_practice = jit(_mm_practice)


def mm_practice(a: Tensor, b: Tensor) -> TensorData:
    (size, _) = a.shape
    threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK)
    blockspergrid = 1
    out = TensorData([0.0 for i in range(size * size)], (size, size))
    out.to_cuda_()
    jit_mm_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, b._tensor._storage, size
    )
    return out


def _tensor_matrix_multiply(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
    """CUDA tensor matrix multiply function.

    Requirements:

    * All data must be first moved to shared memory.
    * Only read each cell in `a` and `b` once.
    * Only write to global memory once per kernel.

    Should work for any tensor shapes that broadcast as long as ::

    ```python
    assert a_shape[-1] == b_shape[-2]
    ```
    Returns:
        None : Fills in `out`
    """
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0
    batch = cuda.blockIdx.z  # Batch dimension for 3D tensors

    BLOCK_DIM = 32
    a_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    b_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)

    # Compute thread and block indices
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x  # Row in output
    j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y  # Column in output
    pi = cuda.threadIdx.x  # Local thread x index
    pj = cuda.threadIdx.y  # Local thread y index

    dim_shared = a_shape[2]  # Shared dimension (a's cols = b's rows)
    tot = 0.0  # Accumulator for dot product

    # Loop over tiles of the shared dimension
    for block_s in range(0, dim_shared, BLOCK_DIM):
        ka = block_s + pj
        kb = block_s + pi

        # Load tiles of `a` and `b` into shared memory
        if i < a_shape[1] and ka < a_shape[2]:
            a_shared[pi, pj] = a_storage[
                (batch * a_batch_stride) + (i * a_strides[1]) + (ka * a_strides[2])
            ]
        else:
            a_shared[pi, pj] = 0.0

        if kb < b_shape[1] and j < b_shape[2]:
            b_shared[pi, pj] = b_storage[
                (batch * b_batch_stride) + (kb * b_strides[1]) + (j * b_strides[2])
            ]
        else:
            b_shared[pi, pj] = 0.0

        # Synchronize threads to ensure tiles are loaded
        cuda.syncthreads()

        # Compute dot product for the current tile
        for k in range(BLOCK_DIM):
            if block_s + k < dim_shared:
                tot += a_shared[pi, k] * b_shared[k, pj]

        # Synchronize threads before loading the next tile
        cuda.syncthreads()

    # Write the computed value to the output tensor
    if i < out_shape[1] and j < out_shape[2]:
        out_idx = (batch * out_strides[0]) + (i * out_strides[1]) + (j * out_strides[2])
        out[out_idx] = tot


tensor_matrix_multiply = cuda.jit(_tensor_matrix_multiply)
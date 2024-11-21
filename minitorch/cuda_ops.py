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


def device_jit(fn: Fn, **kwargs: Any) -> Fn:
    """JIT compile a function for CUDA device execution.

    Args:
    ----
        fn: The function to be compiled.
        **kwargs: Additional keyword arguments to pass to the Numba JIT compiler.

    Returns:
    -------
        The JIT-compiled CUDA device function.

    """
    return _jit(device=True, **kwargs)(fn)  # type: ignore


def jit(fn: Fn, **kwargs: Any) -> FakeCUDAKernel:
    """JIT compile a function for CUDA kernel execution.

    Args:
    ----
        fn: The function to be compiled as a CUDA kernel.
        **kwargs: Additional keyword arguments to pass to the Numba JIT compiler.

    Returns:
    -------
        The JIT-compiled CUDA kernel.

    """
    return _jit(**kwargs)(fn)  # type: ignore


to_index = device_jit(to_index)
index_to_position = device_jit(index_to_position)
broadcast_index = device_jit(broadcast_index)

THREADS_PER_BLOCK = 32


class CudaOps(TensorOps):
    cuda = True

    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """Create a CUDA kernel for applying a unary function to a tensor.

        Args:
        ----
            fn: A function that maps a single float to another float.

        Returns:
        -------
            A function that maps a tensor to an output tensor.

        """
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
        """Create a CUDA kernel for applying a binary function to two tensors.

        Args:
        ----
            fn: A function that maps two floats to a single float.

        Returns:
        -------
            A function that maps two input tensors to an output tensor.

        """
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
        """Create a CUDA kernel for reducing a tensor along a specified dimension.

        Args:
        ----
            fn: A reduction function that maps two floats to a single float.
            start: The initial value for the reduction.

        Returns:
        -------
            A function that reduces a tensor along a specified dimension.

        """
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
        """Perform a matrix multiplication between two tensors using CUDA.

        Args:
        ----
            a: The first tensor.
            b: The second tensor.

        Returns:
        -------
            The resulting tensor from the matrix multiplication.

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
    """CUDA kernel for mapping a unary function over a tensor.

    Args:
    ----
        fn: A unary function to apply to each element of the tensor.

    Returns:
    -------
        A CUDA kernel function for mapping the unary function.

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
        if i < out_size:  # Ensure valid bounds
            to_index(i, out_shape, out_index)
            o = index_to_position(out_index, out_strides)
            broadcast_index(out_index, out_shape, in_shape, in_index)
            j = index_to_position(in_index, in_strides)
            out[o] = fn(in_storage[j])

    return cuda.jit()(_map)  # type: ignore


def tensor_zip(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """CUDA kernel for applying a binary function to two tensors.

    Args:
    ----
        fn: A binary function to apply to elements from two tensors.

    Returns:
    -------
        A CUDA kernel function for applying the binary function.

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
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        if i < out_size:  # Ensure valid bounds
            to_index(i, out_shape, out_index)
            o = index_to_position(out_index, out_strides)
            broadcast_index(out_index, out_shape, a_shape, a_index)
            broadcast_index(out_index, out_shape, b_shape, b_index)
            j = index_to_position(a_index, a_strides)
            k = index_to_position(b_index, b_strides)
            out[o] = fn(a_storage[j], b_storage[k])

    return cuda.jit()(_zip)  # type: ignore


def _sum_practice(out: Storage, a: Storage, size: int) -> None:
    BLOCK_DIM = 32
    cache = cuda.shared.array(BLOCK_DIM, numba.float64)
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    pos = cuda.threadIdx.x

    # Load data into shared memory
    cache[pos] = a[i] if i < size else 0.0
    cuda.syncthreads()

    # Perform reduction in shared memory
    stride = 1
    while stride < BLOCK_DIM:
        if pos % (2 * stride) == 0 and pos + stride < BLOCK_DIM:
            cache[pos] += cache[pos + stride]
        stride *= 2
        cuda.syncthreads()

    # Store the result
    if pos == 0:
        out[cuda.blockIdx.x] = cache[0]


jit_sum_practice = cuda.jit()(_sum_practice)


def sum_practice(a: Tensor) -> TensorData:
    """CUDA kernel for summing elements of a tensor.

    Args:
    ----
        out: The output storage for the sum result.
        a: The input storage containing the tensor elements.
        size: The size of the input tensor.

    Returns:
    -------
        None.

    """
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
        fn: A reduction function that maps two floats to a single float.

    Returns:
    -------
        A tensor reduce function to be executed on CUDA.

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
        init_value: float,
    ) -> None:
        BLOCK_DIM = 1024
        # Shared memory for thread collaboration within a block
        shared_mem = cuda.shared.array(BLOCK_DIM, numba.float64)
        out_index = cuda.local.array(MAX_DIMS, numba.int32)

        # Block and thread identifiers
        block_idx = cuda.blockIdx.x
        thread_idx = cuda.threadIdx.x

        # Initialize shared memory with the initial reduction value
        shared_mem[thread_idx] = init_value

        if block_idx < out_size:
            # Calculate the output index
            to_index(block_idx, out_shape, out_index)
            dim_length = a_shape[reduce_dim]

            # Map thread position to the reduction dimension
            out_index[reduce_dim] = thread_idx + out_index[reduce_dim] * BLOCK_DIM

            if out_index[reduce_dim] < dim_length:
                # Convert the calculated index to storage position
                input_pos = index_to_position(out_index, a_strides)
                shared_mem[thread_idx] = a_storage[input_pos]

            # Synchronize all threads in the block before reduction
            cuda.syncthreads()

            # Perform reduction using shared memory
            step = 1
            while step < BLOCK_DIM:
                if thread_idx % (2 * step) == 0:
                    if thread_idx + step < BLOCK_DIM:
                        shared_mem[thread_idx] = fn(
                            shared_mem[thread_idx], shared_mem[thread_idx + step]
                        )
                step *= 2
                cuda.syncthreads()

            # Write the reduced result to the output tensor
            if thread_idx == 0:
                output_pos = index_to_position(out_index, out_strides)
                out[output_pos] = shared_mem[0]

    return cuda.jit()(_reduce)  # type: ignore


def _mm_practice(out: Storage, a: Storage, b: Storage, size: int) -> None:
    """Practice CUDA kernel for matrix multiplication using shared memory.

    Args:
    ----
        out (Storage): Output matrix storage.
        a (Storage): Input matrix A storage.
        b (Storage): Input matrix B storage.
        size (int): Size of the square matrix (size x size).

    """
    # Define the block size for shared memory arrays
    BLOCK_DIM = 32

    # Allocate shared memory for blocks of A and B
    a_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    b_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)

    # Thread indices within the block
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y

    # Global row and column indices for the current thread
    row = cuda.blockIdx.y * cuda.blockDim.y + ty
    col = cuda.blockIdx.x * cuda.blockDim.x + tx

    # Accumulate the computed value for the current element
    temp = 0.0

    # Iterate over tiles of the matrix
    for tile in range((size + BLOCK_DIM - 1) // BLOCK_DIM):
        # Load tile of matrix A into shared memory
        if row < size and tile * BLOCK_DIM + tx < size:
            a_shared[ty, tx] = a[row * size + tile * BLOCK_DIM + tx]
        else:
            a_shared[ty, tx] = 0.0  # Load 0 if out of bounds

        # Load tile of matrix B into shared memory
        if col < size and tile * BLOCK_DIM + ty < size:
            b_shared[ty, tx] = b[(tile * BLOCK_DIM + ty) * size + col]
        else:
            b_shared[ty, tx] = 0.0  # Load 0 if out of bounds

        # Synchronize threads to ensure all threads finish loading
        cuda.syncthreads()

        # Perform partial computation for the current tile
        for k in range(BLOCK_DIM):
            temp += a_shared[ty, k] * b_shared[k, tx]

        # Synchronize threads to prevent race conditions in shared memory
        cuda.syncthreads()

    # Store the computed value into the output matrix
    if row < size and col < size:
        out[row * size + col] = temp


jit_mm_practice = jit(_mm_practice)


def mm_practice(a: Tensor, b: Tensor) -> TensorData:
    """Wrapper for the CUDA matrix multiplication kernel.

    Args:
    ----
        a (Tensor): Input tensor A.
        b (Tensor): Input tensor B.

    Returns:
    -------
        TensorData: Result of the matrix multiplication.

    """
    (size, _) = a.shape  # Assume square matrices

    # Define block and grid dimensions
    threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK)  # Threads per block
    blockspergrid = 1  # Single grid for now

    # Allocate output tensor data
    out = TensorData([0.0 for i in range(size * size)], (size, size))

    # Move output tensor to GPU memory
    out.to_cuda_()

    # Launch the CUDA kernel
    jit_mm_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, b._tensor._storage, size
    )

    # Return the result tensor
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
    """CUDA tensor matrix multiplication function using shared memory.

    Args:
    ----
        out (Storage): Output tensor storage.
        out_shape (Shape): Shape of the output tensor.
        out_strides (Strides): Strides of the output tensor.
        out_size (int): Total elements in the output tensor.
        a_storage (Storage): Input tensor A storage.
        a_shape (Shape): Shape of tensor A.
        a_strides (Strides): Strides of tensor A.
        b_storage (Storage): Input tensor B storage.
        b_shape (Shape): Shape of tensor B.
        b_strides (Strides): Strides of tensor B.

    """
    BLOCK_DIM = 32

    # Shared memory for tiles of A and B
    a_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    b_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)

    # Batch index
    batch = cuda.blockIdx.z
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0

    # Global row and column indices
    i = cuda.blockIdx.x * BLOCK_DIM + cuda.threadIdx.x
    j = cuda.blockIdx.y * BLOCK_DIM + cuda.threadIdx.y

    # Local thread indices within the block
    pi = cuda.threadIdx.x
    pj = cuda.threadIdx.y

    val = 0.0  # Accumulator for the result
    shared_dim = a_shape[2]  # Shared dimension for matrix multiplication

    # Iterate over tiles in the shared dimension
    for k0 in range(0, shared_dim, BLOCK_DIM):
        # Load sub-block of A into shared memory
        if i < a_shape[1] and (k0 + pj) < shared_dim:
            a_shared[pi, pj] = a_storage[
                batch * a_batch_stride + i * a_strides[1] + (k0 + pj) * a_strides[2]
            ]
        else:
            a_shared[pi, pj] = 0.0  # Load 0 if out of bounds

        # Load sub-block of B into shared memory
        if j < b_shape[2] and (k0 + pi) < shared_dim:
            b_shared[pi, pj] = b_storage[
                batch * b_batch_stride + (k0 + pi) * b_strides[1] + j * b_strides[2]
            ]
        else:
            b_shared[pi, pj] = 0.0  # Load 0 if out of bounds

        # Synchronize threads to ensure all threads finish loading
        cuda.syncthreads()

        # Compute partial result for the current tile
        for k in range(BLOCK_DIM):
            if (k0 + k) < shared_dim:
                val += a_shared[pi, k] * b_shared[k, pj]

        # Synchronize threads before loading the next tile
        cuda.syncthreads()

    # Write the result back to global memory
    if i < out_shape[1] and j < out_shape[2]:
        out_idx = batch * out_strides[0] + i * out_strides[1] + j * out_strides[2]
        out[out_idx] = val


tensor_matrix_multiply = cuda.jit(_tensor_matrix_multiply)

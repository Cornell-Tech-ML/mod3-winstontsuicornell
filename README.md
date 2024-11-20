# MiniTorch Module 3

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module3.html


You will need to modify `tensor_functions.py` slightly in this assignment.

* Tests:

```
python run_tests.py
```

* Note:

Several of the tests for this assignment will only run if you are on a GPU machine and will not
run on github's test infrastructure. Please follow the instructions to setup up a colab machine
to run these tests.

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/tensor.py minitorch/datasets.py minitorch/testing.py minitorch/optim.py

<br>
<br>

<details>
<summary>Diagnostics output from script:</summary>
MAP

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_map.<locals>._map,
/Users/winstontsui/Desktop/CS 5781 Machine Learning
Engineering/mod3-winstontsuicornell/minitorch/fast_ops.py (172)
================================================================================


Parallel loop listing for  Function tensor_map.<locals>._map, /Users/winstontsui/Desktop/CS 5781 Machine Learning Engineering/mod3-winstontsuicornell/minitorch/fast_ops.py (172)
----------------------------------------------------------------------------------|loop #ID
    def _map(                                                                     |
        out: Storage,                                                             |
        out_shape: Shape,                                                         |
        out_strides: Strides,                                                     |
        in_storage: Storage,                                                      |
        in_shape: Shape,                                                          |
        in_strides: Strides,                                                      |
    ) -> None:                                                                    |
        # Task 3.1.                                                               |
        if (                                                                      |
            (len(in_shape) == len(out_shape))                                     |
            and (in_shape == out_shape).all()-------------------------------------| #0
            and (in_strides == out_strides).all()---------------------------------| #1
        ):                                                                        |
            for i in prange(len(out)):--------------------------------------------| #2
                out[i] = fn(in_storage[i])                                        |
        else:                                                                     |
            for i in prange(len(out)):--------------------------------------------| #3
                in_idx = in_shape.copy()                                          |
                out_idx = out_shape.copy()                                        |
                to_index(i, out_shape, out_idx)                                   |
                broadcast_index(out_idx, out_shape, in_shape, in_idx)             |
                out[i] = fn(in_storage[index_to_position(in_idx, in_strides)])    |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 4 parallel for-
loop(s) (originating from loops labelled: #0, #1, #2, #3).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None
ZIP

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_zip.<locals>._zip,
/Users/winstontsui/Desktop/CS 5781 Machine Learning
Engineering/mod3-winstontsuicornell/minitorch/fast_ops.py (225)
================================================================================


Parallel loop listing for  Function tensor_zip.<locals>._zip, /Users/winstontsui/Desktop/CS 5781 Machine Learning Engineering/mod3-winstontsuicornell/minitorch/fast_ops.py (225)
-----------------------------------------------------------------------------------------------------------------------------------------------|loop #ID
    def _zip(                                                                                                                                  |
        out: Storage,                                                                                                                          |
        out_shape: Shape,                                                                                                                      |
        out_strides: Strides,                                                                                                                  |
        a_storage: Storage,                                                                                                                    |
        a_shape: Shape,                                                                                                                        |
        a_strides: Strides,                                                                                                                    |
        b_storage: Storage,                                                                                                                    |
        b_shape: Shape,                                                                                                                        |
        b_strides: Strides,                                                                                                                    |
    ) -> None:                                                                                                                                 |
        # TODO: Implement for Task 3.1.                                                                                                        |
        # out_index = np.zeros(len(out_shape), dtype=np.int32)  # Buffer for output index                                                      |
        # for i in prange(len(out)):  # Parallel loop                                                                                          |
        #     to_index(i, out_shape, out_index)  # Convert flat index to multidimensional index                                                |
        #     a_pos = index_to_position(out_index, a_strides)  # Get position in a_storage                                                     |
        #     b_pos = index_to_position(out_index, b_strides)  # Get position in b_storage                                                     |
        #     out[i] = fn(a_storage[a_pos], b_storage[b_pos])  # Apply function and store result                                               |
        out_size = len(out)                                                                                                                    |
        out_len = len(out_shape)                                                                                                               |
        a_len = len(a_shape)                                                                                                                   |
        b_len = len(b_shape)                                                                                                                   |
        if ((a_len == b_len) and (a_strides == b_strides).all() and (a_shape == b_shape).all() and (b_strides == out_strides).all()):----------| #4, 5, 6
            for i in prange(out_size):---------------------------------------------------------------------------------------------------------| #7
                out[i] = fn(a_storage[i], b_storage[i])                                                                                        |
        else:                                                                                                                                  |
            for i in prange(out_size):---------------------------------------------------------------------------------------------------------| #8
                out_index = np.empty(out_len, np.int32)                                                                                        |
                a_index = np.empty(a_len, np.int32)                                                                                            |
                b_index = np.empty(b_len, np.int32)                                                                                            |
                # Gaining the out_index                                                                                                        |
                to_index(i, out_shape, out_index)                                                                                              |
                # Gain the a_index from the out_index                                                                                          |
                broadcast_index(out_index, out_shape, a_shape, a_index)                                                                        |
                # Gain the b_index from the out_index                                                                                          |
                broadcast_index(out_index, out_shape, b_shape, b_index)                                                                        |
                # Finding the a_storage position from the a_index                                                                              |
                a_pos = index_to_position(a_index, a_strides)                                                                                  |
                # Finding the b_storage position from the b_index                                                                              |
                b_pos = index_to_position(b_index, b_strides)                                                                                  |
                # Finding the out_storage position from the out_index                                                                          |
                #out_pos = index_to_position(out_index, out_strides)                                                                           |
                # Throwing it in storage                                                                                                       |
                out[i] = fn(a_storage[a_pos], b_storage[b_pos])                                                                                |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 5 parallel for-
loop(s) (originating from loops labelled: #4, #5, #6, #7, #8).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at
/Users/winstontsui/Desktop/CS 5781 Machine Learning
Engineering/mod3-winstontsuicornell/minitorch/fast_ops.py (252) is hoisted out
of the parallel loop labelled #8 (it will be performed before the loop is
executed and reused inside the loop):
   Allocation:: out_index = np.empty(out_len, np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at
/Users/winstontsui/Desktop/CS 5781 Machine Learning
Engineering/mod3-winstontsuicornell/minitorch/fast_ops.py (253) is hoisted out
of the parallel loop labelled #8 (it will be performed before the loop is
executed and reused inside the loop):
   Allocation:: a_index = np.empty(a_len, np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at
/Users/winstontsui/Desktop/CS 5781 Machine Learning
Engineering/mod3-winstontsuicornell/minitorch/fast_ops.py (254) is hoisted out
of the parallel loop labelled #8 (it will be performed before the loop is
executed and reused inside the loop):
   Allocation:: b_index = np.empty(b_len, np.int32)
    - numpy.empty() is used for the allocation.
None
REDUCE

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_reduce.<locals>._reduce,
/Users/winstontsui/Desktop/CS 5781 Machine Learning
Engineering/mod3-winstontsuicornell/minitorch/fast_ops.py (297)
================================================================================


Parallel loop listing for  Function tensor_reduce.<locals>._reduce, /Users/winstontsui/Desktop/CS 5781 Machine Learning Engineering/mod3-winstontsuicornell/minitorch/fast_ops.py (297)
------------------------------------------------------------------------------------------------------|loop #ID
    def _reduce(                                                                                      |
        out: Storage,                                                                                 |
        out_shape: Shape,                                                                             |
        out_strides: Strides,                                                                         |
        a_storage: Storage,                                                                           |
        a_shape: Shape,                                                                               |
        a_strides: Strides,                                                                           |
        reduce_dim: int,                                                                              |
    ) -> None:                                                                                        |
        # TODO: Implement for Task 3.1.                                                               |
        # out_index = np.zeros(len(out_shape), dtype=np.int32)  # Buffer for output index             |
        # in_index = np.zeros(len(a_shape), dtype=np.int32)  # Buffer for input index                 |
        # for i in prange(len(out)):  # Parallel loop                                                 |
        #     to_index(i, out_shape, out_index)  # Convert flat index to multidimensional index       |
        #     reduced_value = out[i]  # Start with the initial value in the output tensor             |
        #     for j in range(a_shape[reduce_dim]):  # Iterate along the reduction dimension           |
        #         in_index[:] = out_index  # Copy current index                                       |
        #         in_index[reduce_dim] = j  # Update index for the reduction dimension                |
        #         in_pos = index_to_position(in_index, a_strides)  # Get position in input storage    |
        #         reduced_value = fn(reduced_value, a_storage[in_pos])  # Apply reduction             |
        #     out[i] = reduced_value  # Write reduced result to output tensor                         |
                                                                                                      |
        reduce_size = a_shape[reduce_dim]                                                             |
        for i in prange(len(out)):--------------------------------------------------------------------| #10
            out_index: Index = np.zeros(MAX_DIMS, np.int32)-------------------------------------------| #9
            to_index(i, out_shape, out_index)                                                         |
            o = index_to_position(out_index, out_strides)                                             |
            for s in range(reduce_size):                                                              |
                out_index[reduce_dim] = s                                                             |
                j = index_to_position(out_index, a_strides)                                           |
                out[o] = fn(out[o], a_storage[j])                                                     |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 2 parallel for-
loop(s) (originating from loops labelled: #10, #9).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...

+--10 is a parallel loop
   +--9 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--10 (parallel)
   +--9 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--10 (parallel)
   +--9 (serial)



Parallel region 0 (loop #10) had 0 loop(s) fused and 1 loop(s) serialized as
part of the larger parallel loop (#10).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at
/Users/winstontsui/Desktop/CS 5781 Machine Learning
Engineering/mod3-winstontsuicornell/minitorch/fast_ops.py (321) is hoisted out
of the parallel loop labelled #10 (it will be performed before the loop is
executed and reused inside the loop):
   Allocation:: out_index: Index = np.zeros(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
None
MATRIX MULTIPLY

================================================================================
 Parallel Accelerator Optimizing:  Function _tensor_matrix_multiply,
/Users/winstontsui/Desktop/CS 5781 Machine Learning
Engineering/mod3-winstontsuicornell/minitorch/fast_ops.py (334)
================================================================================


Parallel loop listing for  Function _tensor_matrix_multiply, /Users/winstontsui/Desktop/CS 5781 Machine Learning Engineering/mod3-winstontsuicornell/minitorch/fast_ops.py (334)
----------------------------------------------------------------------------------------------|loop #ID
def _tensor_matrix_multiply(                                                                  |
    out: Storage,                                                                             |
    out_shape: Shape,                                                                         |
    out_strides: Strides,                                                                     |
    a_storage: Storage,                                                                       |
    a_shape: Shape,                                                                           |
    a_strides: Strides,                                                                       |
    b_storage: Storage,                                                                       |
    b_shape: Shape,                                                                           |
    b_strides: Strides,                                                                       |
) -> None:                                                                                    |
    """NUMBA tensor matrix multiply function.                                                 |
                                                                                              |
    Args:                                                                                     |
        out (Storage): storage for `out` tensor                                               |
        out_shape (Shape): shape for `out` tensor                                             |
        out_strides (Strides): strides for `out` tensor                                       |
        a_storage (Storage): storage for `a` tensor                                           |
        a_shape (Shape): shape for `a` tensor                                                 |
        a_strides (Strides): strides for `a` tensor                                           |
        b_storage (Storage): storage for `b` tensor                                           |
        b_shape (Shape): shape for `b` tensor                                                 |
        b_strides (Strides): strides for `b` tensor                                           |
                                                                                              |
    Returns:                                                                                  |
        None: Fills in `out`.                                                                 |
                                                                                              |
    """                                                                                       |
    # batch_size, out_rows, out_cols = out_shape                                              |
    # a_rows, a_cols = a_shape[-2], a_shape[-1]                                               |
    # b_rows, b_cols = b_shape[-2], b_shape[-1]                                               |
                                                                                              |
    # # Ensure matrix dimensions match for multiplication                                     |
    # assert a_cols == b_rows                                                                 |
                                                                                              |
    # for n in prange(batch_size):  # Parallelize over batches                                |
    #     for i in range(out_rows):  # Iterate over rows of the output                        |
    #         for j in range(out_cols):  # Iterate over columns of the output                 |
    #             out_index = n * out_strides[0] + i * out_strides[1] + j * out_strides[2]    |
    #             sum_value = 0.0  # Local variable to accumulate the dot product             |
                                                                                              |
    #             for k in range(a_cols):  # Iterate over the inner dimension                 |
    #                 a_index = n * a_strides[0] + i * a_strides[1] + k * a_strides[2]        |
    #                 b_index = n * b_strides[0] + k * b_strides[1] + j * b_strides[2]        |
                                                                                              |
    #                 sum_value += a_storage[a_index] * b_storage[b_index]                    |
                                                                                              |
    #             out[out_index] = sum_value  # Write the result to the output tensor         |
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0                                    |
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0                                    |
    for x in prange(out_shape[0]):------------------------------------------------------------| #13
        for y in prange(out_shape[1]):--------------------------------------------------------| #12
            for z in prange(out_shape[2]):----------------------------------------------------| #11
                val = 0.0                                                                     |
                posA = x * a_batch_stride + y * a_strides[1]                                  |
                posB = x * b_batch_stride + z * b_strides[2]                                  |
                for a in range(a_shape[2]):                                                   |
                    val += a_storage[posA] * b_storage[posB]                                  |
                    posA += a_strides[2]                                                      |
                    posB += b_strides[1]                                                      |
                outPos = x * out_strides[0] + y * out_strides[1] + z * out_strides[2]         |
                out[outPos] = val                                                             |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 2 parallel for-
loop(s) (originating from loops labelled: #13, #12).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...

+--13 is a parallel loop
   +--12 --> rewritten as a serial loop
      +--11 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--13 (parallel)
   +--12 (parallel)
      +--11 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--13 (parallel)
   +--12 (serial)
      +--11 (serial)



Parallel region 0 (loop #13) had 0 loop(s) fused and 2 loop(s) serialized as
part of the larger parallel loop (#13).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None


</details>





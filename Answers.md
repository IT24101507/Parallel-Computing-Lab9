# SE3082 - Lab 09: CUDA Questions and Answers

## Exercise 2 - THINK

**Q1. Why do we need separate variables `d_a`, `d_b`, and `d_c` on the device?**

The CPU and GPU have separate memory spaces. We allocate device memory and copy the input values to the GPU before executing the kernel.

**Q2. Why use `&a` in `cudaMemcpy()` but `d_a` in the kernel?**

`&a` is the address of the CPU variable. `d_a` is a pointer to GPU memory, which the kernel uses to access the value.

**Q3. How many threads execute the add kernel? What happens if we use 10 threads?**

The original kernel uses one thread. To launch 10 threads, we can use `add<<<1, 10>>>`. The expected answer is still 9, but all 10 threads would write to the same location, causing a data race.

## Exercise 3 - PREDICT BEFORE YOU RUN

**Q1. How many blocks and threads does `add<<<N, 1>>>` launch?**

It launches 512 blocks with one thread each, giving 512 threads in total.

**Q2. What happens when a block contains only one thread?**

CUDA executes threads in warps of 32. A block containing only one thread leaves most of the warp's execution capacity unused, making this arrangement inefficient.

## Exercise 4 - Experiment

**What happens when N is changed from 512 to 2048?**

The kernel launch fails with an invalid configuration argument.

The NVIDIA Tesla T4 supports a maximum of 1024 threads per block. Launching 2048 threads in one block exceeds this limit.

The program reported 2048 mismatches because the kernel did not execute successfully. The output values copied from uninitialized GPU memory are not reliable.

The value of N was restored to 512 after the experiment.

## Exercise 5 - THINK

**Q1. Why is using a multiple of 32 threads per block a good idea?**

CUDA threads execute in groups of 32 called warps. Using a multiple of 32 avoids partially filled warps within a block, improving resource utilization.

**Q2. What happens if we use 500 threads instead of 512?**

We need 20,000 blocks to cover 10,000,000 elements.

Total threads = 20,000 × 500 = 10,000,000.

There are no extra threads beyond the array size, but each block's last warp is only partially filled because 500 is not a multiple of 32.

## Exercise 6 - Matrix Multiplication


The program uses a two-dimensional CUDA grid to multiply corresponding elements of two 10,000 × 10,000 matrices.

- Threads per block: 32 × 16 = 512
- Grid size: 313 × 625 blocks
- Total elements: 100,000,000
- Global row: `blockIdx.y * blockDim.y + threadIdx.y`
- Global column: `blockIdx.x * blockDim.x + threadIdx.x`
- Flat index: `row * cols + col`

A bounds check prevents threads outside the matrix dimensions from accessing invalid memory.


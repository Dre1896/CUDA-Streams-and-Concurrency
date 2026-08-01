# CUDA-Streams-and-Concurrency

I built a serial and a stream based version of the same GPU workload to measure whether overlapping data transfer with computation actually produces observable concurrency, not just theoretical concurrency. This project documents the process: writing both versions, profiling each with Nsight Systems, and proving the difference with real timestamps rather than just describing what streams are supposed to do.

**Result: the serial version issues each operation roughly 1.2 to 1.5 milliseconds apart, waiting for the previous one to fully complete first. The streamed version issues operations only microseconds apart, confirming genuine overlap rather than hidden serialization.**

## Why I built this

Nsight Systems is the tool I would reach for to answer "is this GPU actually idle waiting on the host, or is work genuinely overlapping." That question comes up constantly in distributed and multi-stage GPU workloads, and I wanted direct, hands on evidence of what overlap looks like in a profiler, on a system small enough that I fully control every variable.

## Architecture

```
CUDA workload (8 chunks, pinned host memory)
       |
       v
Serial version        — cudaMemcpy (blocking), one chunk fully completes before the next starts
       |
Streamed version       — cudaMemcpyAsync + per chunk CUDA streams, operations issued without waiting
       |
       v
Nsight Systems         — Timeline View and CUDA API Summary confirm actual overlap versus sequencing
```

Both versions run the identical kernel. The kernel performs artificial, non meaningful arithmetic solely to create measurable execution time for profiling, it is a controlled test harness, not a real computational workload. The only difference between the two versions is entirely in how the host issues and schedules the copy and compute operations, not in what the GPU actually computes.

## Methodology

I wrote a serial version that processes 8 chunks of data one at a time: copy to device, launch kernel, copy back, then move to the next chunk. Every memory copy uses blocking `cudaMemcpy`, and every kernel launches into the default stream, so each step waits for the previous one to fully finish.

I then wrote a streamed version of the same workload. Each of the 8 chunks gets its own `cudaStream_t`. Memory copies use `cudaMemcpyAsync`, which returns immediately instead of blocking, and each kernel launch is issued into its chunk's own stream. Both host and device memory allocations use pinned memory (`cudaMallocHost`), which is required for asynchronous transfers to actually overlap with compute, pageable memory forces synchronous copies regardless of streams.

I profiled both versions with Nsight Systems, then used the CUDA API Summary view to get aggregate call counts and durations per function, and the Events View with Start and Duration columns to compare exactly when each operation was issued.

## Results

### Serial vs streamed timeline

<!-- Add Timeline View screenshots here: serial shows a staircase of sequential blocks, streamed shows compressed, overlapping blocks -->

### CUDA API Summary comparison

| Function | Serial: Total Time | Serial: Avg | Streamed: Total Time | Streamed: Avg |
|---|---|---|---|---|
| cudaHostAlloc | 378.930 ms | 47.366 ms | 483.402 ms | 60.425 ms |
| cudaLaunchKernel | 262.849 ms | 32.856 ms | 6.720 ms | 839.950 μs |
| cudaMemcpy / cudaMemcpyAsync | 23.467 ms | 1.467 ms | 4.558 ms | 284.865 μs |
| cudaDeviceSynchronize | 4.668 μs | 4.668 μs | 17.630 ms | 17.630 ms |

Both memcpy rows show 16 calls, two per chunk, one host to device and one device to host, across all 8 chunks, confirming the loop ran correctly in both versions.

### Event timing pattern

Serial `cudaMemcpy` calls, consecutive Start timestamps:

```
2.76191s
2.76343s   (+1.52ms)
2.76474s   (+1.31ms)
2.76609s   (+1.35ms)
```

Streamed `cudaMemcpyAsync` calls, consecutive Start timestamps:

```
2.54041s
2.54046s   (+50μs)
2.54051s   (+50μs)
2.54052s   (+10μs)
```

The serial version issues operations on a roughly 1.2 to 1.5 millisecond cadence, each one starting only after the previous chunk's full copy, compute, copy back cycle finishes. The streamed version issues operations within tens of microseconds of each other, two to three orders of magnitude faster, because `cudaMemcpyAsync` returns immediately and does not wait for the transfer to complete before the next line of host code runs.

## Key findings

The event timing data is the clearest evidence of real overlap. A profiler cannot lie about when a CUDA API call was actually issued, and the gap between consecutive Start timestamps is exactly what changes between blocking and asynchronous calls. Serial spacing on the order of milliseconds versus streamed spacing on the order of microseconds is a direct, measurable signature of the host no longer waiting on each operation before issuing the next.

One finding worth explaining honestly rather than reporting at face value: `cudaLaunchKernel` shows 262.849 ms total time in the serial run versus only 6.720 ms in the streamed run, a difference that looks like the kernel itself somehow got dramatically faster. It did not. Nsight Systems flagged `Runtime Triggered Module Loading` and `JIT Compilation` blocks directly in the serial run's timeline, meaning the first kernel launch paid a one time compilation cost. This is the same measurement artifact I found in earlier profiling work: JIT compilation inflates the first call's duration regardless of which version of a kernel is being compiled. The fair comparison is the event spacing shown above, not the raw `cudaLaunchKernel` total.

`cudaHostAlloc` is the single largest cost in both versions, and its StdDev is nearly as large as its average in both runs, one allocation, almost always the first, is dramatically slower than the rest. This is consistent with a one time OS level cost to set up pinned memory infrastructure, separate from the actual per chunk allocation cost. It is a real cost of using pinned memory, but it is a fixed setup cost, not something that scales with chunk count.

<details>
<summary>A plainer language explanation of what changed</summary>

Picture a kitchen with one cook working through 8 orders. In the serial version, the cook fully finishes plating one order, walks it to the pass, comes back, and only then starts the next order. Nothing happens until the previous thing is completely done.

In the streamed version, the cook drops off an order at the pass without waiting to see it served, and immediately starts the next one. Multiple orders end up cooking and moving through the kitchen at overlapping times, because the cook stopped waiting around after each handoff.

The kitchen itself, the recipe, the ingredients, none of that changed. What changed is whether the cook waits after each step or keeps moving immediately to the next one.

</details>

## Design notes

I used artificial, repeated arithmetic inside the kernel specifically to create measurable execution time, not to solve anything real. Without deliberately padding the kernel's runtime, both versions would finish too quickly to produce a meaningful profiler timeline, and the overlap this project is trying to demonstrate would be invisible.

I chose event Start timestamps as the primary evidence rather than relying only on a timeline screenshot, because exact numbers are verifiable in a way that a colored bar chart is not. A reader can check the math themselves against the raw data.

Pinned memory (`cudaMallocHost`) is required, not optional, for this comparison to be meaningful. Asynchronous copies from pageable memory silently fall back to synchronous behavior, which would have made the streamed version behave identically to the serial one and hidden the entire point of the project.

## Getting started

Requirements:
- NVIDIA GPU with current drivers
- CUDA Toolkit installed
- Nsight Systems installed

```bash
git clone https://github.com/Dre1896/CUDA-Streams-Concurrency.git
cd CUDA-Streams-Concurrency
nvcc serial_streams.cu -o serial_version
nvcc concurrent_streams.cu -o streamed_version
```

To profile either version:

```bash
nsys profile -o serial_profile ./serial_version
nsys profile -o streamed_profile ./streamed_version
```

Open the resulting `.nsys-rep` files in Nsight Systems. Expand the CUDA API row in Timeline View to see individual calls, or switch to Stats System View and select CUDA API Summary for aggregate counts and durations.

## Next steps

- Extend the same comparison to a workload with real computational value instead of artificial padding, to confirm the overlap behavior holds under a realistic kernel
- Apply the same DCGM, Nsight Systems, and Nsight Compute diagnostic chain used in other projects to this workload, to see whether GPU level telemetry also reflects the overlap during the streamed run
- Investigate the cudaHostAlloc first call overhead directly, to understand whether it is fully fixed cost or partially scales with allocation size

## License

MIT, see LICENSE.

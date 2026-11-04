# Intel XPU Hardware Reference (`Xe` PVC & `Xe2` BMG)

This file summarizes the hardware topology, caches, memory bandwidths, execution pipes, and Roofline ridge points across Intel Data Center GPU Max (`Xe`, Ponte Vecchio) and Intel Arc B-Series (`Xe2`, Battlemage), matching `helion/_hardware.py` and `pretuned_kernels/xpu_hw_collect.py`.

## 1. SKU Comparison Table

| Property | PVC Max 1550 (2-Stack) | PVC Max 1550 (1-Stack / Flat Tile) | PVC Max 1100 | Arc Pro B70 (BMG-G31) | Arc B580 (BMG-G21) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Compute Capability** | `Xe` | `Xe` | `Xe` | `Xe2` | `Xe2` |
| **Xe-cores (`gpu_subslice_count`)** | 128 | 64 | 56 | 32 | 20 |
| **XVEs (`gpu_eu_count`, 8/core)** | 1024 | 512 | 448 | 256 | 160 |
| **HW Threads (128-GRF / 256-GRF)** | 8192 / 4096 | 4096 / 2048 | 3584 / 1792 | 2048 / 1024 | 1280 / 640 |
| **Max Clock** | 1600 MHz | 1600 MHz | 1550 MHz | 2800 MHz | 2850 MHz |
| **Subgroup Sizes (`SIMD`)** | 16, 32 | 16, 32 | 16, 32 | 16, 32 | 16, 32 |
| **SLM per Xe-core** | 128 KB | 128 KB | 128 KB | 128 KB | 128 KB |
| **L1 / LSC per Xe-core** | 256 KB (64B line) | 256 KB (64B line) | 256 KB (64B line) | 256 KB (64B line) | 256 KB (64B line) |
| **Last-Level Cache (L2/L3)** | 408 MiB | 192 MiB | 108 MiB | 24 MiB | 18 MiB |
| **Peak Memory Bandwidth** | 3276.8 GB/s (HBM2e) | 1638.4 GB/s (HBM2e) | 1228.8 GB/s (HBM2e) | 608.0 GB/s (GDDR6) | 456.0 GB/s (GDDR6) |
| **Peak XMX FP16 / BF16** | 839.0 TFLOP/s | 419.5 TFLOP/s | 311.0 TFLOP/s | 183.5 TFLOP/s | 117.0 TFLOP/s |
| **Peak XMX FP8 / INT8** | 1678.0 TOP/s | 839.0 TOP/s | 622.0 TOP/s | 367.0 TOP/s | 233.0 TOP/s |
| **Peak Vector FP32 (`ALU0`)** | 52.4 TFLOP/s | 26.2 TFLOP/s | 19.4 TFLOP/s | 22.94 TFLOP/s | 14.6 TFLOP/s |
| **Ridge Point (XMX FP16)** | 256.0 FLOP/B | 256.0 FLOP/B | 253.1 FLOP/B | 301.8 FLOP/B | 256.6 FLOP/B |
| **Ridge Point (Vector FP32)** | 16.0 FLOP/B | 16.0 FLOP/B | 15.8 FLOP/B | 37.7 FLOP/B | 32.0 FLOP/B |

## 2. Thread & Register File Hierarchy

- **Xe-core Structure**: Each Xe-core has **8 Vector Engines (XVEs)** and **8 Matrix Engines (XMX)**.
- **Hardware Threads per XVE**:
  - **128-GRF mode (`grf_mode="default"` / `"128"`)**: 8 hardware threads per XVE = **64 hardware threads per Xe-core** (`128 x 64B = 8 KB` register file per thread).
  - **256-GRF mode (`grf_mode="256"`, Large GRF)**: 4 hardware threads per XVE = **32 hardware threads per Xe-core** (`256 x 64B = 16 KB` register file per thread).
- **Triton `num_warps` Mapping**:
  - In Triton XPU, 1 "warp" corresponds to 1 subgroup (`SIMD32` = 32 work-items when `SIMD32` is selected by IGC, or `SIMD16` = 16 work-items).
  - With `SIMD32`: `threads_per_wg = num_warps` (since 1 warp = 32 lanes = 1 hardware thread).
  - With `SIMD16`: `threads_per_wg = 2 * num_warps` (if 32 lanes/warp) or `num_warps`. Always verify `SIMD` and `local` in `unitrace -d -v` (`07_kernel_occupancy.json`).
- **SLM Allocation Granularity**:
  - Hardware allocates SLM per workgroup in discrete chunks: `(0, 1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128)` KB.
  - Allocating `8.1 KB` rounds up to `16 KB`, halving resident workgroups per Xe-core from 16 to 8.

## 3. On-Chip & Memory Latencies (Measured Reference)

| Level / Operation | Unloaded Latency (Cycles) | Notes |
| :--- | :--- | :--- |
| **L1 / LSC Hit** | ~98–128 clk | 256 B/clk/Xe-core read, 128 B/clk/Xe-core write |
| **L3 (Device Cache) Hit** | ~260–310 clk | 64B line, DW (4B) write enables only |
| **DRAM (HBM2e / GDDR6)** | ~900 clk (unloaded), 1100–2500 clk (loaded) | 256B DRAM access granularity |
| **Subgroup Shuffle / Broadcast** | ~12 clk (direct region), +36.5 clk (`r[a0]` indirect) | Runtime shuffle distances or mask 16 on SIMD32 trigger indirect addressing |
| **Subgroup Reduce (32-lane)** | ~83–94 clk | ~6x faster than workgroup SLM + barrier reduce (~380–580 clk) |
| **Workgroup Barrier (`sync.bar`)** | ~50–82 clk | Compiles to `fence.slm + fence.ugm + send.gtwy + sync.bar` |
| **DPAS (`dpas.8x8`)** | 36 clk latency, 16 clk throughput/XVE | 4096 FLOP/instruction (BF16/FP16) |

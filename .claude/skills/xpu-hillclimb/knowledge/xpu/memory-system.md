# Intel XPU Memory System Rules & Helion/Triton Levers

These rules govern real DRAM/L3/L1 traffic on Intel `Xe` (PVC) and `Xe2` (BMG B580 / Pro B70). Each rule lists the **hardware counter / ASM signature** produced by `pretuned_kernels/xpu_hw_collect.py` and the corresponding **Helion compiler / autotuner lever**.

## M1. Partial Cache-Line Stores Trigger Extra DRAM Reads (RFO on Eviction)
- **Mechanism**: L3 cache lines are `64 B`. A store message that does not cover a full `64 B` line leaves a partially dirty L3 line; when evicted to DRAM, the memory controller must read the line from DRAM to merge. Sixteen separate `4 B` stores still suffer eviction mid-fill under streaming pressure; only a single store message covering a full `64 B` line is immune.
- **Signature (`09_memory_load_store.json`)**:
  - `measured_read_bytes` exceeds `projected_read_bytes` by up to `projected_write_bytes`.
  - `internal_set5.std.l3.wr.partial` is high (on `Xe2`).
- **Helion / Triton Lever**:
  - Ensure inner contiguous tile dimension (`block_sizes` along stride-1 dim) writes at least `64 B` per subgroup send (e.g., `>= 32` elements for FP16/BF16, `>= 64` for FP8/INT8).
  - Prefer `indexing="tensor_descriptor"` (2D block store when aligned and >= 64B row width) or contiguous pointer stores over scattered stores.

## M2. 256B DRAM Sector Granularity & Sparse Writes
- **Mechanism**: DRAM is accessed in `256 B` blocks (four `64 B` L3 lines). Writing only part of a `256 B` block (e.g., strided 64B writes) or reading narrow `< 256 B` tiles in column-major order causes `2.0x` DRAM traffic amplification when the working set exceeds L3.
- **Signature (`09_memory_load_store.json`)**:
  - `read_amplification` $\approx 2.0\times$ despite coalesced loads.
- **Helion / Triton Lever**:
  - Keep inner contiguous `block_size * elem_bytes >= 256 B` (e.g., `block_size >= 128` for FP16/BF16, `>= 256` for FP8).
  - Order grid/tile traversal (`l2_groupings` or `loop_orders`) so adjacent workgroups access contiguous `256 B` blocks concurrently.

## M3. Scalar 16-bit Stores (`d16u32`) Cause 4x L3 Write Amplification
- **Mechanism**: L3 has `4 B` (dword) write enables only. When IGC emits `store.ugm.d16u32` (2-byte store in a 4-byte container), a `SIMD32` store to a `64 B` line splits into **4 partial L3 writes** (even/odd lanes per SIMD16 half), triggering M1 extra DRAM merge reads.
- **Signature (`09_memory_load_store.json` & `11_asm_source_mapping.json`)**:
  - `l3_write_over_min` $\approx 4.0$.
  - `asm_census.d16u32_stores > 0` in `11_asm_source_mapping.json`.
- **Helion / Triton Lever**:
  - Ensure contiguous dimension tiling and `num_warps` allow Triton to vectorize 16-bit/8-bit stores (`>= 2` or `4` elements per lane along the contiguous dimension) so IGC emits `store.ugm.d32` / `d32x2` / `d32x4` or `store_block2d` instead of `d16u32`.

## M4. Multi-Pass L1/L3 Residency vs. Static Config Traffic Amplification
- **Mechanism**: Many normalization/softmax/quantization kernels (e.g. `rms_norm_dynamic_per_token_quant`, `layer_norm`, `softmax`) iterate over a row 2–3 times. Helion's static `BoundKernel.roofline()` sums all `hl.load` calls across loops and reports `traffic_amplification = 2.0x–3.7x` (`tile_traffic_bound`).
  - **When row footprint per WG fits in L1 (`<= 32–64 KB/WG`) or L3 (`resident_WGs * row_bytes <= LLC`)**: Pass 2 and Pass 3 hit in L1/L3!
- **Signature (`09_memory_load_store.json`)**:
  - `05_static_roofline.json` says `traffic_amplification > 2.0x` (`tile_traffic_bound`), **BUT** `09_memory_load_store.json` shows `read_amplification` $\approx 1.0\times$ and `measured_over_config_predicted` $\ll 1.0$ (e.g. `0.27x`).
- **Helion / Triton Lever**:
  - **Do NOT try to shrink DRAM traffic when `read_amplification` is already ~1.0x!** Instead, check `10_instructions_measurement.json`: the kernel is either **Instruction/Pipe bound** (`T_instruction_ms > T_mem_ms`) or **MLP/Latency bound** (`num_warps` too small).
  - Conversely, if `read_amplification` in `09_memory_load_store.json` really IS $> 1.3\times$ on a multi-pass kernel, the per-WG row footprint evicted L1/L3: use `load_eviction_policies` (`"last"` / `"first"`) or reduce concurrent resident footprint per WG.

## M5. Bytes in Flight (Little's Law: ~2 KB Outstanding per XVE)
- **Mechanism**: To saturate DRAM bandwidth ($BW \times L$), each XVE needs roughly `2 KB` of bytes in flight (`bytes_per_load_message * active_threads_per_XVE * independent_loads_in_flight`).
- **Signature (`09_memory_load_store.json` & `10_instructions_measurement.json`)**:
  - `read_amplification` $\approx 1.0\times$, `bw_utilization < 0.50`, `T_instruction_ms << T_actual_ms`, and `SbidStall` is high right after loads.
- **Helion / Triton Lever**:
  - Increase `block_sizes` or `num_warps` so each subgroup issues wider vector loads (`load.ugm.d32x2` / `d32x4`) or more concurrent subgroups per WG.
  - Increase `num_stages` / `range_num_stages` or `range_unroll_factors` so multiple loads are issued before the first consumption wait (`SbidStall`).
  - Avoid `grf_mode="256"` on memory-bound kernels if `128` GRF does not spill, since `256` GRF cuts max threads per XVE from 8 to 4.

## M6. Hardware Lossless Compression (`ComputeBasic` vs `ComputeBasic.nocomp`)
- **Mechanism**: Intel XPU has a hardware compressor between L3 and DRAM. Zero/constant buffers or small-valued `int64` index tensors compress 2–4x.
- **Signature (`09_memory_load_store.json`)**:
  - `measured_write_bytes < projected_write_bytes` while `compression_off.measured_write_bytes` $\approx$ `projected_write_bytes`.
- **Helion / Triton Rule**:
  - Always benchmark with `torch.randn` (non-constant random values). Use `compression_off` in `09_memory_load_store.json` to verify logical traffic accuracy.

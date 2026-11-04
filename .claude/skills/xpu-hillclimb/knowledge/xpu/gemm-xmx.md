# Intel XPU XMX / DPAS, 2D Block Messages & GEMM Rules

Reference for optimizing matrix multiplication (`scaled_mm`, `matmul`, `grouped_gemm`, attention/linear projections) on Intel XPU (`Xe` PVC and `Xe2` BMG B580 / Pro B70).

## 1. XMX / DPAS Throughput & Accumulator Sizing

- **Instruction**: `dpas.8x8` (`SIMD16` execution width) computes an $8 \times 8 \times 16$ (FP16/BF16) or $8 \times 8 \times 32$ (FP8/INT8) matrix multiply-accumulate:
  - **FP16 / BF16**: `4096 FLOPs` per `dpas.8x8` instruction, throughput `16 clocks / XVE`.
  - **FP8 / INT8**: `8192 OPs` per `dpas.8x8` instruction, throughput `16 clocks / XVE`.
- **Counter Accounting (`10_instructions_measurement.json`)**:
  - `XVE_INST_EXECUTED_ALU2_ALL[events]` (`Xe2`) or `XVE_INST_EXECUTED_XMX_ALL[events]` (`Xe`) records **8 events per `dpas.8x8`** (`16 clocks` busy -> `2 clocks / event`).
  - L1-resident GEMM reaches >98% of theoretical XMX peak when `ALU2` / `XMX` utilization is near 100%.
- **Register Footprint & `grf_mode="256"`**:
  - A $64 \times 32$ FP32 accumulator tile per thread consumes `128 GRF` (`8 KB`).
  - Therefore, XMX GEMM kernels with large register tiles (`block_m >= 128`, `block_n >= 128`) **require `grf_mode="256"`** (`Large GRF`) to avoid catastrophic register spilling (`spill_bytes_per_thread > 0`).
  - In `helion/autotuner/`, `TritonXpuMatmulHeuristic` seeds `grf_mode="256"` for large GEMM tiles.

## 2. 2D Block Load / Store (`load_block2d` / `store_block2d`)

- **Hardware Support**: Both `Xe` (PVC) and `Xe2` (BMG/B70) feature hardware 2D Block Load/Store Engines in the LSC (`load_block2d.ugm`, `store_block2d.ugm`, `prefetch_block2d`), surfaced in Helion via `indexing="tensor_descriptor"` (or `block_ptr`).
- **Geometry & Alignment Rules**:
  - Base pointer must be `64 B` aligned; row pitch must be `16 B` aligned.
  - Store width per row should be a multiple of `64 B` (e.g. `>= 32` FP16/BF16 columns or `>= 64` FP8 columns) to avoid partial-line L3 write penalties (M1).
- **Signature in `11_asm_source_mapping.json`**:
  - `asm_census.load_block2d > 0` and `asm_census.store_block2d > 0`.
  - If `indexing="tensor_descriptor"` is configured in Helion but `asm_census.load_block2d == 0`, check alignment/stride constraints or Triton XPU lowering fallbacks.

## 3. Two-Level Blocking & L3 Panel Reuse (`l2_groupings`)

- **Roofline Ridge Point**:
  - XMX FP16 Ridge Point is **~256 FLOP/B** on PVC 1550 / B580 and **~302 FLOP/B** on Arc Pro B70.
  - To reach compute-bound XMX peak when streaming from DRAM, workgroups must share $A$ and $B$ panels in the L3 cache (`18 MiB` on B580, `24 MiB` on B70, `192 MiB` on 1-stack PVC).
- **Helion / Autotuner Lever**:
  - Set `l2_groupings` (e.g. `4`, `8`, `16`) in `Config` so concurrent workgroups traverse the output grid in panel-major order, keeping an $\approx 512 \times 512$ panel resident in L3 and reducing DRAM read amplification from $3\text{--}5\times$ down to $\approx 1.1\times$.
  - Verify in `09_memory_load_store.json` that `measured_read_bytes` drops toward `B_bytes + num_panels * A_bytes` while `l3_read_bytes` serves the inter-WG reuse.

## 4. Epilogue Overhead in Quantized / Scaled GEMM (`scaled_mm`)

- **Problem**: In `scaled_mm` ($D = \text{cast}(\alpha \cdot (A \times B) \cdot \beta)$), when per-row/per-col scales are applied in the epilogue or inside the K-loop, broadcasting or uncoalesced scale loads and software type conversions can dominate small/medium $M, N$ shapes or prevent `store_block2d`.
- **Diagnosis**:
  - Check `10_instructions_measurement.json` (`MATRIX` vs `ALU0`/`ALU1` pipe time) and `11_asm_source_mapping.json` (`hot_source_locations` in epilogue vs `tl.dot`).
  - If `T_pipe(ALU0/ALU1) > T_pipe(MATRIX)` on a GEMM kernel, the epilogue or pointer math is bottlenecking XMX issue rate.

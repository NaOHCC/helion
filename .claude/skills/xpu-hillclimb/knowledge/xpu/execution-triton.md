# Intel XPU Execution Pipes, Stalls & Helion/Triton Codegen Rules

This guide maps hardware execution bottlenecks reported in `07_kernel_occupancy.json`, `08_register_bank_conflict.json`, `10_instructions_measurement.json`, and `11_asm_source_mapping.json` directly to **Helion compiler (`helion/_compiler/`) and autotuner (`helion/autotuner/`)** optimizations.

## 1. XVE Execution Pipes Overview

| Pipe | Executed Counter | Native Work |
| :--- | :--- | :--- |
| **`ALU0` (FP)** | `XVE_INST_EXECUTED_ALU0_ALL[events]` | FP32 (`SIMD16`), FP16/BF16 arithmetic |
| **`ALU1` (INT)** | `XVE_INST_EXECUTED_ALU1_ALL[events]` | INT32/16/8, bit-casts (e.g. FP8 conversion), direct `mov`, indirect `r[a0]` reads, loop control (`add`/`cmp`), FP64/INT64 (`:q`) address math |
| **`MATH` (SFU)** | `XVE_INST_EXECUTED_MATH[events]` | `math.inv` (rcp/div), `rsqrt`, `sqrt`, `exp`, `log`, `sin`, `cos` (native `SIMD4` -> 1 `SIMD32` instruction takes **8 issue slots**) |
| **`MATRIX` (XMX)** | `XVE_INST_EXECUTED_ALU2_ALL` (`Xe2`) / `XMX_ALL` (`Xe`) | `dpas.8x8` matrix multiply-accumulate (1 event = 2 clocks on XMX) |
| **`SEND` (Memory)** | `XVE_INST_EXECUTED_SEND_ALL[events]` | Global UGM loads/stores, 2D block loads/stores, SLM loads/stores, gateway barriers |
| **`CONTROL` (Jump)** | `XVE_INST_EXECUTED_CONTROL_ALL[events]` | `jmpi`, `goto`, `join` branch instructions |

- **Pipe Lower-Bound Formula**:
  $$T_{\text{pipe}} = \frac{\text{slots} \times \text{clk\_per\_slot}}{\text{num\_XVE} \times \text{clock\_Hz}}$$
  When $\max(T_{\text{pipe}}) \ge 0.8 \times T_{\text{mem}}$, `10_instructions_measurement.json` reports `primary_bound: "instruction"`.

---

## E1. Persistent Grid Oversubscription on Small/Medium Grids
- **Problem**: When a config uses `pid_type="persistent_blocked"` or `"persistent_interleaved"` with a large `num_sm_multiplier` (e.g. `num_sm_multiplier=8` or `32` -> `64 * 32 = 2048` workgroups launched) on a shape that only has `P < launched_WGs` logical tiles (e.g. `num_tokens = 128`), `launched_WGs - P` workgroups are dispatched to hardware only to immediately exit the loop after paying thread-dispatch and prologue payload-load (`load.ugm...bti[255]`) overhead.
- **Signature (`07_kernel_occupancy.json` & `10_instructions_measurement.json`)**:
  - `launch_params.num_workgroups` $\gg$ logical tile count in `05_static_roofline.json`.
  - `xve_active_pct < 25%`, `xve_stall_pct > 60%`, high `InstrFetchStall` / `DistStall`.
- **Helion Compiler / Autotuner Lever**:
  - In Helion's persistent grid launcher / codegen (`helion/_compiler/`), clamp the launched persistent grid size to `min(persistent_programs, total_pids)` so empty workgroups are never launched!
  - In Autotuner search space / XPU seeds, prefer `pid_type="flat"` when `total_pids <= 2 * num_xe_cores`.

## E2. Runtime Floating-Point & Integer Divisions (`math.inv`) Inside Inner Loops
- **Problem**: `math.inv` executes on the narrow `SIMD4` `MATH` pipe (8 slots per `SIMD32` instruction) and integer division/modulo (`a // b`, `a % b` with non-constexpr divisor) expands to ~15 ALU1 + MATH instructions per lane.
  - Common case in quantization / normalization kernels: `y_blk = x_blk / s_blk[:, None]` inside a `for tile_n in hl.tile(hidden_size):` loop where `s_blk` is loop-invariant across `tile_n`!
- **Signature (`11_asm_source_mapping.json`)**:
  - `asm_census.math_inv` is high, and `hot_source_locations` shows `PipeStall` / `DistStall` on a division line inside an inner `tl.range` loop.
- **Helion Compiler / Autotuner Lever**:
  - In Helion's DeviceIR / loop-invariant code motion or Triton lowering (`helion/_compiler/`), hoist loop-invariant reciprocals (`1.0 / s_blk`) out of inner reduction/tile loops or strength-reduce division by loop-invariant tensors into multiplication by reciprocal.
  - Specialize or power-of-two tile divisors to avoid runtime integer `div`/`rem`.

## E3. `ALU1` Saturation & `PipeStall` from Software FP8 Conversion (`tl.cast(..., tl.float8e4nv)`) and 64-bit (`:q`) Address Math
- **Problem**:
  1. On XPU platforms without native FP8 conversion instructions (or when lowered via bit manipulation), `tl.cast(val, tl.float8e4nv)` expands into a dense sequence of `ALU1` integer shifts/masks/selects, saturating `ALU1` and causing `PipeStall`.
  2. 64-bit integer address math (`add (16|M0) r16:q ...`, counted in `asm_census.qword_ops`) writes 2-GRF destinations and holds the GRF write port ~3.7 extra cycles, amplifying `PipeStall` at high occupancy.
- **Signature (`10_instructions_measurement.json` & `11_asm_source_mapping.json`)**:
  - `dominant_pipe == "ALU1"`, `PipeStall` > 15%.
  - `asm_census.qword_ops` is high or `hot_source_locations` points to `tl.cast(..., tl.float8e4nv)` or pointer offset additions.
- **Helion Compiler / Autotuner Lever**:
  - Use 32-bit index arithmetic (`index_dtype=torch.int32`) whenever tensor `numel < 2^31`.
  - Test `indexing="tensor_descriptor"` vs `"pointer"` in autotuner seeds to amortize per-lane 64-bit address arithmetic.
  - Tune `num_warps` and `block_sizes` so multiple subgroups share workgroup progress rather than 1 subgroup executing a 4000-instruction unrolled loop (`InstrFetchStall`).

## E4. `grf_mode` (`"default"` 128-GRF vs `"256"` Large GRF) Trade-off
- **Problem**:
  - Setting `grf_mode="256"` doubles registers per thread (`128 -> 256`), which eliminates register spilling (`spill_bytes_per_thread == 0`) for large tile sizes or GEMM accumulators, **but halves max resident hardware threads per Xe-core (`64 -> 32`, capping theoretical occupancy at 50%)**.
  - Conversely, IGC's `-cl-intel-enable-auto-large-GRF-mode` may automatically switch a kernel with large unrolled loops to 256-GRF mode even when `grf_mode="default"` was passed! Always check `launch_params.registers_per_thread` in `07_kernel_occupancy.json`.
- **Signature (`07_kernel_occupancy.json` & `08_register_bank_conflict.json`)**:
  - `registers_per_thread == 256` (`large_grf == True`), `spill_bytes_per_thread == 0`, while `occupancy <= 0.50` and latency stalls (`SbidStall`) are high.
  - Or `registers_per_thread == 128` with `spill_bytes_per_thread > 0` / `scratch_sends > 0` / `lsc_byte_write > 0`.
- **Helion Compiler / Autotuner Lever**:
  - When `spill_bytes_per_thread > 0` or `scratch_sends > 0`: reduce live register pressure (smaller `block_sizes`, smaller `range_unroll_factors`, fewer `num_stages`) OR enable `grf_mode="256"`.
  - When `large_grf == True` with zero spills on a latency/memory-bound kernel: shrink `block_sizes` / `range_unroll_factors` or explicitly support pinning 128-GRF mode so occupancy rises back to 64 threads/Xe-core.

## E5. Single-Subgroup (`num_warps=1`) Under-Parallelization vs. Workgroup Barrier Overhead
- **Problem**:
  - With `num_warps=1` (`Local [32, 1, 1]`, 1 hardware thread/WG), cross-subgroup SLM reductions and `sync.bar` barriers are completely avoided (`barrier_wait == 0`), which is fast when `num_workgroups >= 32 * xe_cores` AND `hidden_size` is small (`<= 1024`).
  - However, when `hidden_size` is large (`4096–8192`) or `num_workgroups` is small (`< 4 * xe_cores`), `num_warps=1` forces a single subgroup to loop dozens of times serially per row and leaves XVEs starved!
- **Signature (`07_kernel_occupancy.json` & `11_asm_source_mapping.json`)**:
  - `threads_per_wg == 1`, `asm_census.instructions > 3000`, high loop trip count, low bandwidth utilization.
- **Helion Compiler / Autotuner Lever**:
  - Ensure XPU autotuner seeds cover both `num_warps=1` (for high-token-count, small-row shapes where avoiding SLM barrier wins) and `num_warps=4, 8, 16` (for small-token-count or wide-row shapes where intra-row parallelism is needed to fill XVEs).

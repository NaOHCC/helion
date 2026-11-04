---
name: xpu-hillclimb
description: Iteratively optimize a Helion kernel on Intel XPU (HELION_BACKEND=triton) using Roofline and unitrace/ocloc hardware counter attribution until it matches or beats the best available baseline — via general compiler/autotuner changes (knobs, seeds, and lowering passes, not kernel-specific hacks) — then clean up history and report before/after numbers.
hooks:
  Stop:
    - hooks:
        - type: command
          command: python3 "$CLAUDE_PROJECT_DIR/.claude/skills/xpu-hillclimb/scripts/check_goal.py" --gate
---

Your task is to improve the performance of a specific Helion kernel on Intel XPU (`Xe` Data Center GPU Max 1550/1100 or `Xe2` Arc B580 / Pro B70 under `HELION_BACKEND=triton`) until it matches or beats the best available baseline. Limit your changes to the **Helion compiler/backend and autotuner** (`helion/_compiler/`, `helion/autotuner/`, `helion/runtime/`); do not make changes to the Helion frontend language or target kernel DSL implementation. Make your changes as general as possible, so optimizations apply to many different kernels — don't over-specialize to one specific kernel. When you hit a meaningful choice (e.g. A is better for one shape/kernel and B is better for another), add an autotuner knob so we can find the answer automatically. Avoid hardcoding shape- or kernel-specific heuristics. The one place heuristics are OK is autotuner seeds: a seed gives the autotuner a head start without hardcoding the choice. If Helion's Triton XPU codegen or autotuner design is a limiting factor, feel free to refactor and improve it.

# Goal contract — do not stop early

Completion is defined operationally and enforced mechanically:

- **Definition of done:** for EVERY variant/shape, Helion on XPU matches or beats the best baseline (`torch`, `torch_compile`, `vllm`, etc.) within a 1% noise margin (`ratio >= 0.99`), AND the geomean speedup is `>= 1.0` (plus any optional `sol_pct_min` Roofline floor configured in `criteria`). Both come from **cold-full-autotune** numbers (`HELION_AUTOTUNE_EFFORT=full HELION_SKIP_CACHE=1`); where interleaved ABAB verify runs exist, they are the ground truth. "Geomean >= 1 but some variants behind" is NOT done.
- **Goal file:** in Step 1 you create `artifacts/goal.json` with one entry per variant/shape. After every reported measurement, update that variant's `ratio` (`baseline_time / helion_time`; `>1` means Helion is faster), optional `sol_pct` (from `BoundKernel.roofline()` / `run_roofline.py`), and `artifact` (path to the raw result file that backs it). Never record a ratio without an artifact.
- **Stop gate:** this skill registers a Stop hook that runs `scripts/check_goal.py --gate` whenever you try to end your turn. While any variant fails, it blocks the stop and lists the failing variants: go back to Step 2 instead of summarizing. Run `python3 .claude/skills/xpu-hillclimb/scripts/check_goal.py` yourself at any time to see the current verdict.
- **Only two legitimate ways to end with the goal unmet:**
  1. The user explicitly tells you to stop or pause → set `"status": "paused"` in the goal file, quoting their instruction in `"paused_reason"`.
  2. You have falsification evidence the goal is unreachable within scope → set `"status": "blocked"` and put a concrete question in `"question_for_user"`, together with the evidence. Falsification evidence means: you hand-edited the generated Triton kernel (`kernels/k0_triton.py`) for the worst failing variant to adopt the optimal structure and confirmed via `unitrace` / `ocloc` that the remaining gap comes from an upstream Triton XPU / IGC compiler limitation outside Helion's control, and included that ablation in the evidence. If the hand-edited Triton kernel DOES reach parity, that is not impossibility evidence — it is the specification of your next Helion compiler pass or autotuner knob; keep going.
  Never silently downgrade the goal, and never present a final report while the checker says a variant fails.
- **Context-loss protection:** long hillclimbs get compacted. At the start of every optimization iteration re-read `artifacts/goal.json` and the iteration log in your plan doc; after every iteration append one log line (worst variant, hardware bottleneck signature, idea tried, cold-autotune result, checker verdict). If a summary in your context claims success, trust the checker output, not the summary.

# Hardware knowledge & profiling harness

Read the Intel XPU hardware knowledge base under `.claude/skills/xpu-hillclimb/knowledge/xpu/` before diagnosing or designing changes:
- `knowledge/xpu/hardware.md`: Topology, clocks, SLM, L1/LSC, L3, and Roofline ridge points for PVC Max 1550/1100 (`Xe`) and Arc B580 / Pro B70 (`Xe2`).
- `knowledge/xpu/memory-system.md`: M1–M6 rules (64B partial-line write RFO penalty, 256B DRAM granularity, `d16u32` 4x L3 write amplification, L1/L3 multi-pass residency vs static config traffic amplification, Little's law ~2KB/XVE bytes-in-flight, hardware compression).
- `knowledge/xpu/execution-triton.md`: E1–E5 rules mapped directly to Helion compiler & autotuner levers (`ALU0`/`ALU1`/`XMX`/`SEND` pipe slots, persistent grid oversubscription clamping, loop-invariant division/`math.inv` hoisting, FP8 cast & 64-bit `:q` `ALU1` `PipeStall`, `grf_mode="default"` vs `"256"`, `num_warps` single-subgroup vs multi-subgroup tradeoffs).
- `knowledge/xpu/gemm-xmx.md`: XMX `dpas.8x8` throughput, 2D block loads/stores (`indexing="tensor_descriptor"`), `l2_groupings` panel traversal, and GRF banking.

Use the built-in XPU profiling tools:
- `python pretuned_kernels/run_roofline.py --kernels <name> --output artifacts/<run-name>/roofline.json` — sweeps all shapes, comparing Helion against baselines and computing static + empirical Roofline `SOL%`, `CfgSOL%`, traffic amplification, and waves/Xe-core.
- `python pretuned_kernels/xpu_hw_collect.py collect --kernel <name> --shape "<shape>" --out artifacts/<run-name>/hw_<shape>` — runs un-instrumented timing + `BoundKernel.roofline()`, `unitrace -d -v` timeline/occupancy, `unitrace -q` hardware counter groups (`ComputeBasic`, `L1Profile*` / `VectorEngine*`, `InternalSet*`), `unitrace --stall-sampling`, and `DumpZEBin=1 + ocloc disasm`, writing `05..11_*.json`, `kernels/k0_triton.py`, `kernels/k0.stall.asm`, and `hw_profile_report.md`.

# General notes

## Scratch artifacts

Pick a run name at the start (e.g. `rms-norm-quant-xpu-2026-11-04`) and write ALL non-committed outputs — raw benchmark JSONs, `xpu_hw_collect.py` directories, winning autotuner configs, probe logs — under `artifacts/<run-name>/` in the repo root; the goal file sits at the fixed path `artifacts/goal.json`. The `artifacts/` directory is gitignored; never delete or overwrite the raw files backing reported numbers.

## Shared machine & XPU hygiene

- Check for other XPU users (`ps aux | grep python`) and pin to a single XPU device/tile via `ZE_AFFINITY_MASK=0` (or `ONEAPI_DEVICE_SELECTOR=level_zero:0`) so measurements are consistent.
- **Avoid XPU Level Zero graph capture with 256MB L2 flushes:** `tritonbench`'s `_do_bench_cudagraph_with_cache_clear` can trigger `UR_RESULT_ERROR_DEVICE_LOST` or `OUT_OF_DEVICE_MEMORY` on XPU; `run_roofline.py` and `xpu_hw_collect.py` automatically use cold-L2 `triton.testing.do_bench` or `stride2` rotating buffer sets on XPU.
- **Counter perturbation:** `unitrace -q` and `--stall-sampling` perturb kernel execution time by 10–35%. Always take latency/speedup numbers from un-instrumented runs (`run_roofline.py` or `05_static_roofline.json`), and use `unitrace` outputs strictly for counter/stall/traffic attribution.
- **One config per process:** `kernel.bind(args)` memoizes by args; compare configs in separate processes.

## Commit your work as you go

Group work into logical, complete chunks where everything is working, and commit each chunk to save progress. Use the `xpu-verify` skill to test and review each chunk before committing it.

# Hillclimbing task

## Step 1: Establish a target kernel, baseline, and starting `goal.json`

Decide what to optimize and establish the baseline:
1. Identify the target Helion kernel (in `pretuned_kernels/<name>/`, `examples/`, or `benchmarks/`) and select representative shapes (use 8 representative shapes covering small/latency-bound, medium, and large/bandwidth-or-compute-bound regimes if not otherwise specified, or the full benchmark shape sweep).
2. Verify numeric correctness against the PyTorch reference (`correctness_check()` or `check_example`).
3. Measure and record the starting cold-autotune performance (or current heuristic baseline if starting from a pretuned kernel, followed by a cold full autotune run `HELION_AUTOTUNE_EFFORT=full HELION_SKIP_CACHE=1`):
   - Save raw benchmark + Roofline JSON under `artifacts/<run-name>/before.json` and winning configs under `artifacts/<run-name>/before_configs.json`.
4. Create `artifacts/goal.json` (archive any previous `artifacts/goal.json` into its old run directory first):

```json
{
  "task": "rms_norm_dynamic_per_token_quant",
  "status": "running",
  "criteria": {"geomean_min": 1.0},
  "shapes": [
    {
      "name": "5120x2048",
      "ratio": 0.83,
      "sol_pct": 17.8,
      "artifact": "artifacts/rms-norm-quant-xpu/before.json"
    }
  ]
}
```

## Step 2: Diagnose with `xpu_hw_collect.py` and optimize the worst failing variant

Pick the shape/variant with the largest performance gap (`lowest ratio`) to the best baseline:

1. **Collect hardware counters, stalls, and disassembly:**
   ```bash
   python pretuned_kernels/xpu_hw_collect.py collect \
       --kernel <name> --shape "<worst_shape>" \
       --out artifacts/<run-name>/iterN_hw
   ```
2. **Formulate a quantitative hardware theory** from `hw_profile_report.md` and `05..11_*.json`:
   - Compare **Static Config Traffic (`05_static_roofline.json`)** vs **Measured DRAM & L3 Traffic (`09_memory_load_store.json`)**:
     - Does multi-pass traffic hit in L1/L3 (`read_amplification ~ 1.0x`), or does it spill to DRAM (`read_amplification >> 1.0x`)?
     - Are there partial-line writes or `d16u32` store amplifications (`l3_write_over_min >> 1.0`)?
   - Check **Occupancy & Launch Parameters (`07_kernel_occupancy.json` & `08_register_bank_conflict.json`)**:
     - Did `pid_type="persistent_*"` with `num_sm_multiplier` launch thousands of empty workgroups for a small grid (E1)?
     - Did `grf_mode="256"` halve occupancy unnecessarily when `spill_bytes == 0`, or did `grf_mode="default"` spill registers (`spill_bytes > 0` / `scratch_sends > 0`) (E4)?
     - Is `num_warps=1` starving XVEs on a wide row, or are multi-warp SLM barriers (`barrier_wait`) dominating a small row (E5)?
   - Check **Pipe Utilization & Stalls (`10_instructions_measurement.json` & `11_asm_source_mapping.json`)**:
     - Which pipe (`ALU0`, `ALU1`, `MATH`, `MATRIX`, `SEND`) has the highest `T_ms` and utilization?
     - Which exact Helion DSL / Triton lines in `hot_source_locations` account for the top `PipeStall`, `SbidStall`, `InstrFetchStall`, or `DistStall` events (E2, E3)?
3. **Prototype and implement a general compiler / autotuner fix:**
   - You can quickly test hypotheses by editing the dumped Triton kernel (`artifacts/<run-name>/iterN_hw/kernels/k0_triton.py`) and timing it on the same rotating sets.
   - Once a mechanism is proven, implement it **generally in Helion's compiler (`helion/_compiler/`) or autotuner (`helion/autotuner/`)**:
     - General compiler passes (e.g. clamping persistent grid size to logical tile count, hoisting loop-invariant reciprocals/divisions, using `int32` index offsets when `numel < 2^31`, improving XPU block_ptr / tensor_descriptor lowering).
     - Autotuner search space knobs & XPU heuristic seeds (`helion/autotuner/`).
4. **Verify via cold full autotuning:**
   - An optimization only counts once a **cold full autotune** (`HELION_AUTOTUNE_EFFORT=full HELION_SKIP_CACHE=1` with a fresh seed) finds the winning configuration automatically.
   - Re-run `xpu_hw_collect.py collect` on the optimized kernel to confirm the predicted hardware counter/stall signature moved as expected.

## Step 3: Completion check, or loop again

Refresh `artifacts/goal.json` from your newest cold-full-autotune numbers across all target shapes, then run the checker:

```bash
python3 .claude/skills/xpu-hillclimb/scripts/check_goal.py
```

When a variant sits within ~2% of the `0.99` bar, settle it with an interleaved ABAB verify run on the same XPU tile and record that ratio. Keep looping back to Step 2 on the worst failing variant until `check_goal.py` prints `GOAL MET` (or a legitimate `paused`/`blocked` condition is met).

## Step 4: Clean up git history & verify

Rewrite your git history (relative to `origin/main` or the starting branch point) into clear, logical commits:
- Group compiler pass improvements, autotuner knob additions, and XPU seed improvements into clean, self-contained commits.
- Raw benchmark artifacts, `unitrace` logs, and `artifacts/goal.json` stay uncommitted under `artifacts/`.
- Run the **`xpu-verify`** skill to execute `./lint.sh`, XPU/Roofline unit tests, and a code review subagent on your commits.

## Step 5: Final report

Write the final report only once `check_goal.py` prints `GOAL MET` (or the user has accepted a `paused`/`blocked` outcome). Include:
1. The exact `check_goal.py` output.
2. A per-shape comparison table: `Shape | Baseline (us) | Helion Before (us) | Helion After (us) | Speedup Before | Speedup After | SOL% Before | SOL% After | Artifact`, ending with geomean for speedups and arithmetic mean for SOL% / GB/s / TFLOP/s.
3. Hardware counter attribution summary (before vs after DRAM/L3 traffic, occupancy, dominant pipe slots, and stall percentages from `xpu_hw_collect.py`).
4. Summary of the general Helion compiler and autotuner changes made.

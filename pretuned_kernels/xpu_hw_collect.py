#!/usr/bin/env python3
"""Intel XPU hardware performance counter, stall, and assembly collection harness.

Inspired by ``kernelbench-torch`` (``kbtorch.profile`` and ``kbtorch.steps``),
this standalone tool wraps Helion kernels (in ``pretuned_kernels``) with Intel
``unitrace`` and ``ocloc disasm``, combining Helion's static/empirical
``BoundKernel.roofline()`` model with hardware-level measurements:

- **Un-instrumented benchmark & static Roofline** (``05_static_roofline.json``)
- **Timeline & Kernel Launch Properties** via ``unitrace -d -v``
- **Hardware Counter Groups** via ``unitrace -q -g <Group>`` (auto-adapted for
  both ``Xe`` PVC Max 1550/1100 and ``Xe2`` BMG Arc B580 / Pro B70)
  (``06_kernel_profiler_parser.json``)
- **Xe-core Occupancy & Wave Analysis** (``07_kernel_occupancy.json``)
- **Register File (GRF 128/256), Spill & Bank Conflicts**
  (``08_register_bank_conflict.json``)
- **Memory Hierarchy Traffic (DRAM vs L3 vs L1 LSC vs SLM) & Amplification**
  (``09_memory_load_store.json``)
- **XVE Pipe Slot Utilization & Stall Breakdown**
  (``10_instructions_measurement.json``)
- **GEN ISA Disassembly Census & Stall-to-Triton/Helion Source Mapping**
  (``11_asm_source_mapping.json``)
- **Human/Agent-readable Diagnostic Summary** (``hw_profile_report.md``)

Usage:
    python pretuned_kernels/xpu_hw_collect.py list --kernel rms_norm_dynamic_per_token_quant
    python pretuned_kernels/xpu_hw_collect.py run --kernel rms_norm_dynamic_per_token_quant --shape-index -1
    python pretuned_kernels/xpu_hw_collect.py collect --kernel rms_norm_dynamic_per_token_quant --out /tmp/rms_quant_hw
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
from typing import TYPE_CHECKING
from typing import Any

import torch

PRETUNED_KERNELS_DIR = Path(__file__).resolve().parent
REPO_ROOT = PRETUNED_KERNELS_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(PRETUNED_KERNELS_DIR) not in sys.path:
    sys.path.insert(0, str(PRETUNED_KERNELS_DIR))

import _bench  # noqa: E402  # pyrefly: ignore[missing-import]
from run import (  # noqa: E402
    _import_kernel_module,  # pyrefly: ignore[missing-module-attribute]
)

from helion._hardware import get_hardware_roofline_spec  # noqa: E402
from helion.runtime.kernel import BoundKernel  # noqa: E402
from helion.runtime.kernel import Kernel  # noqa: E402

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterable
    from collections.abc import Iterator
    from collections.abc import Sequence

    from helion._compiler.roofline import RooflineReport
    from helion._hardware import HardwareRooflineSpec
    from helion.runtime.config import Config


# ---------------------------------------------------------------------------
# Hardware Architecture & Group Defaults
# ---------------------------------------------------------------------------

XE2_DEFAULT_GROUPS = (
    "ComputeBasic",
    "VectorEngineProfile",
    "VectorEngineStalls",
    "InternalSet5",
    "InternalSet3",
)
XE_PVC_DEFAULT_GROUPS = (
    "ComputeBasic",
    "L1ProfileReads",
    "L1ProfileWrites",
    "L1ProfileSlmBankConflicts",
)

THREADS_PER_CORE = 64
MAX_WGS_PER_CORE = 64
SLM_PER_CORE_KB = 128
SLM_SIZES_KB = (0, 1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128)


def find_unitrace() -> str | None:
    """Locate the ``unitrace`` binary on PATH or common build locations."""
    env_ut = os.environ.get("HELION_UNITRACE") or os.environ.get("KBTORCH_UNITRACE")
    if env_ut and Path(env_ut).is_file():
        return env_ut
    which_ut = shutil.which("unitrace")
    if which_ut:
        return which_ut
    candidates = [
        Path("/usr/local/bin/unitrace"),
        Path("/workspaces/pytorch/pti-gpu/tools/unitrace/build/unitrace"),
        Path("/workspaces/helion/pti-gpu/tools/unitrace/build/unitrace"),
    ]
    for cand in candidates:
        if cand.is_file():
            return str(cand)
    return None


def detect_aot_device(hw_spec: HardwareRooflineSpec) -> str:
    """Return the ``ocloc disasm -device`` target name (`pvc` or `bmg`)."""
    if hw_spec.compute_capability.lower() == "xe2":
        return "bmg"
    return "pvc"


def default_counter_groups(hw_spec: HardwareRooflineSpec) -> tuple[str, ...]:
    """Return default ``unitrace -q -g`` metric groups for the active XPU."""
    if hw_spec.compute_capability.lower() == "xe2":
        return XE2_DEFAULT_GROUPS
    return XE_PVC_DEFAULT_GROUPS


def get_xpu_device_topology(
    hw_spec: HardwareRooflineSpec | None = None,
) -> dict[str, Any]:
    """Return Xe-core, XVE, SLM, LLC, and clock parameters for the XPU device."""
    spec = hw_spec or get_hardware_roofline_spec()
    xe_cores = max(1, spec.num_compute_units)
    num_xve = xe_cores * 8
    slm_kb = SLM_PER_CORE_KB
    llc_bytes = 24 << 20
    clock_ghz = 1.6 if spec.compute_capability == "Xe" else 2.80

    if torch.xpu.is_available():
        with contextlib.suppress(Exception):
            props = torch.xpu.get_device_properties(0)
            subslice = int(getattr(props, "gpu_subslice_count", 0) or 0)
            eu_count = int(getattr(props, "gpu_eu_count", 0) or 0)
            local_mem = int(getattr(props, "local_mem_size", 0) or 0)
            llc = int(getattr(props, "last_level_cache_size", 0) or 0)
            if subslice > 0:
                xe_cores = subslice
            if eu_count > 0:
                num_xve = eu_count
            else:
                num_xve = xe_cores * 8
            if local_mem > 0:
                slm_kb = local_mem // 1024
            if llc > 0:
                llc_bytes = llc

    name_upper = spec.hardware_name.upper()
    if "B580" in name_upper:
        clock_ghz = 2.85
    elif "B70" in name_upper or "B60" in name_upper:
        clock_ghz = 2.80
    elif "1550" in name_upper:
        clock_ghz = 1.60
    elif "1100" in name_upper:
        clock_ghz = 1.55

    return {
        "hardware_name": spec.hardware_name,
        "compute_capability": spec.compute_capability,
        "xe_cores": xe_cores,
        "num_xve": num_xve,
        "threads_per_core": THREADS_PER_CORE,
        "max_wgs_per_core": MAX_WGS_PER_CORE,
        "slm_per_core_kb": slm_kb,
        "llc_bytes": llc_bytes,
        "clock_ghz": clock_ghz,
        "peak_memory_bw_gbps": spec.peak_memory_bandwidth_gbps,
        "peak_matrix_fp16_tflops": spec.matrix_tflops(torch.bfloat16),
        "peak_vector_fp32_tflops": spec.vector_tflops(torch.float32),
    }


def stride2_order(n: int) -> list[int]:
    """Return ``[0, 2, 4, ..., 1, 3, 5, ...]`` so consecutive calls never touch adjacent sets."""
    return list(range(0, n, 2)) + list(range(1, n, 2))


def compute_num_sets(
    per_set_bytes: int,
    llc_bytes: int,
    *,
    max_total_bytes: int = 1024 << 20,
    max_sets: int = 256,
) -> int:
    """Compute rotating set count so total footprint exceeds ``2 * llc_bytes`` (>= 64 MiB)."""
    if per_set_bytes <= 0:
        return 2
    target_footprint = min(max_total_bytes, max(64 << 20, 2 * llc_bytes))
    needed = math.ceil(target_footprint / per_set_bytes)
    capped_by_mem = max(2, max_total_bytes // per_set_bytes)
    return max(2, min(needed, capped_by_mem, max_sets))


# ---------------------------------------------------------------------------
# Pretuned Kernel Shape & Call Discovery
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _capture_kernel_dispatch(
    captured: list[tuple[BoundKernel[object], Config, tuple[object, ...]]],
) -> Iterator[None]:
    """Capture ``(bound_kernel, config, normalized_args)`` when a Helion kernel executes."""
    orig_call = Kernel.__call__

    def _wrapped_call(self: Kernel[object], *args: object, **kwargs: object) -> object:
        result = orig_call(self, *args, **kwargs)
        with contextlib.suppress(Exception):
            norm_args = self.normalize_args(*args, **kwargs)
            bound = self.bind(norm_args)
            cfg = (
                bound._config
                or bound._implicit_config()
                or bound.config_spec.default_config()
            )
            captured.append((bound, cfg, norm_args))
        return result

    Kernel.__call__ = _wrapped_call  # type: ignore[method-assign]
    try:
        yield
    finally:
        Kernel.__call__ = orig_call  # type: ignore[method-assign]


class _SweepCaptured(Exception):
    """Internal control-flow exception used to intercept ``_bench.run_sweep`` parameters."""

    def __init__(
        self,
        shapes: list[Any],
        make_calls: Callable[
            [Any],
            tuple[
                Callable[[], object],
                list[tuple[str, Callable[[], object]]],
                str,
            ],
        ],
        shape_header: str,
        use_cudagraph: bool,
        pre_captured_cudagraph: bool,
        make_resets: Callable[[Any], Sequence[Callable[[], object] | None]] | None,
        warmup: int,
        rep: int,
    ) -> None:
        super().__init__()
        self.shapes = shapes
        self.make_calls = make_calls
        self.shape_header = shape_header
        self.use_cudagraph = use_cudagraph
        self.pre_captured_cudagraph = pre_captured_cudagraph
        self.make_resets = make_resets
        self.warmup = warmup
        self.rep = rep


def discover_kernel_sweep(kernel_name: str) -> _SweepCaptured:
    """Import ``pretuned_kernels/<kernel_name>`` and capture its ``run_sweep`` arguments."""
    module = _import_kernel_module(kernel_name)
    orig_run_sweep = _bench.run_sweep

    def _intercept_run_sweep(
        shapes: Iterable[Any],
        make_calls: Callable[
            [Any],
            tuple[
                Callable[[], object],
                list[tuple[str, Callable[[], object]]],
                str,
            ],
        ],
        *,
        use_cudagraph: bool,
        pre_captured_cudagraph: bool = False,
        make_resets: Callable[[Any], Sequence[Callable[[], object] | None]]
        | None = None,
        shape_header: str,
        warmup: int = 25,
        rep: int = 100,
        thermal_warmup_ms: int = 0,
        verbose: bool = True,
    ) -> dict[str, Any]:
        raise _SweepCaptured(
            shapes=list(shapes),
            make_calls=make_calls,
            shape_header=shape_header,
            use_cudagraph=use_cudagraph,
            pre_captured_cudagraph=pre_captured_cudagraph,
            make_resets=make_resets,
            warmup=warmup,
            rep=rep,
        )

    _bench.run_sweep = _intercept_run_sweep  # type: ignore[assignment]
    try:
        module.main(verbose=False)
    except _SweepCaptured as captured:
        return captured
    finally:
        _bench.run_sweep = orig_run_sweep  # type: ignore[assignment]
    raise RuntimeError(f"Kernel {kernel_name!r} did not invoke _bench.run_sweep")


def select_shape_entry(
    sweep: _SweepCaptured,
    *,
    shape_index: int | None = None,
    shape_filter: str | None = None,
) -> tuple[int, Any, Callable[[], object], list[tuple[str, Callable[[], object]]], str]:
    """Select one shape by index or substring and return ``(idx, shape, helion_call, base_calls, label)``."""
    if not sweep.shapes:
        raise ValueError("Kernel sweep has no shapes")

    if shape_filter:
        norm_filter = " ".join(shape_filter.split())
        for idx, sh in enumerate(sweep.shapes):
            helion_call, base_calls, cells = sweep.make_calls(sh)
            norm_cells = " ".join(cells.split())
            if norm_filter in norm_cells or shape_filter in str(sh):
                return idx, sh, helion_call, base_calls, norm_cells
        raise ValueError(f"No shape matching {shape_filter!r} found in sweep")

    idx = -1 if shape_index is None else shape_index
    if idx < 0:
        idx = len(sweep.shapes) + idx
    if not (0 <= idx < len(sweep.shapes)):
        raise IndexError(
            f"shape_index {shape_index} out of range (0..{len(sweep.shapes) - 1})"
        )
    sh = sweep.shapes[idx]
    helion_call, base_calls, cells = sweep.make_calls(sh)
    return idx, sh, helion_call, base_calls, " ".join(cells.split())


def clone_arg_sets(
    norm_args: tuple[object, ...],
    n_sets: int,
) -> list[tuple[object, ...]]:
    """Create ``n_sets`` independent device buffer copies of ``norm_args``."""
    sets: list[tuple[object, ...]] = [norm_args]
    for _ in range(1, n_sets):
        cloned = tuple(
            arg.clone() if isinstance(arg, torch.Tensor) else arg for arg in norm_args
        )
        sets.append(cloned)
    return sets


def args_tensor_bytes(norm_args: tuple[object, ...]) -> int:
    """Sum of bytes of all distinct tensor arguments in ``norm_args``."""
    seen_ptrs: set[int] = set()
    total = 0
    for arg in norm_args:
        if isinstance(arg, torch.Tensor):
            ptr = arg.data_ptr()
            if ptr not in seen_ptrs:
                seen_ptrs.add(ptr)
                total += arg.numel() * arg.element_size()
    return total


# ---------------------------------------------------------------------------
# Unitrace Log & Disassembly Parsers
# ---------------------------------------------------------------------------


def _quoted_rows(text: str, section: str) -> list[tuple[str, list[str]]]:
    """Parse ``"name", v1, v2, ...`` rows of the unitrace table following ``section``."""
    out: list[tuple[str, list[str]]] = []
    on = False
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("==="):
            on = section in s
            continue
        if on and s.startswith('"'):
            end = s.rfind('"')
            if end > 0:
                out.append(
                    (
                        s[1:end],
                        [c.strip() for c in s[end + 1 :].lstrip(",").split(",")],
                    )
                )
    return out


def _parse_launch_signature(
    name: str,
) -> tuple[int | None, list[int] | None, list[int] | None]:
    """Extract ``(simd, [gx, gy, gz], [lx, ly, lz])`` from ``Foo[SIMD32 {g0; g1; g2} {l0; l1; l2}]``."""
    m = re.search(
        r"\[SIMD(\d+) \{(\d+); (\d+); (\d+)\} \{(\d+); (\d+); (\d+)\}\]$", name
    )
    if not m:
        return None, None, None
    vals = [int(x) for x in m.groups()]
    return vals[0], vals[1:4], vals[4:7]


def parse_timeline(text: str, tag: str) -> list[dict[str, Any]]:
    """Parse ``unitrace -d -v`` output for kernels matching ``tag``."""
    props = dict(_quoted_rows(text, "Kernel Properties"))
    kernels: list[dict[str, Any]] = []
    for name, v in _quoted_rows(text, "Device Timing Summary"):
        if tag not in name or name not in props:
            continue
        simd, groups, local = _parse_launch_signature(name)
        p = props[name]
        # p: Compiled, SIMD, Args, SLM/WG, Private/thread, Spill/thread, GRF/thread
        kernels.append(
            {
                "name": name,
                "base": name.split("[")[0],
                "calls": int(v[0]),
                "total_ns": float(v[1]),
                "avg_ns": float(v[3]),
                "min_ns": float(v[4]) if len(v) > 4 else float(v[3]),
                "max_ns": float(v[5]) if len(v) > 5 else float(v[3]),
                "simd": simd if simd is not None else int(p[1]),
                "groups": groups,
                "local": local,
                "compiled": p[0],
                "num_args": int(p[2]),
                "slm_bytes": int(p[3]),
                "private_bytes": int(p[4]),
                "spill_bytes": int(p[5]),
                "grf": int(p[6]),
            }
        )
    kernels.sort(key=lambda k: -k["total_ns"])
    return kernels


def _split_kernel_row(line: str, n_numeric: int) -> tuple[str, list[str]]:
    """Split a CSV row right-to-left so commas inside kernel names are preserved."""
    fields = line.split(",")
    name = ",".join(fields[: len(fields) - n_numeric]).strip().strip('"')
    return name, [f.strip() for f in fields[len(fields) - n_numeric :]]


def parse_unitrace_metrics(log_text: str) -> tuple[list[str], list[dict[str, str]]]:
    """Parse a ``unitrace -q`` metric log (supports both <=2.4.0 and 2.4.1+ 4-line block layouts)."""
    lines = [line.rstrip("\n") for line in log_text.splitlines()]
    starts = [
        i
        for i, line in enumerate(lines)
        if re.match(r"^=== Device #\d+ Metrics ===$", line.strip())
    ]
    if not starts:
        raise ValueError("No '=== Device #N Metrics ===' section found in log")

    header_fields: list[str] | None = None
    rows: list[dict[str, str]] = []
    metric_header: list[str] | None = None

    for st in starts:
        i = st + 1
        while i < len(lines) and not lines[i].strip().startswith("=== Device #"):
            line_s = lines[i].strip()
            if not line_s:
                i += 1
                continue
            cols = [c.strip() for c in line_s.split(",")]
            if cols[0] == "Kernel" and len(cols) == 2 and i + 3 < len(lines):
                # unitrace 2.4.1+ 4-line block layout
                name, (gid,) = _split_kernel_row(lines[i + 1].strip(), 1)
                mnames = [c.strip() for c in lines[i + 2].split(",")]
                mvals = [c.strip() for c in lines[i + 3].split(",")]
                if len(mnames) == len(mvals):
                    row: dict[str, str] = {"Kernel": name, "GlobalInstanceId": gid}
                    row.update(zip(mnames, mvals, strict=True))
                    rows.append(row)
                    metric_header = ["Kernel", "GlobalInstanceId", *mnames]
                i += 4
                continue
            if cols[0] == "Kernel" and len(cols) > 2:
                header_fields = cols if cols[1] == "GlobalInstanceId" else None
                i += 1
                continue
            if (
                header_fields
                and line_s.startswith('"')
                and len(header_fields) > 2
                and header_fields[1] == "GlobalInstanceId"
            ):
                name, vals = _split_kernel_row(line_s, len(header_fields) - 1)
                if len(vals) == len(header_fields) - 1:
                    row = {"Kernel": name}
                    row.update(zip(header_fields[1:], vals, strict=True))
                    rows.append(row)
                    metric_header = header_fields
            i += 1

    if not rows or metric_header is None:
        raise ValueError("Found a metrics section but no metric rows")
    return metric_header, rows


def skip_warmup(rows: list[dict[str, str]], n: int = 1) -> list[dict[str, str]]:
    """Skip the first ``n`` instances of each kernel."""
    seen_counts: dict[str, int] = {}
    result: list[dict[str, str]] = []
    for row in rows:
        k = row["Kernel"]
        seen_counts[k] = seen_counts.get(k, 0) + 1
        if seen_counts[k] > n:
            result.append(row)
    return result


def _is_preempted(row: dict[str, str]) -> bool:
    try:
        return float(row.get("MidQueryCtxSwitch") or 0) > 0
    except ValueError:
        return False


def compute_median_summary(
    rows: list[dict[str, str]],
    header_fields: list[str],
) -> dict[str, Any]:
    """Compute the median of each numeric column across ``rows``."""
    if not rows:
        return {}
    summary: dict[str, Any] = {"Kernel": rows[0]["Kernel"], "count": len(rows)}
    for col in header_fields[1:]:
        col_s = col.strip()
        if col_s == "GlobalInstanceId":
            continue
        values: list[float] = []
        for r in rows:
            with contextlib.suppress(ValueError, TypeError):
                values.append(float(r.get(col_s, "")))
        if values:
            summary[col_s] = statistics.median(values)
    return summary


def parse_counters_for_kernels(
    log_path: Path,
    kernels: list[dict[str, Any]],
    warmup: int,
    preempt_out: dict[str, dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Parse a ``unitrace -q`` log and return per-kernel median counter dicts."""
    log_text = log_path.read_text(errors="replace")
    _, rows = parse_unitrace_metrics(log_text)
    out: dict[str, dict[str, Any]] = {}
    for k in kernels:
        matched = [r for r in rows if r["Kernel"] == k["name"]]
        if not matched:
            matched = [
                r
                for r in rows
                if r["Kernel"].split("[")[0] == k["base"] or k["base"] in r["Kernel"]
            ]
        rk = skip_warmup(
            matched,
            int(k.get("prefill", 0)) + warmup * int(k.get("per_call", 1)),
        )
        clean = [r for r in rk if not _is_preempted(r)]
        if preempt_out is not None and rk and len(clean) < len(rk):
            preempt_out[k["name"]] = {
                "dropped": len(rk) - len(clean),
                "instances": len(rk),
                "all_preempted": not clean,
            }
        rk = clean or rk
        if rk:
            hdr = sorted(
                {c for r in rk for c in r if c not in ("Kernel", "GlobalInstanceId")}
            )
            out[k["name"]] = compute_median_summary(rk, ["Kernel", *hdr])
    return out


# ---------------------------------------------------------------------------
# Occupancy, Memory, and Pipe Step Derivations
# ---------------------------------------------------------------------------


def compute_occupancy(
    kernel_info: dict[str, Any],
    topology: dict[str, Any],
) -> dict[str, Any] | None:
    """Compute Intel XPU Xe-core and device occupancy from timeline launch properties."""
    local = kernel_info.get("local")
    groups = kernel_info.get("groups")
    simd = kernel_info.get("simd")
    if not local or not groups or not simd:
        return None

    wg_size = math.prod(local)
    num_wgs = math.prod(groups)
    grf = int(kernel_info.get("grf", 128))
    slm_bytes = int(kernel_info.get("slm_bytes", 0))
    large_grf = grf > 128

    xe_cores = int(topology["xe_cores"])
    threads_per_core = int(topology["threads_per_core"])
    max_wgs_per_core = int(topology["max_wgs_per_core"])
    slm_per_core_kb = int(topology["slm_per_core_kb"])

    max_threads = threads_per_core // 2 if large_grf else threads_per_core
    threads_per_wg = math.ceil(wg_size / simd)
    slm_kb = (
        next((s for s in SLM_SIZES_KB if s * 1024 >= slm_bytes), slm_per_core_kb)
        if slm_bytes > 0
        else 0
    )

    lim_threads = max_threads // threads_per_wg if threads_per_wg <= max_threads else 0
    lim_smem = slm_per_core_kb // slm_kb if slm_kb > 0 else max_wgs_per_core
    per_core_launched = math.ceil(num_wgs / xe_cores)
    resident_per_core = min(lim_threads, lim_smem, max_wgs_per_core)
    active_wgs_per_core = min(resident_per_core, per_core_launched)

    core_occ = round(threads_per_wg * active_wgs_per_core / threads_per_core, 4)
    active_xe_cores = min(xe_cores, num_wgs)
    device_occ = round(
        min(1.0, (num_wgs * threads_per_wg) / (xe_cores * threads_per_core)), 4
    )
    waves = (
        round(num_wgs / (resident_per_core * xe_cores), 2)
        if resident_per_core > 0
        else None
    )

    if device_occ >= 0.5:
        classification = "Good"
    elif device_occ >= 0.25:
        classification = "Moderate"
    else:
        classification = "Low"

    limiting_factors: list[str] = []
    if num_wgs < xe_cores:
        limiting_factors.append(
            f"under-populated grid ({num_wgs} WGs < {xe_cores} Xe-cores)"
        )
    if large_grf:
        limiting_factors.append(
            "256-GRF mode halves max hardware threads per Xe-core (32 vs 64)"
        )
    if slm_kb > 0 and lim_smem < lim_threads:
        limiting_factors.append(
            f"SLM allocation ({slm_kb} KB/WG limits to {lim_smem} WGs/Xe-core)"
        )
    if not limiting_factors:
        limiting_factors.append(
            f"thread slots ({threads_per_wg} threads/WG, {resident_per_core} resident WGs/Xe-core)"
        )

    return {
        "launch_params": {
            "workgroup_size": wg_size,
            "local_size": local,
            "group_count": groups,
            "subgroup_size": simd,
            "registers_per_thread": grf,
            "smem_per_workgroup_bytes": slm_bytes,
            "smem_allocated_kb": slm_kb,
            "num_workgroups": num_wgs,
            "large_grf": large_grf,
            "grf_note": "Register File Size Per Thread = GRF mode (128 or 256), not exact live registers",
        },
        "resource_limits": {
            "xe_cores": xe_cores,
            "max_threads_per_core": max_threads,
            "threads_per_wg": threads_per_wg,
            "limit_threads_wgs_per_core": lim_threads,
            "limit_smem_wgs_per_core": lim_smem if slm_kb > 0 else None,
            "max_wgs_per_core": max_wgs_per_core,
            "workgroups_per_core_launched": per_core_launched,
            "resident_wgs_per_core": resident_per_core,
        },
        "occupancy": core_occ,
        "active_xe_cores": active_xe_cores,
        "device_occupancy": device_occ,
        "waves": waves,
        "classification": classification,
        "limiting_factors": limiting_factors,
    }


def _counter_val(d: dict[str, Any], *names: str) -> float | None:
    for n in names:
        if n in d and d[n] is not None:
            try:
                return float(d[n])
            except (ValueError, TypeError):
                pass
    return None


def derive_memory_and_pipe_metrics(
    counters_by_group: dict[str, dict[str, Any]],
    *,
    static_roofline: dict[str, Any],
    topology: dict[str, Any],
    runner_time_us: float | None = None,
    stall_sampling_pct: dict[str, float] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Derive Step 9 (Memory/Cache) and Step 10 (Pipes/Stalls) metrics from parsed counters."""
    cb = counters_by_group.get("ComputeBasic", {})
    vep = counters_by_group.get("VectorEngineProfile", {})
    ves = counters_by_group.get("VectorEngineStalls", {})
    l1r = counters_by_group.get("L1ProfileReads", {})
    l1w = counters_by_group.get("L1ProfileWrites", {})
    l1slm = counters_by_group.get("L1ProfileSlmBankConflicts", {})

    t_ns = _counter_val(cb, "GpuTime[ns]")
    rd = _counter_val(cb, "GPU_MEMORY_BYTE_READ[bytes]")
    wr = _counter_val(cb, "GPU_MEMORY_BYTE_WRITE[bytes]")

    l3_rd_bytes = _counter_val(cb, "L3_BYTE_READ[bytes]")
    l3_wr_bytes = _counter_val(cb, "L3_BYTE_WRITE[bytes]")
    l3_rd_lines = _counter_val(cb, "L3_READ[events]")
    l3_wr_lines = _counter_val(cb, "L3_WRITE[events]")
    if l3_rd_bytes is None and l3_rd_lines is not None:
        l3_rd_bytes = l3_rd_lines * 64.0
    if l3_wr_bytes is None and l3_wr_lines is not None:
        l3_wr_bytes = l3_wr_lines * 64.0
    if l3_wr_lines is None and l3_wr_bytes is not None:
        l3_wr_lines = round(l3_wr_bytes / 64.0, 1)

    lsc_rd_bytes = _counter_val(
        cb, "LOAD_STORE_CACHE_BYTE_READ[bytes]"
    ) or _counter_val(l1r, "LOAD_STORE_CACHE_BYTE_READ[bytes]")
    lsc_wr_bytes = _counter_val(
        cb, "LOAD_STORE_CACHE_BYTE_WRITE[bytes]"
    ) or _counter_val(l1w, "LOAD_STORE_CACHE_BYTE_WRITE[bytes]")
    slm_rd_bytes = _counter_val(cb, "SLM_BYTE_READ[bytes]") or _counter_val(
        l1r, "SLM_BYTE_READ[bytes]"
    )
    slm_wr_bytes = _counter_val(cb, "SLM_BYTE_WRITE[bytes]") or _counter_val(
        l1w, "SLM_BYTE_WRITE[bytes]"
    )
    slm_bank_conflicts = _counter_val(
        cb, "SLM_BANK_CONFLICT_COUNT[events]"
    ) or _counter_val(l1slm, "SLM_BANK_CONFLICT_COUNT[events]")

    algo_info = (
        static_roofline.get("workload") or static_roofline.get("algorithmic") or {}
    )
    cfg_info = static_roofline.get("config_traffic") or {}
    min_read = float(
        algo_info.get("algorithmic_read_bytes") or algo_info.get("read_bytes") or 0
    )
    min_write = float(
        algo_info.get("algorithmic_write_bytes") or algo_info.get("write_bytes") or 0
    )
    cfg_read = float(
        cfg_info.get("config_read_bytes") or cfg_info.get("read_bytes") or 0
    )
    cfg_write = float(
        cfg_info.get("config_write_bytes") or cfg_info.get("write_bytes") or 0
    )
    peak_bw = float(topology["peak_memory_bw_gbps"])

    clock_mhz = _counter_val(cb, "AvgGpuCoreFrequencyMHz[MHz]", "CoreFrequencyMHz[MHz]")
    clock_ghz = (
        clock_mhz / 1000.0
        if clock_mhz and clock_mhz > 100
        else float(topology["clock_ghz"])
    )
    latency_us = 2000.0 / (clock_ghz * 1e3)
    t_mem_us = latency_us + (min_read + min_write) / (peak_bw * 1e3)
    t_mem_ms = t_mem_us / 1e3

    t_s = (t_ns * 1e-9) if t_ns and t_ns > 0 else None
    runner_s = (
        (runner_time_us * 1e-6) if runner_time_us and runner_time_us > 0 else None
    )
    eff_s = runner_s or t_s

    step9_memory: dict[str, Any] = {
        "measured_read_bytes": rd,
        "measured_write_bytes": wr,
        "measured_total_bytes": (rd + wr)
        if rd is not None and wr is not None
        else None,
        "projected_read_bytes": min_read,
        "projected_write_bytes": min_write,
        "config_predicted_read_bytes": cfg_read,
        "config_predicted_write_bytes": cfg_write,
        "read_amplification": (
            round(rd / min_read, 4) if rd is not None and min_read > 0 else None
        ),
        "write_amplification": (
            round(wr / min_write, 4) if wr is not None and min_write > 0 else None
        ),
        "total_amplification_vs_algo": (
            round((rd + wr) / (min_read + min_write), 4)
            if rd is not None and wr is not None and (min_read + min_write) > 0
            else None
        ),
        "measured_over_config_predicted": (
            round((rd + wr) / (cfg_read + cfg_write), 4)
            if rd is not None and wr is not None and (cfg_read + cfg_write) > 0
            else None
        ),
        "l3_read_bytes": l3_rd_bytes,
        "l3_write_bytes": l3_wr_bytes,
        "l3_write_lines": l3_wr_lines,
        "l3_write_over_min": (
            round(l3_wr_lines / (min_write / 64.0), 4)
            if l3_wr_lines is not None and min_write > 0
            else None
        ),
        "lsc_read_bytes": lsc_rd_bytes,
        "lsc_write_bytes": lsc_wr_bytes,
        "slm_read_bytes": slm_rd_bytes,
        "slm_write_bytes": slm_wr_bytes,
        "slm_bank_conflicts": slm_bank_conflicts,
        "dram_read_bw_gbps": (
            round(rd / eff_s / 1e9, 2) if rd is not None and eff_s else None
        ),
        "dram_write_bw_gbps": (
            round(wr / eff_s / 1e9, 2) if wr is not None and eff_s else None
        ),
        "dram_total_bw_gbps": (
            round((rd + wr) / eff_s / 1e9, 2)
            if rd is not None and wr is not None and eff_s
            else None
        ),
        "l3_total_bw_gbps": (
            round((l3_rd_bytes + l3_wr_bytes) / eff_s / 1e9, 2)
            if l3_rd_bytes is not None and l3_wr_bytes is not None and eff_s
            else None
        ),
        "peak_bw_gbps": peak_bw,
        "bw_utilization": (
            round((rd + wr) / eff_s / 1e9 / peak_bw, 4)
            if rd is not None and wr is not None and eff_s and peak_bw > 0
            else None
        ),
        "T_mem_ms": round(t_mem_ms, 6),
        "T_counter_gpu_ms": round(t_ns / 1e6, 6) if t_ns is not None else None,
        "T_runner_ms": round(runner_time_us / 1e3, 6)
        if runner_time_us is not None
        else None,
        "T_actual_over_T_mem": (
            round((eff_s * 1e3) / t_mem_ms, 4)
            if eff_s is not None and t_mem_ms > 0
            else None
        ),
        "xve_active_pct": _counter_val(cb, "XVE_ACTIVE[%]"),
        "xve_stall_pct": _counter_val(cb, "XVE_STALL[%]"),
        "occupancy_pct": _counter_val(cb, "XVE_THREADS_OCCUPANCY_ALL[%]"),
        "atomic_messages": _counter_val(cb, "XVE_ATOMIC_ACCESS_COUNT[messages]"),
        "barrier_messages": _counter_val(cb, "XVE_BARRIER_MESSAGE_COUNT[messages]"),
    }

    src = vep or cb
    slots = {
        "ALU0": _counter_val(src, "XVE_INST_EXECUTED_ALU0_ALL[events]"),
        "ALU1": _counter_val(src, "XVE_INST_EXECUTED_ALU1_ALL[events]"),
        "MATH": _counter_val(src, "XVE_INST_EXECUTED_MATH[events]"),
        "MATRIX": _counter_val(
            src,
            "XVE_INST_EXECUTED_ALU2_ALL[events]",
            "XVE_INST_EXECUTED_XMX_ALL[events]",
        ),
        "SEND": _counter_val(src, "XVE_INST_EXECUTED_SEND_ALL[events]"),
        "CONTROL": _counter_val(src, "XVE_INST_EXECUTED_CONTROL_ALL[events]"),
    }
    pipe_util_pct = {
        "ALU0": _counter_val(cb, "XVE_INST_EXECUTED_ALU0_ALL_UTILIZATION[%]"),
        "ALU1": _counter_val(cb, "XVE_INST_EXECUTED_ALU1_ALL_UTILIZATION[%]"),
        "MATRIX": _counter_val(
            cb,
            "XVE_INST_EXECUTED_XMX_ALL_UTILIZATION[%]",
            "XVE_INST_EXECUTED_ALU2_ALL_UTILIZATION[%]",
        ),
        "SEND": _counter_val(cb, "XVE_INST_EXECUTED_SEND_ALL_UTILIZATION[%]"),
        "CONTROL": _counter_val(cb, "XVE_INST_EXECUTED_CONTROL_ALL_UTILIZATION[%]"),
    }

    num_xve = int(topology["num_xve"])
    rate = num_xve * clock_ghz * 1e9
    clk_per_slot = {"MATRIX": 2}

    per_pipe_xve: dict[str, dict[str, Any]] = {}
    for k_pipe, v_slots in slots.items():
        if v_slots is not None:
            per_pipe_xve[k_pipe] = {
                "slots": v_slots,
                "T_ms": round(v_slots * clk_per_slot.get(k_pipe, 1) / rate * 1e3, 6),
                "utilization_pct": pipe_util_pct.get(k_pipe),
            }

    if "ALU1" in per_pipe_xve and slots["MATH"] is not None:
        v_comb = (slots["ALU1"] or 0.0) + (slots["MATH"] or 0.0)
        per_pipe_xve["ALU1+MATH"] = {
            "slots": v_comb,
            "T_ms": round(v_comb / rate * 1e3, 6),
            "utilization_pct": pipe_util_pct.get("ALU1"),
        }

    candidates = [
        k
        for k in per_pipe_xve
        if k not in (("ALU1", "MATH") if "ALU1+MATH" in per_pipe_xve else ())
    ]
    if candidates:
        dom_pipe = max(candidates, key=lambda k: per_pipe_xve[k]["T_ms"])
        t_instr_ms = per_pipe_xve[dom_pipe]["T_ms"]
    else:
        dom_pipe, t_instr_ms = None, None

    generic_map = {
        "FP": "ALU0",
        "INT": "ALU1+MATH" if "ALU1+MATH" in per_pipe_xve else "ALU1",
        "MATH": "MATH",
        "MATRIX": "MATRIX",
        "MEMORY": "SEND",
        "CONTROL": "CONTROL",
    }
    per_pipe_generic = {
        g_name: per_pipe_xve[x_name]
        for g_name, x_name in generic_map.items()
        if x_name in per_pipe_xve
    }

    ve_stalls_pct = {
        k.split("[")[0]: float(v)
        for k, v in ves.items()
        if k.startswith("XVE_STALL_") and k.endswith("[%]") and v is not None
    }

    step10_instructions: dict[str, Any] = {
        "device_xve_count": num_xve,
        "device_freq_ghz": round(clock_ghz, 3),
        "peak_slots_per_s": round(rate, 1),
        "per_pipe": per_pipe_generic,
        "per_pipe_xve": per_pipe_xve,
        "dominant_pipe": dom_pipe,
        "T_instruction_ms": t_instr_ms,
        "T_mem_ms": round(t_mem_ms, 6),
        "primary_bound": (
            None
            if t_instr_ms is None
            else ("instruction" if t_instr_ms > 0.8 * t_mem_ms else "memory")
        ),
        "xve_active_pct": step9_memory["xve_active_pct"],
        "xve_stall_pct": step9_memory["xve_stall_pct"],
        "alu0_and_alu1_overlap_pct": _counter_val(
            cb, "XVE_PIPE_ALU0_AND_ALU1_ACTIVE[%]"
        ),
        "alu0_and_xmx_overlap_pct": _counter_val(cb, "XVE_PIPE_ALU0_AND_XMX_ACTIVE[%]"),
        "stall_breakdown_counter_pct": ve_stalls_pct or None,
        "stall_breakdown_sampling_pct": stall_sampling_pct,
    }
    return step9_memory, step10_instructions


# ---------------------------------------------------------------------------
# Disassembly Census & Stall-Sampling Source Mapper
# ---------------------------------------------------------------------------

ASM_PATTERNS: dict[str, str] = {
    "load_ugm": r"; load\.ugm\.",
    "store_ugm": r"; store\.ugm\.",
    "load_slm": r"; load\.slm\.",
    "store_slm": r"; store\.slm\.",
    "load_block2d": r"; load_block2d\.",
    "store_block2d": r"; store_block2d\.",
    "atomic_sends": r"; atomic_\w+\.",
    "fence": r"; fence\.",
    "payload_loads": r"; load\.ugm\.\S*bti\[255\]",
    "prefetch": r"\bsend\S* \([^)]*\)\s+null\b.*; load",
    "prefetch_block2d": r"\bsend\S* \([^)]*\)\s+null\b.*; load_block2d\.",
    "scratch_sends": r"\bscratch\b|ss\[a0",
    "d16u32_loads": r"; load\.\S*\.d16u32",
    "d16u32_stores": r"; store\.\S*\.d16u32",
    "barrier_signal": r"signal barrier|\bbarrier\b(?!.*sync\.bar)",
    "barrier_wait": r"\bsync\.bar\b",
    "math": r"\bmath\.",
    "math_inv": r"\bmath\.inv",
    "dpas": r"\bdpas\b",
    "qword_ops": r":u?q\b",
    "fp64_mov": r"\bmov \(.*:df\b",
    "mov_byte_dst": r"\bmov \(\S+\)\s+\S+:u?b\b",
    "indirect_reg": r"r\[a0",
    "jmpi": r"\bjmpi\b",
    "goto_join": r"\b(goto|join)\b",
    "sync_nop": r"\bsync\.nop\b",
}

ASM_TOTALS: dict[str, tuple[str, ...]] = {
    "load_sends": ("load_ugm", "load_slm", "load_block2d"),
    "store_sends": ("store_ugm", "store_slm", "store_block2d"),
    "d16u32_sends": ("d16u32_loads", "d16u32_stores"),
    "fp64_ops": ("fp64_arith", "fp64_mov"),
    "barrier": ("barrier_signal", "barrier_wait"),
}

_INSTR_RE = re.compile(
    r"^\s*(?:/\*\s*#\d+\s*\[([0-9a-fA-F]+)\]\s*\*/)?\s*(\(.*?\)\s*)?([a-z][\w.]*)"
)


def parse_asm_lines(text: str) -> list[tuple[int, int, str, str]]:
    """Parse GEN ISA disassembly lines into ``(line_no, byte_offset, opcode, text)``."""
    out: list[tuple[int, int, str, str]] = []
    off = 0
    for i, line in enumerate(text.splitlines(), 1):
        s = line.strip()
        m_lbl = re.match(r"^L(\d+):", s)
        if m_lbl:
            off = int(m_lbl.group(1))
            continue
        if not s or s.startswith("//"):
            continue
        m = _INSTR_RE.match(line)
        if not m:
            continue
        if m.group(1):
            off = int(m.group(1), 16)
        out.append((i, off, m.group(3), line))
        off += 8 if "Compacted" in line else 16
    return out


def compute_asm_census(
    lines: list[tuple[int, int, str, str]],
    *,
    opcodes_top: int = 15,
) -> dict[str, Any]:
    """Compute instruction category counts and top opcodes over ``parse_asm_lines`` output."""
    c: dict[str, Any] = {
        name: sum(1 for *_, line in lines if re.search(pat, line))
        for name, pat in ASM_PATTERNS.items()
    }
    c["fp64_arith"] = sum(
        1
        for _, _, op, line in lines
        if ":df" in line.split("//")[0] and op.split(".")[0] != "mov"
    )
    for total_name, parts in ASM_TOTALS.items():
        c[total_name] = sum(int(c[p]) for p in parts)
    c["instructions"] = len(lines)
    ops: dict[str, int] = {}
    for _, _, op, _ in lines:
        root_op = op.split(".")[0]
        ops[root_op] = ops.get(root_op, 0) + 1
    c["opcodes"] = dict(sorted(ops.items(), key=lambda x: -x[1])[:opcodes_top])
    return c


def extract_send_kinds(text: str) -> list[str]:
    """Return sorted unique LSC send message descriptors from disassembly comments."""
    return sorted(
        {
            m.group(1)
            for m in re.finditer(
                r"; ((?:load|store|load_block2d|store_block2d|atomic_\w+)\.\S+)", text
            )
        }
    )


def build_triton_line_map(triton_code: str | None) -> dict[int, dict[str, str]]:
    """Map 1-based line numbers in generated Triton code to ``(triton_line, helion_src)``."""
    if not triton_code:
        return {}
    mapping: dict[int, dict[str, str]] = {}
    pending_srcs: list[str] = []
    current_helion_src = ""
    for idx, line in enumerate(triton_code.splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith("# src["):
            comment_body = stripped.removeprefix("# ")
            if not comment_body.endswith(": ...") or not pending_srcs:
                pending_srcs.append(comment_body)
        else:
            if pending_srcs:
                if len(pending_srcs) == 1:
                    current_helion_src = pending_srcs[0]
                else:
                    # Combine multi-line # src[file:l1]: ... # src[file:l2]: ...
                    first = pending_srcs[0]
                    parts = [
                        p.split(": ", 1)[1].strip() for p in pending_srcs if ": " in p
                    ]
                    prefix = first.split(": ", 1)[0]
                    current_helion_src = f"{prefix}: {' '.join(parts)}"
                pending_srcs = []
        mapping[idx] = {
            "triton_line": stripped,
            "helion_src": current_helion_src,
        }
    return mapping


def _find_stall_asm_files(stall_dir: Path | None) -> dict[str, Path]:
    files: dict[str, Path] = {}
    if not stall_dir or not stall_dir.is_dir():
        return files
    for f in sorted(stall_dir.glob("asm.*/*.asm")):
        head = f.read_text(errors="replace")[:2000]
        m = re.search(r"^// Kernel: (.+)$", head, re.MULTILINE)
        if m:
            files.setdefault(m.group(1).strip(), f)
    return files


def collect_asm_census(
    asm_dir: Path | None,
    stall_dir: Path | None,
    kernels: list[dict[str, Any]],
    out_dir: Path,
) -> dict[str, dict[str, Any]]:
    """Locate disassembly for each kernel from ``ocloc disasm`` or ``--stall-sampling`` dumps."""
    files = sorted(asm_dir.rglob(".text.*.asm")) if asm_dir and asm_dir.is_dir() else []
    stall_asm = _find_stall_asm_files(stall_dir)
    out: dict[str, dict[str, Any]] = {}
    kernels_dir = out_dir / "kernels"
    kernels_dir.mkdir(exist_ok=True)

    def _record(k: dict[str, Any], f: Path, note: str | None = None) -> None:
        text = f.read_text(errors="replace")
        idx = kernels.index(k)
        vis = kernels_dir / f"k{idx}.asm"
        vis.write_text(text)
        lines = parse_asm_lines(text)
        entry: dict[str, Any] = {
            "asm_file": str(vis),
            "asm_file_original": str(f),
            "census": compute_asm_census(lines),
            "send_kinds": extract_send_kinds(text),
        }
        if note:
            entry["note"] = note
        out[k["name"]] = entry

    for f in files:
        mangled = f.name[len(".text.") : -len(".asm")]
        for k in kernels:
            if (k["base"] == mangled or k["base"] in mangled) and k["name"] not in out:
                _record(k, f)

    for k in kernels:
        if k["name"] not in out and k["base"] in stall_asm:
            _record(
                k,
                stall_asm[k["base"]],
                note="from stall-sampling dump",
            )
    return out


def parse_stall_sampling(
    stall_dir: Path | None,
    kernels: list[dict[str, Any]],
    out_dir: Path,
    *,
    triton_code: str | None = None,
    cover: float = 0.8,
    cap: int = 40,
) -> dict[str, dict[str, Any]]:
    """Parse ``unitrace --stall-sampling`` output and map hot IPs to Triton/Helion lines."""
    if not stall_dir:
        return {}
    files = sorted(stall_dir.glob("*.metrics.*.csv")) if stall_dir.is_dir() else []
    log = out_dir / "stall.log"
    if not files and log.exists():
        files = [log]
    if not files:
        return {}

    blocks: dict[str, list[dict[str, str]]] = {}
    for f in files:
        name: str | None = None
        hdr: list[str] | None = None
        for line in f.read_text(errors="replace").splitlines():
            if line.startswith("Kernel,Pid"):
                name, hdr = "", None
                continue
            if name == "" and line.startswith('"'):
                name = next(csv.reader(io.StringIO(line)))[0]
                continue
            if line.startswith("Source,Offset"):
                hdr = [c.strip() for c in next(csv.reader(io.StringIO(line)))]
                continue
            if hdr and name and line.startswith('"'):
                row = dict(zip(hdr, next(csv.reader(io.StringIO(line))), strict=False))
                blocks.setdefault(name, []).append(row)

    stall_asm = _find_stall_asm_files(stall_dir)
    triton_map = build_triton_line_map(triton_code)
    kernels_dir = out_dir / "kernels"
    kernels_dir.mkdir(exist_ok=True)
    out: dict[str, dict[str, Any]] = {}

    for idx, k in enumerate(kernels):
        rows = blocks.get(k["base"]) or blocks.get(k["name"])
        if not rows:
            continue
        cols = [c for c in rows[0] if c.endswith("[Events]")]
        stall_cols = [c for c in cols if c != "Active[Events]"]
        tot = {c.split("[")[0]: sum(int(r.get(c) or 0) for r in rows) for c in cols}
        all_ev = sum(tot.values()) or 1
        all_stall_ev = sum(v for k_c, v in tot.items() if k_c != "Active") or 1

        def _ev(
            row_d: dict[str, str],
            _cols: list[str] = cols,
        ) -> int:
            return sum(int(row_d.get(col_k) or 0) for col_k in _cols)

        def _top_stall(
            row_d: dict[str, str],
            _stall_cols: list[str] = stall_cols,
        ) -> str | None:
            best_c = ""
            best_v = 0
            for col_k in _stall_cols:
                val = int(row_d.get(col_k) or 0)
                if val > best_v:
                    best_v = val
                    best_c = col_k
            return best_c.split("[")[0] if best_v > 0 else None

        by_off = {int(r["Offset"], 16): r for r in rows if "Offset" in r}
        src_asm = stall_asm.get(k["base"])
        if src_asm is None and (kernels_dir / f"k{idx}.asm").exists():
            src_asm = kernels_dir / f"k{idx}.asm"

        line_of: dict[int, int] = {}
        asm_out: Path | None = None
        if src_asm is not None:
            raw_lines = src_asm.read_text(errors="replace").splitlines()
            parsed_ins = parse_asm_lines("\n".join(raw_lines))
            pos = {ln: (off, op, text) for ln, off, op, text in parsed_ins}
            ann: list[str] = []
            for ln_no, raw_l in enumerate(raw_lines, 1):
                if ln_no in pos:
                    off, _op, _ = pos[ln_no]
                    r = by_off.get(off)
                    if r is not None:
                        line_of[off] = ln_no
                        e = _ev(r)
                        nz = {
                            c.split("[")[0]: int(r[c])
                            for c in stall_cols
                            if int(r.get(c) or 0) > 0
                        }
                        tag_s = f"{e:7d} {100 * e / all_ev:5.1f}% " + ",".join(
                            f"{c}={v}" for c, v in nz.items()
                        )
                        ann.append(f"/* 0x{off:06x} {tag_s:<60s} */ {raw_l.strip()}")
                    else:
                        ann.append(f"/* 0x{off:06x} {'':<60s} */ {raw_l.strip()}")
                else:
                    ann.append(raw_l)
            asm_out = kernels_dir / f"k{idx}.stall.asm"
            asm_out.write_text("\n".join(ann) + "\n")

        hot: list[dict[str, Any]] = []
        source_hotspots: dict[str, dict[str, Any]] = {}
        acc = 0
        for r in sorted(rows, key=_ev, reverse=True):
            e = _ev(r)
            if e == 0:
                break
            src_loc = r.get("Source", "")
            triton_line_txt = ""
            helion_src_txt = ""
            m_src = re.search(r"([^/\\]+\.py):(\d+)(?::\d+)?$", src_loc)
            if m_src:
                fname, lineno_s = m_src.group(1), m_src.group(2)
                t_lineno = int(lineno_s)
                if fname in ("helpers.py", "standard.py", "libdevice.py"):
                    helion_src_txt = f"{fname}:{t_lineno}"
                    src_file_path = Path(src_loc.split(":")[0])
                    if src_file_path.is_file():
                        f_lines = src_file_path.read_text(errors="replace").splitlines()
                        if 1 <= t_lineno <= len(f_lines):
                            triton_line_txt = f_lines[t_lineno - 1].strip()
                            helion_src_txt = f"{fname}:{t_lineno}: {triton_line_txt}"
                elif triton_map:
                    info = triton_map.get(t_lineno)
                    if info:
                        triton_line_txt = info["triton_line"]
                        helion_src_txt = info["helion_src"]

            if helion_src_txt or triton_line_txt:
                if (
                    helion_src_txt
                    and triton_line_txt
                    and not helion_src_txt.startswith(("helpers.py", "standard.py"))
                ):
                    key = f"{helion_src_txt}  [triton: {triton_line_txt}]"
                else:
                    key = helion_src_txt or f"{src_loc}: {triton_line_txt}"
                agg = source_hotspots.setdefault(
                    key,
                    {
                        "source": key,
                        "triton_line": triton_line_txt,
                        "events": 0,
                        "stalls": {},
                    },
                )
                agg["events"] += e
                for c in stall_cols:
                    v_c = int(r.get(c) or 0)
                    if v_c > 0:
                        cname = c.split("[")[0]
                        agg["stalls"][cname] = agg["stalls"].get(cname, 0) + v_c

            if acc < cover * all_ev and len(hot) < cap:
                acc += e
                off_val = int(r["Offset"], 16)
                hot.append(
                    {
                        "offset": r["Offset"],
                        "asm_line": line_of.get(off_val),
                        "instruction": r.get("Instruction", "")[:120],
                        "dependency": r.get("Dependency", "")[:120],
                        "source": src_loc,
                        "triton_line": triton_line_txt or None,
                        "helion_src": helion_src_txt or None,
                        "events": e,
                        "pct": round(100.0 * e / all_ev, 2),
                        "top_stall": _top_stall(r),
                    }
                )

        top_sources = sorted(source_hotspots.values(), key=lambda x: -int(x["events"]))[
            :15
        ]
        for s_item in top_sources:
            s_item["pct"] = round(100.0 * int(s_item["events"]) / all_ev, 2)

        out[k["name"]] = {
            "samples": all_ev,
            "totals": tot,
            "stall_pct_of_all_samples": {
                c: round(100.0 * v / all_ev, 2) for c, v in tot.items() if c != "Active"
            },
            "stall_pct_of_stalls": {
                c: round(100.0 * v / all_stall_ev, 2)
                for c, v in tot.items()
                if c != "Active"
            },
            "annotated_asm": str(asm_out) if asm_out else None,
            "hottest_cover_pct": round(100.0 * acc / all_ev, 1),
            "hottest_instructions": hot,
            "hot_source_locations": top_sources,
        }
    return out


# ---------------------------------------------------------------------------
# Report Generation
# ---------------------------------------------------------------------------


def build_markdown_report(
    summary: dict[str, Any],
    s05_roofline: dict[str, Any],
    s07_occupancy: dict[str, Any],
    s08_reg: dict[str, Any],
    s09_mem: dict[str, Any],
    s10_inst: dict[str, Any],
    s11_asm: dict[str, Any],
) -> str:
    """Format a comprehensive Markdown report comparing Static Roofline vs Hardware Counters."""
    lines: list[str] = []
    kname = summary["kernel"]
    shape_str = summary["shape"]
    dev_name = summary.get("device", "Intel XPU")
    lines.extend(
        [
            f"# Intel XPU Hardware Profile Report: `{kname}` ({shape_str})",
            "",
            f"- **Device**: {dev_name} (`{summary.get('compute_capability')}`)",
            f"- **Dominant Kernel**: `{summary.get('dominant_kernel_name')}`",
        ]
    )
    if summary.get("helion_us") is not None:
        base_str = ", ".join(
            f"{b}={t:.2f} us" for b, t in (summary.get("baselines_us") or {}).items()
        )
        lines.append(
            f"- **Un-instrumented Runner Time**: `{summary['helion_us']:.2f} us` "
            f"(Speedup: `{summary.get('speedup', 0.0):.2f}x` vs `{summary.get('best_baseline')}`; {base_str})"
        )
    if s09_mem.get("T_counter_gpu_ms") is not None:
        lines.append(
            f"- **Counter-mode Median GPU Time**: `{s09_mem['T_counter_gpu_ms'] * 1000.0:.2f} us`"
        )

    # 1. Static Roofline vs Hardware Measurement
    algo = s05_roofline.get("workload") or s05_roofline.get("algorithmic") or {}
    cfg_t = s05_roofline.get("config_traffic") or {}
    emp = s05_roofline.get("empirical") or {}
    algo_rd = int(algo.get("algorithmic_read_bytes") or algo.get("read_bytes") or 0)
    algo_wr = int(algo.get("algorithmic_write_bytes") or algo.get("write_bytes") or 0)
    cfg_rd = int(cfg_t.get("config_read_bytes") or cfg_t.get("read_bytes") or 0)
    cfg_wr = int(cfg_t.get("config_write_bytes") or cfg_t.get("write_bytes") or 0)
    lines.extend(
        [
            "",
            "## 1. Static Roofline vs. Hardware Counter Comparison",
            "",
            "| Metric | Static Algorithmic | Static Config (Tiled) | Hardware Measured (`unitrace`) |",
            "| :--- | ---: | ---: | ---: |",
            (
                f"| Read Bytes | {algo_rd:,} B | {cfg_rd:,} B | "
                f"DRAM: {int(s09_mem.get('measured_read_bytes') or 0):,} B / L3: {int(s09_mem.get('l3_read_bytes') or 0):,} B |"
            ),
            (
                f"| Write Bytes | {algo_wr:,} B | {cfg_wr:,} B | "
                f"DRAM: {int(s09_mem.get('measured_write_bytes') or 0):,} B / L3: {int(s09_mem.get('l3_write_bytes') or 0):,} B |"
            ),
            (
                f"| Traffic Amplification | 1.00x | {cfg_t.get('traffic_amplification', 0.0):.2f}x | "
                f"DRAM vs Algo: {s09_mem.get('total_amplification_vs_algo') or 0.0:.2f}x "
                f"(DRAM/Config: {s09_mem.get('measured_over_config_predicted') or 0.0:.2f}x) |"
            ),
            (
                f"| Effective Bandwidth | {emp.get('achieved_algo_gbps') or 0.0:.1f} GB/s | "
                f"{emp.get('achieved_config_gbps') or 0.0:.1f} GB/s | "
                f"DRAM: {s09_mem.get('dram_total_bw_gbps') or 0.0:.1f} GB/s / L3: {s09_mem.get('l3_total_bw_gbps') or 0.0:.1f} GB/s |"
            ),
            (
                f"| SOL% / Bound Diagnosis | SOL: {emp.get('sol_pct') or 0.0:.1f}% | "
                f"CfgSOL: {emp.get('config_sol_pct') or 0.0:.1f}% (`{s05_roofline.get('bottleneck')}`) | "
                f"Primary: `{s10_inst.get('primary_bound')}` (Dom Pipe: `{s10_inst.get('dominant_pipe')}`) |"
            ),
            "",
            "## 2. Occupancy, Launch Config & Register Pressure",
            "",
        ]
    )

    # 2. Occupancy & Register File
    lp = s07_occupancy.get("launch_params") or {}
    rl = s07_occupancy.get("resource_limits") or {}
    spill = s08_reg.get("spill_measured") or {}
    lines.extend(
        [
            (
                f"- **Launch Grid**: Groups `{lp.get('group_count')}` ({lp.get('num_workgroups')} WGs), "
                f"Local `{lp.get('local_size')}` ({lp.get('workgroup_size')} work-items/WG), "
                f"`SIMD{lp.get('subgroup_size')}` ({rl.get('threads_per_wg')} HW threads/WG)"
            ),
            (
                f"- **GRF & Spill**: `{lp.get('registers_per_thread')} GRF/thread` "
                f"(Large GRF: `{lp.get('large_grf')}`), "
                f"Spill: `{spill.get('spill_bytes_per_thread', 0)} B/thread`, "
                f"Scratch Sends: `{spill.get('scratch_sends', 0)}`, "
                f"SLM: `{lp.get('smem_per_workgroup_bytes', 0)} B/WG`"
            ),
            (
                f"- **Theoretical Occupancy**: Core `{(s07_occupancy.get('occupancy') or 0.0) * 100:.1f}%`, "
                f"Device `{(s07_occupancy.get('device_occupancy') or 0.0) * 100:.1f}%` "
                f"(`{s07_occupancy.get('active_xe_cores')}/{rl.get('xe_cores')}` active Xe-cores, "
                f"`{s07_occupancy.get('waves')}` waves, `{s07_occupancy.get('classification')}`)"
            ),
        ]
    )
    if s07_occupancy.get("measured_occupancy_pct") is not None:
        lines.append(
            f"- **Measured Thread Occupancy (`XVE_THREADS_OCCUPANCY_ALL`)**: "
            f"`{s07_occupancy['measured_occupancy_pct']:.1f}%`"
        )

    # 3. XVE Pipe Utilization & Stall Breakdown
    lines.extend(
        [
            "",
            "## 3. XVE Execution Pipes & Stall Breakdown",
            "",
            (
                f"- **XVE Activity**: Active `{s10_inst.get('xve_active_pct') or 0.0:.1f}%`, "
                f"Stall `{s10_inst.get('xve_stall_pct') or 0.0:.1f}%`, "
                f"ALU0+ALU1 Overlap `{s10_inst.get('alu0_and_alu1_overlap_pct') or 0.0:.1f}%`"
            ),
            "",
            "| Pipe | Executed Slots | Pipe Lower Bound (`T_ms`) | Counter Utilization (%) |",
            "| :--- | ---: | ---: | ---: |",
        ]
    )
    for p_name, p_info in (s10_inst.get("per_pipe_xve") or {}).items():
        util_s = (
            f"{p_info['utilization_pct']:.2f}%"
            if p_info.get("utilization_pct") is not None
            else "N/A"
        )
        lines.append(
            f"| `{p_name}` | {int(p_info.get('slots') or 0):,} | "
            f"{p_info.get('T_ms', 0.0):.4f} ms | {util_s} |"
        )
    lines.append("")

    st_samp = s10_inst.get("stall_breakdown_sampling_pct") or {}
    st_cnt = s10_inst.get("stall_breakdown_counter_pct") or {}
    if st_cnt or st_samp:
        lines.append("### Stall Breakdown")
        for k_st, v_st in sorted(
            (st_cnt or st_samp).items(), key=lambda x: -float(x[1] or 0)
        ):
            lines.append(f"- `{k_st}`: `{v_st:.2f}%`")
        lines.append("")

    # 4. ASM Census & Hot Source Lines
    census = s11_asm.get("asm_census") or {}
    if census:
        lines.extend(
            [
                "## 4. GEN ISA Disassembly Census",
                "",
                (
                    f"- **Total Instructions**: `{census.get('instructions', 0)}` "
                    f"(Load UGM: `{census.get('load_ugm', 0)}`, Store UGM: `{census.get('store_ugm', 0)}`, "
                    f"Load SLM: `{census.get('load_slm', 0)}`, Store SLM: `{census.get('store_slm', 0)}`, "
                    f"Block2D Load/Store: `{census.get('load_block2d', 0)}/{census.get('store_block2d', 0)}`)"
                ),
                (
                    f"- **Special Ops**: Math `{census.get('math', 0)}` (`math.inv`: `{census.get('math_inv', 0)}`), "
                    f"DPAS `{census.get('dpas', 0)}`, Barrier Wait `{census.get('barrier_wait', 0)}`, "
                    f"QWord (64-bit) Ops `{census.get('qword_ops', 0)}`, `d16u32` Loads/Stores `{census.get('d16u32_sends', 0)}`"
                ),
                f"- **Send Descriptors**: `{', '.join(s11_asm.get('send_kinds') or [])}`",
                "",
            ]
        )

    stall_info = s11_asm.get("stall_sampling") or {}
    hot_srcs = stall_info.get("hot_source_locations") or []
    if hot_srcs:
        lines.extend(
            [
                "## 5. Top Stalled Helion / Triton Source Locations",
                "",
                "| Share (%) | Samples | Dominant Stalls | Helion / Triton Source |",
                "| ---: | ---: | :--- | :--- |",
            ]
        )
        for hs in hot_srcs[:10]:
            top_st = ", ".join(
                f"{k}={v}"
                for k, v in sorted(
                    (hs.get("stalls") or {}).items(), key=lambda x: -x[1]
                )[:3]
            )
            lines.append(
                f"| {hs.get('pct', 0.0):.1f}% | {hs.get('events', 0)} | `{top_st}` | "
                f"`{hs.get('source')}` |"
            )
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI Commands: list, run, collect
# ---------------------------------------------------------------------------


def cmd_list(args: argparse.Namespace) -> None:
    sweep = discover_kernel_sweep(args.kernel)
    hw_spec = get_hardware_roofline_spec()
    topology = get_xpu_device_topology(hw_spec)

    print(
        f"Kernel: {args.kernel} on {hw_spec.hardware_name} "
        f"(LLC={topology['llc_bytes'] // (1024 * 1024)} MiB)"
    )
    print(
        f"{'idx':>4s}  {sweep.shape_header}  {'sets':>5s}  "
        f"{'algo_read':>12s}  {'algo_write':>12s}  {'cfg_read':>12s}  "
        f"{'flops':>14s}  {'bottleneck':<22s}"
    )

    for idx, sh in enumerate(sweep.shapes):
        helion_call, _base_calls, cells = sweep.make_calls(sh)
        captured: list[tuple[BoundKernel[object], Config, tuple[object, ...]]] = []
        with _capture_kernel_dispatch(captured):
            helion_call()
        if not captured:
            continue
        bound, cfg, norm_args = captured[-1]
        rep: RooflineReport = bound.roofline(
            cfg, args=norm_args, measure=False, hardware_spec=hw_spec
        )
        per_set = args_tensor_bytes(norm_args)
        n_sets = compute_num_sets(per_set, int(topology["llc_bytes"]))
        print(
            f"{idx:>4d}  {cells}  {n_sets:>5d}  "
            f"{rep.algorithmic_read_bytes:>12,d}  {rep.algorithmic_write_bytes:>12,d}  "
            f"{rep.config_read_bytes:>12,d}  {rep.total_flops:>14,d}  "
            f"{rep.bottleneck:<22s}"
        )


def cmd_run(args: argparse.Namespace) -> None:
    sweep = discover_kernel_sweep(args.kernel)
    idx, _sh, helion_call, _base_calls, shape_label = select_shape_entry(
        sweep,
        shape_index=args.shape_index,
        shape_filter=args.shape,
    )
    hw_spec = get_hardware_roofline_spec()
    topology = get_xpu_device_topology(hw_spec)

    captured: list[tuple[BoundKernel[object], Config, tuple[object, ...]]] = []
    with torch.no_grad(), _capture_kernel_dispatch(captured):
        helion_call()

    if not captured:
        raise RuntimeError(f"Kernel {args.kernel!r} did not dispatch a Helion kernel")

    bound, cfg, norm_args = captured[-1]
    if args.dump_triton:
        Path(args.dump_triton).write_text(bound.to_triton_code(cfg))

    compiled_fn = bound.compile_config(cfg, allow_print=False)
    per_set_bytes = args_tensor_bytes(norm_args)
    n_sets = (
        args.sets
        if args.sets and args.sets > 0
        else compute_num_sets(per_set_bytes, int(topology["llc_bytes"]))
    )

    with torch.no_grad():
        arg_sets = clone_arg_sets(norm_args, n_sets)
        order = [arg_sets[k] for k in stride2_order(len(arg_sets))]
        warmup = args.warmup if args.warmup >= 0 else len(arg_sets) + 1
        iters = args.iters if args.iters > 0 else max(10, len(arg_sets))

        for j in range(warmup):
            compiled_fn(*order[j % len(order)])
        if torch.xpu.is_available():
            torch.xpu.synchronize()

        if args.time:
            t0 = torch.xpu.Event(enable_timing=True)
            t1 = torch.xpu.Event(enable_timing=True)
            t0.record()
            for i in range(iters):
                compiled_fn(*order[(warmup + i) % len(order)])
            t1.record()
            torch.xpu.synchronize()
            us = (t0.elapsed_time(t1) * 1000.0) / iters
            print(
                f"{args.kernel} [{idx}] ({shape_label}): {us:.3f} us/call, "
                f"{per_set_bytes} buffer bytes/call, sets={len(arg_sets)}"
            )
        else:
            for i in range(iters):
                compiled_fn(*order[(warmup + i) % len(order)])
                if args.sync_each and torch.xpu.is_available():
                    torch.xpu.synchronize()
            if torch.xpu.is_available():
                torch.xpu.synchronize()

    print(
        f"PROFILE_RUN_DONE kernel={args.kernel} shape_index={idx} "
        f"warmup={warmup} iters={iters} sets={len(arg_sets)}"
    )


def _measure_uninstrumented_shape(
    kernel_name: str,
    shape_index: int | None,
    shape_filter: str | None,
    out_dir: Path,
) -> dict[str, Any]:
    """Benchmark the selected shape without unitrace instrumentation and compute static Roofline."""
    sweep = discover_kernel_sweep(kernel_name)
    idx, _sh, helion_call, base_calls, shape_label = select_shape_entry(
        sweep,
        shape_index=shape_index,
        shape_filter=shape_filter,
    )
    hw_spec = get_hardware_roofline_spec()
    topology = get_xpu_device_topology(hw_spec)

    captured: list[tuple[BoundKernel[object], Config, tuple[object, ...]]] = []
    with torch.no_grad(), _capture_kernel_dispatch(captured):
        helion_call()

    if not captured:
        raise RuntimeError(f"Kernel {kernel_name!r} did not dispatch a Helion kernel")

    bound, cfg, norm_args = captured[-1]
    kernels_dir = out_dir / "kernels"
    kernels_dir.mkdir(parents=True, exist_ok=True)
    triton_code = bound.to_triton_code(cfg)
    triton_path = kernels_dir / "k0_triton.py"
    triton_path.write_text(triton_code)

    # Use plain do_bench on XPU to avoid Level Zero graph capture issues with 256MB L2 flushes
    effective_cudagraph = sweep.use_cudagraph and hw_spec.device_kind != "xpu"
    ms_helion = _bench._bench(helion_call, effective_cudagraph, sweep.warmup, sweep.rep)
    base_ms = {
        name: _bench._bench(call, effective_cudagraph, sweep.warmup, sweep.rep)
        for name, call in base_calls
    }
    best_name = min(base_ms, key=lambda n: base_ms[n]) if base_ms else "none"
    speedup = (
        base_ms[best_name] / ms_helion if base_ms and ms_helion > 0 else float("nan")
    )

    report = bound.roofline(
        cfg,
        args=norm_args,
        measure=False,
        latency_ms=ms_helion,
        hardware_spec=hw_spec,
    )
    s05_dict = report.to_dict()
    s05_dict["config"] = dict(cfg)
    s05_dict["shape_index"] = idx
    s05_dict["shape"] = shape_label
    s05_dict["helion_us"] = round(ms_helion * 1000.0, 3)
    s05_dict["baselines_us"] = {n: round(v * 1000.0, 3) for n, v in base_ms.items()}
    s05_dict["best_baseline"] = best_name
    s05_dict["speedup"] = round(speedup, 4)
    s05_dict["triton_source_file"] = str(triton_path)
    (out_dir / "05_static_roofline.json").write_text(
        json.dumps(s05_dict, indent=2) + "\n"
    )

    per_set_bytes = args_tensor_bytes(norm_args)
    n_sets = compute_num_sets(per_set_bytes, int(topology["llc_bytes"]))
    return {
        "shape_index": idx,
        "shape": shape_label,
        "helion_us": round(ms_helion * 1000.0, 3),
        "baselines_us": s05_dict["baselines_us"],
        "best_baseline": best_name,
        "speedup": round(speedup, 4),
        "static_roofline": s05_dict,
        "triton_code": triton_code,
        "triton_source_file": str(triton_path),
        "per_set_bytes": per_set_bytes,
        "recommended_sets": n_sets,
        "topology": topology,
        "hw_spec": hw_spec,
    }


def cmd_collect(args: argparse.Namespace) -> None:
    unitrace = find_unitrace()
    if not unitrace:
        raise SystemExit(
            "unitrace not found on PATH or standard paths (set HELION_UNITRACE)"
        )

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[1/5] Benchmarking un-instrumented '{args.kernel}' and computing static Roofline..."
    )
    base_info = _measure_uninstrumented_shape(
        args.kernel, args.shape_index, args.shape, out_dir
    )
    hw_spec: HardwareRooflineSpec = base_info["hw_spec"]
    topology: dict[str, Any] = base_info["topology"]
    s05_roofline: dict[str, Any] = base_info["static_roofline"]
    shape_idx = int(base_info["shape_index"])
    shape_label = str(base_info["shape"])

    n_sets = (
        args.sets if args.sets and args.sets > 0 else int(base_info["recommended_sets"])
    )
    warmup = args.warmup if args.warmup >= 0 else min(8, n_sets)
    iters = args.iters if args.iters > 0 else max(12, min(24, n_sets))

    run_cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "run",
        "--kernel",
        args.kernel,
        "--shape-index",
        str(shape_idx),
        "--sets",
        str(n_sets),
        "--warmup",
        str(warmup),
        "--iters",
        str(iters),
    ]
    if not args.no_sync_each:
        run_cmd.append("--sync-each")

    tag = args.kernel_filter or f"_helion_{args.kernel}"
    env = dict(os.environ)

    summary: dict[str, Any] = {
        "kernel": args.kernel,
        "shape_index": shape_idx,
        "shape": shape_label,
        "kernel_filter": tag,
        "device": topology["hardware_name"],
        "compute_capability": topology["compute_capability"],
        "helion_us": base_info["helion_us"],
        "baselines_us": base_info["baselines_us"],
        "best_baseline": base_info["best_baseline"],
        "speedup": base_info["speedup"],
        "warmup": warmup,
        "iters": iters,
        "sets": n_sets,
        "sync_each": not args.no_sync_each,
        "files": {
            "static_roofline": str(out_dir / "05_static_roofline.json"),
            "triton_source": base_info["triton_source_file"],
        },
    }

    def _run_unitrace(
        ut_args: list[str],
        log_name: str,
        *,
        optional: bool = False,
        extra_env: dict[str, str] | None = None,
    ) -> str | None:
        log_path = out_dir / f"{log_name}.log"
        merged_env = dict(env, **(extra_env or {}))
        with open(log_path, "w") as f:
            rc = subprocess.run(
                [unitrace, *ut_args, *run_cmd],
                cwd=out_dir,
                env=merged_env,
                stdout=f,
                stderr=subprocess.STDOUT,
                check=False,
            ).returncode
        text = log_path.read_text(errors="replace")
        if optional and (
            "Unable to get metric group" in text
            or "No '=== Device #" in text
            or "Metric group" in text
            and "is not supported" in text
        ):
            summary.setdefault("missing", {})[log_name] = (
                text.strip().splitlines()[-1][:300] if text.strip() else "unavailable"
            )
            log_path.rename(out_dir / f"{log_name}.unavailable.log")
            return None
        if rc != 0 or "PROFILE_RUN_DONE" not in text:
            if optional:
                summary.setdefault("missing", {})[log_name] = f"exit code {rc}"
                return None
            raise SystemExit(f"unitrace run failed ({log_path}); check log for details")
        summary["files"][log_name] = str(log_path)
        return text

    print(f"[2/5] Collecting unitrace timeline and launch properties (tag={tag})...")
    timeline_text = _run_unitrace(["-d", "-v"], "timeline")
    assert timeline_text is not None
    kernels = parse_timeline(timeline_text, tag)
    if not kernels:
        raise SystemExit(
            f"No kernel matching {tag!r} found in {out_dir / 'timeline.log'}"
        )
    total_calls = max(1, warmup + iters)
    for k in kernels:
        k["per_call"] = max(1, k["calls"] // total_calls)
    summary["kernels"] = [k["name"] for k in kernels]
    summary["dominant_kernel_name"] = kernels[0]["name"]

    if args.groups == "auto":
        groups_to_collect = list(default_counter_groups(hw_spec))
    else:
        groups_to_collect = [g.strip() for g in args.groups.split(",") if g.strip()]

    print(
        f"[3/5] Collecting unitrace hardware counter groups: {', '.join(groups_to_collect)}..."
    )
    preempt: dict[str, dict[str, Any]] = {}
    parsed_groups: dict[str, dict[str, Any]] = {}
    for g in groups_to_collect:
        g_text = _run_unitrace(
            ["-q", "-g", g, "--include-kernels", tag],
            g,
            optional=True,
        )
        if g_text is not None:
            with contextlib.suppress(ValueError):
                pk = parse_counters_for_kernels(
                    out_dir / f"{g}.log",
                    kernels,
                    warmup,
                    preempt.setdefault(g, {}),
                )
                if kernels[0]["name"] in pk:
                    parsed_groups[g] = pk[kernels[0]["name"]]
                    g_json = out_dir / f"{g}.call.json"
                    g_json.write_text(
                        json.dumps(pk[kernels[0]["name"]], indent=2) + "\n"
                    )
                    summary["files"][f"{g}_json"] = str(g_json)

    nocomp_counters: dict[str, Any] | None = None
    if not args.no_nocomp:
        nc_text = _run_unitrace(
            ["-q", "-g", "ComputeBasic", "--include-kernels", tag],
            "ComputeBasic.nocomp",
            optional=True,
            extra_env={
                "NEOReadDebugKeys": "1",
                "RenderCompressedBuffersEnabled": "0",
            },
        )
        if nc_text is not None:
            with contextlib.suppress(ValueError):
                pk_nc = parse_counters_for_kernels(
                    out_dir / "ComputeBasic.nocomp.log", kernels, warmup
                )
                nocomp_counters = pk_nc.get(kernels[0]["name"])

    stall_dir: Path | None = None
    if not args.no_stall:
        print("[4/5] Collecting unitrace EU stall sampling...")
        stall_dir = out_dir / "stall"
        shutil.rmtree(stall_dir, ignore_errors=True)
        stall_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "stall.log", "w") as f:
            subprocess.run(
                [unitrace, "--stall-sampling", "--include-kernels", tag, *run_cmd],
                cwd=stall_dir,
                env=dict(env, ZE_AFFINITY_MASK=env.get("ZE_AFFINITY_MASK", "0")),
                stdout=f,
                stderr=subprocess.STDOUT,
                check=False,
            )
        summary["files"]["stall_dir"] = str(stall_dir)
        summary["files"]["stall_log"] = str(out_dir / "stall.log")

    asm_dir: Path | None = None
    if not args.no_asm:
        print("[5/5] Extracting ZE binary & running ocloc disasm...")
        dump_dir = out_dir / "asm_dump"
        shutil.rmtree(dump_dir, ignore_errors=True)
        dump_dir.mkdir(parents=True, exist_ok=True)
        short_run = [
            sys.executable,
            str(Path(__file__).resolve()),
            "run",
            "--kernel",
            args.kernel,
            "--shape-index",
            str(shape_idx),
            "--sets",
            "2",
            "--warmup",
            "0",
            "--iters",
            "1",
        ]
        with open(out_dir / "asm.log", "w") as f:
            subprocess.run(
                short_run,
                cwd=dump_dir,
                env=dict(env, DumpZEBin="1", NEOReadDebugKeys="1"),
                stdout=f,
                stderr=subprocess.STDOUT,
                check=False,
            )
        asm_dir = out_dir / "asm"
        shutil.rmtree(asm_dir, ignore_errors=True)
        asm_dir.mkdir(parents=True, exist_ok=True)
        aot_dev = detect_aot_device(hw_spec)
        ocloc_bin = shutil.which("ocloc") or "/usr/bin/ocloc"
        if Path(ocloc_bin).is_file():
            for elf in sorted(dump_dir.glob("dumped_zebin_module*.elf")):
                if tag.encode() in elf.read_bytes():
                    target_sub = asm_dir / elf.stem
                    target_sub.mkdir(parents=True, exist_ok=True)
                    subprocess.run(
                        [
                            ocloc_bin,
                            "disasm",
                            "-file",
                            str(elf),
                            "-dump",
                            str(target_sub),
                            "-device",
                            aot_dev,
                        ],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        check=False,
                    )
        shutil.rmtree(dump_dir, ignore_errors=True)
        summary["files"]["asm_dir"] = str(asm_dir)

    # Assemble step JSONs (06..11)
    dom = kernels[0]
    asm_by_kernel = collect_asm_census(asm_dir, stall_dir, kernels, out_dir)
    stalls_by_kernel = parse_stall_sampling(
        stall_dir,
        kernels,
        out_dir,
        triton_code=base_info["triton_code"],
    )
    dom_stall = stalls_by_kernel.get(dom["name"], {})
    dom_asm = asm_by_kernel.get(dom["name"], {})

    s09_mem, s10_inst = derive_memory_and_pipe_metrics(
        parsed_groups,
        static_roofline=s05_roofline,
        topology=topology,
        runner_time_us=base_info["helion_us"],
        stall_sampling_pct=dom_stall.get("stall_pct_of_all_samples"),
    )

    if nocomp_counters is not None:
        s09_mem["compression_off"] = {
            "measured_read_bytes": _counter_val(
                nocomp_counters, "GPU_MEMORY_BYTE_READ[bytes]"
            ),
            "measured_write_bytes": _counter_val(
                nocomp_counters, "GPU_MEMORY_BYTE_WRITE[bytes]"
            ),
            "note": "ComputeBasic with RenderCompressedBuffersEnabled=0",
        }

    s06_parser = {
        "step": "06_kernel_profiler_parser",
        "dominant_kernel_name": dom["name"],
        "kernels_per_call": [k["name"] for k in kernels],
        "raw_log_files": {g: str(out_dir / f"{g}.log") for g in parsed_groups},
        "parsed_counter_files": {
            g: str(out_dir / f"{g}.call.json") for g in parsed_groups
        },
        "preempted_instances": {k: v for k, v in preempt.items() if v} or None,
        "summary": {
            "gpu_time_ns": _counter_val(
                parsed_groups.get("ComputeBasic", {}), "GpuTime[ns]"
            ),
            "dram_read_bytes": s09_mem["measured_read_bytes"],
            "dram_write_bytes": s09_mem["measured_write_bytes"],
            "l3_read_bytes": s09_mem["l3_read_bytes"],
            "l3_write_bytes": s09_mem["l3_write_bytes"],
        },
        "per_kernel": kernels,
    }

    occ_dom = compute_occupancy(dom, topology) or {}
    s07_occupancy = {
        "step": "07_kernel_occupancy",
        "dominant_kernel_name": dom["name"],
        "device": topology["hardware_name"],
        **occ_dom,
        "measured_occupancy_pct": s09_mem.get("occupancy_pct"),
        "barrier_waits_in_asm": (dom_asm.get("census") or {}).get("barrier_wait"),
    }

    is3 = parsed_groups.get("InternalSet3", {})
    grf_rd = _counter_val(is3, "std.eu.grf.rd[events]")
    grf_wr = _counter_val(is3, "std.eu.grf.wr[events]")
    conf_rd = _counter_val(is3, "std.eu.grf.conflict.rd[events]")
    conf_wr = _counter_val(is3, "std.eu.grf.conflict.wr[events]")
    s08_reg = {
        "step": "08_register_bank_conflict",
        "dominant_kernel_name": dom["name"],
        "launch_params": {
            "registers_per_thread": dom["grf"],
            "spill_memory_per_thread_bytes": dom["spill_bytes"],
            "private_memory_per_thread_bytes": dom["private_bytes"],
            "large_register_file": dom["grf"] > 128,
        },
        "spill_measured": {
            "spilled": dom["spill_bytes"] > 0,
            "spill_bytes_per_thread": dom["spill_bytes"],
            "scratch_sends": (dom_asm.get("census") or {}).get("scratch_sends"),
            "lsc_byte_write": s09_mem.get("lsc_write_bytes"),
        },
        "slm_bank_conflicts": s09_mem.get("slm_bank_conflicts"),
        "grf_conflicts_measured": (
            {
                "conflict_rd_events": conf_rd,
                "conflict_wr_events": conf_wr,
                "grf_rd_events": grf_rd,
                "grf_wr_events": grf_wr,
                "conflict_rd_pct": (
                    round(100.0 * conf_rd / grf_rd, 2)
                    if conf_rd is not None and grf_rd
                    else None
                ),
                "conflict_wr_pct": (
                    round(100.0 * conf_wr / grf_wr, 2)
                    if conf_wr is not None and grf_wr
                    else None
                ),
            }
            if is3
            else None
        ),
    }

    s09_full = {
        "step": "09_memory_load_store",
        "dominant_kernel_name": dom["name"],
        **s09_mem,
    }
    s10_full = {
        "step": "10_instructions_measurement",
        "dominant_kernel_name": dom["name"],
        **s10_inst,
    }
    s11_full = {
        "step": "11_asm_source_mapping",
        "dominant_kernel_name": dom["name"],
        "triton_source_file": base_info["triton_source_file"],
        "asm_file": dom_asm.get("asm_file"),
        "asm_census": dom_asm.get("census"),
        "send_kinds": dom_asm.get("send_kinds"),
        "stall_sampling": dom_stall or None,
    }

    step_outputs = [
        s06_parser,
        s07_occupancy,
        s08_reg,
        s09_full,
        s10_full,
        s11_full,
    ]
    for st_obj in step_outputs:
        st_name = str(st_obj["step"])
        st_path = out_dir / f"{st_name}.json"
        st_path.write_text(json.dumps(st_obj, indent=2) + "\n")
        summary["files"][st_name] = str(st_path)

    md_report = build_markdown_report(
        summary,
        s05_roofline,
        s07_occupancy,
        s08_reg,
        s09_full,
        s10_full,
        s11_full,
    )
    md_path = out_dir / "hw_profile_report.md"
    md_path.write_text(md_report + "\n")
    summary["files"]["markdown_report"] = str(md_path)

    (out_dir / "collect.json").write_text(json.dumps(summary, indent=2) + "\n")
    print()
    print(md_report)
    print(f"\nSaved hardware profile artifacts to: {out_dir}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Intel XPU hardware counter, stall, and assembly collection harness for Helion kernels."
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    pl = sub.add_parser(
        "list", help="List shapes and static Roofline metrics for a kernel"
    )
    pl.add_argument(
        "--kernel",
        required=True,
        help="Pretuned kernel name (e.g. rms_norm_dynamic_per_token_quant)",
    )

    pr = sub.add_parser(
        "run", help="Run one shape over rotating input sets (for unitrace wrapping)"
    )
    pr.add_argument("--kernel", required=True, help="Pretuned kernel name")
    pr.add_argument(
        "--shape-index",
        type=int,
        default=-1,
        help="Shape index (default: -1, largest shape)",
    )
    pr.add_argument("--shape", default=None, help="Shape substring filter")
    pr.add_argument(
        "--sets", type=int, default=0, help="Override rotating buffer set count"
    )
    pr.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Warmup calls (-1: one full set rotation + 1)",
    )
    pr.add_argument(
        "--iters",
        type=int,
        default=20,
        help="Profiled iterations (0: one full set rotation)",
    )
    pr.add_argument(
        "--sync-each", action="store_true", help="Synchronize after each iteration"
    )
    pr.add_argument(
        "--time", action="store_true", help="Measure and print XPU event time per call"
    )
    pr.add_argument(
        "--dump-triton", default=None, help="Write generated Triton kernel code to file"
    )

    pc = sub.add_parser(
        "collect",
        help="Full hardware collection (Roofline + unitrace + ocloc + report)",
    )
    pc.add_argument("--kernel", required=True, help="Pretuned kernel name")
    pc.add_argument(
        "--shape-index",
        type=int,
        default=-1,
        help="Shape index (default: -1, largest shape)",
    )
    pc.add_argument("--shape", default=None, help="Shape substring filter")
    pc.add_argument(
        "--out", required=True, help="Output directory for logs, step JSONs, and report"
    )
    pc.add_argument(
        "--groups",
        default="auto",
        help="Comma-separated unitrace counter groups (default: auto per Xe/Xe2 architecture)",
    )
    pc.add_argument(
        "--kernel-filter",
        default=None,
        help="Kernel name substring filter (default: _helion_<kernel>)",
    )
    pc.add_argument(
        "--sets", type=int, default=0, help="Override rotating buffer set count"
    )
    pc.add_argument(
        "--warmup", type=int, default=-1, help="Warmup calls before counted instances"
    )
    pc.add_argument("--iters", type=int, default=0, help="Counted iterations")
    pc.add_argument(
        "--no-sync-each", action="store_true", help="Do not sync after each call"
    )
    pc.add_argument(
        "--no-nocomp",
        action="store_true",
        help="Skip compression-disabled ComputeBasic run",
    )
    pc.add_argument(
        "--no-stall", action="store_true", help="Skip unitrace --stall-sampling run"
    )
    pc.add_argument(
        "--no-asm", action="store_true", help="Skip DumpZEBin + ocloc disassembly"
    )
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    dispatch = {
        "list": cmd_list,
        "run": cmd_run,
        "collect": cmd_collect,
    }
    dispatch[args.cmd](args)


if __name__ == "__main__":
    main()

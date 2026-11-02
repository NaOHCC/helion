#!/usr/bin/env python3
"""Run pretuned kernels benchmark sweep with Roofline performance analysis.

Combines ``pretuned_kernels/run.py`` baseline comparisons with Helion's
built-in ``BoundKernel.roofline()`` static + empirical (SOL%) analyzer,
reporting per-shape and per-kernel Speed-of-Light (SOL%), achieved TFLOP/s
and GB/s, tile traffic amplification, Xe-core wave occupancy, and bottleneck
classifications.

Usage:
    python pretuned_kernels/run_roofline.py
    python pretuned_kernels/run_roofline.py --kernels vector_add,softmax,scaled_mm
    python pretuned_kernels/run_roofline.py --output pretuned-roofline.json --markdown pretuned-roofline.md
"""

from __future__ import annotations

import argparse
from collections import Counter
import contextlib
import gc
import json
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
from typing import TYPE_CHECKING
from typing import Any

import torch

PRETUNED_KERNELS_DIR = Path(__file__).resolve().parent
REPO_ROOT = PRETUNED_KERNELS_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(PRETUNED_KERNELS_DIR) not in sys.path:
    sys.path.insert(0, str(PRETUNED_KERNELS_DIR))

import _bench  # noqa: E402
from run import _HARDWARE_BY_COMPUTE  # noqa: E402
from run import KERNELS  # noqa: E402
from run import _import_kernel_module  # noqa: E402
from run import _supported_hardware  # noqa: E402
from run import parse_kernel_names  # noqa: E402

from helion._hardware import get_hardware_info  # noqa: E402
from helion._hardware import get_hardware_roofline_spec  # noqa: E402
from helion.runtime.kernel import BoundKernel  # noqa: E402
from helion.runtime.kernel import Kernel  # noqa: E402

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterable
    from collections.abc import Iterator
    from collections.abc import Sequence

    from helion._compiler.roofline import RooflineReport
    from helion.runtime.config import Config


@contextlib.contextmanager
def _capture_kernel_dispatch(
    captured: list[tuple[BoundKernel[object], Config, tuple[object, ...]]],
) -> Iterator[None]:
    """Capture ``(bound_kernel, config, normalized_args)`` during warmup only."""
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


def _make_roofline_run_sweep(
    shape_records_out: list[dict[str, object]],
    *,
    max_shapes: int = 0,
    allow_xpu_graph: bool = False,
) -> Callable[..., dict[str, Any]]:
    """Build a ``_bench.run_sweep`` replacement that also computes Roofline metrics."""

    def roofline_run_sweep(
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
        if use_cudagraph and pre_captured_cudagraph:
            raise ValueError(
                "use_cudagraph and pre_captured_cudagraph are mutually exclusive"
            )
        if make_resets is not None and not pre_captured_cudagraph:
            raise ValueError("make_resets requires pre_captured_cudagraph=True")

        def _p(*args: object) -> None:
            if verbose:
                print(*args)

        hw_spec = get_hardware_roofline_spec()
        effective_cudagraph = use_cudagraph and (
            allow_xpu_graph or hw_spec.device_kind != "xpu"
        )
        if verbose:
            _p(
                f"GPU: {hw_spec.hardware_name} [{hw_spec.compute_capability}] "
                f"({hw_spec.num_compute_units} CUs, "
                f"Peak XMX FP16={hw_spec.matrix_tflops():.1f} TFLOP/s, "
                f"Peak Vec FP32={hw_spec.vector_tflops():.2f} TFLOP/s, "
                f"Peak BW={hw_spec.peak_memory_bandwidth_gbps:.1f} GB/s)"
            )

        speedups_by_base: dict[str, list[float]] = {}
        best_speedups: list[float] = []
        helion_wins = 0
        best_speedup = 0.0
        header_printed = False

        shape_list = list(shapes)
        if max_shapes > 0:
            shape_list = shape_list[:max_shapes]

        for shape in shape_list:
            helion_call, baseline_calls, shape_cells = make_calls(shape)
            names = [n for n, _ in baseline_calls]
            if not header_printed:
                for n in names:
                    speedups_by_base[n] = []
                base_hdr = "  ".join(f"{n + ' (us)':>13s}" for n in names)
                _p(
                    f"{shape_header}  {'helion (us)':>12s}  {base_hdr}  "
                    f"{'speedup':>8s}  {'SOL%':>6s}  {'CfgSOL%':>7s}  "
                    f"{'TFLOP/s':>8s}  {'AlgoGB/s':>9s}  {'Amp':>5s}  "
                    f"{'Waves':>6s}  {'Bottleneck':<24s}"
                )
                header_printed = True

            captured: list[tuple[BoundKernel[object], Config, tuple[object, ...]]] = []
            with _capture_kernel_dispatch(captured):
                helion_call()  # warmup / compile & capture bound kernel + args

            if pre_captured_cudagraph:
                calls = [helion_call, *(call for _name, call in baseline_calls)]
                resets = (
                    list(make_resets(shape))
                    if make_resets is not None
                    else [None] * len(calls)
                )
                _bench.thermal_warmup(thermal_warmup_ms)
                timings = _bench.bench_pre_captured_cudagraphs(
                    calls,
                    rep=rep,
                    resets=resets if make_resets is not None else None,
                )
                ms_helion, *baseline_timings = timings
                base_ms = dict(zip(names, baseline_timings, strict=True))
            else:
                ms_helion = _bench._bench(helion_call, effective_cudagraph, warmup, rep)
                base_ms = {
                    name: _bench._bench(call, effective_cudagraph, warmup, rep)
                    for name, call in baseline_calls
                }

            for name in names:
                speedups_by_base[name].append(
                    base_ms[name] / ms_helion if ms_helion > 0 else float("nan")
                )
            best_name = min(base_ms, key=lambda name: base_ms[name])
            speedup = base_ms[best_name] / ms_helion if ms_helion > 0 else float("nan")
            best_speedups.append(speedup)
            if speedup > 1.0:
                helion_wins += 1
            best_speedup = max(best_speedup, speedup)

            report: RooflineReport | None = None
            if captured:
                bound, cfg, norm_args = captured[-1]
                with contextlib.suppress(Exception):
                    report = bound.roofline(
                        cfg,
                        args=norm_args,
                        measure=False,
                        latency_ms=ms_helion,
                        hardware_spec=hw_spec,
                    )

            base_cols = "  ".join(f"{base_ms[n] * 1000:>13.2f}" for n in names)
            if report is not None:
                sol_str = f"{(report.sol_pct or 0.0):>5.1f}%"
                cfg_sol_str = f"{(report.config_sol_pct or 0.0):>6.1f}%"
                tflops_str = f"{(report.achieved_tflops or 0.0):>8.2f}"
                gbps_str = f"{(report.achieved_algo_gbps or 0.0):>9.1f}"
                amp_str = f"{report.traffic_amplification:>4.2f}x"
                waves_str = f"{report.waves_per_compute_unit:>6.2f}"
                bneck_str = f"{report.bottleneck:<24s}"
            else:
                sol_str = f"{'N/A':>6s}"
                cfg_sol_str = f"{'N/A':>7s}"
                tflops_str = f"{'N/A':>8s}"
                gbps_str = f"{'N/A':>9s}"
                amp_str = f"{'N/A':>5s}"
                waves_str = f"{'N/A':>6s}"
                bneck_str = f"{'N/A':<24s}"

            _p(
                f"{shape_cells}  {ms_helion * 1000:>12.2f}  {base_cols}  "
                f"{speedup:>7.2f}x  {sol_str}  {cfg_sol_str}  "
                f"{tflops_str}  {gbps_str}  {amp_str}  {waves_str}  {bneck_str}"
            )

            shape_records_out.append(
                {
                    "shape": " ".join(shape_cells.split()),
                    "helion_us": round(ms_helion * 1000.0, 3),
                    "baselines_us": {n: round(base_ms[n] * 1000.0, 3) for n in names},
                    "best_baseline": best_name,
                    "speedup": round(speedup, 4),
                    "roofline": report.to_dict() if report is not None else None,
                }
            )

        names = list(speedups_by_base)
        per_baseline = {
            n: {
                "wins": sum(1 for s in speedups_by_base[n] if s > 1.0),
                "total": len(speedups_by_base[n]),
                "geomean": round(_bench.geomean(speedups_by_base[n]), 4),
                "best_speedup": round(max(speedups_by_base[n], default=0.0), 4),
            }
            for n in names
        }
        for n in names:
            m = per_baseline[n]
            _p(
                f"vs {n}: wins={m['wins']}/{m['total']} "
                f"geomean={m['geomean']:.3f}x best={m['best_speedup']:.2f}x"
            )
        total = len(best_speedups)
        gm = _bench.geomean(best_speedups)
        _p(
            f"\nHelion faster on {helion_wins}/{total} shapes vs the best baseline; "
            f"geomean speedup {gm:.3f}x; best speedup {best_speedup:.2f}x."
        )
        return {
            "helion_wins": helion_wins,
            "total": total,
            "geomean": round(gm, 4),
            "best_speedup": round(best_speedup, 4),
            "baselines": per_baseline,
        }

    return roofline_run_sweep


def _summarize_kernel_roofline(
    shape_records: list[dict[str, object]],
) -> dict[str, object] | None:
    reports = [
        r["roofline"] for r in shape_records if isinstance(r.get("roofline"), dict)
    ]
    if not reports:
        return None

    sols = [
        float(rep["empirical"]["sol_pct"])
        for rep in reports
        if isinstance(rep, dict)
        and isinstance(rep.get("empirical"), dict)
        and rep["empirical"].get("sol_pct") is not None
    ]
    cfg_sols = [
        float(rep["empirical"]["config_sol_pct"])
        for rep in reports
        if isinstance(rep, dict)
        and isinstance(rep.get("empirical"), dict)
        and rep["empirical"].get("config_sol_pct") is not None
    ]
    tflops_list = [
        float(rep["empirical"]["achieved_tflops"])
        for rep in reports
        if isinstance(rep, dict)
        and isinstance(rep.get("empirical"), dict)
        and rep["empirical"].get("achieved_tflops") is not None
    ]
    gbps_list = [
        float(rep["empirical"]["achieved_algo_gbps"])
        for rep in reports
        if isinstance(rep, dict)
        and isinstance(rep.get("empirical"), dict)
        and rep["empirical"].get("achieved_algo_gbps") is not None
    ]
    amps = [
        float(rep["config_traffic"]["traffic_amplification"])
        for rep in reports
        if isinstance(rep, dict)
        and isinstance(rep.get("config_traffic"), dict)
        and rep["config_traffic"].get("traffic_amplification") is not None
    ]
    waves = [
        float(rep["config_traffic"]["waves_per_compute_unit"])
        for rep in reports
        if isinstance(rep, dict)
        and isinstance(rep.get("config_traffic"), dict)
        and rep["config_traffic"].get("waves_per_compute_unit") is not None
    ]
    bottlenecks = Counter(
        str(rep["bottleneck"])
        for rep in reports
        if isinstance(rep, dict) and rep.get("bottleneck") is not None
    )
    hints: list[str] = []
    for rep in reports:
        if isinstance(rep, dict) and isinstance(rep.get("hints"), list):
            for h in rep["hints"]:
                if isinstance(h, str) and h not in hints:
                    hints.append(h)

    dominant_bottleneck = bottlenecks.most_common(1)[0][0] if bottlenecks else "unknown"
    return {
        "mean_sol_pct": round(statistics.mean(sols), 2) if sols else 0.0,
        "geomean_sol_pct": round(_bench.geomean(sols), 2) if sols else 0.0,
        "min_sol_pct": round(min(sols), 2) if sols else 0.0,
        "max_sol_pct": round(max(sols), 2) if sols else 0.0,
        "mean_config_sol_pct": (
            round(statistics.mean(cfg_sols), 2) if cfg_sols else 0.0
        ),
        "peak_achieved_tflops": round(max(tflops_list), 3) if tflops_list else 0.0,
        "mean_achieved_tflops": (
            round(statistics.mean(tflops_list), 3) if tflops_list else 0.0
        ),
        "peak_achieved_algo_gbps": round(max(gbps_list), 2) if gbps_list else 0.0,
        "mean_achieved_algo_gbps": (
            round(statistics.mean(gbps_list), 2) if gbps_list else 0.0
        ),
        "mean_traffic_amplification": (
            round(statistics.mean(amps), 3) if amps else 1.0
        ),
        "mean_waves_per_cu": round(statistics.mean(waves), 2) if waves else 0.0,
        "dominant_bottleneck": dominant_bottleneck,
        "bottleneck_counts": dict(bottlenecks),
        "hints": hints,
    }


def run_kernel_with_roofline(
    name: str,
    current_hardware: str,
    *,
    max_shapes: int = 0,
    allow_xpu_graph: bool = False,
) -> dict[str, object]:
    record: dict[str, object] = {"kernel": name}
    supported = sorted(_supported_hardware(name))
    record["pretuned_hardware"] = supported
    if current_hardware not in supported:
        record["skipped"] = (
            f"pretuned for {supported}; current hardware is {current_hardware}"
        )
        print(f"Skipping {name}: {record['skipped']}")
        return record

    shape_records: list[dict[str, object]] = []
    orig_run_sweep = _bench.run_sweep
    _bench.run_sweep = _make_roofline_run_sweep(
        shape_records,
        max_shapes=max_shapes,
        allow_xpu_graph=allow_xpu_graph,
    )
    try:
        module = _import_kernel_module(name)
        metrics = module.main(verbose=True)
        record.update(
            helion_wins=int(metrics["helion_wins"]),
            total=int(metrics["total"]),
            geomean=float(metrics["geomean"]),
            best_speedup=float(metrics["best_speedup"]),
            cudagraph=bool(module.use_cudagraph()),
        )
        if metrics.get("baselines"):
            record["baselines"] = metrics["baselines"]
        roofline_summary = _summarize_kernel_roofline(shape_records)
        if roofline_summary is not None:
            record["roofline_summary"] = roofline_summary
            print(
                f"Roofline Summary ({name}): "
                f"Mean SOL={roofline_summary['mean_sol_pct']:.1f}% "
                f"(Min={roofline_summary['min_sol_pct']:.1f}%, Max={roofline_summary['max_sol_pct']:.1f}%, "
                f"CfgSOL={roofline_summary['mean_config_sol_pct']:.1f}%) | "
                f"Peak={roofline_summary['peak_achieved_tflops']:.2f} TFLOP/s, "
                f"{roofline_summary['peak_achieved_algo_gbps']:.1f} GB/s | "
                f"Amp={roofline_summary['mean_traffic_amplification']:.2f}x | "
                f"Bottleneck={roofline_summary['dominant_bottleneck']}"
            )
        record["shapes"] = shape_records
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
        print(f"ERROR running {name}: {record['error']}", file=sys.stderr)
    finally:
        _bench.run_sweep = orig_run_sweep
        gc.collect()
        if torch.xpu.is_available():
            torch.xpu.empty_cache()
    return record


def _run_kernel_subprocess(
    name: str,
    current_hardware: str,
    *,
    max_shapes: int = 0,
    allow_xpu_graph: bool = False,
) -> dict[str, object]:
    supported = sorted(_supported_hardware(name))
    if current_hardware not in supported:
        record: dict[str, object] = {
            "kernel": name,
            "pretuned_hardware": supported,
            "skipped": (
                f"pretuned for {supported}; current hardware is {current_hardware}"
            ),
        }
        print(f"Skipping {name}: {record['skipped']}")
        return record

    with tempfile.NamedTemporaryFile(
        prefix=f"roofline_{name}_", suffix=".json", delete=False
    ) as tmp:
        tmp_path = Path(tmp.name)

    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--kernels",
        name,
        "--hardware",
        current_hardware,
        "--no-isolate",
        "--output",
        str(tmp_path),
    ]
    if max_shapes > 0:
        cmd.extend(["--max-shapes", str(max_shapes)])
    if allow_xpu_graph:
        cmd.append("--allow-xpu-graph")

    try:
        subprocess.run(cmd, check=False)
        if tmp_path.is_file() and tmp_path.stat().st_size > 0:
            items = json.loads(tmp_path.read_text())
            if isinstance(items, list) and items:
                return items[0]
        return {
            "kernel": name,
            "pretuned_hardware": supported,
            "error": "Subprocess exited without producing output JSON",
        }
    finally:
        with contextlib.suppress(OSError):
            tmp_path.unlink()


def render_markdown_report(
    records: list[dict[str, object]],
    hw_name: str,
    compute: str,
) -> str:
    hw_spec = get_hardware_roofline_spec()
    lines = [
        f"# Pretuned Kernels Roofline Report — {hw_name} (`{compute}`)",
        "",
        f"- **Compute Units**: {hw_spec.num_compute_units} Xe-cores",
        f"- **Peak Matrix (XMX FP16/BF16)**: {hw_spec.matrix_tflops():.1f} TFLOP/s",
        f"- **Peak Vector (FP32)**: {hw_spec.vector_tflops():.2f} TFLOP/s",
        f"- **Peak Memory Bandwidth**: {hw_spec.peak_memory_bandwidth_gbps:.1f} GB/s",
        "",
        "| Kernel | Wins/Total | Geomean Speedup | Mean SOL% | Min–Max SOL% | Mean CfgSOL% | Peak TFLOP/s | Peak Algo GB/s | Traffic Amp | Dominant Bottleneck |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |",
    ]
    for rec in records:
        if "skipped" in rec:
            continue
        kname = str(rec["kernel"])
        if "error" in rec:
            lines.append(
                f"| `{kname}` | ERROR | - | - | - | - | - | - | - | `{rec['error']}` |"
            )
            continue
        wins = f"{rec.get('helion_wins', 0)}/{rec.get('total', 0)}"
        gm = f"{float(rec.get('geomean', 0.0)):.3f}x"
        rf = rec.get("roofline_summary")
        if isinstance(rf, dict):
            mean_sol = f"{float(rf.get('mean_sol_pct', 0.0)):.1f}%"
            minmax_sol = (
                f"{float(rf.get('min_sol_pct', 0.0)):.1f}%–"
                f"{float(rf.get('max_sol_pct', 0.0)):.1f}%"
            )
            cfg_sol = f"{float(rf.get('mean_config_sol_pct', 0.0)):.1f}%"
            peak_tf = f"{float(rf.get('peak_achieved_tflops', 0.0)):.2f}"
            peak_bw = f"{float(rf.get('peak_achieved_algo_gbps', 0.0)):.1f}"
            amp = f"{float(rf.get('mean_traffic_amplification', 1.0)):.2f}x"
            bneck = f"`{rf.get('dominant_bottleneck', 'unknown')}`"
        else:
            mean_sol = minmax_sol = cfg_sol = peak_tf = peak_bw = amp = bneck = "N/A"
        lines.append(
            f"| `{kname}` | {wins} | {gm} | {mean_sol} | {minmax_sol} | "
            f"{cfg_sol} | {peak_tf} | {peak_bw} | {amp} | {bneck} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run pretuned kernels with Helion Roofline analysis."
    )
    parser.add_argument(
        "--output",
        default="pretuned-roofline.json",
        help="Path to write JSON results (default: pretuned-roofline.json).",
    )
    parser.add_argument(
        "--markdown",
        default="",
        help="Optional path to write Markdown summary table.",
    )
    parser.add_argument(
        "--kernels",
        default=",".join(KERNELS),
        help="Comma-separated kernel names to run (default: all).",
    )
    parser.add_argument(
        "--hardware",
        default="",
        help="Hardware alias to match against each kernel's checked-in heuristic files.",
    )
    parser.add_argument(
        "--max-shapes",
        type=int,
        default=0,
        help="Optional limit on number of shapes per kernel (0 = all shapes).",
    )
    parser.add_argument(
        "--no-isolate",
        action="store_true",
        help="Run all selected kernels in a single process instead of isolated subprocesses.",
    )
    parser.add_argument(
        "--allow-xpu-graph",
        action="store_true",
        help="Allow bench_cudagraph on XPU (disabled by default due to Level Zero 256MB cache-clear graph issues).",
    )
    args = parser.parse_args()

    hardware_info = get_hardware_info()
    hardware_name = hardware_info.hardware_name
    compute = hardware_info.compute_capability
    current_hardware = args.hardware or _HARDWARE_BY_COMPUTE.get(compute, compute)
    print(f"GPU: {hardware_name} ({compute}); hardware={current_hardware}")

    kernels = parse_kernel_names(args.kernels)
    isolate = not args.no_isolate and len(kernels) > 1
    records: list[dict[str, object]] = []
    for name in kernels:
        print(f"\n{'=' * 80}\nBenchmarking + Roofline: {name}\n{'=' * 80}", flush=True)
        if isolate:
            record = _run_kernel_subprocess(
                name,
                current_hardware,
                max_shapes=args.max_shapes,
                allow_xpu_graph=args.allow_xpu_graph,
            )
        else:
            record = run_kernel_with_roofline(
                name,
                current_hardware,
                max_shapes=args.max_shapes,
                allow_xpu_graph=args.allow_xpu_graph,
            )
        record["device"] = hardware_name
        record["compute_capability"] = compute
        records.append(record)

    if not args.no_isolate or len(kernels) > 1:
        md_report = render_markdown_report(records, hardware_name, compute)
        print(f"\n{'=' * 80}\n{md_report}\n{'=' * 80}")
    else:
        md_report = render_markdown_report(records, hardware_name, compute)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(records, indent=2) + "\n")
    print(f"Wrote JSON report to {out_path}")

    if args.markdown:
        md_path = Path(args.markdown)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(md_report + "\n")
        print(f"Wrote Markdown report to {md_path}")


if __name__ == "__main__":
    main()

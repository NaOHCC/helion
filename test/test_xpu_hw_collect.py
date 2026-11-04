from __future__ import annotations

import importlib.util
from pathlib import Path

import pretuned_kernels.xpu_hw_collect as xhc


def test_parse_timeline_extracts_launch_and_properties() -> None:
    sample_timeline = """
=== Device Timing Summary ===

                                    Kernel,        Calls,            Time (ns),     Time (%),         Average (ns),             Min (ns),             Max (ns)
"_helion_rms_norm_dynamic_per_token_quant[SIMD32 {2048; 1; 1} {32; 1; 1}]",           15,              1050080,    90.000000,                70005,                60160,                89280
"at::native::xpu::FillFunctor<float>[SIMD32 {2; 1; 1} {256; 1; 1}]",            1,                 6080,    10.000000,                 6080,                 6080,                 6080

=== Kernel Properties ===

                                    Kernel, Compiled, SIMD, Number of Arguments, SLM Per Work Group, Private Memory Per Thread, Spill Memory Per Thread, Register File Size Per Thread
"_helion_rms_norm_dynamic_per_token_quant[SIMD32 {2048; 1; 1} {32; 1; 1}]",      AOT,   32,                  12,               4096,                         0,                       0,                           128
"at::native::xpu::FillFunctor<float>[SIMD32 {2; 1; 1} {256; 1; 1}]",      AOT,   32,                   3,                  0,                         0,                       0,                           128
"""
    kernels = xhc.parse_timeline(
        sample_timeline, "_helion_rms_norm_dynamic_per_token_quant"
    )
    assert len(kernels) == 1
    k = kernels[0]
    assert k["base"] == "_helion_rms_norm_dynamic_per_token_quant"
    assert k["calls"] == 15
    assert k["simd"] == 32
    assert k["groups"] == [2048, 1, 1]
    assert k["local"] == [32, 1, 1]
    assert k["slm_bytes"] == 4096
    assert k["grf"] == 128
    assert k["spill_bytes"] == 0


def test_parse_unitrace_metrics_v241_and_legacy_layouts() -> None:
    v241_log = """
PROFILE_RUN_DONE

=== Device #0 Metrics ===

Kernel, GlobalInstanceId
"_helion_test_kernel<float, 2>[SIMD32 {256; 1; 1} {128; 1; 1}]", 1
GpuTime[ns], GPU_MEMORY_BYTE_READ[bytes], GPU_MEMORY_BYTE_WRITE[bytes], MidQueryCtxSwitch
50000, 1000000, 500000, 0

Kernel, GlobalInstanceId
"_helion_test_kernel<float, 2>[SIMD32 {256; 1; 1} {128; 1; 1}]", 2
GpuTime[ns], GPU_MEMORY_BYTE_READ[bytes], GPU_MEMORY_BYTE_WRITE[bytes], MidQueryCtxSwitch
20000, 2000000, 1000000, 0

Kernel, GlobalInstanceId
"_helion_test_kernel<float, 2>[SIMD32 {256; 1; 1} {128; 1; 1}]", 3
GpuTime[ns], GPU_MEMORY_BYTE_READ[bytes], GPU_MEMORY_BYTE_WRITE[bytes], MidQueryCtxSwitch
30000, 9999999, 9999999, 1

Kernel, GlobalInstanceId
"_helion_test_kernel<float, 2>[SIMD32 {256; 1; 1} {128; 1; 1}]", 4
GpuTime[ns], GPU_MEMORY_BYTE_READ[bytes], GPU_MEMORY_BYTE_WRITE[bytes], MidQueryCtxSwitch
22000, 2200000, 1100000, 0
"""
    hdr, rows = xhc.parse_unitrace_metrics(v241_log)
    assert "GpuTime[ns]" in hdr
    assert len(rows) == 4
    assert (
        rows[0]["Kernel"]
        == "_helion_test_kernel<float, 2>[SIMD32 {256; 1; 1} {128; 1; 1}]"
    )

    # Also verify legacy single-header CSV layout
    legacy_log = """
=== Device #0 Metrics ===
Kernel,GlobalInstanceId,GpuTime[ns],GPU_MEMORY_BYTE_READ[bytes]
"_helion_legacy<int, 3>[SIMD16 {64; 1; 1} {64; 1; 1}]",1,10000,4096
"_helion_legacy<int, 3>[SIMD16 {64; 1; 1} {64; 1; 1}]",2,12000,8192
"""
    hdr_leg, rows_leg = xhc.parse_unitrace_metrics(legacy_log)
    assert len(rows_leg) == 2
    summary_leg = xhc.compute_median_summary(xhc.skip_warmup(rows_leg, 1), hdr_leg)
    assert summary_leg["GpuTime[ns]"] == 12000.0
    assert summary_leg["GPU_MEMORY_BYTE_READ[bytes]"] == 8192.0


def test_occupancy_and_memory_pipe_derivation() -> None:
    topology = {
        "hardware_name": "Intel(R) Data Center GPU Max 1550",
        "compute_capability": "Xe",
        "xe_cores": 64,
        "num_xve": 512,
        "threads_per_core": 64,
        "max_wgs_per_core": 64,
        "slm_per_core_kb": 128,
        "llc_bytes": 192 << 20,
        "clock_ghz": 1.60,
        "peak_memory_bw_gbps": 1638.4,
    }
    kernel_info = {
        "name": "_helion_rms_norm[SIMD32 {2048; 1; 1} {32; 1; 1}]",
        "base": "_helion_rms_norm",
        "simd": 32,
        "groups": [2048, 1, 1],
        "local": [32, 1, 1],
        "grf": 128,
        "slm_bytes": 0,
    }
    occ = xhc.compute_occupancy(kernel_info, topology)
    assert occ is not None
    assert occ["active_xe_cores"] == 64
    assert occ["resource_limits"]["threads_per_wg"] == 1
    assert occ["occupancy"] == 0.5  # 32 WGs/core launched out of 64 threads/core
    assert occ["device_occupancy"] == 0.5

    counters = {
        "ComputeBasic": {
            "GpuTime[ns]": 65000.0,
            "AvgGpuCoreFrequencyMHz[MHz]": 1600.0,
            "GPU_MEMORY_BYTE_READ[bytes]": 16793600.0,
            "GPU_MEMORY_BYTE_WRITE[bytes]": 8396800.0,
            "L3_BYTE_READ[bytes]": 17892352.0,
            "L3_BYTE_WRITE[bytes]": 8539520.0,
            "XVE_ACTIVE[%]": 55.3,
            "XVE_STALL[%]": 24.5,
            "XVE_THREADS_OCCUPANCY_ALL[%]": 40.2,
            "XVE_INST_EXECUTED_ALU0_ALL[events]": 17790976.0,
            "XVE_INST_EXECUTED_ALU1_ALL[events]": 19947520.0,
            "XVE_INST_EXECUTED_XMX_ALL[events]": 0.0,
            "XVE_INST_EXECUTED_SEND_ALL[events]": 935936.0,
            "XVE_INST_EXECUTED_CONTROL_ALL[events]": 40960.0,
            "XVE_INST_EXECUTED_ALU0_ALL_UTILIZATION[%]": 33.2,
            "XVE_INST_EXECUTED_ALU1_ALL_UTILIZATION[%]": 37.3,
        }
    }
    static_roofline = {
        "algorithmic": {"read_bytes": 16785408, "write_bytes": 8396800},
        "config_traffic": {
            "read_bytes": 50356224,
            "write_bytes": 8396800,
            "traffic_amplification": 2.33,
        },
        "empirical": {
            "sol_pct": 25.0,
            "config_sol_pct": 58.0,
            "achieved_algo_gbps": 400.0,
            "achieved_config_gbps": 930.0,
        },
        "bottleneck": "tile_traffic_bound",
    }
    s09, s10 = xhc.derive_memory_and_pipe_metrics(
        counters,
        static_roofline=static_roofline,
        topology=topology,
        runner_time_us=62.0,
    )
    assert s09["read_amplification"] is not None
    assert abs(s09["read_amplification"] - 1.0005) < 0.01
    assert s09["measured_over_config_predicted"] is not None
    assert s09["measured_over_config_predicted"] < 0.5
    assert s10["dominant_pipe"] == "ALU1"
    assert s10["T_instruction_ms"] is not None and s10["T_instruction_ms"] > 0


def test_asm_census_and_triton_line_map() -> None:
    sample_asm = """
L0:
/* #0 [0000] */ (W) send.ugm (1|M0) r1 r127 null:0 0xFF000000 0x6228E500 {A@1,$0} // wr:1+0, rd:2; load.ugm.d32x32t.a32.ca.ca.bti[255]
/* #1 [0010] */ send.ugm (32|M0) r36 r16 null:0 0x0 0x08803580 {A@1,$5} // wr:2+0, rd:2; load.ugm.d32x2.a64.ca.ca
/* #2 [0020] */ math.inv (16|M0) r10.0<1>:f r8.0<8;8,1>:f {Compacted}
/* #3 [0028] */ send.ugm (32|M0) null r20 r36:2 0x0 0x08043584 {$6} // wr:2+2, rd:0; store.ugm.d8u32.a64.wb.wb
"""
    lines = xhc.parse_asm_lines(sample_asm)
    assert len(lines) == 4
    census = xhc.compute_asm_census(lines)
    assert census["load_ugm"] == 2
    assert census["store_ugm"] == 1
    assert census["math"] == 1
    assert census["math_inv"] == 1
    sends = xhc.extract_send_kinds(sample_asm)
    assert "load.ugm.d32x2.a64.ca.ca" in sends
    assert "store.ugm.d8u32.a64.wb.wb" in sends

    sample_triton = """@triton.jit
def _helion_foo(x, y):
    # src[foo.py:12]: val = x[tile]
    load = tl.load(x)
    # src[foo.py:13]: y[tile] = val
    tl.store(y, load)
"""
    tmap = xhc.build_triton_line_map(sample_triton)
    assert tmap[4]["triton_line"] == "load = tl.load(x)"
    assert tmap[4]["helion_src"] == "src[foo.py:12]: val = x[tile]"
    assert tmap[6]["helion_src"] == "src[foo.py:13]: y[tile] = val"


def test_xpu_hillclimb_check_goal_gate(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parent.parent
    script_path = (
        repo_root / ".claude" / "skills" / "xpu-hillclimb" / "scripts" / "check_goal.py"
    )
    spec = importlib.util.spec_from_file_location("xpu_check_goal", script_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    art_dir = tmp_path / "artifacts" / "run1"
    art_dir.mkdir(parents=True)
    art_file = art_dir / "res.json"
    art_file.write_text("{}")

    goal_pass = {
        "task": "rms_norm_dynamic_per_token_quant",
        "status": "running",
        "criteria": {"geomean_min": 1.0, "sol_pct_min": 15.0},
        "shapes": [
            {
                "name": "5120x2048",
                "ratio": 1.05,
                "sol_pct": 22.0,
                "artifact": "artifacts/run1/res.json",
            },
            {
                "name": "4096x1024",
                "ratio": 0.995,
                "sol_pct": 18.5,
                "artifact": "artifacts/run1/res.json",
            },
        ],
    }
    met, _lines, fails = mod.evaluate(goal_pass, str(tmp_path))
    assert met is True
    assert not fails

    goal_fail_sol = {
        "task": "rms_norm_dynamic_per_token_quant",
        "status": "running",
        "criteria": {"geomean_min": 1.0, "sol_pct_min": 25.0},
        "shapes": [
            {
                "name": "5120x2048",
                "ratio": 1.05,
                "sol_pct": 20.0,
                "artifact": "artifacts/run1/res.json",
            }
        ],
    }
    met2, _lines2, fails2 = mod.evaluate(goal_fail_sol, str(tmp_path))
    assert met2 is False
    assert any("sol_pct" in f for f in fails2)

    goal_path = tmp_path / "artifacts" / "goal.json"
    goal_path.write_text(
        '{"task": "t", "status": "paused", "paused_reason": "user requested pause", "shapes": []}'
    )
    assert mod._gate_verdict(str(goal_path), str(tmp_path)) == 0

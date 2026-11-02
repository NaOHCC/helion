from __future__ import annotations

import os
from unittest.mock import patch

import torch

import helion
from helion._hardware import HardwareInfo
from helion._hardware import HardwareRooflineSpec
from helion._hardware import get_hardware_roofline_spec
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import onlyBackends
from helion.autotuner.metrics import AutotuneMetrics
from helion.autotuner.metrics import register_post_autotune_hook
from helion.autotuner.metrics import remove_post_autotune_hook
import helion.language as hl


@helion.kernel(autotune_effort="none")
def _pointwise_add_exp(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(out.size()):
        out[tile] = torch.exp(x[tile] + y[tile])
    return out


@helion.kernel(autotune_effort="none")
def _matmul_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
        out[tile_m, tile_n] = acc.to(x.dtype)
    return out


@helion.kernel(autotune_effort="none")
def _row_softmax_kernel(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        row = x[tile_m, :].to(torch.float32)
        row_max = torch.amax(row, dim=-1, keepdim=True)
        shifted = torch.exp(row - row_max)
        denom = torch.sum(shifted, dim=-1, keepdim=True)
        out[tile_m, :] = (shifted / denom).to(x.dtype)
    return out


class TestXpuRooflineHardwareSpecs(TestCase):
    def test_pvc_max_1550_2stack_and_1stack_scaling(self) -> None:
        pvc_info = HardwareInfo(
            device_kind="xpu",
            hardware_name="Intel(R) Data Center GPU Max 1550",
            runtime_version="1.3",
            compute_capability="Xe",
        )
        spec_2stack = get_hardware_roofline_spec(
            hardware_info=pvc_info,
            num_compute_units=128,
        )
        self.assertEqual(spec_2stack.num_compute_units, 128)
        self.assertAlmostEqual(spec_2stack.matrix_tflops(torch.float16), 839.0)
        self.assertAlmostEqual(spec_2stack.matrix_tflops(torch.bfloat16), 839.0)
        self.assertAlmostEqual(spec_2stack.vector_tflops(torch.float32), 52.4)
        self.assertAlmostEqual(spec_2stack.peak_memory_bandwidth_gbps, 3276.8)

        # Single tile/stack PVC exposes 64 Xe-cores (subslices) -> half compute & BW
        spec_1stack = get_hardware_roofline_spec(
            hardware_info=pvc_info,
            num_compute_units=64,
        )
        self.assertEqual(spec_1stack.num_compute_units, 64)
        self.assertAlmostEqual(spec_1stack.matrix_tflops(torch.float16), 419.5)
        self.assertAlmostEqual(spec_1stack.vector_tflops(torch.float32), 26.2)
        self.assertAlmostEqual(spec_1stack.peak_memory_bandwidth_gbps, 1638.4)

    def test_pvc_max_1100_and_bmg_b580_specs(self) -> None:
        pvc_1100_info = HardwareInfo(
            device_kind="xpu",
            hardware_name="Intel(R) Data Center GPU Max 1100",
            runtime_version="1.3",
            compute_capability="Xe",
        )
        spec_1100 = get_hardware_roofline_spec(hardware_info=pvc_1100_info)
        self.assertEqual(spec_1100.num_compute_units, 56)
        self.assertAlmostEqual(spec_1100.matrix_tflops(torch.float16), 311.0)
        self.assertAlmostEqual(spec_1100.peak_memory_bandwidth_gbps, 1228.8)

        bmg_info = HardwareInfo(
            device_kind="xpu",
            hardware_name="Intel(R) Arc(TM) B580 Graphics",
            runtime_version="1.6",
            compute_capability="Xe2",
        )
        spec_bmg = get_hardware_roofline_spec(hardware_info=bmg_info)
        self.assertEqual(spec_bmg.num_compute_units, 20)
        self.assertAlmostEqual(spec_bmg.matrix_tflops(torch.float16), 117.0)
        self.assertAlmostEqual(spec_bmg.vector_tflops(torch.float32), 14.6)
        self.assertAlmostEqual(spec_bmg.peak_memory_bandwidth_gbps, 456.0)
        self.assertAlmostEqual(
            spec_bmg.ridge_point(use_matrix_engine=True, dtype=torch.float16),
            (117.0 * 1e3) / 456.0,
        )

    def test_env_overrides(self) -> None:
        bmg_info = HardwareInfo(
            device_kind="xpu",
            hardware_name="Intel(R) Arc(TM) B580 Graphics",
            runtime_version="1.6",
            compute_capability="Xe2",
        )
        with patch.dict(
            os.environ,
            {
                "HELION_ROOFLINE_PEAK_TFLOPS": "200.0",
                "HELION_ROOFLINE_PEAK_VECTOR_TFLOPS": "25.0",
                "HELION_ROOFLINE_PEAK_BW_GBPS": "500.0",
                "HELION_ROOFLINE_XE_CORES": "24",
            },
        ):
            spec = get_hardware_roofline_spec(hardware_info=bmg_info)
            self.assertEqual(spec.num_compute_units, 24)
            self.assertAlmostEqual(spec.matrix_tflops(torch.float16), 200.0)
            self.assertAlmostEqual(spec.vector_tflops(torch.float32), 25.0)
            self.assertAlmostEqual(spec.peak_memory_bandwidth_gbps, 500.0)


@onlyBackends(["triton"])
class TestKernelRooflineAnalysis(TestCase):
    def _pvc_1stack_spec(self) -> HardwareRooflineSpec:
        pvc_info = HardwareInfo(
            device_kind="xpu",
            hardware_name="Intel(R) Data Center GPU Max 1550",
            runtime_version="1.3",
            compute_capability="Xe",
        )
        return get_hardware_roofline_spec(
            hardware_info=pvc_info,
            num_compute_units=64,
        )

    def test_pointwise_roofline_static_and_sol(self) -> None:
        m, n = 1024, 1024
        numel = m * n
        x = torch.randn(m, n, device=DEVICE, dtype=torch.float32)
        y = torch.randn(m, n, device=DEVICE, dtype=torch.float32)
        bound = _pointwise_add_exp.bind((x, y))
        config = helion.Config(block_sizes=[32, 32], num_warps=4)

        report = bound.roofline(
            config,
            measure=False,
            latency_ms=0.015,
            hardware_spec=self._pvc_1stack_spec(),
        )
        # 2 ops per element: 1 add + 1 exp (SFU)
        self.assertEqual(report.matmul_flops, 0)
        self.assertEqual(report.vector_flops, 2 * numel)
        self.assertEqual(report.sfu_flops, numel)
        self.assertEqual(report.total_flops, 2 * numel)
        # 2 float32 reads + 1 float32 write = 12 bytes/element
        self.assertEqual(report.algorithmic_read_bytes, 2 * numel * 4)
        self.assertEqual(report.algorithmic_write_bytes, numel * 4)
        self.assertEqual(report.algorithmic_bytes, 3 * numel * 4)
        self.assertAlmostEqual(report.traffic_amplification, 1.0)
        self.assertEqual(report.bottleneck, "memory_bound")
        self.assertIsNotNone(report.sol_pct)
        assert report.sol_pct is not None
        self.assertGreater(report.sol_pct, 0.0)
        summary = report.summary()
        self.assertIn("Roofline Report", summary)
        self.assertIn("memory_bound", summary)

    def test_matmul_roofline_traffic_amplification_and_l2_grouping(self) -> None:
        m, k, n = 2048, 2048, 2048
        x = torch.randn(m, k, device=DEVICE, dtype=torch.float16)
        y = torch.randn(k, n, device=DEVICE, dtype=torch.float16)
        bound = _matmul_kernel.bind((x, y))

        # Large tile: [256, 256, 32] -> 8x8=64 workgroups (1 wave on 64 Xe-cores)
        cfg_large = helion.Config(
            block_sizes=[256, 256, 32],
            num_warps=32,
            num_stages=2,
            l2_groupings=[4],
        )
        rep_large = bound.roofline(
            cfg_large,
            measure=False,
            latency_ms=0.08,
            hardware_spec=self._pvc_1stack_spec(),
        )
        self.assertEqual(rep_large.matmul_flops, 2 * m * n * k)
        expected_algo_read = (m * k + k * n) * 2
        expected_algo_write = m * n * 2
        self.assertEqual(rep_large.algorithmic_read_bytes, expected_algo_read)
        self.assertEqual(rep_large.algorithmic_write_bytes, expected_algo_write)
        # A is read (N / 256) = 8 times, B is read (M / 256) = 8 times
        expected_cfg_read = (m * k * 8 + k * n * 8) * 2
        self.assertEqual(rep_large.config_read_bytes, expected_cfg_read)
        self.assertLess(rep_large.l2_estimated_bytes, rep_large.config_bytes)
        self.assertEqual(rep_large.num_workgroups, 64)
        self.assertAlmostEqual(rep_large.waves_per_compute_unit, 1.0)
        self.assertEqual(rep_large.bottleneck, "compute_bound")
        # Should include XPU grf_mode / tensor_descriptor hints when not set
        self.assertTrue(any("grf_mode='256'" in h for h in rep_large.hints))
        self.assertTrue(any("tensor_descriptor" in h for h in rep_large.hints))

        # Tiny tile: [16, 16, 16] -> massive traffic amplification (128x re-reads)
        cfg_small = helion.Config(
            block_sizes=[16, 16, 16],
            num_warps=4,
            num_stages=2,
        )
        rep_small = bound.roofline(
            cfg_small,
            measure=False,
            hardware_spec=self._pvc_1stack_spec(),
        )
        self.assertGreater(
            rep_small.traffic_amplification, rep_large.traffic_amplification
        )
        self.assertEqual(rep_small.bottleneck, "tile_traffic_bound")

    def test_reduction_persistent_vs_looped_traffic(self) -> None:
        m, n = 512, 2048
        x = torch.randn(m, n, device=DEVICE, dtype=torch.float16)
        bound = _row_softmax_kernel.bind((x,))

        cfg_persistent = helion.Config(
            block_sizes=[1],
            reduction_loops=[None],
            num_warps=4,
        )
        rep_pers = bound.roofline(
            cfg_persistent,
            measure=False,
            hardware_spec=self._pvc_1stack_spec(),
        )
        self.assertAlmostEqual(rep_pers.traffic_amplification, 1.0)

        cfg_looped = helion.Config(
            block_sizes=[1],
            reduction_loops=[256],
            num_warps=4,
        )
        rep_looped = bound.roofline(
            cfg_looped,
            measure=False,
            hardware_spec=self._pvc_1stack_spec(),
        )
        # Looped softmax without reduction_fusion reads x in 2 reduction loops
        self.assertGreater(rep_looped.config_read_bytes, rep_pers.config_read_bytes)
        self.assertAlmostEqual(rep_looped.traffic_amplification, 1.5)
        self.assertTrue(any("reduction" in h for h in rep_looped.hints))

    def test_low_occupancy_and_empirical_measure_with_do_bench(self) -> None:
        m, n = 64, 64
        x = torch.randn(m, n, device=DEVICE, dtype=torch.float32)
        y = torch.randn(m, n, device=DEVICE, dtype=torch.float32)
        bound = _pointwise_add_exp.bind((x, y))
        cfg = helion.Config(block_sizes=[32, 32], num_warps=4)

        with (
            patch.object(bound, "compile_config", return_value=lambda *_a: None),
            patch(
                "helion.autotuner.benchmarking.do_bench",
                return_value=0.05,
            ) as mock_bench,
        ):
            report = bound.roofline(
                cfg,
                measure=True,
                hardware_spec=self._pvc_1stack_spec(),
            )
            mock_bench.assert_called_once()
        self.assertEqual(report.num_workgroups, 4)
        self.assertLess(report.waves_per_compute_unit, 1.0)
        self.assertEqual(report.bottleneck, "occupancy_or_latency_bound")
        self.assertAlmostEqual(report.measured_ms or 0.0, 0.05)

    def test_autotune_metrics_serialization(self) -> None:
        m, n = 256, 256
        x = torch.randn(m, n, device=DEVICE, dtype=torch.float32)
        y = torch.randn(m, n, device=DEVICE, dtype=torch.float32)
        bound = _pointwise_add_exp.bind((x, y))
        report = bound.roofline(
            helion.Config(block_sizes=[32, 32]),
            measure=False,
            latency_ms=0.01,
            hardware_spec=self._pvc_1stack_spec(),
        )
        metrics = AutotuneMetrics(
            kernel_name="_pointwise_add_exp",
            best_perf_ms=0.01,
            roofline=report.to_dict(),
        )
        as_dict = metrics.to_dict()
        self.assertIn("roofline", as_dict)
        roofline_dict = as_dict["roofline"]
        assert isinstance(roofline_dict, dict)
        self.assertEqual(roofline_dict["bottleneck"], "memory_bound")
        self.assertIn("workload", roofline_dict)
        self.assertIn("config_traffic", roofline_dict)
        self.assertIn("empirical", roofline_dict)

        captured: list[AutotuneMetrics] = []
        register_post_autotune_hook(captured.append)
        try:
            self.assertEqual(len(captured), 0)
        finally:
            remove_post_autotune_hook(captured.append)

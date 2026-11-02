from __future__ import annotations

from unittest.mock import patch

import torch

import helion
from helion._compiler.compile_environment import CompileEnvironment
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import is_xpu
from helion._testing import onlyBackends
from helion._testing import skipIfFn
import helion.language as hl


@helion.kernel(
    autotune_effort="none",
    config=helion.Config(block_sizes=[32, 32], grf_mode="256"),
)
def _add_large_grf(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(out.size()):
        out[tile] = x[tile] + y[tile]
    return out


@helion.kernel(autotune_effort="none")
def _matmul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
        out[tile_m, tile_n] = acc.to(x.dtype)
    return out


@onlyBackends(["triton"])
class TestXPUTunables(TestCase):
    @skipIfFn(lambda: not is_xpu(), "Test requires Intel XPU")
    def test_xpu_matmul_seeds(self) -> None:
        x = torch.randn(2048, 2048, device=DEVICE, dtype=torch.float16)
        y = torch.randn(2048, 2048, device=DEVICE, dtype=torch.float16)
        spec = _matmul.bind((x, y)).config_spec
        self.assertIn("triton_xpu_matmul", spec.autotuner_heuristics)
        seeds = [
            seed
            for seed in spec.compiler_seed_configs
            if seed.config.get("num_warps") == 32
        ]
        self.assertTrue(seeds)
        for seed in seeds:
            result = _matmul.bind((x, y)).compile_config(seed)(x, y)
            torch.testing.assert_close(result, x @ y, atol=1e-1, rtol=1e-2)

    @skipIfFn(lambda: not is_xpu(), "Test requires Intel XPU")
    def test_grf_mode_in_kernel(self) -> None:
        x = torch.randn(128, 128, device=DEVICE)
        y = torch.randn(128, 128, device=DEVICE)
        code, result = code_and_output(_add_large_grf, (x, y))
        torch.testing.assert_close(result, x + y)
        self.assertIn("grf_mode='256'", code)

    def test_grf_mode_fragment(self) -> None:
        with patch("helion._compat.supports_intel_xpu_tunables", return_value=True):
            env = CompileEnvironment(DEVICE, helion.Settings(backend="triton"))
            fragments = env.config_spec.backend_tunable_fragments
            self.assertEqual(fragments["grf_mode"].choices, ("default", "256"))

            config = helion.Config(grf_mode="256")
            env.config_spec.normalize(config)
            self.assertEqual(config["grf_mode"], "256")

            config = helion.Config()
            env.config_spec.normalize(config)
            self.assertEqual(config["grf_mode"], "default")

    def test_grf_mode_rejected_without_xpu(self) -> None:
        with patch("helion._compat.supports_intel_xpu_tunables", return_value=False):
            env = CompileEnvironment(DEVICE, helion.Settings(backend="triton"))
            self.assertNotIn("grf_mode", env.config_spec.backend_tunable_fragments)
            with self.assertRaisesRegex(
                helion.exc.InvalidConfig,
                rf"Unsupported config keys for backend '{env.backend_name}': \['grf_mode'\]",
            ):
                env.config_spec.normalize(helion.Config(grf_mode="256"))

    def test_grf_mode_validation(self) -> None:
        with patch("helion._compat.supports_intel_xpu_tunables", return_value=True):
            env = CompileEnvironment(DEVICE, helion.Settings(backend="triton"))
            with self.assertRaisesRegex(helion.exc.InvalidConfig, "grf_mode"):
                env.config_spec.normalize(helion.Config(grf_mode="64"))
            with self.assertRaisesRegex(helion.exc.InvalidConfig, "num_warps > 32"):
                env.config_spec.normalize(helion.Config(grf_mode="256", num_warps=64))

            config = helion.Config(grf_mode="256", num_warps=64)
            env.config_spec.normalize(config, _fix_invalid=True)
            self.assertEqual(config["grf_mode"], "default")

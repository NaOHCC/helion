from __future__ import annotations

import dataclasses
import functools
import os

import torch

# Linear compute-capability fallback lists (newest to oldest). ``sm103`` is not
# a safe generic fallback for arbitrary future CUDA architectures, so it stays
# outside this chain: an exact sm103 target still falls back to sm100 through
# the unknown-current path, without offering sm103 artifacts to future targets.
_CUDA_COMPUTE_CAPS: list[str] = [
    "sm100",
    "sm90",
    "sm89",
    "sm87",
    "sm86",
    "sm80",
    "sm75",
    "sm72",
    "sm70",
]

_ROCM_ARCHS: list[str] = [
    "gfx950",
    "gfx942",
    "gfx941",
    "gfx940",
    "gfx90a",
    "gfx908",
    "gfx906",
    "gfx900",
]


@dataclasses.dataclass(frozen=True)
class HardwareInfo:
    """
    Hardware information for cache keys and heuristic selection.

    Attributes:
        device_kind: Device type ('cuda', 'rocm', 'xpu')
        hardware_name: Device name (e.g., 'NVIDIA H100', 'gfx90a')
        runtime_version: Runtime version (e.g., '12.4', 'gfx90a')
        compute_capability: Compute capability for heuristics (e.g., 'sm90', 'gfx90a')
    """

    device_kind: str
    hardware_name: str
    runtime_version: str
    compute_capability: str

    @property
    def hardware_id(self) -> str:
        """Get a unique identifier string for this hardware."""
        safe_name = self.hardware_name.replace(" ", "_")
        return f"{self.device_kind}_{safe_name}_{self.runtime_version}"

    def get_compatible_compute_ids(self) -> list[str]:
        """
        Get a list of compatible compute IDs for fallback, ordered from current to oldest.

        For CUDA/ROCm, returns the current compute capability followed by all older
        compatible architectures. This allows using heuristics tuned on older hardware
        when newer hardware-specific heuristics aren't available.
        """
        if self.device_kind == "cuda":
            arch_list = _CUDA_COMPUTE_CAPS
        elif self.device_kind == "rocm":
            arch_list = _ROCM_ARCHS
        else:
            return [self.compute_capability]

        try:
            current_idx = arch_list.index(self.compute_capability)
            return arch_list[current_idx:]
        except ValueError:
            return [self.compute_capability, *arch_list]


@functools.cache
def get_hardware_info(device: torch.device | None = None) -> HardwareInfo:
    """
    Get hardware information for the current or specified device.

    Args:
        device: Optional device to get info for. If None, uses first available GPU or TPU.

    Returns:
        HardwareInfo with device details for caching and heuristic lookup.
    """
    if device is None:
        # CUDA/ROCm path
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        # XPU (Intel) path
        elif torch.xpu.is_available():
            device = torch.device("xpu:0")

    if device is not None:
        if device.type == "cuda":
            props = torch.cuda.get_device_properties(device)

            if torch.version.cuda is not None:
                return HardwareInfo(
                    device_kind="cuda",
                    hardware_name=props.name,
                    runtime_version=str(torch.version.cuda),
                    compute_capability=f"sm{props.major}{props.minor}",
                )

            if torch.version.hip is not None:
                return HardwareInfo(
                    device_kind="rocm",
                    hardware_name=props.gcnArchName,
                    runtime_version=torch.version.hip,
                    compute_capability=props.gcnArchName,
                )
        if device.type == "xpu":
            import inspect

            from torch.testing._internal.common_xpu import get_xpu_arch

            props = torch.xpu.get_device_properties(device)
            # Older torch builds expose get_xpu_arch() without a device argument.
            arch = (
                get_xpu_arch(device)
                if inspect.signature(get_xpu_arch).parameters
                else get_xpu_arch()
            )
            return HardwareInfo(
                device_kind="xpu",
                hardware_name=props.name,
                runtime_version=props.driver_version,
                compute_capability=arch.name,
            )

    # TPU / Pallas path
    try:
        import jax

        tpu_devices = [d for d in jax.devices() if d.platform == "tpu"]
        if tpu_devices:
            first_tpu = tpu_devices[0]
            return HardwareInfo(
                device_kind="tpu",
                hardware_name=first_tpu.device_kind,
                runtime_version=jax.__version__,
                compute_capability=first_tpu.device_kind,
            )
    except ImportError:
        pass

    raise RuntimeError(
        "No supported GPU or TPU device found. Helion requires CUDA, ROCm, XPU, or TPU."
    )


@dataclasses.dataclass(frozen=True)
class HardwareRooflineSpec:
    """Theoretical hardware roofline ceilings for compute and memory bandwidth.

    Attributes:
        device_kind: Device family (e.g. ``'xpu'``).
        hardware_name: Human-readable device SKU name.
        compute_capability: Architecture identifier (e.g. ``'Xe'``, ``'Xe2'``).
        num_compute_units: Number of Xe-cores / subslices (or SMs).
        peak_memory_bandwidth_gbps: Peak HBM/GDDR bandwidth in GB/s (1e9 bytes/s).
        peak_matrix_tflops: XMX / DPAS matrix engine peak TFLOP/s keyed by dtype.
        peak_vector_tflops: Vector engine (FPU) peak TFLOP/s keyed by dtype.
    """

    device_kind: str
    hardware_name: str
    compute_capability: str
    num_compute_units: int
    peak_memory_bandwidth_gbps: float
    peak_matrix_tflops: dict[torch.dtype, float]
    peak_vector_tflops: dict[torch.dtype, float]

    def matrix_tflops(self, dtype: torch.dtype | None = None) -> float:
        """Return XMX/matrix engine peak TFLOP/s for ``dtype`` (defaults to FP16/BF16)."""
        if dtype is not None and dtype in self.peak_matrix_tflops:
            return self.peak_matrix_tflops[dtype]
        if torch.float16 in self.peak_matrix_tflops:
            return self.peak_matrix_tflops[torch.float16]
        if self.peak_matrix_tflops:
            return next(iter(self.peak_matrix_tflops.values()))
        return self.vector_tflops(dtype)

    def vector_tflops(self, dtype: torch.dtype | None = None) -> float:
        """Return vector engine peak TFLOP/s for ``dtype`` (defaults to FP32)."""
        if dtype is not None and dtype in self.peak_vector_tflops:
            return self.peak_vector_tflops[dtype]
        if torch.float32 in self.peak_vector_tflops:
            return self.peak_vector_tflops[torch.float32]
        if self.peak_vector_tflops:
            return next(iter(self.peak_vector_tflops.values()))
        return 1.0

    def ridge_point(
        self,
        *,
        use_matrix_engine: bool = False,
        dtype: torch.dtype | None = None,
    ) -> float:
        """Return the Roofline ridge point in FLOP/Byte."""
        peak_tflops = (
            self.matrix_tflops(dtype)
            if use_matrix_engine
            else self.vector_tflops(dtype)
        )
        return (peak_tflops * 1e3) / max(1e-9, self.peak_memory_bandwidth_gbps)


@dataclasses.dataclass(frozen=True)
class _XpuSkuBaseline:
    reference_xe_cores: int
    peak_memory_bandwidth_gbps: float
    peak_matrix_tflops: dict[torch.dtype, float]
    peak_vector_tflops: dict[torch.dtype, float]
    scale_bandwidth_with_cores: bool = True


# Reference Intel XPU SKU baselines:
# - Data Center GPU Max 1550 (Ponte Vecchio 2-stack, 128 Xe-cores; 1-stack tile exposes 64 Xe-cores)
# - Data Center GPU Max 1100 (Ponte Vecchio PCIe, 56 Xe-cores)
# - Arc B580 (Battlemage BMG-G21, 20 Xe2-cores)
_PVC_MAX_1550_BASELINE = _XpuSkuBaseline(
    reference_xe_cores=128,
    peak_memory_bandwidth_gbps=3276.8,
    peak_matrix_tflops={
        torch.float16: 839.0,
        torch.bfloat16: 839.0,
        torch.float32: 419.5,
        torch.float8_e4m3fn: 1678.0,
        torch.float8_e5m2: 1678.0,
        torch.int8: 1678.0,
        torch.float64: 52.4,
    },
    peak_vector_tflops={
        torch.float32: 52.4,
        torch.float16: 104.8,
        torch.bfloat16: 104.8,
        torch.float64: 52.4,
    },
    scale_bandwidth_with_cores=True,
)

_PVC_MAX_1100_BASELINE = _XpuSkuBaseline(
    reference_xe_cores=56,
    peak_memory_bandwidth_gbps=1228.8,
    peak_matrix_tflops={
        torch.float16: 311.0,
        torch.bfloat16: 311.0,
        torch.float32: 155.5,
        torch.float8_e4m3fn: 622.0,
        torch.float8_e5m2: 622.0,
        torch.int8: 622.0,
        torch.float64: 19.4,
    },
    peak_vector_tflops={
        torch.float32: 19.4,
        torch.float16: 38.8,
        torch.bfloat16: 38.8,
        torch.float64: 19.4,
    },
    scale_bandwidth_with_cores=True,
)

_BMG_ARC_B580_BASELINE = _XpuSkuBaseline(
    reference_xe_cores=20,
    peak_memory_bandwidth_gbps=456.0,
    peak_matrix_tflops={
        torch.float16: 117.0,
        torch.bfloat16: 117.0,
        torch.float32: 58.5,
        torch.float8_e4m3fn: 233.0,
        torch.float8_e5m2: 233.0,
        torch.int8: 233.0,
    },
    peak_vector_tflops={
        torch.float32: 14.6,
        torch.float16: 29.2,
        torch.bfloat16: 29.2,
    },
    scale_bandwidth_with_cores=False,
)


def _match_xpu_sku_baseline(
    hardware_name: str, compute_capability: str
) -> _XpuSkuBaseline:
    lowered = hardware_name.lower()
    if "1100" in lowered:
        return _PVC_MAX_1100_BASELINE
    if "1550" in lowered or "max" in lowered or "pvc" in lowered:
        return _PVC_MAX_1550_BASELINE
    if "b580" in lowered or "b570" in lowered or "bmg" in lowered or "arc b" in lowered:
        return _BMG_ARC_B580_BASELINE
    if compute_capability.lower().startswith("xe2"):
        return _BMG_ARC_B580_BASELINE
    return _PVC_MAX_1550_BASELINE


def _parse_positive_float_env(name: str) -> float | None:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return None
    value = float(raw.strip())
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {raw!r}")
    return value


def _parse_positive_int_env(name: str) -> int | None:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return None
    value = int(raw.strip())
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {raw!r}")
    return value


def get_hardware_roofline_spec(
    device: torch.device | None = None,
    *,
    hardware_info: HardwareInfo | None = None,
    num_compute_units: int | None = None,
    override: HardwareRooflineSpec | None = None,
) -> HardwareRooflineSpec:
    """Resolve the hardware roofline specification for ``device`` (focused on Intel XPU).

    Supports automatic lookup for Intel Data Center GPU Max (PVC 1550 / 1100) and
    Intel Arc B-series (BMG / B580) with subslice-count scaling (e.g. 1-stack 64 Xe-core
    vs 2-stack 128 Xe-core PVC), plus environment variable overrides:
    - ``HELION_ROOFLINE_PEAK_TFLOPS``: override matrix peak TFLOP/s
    - ``HELION_ROOFLINE_PEAK_VECTOR_TFLOPS``: override vector peak TFLOP/s
    - ``HELION_ROOFLINE_PEAK_BW_GBPS``: override memory bandwidth in GB/s
    - ``HELION_ROOFLINE_XE_CORES``: override compute unit (Xe-core) count
    """
    if override is not None:
        return override

    info = hardware_info if hardware_info is not None else get_hardware_info(device)
    baseline = _match_xpu_sku_baseline(info.hardware_name, info.compute_capability)

    resolved_cores = _parse_positive_int_env("HELION_ROOFLINE_XE_CORES")
    if resolved_cores is None:
        if num_compute_units is not None and num_compute_units > 0:
            resolved_cores = num_compute_units
        elif device is not None and device.type == "xpu" and torch.xpu.is_available():
            props = torch.xpu.get_device_properties(device)
            if props.gpu_subslice_count > 0:
                resolved_cores = int(props.gpu_subslice_count)

    if resolved_cores is None or resolved_cores <= 0:
        resolved_cores = baseline.reference_xe_cores

    core_scale = resolved_cores / max(1, baseline.reference_xe_cores)
    bw_scale = core_scale if baseline.scale_bandwidth_with_cores else 1.0

    peak_bw = (
        _parse_positive_float_env("HELION_ROOFLINE_PEAK_BW_GBPS")
        or baseline.peak_memory_bandwidth_gbps * bw_scale
    )

    matrix_override = _parse_positive_float_env("HELION_ROOFLINE_PEAK_TFLOPS")
    if matrix_override is not None:
        peak_matrix = dict.fromkeys(baseline.peak_matrix_tflops, matrix_override)
    else:
        peak_matrix = {
            dtype: tflops * core_scale
            for dtype, tflops in baseline.peak_matrix_tflops.items()
        }

    vector_override = _parse_positive_float_env("HELION_ROOFLINE_PEAK_VECTOR_TFLOPS")
    if vector_override is not None:
        peak_vector = dict.fromkeys(baseline.peak_vector_tflops, vector_override)
    else:
        peak_vector = {
            dtype: tflops * core_scale
            for dtype, tflops in baseline.peak_vector_tflops.items()
        }

    return HardwareRooflineSpec(
        device_kind=info.device_kind,
        hardware_name=info.hardware_name,
        compute_capability=info.compute_capability,
        num_compute_units=resolved_cores,
        peak_memory_bandwidth_gbps=peak_bw,
        peak_matrix_tflops=peak_matrix,
        peak_vector_tflops=peak_vector,
    )

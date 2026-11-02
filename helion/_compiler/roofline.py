from __future__ import annotations

import contextlib
import dataclasses
import math
import operator
from typing import TYPE_CHECKING
from typing import Literal

import torch

from .._hardware import HardwareRooflineSpec
from .._hardware import get_hardware_roofline_spec
from ..language import _tracing_ops
from ..language import memory_ops
from ..language import tile_ops
from ..language import view_ops
from ..language.atomic_ops import ATOMIC_OPS
from ..language.matmul_ops import MATMUL_FACT_ID_META
from .compile_environment import CompileEnvironment
from .device_ir import ForLoopGraphInfo
from .device_ir import ReductionLoopGraphInfo
from .device_ir import control_flow_parent_ids
from .device_ir_analysis import DeviceIRAnalysis
from .device_ir_analysis import _accessed_tensor_fake
from .device_ir_analysis import matmul_operand_positions
from .inductor_lowering import ReductionLowering

if TYPE_CHECKING:
    from ..autotuner.config_spec import ConfigSpec
    from ..autotuner.config_spec import MemoryOpFact
    from ..runtime.config import Config
    from .device_ir import DeviceIR
    from .host_function import HostFunction

RooflineBottleneck = Literal[
    "compute_bound",
    "memory_bound",
    "tile_traffic_bound",
    "occupancy_or_latency_bound",
]

_SFU_OP_NAMES: frozenset[str] = frozenset(
    {
        "sin",
        "cos",
        "tan",
        "tanh",
        "asin",
        "acos",
        "atan",
        "sinh",
        "cosh",
        "atanh",
        "asinh",
        "acosh",
        "exp",
        "exp2",
        "expm1",
        "log",
        "log2",
        "log10",
        "log1p",
        "sqrt",
        "rsqrt",
        "sigmoid",
        "erf",
        "erfc",
        "pow",
        "reciprocal",
    }
)

_NON_COMPUTE_TARGETS: frozenset[object] = frozenset(
    {
        memory_ops.load,
        memory_ops.store,
        memory_ops.prefetch,
        view_ops.subscript,
        view_ops.join,
        view_ops.split,
        tile_ops.tile_index,
        tile_ops.tile_id,
        tile_ops.tile_begin,
        tile_ops.tile_end,
        tile_ops.tile_block_size,
        _tracing_ops._host_tensor,
        _tracing_ops._get_symnode,
        _tracing_ops._for_loop,
        _tracing_ops._if,
        _tracing_ops._new_var,
        _tracing_ops._phi,
        _tracing_ops._constant_tensor,
        _tracing_ops._mask_to,
        operator.getitem,
        torch.ops.prims.iota.default,
        torch.ops.prims.convert_element_type.default,
        torch.ops.aten.sym_size.int,
        torch.ops.aten.full.default,
        torch.ops.aten.zeros.default,
        torch.ops.aten.ones.default,
        torch.ops.aten.empty.memory_format,
        torch.ops.aten.expand.default,
        torch.ops.aten.broadcast_tensors.default,
        torch.ops.aten.view.default,
        torch.ops.aten.reshape.default,
        torch.ops.aten.permute.default,
        torch.ops.aten.transpose.int,
        torch.ops.aten.t.default,
        torch.ops.aten.unsqueeze.default,
        torch.ops.aten.squeeze.dim,
        torch.ops.aten.squeeze.dims,
        torch.ops.aten.squeeze.default,
        torch.ops.aten.slice.Tensor,
        torch.ops.aten.to.dtype,
        torch.ops.aten.alias.default,
        torch.ops.aten.clone.default,
        torch.ops.aten.contiguous.default,
        torch.ops.aten.where.self,
    }
)


@dataclasses.dataclass(frozen=True)
class RooflineReport:
    """Static and empirical Roofline performance report for a Helion kernel config."""

    # Algorithmic (config-independent) workload metrics
    matmul_flops: int
    vector_flops: int
    sfu_flops: int
    total_flops: int
    algorithmic_read_bytes: int
    algorithmic_write_bytes: int
    algorithmic_bytes: int
    algorithmic_arithmetic_intensity: float
    dominant_matmul_dtype: torch.dtype | None
    dominant_vector_dtype: torch.dtype | None

    # Config-dependent tile traffic & parallelism metrics
    config: Config
    config_read_bytes: int
    config_write_bytes: int
    config_bytes: int
    l2_estimated_bytes: int
    config_arithmetic_intensity: float
    traffic_amplification: float
    num_workgroups: int
    waves_per_compute_unit: float

    # Hardware roofline ceilings & ideal latency bounds
    hardware: HardwareRooflineSpec
    peak_matrix_tflops: float
    peak_vector_tflops: float
    effective_peak_tflops: float
    peak_bandwidth_gbps: float
    ridge_point_flops_per_byte: float
    compute_bound_ms: float
    algo_memory_bound_ms: float
    config_memory_bound_ms: float
    ideal_ms: float
    config_ideal_ms: float

    # Empirical timing & SOL% (populated when latency_ms is measured or supplied)
    measured_ms: float | None
    achieved_tflops: float | None
    achieved_algo_gbps: float | None
    achieved_config_gbps: float | None
    sol_pct: float | None
    config_sol_pct: float | None

    # Diagnosis
    bottleneck: RooflineBottleneck
    hints: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Serialize the report to a JSON-compatible dictionary."""
        return {
            "hardware": {
                "device_kind": self.hardware.device_kind,
                "hardware_name": self.hardware.hardware_name,
                "compute_capability": self.hardware.compute_capability,
                "num_compute_units": self.hardware.num_compute_units,
                "peak_matrix_tflops": self.peak_matrix_tflops,
                "peak_vector_tflops": self.peak_vector_tflops,
                "effective_peak_tflops": self.effective_peak_tflops,
                "peak_bandwidth_gbps": self.peak_bandwidth_gbps,
                "ridge_point_flops_per_byte": self.ridge_point_flops_per_byte,
            },
            "workload": {
                "matmul_flops": self.matmul_flops,
                "vector_flops": self.vector_flops,
                "sfu_flops": self.sfu_flops,
                "total_flops": self.total_flops,
                "algorithmic_read_bytes": self.algorithmic_read_bytes,
                "algorithmic_write_bytes": self.algorithmic_write_bytes,
                "algorithmic_bytes": self.algorithmic_bytes,
                "algorithmic_arithmetic_intensity": (
                    self.algorithmic_arithmetic_intensity
                ),
                "dominant_matmul_dtype": (
                    str(self.dominant_matmul_dtype)
                    if self.dominant_matmul_dtype is not None
                    else None
                ),
                "dominant_vector_dtype": (
                    str(self.dominant_vector_dtype)
                    if self.dominant_vector_dtype is not None
                    else None
                ),
            },
            "config_traffic": {
                "config": dict(self.config),
                "config_read_bytes": self.config_read_bytes,
                "config_write_bytes": self.config_write_bytes,
                "config_bytes": self.config_bytes,
                "l2_estimated_bytes": self.l2_estimated_bytes,
                "config_arithmetic_intensity": self.config_arithmetic_intensity,
                "traffic_amplification": self.traffic_amplification,
                "num_workgroups": self.num_workgroups,
                "waves_per_compute_unit": self.waves_per_compute_unit,
            },
            "bounds_ms": {
                "compute_bound_ms": self.compute_bound_ms,
                "algo_memory_bound_ms": self.algo_memory_bound_ms,
                "config_memory_bound_ms": self.config_memory_bound_ms,
                "ideal_ms": self.ideal_ms,
                "config_ideal_ms": self.config_ideal_ms,
            },
            "empirical": {
                "measured_ms": self.measured_ms,
                "achieved_tflops": self.achieved_tflops,
                "achieved_algo_gbps": self.achieved_algo_gbps,
                "achieved_config_gbps": self.achieved_config_gbps,
                "sol_pct": self.sol_pct,
                "config_sol_pct": self.config_sol_pct,
            },
            "bottleneck": self.bottleneck,
            "hints": list(self.hints),
        }

    def summary(self) -> str:
        """Render a concise human- and agent-readable summary."""
        lines = [
            (
                f"=== Roofline Report ({self.hardware.hardware_name} "
                f"[{self.hardware.compute_capability}], "
                f"{self.hardware.num_compute_units} CUs) ==="
            ),
            (
                f"Workload   : {self.total_flops / 1e9:.3f} GFLOPs "
                f"(matmul={self.matmul_flops / 1e9:.3f}G, "
                f"vector={self.vector_flops / 1e9:.3f}G, "
                f"sfu={self.sfu_flops / 1e9:.3f}G)"
            ),
            (
                f"Algo Memory: {self.algorithmic_bytes / 1e6:.3f} MB "
                f"(read={self.algorithmic_read_bytes / 1e6:.3f} MB, "
                f"write={self.algorithmic_write_bytes / 1e6:.3f} MB) | "
                f"AI={self.algorithmic_arithmetic_intensity:.2f} FLOP/B"
            ),
            (
                f"Tile Memory: {self.config_bytes / 1e6:.3f} MB "
                f"(L2-est={self.l2_estimated_bytes / 1e6:.3f} MB, "
                f"amp={self.traffic_amplification:.2f}x) | "
                f"Config AI={self.config_arithmetic_intensity:.2f} FLOP/B"
            ),
            (
                f"Occupancy  : {self.num_workgroups} workgroups "
                f"({self.waves_per_compute_unit:.2f} waves/CU)"
            ),
            (
                f"Roofline   : Peak Compute={self.effective_peak_tflops:.1f} TFLOP/s, "
                f"Peak BW={self.peak_bandwidth_gbps:.1f} GB/s, "
                f"Ridge={self.ridge_point_flops_per_byte:.1f} FLOP/B"
            ),
            (
                f"Ideal Time : {self.ideal_ms * 1e3:.2f} us "
                f"(compute={self.compute_bound_ms * 1e3:.2f} us, "
                f"algo_mem={self.algo_memory_bound_ms * 1e3:.2f} us, "
                f"tile_mem={self.config_memory_bound_ms * 1e3:.2f} us)"
            ),
        ]
        if self.measured_ms is not None:
            lines.append(
                f"Measured   : {self.measured_ms * 1e3:.2f} us | "
                f"Achieved={self.achieved_tflops or 0.0:.2f} TFLOP/s, "
                f"Algo BW={self.achieved_algo_gbps or 0.0:.1f} GB/s "
                f"(Tile BW={self.achieved_config_gbps or 0.0:.1f} GB/s) | "
                f"SOL={self.sol_pct or 0.0:.1f}% (Config SOL={self.config_sol_pct or 0.0:.1f}%)"
            )
        lines.append(f"Bottleneck : {self.bottleneck}")
        for hint in self.hints:
            lines.append(f"  - Hint: {hint}")
        return "\n".join(lines)


def _block_extent(env: CompileEnvironment, spec: ConfigSpec, block_id: int) -> int:
    canonical_id = env.canonical_block_id(block_id)
    if 0 <= canonical_id < len(env.block_sizes):
        info = env.block_sizes[canonical_id]
        if isinstance(info.size, (int, torch.SymInt)):
            return max(1, env.size_hint(info.size))
    try:
        return max(1, spec.block_sizes.block_id_lookup(canonical_id).size_hint)
    except KeyError:
        return 1


def _block_tile_size(
    env: CompileEnvironment,
    spec: ConfigSpec,
    config: Config,
    block_id: int,
) -> int:
    canonical_id = env.canonical_block_id(block_id)
    ext = _block_extent(env, spec, canonical_id)
    if not (0 <= canonical_id < len(env.block_sizes)):
        return ext
    info = env.block_sizes[canonical_id]
    try:
        val = info.from_config(config)
    except Exception:
        val = None
    if val is None:
        return ext
    if isinstance(val, int):
        return max(1, min(ext, val))
    if isinstance(val, torch.SymInt):
        return max(1, min(ext, env.size_hint(val)))
    return ext


def _block_trips(
    env: CompileEnvironment,
    spec: ConfigSpec,
    config: Config,
    block_id: int,
) -> int:
    ext = _block_extent(env, spec, block_id)
    tile = _block_tile_size(env, spec, config, block_id)
    return math.ceil(ext / max(1, tile))


def _enclosing_loops_by_graph(
    device_ir: DeviceIR,
    env: CompileEnvironment,
) -> dict[int, tuple[tuple[int, ...], tuple[int, ...]]]:
    """Map each original graph_id to ``(grid_block_ids, inner_loop_block_ids)``."""
    parent_ids = control_flow_parent_ids(device_ir.graphs)
    root_grid_by_id: dict[int, tuple[int, ...]] = {
        root_id: tuple(env.canonical_block_id(bid) for bid in block_ids)
        for root_id, block_ids in zip(
            device_ir.root_ids, device_ir.grid_block_ids, strict=True
        )
    }
    graph_by_id = {g.graph_id: g for g in device_ir.graphs}
    result: dict[int, tuple[tuple[int, ...], tuple[int, ...]]] = {}

    for graph_info in device_ir.graphs:
        if isinstance(graph_info, ReductionLoopGraphInfo):
            continue
        current = graph_info.graph_id
        inner_loops: list[int] = []
        grid_loops: tuple[int, ...] = ()
        seen: set[int] = set()
        while current not in seen:
            seen.add(current)
            if current in root_grid_by_id:
                grid_loops = root_grid_by_id[current]
                break
            info = graph_by_id.get(current)
            if isinstance(info, ForLoopGraphInfo) and not isinstance(
                info, ReductionLoopGraphInfo
            ):
                for bid in reversed(info.block_ids):
                    cbid = env.canonical_block_id(bid)
                    if cbid not in inner_loops:
                        inner_loops.append(cbid)
            parent = parent_ids.get(current)
            if parent is None:
                break
            current = parent
        inner_loops.reverse()
        dedup_inner = tuple(bid for bid in inner_loops if bid not in grid_loops)
        result[graph_info.graph_id] = (grid_loops, dedup_inner)
    return result


def _op_base_name(target: object) -> str:
    name = getattr(target, "__name__", None) or getattr(target, "name", None)
    if not isinstance(name, str):
        name = str(target)
    return name.split(".")[0].lower()


def _collect_data_compute_nodes(
    device_ir: DeviceIR,
) -> set[int]:
    """Identify FX nodes on the value data-path (excluding index math and boolean masks)."""
    data_nodes: set[int] = set()
    for graph_info in device_ir.graphs:
        if isinstance(graph_info, ReductionLoopGraphInfo):
            continue
        nodes = list(graph_info.graph.nodes)
        for node in reversed(nodes):
            if node.op == "output":
                data_nodes.update(id(arg) for arg in node.all_input_nodes)
                continue
            if node.op != "call_function":
                continue
            if node.target is memory_ops.store:
                # store(tensor, index_list, value, extra_mask) -> args[2] is the stored value
                if len(node.args) >= 3 and isinstance(node.args[2], torch.fx.Node):
                    data_nodes.add(id(node.args[2]))
                continue
            if node.target in ATOMIC_OPS:
                for arg in node.args[2:]:
                    if isinstance(arg, torch.fx.Node):
                        data_nodes.add(id(arg))
                continue
            if node.target is memory_ops.load:
                # Do not traverse into load address subscripts (args[1]) or mask (args[2])
                continue
            if id(node) in data_nodes:
                data_nodes.update(id(inp) for inp in node.all_input_nodes)
    return data_nodes


def _tensor_logical_numel_and_blocks(
    env: CompileEnvironment,
    spec: ConfigSpec,
    tensor: torch.Tensor,
) -> tuple[int, set[int]]:
    numel = 1
    used_blocks: set[int] = set()
    for dim in tensor.shape:
        bid = env.resolve_block_id(dim)
        if bid is not None:
            cbid = env.canonical_block_id(bid)
            used_blocks.add(cbid)
            numel *= _block_extent(env, spec, cbid)
        else:
            numel *= max(1, env.size_hint(dim))
    return numel, used_blocks


def _resolve_matmul_dim_extent(
    env: CompileEnvironment,
    spec: ConfigSpec,
    dim: int | torch.SymInt,
    fact_bid: int | None,
    fact_static: int | None,
    used_bids: set[int],
) -> int:
    bid = fact_bid if fact_bid is not None else env.resolve_block_id(dim)
    if bid is not None:
        cbid = env.canonical_block_id(bid)
        used_bids.add(cbid)
        return _block_extent(env, spec, cbid)
    if fact_static is not None and fact_static > 0:
        return fact_static
    return max(1, env.size_hint(dim))


def _compute_static_flops(
    device_ir: DeviceIR,
    env: CompileEnvironment,
    spec: ConfigSpec,
    config: Config,
    enclosing_by_graph: dict[int, tuple[tuple[int, ...], tuple[int, ...]]],
) -> tuple[int, int, int, torch.dtype | None, torch.dtype | None]:
    """Return ``(matmul_flops, vector_flops, sfu_flops, dominant_matmul_dtype, dominant_vector_dtype)``."""
    operand_positions = matmul_operand_positions()
    data_nodes = _collect_data_compute_nodes(device_ir)

    matmul_flops = 0
    vector_flops = 0
    sfu_flops = 0
    matmul_flops_by_dtype: dict[torch.dtype, int] = {}
    vector_flops_by_dtype: dict[torch.dtype, int] = {}

    dot_nodes: set[torch.fx.Node] = set()
    for graph_info in device_ir.graphs:
        if isinstance(graph_info, ReductionLoopGraphInfo):
            continue
        for node in graph_info.graph.nodes:
            if node.op == "call_function" and node.target in operand_positions:
                dot_nodes.add(node)

    for graph_info in device_ir.graphs:
        if isinstance(graph_info, ReductionLoopGraphInfo):
            continue
        grid_bids, inner_bids = enclosing_by_graph.get(graph_info.graph_id, ((), ()))
        enclosing_bids = (*grid_bids, *inner_bids)

        for node in graph_info.graph.nodes:
            if node.op != "call_function":
                continue

            if node.target in operand_positions:
                lhs_pos, rhs_pos = operand_positions[node.target]
                if len(node.args) <= max(lhs_pos, rhs_pos):
                    continue
                lhs_arg = node.args[lhs_pos]
                rhs_arg = node.args[rhs_pos]
                lhs_val = (
                    lhs_arg.meta.get("val")
                    if isinstance(lhs_arg, torch.fx.Node)
                    else None
                )
                rhs_val = (
                    rhs_arg.meta.get("val")
                    if isinstance(rhs_arg, torch.fx.Node)
                    else None
                )
                if not isinstance(lhs_val, torch.Tensor) or not isinstance(
                    rhs_val, torch.Tensor
                ):
                    continue

                fact_id = node.meta.get(MATMUL_FACT_ID_META)
                fact = (
                    spec.matmul_facts[fact_id]
                    if isinstance(fact_id, int)
                    and 0 <= fact_id < len(spec.matmul_facts)
                    else None
                )

                used_bids: set[int] = set()
                m_ext = _resolve_matmul_dim_extent(
                    env,
                    spec,
                    lhs_val.shape[-2],
                    fact.m_block_id if fact is not None else None,
                    fact.static_m if fact is not None else None,
                    used_bids,
                )
                k_ext = _resolve_matmul_dim_extent(
                    env,
                    spec,
                    lhs_val.shape[-1],
                    fact.k_block_id if fact is not None else None,
                    fact.static_k if fact is not None else None,
                    used_bids,
                )
                n_ext = _resolve_matmul_dim_extent(
                    env,
                    spec,
                    rhs_val.shape[-1],
                    fact.n_block_id if fact is not None else None,
                    fact.static_n if fact is not None else None,
                    used_bids,
                )

                batch_ext = 1
                leading_shape = (
                    lhs_val.shape[:-2]
                    if lhs_val.ndim >= rhs_val.ndim
                    else rhs_val.shape[:-2]
                )
                for bdim in leading_shape:
                    batch_ext *= _resolve_matmul_dim_extent(
                        env, spec, bdim, None, None, used_bids
                    )

                for extra_bid in enclosing_bids:
                    if extra_bid not in used_bids:
                        batch_ext *= _block_trips(env, spec, config, extra_bid)

                op_flops = 2 * batch_ext * m_ext * n_ext * k_ext
                matmul_flops += op_flops
                dtype = lhs_val.dtype
                matmul_flops_by_dtype[dtype] = (
                    matmul_flops_by_dtype.get(dtype, 0) + op_flops
                )
                continue

            if node.target in _NON_COMPUTE_TARGETS or id(node) not in data_nodes:
                continue

            # Skip loop-carried accumulation `acc + dot(...)` which lowers into FMA inside tl.dot
            if node.target in (torch.ops.aten.add.Tensor, operator.add) and any(
                inp in dot_nodes for inp in node.all_input_nodes
            ):
                continue

            val = node.meta.get("val")
            if not isinstance(val, torch.Tensor) or val.dtype == torch.bool:
                continue

            lowering = node.meta.get("lowering")
            if isinstance(lowering, ReductionLowering) and node.all_input_nodes:
                first_in = node.all_input_nodes[0].meta.get("val")
                ref_tensor = first_in if isinstance(first_in, torch.Tensor) else val
            else:
                ref_tensor = val

            numel, used_bids = _tensor_logical_numel_and_blocks(env, spec, ref_tensor)
            for extra_bid in enclosing_bids:
                if extra_bid not in used_bids:
                    numel *= _block_trips(env, spec, config, extra_bid)

            base_name = _op_base_name(node.target)
            weight = 2 if base_name in ("addmm", "baddbmm", "fma", "addcmul") else 1
            op_flops = numel * weight
            vector_flops += op_flops
            if base_name in _SFU_OP_NAMES:
                sfu_flops += op_flops
            if ref_tensor.dtype.is_floating_point:
                vector_flops_by_dtype[ref_tensor.dtype] = (
                    vector_flops_by_dtype.get(ref_tensor.dtype, 0) + op_flops
                )

    dominant_matmul_dtype = (
        max(matmul_flops_by_dtype.items(), key=operator.itemgetter(1))[0]
        if matmul_flops_by_dtype
        else None
    )
    dominant_vector_dtype = (
        max(vector_flops_by_dtype.items(), key=operator.itemgetter(1))[0]
        if vector_flops_by_dtype
        else None
    )
    return (
        matmul_flops,
        vector_flops,
        sfu_flops,
        dominant_matmul_dtype,
        dominant_vector_dtype,
    )


def _memory_op_used_block_ids(
    env: CompileEnvironment,
    fact: MemoryOpFact,
) -> set[int]:
    used: set[int] = set()
    for bid in (
        *fact.indexed_block_ids,
        *fact.subscript_block_ids,
        *fact.subscript_affine_block_ids,
    ):
        if bid is not None:
            used.add(env.canonical_block_id(bid))
    used.update(env.canonical_block_id(r_bid) for r_bid, _count in fact.reductions_fed)
    return used


def _memory_op_nodes_by_index(device_ir: DeviceIR) -> dict[int, torch.fx.Node]:
    nodes: dict[int, torch.fx.Node] = {}
    idx = 0
    for graph_info in device_ir.graphs:
        for node in graph_info.graph.nodes:
            if node.op == "call_function" and node.target in (
                memory_ops.load,
                memory_ops.store,
            ):
                nodes[idx] = node
                idx += 1
    return nodes


def _compute_memory_traffic(
    device_ir: DeviceIR,
    env: CompileEnvironment,
    spec: ConfigSpec,
    config: Config,
    enclosing_by_graph: dict[int, tuple[tuple[int, ...], tuple[int, ...]]],
) -> tuple[int, int, int, int, int]:
    """Return ``(algo_read, algo_write, config_read, config_write, l2_estimated_total)`` in bytes."""
    unique_reads: dict[object, int] = {}
    unique_writes: dict[object, int] = {}
    config_read_bytes = 0
    config_write_bytes = 0
    l2_read_bytes = 0

    l2_groupings = getattr(config, "l2_groupings", ()) or ()
    max_l2_group = (
        max((int(g) for g in l2_groupings if isinstance(g, int) and g > 1), default=1)
        if isinstance(l2_groupings, (list, tuple))
        else 1
    )
    reduction_fusion = bool(getattr(config, "reduction_fusion", False))
    op_nodes = _memory_op_nodes_by_index(device_ir)

    for fact in spec.memory_op_facts:
        if fact.dtype is None or fact.accessed_numel <= 0:
            continue
        if not (0 <= fact.graph_id < len(device_ir.graphs)):
            continue
        graph_info = device_ir.graphs[fact.graph_id]
        if isinstance(graph_info, ReductionLoopGraphInfo):
            continue

        used_bids = _memory_op_used_block_ids(env, fact)
        effective_numel = fact.accessed_numel
        storage_key: object = fact.tensor_name or f"slot_{fact.indexing_index}"

        node = op_nodes.get(fact.indexing_index)
        if node is not None:
            accessed_fake = _accessed_tensor_fake(node)
            if accessed_fake is not None:
                with contextlib.suppress(Exception):
                    storage_key = (
                        "storage",
                        int(accessed_fake.untyped_storage()._cdata),
                    )
            tile_val = (
                node.meta.get("val")
                if fact.kind == "load"
                else (
                    node.args[2].meta.get("val")
                    if len(node.args) >= 3 and isinstance(node.args[2], torch.fx.Node)
                    else None
                )
            )
            if isinstance(tile_val, torch.Tensor):
                tile_numel, tile_bids = _tensor_logical_numel_and_blocks(
                    env, spec, tile_val
                )
                used_bids |= tile_bids
                if tile_numel > 0:
                    effective_numel = min(effective_numel, tile_numel)

        base_bytes = effective_numel * fact.dtype.itemsize
        if fact.kind == "load":
            unique_reads[storage_key] = max(
                unique_reads.get(storage_key, 0), base_bytes
            )
        else:
            unique_writes[storage_key] = max(
                unique_writes.get(storage_key, 0), base_bytes
            )

        grid_bids, inner_bids = enclosing_by_graph.get(fact.graph_id, ((), ()))
        enclosing_bids = (*grid_bids, *inner_bids)

        repeat_factor = 1
        grid_repeat_factor = 1
        for loop_bid in enclosing_bids:
            if loop_bid not in used_bids:
                trips = _block_trips(env, spec, config, loop_bid)
                repeat_factor *= trips
                if loop_bid in grid_bids:
                    grid_repeat_factor *= trips

        # Account for multi-pass rolled reductions when reduction_fusion is disabled
        if fact.kind == "load" and not reduction_fusion:
            for r_bid, fed_count in fact.reductions_fed:
                if fed_count > 1:
                    r_trips = _block_trips(env, spec, config, r_bid)
                    if r_trips > 1:
                        repeat_factor *= fed_count

        op_config_bytes = base_bytes * repeat_factor
        if fact.kind == "load":
            config_read_bytes += op_config_bytes
            if max_l2_group > 1 and grid_repeat_factor > 1:
                l2_saving = min(max_l2_group, grid_repeat_factor)
                l2_read_bytes += max(base_bytes, op_config_bytes // l2_saving)
            else:
                l2_read_bytes += op_config_bytes
        else:
            config_write_bytes += op_config_bytes

    algo_read = sum(unique_reads.values())
    algo_write = sum(unique_writes.values())
    l2_estimated_total = l2_read_bytes + config_write_bytes
    return (
        algo_read,
        algo_write,
        config_read_bytes,
        config_write_bytes,
        l2_estimated_total,
    )


def _compute_total_workgroups(
    device_ir: DeviceIR,
    env: CompileEnvironment,
    spec: ConfigSpec,
    config: Config,
) -> int:
    if not device_ir.grid_block_ids:
        return 1
    total_workgroups = 0
    for group in device_ir.grid_block_ids:
        group_ctas = 1
        for bid in group:
            group_ctas *= _block_trips(env, spec, config, bid)
        total_workgroups += max(1, group_ctas)
    return max(1, total_workgroups)


def _diagnose_bottleneck_and_hints(
    *,
    env: CompileEnvironment,
    spec: ConfigSpec,
    config: Config,
    hardware: HardwareRooflineSpec,
    matmul_flops: int,
    total_flops: int,
    compute_bound_ms: float,
    algo_memory_bound_ms: float,
    config_memory_bound_ms: float,
    traffic_amplification: float,
    waves_per_cu: float,
    sol_pct: float | None,
) -> tuple[RooflineBottleneck, tuple[str, ...]]:
    hints: list[str] = []

    if waves_per_cu < 1.0 and (sol_pct is None or sol_pct < 60.0):
        bottleneck: RooflineBottleneck = "occupancy_or_latency_bound"
        hints.append(
            f"Grid launches only {waves_per_cu:.2f} waves across {hardware.num_compute_units} "
            "compute units (Xe-cores); consider smaller tile sizes (block_sizes) to expose more parallelism."
        )
    elif (
        traffic_amplification > 1.25
        and config_memory_bound_ms > compute_bound_ms
        and config_memory_bound_ms > algo_memory_bound_ms * 1.25
    ):
        bottleneck = "tile_traffic_bound"
        hints.append(
            f"Tiling amplifies memory traffic by {traffic_amplification:.2f}x over algorithmic minimum, "
            "shifting the kernel into a memory-traffic bottleneck; consider larger M/N block_sizes or L2 grouping (l2_groupings)."
        )
    elif compute_bound_ms >= config_memory_bound_ms and total_flops > 0:
        bottleneck = "compute_bound"
    else:
        bottleneck = "memory_bound"

    # Intel XPU-specific tuning hints for Triton backend
    if hardware.device_kind == "xpu":
        grf_mode = config.get("grf_mode", "default")
        num_warps = config.get("num_warps", 4)
        if (
            matmul_flops > 0
            and grf_mode == "default"
            and isinstance(num_warps, int)
            and num_warps <= 32
        ):
            hints.append(
                "On Intel XPU, large DPAS/XMX matmul or high-register tiles can benefit from "
                "grf_mode='256' (large GRF mode) when num_warps <= 32 to avoid register spills."
            )
        indexing = config.get("indexing")
        uses_td = indexing == "tensor_descriptor" or (
            isinstance(indexing, (list, tuple)) and "tensor_descriptor" in indexing
        )
        if matmul_flops > 0 and not uses_td:
            hints.append(
                "On Intel XPU (Xe/Xe2), setting indexing='tensor_descriptor' enables 2D block I/O "
                "for matmul operands and typically improves XMX feed efficiency."
            )
        if (
            bottleneck == "tile_traffic_bound"
            and matmul_flops > 0
            and not getattr(config, "l2_groupings", None)
        ):
            hints.append(
                "Consider setting l2_groupings (e.g. 4 or 8) to improve Xe L2 cache reuse across workgroups."
            )
        if (
            bottleneck in ("memory_bound", "tile_traffic_bound")
            and spec.reduction_loops
            and not getattr(config, "reduction_fusion", False)
            and traffic_amplification > 1.1
        ):
            hints.append(
                "Looped reduction re-reads input tensors across passes; consider persistent reduction "
                "(reduction_loops=[None]) or enabling reduction_fusion if register budget permits."
            )

    return bottleneck, tuple(hints)


def analyze_roofline(
    *,
    env: CompileEnvironment,
    device_ir: DeviceIR,
    host_function: HostFunction | None = None,
    config: Config | None = None,
    latency_ms: float | None = None,
    hardware_spec: HardwareRooflineSpec | None = None,
) -> RooflineReport:
    """Compute the static and (optionally) empirical Roofline report for a compiled kernel."""
    spec = env.config_spec
    if config is None:
        config = spec.default_config()

    resolved_host = host_function or device_ir.host_function
    env_ctx = (
        contextlib.nullcontext()
        if CompileEnvironment.has_current() and CompileEnvironment.current() is env
        else env
    )
    host_ctx = resolved_host if resolved_host is not None else contextlib.nullcontext()

    hardware = get_hardware_roofline_spec(
        env.device,
        override=hardware_spec,
    )

    with env_ctx, host_ctx:
        if not spec.memory_op_facts and resolved_host is not None:
            analysis = DeviceIRAnalysis.build(device_ir, env)
            spec.memory_op_facts = analysis.memory_op_facts(env, resolved_host)

        enclosing_by_graph = _enclosing_loops_by_graph(device_ir, env)
        (
            matmul_flops,
            vector_flops,
            sfu_flops,
            dominant_matmul_dtype,
            dominant_vector_dtype,
        ) = _compute_static_flops(device_ir, env, spec, config, enclosing_by_graph)
        (
            algo_read_bytes,
            algo_write_bytes,
            config_read_bytes,
            config_write_bytes,
            l2_estimated_bytes,
        ) = _compute_memory_traffic(device_ir, env, spec, config, enclosing_by_graph)
        num_workgroups = _compute_total_workgroups(device_ir, env, spec, config)

    total_flops = matmul_flops + vector_flops
    algo_bytes = algo_read_bytes + algo_write_bytes
    config_bytes = config_read_bytes + config_write_bytes

    algo_ai = total_flops / max(1, algo_bytes)
    config_ai = total_flops / max(1, config_bytes)
    traffic_amp = config_bytes / max(1, algo_bytes) if algo_bytes > 0 else 1.0
    waves_per_cu = num_workgroups / max(1, hardware.num_compute_units)

    peak_matrix_tflops = hardware.matrix_tflops(dominant_matmul_dtype)
    peak_vector_tflops = hardware.vector_tflops(dominant_vector_dtype)
    peak_bw_gbps = hardware.peak_memory_bandwidth_gbps

    matmul_compute_ms = (
        (matmul_flops / (peak_matrix_tflops * 1e12)) * 1e3
        if matmul_flops > 0 and peak_matrix_tflops > 0
        else 0.0
    )
    vector_compute_ms = (
        (vector_flops / (peak_vector_tflops * 1e12)) * 1e3
        if vector_flops > 0 and peak_vector_tflops > 0
        else 0.0
    )
    compute_bound_ms = matmul_compute_ms + vector_compute_ms

    if compute_bound_ms > 0 and total_flops > 0:
        effective_peak_tflops = (total_flops / (compute_bound_ms * 1e-3)) / 1e12
    elif matmul_flops > 0:
        effective_peak_tflops = peak_matrix_tflops
    else:
        effective_peak_tflops = peak_vector_tflops

    ridge_point = (effective_peak_tflops * 1e3) / max(1e-9, peak_bw_gbps)

    algo_memory_bound_ms = (
        (algo_bytes / (peak_bw_gbps * 1e9)) * 1e3 if peak_bw_gbps > 0 else 0.0
    )
    config_memory_bound_ms = (
        (l2_estimated_bytes / (peak_bw_gbps * 1e9)) * 1e3 if peak_bw_gbps > 0 else 0.0
    )

    ideal_ms = max(compute_bound_ms, algo_memory_bound_ms)
    config_ideal_ms = max(compute_bound_ms, config_memory_bound_ms)

    measured: float | None = None
    achieved_tflops: float | None = None
    achieved_algo_gbps: float | None = None
    achieved_config_gbps: float | None = None
    sol_pct: float | None = None
    config_sol_pct: float | None = None

    if latency_ms is not None and math.isfinite(latency_ms) and latency_ms > 0:
        measured = float(latency_ms)
        sec = measured * 1e-3
        achieved_tflops = (total_flops / sec) / 1e12
        achieved_algo_gbps = (algo_bytes / sec) / 1e9
        achieved_config_gbps = (config_bytes / sec) / 1e9
        sol_pct = (ideal_ms / measured) * 100.0 if ideal_ms > 0 else 0.0
        config_sol_pct = (
            (config_ideal_ms / measured) * 100.0 if config_ideal_ms > 0 else 0.0
        )

    bottleneck, hints = _diagnose_bottleneck_and_hints(
        env=env,
        spec=spec,
        config=config,
        hardware=hardware,
        matmul_flops=matmul_flops,
        total_flops=total_flops,
        compute_bound_ms=compute_bound_ms,
        algo_memory_bound_ms=algo_memory_bound_ms,
        config_memory_bound_ms=config_memory_bound_ms,
        traffic_amplification=traffic_amp,
        waves_per_cu=waves_per_cu,
        sol_pct=sol_pct,
    )

    return RooflineReport(
        matmul_flops=matmul_flops,
        vector_flops=vector_flops,
        sfu_flops=sfu_flops,
        total_flops=total_flops,
        algorithmic_read_bytes=algo_read_bytes,
        algorithmic_write_bytes=algo_write_bytes,
        algorithmic_bytes=algo_bytes,
        algorithmic_arithmetic_intensity=algo_ai,
        dominant_matmul_dtype=dominant_matmul_dtype,
        dominant_vector_dtype=dominant_vector_dtype,
        config=config,
        config_read_bytes=config_read_bytes,
        config_write_bytes=config_write_bytes,
        config_bytes=config_bytes,
        l2_estimated_bytes=l2_estimated_bytes,
        config_arithmetic_intensity=config_ai,
        traffic_amplification=traffic_amp,
        num_workgroups=num_workgroups,
        waves_per_compute_unit=waves_per_cu,
        hardware=hardware,
        peak_matrix_tflops=peak_matrix_tflops,
        peak_vector_tflops=peak_vector_tflops,
        effective_peak_tflops=effective_peak_tflops,
        peak_bandwidth_gbps=peak_bw_gbps,
        ridge_point_flops_per_byte=ridge_point,
        compute_bound_ms=compute_bound_ms,
        algo_memory_bound_ms=algo_memory_bound_ms,
        config_memory_bound_ms=config_memory_bound_ms,
        ideal_ms=ideal_ms,
        config_ideal_ms=config_ideal_ms,
        measured_ms=measured,
        achieved_tflops=achieved_tflops,
        achieved_algo_gbps=achieved_algo_gbps,
        achieved_config_gbps=achieved_config_gbps,
        sol_pct=sol_pct,
        config_sol_pct=config_sol_pct,
        bottleneck=bottleneck,
        hints=hints,
    )

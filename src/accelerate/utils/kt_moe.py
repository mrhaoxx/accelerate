# Copyright 2026 the KTransformers team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
KTransformers MoE Backend Integration via KTMoEWrapper.
"""

from __future__ import annotations

import gc
import importlib.util as _u
import inspect
import math
import os
import weakref
import time
from datetime import datetime
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any
from threading import Lock

import torch
import torch.nn as nn
import torch.nn.functional as F

import logging as _logging

from .dataclasses import KTransformersPlugin

logger = _logging.getLogger(__name__)
KT_DEBUG = os.environ.get("ACCELERATE_KT_DEBUG", "0") == "1"
KT_MEM_LOG = os.environ.get("ACCELERATE_KT_MEM_LOG", "0") == "1"
KT_LIFECYCLE_LOG = os.environ.get("ACCELERATE_KT_LIFECYCLE_LOG", "0") == "1"
KT_LIFECYCLE_REFERRERS = os.environ.get("ACCELERATE_KT_LIFECYCLE_REFERRERS", "0") == "1"

_LIFECYCLE_LOCK = Lock()
_LIFECYCLE_SEQ = 0
_LIFECYCLE_BACKWARD_COUNT = 0
_LIFECYCLE_TRACKED: dict[int, dict[str, Any]] = {}
_CTX_SEQ = 0
_CTX_LIVE: dict[int, dict[str, Any]] = {}
_KT_MEM_LOG_LOCK = Lock()
_KT_MEM_LOG_CLEARED_PATHS: set[str] = set()

# Check if kt_kernel is available
KT_KERNEL_AVAILABLE = _u.find_spec("kt_kernel") is not None

if KT_KERNEL_AVAILABLE:
    try:
        from kt_kernel.experts import KTMoEWrapper
    except ImportError:
        KT_KERNEL_AVAILABLE = False
        KTMoEWrapper = None

# Check if safetensors is available
try:
    from safetensors import safe_open

    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False
    safe_open = None


# =============================================================================
# Exception Classes
# =============================================================================


class KTAMXError(Exception):
    """Base exception for KT AMX errors."""


class KTAMXNotAvailableError(KTAMXError):
    """kt_kernel not installed or AMX not supported."""


class KTAMXModelNotSupportedError(KTAMXError):
    """Model architecture not supported."""


class KTAMXConfigError(KTAMXError):
    """Configuration error."""


# =============================================================================
# MoE Configuration
# =============================================================================


@dataclass
class MOEArchConfig:
    """MoE architecture configuration for different model types."""

    moe_layer_attr: str
    router_attr: str
    experts_attr: str
    weight_names: tuple[str, str, str]
    expert_num: int
    intermediate_size: int
    num_experts_per_tok: int
    has_shared_experts: bool = False
    router_type: str = "linear"


def get_moe_arch_config(config) -> MOEArchConfig:
    """
    Get MoE architecture configuration based on model type.

    Args:
        config: HuggingFace model configuration

    Returns:
        MOEArchConfig for the model

    Raises:
        KTAMXModelNotSupportedError: If model architecture is not supported
    """
    arch = config.architectures[0] if getattr(config, "architectures", None) else ""

    if "DeepseekV2" in arch:
        return MOEArchConfig(
            moe_layer_attr="mlp",
            router_attr="gate",
            experts_attr="experts",
            weight_names=("gate_proj", "up_proj", "down_proj"),
            expert_num=config.n_routed_experts,
            intermediate_size=config.moe_intermediate_size,
            num_experts_per_tok=config.num_experts_per_tok,
            has_shared_experts=getattr(config, "n_shared_experts", 0) > 0,
            router_type="deepseek_gate",
        )
    if "DeepseekV3" in arch:
        return MOEArchConfig(
            moe_layer_attr="mlp",
            router_attr="gate",
            experts_attr="experts",
            weight_names=("gate_proj", "up_proj", "down_proj"),
            expert_num=config.n_routed_experts,
            intermediate_size=config.moe_intermediate_size,
            num_experts_per_tok=config.num_experts_per_tok,
            has_shared_experts=getattr(config, "n_shared_experts", 0) > 0,
            router_type="deepseek_gate",
        )
    if "Qwen2Moe" in arch or "Qwen3Moe" in arch:
        return MOEArchConfig(
            moe_layer_attr="mlp",
            router_attr="gate",
            experts_attr="experts",
            weight_names=("gate_proj", "up_proj", "down_proj"),
            expert_num=config.num_experts,
            intermediate_size=config.moe_intermediate_size,
            num_experts_per_tok=config.num_experts_per_tok,
            has_shared_experts=getattr(config, "shared_expert_intermediate_size", 0) > 0,
        )
    if "Mixtral" in arch:
        return MOEArchConfig(
            moe_layer_attr="block_sparse_moe",
            router_attr="gate",
            experts_attr="experts",
            weight_names=("w1", "w3", "w2"),
            expert_num=config.num_local_experts,
            intermediate_size=config.intermediate_size,
            num_experts_per_tok=config.num_experts_per_tok,
            has_shared_experts=False,
        )

    raise KTAMXModelNotSupportedError(
        f"Model architecture {arch} not supported for KT AMX. "
        "Supported architectures: DeepseekV2, DeepseekV3, Qwen2Moe, Qwen3Moe, Mixtral"
    )


def get_moe_module(layer: nn.Module, moe_config: MOEArchConfig) -> nn.Module | None:
    """Get MoE module from transformer layer."""
    moe_module = getattr(layer, moe_config.moe_layer_attr, None)
    if moe_module is None:
        return None
    if not hasattr(moe_module, moe_config.experts_attr):
        return None
    return moe_module


def _tensor_nbytes(obj: Any) -> int:
    if torch.is_tensor(obj):
        return obj.numel() * obj.element_size()
    if isinstance(obj, (list, tuple)):
        total = 0
        for x in obj:
            total += _tensor_nbytes(x)
        return total
    return 0


def _kt_mem_log(rank: int, tag: str, **stats: Any) -> None:
    if not KT_MEM_LOG:
        return
    path_tpl = os.environ.get("ACCELERATE_KT_MEM_LOG_FILE", "kt_mem_rank{rank}.log")
    path = path_tpl.format(rank=rank)
    try:
        # Clear previous run log once per process/path so each launch starts fresh.
        with _KT_MEM_LOG_LOCK:
            if path not in _KT_MEM_LOG_CLEARED_PATHS:
                with open(path, "w", encoding="utf-8"):
                    pass
                _KT_MEM_LOG_CLEARED_PATHS.add(path)
        pieces: list[str] = []
        for k, v in stats.items():
            if isinstance(v, int) and ("bytes" in k or "nbytes" in k):
                pieces.append(f"{k}={v/1024/1024:.2f}MB")
            else:
                pieces.append(f"{k}={v}")
        line = (
            f"{datetime.now().isoformat()} pid={os.getpid()} rank={rank} tag={tag} "
            + " ".join(pieces)
            + "\n"
        )
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)
    except Exception:
        return


def _cuda_mem_stats(device: torch.device | None) -> dict[str, int]:
    if device is None or device.type != "cuda" or not torch.cuda.is_available():
        return {}
    return {
        "mem_allocated_bytes": torch.cuda.memory_allocated(device),
        "mem_reserved_bytes": torch.cuda.memory_reserved(device),
        "mem_max_allocated_bytes": torch.cuda.max_memory_allocated(device),
    }


def _wrapper_cache_stats(wrapper: Any) -> dict[str, Any]:
    if wrapper is None:
        return {"wrapper_present": False}

    pending_buffer = getattr(wrapper, "_pending_buffer", None)
    pending_qlen = getattr(wrapper, "_pending_qlen", None)
    hidden_size = getattr(wrapper, "hidden_size", None)
    num_experts_per_tok = getattr(wrapper, "num_experts_per_tok", None)

    stats: dict[str, Any] = {
        "wrapper_present": True,
        "cache_depth": getattr(wrapper, "_cache_depth", None),
        "max_cache_depth": getattr(wrapper, "max_cache_depth", None),
        "pending_exists": pending_buffer is not None,
        "pending_save_for_backward": getattr(wrapper, "_pending_save_for_backward", None),
        "pending_qlen": pending_qlen,
    }
    if isinstance(pending_qlen, int) and isinstance(hidden_size, int):
        stats["pending_input_bytes_est"] = pending_qlen * hidden_size * 2  # bf16
        stats["pending_output_bytes_est"] = pending_qlen * hidden_size * 2  # bf16
    if isinstance(pending_qlen, int) and isinstance(num_experts_per_tok, int):
        stats["pending_expert_ids_bytes_est"] = pending_qlen * num_experts_per_tok * 8  # int64
        stats["pending_weights_bytes_est"] = pending_qlen * num_experts_per_tok * 4  # float32
    return stats


def _kt_mem_log_wrapper(rank: int, tag: str, wrapper: Any, device: torch.device | None = None, **stats: Any) -> None:
    if not KT_MEM_LOG:
        return
    merged: dict[str, Any] = {}
    merged.update(_wrapper_cache_stats(wrapper))
    merged.update(_cuda_mem_stats(device))
    merged.update(stats)
    _kt_mem_log(rank, tag, **merged)


def _all_gather_qlens(local_qlen: int, device: torch.device, world_size: int) -> list[int]:
    import torch.distributed as dist

    local_qlen_t = torch.tensor([int(local_qlen)], device=device, dtype=torch.int64)
    gathered = [torch.empty(1, device=device, dtype=torch.int64) for _ in range(world_size)]
    dist.all_gather(gathered, local_qlen_t)
    return [int(t.item()) for t in gathered]


def _qlen_offsets(all_qlens: list[int]) -> list[int]:
    offsets = [0]
    for q in all_qlens:
        offsets.append(offsets[-1] + int(q))
    return offsets


def _dist_gather_varlen_to_rank0(
    local_tensor: torch.Tensor,
    *,
    all_qlens: list[int],
    rank: int,
    world_size: int,
) -> list[torch.Tensor] | None:
    import torch.distributed as dist

    local_tensor = local_tensor.contiguous()
    local_expected = int(all_qlens[rank])
    if local_tensor.shape[0] != local_expected:
        raise RuntimeError(
            f"Local leading dim mismatch on rank {rank}: got {local_tensor.shape[0]}, expected {local_expected}"
        )

    if rank == 0:
        gathered: list[torch.Tensor | None] = [None] * world_size
        gathered[0] = local_tensor
        ops: list[dist.P2POp] = []
        for src in range(1, world_size):
            qlen_src = int(all_qlens[src])
            recv_shape = (qlen_src, *local_tensor.shape[1:])
            recv = torch.empty(recv_shape, device=local_tensor.device, dtype=local_tensor.dtype)
            gathered[src] = recv
            if qlen_src > 0:
                ops.append(dist.P2POp(dist.irecv, recv, src))
        if ops:
            reqs = dist.batch_isend_irecv(ops)
            for req in reqs:
                req.wait()
        out: list[torch.Tensor] = []
        for idx, t in enumerate(gathered):
            if t is None:
                raise RuntimeError(f"Missing gathered tensor for rank {idx} on rank0.")
            out.append(t)
        return out

    if local_expected > 0:
        reqs = dist.batch_isend_irecv([dist.P2POp(dist.isend, local_tensor, 0)])
        for req in reqs:
            req.wait()
    return None


def _dist_scatter_varlen_from_rank0(
    *,
    rank0_chunks: list[torch.Tensor] | None,
    all_qlens: list[int],
    rank: int,
    world_size: int,
    feature_shape: tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    import torch.distributed as dist

    local_qlen = int(all_qlens[rank])
    local_out = torch.empty((local_qlen, *feature_shape), device=device, dtype=dtype)

    if rank == 0:
        if rank0_chunks is None or len(rank0_chunks) != world_size:
            raise RuntimeError("rank0_chunks must contain one chunk per rank on rank0.")
        if int(rank0_chunks[0].shape[0]) != local_qlen:
            raise RuntimeError(
                f"Rank0 local chunk mismatch: got {rank0_chunks[0].shape[0]}, expected {local_qlen}"
            )
        if local_qlen > 0:
            local_out.copy_(rank0_chunks[0])
        ops: list[dist.P2POp] = []
        for dst in range(1, world_size):
            qlen_dst = int(all_qlens[dst])
            if qlen_dst <= 0:
                continue
            chunk = rank0_chunks[dst].contiguous()
            if int(chunk.shape[0]) != qlen_dst:
                raise RuntimeError(
                    f"Rank{dst} chunk mismatch on rank0: got {chunk.shape[0]}, expected {qlen_dst}"
                )
            ops.append(dist.P2POp(dist.isend, chunk, dst))
        if ops:
            reqs = dist.batch_isend_irecv(ops)
            for req in reqs:
                req.wait()
        return local_out

    if local_qlen > 0:
        reqs = dist.batch_isend_irecv([dist.P2POp(dist.irecv, local_out, 0)])
        for req in reqs:
            req.wait()
    return local_out


def _is_in_checkpoint_first_forward() -> bool:
    """Best-effort detection for non-reentrant checkpoint first forward.

    Recompute (during backward) does not traverse our Python GC wrapper call path,
    while the first forward does.
    """
    try:
        for frame_info in inspect.stack(context=0):
            fn = frame_info.function
            file = frame_info.filename or ""
            # LLaMA-Factory GC wrapper (first forward only).
            if fn == "custom_gradient_checkpointing_func" and file.endswith("checkpointing.py"):
                return True
    except Exception:
        return False
    return False


def _checkpoint_hook_mode() -> str:
    """Infer checkpoint phase from current saved_tensors_hooks top.

    Returns one of:
      - "first_forward": non-reentrant checkpoint's _checkpoint_hook
      - "recompute": non-reentrant checkpoint's _recomputation_hook
      - "none": no default saved_tensors_hooks on top
      - "other": unknown hook stack entry
      - "error": failed to query hook stack
    """
    try:
        top = torch._C._autograd._top_saved_tensors_default_hooks(False)
    except Exception:
        return "error"
    if top is None:
        return "none"
    try:
        pack_fn, _ = top
        mod = getattr(pack_fn, "__module__", "")
        qual = getattr(pack_fn, "__qualname__", getattr(pack_fn, "__name__", ""))
        tag = f"{mod}.{qual}"
    except Exception:
        return "other"
    if "_recomputation_hook.__init__.<locals>.pack_hook" in tag:
        return "recompute"
    if "_checkpoint_hook.__init__.<locals>.pack_hook" in tag:
        return "first_forward"
    return "other"


def _lifecycle_referrer_types(obj: Any, limit: int = 6) -> list[str]:
    if not KT_LIFECYCLE_REFERRERS:
        return []
    try:
        ref_types: dict[str, int] = {}
        for ref in gc.get_referrers(obj):
            tname = type(ref).__name__
            ref_types[tname] = ref_types.get(tname, 0) + 1
        return [f"{k}:{v}" for k, v in sorted(ref_types.items(), key=lambda kv: kv[1], reverse=True)[:limit]]
    except Exception:
        return []


def _lifecycle_finalize(tensor_id: int, rank: int, layer: int, label: str, seq: int, bytes_size: int) -> None:
    with _LIFECYCLE_LOCK:
        _LIFECYCLE_TRACKED.pop(tensor_id, None)
    _kt_mem_log(
        rank,
        "lifecycle_release",
        layer=layer,
        label=label,
        seq=seq,
        tensor_id=tensor_id,
        tensor_bytes=bytes_size,
    )


def _track_lifecycle_tensor(
    rank: int,
    layer: int,
    label: str,
    tensor: torch.Tensor | None,
) -> None:
    if not KT_LIFECYCLE_LOG:
        return
    if not torch.is_tensor(tensor):
        return
    if not tensor.is_cuda:
        return
    tensor_id = id(tensor)
    tensor_bytes = _tensor_nbytes(tensor)
    with _LIFECYCLE_LOCK:
        global _LIFECYCLE_SEQ
        _LIFECYCLE_SEQ += 1
        seq = _LIFECYCLE_SEQ
        finalizer = weakref.finalize(
            tensor,
            _lifecycle_finalize,
            tensor_id,
            rank,
            layer,
            label,
            seq,
            tensor_bytes,
        )
        _LIFECYCLE_TRACKED[tensor_id] = {
            "seq": seq,
            "rank": rank,
            "layer": layer,
            "label": label,
            "bytes": tensor_bytes,
            "shape": tuple(tensor.shape),
            "dtype": str(tensor.dtype),
            "device": str(tensor.device),
            "created_at": time.time(),
            "weak": weakref.ref(tensor),
            "finalizer": finalizer,
        }
    _kt_mem_log(
        rank,
        "lifecycle_track",
        layer=layer,
        label=label,
        seq=seq,
        tensor_id=tensor_id,
        tensor_shape=tuple(tensor.shape),
        tensor_dtype=str(tensor.dtype),
        tensor_device=str(tensor.device),
        tensor_bytes=tensor_bytes,
    )


def _track_lifecycle_collection(
    rank: int,
    layer: int,
    label_prefix: str,
    tensors: Any,
    *,
    max_items: int = 8,
) -> None:
    if not KT_LIFECYCLE_LOG:
        return
    if tensors is None:
        return
    if torch.is_tensor(tensors):
        _track_lifecycle_tensor(rank, layer, label_prefix, tensors)
        return
    if not isinstance(tensors, (list, tuple)):
        return
    for i, t in enumerate(tensors):
        if i >= max_items:
            break
        _track_lifecycle_tensor(rank, layer, f"{label_prefix}[{i}]", t)


def _log_lifecycle_snapshot(rank: int, *, layer: int, tag: str, topk: int = 8) -> None:
    if not KT_LIFECYCLE_LOG:
        return
    now = time.time()
    with _LIFECYCLE_LOCK:
        live = list(_LIFECYCLE_TRACKED.values())
    if not live:
        _kt_mem_log(rank, "lifecycle_snapshot", layer=layer, snapshot_tag=tag, live_count=0, live_bytes=0)
        return

    live_bytes = sum(entry["bytes"] for entry in live)
    _kt_mem_log(
        rank,
        "lifecycle_snapshot",
        layer=layer,
        snapshot_tag=tag,
        live_count=len(live),
        live_bytes=live_bytes,
    )

    # Focus on largest and oldest live tensors to surface persistent holders.
    ranked = sorted(live, key=lambda x: (x["bytes"], now - x["created_at"]), reverse=True)[:topk]
    for entry in ranked:
        age_s = max(0.0, now - entry["created_at"])
        ref_types: list[str] = []
        obj = entry["weak"]()
        if obj is not None:
            ref_types = _lifecycle_referrer_types(obj)
        _kt_mem_log(
            rank,
            "lifecycle_live_tensor",
            layer=entry["layer"],
            label=entry["label"],
            seq=entry["seq"],
            tensor_bytes=entry["bytes"],
            tensor_shape=entry["shape"],
            tensor_dtype=entry["dtype"],
            tensor_device=entry["device"],
            age_s=round(age_s, 6),
            referrer_types="|".join(ref_types) if ref_types else "",
        )


def _ctx_register(
    rank: int,
    *,
    layer: int,
    topk_ids_id: int,
    topk_ids_shape: tuple[int, ...],
    topk_ids_bytes: int,
    topk_weights: torch.Tensor,
) -> int:
    with _LIFECYCLE_LOCK:
        global _CTX_SEQ
        _CTX_SEQ += 1
        ctx_uid = _CTX_SEQ
        _CTX_LIVE[ctx_uid] = {
            "created_at": time.time(),
            "layer": layer,
            "topk_ids_id": topk_ids_id,
            "topk_ids_shape": topk_ids_shape,
            "topk_ids_bytes": topk_ids_bytes,
            "topk_weights_id": id(topk_weights),
            "topk_weights_bytes": _tensor_nbytes(topk_weights),
        }
    _kt_mem_log(
        rank,
        "ctx_forward",
        ctx_uid=ctx_uid,
        layer=layer,
        topk_ids_id=topk_ids_id,
        topk_ids_shape=topk_ids_shape,
        topk_ids_bytes=topk_ids_bytes,
        topk_weights_id=id(topk_weights),
        topk_weights_shape=tuple(topk_weights.shape),
        topk_weights_bytes=_tensor_nbytes(topk_weights),
    )
    return ctx_uid


def _ctx_backward_enter(rank: int, *, layer: int, ctx_uid: int) -> None:
    with _LIFECYCLE_LOCK:
        entry = _CTX_LIVE.get(ctx_uid)
    age_s = -1.0
    if entry is not None:
        age_s = max(0.0, time.time() - float(entry["created_at"]))
    _kt_mem_log(
        rank,
        "ctx_backward_enter",
        ctx_uid=ctx_uid,
        layer=layer,
        ctx_found=entry is not None,
        age_s=round(age_s, 6),
    )


def _ctx_backward_exit(rank: int, *, layer: int, ctx_uid: int) -> None:
    with _LIFECYCLE_LOCK:
        entry = _CTX_LIVE.pop(ctx_uid, None)
        live_count = len(_CTX_LIVE)
    age_s = -1.0
    if entry is not None:
        age_s = max(0.0, time.time() - float(entry["created_at"]))
    _kt_mem_log(
        rank,
        "ctx_backward_exit",
        ctx_uid=ctx_uid,
        layer=layer,
        ctx_found=entry is not None,
        age_s=round(age_s, 6),
        live_ctx_count=live_count,
    )


def _ctx_snapshot(rank: int, *, layer: int, tag: str, topk: int = 8) -> None:
    with _LIFECYCLE_LOCK:
        live = list(_CTX_LIVE.items())
    _kt_mem_log(
        rank,
        "ctx_snapshot",
        layer=layer,
        snapshot_tag=tag,
        live_ctx_count=len(live),
    )
    if not live:
        return
    now = time.time()
    ranked = sorted(live, key=lambda kv: now - float(kv[1]["created_at"]), reverse=True)[:topk]
    for ctx_uid, entry in ranked:
        age_s = max(0.0, now - float(entry["created_at"]))
        _kt_mem_log(
            rank,
            "ctx_live",
            ctx_uid=ctx_uid,
            layer=entry["layer"],
            age_s=round(age_s, 6),
            topk_ids_id=entry["topk_ids_id"],
            topk_ids_bytes=entry["topk_ids_bytes"],
            topk_weights_id=entry["topk_weights_id"],
            topk_weights_bytes=entry["topk_weights_bytes"],
            precomputed_output_id=entry.get("precomputed_output_id", -1),
            precomputed_output_bytes=entry.get("precomputed_output_bytes", 0),
        )


# =============================================================================
# Device Map Helpers
# =============================================================================


def _get_layers_prefix(config) -> str:
    arch = config.architectures[0] if getattr(config, "architectures", None) else ""
    if any(x in arch for x in ["Deepseek", "Qwen", "Mixtral", "Llama"]):
        return "model.layers"
    return "model.layers"


def _maybe_zero3_gathered_parameters(params: list[torch.nn.Parameter]):
    if not params:
        return nullcontext()
    try:
        from transformers.integrations import is_deepspeed_zero3_enabled
    except Exception:
        return nullcontext()
    if not is_deepspeed_zero3_enabled():
        return nullcontext()
    try:
        import deepspeed  # type: ignore
    except Exception:
        return nullcontext()
    return deepspeed.zero.GatheredParameters(params, modifier_rank=0)


def build_kt_device_map(config, kt_plugin, device: str = "cuda:0") -> dict[str, str | int]:
    """
    Build device_map for KT model loading with hybrid GPU/CPU expert placement.
    """
    moe_config = get_moe_arch_config(config)
    layers_prefix = _get_layers_prefix(config)
    num_layers = config.num_hidden_layers
    num_experts = moe_config.expert_num
    num_gpu_experts = getattr(kt_plugin, "kt_num_gpu_experts", 0) or 0

    device_map: dict[str, str | int] = {}

    device_map["model.embed_tokens"] = device
    device_map["model.norm"] = device
    device_map["lm_head"] = device

    for layer_idx in range(num_layers):
        layer_prefix = f"{layers_prefix}.{layer_idx}"
        device_map[layer_prefix] = device
        moe_prefix = f"{layer_prefix}.{moe_config.moe_layer_attr}"

        for expert_idx in range(num_experts):
            expert_key = f"{moe_prefix}.{moe_config.experts_attr}.{expert_idx}"
            if expert_idx < num_gpu_experts:
                device_map[expert_key] = device
            else:
                device_map[expert_key] = "cpu"

    logger.info(
        f"Built KT device_map: {num_gpu_experts} GPU experts, {num_experts - num_gpu_experts} CPU experts"
    )

    return device_map


def build_kt_device_map_simplified(config, kt_plugin, device: str = "cuda:0") -> dict[str, str | int]:
    """
    Simplified device_map builder: map full layers to GPU, override routed experts to CPU.
    """
    moe_config = get_moe_arch_config(config)
    layers_prefix = _get_layers_prefix(config)
    num_layers = config.num_hidden_layers
    num_gpu_experts = getattr(kt_plugin, "kt_num_gpu_experts", 0) or 0

    device_map: dict[str, str | int] = {}

    device_map["model.embed_tokens"] = device
    device_map["model.norm"] = device
    device_map["lm_head"] = device

    for layer_idx in range(num_layers):
        layer_prefix = f"{layers_prefix}.{layer_idx}"
        device_map[layer_prefix] = device

        experts_prefix = f"{layer_prefix}.{moe_config.moe_layer_attr}.{moe_config.experts_attr}"

        if num_gpu_experts == 0:
            device_map[experts_prefix] = "cpu"
        else:
            return build_kt_device_map(config, kt_plugin, device=device)

    logger.info("Built simplified KT device_map: all layers on GPU, routed experts on CPU")
    return device_map


def get_kt_loading_kwargs(
    config,
    kt_plugin,
    torch_dtype: torch.dtype | str | None = torch.bfloat16,
    trust_remote_code: bool | None = None,
    token: str | None = None,
) -> dict[str, Any]:
    """
    Get kwargs for AutoModel.from_pretrained() for KT loading.

    Defaults to loading all weights on CPU to avoid meta tensors, then you can
    call move_non_experts_to_gpu().
    """
    kwargs: dict[str, Any] = {
        "config": config,
        "torch_dtype": torch_dtype,
        "device_map": "cpu",
        "low_cpu_mem_usage": True,
    }
    if trust_remote_code is not None:
        kwargs["trust_remote_code"] = trust_remote_code
    if token is not None:
        kwargs["token"] = token
    return kwargs


def _get_model_container_and_layers(model: nn.Module, *, purpose: str) -> tuple[nn.Module, Any]:
    """
    Resolve the transformer layer container for KT integration.

    KT expects the transformer block stack to be accessible as `<container>.layers`. Some common
    wrappers (e.g. PEFT `PeftModel`, TRL value-head models, or DDP wrappers) may sit on top of the
    underlying HF model and hide `.layers` behind one or more indirections.
    """
    to_visit: list[nn.Module] = [model]
    visited: set[int] = set()
    visited_types: list[str] = []

    while to_visit:
        current = to_visit.pop(0)
        if id(current) in visited:
            continue
        visited.add(id(current))
        visited_types.append(type(current).__name__)

        layers = getattr(current, "layers", None)
        if layers is not None and isinstance(layers, (list, tuple, nn.ModuleList)):
            return current, layers

        for attr in ("model", "base_model", "pretrained_model", "module"):
            child = getattr(current, attr, None)
            if isinstance(child, nn.Module) and child is not current:
                to_visit.append(child)

        get_base_model = getattr(current, "get_base_model", None)
        if callable(get_base_model):
            try:
                base = get_base_model()
            except Exception:
                base = None
            if isinstance(base, nn.Module) and base is not current:
                to_visit.append(base)

    visited_preview = ", ".join(visited_types[:6])
    if len(visited_types) > 6:
        visited_preview += ", ..."

    raise KTAMXConfigError(
        f"Model does not expose a .model.layers or .layers attribute for KT {purpose}. "
        "Tried unwrapping via model/base_model/pretrained_model/module/get_base_model; "
        f"visited: {visited_preview}"
    )


def move_non_experts_to_gpu(
    model: nn.Module,
    moe_config: MOEArchConfig | None = None,
    device: str = "cuda:0",
) -> None:
    """
    Move non-expert parameters to GPU after loading (experts stay on CPU).
    """
    if moe_config is None:
        config = getattr(model, "config", None)
        if config is None:
            raise KTAMXConfigError("Model config is required to infer MoE architecture.")
        moe_config = get_moe_arch_config(config)

    container, layers = _get_model_container_and_layers(model, purpose="placement")

    if hasattr(container, "embed_tokens"):
        container.embed_tokens.to(device)
    if hasattr(container, "norm"):
        container.norm.to(device)
    if hasattr(model, "lm_head"):
        model.lm_head.to(device)

    for layer in layers:
        if hasattr(layer, "self_attn"):
            layer.self_attn.to(device)

        if hasattr(layer, "input_layernorm"):
            layer.input_layernorm.to(device)
        if hasattr(layer, "post_attention_layernorm"):
            layer.post_attention_layernorm.to(device)

        moe_module = getattr(layer, moe_config.moe_layer_attr, None)
        if moe_module is None or not hasattr(moe_module, moe_config.experts_attr):
            if hasattr(layer, "mlp"):
                layer.mlp.to(device)
            continue

        router = getattr(moe_module, moe_config.router_attr, None)
        if router is not None:
            router.to(device)

        if hasattr(moe_module, "shared_experts") and moe_module.shared_experts is not None:
            moe_module.shared_experts.to(device)

    logger.info(f"Moved non-expert parameters to {device}")


def get_expert_device(model: nn.Module, moe_config: MOEArchConfig | None = None) -> str:
    """
    Get the device type of MoE experts.
    """
    if moe_config is None:
        config = getattr(model, "config", None)
        if config is None:
            return "unknown"
        moe_config = get_moe_arch_config(config)

    try:
        _, layers = _get_model_container_and_layers(model, purpose="expert device probing")
    except KTAMXConfigError:
        return "unknown"

    for layer in layers:
        moe_module = getattr(layer, moe_config.moe_layer_attr, None)
        if moe_module is None:
            continue
        experts = getattr(moe_module, moe_config.experts_attr, None)
        if not experts:
            continue
        first_expert = experts[0]
        gate_name = moe_config.weight_names[0]
        gate_proj = getattr(first_expert, gate_name, None)
        if gate_proj is not None:
            return str(gate_proj.weight.device.type)

    return "unknown"


# =============================================================================
# Weight Extraction
# =============================================================================


def extract_moe_weights(
    moe_module: nn.Module, moe_config: MOEArchConfig
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extract MoE expert weights from the module.

    Returns (gate_proj, up_proj, down_proj) with shape
    [expert_num, out_features, in_features].
    """
    experts = getattr(moe_module, moe_config.experts_attr)
    gate_name, up_name, down_name = moe_config.weight_names

    gather_params: list[torch.nn.Parameter] = []
    for expert in experts:
        for weight_name in (gate_name, up_name, down_name):
            proj = getattr(expert, weight_name, None)
            if proj is not None and hasattr(proj, "weight"):
                # Handle PEFT LoRA wrapped modules
                weight = proj.weight
                if isinstance(weight, torch.Tensor):
                    gather_params.append(weight)
                elif hasattr(weight, "data"):
                    gather_params.append(weight.data)

    with _maybe_zero3_gathered_parameters(gather_params):
        gate_weights = []
        up_weights = []
        down_weights = []

        for expert in experts:
            # Handle PEFT LoRA wrapped modules - get weight tensor properly
            gate_proj = getattr(expert, gate_name)
            up_proj_mod = getattr(expert, up_name)
            down_proj_mod = getattr(expert, down_name)

            # Get weight tensors, handling both regular Linear and PEFT LoRA wrapped
            def get_weight_tensor(mod):
                weight = mod.weight
                if isinstance(weight, torch.Tensor):
                    return weight.data
                elif hasattr(weight, "data"):
                    return weight.data
                else:
                    raise ValueError(f"Cannot extract weight from {type(mod)}, weight type={type(weight)}")

            gate_weights.append(get_weight_tensor(gate_proj))
            up_weights.append(get_weight_tensor(up_proj_mod))
            down_weights.append(get_weight_tensor(down_proj_mod))

    gate_proj = torch.stack(gate_weights, dim=0)
    up_proj = torch.stack(up_weights, dim=0)
    down_proj = torch.stack(down_weights, dim=0)

    return gate_proj, up_proj, down_proj


def _clear_original_expert_weights(moe_module: nn.Module, moe_config: MOEArchConfig) -> None:
    """
    Clear original expert weights to free memory after KT weights are loaded.
    """
    experts = getattr(moe_module, moe_config.experts_attr, None)
    if experts is None:
        return

    def _iter_weight_params():
        for expert in experts:
            for weight_name in moe_config.weight_names:
                proj = getattr(expert, weight_name, None)
                if proj is None or not hasattr(proj, "weight"):
                    continue

                parametrizations = getattr(proj, "parametrizations", None)
                parametrized_weight = getattr(parametrizations, "weight", None) if parametrizations is not None else None
                if parametrized_weight is not None:
                    original = getattr(parametrized_weight, "original", None)
                    if isinstance(original, torch.nn.Parameter):
                        yield proj, parametrized_weight, "original", original
                        continue

                direct_weight = getattr(proj, "_parameters", {}).get("weight")
                if isinstance(direct_weight, torch.nn.Parameter):
                    yield proj, proj, "weight", direct_weight
                    continue

                # Fallback: `weight` can be a non-settable property (e.g. parametrizations) or a non-Parameter.
                weight_attr = getattr(proj, "weight", None)
                if isinstance(weight_attr, torch.nn.Parameter):
                    yield proj, proj, "weight", weight_attr

    gather_params: list[torch.nn.Parameter] = []
    for _, _, _, weight_param in _iter_weight_params():
        gather_params.append(weight_param)

    replaced_count = 0

    with _maybe_zero3_gathered_parameters(gather_params):
        for proj, container, param_name, weight_param in _iter_weight_params():
            original_dtype = weight_param.dtype

            # Create a CPU tensor with the correct shape but NO physical memory.
            # torch.empty(shape, device="cpu") unfortunately touches pages via the
            # allocator, consuming real RSS.  Instead, allocate a 1-byte storage and
            # use set_ to give it the original shape with zero strides.  The tensor
            # is "valid" (correct dtype, device, shape) so PEFT can discover
            # in/out features, but its storage is essentially zero-cost.
            # NOTE: reading element values from this tensor is undefined — it is
            # only used for shape/dtype discovery by PEFT.
            tiny_storage = torch.UntypedStorage(1, device="cpu")
            fake_tensor = torch.tensor([], dtype=original_dtype, device="cpu").set_(
                tiny_storage, storage_offset=0, size=weight_param.shape,
                stride=[0] * len(weight_param.shape),
            )
            new_param = nn.Parameter(fake_tensor, requires_grad=False)
            replaced_count += 1

            # Avoid `KeyError: attribute 'weight' already exists` for parametrized modules
            # where `weight` is a property and the real parameter lives elsewhere.
            container_params = getattr(container, "_parameters", {})
            if isinstance(container_params, dict) and param_name in container_params:
                container_params[param_name] = new_param
                continue

            if hasattr(container, param_name):
                logger.debug(
                    f"Skipping clearing expert weight {type(proj).__name__}.{param_name}: "
                    "attribute exists but is not a registered parameter."
                )
                continue

            try:
                setattr(container, param_name, new_param)
            except Exception as exc:
                logger.warning(
                    f"Failed to clear expert weight {type(proj).__name__}.{param_name}: {exc}"
                )

    logger.info(f"Replaced {replaced_count} expert weight params")


# =============================================================================
# LoRA Experts Modules
# =============================================================================


class LoRAExpertMLP(nn.Module):
    """Single LoRA Expert with SwiGLU activation structure."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.le_gate = nn.Linear(hidden_size, intermediate_size, bias=False, device=device, dtype=dtype)
        self.le_up = nn.Linear(hidden_size, intermediate_size, bias=False, device=device, dtype=dtype)
        self.le_down = nn.Linear(intermediate_size, hidden_size, bias=False, device=device, dtype=dtype)
        self.act_fn = nn.SiLU()

        nn.init.zeros_(self.le_down.weight)
        nn.init.kaiming_uniform_(self.le_gate.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.le_up.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.le_down(self.act_fn(self.le_gate(x)) * self.le_up(x))


class LoRAExperts(nn.Module):
    """LoRA Experts module containing multiple LoRA Expert MLPs."""

    def __init__(
        self,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.experts = nn.ModuleList(
            [LoRAExpertMLP(hidden_size, intermediate_size, device, dtype) for _ in range(num_experts)]
        )
        self.num_experts = num_experts

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        output = torch.zeros_like(hidden_states)
        for expert in self.experts:
            output = output + expert(hidden_states)
        return output / self.num_experts


# =============================================================================
# kt_weight_path Loading Functions
# =============================================================================


@dataclass
class INT8ExpertWeights:
    """Container for INT8 expert weights with scales."""

    gate_proj: torch.Tensor
    gate_scale: torch.Tensor
    up_proj: torch.Tensor
    up_scale: torch.Tensor
    down_proj: torch.Tensor
    down_scale: torch.Tensor


def _find_safetensor_files(kt_weight_path: str) -> list[str]:
    if not os.path.isdir(kt_weight_path):
        raise FileNotFoundError(f"kt_weight_path directory not found: {kt_weight_path}")

    safetensor_files = []
    for file in sorted(os.listdir(kt_weight_path)):
        if file.endswith(".safetensors"):
            safetensor_files.append(os.path.join(kt_weight_path, file))

    if not safetensor_files:
        raise FileNotFoundError(f"No safetensors files found in {kt_weight_path}")

    return safetensor_files


def _load_kt_weight_index(kt_weight_path: str) -> dict[str, str]:
    if not SAFETENSORS_AVAILABLE:
        raise ImportError("safetensors is required for loading kt_weight_path")

    index = {}
    safetensor_files = _find_safetensor_files(kt_weight_path)

    for file_path in safetensor_files:
        with safe_open(file_path, framework="pt") as f:
            for key in f.keys():
                index[key] = file_path

    logger.info(f"Indexed {len(index)} tensors from {len(safetensor_files)} safetensors files")
    return index


def _resolve_checkpoint_files(
    model_name_or_path: str,
    cache_dir: str | None = None,
    revision: str | None = None,
    token: str | None = None,
    trust_remote_code: bool | None = None,
) -> tuple[list[str] | None, dict | None]:
    try:
        from transformers.modeling_utils import _get_resolved_checkpoint_files
    except Exception:
        return None, None

    try:
        checkpoint_files, sharded_metadata = _get_resolved_checkpoint_files(
            pretrained_model_name_or_path=model_name_or_path,
            subfolder="",
            variant=None,
            gguf_file=None,
            from_tf=False,
            from_flax=False,
            use_safetensors=None,
            cache_dir=cache_dir,
            force_download=False,
            proxies=None,
            local_files_only=False,
            token=token,
            user_agent={"file_type": "model", "framework": "pytorch"},
            revision=revision or "main",
            commit_hash=None,
            is_remote_code=bool(trust_remote_code),
            transformers_explicit_filename=None,
        )
    except Exception:
        return None, None

    return checkpoint_files, sharded_metadata


def _dequant_fp8_experts(weights: list[torch.Tensor], scales: list[torch.Tensor | None], block_size: tuple[int, int]) -> torch.Tensor:
    """Dequantize a list of FP8 expert weights and stack them (batched, vectorized).

    Args:
        weights: list of [out, in] float8_e4m3fn tensors (one per expert)
        scales: list of [out//bs_m, in//bs_n] scale_inv tensors (one per expert, may be None)
        block_size: (bs_m, bs_n)

    Returns:
        Stacked BF16 tensor of shape [num_experts, out, in]
    """
    has_scales = scales[0] is not None
    if not has_scales:
        return torch.stack(weights, dim=0).to(torch.bfloat16).cpu().contiguous()

    bs_m, bs_n = block_size
    n = len(weights)
    out_features, in_features = weights[0].shape

    # Stack all experts: [N, out, in] fp8 → reshape to blocks → bf16
    w = torch.stack(weights, dim=0)  # [N, out, in] fp8
    w = w.reshape(n, out_features // bs_m, bs_m, in_features // bs_n, bs_n)
    w = w.to(torch.bfloat16)

    # Stack all scales: [N, out//bs_m, in//bs_n] → bf16, broadcast multiply
    s = torch.stack(scales, dim=0).to(torch.bfloat16)  # [N, out//bs_m, in//bs_n]
    w = w * s[:, :, None, :, None]

    return w.reshape(n, out_features, in_features).contiguous()


def load_experts_from_checkpoint_files(
    checkpoint_files: list[str],
    sharded_metadata: dict | None,
    layers_prefix: str,
    moe_config: MOEArchConfig,
    layer_idx: int,
    block_size: tuple[int, int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not SAFETENSORS_AVAILABLE:
        raise ImportError("safetensors is required for loading experts from checkpoint files")

    if not checkpoint_files:
        raise FileNotFoundError("checkpoint_files is empty")

    t0 = time.time()

    weight_map = None
    base_dir = os.path.dirname(checkpoint_files[0])
    if sharded_metadata is not None:
        weight_map = sharded_metadata.get("weight_map", None)

    gate_name, up_name, down_name = moe_config.weight_names
    keys = []
    for expert_idx in range(moe_config.expert_num):
        base = f"{layers_prefix}.{layer_idx}.{moe_config.moe_layer_attr}.{moe_config.experts_attr}.{expert_idx}"
        keys.append(f"{base}.{gate_name}.weight")
        keys.append(f"{base}.{gate_name}.weight_scale_inv")
        keys.append(f"{base}.{up_name}.weight")
        keys.append(f"{base}.{up_name}.weight_scale_inv")
        keys.append(f"{base}.{down_name}.weight")
        keys.append(f"{base}.{down_name}.weight_scale_inv")

    keys_by_file: dict[str, list[str]] = {}
    mapped_count = 0
    unmapped_count = 0
    for key in keys:
        if weight_map is not None:
            filename = weight_map.get(key)
            if filename is None:
                unmapped_count += 1
                continue
            mapped_count += 1
            file_path = os.path.join(base_dir, filename)
        else:
            file_path = checkpoint_files[0]
        keys_by_file.setdefault(file_path, []).append(key)

    print(
        f"[kt_moe] Layer {layer_idx}: key mapping done in {time.time()-t0:.1f}s — "
        f"total_keys={len(keys)}, mapped={mapped_count}, unmapped={unmapped_count}, "
        f"files_to_open={len(keys_by_file)}",
        flush=True,
    )

    t1 = time.time()
    tensor_map: dict[str, torch.Tensor] = {}
    for file_idx, (file_path, file_keys) in enumerate(keys_by_file.items()):
        with safe_open(file_path, framework="pt") as f:
            available_keys = set(f.keys())
            for key in file_keys:
                if key in available_keys:
                    tensor_map[key] = f.get_tensor(key)
        if file_idx == 0:
            print(
                f"[kt_moe] Layer {layer_idx}: first file loaded ({os.path.basename(file_path)}, "
                f"{len(file_keys)} keys) in {time.time()-t1:.1f}s",
                flush=True,
            )

    print(
        f"[kt_moe] Layer {layer_idx}: all files loaded in {time.time()-t1:.1f}s — "
        f"tensor_map has {len(tensor_map)} tensors",
        flush=True,
    )

    gate_weights = []
    up_weights = []
    down_weights = []
    gate_scales = []
    up_scales = []
    down_scales = []
    for expert_idx in range(moe_config.expert_num):
        base = f"{layers_prefix}.{layer_idx}.{moe_config.moe_layer_attr}.{moe_config.experts_attr}.{expert_idx}"
        gate_key = f"{base}.{gate_name}.weight"
        up_key = f"{base}.{up_name}.weight"
        down_key = f"{base}.{down_name}.weight"
        if gate_key not in tensor_map or up_key not in tensor_map or down_key not in tensor_map:
            raise FileNotFoundError(f"Missing expert weights for layer {layer_idx}, expert {expert_idx}")
        gate_weights.append(tensor_map[gate_key])
        up_weights.append(tensor_map[up_key])
        down_weights.append(tensor_map[down_key])
        gate_scales.append(tensor_map.get(f"{base}.{gate_name}.weight_scale_inv"))
        up_scales.append(tensor_map.get(f"{base}.{up_name}.weight_scale_inv"))
        down_scales.append(tensor_map.get(f"{base}.{down_name}.weight_scale_inv"))

    # Check if weights are FP8 and need dequantization
    t2 = time.time()
    is_fp8 = gate_weights[0].dtype == torch.float8_e4m3fn
    if is_fp8:
        if block_size is None:
            block_size = (128, 128)
        print(
            f"[kt_moe] Layer {layer_idx}: FP8 expert weights detected, "
            f"dequantizing with block_size={block_size} "
            f"(has_scales={gate_scales[0] is not None})",
            flush=True,
        )
        gate_proj = _dequant_fp8_experts(gate_weights, gate_scales, block_size)
        up_proj = _dequant_fp8_experts(up_weights, up_scales, block_size)
        down_proj = _dequant_fp8_experts(down_weights, down_scales, block_size)
    else:
        gate_proj = torch.stack(gate_weights, dim=0).cpu().to(torch.bfloat16).contiguous()
        up_proj = torch.stack(up_weights, dim=0).cpu().to(torch.bfloat16).contiguous()
        down_proj = torch.stack(down_weights, dim=0).cpu().to(torch.bfloat16).contiguous()

    print(
        f"[kt_moe] Layer {layer_idx}: done — dtype={gate_proj.dtype}, shape={gate_proj.shape}, "
        f"dequant={time.time()-t2:.1f}s, total={time.time()-t0:.1f}s",
        flush=True,
    )
    return gate_proj, up_proj, down_proj


def load_experts_from_kt_weight_path(
    kt_weight_path: str,
    layer_idx: int,
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
) -> INT8ExpertWeights:
    """Load INT8 preprocessed expert weights from kt_weight_path for a specific layer."""
    if not SAFETENSORS_AVAILABLE:
        raise ImportError("safetensors is required for loading kt_weight_path")

    index = _load_kt_weight_index(kt_weight_path)

    numa_count = 0
    test_key_prefix = f"blk.{layer_idx}.ffn_gate_exps.0.numa."
    for key in index.keys():
        if key.startswith(test_key_prefix) and key.endswith(".weight"):
            numa_idx = int(key.split("numa.")[1].split(".")[0])
            numa_count = max(numa_count, numa_idx + 1)

    if numa_count == 0:
        raise FileNotFoundError(
            f"No weights found for layer {layer_idx} in {kt_weight_path}. "
            f"Expected keys like 'blk.{layer_idx}.ffn_gate_exps.0.numa.0.weight'"
        )

    logger.info(
        f"Loading INT8 weights for layer {layer_idx}: {num_experts} experts, {numa_count} NUMA partitions"
    )

    gate_weights_list = []
    gate_scales_list = []
    up_weights_list = []
    up_scales_list = []
    down_weights_list = []
    down_scales_list = []

    for expert_idx in range(num_experts):
        gate_w_parts = []
        gate_s_parts = []
        for numa_idx in range(numa_count):
            w_key = f"blk.{layer_idx}.ffn_gate_exps.{expert_idx}.numa.{numa_idx}.weight"
            s_key = f"blk.{layer_idx}.ffn_gate_exps.{expert_idx}.numa.{numa_idx}.scale"

            if w_key not in index:
                raise FileNotFoundError(f"Weight key not found: {w_key}")

            with safe_open(index[w_key], framework="pt") as f:
                gate_w_parts.append(f.get_tensor(w_key))
                gate_s_parts.append(f.get_tensor(s_key))

        gate_w = torch.cat(gate_w_parts, dim=0)
        gate_s = torch.cat(gate_s_parts, dim=0)
        gate_w = gate_w.view(intermediate_size, hidden_size)

        gate_weights_list.append(gate_w)
        gate_scales_list.append(gate_s)

        up_w_parts = []
        up_s_parts = []
        for numa_idx in range(numa_count):
            w_key = f"blk.{layer_idx}.ffn_up_exps.{expert_idx}.numa.{numa_idx}.weight"
            s_key = f"blk.{layer_idx}.ffn_up_exps.{expert_idx}.numa.{numa_idx}.scale"

            if w_key not in index:
                raise FileNotFoundError(f"Weight key not found: {w_key}")

            with safe_open(index[w_key], framework="pt") as f:
                up_w_parts.append(f.get_tensor(w_key))
                up_s_parts.append(f.get_tensor(s_key))

        up_w = torch.cat(up_w_parts, dim=0)
        up_s = torch.cat(up_s_parts, dim=0)
        up_w = up_w.view(intermediate_size, hidden_size)

        up_weights_list.append(up_w)
        up_scales_list.append(up_s)

        down_w_parts = []
        down_s_parts = []
        for numa_idx in range(numa_count):
            w_key = f"blk.{layer_idx}.ffn_down_exps.{expert_idx}.numa.{numa_idx}.weight"
            s_key = f"blk.{layer_idx}.ffn_down_exps.{expert_idx}.numa.{numa_idx}.scale"

            if w_key not in index:
                raise FileNotFoundError(f"Weight key not found: {w_key}")

            with safe_open(index[w_key], framework="pt") as f:
                down_w_parts.append(f.get_tensor(w_key))
                down_s_parts.append(f.get_tensor(s_key))

        down_w = torch.cat(down_w_parts, dim=0)
        down_s = torch.cat(down_s_parts, dim=0)
        down_w = down_w.view(hidden_size, intermediate_size)

        down_weights_list.append(down_w)
        down_scales_list.append(down_s)

    gate_proj = torch.stack(gate_weights_list, dim=0)
    gate_scale = torch.stack(gate_scales_list, dim=0)
    up_proj = torch.stack(up_weights_list, dim=0)
    up_scale = torch.stack(up_scales_list, dim=0)
    down_proj = torch.stack(down_weights_list, dim=0)
    down_scale = torch.stack(down_scales_list, dim=0)

    return INT8ExpertWeights(
        gate_proj=gate_proj,
        gate_scale=gate_scale,
        up_proj=up_proj,
        up_scale=up_scale,
        down_proj=down_proj,
        down_scale=down_scale,
    )


# =============================================================================
# KTMoE Autograd Function
# =============================================================================


class KTMoEFunction(torch.autograd.Function):
    """Unified autograd function for KTMoE forward/backward."""

    @staticmethod
    def forward(
        ctx,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        wrapper: Any,
        lora_ref: torch.Tensor,
        hidden_size: int,
        num_experts_per_tok: int,
        layer_idx: int,
        training: bool,
        train_lora: bool,
        all_qlens: list[int] | tuple[int, ...] | None,
    ) -> torch.Tensor:

        original_device = hidden_states.device
        original_dtype = hidden_states.dtype
        batch_size, seq_len, _ = hidden_states.shape
        qlen = batch_size * seq_len

        import torch.distributed as dist
        dist_on = dist.is_initialized() and dist.get_world_size() > 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist_on else 1

        if KT_MEM_LOG:
            _kt_mem_log_wrapper(
                rank,
                "autograd_forward_state",
                wrapper,
                device=original_device if original_device.type == "cuda" else None,
                layer=layer_idx,
                training=training,
                train_lora=train_lora,
                qlen=qlen,
            )

        if KT_LIFECYCLE_LOG and rank == 0:
            ctx._kt_ctx_uid = _ctx_register(
                rank,
                layer=layer_idx,
                topk_ids_id=id(topk_ids),
                topk_ids_shape=tuple(topk_ids.shape),
                topk_ids_bytes=_tensor_nbytes(topk_ids),
                topk_weights=topk_weights,
            )
        else:
            ctx._kt_ctx_uid = -1

        ctx.use_broadcast = wrapper is None

        # ---- Sync CPU expert result and distribute ----
        if dist_on:
            if all_qlens is None:
                all_qlens_list = _all_gather_qlens(qlen, original_device, world_size)
            else:
                all_qlens_list = [int(q) for q in all_qlens]
                if len(all_qlens_list) != world_size:
                    raise RuntimeError(
                        f"all_qlens length mismatch: got {len(all_qlens_list)}, expected {world_size}"
                    )
            if int(all_qlens_list[rank]) != qlen:
                raise RuntimeError(
                    f"Rank {rank} qlen mismatch: local={qlen}, all_qlens[{rank}]={all_qlens_list[rank]}"
                )
            total_qlen = sum(all_qlens_list)

            # Rank 0: sync CPU result and split by real lengths
            if rank == 0:
                if KT_MEM_LOG:
                    _kt_mem_log_wrapper(
                        rank,
                        "autograd_rank0_before_sync",
                        wrapper,
                        device=original_device if original_device.type == "cuda" else None,
                        layer=layer_idx,
                    )
                cpu_output = wrapper.sync_forward_sft(output_device=original_device)
                cpu_output = cpu_output.to(dtype=original_dtype).view(total_qlen, hidden_size)
                _track_lifecycle_tensor(
                    rank,
                    layer_idx,
                    "autograd.rank0.cpu_output",
                    cpu_output,
                )
                offsets = _qlen_offsets(all_qlens_list)
                scatter_list = [cpu_output[offsets[i] : offsets[i + 1]].contiguous() for i in range(world_size)]
                _track_lifecycle_collection(
                    rank,
                    layer_idx,
                    "autograd.rank0.scatter_list",
                    scatter_list,
                )
                if KT_MEM_LOG:
                    _kt_mem_log_wrapper(
                        rank,
                        "autograd_rank0_after_sync",
                        wrapper,
                        device=original_device if original_device.type == "cuda" else None,
                        layer=layer_idx,
                        cpu_output_bytes=_tensor_nbytes(cpu_output),
                        total_qlen=total_qlen,
                    )
            else:
                scatter_list = None

            output_flat = _dist_scatter_varlen_from_rank0(
                rank0_chunks=scatter_list,
                all_qlens=all_qlens_list,
                rank=rank,
                world_size=world_size,
                feature_shape=(hidden_size,),
                device=original_device,
                dtype=original_dtype,
            )
            _track_lifecycle_tensor(
                rank,
                layer_idx,
                "autograd.output_flat",
                output_flat,
            )
            output = output_flat.view(batch_size, seq_len, hidden_size)
            _track_lifecycle_tensor(
                rank,
                layer_idx,
                "autograd.output",
                output,
            )
            del output_flat
        elif wrapper is not None:
            # Single-GPU: sync directly
            if KT_MEM_LOG:
                _kt_mem_log_wrapper(
                    rank,
                    "autograd_single_before_sync",
                    wrapper,
                    device=original_device if original_device.type == "cuda" else None,
                    layer=layer_idx,
                )
            cpu_output = wrapper.sync_forward_sft(output_device=original_device)
            _track_lifecycle_tensor(
                rank,
                layer_idx,
                "autograd.single.cpu_output",
                cpu_output,
            )
            output = cpu_output.view(batch_size, seq_len, hidden_size).to(dtype=original_dtype)
            _track_lifecycle_tensor(
                rank,
                layer_idx,
                "autograd.output",
                output,
            )
            if KT_MEM_LOG:
                _kt_mem_log_wrapper(
                    rank,
                    "autograd_single_after_sync",
                    wrapper,
                    device=original_device if original_device.type == "cuda" else None,
                    layer=layer_idx,
                    cpu_output_bytes=_tensor_nbytes(cpu_output),
                )
        else:
            # Broadcast-only rank (no wrapper)
            output = torch.empty(
                batch_size, seq_len, hidden_size, device=original_device, dtype=original_dtype
            )

        ctx.wrapper = wrapper
        ctx.hidden_size = hidden_size
        ctx.qlen = qlen
        ctx.batch_size = batch_size
        ctx.seq_len = seq_len
        ctx.original_device = original_device
        ctx.original_dtype = original_dtype
        ctx.weights_shape = topk_weights.shape
        ctx.weights_dtype = topk_weights.dtype
        ctx.weights_device = topk_weights.device
        ctx.dist_on = dist_on
        ctx.world_size = world_size
        ctx.all_qlens = all_qlens_list if dist_on else None
        ctx.num_experts_per_tok = num_experts_per_tok
        ctx.layer_idx = layer_idx

        # Save a sentinel tensor so non-reentrant checkpoint's saved_tensors
        # hooks can intercept it.  When backward accesses ctx.saved_tensors,
        # the checkpoint unpack hook triggers a full recompute of the decoder
        # layer — which re-runs the MoE forward with save_for_backward=True,
        # populating the C++ cache BEFORE this backward proceeds.
        # Without this, MoE backward runs before the recompute (MoE comes
        # after attention in forward order → its backward runs first), and
        # the C++ cache is empty when first-forward cache-skip is active.
        ctx.save_for_backward(hidden_states.new_empty(()))

        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # Access saved_tensors FIRST — under non-reentrant checkpoint this
        # triggers the unpack hook which runs a full decoder-layer recompute,
        # populating the C++ cache before we call wrapper.backward().
        _ = ctx.saved_tensors

        qlen = ctx.qlen
        hidden_size = ctx.hidden_size
        batch_size = ctx.batch_size
        seq_len = ctx.seq_len
        dist_on = ctx.dist_on
        world_size = ctx.world_size
        num_experts_per_tok = ctx.num_experts_per_tok

        import torch.distributed as dist
        rank = dist.get_rank() if dist.is_initialized() else 0
        ctx_uid = int(getattr(ctx, "_kt_ctx_uid", -1))
        if KT_MEM_LOG:
            _kt_mem_log(
                rank,
                "autograd_backward_entry",
                layer=getattr(ctx, "layer_idx", -1),
                ctx_uid=ctx_uid,
                dist_on=bool(dist_on),
                use_broadcast=bool(getattr(ctx, "use_broadcast", False)),
                qlen=qlen,
                hidden_size=hidden_size,
                grad_output_shape=str(tuple(grad_output.shape)),
                grad_output_dtype=str(grad_output.dtype),
                grad_output_requires_grad=bool(getattr(grad_output, "requires_grad", False)),
            )
        if KT_LIFECYCLE_LOG and rank == 0 and ctx_uid >= 0:
            _ctx_backward_enter(rank, layer=getattr(ctx, "layer_idx", -1), ctx_uid=ctx_uid)

        if dist_on:
            all_qlens = getattr(ctx, "all_qlens", None)
            if all_qlens is None or len(all_qlens) != world_size:
                all_qlens = _all_gather_qlens(qlen, ctx.original_device, world_size)
            else:
                all_qlens = [int(q) for q in all_qlens]
            if int(all_qlens[rank]) != qlen:
                raise RuntimeError(
                    f"Backward qlen mismatch on rank {rank}: local={qlen}, all_qlens[{rank}]={all_qlens[rank]}"
                )

            grad_out_flat = grad_output.view(qlen, hidden_size).contiguous()
            _track_lifecycle_tensor(
                rank,
                getattr(ctx, "layer_idx", -1),
                "autograd.backward.grad_out_flat",
                grad_out_flat,
            )

            gathered_go = _dist_gather_varlen_to_rank0(
                grad_out_flat,
                all_qlens=all_qlens,
                rank=rank,
                world_size=world_size,
            )
            if rank == 0:
                _track_lifecycle_collection(
                    rank,
                    getattr(ctx, "layer_idx", -1),
                    "autograd.backward.gathered_go",
                    gathered_go,
                )
                all_go = torch.cat(gathered_go, dim=0)
                _track_lifecycle_tensor(
                    rank,
                    getattr(ctx, "layer_idx", -1),
                    "autograd.backward.rank0.all_go",
                    all_go,
                )
                total_qlen = int(all_go.shape[0])

                if KT_MEM_LOG:
                    _kt_mem_log_wrapper(
                        rank,
                        "backward_rank0_before_wrapper",
                        ctx.wrapper,
                        device=ctx.original_device if ctx.original_device.type == "cuda" else None,
                        layer=getattr(ctx, "layer_idx", -1),
                        qlen=total_qlen,
                        grad_output_bytes=_tensor_nbytes(all_go),
                    )
                backward_out = ctx.wrapper.backward(
                    all_go,
                    output_device=ctx.original_device,
                )
                if isinstance(backward_out, tuple) and len(backward_out) == 2:
                    all_grad_input, all_grad_weights = backward_out
                elif isinstance(backward_out, tuple) and len(backward_out) == 3:
                    all_grad_input, _, all_grad_weights = backward_out
                else:
                    raise ValueError("KTMoEWrapper.backward returned unexpected format.")
                if KT_MEM_LOG:
                    _kt_mem_log_wrapper(
                        rank,
                        "backward_rank0_after_wrapper",
                        ctx.wrapper,
                        device=ctx.original_device if ctx.original_device.type == "cuda" else None,
                        layer=getattr(ctx, "layer_idx", -1),
                        grad_input_bytes=_tensor_nbytes(all_grad_input),
                        grad_weights_bytes=_tensor_nbytes(all_grad_weights),
                    )

                all_grad_input = all_grad_input.to(dtype=ctx.original_dtype).view(total_qlen, hidden_size)
                all_grad_weights = all_grad_weights.to(dtype=torch.bfloat16).view(total_qlen, num_experts_per_tok)
                _track_lifecycle_tensor(
                    rank,
                    getattr(ctx, "layer_idx", -1),
                    "autograd.backward.rank0.all_grad_input",
                    all_grad_input,
                )

                offsets = _qlen_offsets(all_qlens)
                scatter_gi = [all_grad_input[offsets[i] : offsets[i + 1]].contiguous() for i in range(world_size)]
                scatter_gw = [all_grad_weights[offsets[i] : offsets[i + 1]].contiguous() for i in range(world_size)]
                _track_lifecycle_collection(
                    rank,
                    getattr(ctx, "layer_idx", -1),
                    "autograd.backward.rank0.scatter_gi",
                    scatter_gi,
                )
                _track_lifecycle_collection(
                    rank,
                    getattr(ctx, "layer_idx", -1),
                    "autograd.backward.rank0.scatter_gw",
                    scatter_gw,
                )
            else:
                scatter_gi = None
                scatter_gw = None

            grad_input_flat = _dist_scatter_varlen_from_rank0(
                rank0_chunks=scatter_gi,
                all_qlens=all_qlens,
                rank=rank,
                world_size=world_size,
                feature_shape=(hidden_size,),
                device=ctx.original_device,
                dtype=ctx.original_dtype,
            )
            grad_weights_flat = _dist_scatter_varlen_from_rank0(
                rank0_chunks=scatter_gw,
                all_qlens=all_qlens,
                rank=rank,
                world_size=world_size,
                feature_shape=(num_experts_per_tok,),
                device=ctx.weights_device,
                dtype=torch.bfloat16,
            )
            _track_lifecycle_tensor(
                rank,
                getattr(ctx, "layer_idx", -1),
                "autograd.backward.grad_input_flat",
                grad_input_flat,
            )
            _track_lifecycle_tensor(
                rank,
                getattr(ctx, "layer_idx", -1),
                "autograd.backward.grad_weights_flat",
                grad_weights_flat,
            )
            grad_input = grad_input_flat.view(batch_size, seq_len, hidden_size)
            grad_weights = grad_weights_flat.view(ctx.weights_shape).to(dtype=ctx.weights_dtype)
            _track_lifecycle_tensor(
                rank,
                getattr(ctx, "layer_idx", -1),
                "autograd.backward.grad_input",
                grad_input,
            )
            _track_lifecycle_tensor(
                rank,
                getattr(ctx, "layer_idx", -1),
                "autograd.backward.grad_weights",
                grad_weights,
            )

        elif not ctx.use_broadcast:
            # ---- Single-GPU path ----
            grad_output_flat = grad_output.view(qlen, hidden_size)
            _track_lifecycle_tensor(
                rank,
                getattr(ctx, "layer_idx", -1),
                "autograd.backward.grad_output_flat",
                grad_output_flat,
            )
            if KT_MEM_LOG:
                _kt_mem_log_wrapper(
                    rank,
                    "backward_single_before_wrapper",
                    ctx.wrapper,
                    device=ctx.original_device if ctx.original_device.type == "cuda" else None,
                    layer=getattr(ctx, "layer_idx", -1),
                    qlen=qlen,
                    grad_output_bytes=_tensor_nbytes(grad_output_flat),
                )
            backward_out = ctx.wrapper.backward(
                grad_output_flat,
                output_device=ctx.original_device,
            )
            ctx.wrapper._kt_has_cached_forward = False
            if isinstance(backward_out, tuple) and len(backward_out) == 2:
                grad_input, grad_weights = backward_out
            elif isinstance(backward_out, tuple) and len(backward_out) == 3:
                grad_input, _, grad_weights = backward_out
            else:
                raise ValueError("KTMoEWrapper.backward returned unexpected format.")
            if KT_MEM_LOG:
                _kt_mem_log_wrapper(
                    rank,
                    "backward_single_after_wrapper",
                    ctx.wrapper,
                    device=ctx.original_device if ctx.original_device.type == "cuda" else None,
                    layer=getattr(ctx, "layer_idx", -1),
                    grad_input_bytes=_tensor_nbytes(grad_input),
                    grad_weights_bytes=_tensor_nbytes(grad_weights),
                )
            grad_input = grad_input.view(batch_size, seq_len, hidden_size).to(dtype=ctx.original_dtype)
            grad_weights = grad_weights.to(dtype=torch.bfloat16)
            _track_lifecycle_tensor(
                rank,
                getattr(ctx, "layer_idx", -1),
                "autograd.backward.grad_input",
                grad_input,
            )
            _track_lifecycle_tensor(
                rank,
                getattr(ctx, "layer_idx", -1),
                "autograd.backward.grad_weights",
                grad_weights,
            )
        else:
            # No wrapper, no dist — shouldn't happen in normal flow
            grad_input = torch.zeros(batch_size, seq_len, hidden_size, device=ctx.original_device, dtype=ctx.original_dtype)
            grad_weights = torch.zeros(ctx.weights_shape, device=ctx.weights_device, dtype=ctx.weights_dtype)

        if KT_LIFECYCLE_LOG and rank == 0:
            with _LIFECYCLE_LOCK:
                global _LIFECYCLE_BACKWARD_COUNT
                _LIFECYCLE_BACKWARD_COUNT += 1
                backward_count = _LIFECYCLE_BACKWARD_COUNT
            if ctx_uid >= 0:
                _ctx_backward_exit(rank, layer=getattr(ctx, "layer_idx", -1), ctx_uid=ctx_uid)
            if backward_count % 16 == 0:
                _log_lifecycle_snapshot(
                    rank,
                    layer=getattr(ctx, "layer_idx", -1),
                    tag=f"after_backward_{backward_count}",
                    topk=12,
                )
                _ctx_snapshot(
                    rank,
                    layer=getattr(ctx, "layer_idx", -1),
                    tag=f"after_backward_{backward_count}",
                    topk=12,
                )

        return grad_input, None, grad_weights, None, None, None, None, None, None, None, None


# =============================================================================
# KTMoE Layer Wrapper
# =============================================================================


class KTMoELayerWrapper(nn.Module):
    """Wrapper for MoE layer using KTMoEWrapper."""

    def __init__(
        self,
        original_moe: nn.Module,
        wrapper: Any,
        lora_params: dict[str, nn.Parameter] | None,  # Kept for backward compatibility, but ignored
        moe_config: MOEArchConfig,
        hidden_size: int,
        layer_idx: int,
        lora_experts: "LoRAExperts | None" = None,
    ):
        super().__init__()
        self._is_kt_moe_wrapper = True

        self.wrapper = wrapper
        self.moe_config = moe_config
        self.hidden_size = hidden_size
        self.layer_idx = layer_idx
        self.router_type = moe_config.router_type

        # IMPORTANT: Register submodules in the SAME ORDER as original MoE module
        # so that PEFT's named_modules() traversal order matches baseline.
        # This ensures kaiming_uniform_ calls happen in the same sequence.
        # Qwen3MoeSparseMoeBlock order: gate FIRST, then experts.

        # 1. gate/router FIRST - keep original attribute name for PEFT compatibility
        router_attr = moe_config.router_attr  # "gate" for Qwen3/DeepSeek
        setattr(self, router_attr, getattr(original_moe, router_attr, None))
        self._router_attr = router_attr

        # 2. experts SECOND (this is what PEFT targets for LoRA)
        experts_attr = moe_config.experts_attr  # typically "experts"
        setattr(self, experts_attr, getattr(original_moe, experts_attr, None))
        self._experts_attr = experts_attr

        # 3. shared_experts (if any)
        if moe_config.has_shared_experts and hasattr(original_moe, "shared_experts"):
            self.shared_experts = original_moe.shared_experts
        else:
            self.shared_experts = None

        # 4. lora_experts (separate LoRA expert MLPs, different from PEFT LoRA on experts)
        self.lora_experts = lora_experts

        # PEFT LoRA tracking (set by kt_adapt_peft_lora)
        # _peft_lora_modules: {expert_idx: {proj_name: (lora_A, lora_B)}}
        self._peft_lora_modules: dict[int, dict[str, tuple[nn.Module, nn.Module]]] | None = None
        self._peft_lora_rank: int = 0
        self._peft_lora_alpha: float = 0.0
        self._skip_lora: bool = False  # True when using SkipLoRA backend (no LoRA on experts)

        self._lora_pointers_dirty = False

    def _apply(self, fn, recurse=True):
        # Protect experts from device transfer (PEFT LoRA should stay on CPU for KT)
        saved_experts = None
        experts_attr = getattr(self, '_experts_attr', None)

        if experts_attr is not None and getattr(self, experts_attr, None) is not None:
            saved_experts = getattr(self, experts_attr)
            self._modules.pop(experts_attr, None)

        result = super()._apply(fn, recurse)

        if saved_experts is not None:
            self._modules[experts_attr] = saved_experts

        return result

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        
        import torch.distributed as dist
        dist_on = dist.is_initialized() and dist.get_world_size() > 1
        rank = dist.get_rank() if dist.is_initialized() else 0

        if KT_DEBUG:
            logger.warning(
                "[KT DEBUG] rank %s KTMoELayerWrapper.forward start layer=%s hidden=%s %s",
                rank,
                self.layer_idx,
                tuple(hidden_states.shape),
                hidden_states.dtype,
            )

        # Check if we need to use distributed broadcast (only rank 0 has KT kernel)
        use_broadcast = dist_on and self.wrapper is None
        if KT_DEBUG:
            logger.warning(
                "[KT DEBUG] rank %s KTMoELayerWrapper.forward layer=%s use_broadcast=%s",
                rank,
                self.layer_idx,
                use_broadcast,
            )

        if KT_DEBUG and dist_on:
            # Log shapes across ranks (batch dim may differ in DP mode, that's expected)
            shape_t = torch.tensor(tuple(hidden_states.shape), device=hidden_states.device, dtype=torch.int32)
            gathered = [torch.empty_like(shape_t) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered, shape_t)
            shapes = [tuple(t.tolist()) for t in gathered]
            if rank == 0:
                logger.warning(
                    "\033[35m[KT DEBUG] layer=%s hidden_states shapes across ranks=%s\033[0m",
                    self.layer_idx,
                    shapes,
                )

        if KT_DEBUG:
            logger.warning(
                "[KT DEBUG] rank %s KTMoELayerWrapper.forward routing start layer=%s",
                rank,
                self.layer_idx,
            )
        topk_ids, topk_weights = self._compute_routing(hidden_states)
        if KT_DEBUG:
            logger.warning(
                "[KT DEBUG] rank %s KTMoELayerWrapper.forward routing done layer=%s topk_ids=%s %s "
                "topk_weights=%s %s",
                rank,
                self.layer_idx,
                tuple(topk_ids.shape),
                topk_ids.dtype,
                tuple(topk_weights.shape),
                topk_weights.dtype,
            )

        train_lora = self._peft_lora_modules is not None and len(self._peft_lora_modules) > 0
        _track_lifecycle_tensor(
            rank,
            self.layer_idx,
            "forward.hidden_states",
            hidden_states,
        )
        _track_lifecycle_tensor(
            rank,
            self.layer_idx,
            "forward.topk_ids",
            topk_ids,
        )
        _track_lifecycle_tensor(
            rank,
            self.layer_idx,
            "forward.topk_weights",
            topk_weights,
        )
        save_for_backward = (
            self.training
            and torch.is_grad_enabled()
            and (hidden_states.requires_grad or topk_weights.requires_grad or train_lora)
        )
        ckpt_hook_mode = _checkpoint_hook_mode()
        in_ckpt_recompute = ckpt_hook_mode == "recompute"
        in_ckpt_first_forward = ckpt_hook_mode == "first_forward"
        if ckpt_hook_mode in ("none", "other", "error"):
            # Fallback for environments where hook-top probing is unavailable.
            in_ckpt_first_forward = _is_in_checkpoint_first_forward()
        if in_ckpt_recompute:
            # Recompute must be treated as non-first-forward in diagnostics.
            in_ckpt_first_forward = False
        # Keep KT autograd path whenever backward is needed. Disabling it in
        # checkpoint first-forward prevents KTMoEFunction.backward from running.
        use_autograd_path = save_for_backward
        save_for_backward_submit = use_autograd_path
        # Only suppress cache when we have high-confidence first_forward detection
        # via the saved_tensors_hooks stack. The stack-walk fallback is too fragile
        # for a correctness-critical decision — it only logs.
        if ckpt_hook_mode == "first_forward":
            save_for_backward_submit = False
        if KT_MEM_LOG:
            graph_task_id = -1
            try:
                graph_task_id = int(torch._C._current_graph_task_id())
            except Exception:
                graph_task_id = -1
            _kt_mem_log(
                rank,
                "forward_state",
                layer=self.layer_idx,
                training=self.training,
                grad_enabled=torch.is_grad_enabled(),
                hidden_states_requires_grad=hidden_states.requires_grad,
                topk_weights_requires_grad=topk_weights.requires_grad,
                train_lora=train_lora,
                save_for_backward=save_for_backward,
                save_for_backward_submit=save_for_backward_submit,
                use_autograd_path=use_autograd_path,
                in_ckpt_first_forward=in_ckpt_first_forward,
                in_ckpt_recompute=in_ckpt_recompute,
                ckpt_hook_mode=ckpt_hook_mode,
                graph_task_id=graph_task_id,
            )
            _kt_mem_log_wrapper(
                rank,
                "forward_wrapper_state",
                self.wrapper,
                device=hidden_states.device if hidden_states.device.type == "cuda" else None,
                layer=self.layer_idx,
                save_for_backward=save_for_backward_submit,
                qlen=hidden_states.shape[0] * hidden_states.shape[1],
            )
        if KT_DEBUG:
            logger.warning(
                "[KT DEBUG] rank %s KTMoELayerWrapper.forward layer=%s train_lora=%s "
                "save_for_backward=%s",
                rank,
                self.layer_idx,
                train_lora,
                save_for_backward,
            )

        if train_lora and self._lora_pointers_dirty:
            if KT_DEBUG:
                logger.warning(
                    "[KT DEBUG] rank %s KTMoELayerWrapper.forward update_lora_pointers layer=%s",
                    rank,
                    self.layer_idx,
                )
            self.update_lora_pointers()
            self._lora_pointers_dirty = False
            
            


        if KT_DEBUG:
            logger.warning(
                "[KT DEBUG] rank %s KTMoELayerWrapper.forward submit+gpu start layer=%s",
                rank,
                self.layer_idx,
            )

        gpu_output, all_qlens = self._submit_and_compute_gpu(
            hidden_states,
            topk_ids,
            topk_weights,
            save_for_backward_submit,
        )
        _track_lifecycle_tensor(
            rank,
            self.layer_idx,
            "forward.gpu_output",
            gpu_output,
        )
        if KT_DEBUG:
            logger.warning(
                "[KT DEBUG] rank %s KTMoELayerWrapper.forward submit+gpu done layer=%s all_qlens=%s",
                rank,
                self.layer_idx,
                all_qlens,
            )

        # Use KTMoEFunction whenever backward is needed so KT backward and LoRA
        # gradient paths remain connected.
        if use_autograd_path:
            lora_ref = hidden_states.new_empty(())
            if train_lora and self._peft_lora_modules:
                for expert_loras in self._peft_lora_modules.values():
                    for lora_A, lora_B in expert_loras.values():
                        if hasattr(lora_A, 'weight') and lora_A.weight.requires_grad:
                            lora_ref = lora_A.weight
                            break
                    if lora_ref.numel() > 0:
                        break

            moe_output = KTMoEFunction.apply(
                hidden_states,
                topk_ids,
                topk_weights,
                self.wrapper,
                lora_ref,
                self.hidden_size,
                self.moe_config.num_experts_per_tok,
                self.layer_idx,
                save_for_backward,
                train_lora,
                all_qlens,
            )
        else:
            moe_output = self._sync_forward_output_no_autograd(
                hidden_states=hidden_states,
                all_qlens=all_qlens,
            )
        _track_lifecycle_tensor(
            rank,
            self.layer_idx,
            "forward.moe_output",
            moe_output,
        )
        

        if gpu_output is not None:
            if KT_DEBUG:
                logger.warning(
                    "[KT DEBUG] rank %s KTMoELayerWrapper.forward add gpu_output layer=%s",
                    rank,
                    self.layer_idx,
                )
            moe_output = moe_output + gpu_output
        if KT_MEM_LOG:
            grad_fn_name = "None"
            try:
                gf = getattr(moe_output, "grad_fn", None)
                if gf is not None:
                    grad_fn_name = type(gf).__name__
            except Exception:
                grad_fn_name = "error"
            _kt_mem_log(
                rank,
                "forward_output_grad_state",
                layer=self.layer_idx,
                use_autograd_path=use_autograd_path,
                in_ckpt_first_forward=in_ckpt_first_forward,
                in_ckpt_recompute=in_ckpt_recompute,
                ckpt_hook_mode=ckpt_hook_mode,
                moe_output_requires_grad=bool(getattr(moe_output, "requires_grad", False)),
                moe_output_grad_fn=grad_fn_name,
                hidden_states_requires_grad=bool(getattr(hidden_states, "requires_grad", False)),
                topk_weights_requires_grad=bool(getattr(topk_weights, "requires_grad", False)),
                train_lora=bool(train_lora),
            )

        if KT_DEBUG:
            logger.warning(
                "[KT DEBUG] rank %s KTMoELayerWrapper.forward done layer=%s output=%s %s",
                rank,
                self.layer_idx,
                tuple(moe_output.shape),
                moe_output.dtype,
            )
            
        return moe_output

    def _sync_forward_output_no_autograd(
        self,
        hidden_states: torch.Tensor,
        all_qlens: list[int] | tuple[int, ...] | None,
    ) -> torch.Tensor:
        """Sync CPU expert output without creating KTMoEFunction autograd nodes."""
        import torch.distributed as dist

        original_device = hidden_states.device
        original_dtype = hidden_states.dtype
        batch_size, seq_len, _ = hidden_states.shape
        qlen = batch_size * seq_len

        dist_on = dist.is_initialized() and dist.get_world_size() > 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist_on else 1

        if dist_on:
            if all_qlens is None:
                all_qlens_list = _all_gather_qlens(qlen, original_device, world_size)
            else:
                all_qlens_list = [int(q) for q in all_qlens]
                if len(all_qlens_list) != world_size:
                    raise RuntimeError(
                        f"all_qlens length mismatch: got {len(all_qlens_list)}, expected {world_size}"
                    )
            if int(all_qlens_list[rank]) != qlen:
                raise RuntimeError(
                    f"Rank {rank} qlen mismatch: local={qlen}, all_qlens[{rank}]={all_qlens_list[rank]}"
                )
            total_qlen = sum(all_qlens_list)

            if rank == 0:
                if self.wrapper is None:
                    raise RuntimeError("Rank0 wrapper is required in distributed KT overlap path.")
                cpu_output = self.wrapper.sync_forward_sft(output_device=original_device)
                cpu_output = cpu_output.to(dtype=original_dtype).view(total_qlen, self.hidden_size)
                _track_lifecycle_tensor(
                    rank,
                    self.layer_idx,
                    "no_autograd.rank0.cpu_output",
                    cpu_output,
                )
                offsets = _qlen_offsets(all_qlens_list)
                scatter_list = [cpu_output[offsets[i] : offsets[i + 1]].contiguous() for i in range(world_size)]
                _track_lifecycle_collection(
                    rank,
                    self.layer_idx,
                    "no_autograd.rank0.scatter_list",
                    scatter_list,
                )
            else:
                scatter_list = None

            output_flat = _dist_scatter_varlen_from_rank0(
                rank0_chunks=scatter_list,
                all_qlens=all_qlens_list,
                rank=rank,
                world_size=world_size,
                feature_shape=(self.hidden_size,),
                device=original_device,
                dtype=original_dtype,
            )
            _track_lifecycle_tensor(
                rank,
                self.layer_idx,
                "no_autograd.output_flat",
                output_flat,
            )
            output = output_flat.view(batch_size, seq_len, self.hidden_size)
            _track_lifecycle_tensor(
                rank,
                self.layer_idx,
                "no_autograd.output",
                output,
            )
            del output_flat
            return output

        if self.wrapper is not None:
            cpu_output = self.wrapper.sync_forward_sft(output_device=original_device)
            _track_lifecycle_tensor(
                rank,
                self.layer_idx,
                "no_autograd.single.cpu_output",
                cpu_output,
            )
            output = cpu_output.view(batch_size, seq_len, self.hidden_size).to(dtype=original_dtype)
            _track_lifecycle_tensor(
                rank,
                self.layer_idx,
                "no_autograd.output",
                output,
            )
            return output

        return torch.empty(batch_size, seq_len, self.hidden_size, device=original_device, dtype=original_dtype)

    def _compute_routing(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Run routing under no_grad to avoid creating autograd nodes whose
        # SavedVariables become orphan holders inside gradient checkpoint.
        # The gate is frozen during LoRA fine-tuning and the main gradient
        # flows through KTMoEFunction.backward()'s grad_input, so the
        # routing gradient contribution to hidden_states can be safely dropped.
        with torch.no_grad():
            router = getattr(self, self._router_attr)
            if self.router_type == "deepseek_gate":
                # DeepSeek V3's MoEGate has `assert not self.training` in its noaux_tc
                # routing path because the HF model is an inference-only port.
                # For LoRA fine-tuning the router is frozen, so eval() is safe.
                was_training = router.training
                if was_training:
                    router.eval()
                router_output = router(hidden_states)
                if was_training:
                    router.train()
                if len(router_output) == 2:
                    topk_ids, topk_weights = router_output
                else:
                    topk_ids, topk_weights = router_output[0], router_output[1]
                if topk_weights.is_floating_point():
                    topk_weights = topk_weights.to(torch.bfloat16)
                return topk_ids, topk_weights

            router_logits = router(hidden_states.view(-1, self.hidden_size))
            routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float32)
            topk_weights, topk_ids = torch.topk(routing_weights, self.moe_config.num_experts_per_tok, dim=-1)
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
            topk_weights = topk_weights.to(torch.bfloat16)
            return topk_ids, topk_weights

    def _submit_and_compute_gpu(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        save_for_backward: bool,
    ) -> tuple[torch.Tensor | None, list[int] | None]:
        import torch.distributed as dist
        

        batch_size, seq_len, _ = hidden_states.shape
        original_device = hidden_states.device
        original_dtype = hidden_states.dtype

        dist_on = dist.is_initialized() and dist.get_world_size() > 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist_on else 1

        qlen = batch_size * seq_len
        if KT_MEM_LOG and hidden_states.is_cuda:
            dev = hidden_states.device
            _kt_mem_log(
                rank,
                "overlap_enter",
                layer=self.layer_idx,
                qlen=qlen,
                hidden_size=self.hidden_size,
                mem_allocated_bytes=torch.cuda.memory_allocated(dev),
                mem_reserved_bytes=torch.cuda.memory_reserved(dev),
                mem_max_allocated_bytes=torch.cuda.max_memory_allocated(dev),
            )

        if dist_on:
            all_qlens = _all_gather_qlens(qlen, original_device, world_size)
            if int(all_qlens[rank]) != qlen:
                raise RuntimeError(
                    f"Rank {rank} qlen mismatch: local={qlen}, all_qlens[{rank}]={all_qlens[rank]}"
                )
            total_qlen = sum(all_qlens)

            hs_flat = hidden_states.view(qlen, self.hidden_size).contiguous()
            expert_ids = topk_ids.view(qlen, self.moe_config.num_experts_per_tok).contiguous()
            weights = topk_weights.view(qlen, self.moe_config.num_experts_per_tok).contiguous()
            _track_lifecycle_tensor(
                rank,
                self.layer_idx,
                "overlap.hs_flat",
                hs_flat,
            )
            _track_lifecycle_tensor(
                rank,
                self.layer_idx,
                "overlap.expert_ids",
                expert_ids,
            )
            _track_lifecycle_tensor(
                rank,
                self.layer_idx,
                "overlap.weights",
                weights,
            )

            submit_hs = hs_flat.detach()
            submit_ids = expert_ids.detach()
            submit_wts = weights.detach()

            gathered_hs = _dist_gather_varlen_to_rank0(
                submit_hs,
                all_qlens=all_qlens,
                rank=rank,
                world_size=world_size,
            )
            gathered_ids = _dist_gather_varlen_to_rank0(
                submit_ids,
                all_qlens=all_qlens,
                rank=rank,
                world_size=world_size,
            )
            gathered_wts = _dist_gather_varlen_to_rank0(
                submit_wts,
                all_qlens=all_qlens,
                rank=rank,
                world_size=world_size,
            )

            if rank == 0:
                _track_lifecycle_collection(
                    rank,
                    self.layer_idx,
                    "overlap.rank0.gathered_hs",
                    gathered_hs,
                )
                _track_lifecycle_collection(
                    rank,
                    self.layer_idx,
                    "overlap.rank0.gathered_ids",
                    gathered_ids,
                )
                _track_lifecycle_collection(
                    rank,
                    self.layer_idx,
                    "overlap.rank0.gathered_wts",
                    gathered_wts,
                )
                all_hs = torch.cat(gathered_hs, dim=0)
                all_ids = torch.cat(gathered_ids, dim=0)
                all_wts = torch.cat(gathered_wts, dim=0)
                _track_lifecycle_tensor(
                    rank,
                    self.layer_idx,
                    "overlap.rank0.all_hs",
                    all_hs,
                )
                _track_lifecycle_tensor(
                    rank,
                    self.layer_idx,
                    "overlap.rank0.all_ids",
                    all_ids,
                )
                _track_lifecycle_tensor(
                    rank,
                    self.layer_idx,
                    "overlap.rank0.all_wts",
                    all_wts,
                )
                if KT_MEM_LOG:
                    _kt_mem_log(
                        rank,
                        "overlap_rank0_gathered",
                        layer=self.layer_idx,
                        total_qlen=total_qlen,
                        gathered_hs_bytes=_tensor_nbytes(gathered_hs),
                        gathered_ids_bytes=_tensor_nbytes(gathered_ids),
                        gathered_wts_bytes=_tensor_nbytes(gathered_wts),
                        all_hs_bytes=_tensor_nbytes(all_hs),
                        all_ids_bytes=_tensor_nbytes(all_ids),
                        all_wts_bytes=_tensor_nbytes(all_wts),
                    )
                    _kt_mem_log_wrapper(
                        rank,
                        "overlap_rank0_before_submit",
                        self.wrapper,
                        device=original_device if original_device.type == "cuda" else None,
                        layer=self.layer_idx,
                        qlen=total_qlen,
                        save_for_backward=save_for_backward,
                    )
                self.wrapper.submit_forward_sft(
                    all_hs,
                    all_ids,
                    all_wts,
                    save_for_backward=save_for_backward,
                )
                if KT_MEM_LOG:
                    _kt_mem_log_wrapper(
                        rank,
                        "overlap_rank0_after_submit",
                        self.wrapper,
                        device=original_device if original_device.type == "cuda" else None,
                        layer=self.layer_idx,
                        save_for_backward=save_for_backward,
                    )

            # Keep shared/lora experts local to avoid qlen_max-style amplification.
            gpu_output = None
            if self.shared_experts is not None:
                gpu_output = self.shared_experts(hidden_states)
                _track_lifecycle_tensor(
                    rank,
                    self.layer_idx,
                    "overlap.shared_gpu_output",
                    gpu_output,
                )
                gpu_output = gpu_output.to(dtype=original_dtype)

            if self.lora_experts is not None:
                lora_out = self.lora_experts(hidden_states)
                _track_lifecycle_tensor(
                    rank,
                    self.layer_idx,
                    "overlap.lora_out",
                    lora_out,
                )
                gpu_output = lora_out if gpu_output is None else gpu_output + lora_out

            return gpu_output, all_qlens

        else:
            # ---- Single-GPU path: submit + GPU compute ----
            input_flat = hidden_states.view(qlen, self.hidden_size)
            expert_ids = topk_ids.view(qlen, self.moe_config.num_experts_per_tok)
            weights = topk_weights.view(qlen, self.moe_config.num_experts_per_tok)
            _track_lifecycle_tensor(
                rank,
                self.layer_idx,
                "single.input_flat",
                input_flat,
            )
            _track_lifecycle_tensor(
                rank,
                self.layer_idx,
                "single.expert_ids",
                expert_ids,
            )
            _track_lifecycle_tensor(
                rank,
                self.layer_idx,
                "single.weights",
                weights,
            )

            if KT_MEM_LOG:
                _kt_mem_log_wrapper(
                    rank,
                    "submit_single_before",
                    self.wrapper,
                    device=original_device if original_device.type == "cuda" else None,
                    layer=self.layer_idx,
                    qlen=qlen,
                    save_for_backward=save_for_backward,
                )
            # Avoid passing graph-attached tensors into C++ cache.
            submit_hs = input_flat.detach()
            submit_ids = expert_ids.detach()
            submit_wts = weights.detach()
            self.wrapper.submit_forward_sft(
                submit_hs,
                submit_ids,
                submit_wts,
                save_for_backward=save_for_backward,
            )
            if KT_MEM_LOG:
                _kt_mem_log_wrapper(
                    rank,
                    "submit_single_after",
                    self.wrapper,
                    device=original_device if original_device.type == "cuda" else None,
                    layer=self.layer_idx,
                    save_for_backward=save_for_backward,
                )

            # GPU compute: shared_experts + lora_experts
            gpu_output = None
            if self.shared_experts is not None:
                gpu_output = self.shared_experts(hidden_states)
                _track_lifecycle_tensor(
                    rank,
                    self.layer_idx,
                    "single.shared_gpu_output",
                    gpu_output,
                )
            if self.lora_experts is not None:
                lora_out = self.lora_experts(hidden_states)
                _track_lifecycle_tensor(
                    rank,
                    self.layer_idx,
                    "single.lora_out",
                    lora_out,
                )
                gpu_output = lora_out if gpu_output is None else gpu_output + lora_out

            return gpu_output, None

    def update_lora_pointers(self):
        """Sync PEFT LoRA weights to C++ kernel after optimizer update."""
        # Skip if wrapper is None (non-rank-0 processes)
        if self.wrapper is None:
            return
        # Skip if wrapper is not properly initialized
        if not getattr(self.wrapper, "_weights_loaded", False):
            logger.warning(f"Layer {self.layer_idx}: Skipping update_lora_pointers - weights not loaded")
            return
        if not getattr(self.wrapper, "_lora_initialized", False):
            logger.warning(f"Layer {self.layer_idx}: Skipping update_lora_pointers - LoRA not initialized")
            return

        # PEFT weights are views into wrapper's contiguous buffers —
        # optimizer.step() already updated them in-place, just re-sync to C++.
        self.wrapper.update_lora_weights()

        if KT_DEBUG:
            import torch.distributed as dist
            rank = dist.get_rank() if dist.is_initialized() else 0
            logger.warning(
                "\033[35m[KT DEBUG] rank %s update_lora_pointers layer=%s update_lora_weights done\033[0m",
                rank, self.layer_idx,
            )


# =============================================================================
# Main Functions
# =============================================================================


def wrap_moe_layers_with_kt_wrapper(model: nn.Module, kt_plugin: Any) -> list[KTMoELayerWrapper]:
    """
    Replace model's MoE layers with KTMoEWrapper-based wrappers.

    Loads expert weights into the C++ KT kernel. No LoRA initialization —
    LoRA is handled by PEFT and later adapted via kt_adapt_peft_lora().
    Only rank 0 initializes KT kernel and loads weights.
    """
    import torch.distributed as dist

    if not KT_KERNEL_AVAILABLE:
        raise KTAMXNotAvailableError("kt_kernel not found. Please install kt_kernel to enable KT MoE support.")

    # Only rank 0 should initialize KT and load weights
    is_rank_0 = True
    if dist.is_initialized():
        is_rank_0 = dist.get_rank() == 0
        if not is_rank_0:
            logger.info(f"Rank {dist.get_rank()}: Skipping KT initialization (only rank 0 initializes KT)")

    moe_config = get_moe_arch_config(model.config)
    hidden_size = model.config.hidden_size

    # Read lora_rank/lora_alpha for C++ wrapper initialization (buffer allocation only)
    lora_rank = getattr(kt_plugin, "lora_rank", 1) or 1
    lora_alpha = getattr(kt_plugin, "lora_alpha", 1.0) or 1.0

    # Read LoRA Experts configuration
    _raw_le = getattr(kt_plugin, "kt_use_lora_experts", None)
    use_lora_experts = bool(_raw_le) if _raw_le is not None else False
    lora_expert_num = getattr(kt_plugin, "kt_lora_expert_num", 2) or 2
    lora_expert_intermediate_size = getattr(kt_plugin, "kt_lora_expert_intermediate_size", 1024) or 1024

    if is_rank_0:
        print(
            f"[kt_moe LoRA Experts] kt_plugin type={type(kt_plugin).__name__}, "
            f"raw kt_use_lora_experts={_raw_le!r} (type={type(_raw_le).__name__}), "
            f"use_lora_experts={use_lora_experts}, "
            f"num={lora_expert_num}, intermediate_size={lora_expert_intermediate_size}",
            flush=True,
        )

    wrappers: list[KTMoELayerWrapper] = []
    moe_layer_count = 0

    kt_backend_map = {
        "AMXBF16": "AMXBF16_SFT",
        "AMXINT8": "AMXINT8_SFT",
        "AMXINT4": "AMXINT4_SFT",
        "AMXBF16_SkipLoRA": "AMXBF16_SFT_SkipLoRA",
        "AMXINT8_SkipLoRA": "AMXINT8_SFT_SkipLoRA",
        "AMXINT4_SkipLoRA": "AMXINT4_SFT_SkipLoRA",
    }
    # Build case-insensitive lookup to handle common typos like "SkipLora" vs "SkipLoRA"
    _kt_backend_map_lower = {k.lower(): v for k, v in kt_backend_map.items()}
    kt_backend = getattr(kt_plugin, "kt_backend", "AMXBF16")
    kt_method = kt_backend_map.get(kt_backend) or _kt_backend_map_lower.get(kt_backend.lower(), "AMXBF16_SFT")
    if kt_method != kt_backend_map.get(kt_backend):
        logger.warning(
            f"kt_backend '{kt_backend}' matched via case-insensitive lookup → '{kt_method}'. "
            f"Please use the exact name from: {list(kt_backend_map.keys())}"
        )

    if "SkipLoRA" in kt_method:
        logger.info(f"Using SkipLoRA backend: {kt_method} (MoE LoRA gradients will be skipped)")

    threadpool_count = getattr(kt_plugin, "kt_threadpool_count", 1) if getattr(kt_plugin, "kt_tp_enabled", False) else 1

    kt_weight_path = getattr(kt_plugin, "kt_weight_path", None)
    use_kt_weight_path = kt_weight_path is not None
    if use_kt_weight_path:
        logger.info(f"Loading INT8 weights from kt_weight_path: {kt_weight_path}")

    checkpoint_files = getattr(kt_plugin, "kt_checkpoint_files", None)
    sharded_metadata = getattr(kt_plugin, "kt_sharded_metadata", None)

    # When kt_expert_checkpoint_path is set, always resolve from it (overrides any existing
    # checkpoint_files which may come from AttnOnlyBf16 and lack expert weights).
    kt_expert_checkpoint_path = getattr(kt_plugin, "kt_expert_checkpoint_path", None)
    if kt_expert_checkpoint_path:
        print(
            f"[kt_moe] Resolving expert checkpoint files from kt_expert_checkpoint_path={kt_expert_checkpoint_path!r}",
            flush=True,
        )
        resolved_files, resolved_meta = _resolve_checkpoint_files(model_name_or_path=kt_expert_checkpoint_path)
        if resolved_files and all(f.endswith(".safetensors") for f in resolved_files):
            checkpoint_files = resolved_files
            sharded_metadata = resolved_meta
            kt_plugin.kt_checkpoint_files = checkpoint_files
            kt_plugin.kt_sharded_metadata = sharded_metadata
            print(
                f"[kt_moe] Resolved {len(checkpoint_files)} checkpoint files from kt_expert_checkpoint_path",
                flush=True,
            )
        else:
            logger.warning(f"Failed to resolve checkpoint files from kt_expert_checkpoint_path={kt_expert_checkpoint_path!r}")

    use_checkpoint_files = bool(checkpoint_files) and not use_kt_weight_path

    print(
        f"[kt_moe] kt_weight_path={kt_weight_path!r} "
        f"(is_dir={os.path.isdir(kt_weight_path) if kt_weight_path else 'N/A'})",
        flush=True,
    )
    print(
        f"[kt_moe] kt_expert_checkpoint_path={kt_expert_checkpoint_path!r}",
        flush=True,
    )
    print(
        f"[kt_moe] checkpoint_files count={len(checkpoint_files) if checkpoint_files else 0}",
        flush=True,
    )
    print(
        f"[kt_moe] use_kt_weight_path={use_kt_weight_path}, use_checkpoint_files={use_checkpoint_files}",
        flush=True,
    )

    if use_checkpoint_files:
        logger.info("Loading expert weights from checkpoint files (online conversion).")
    elif use_kt_weight_path and bool(checkpoint_files):
        logger.info("BF16 checkpoint files available for backward gradient computation.")
    elif (not use_kt_weight_path) and bool(getattr(kt_plugin, "kt_skip_expert_loading", False)):
        # If HF expert weights were skipped during `from_pretrained`, we must source expert weights externally.
        model_name_or_path = getattr(getattr(model, "config", None), "name_or_path", None)
        if model_name_or_path:
            resolved_files, resolved_meta = _resolve_checkpoint_files(model_name_or_path=model_name_or_path)
            if resolved_files and all(f.endswith(".safetensors") for f in resolved_files):
                checkpoint_files = resolved_files
                sharded_metadata = resolved_meta
                kt_plugin.kt_checkpoint_files = checkpoint_files
                kt_plugin.kt_sharded_metadata = sharded_metadata
                use_checkpoint_files = True
                logger.info("KT skip_expert_loading enabled; using checkpoint files for online expert loading.")

        if not use_checkpoint_files:
            raise KTAMXConfigError(
                "KT skip_expert_loading is enabled but no `kt_weight_path` was provided and no safetensors checkpoint "
                "files could be resolved for on-the-fly expert loading."
            )

    import torch.distributed as _dist
    _rank = _dist.get_rank() if _dist.is_initialized() else 0

    model_container, layers = _get_model_container_and_layers(model, purpose="wrapping")
    print(f"[kt_moe rank={_rank}] Total layers={len(layers)}, is_rank_0={is_rank_0}", flush=True)

    for layer_idx, layer in enumerate(layers):
        moe_module = get_moe_module(layer, moe_config)
        if moe_module is None:
            if layer_idx < 3 or layer_idx == len(layers) - 1:
                print(f"[kt_moe rank={_rank}] Layer {layer_idx}: no MoE module, skipping", flush=True)
            continue

        if is_rank_0:
            print(
                f"[kt_moe rank={_rank}] Wrapping MoE layer {layer_idx} "
                f"(method={kt_method}, checkpoint_files={'yes' if checkpoint_files else 'no'})",
                flush=True,
            )

        # Only rank 0 loads weights and initializes KT kernel
        gate_proj, up_proj, down_proj = None, None, None
        wrapper = None

        if is_rank_0:
            # Get block_size from quantization_config if available (for FP8 dequant)
            _quant_cfg = getattr(model.config, "quantization_config", None)
            _block_size = None
            if _quant_cfg is not None:
                _block_size = getattr(_quant_cfg, "weight_block_size", None)

            if use_kt_weight_path:
                # kt_weight_path has pre-quantized forward + backward .kt files.
                # C++ loads them directly — no BF16 from checkpoint needed.
                print(
                    f"[kt_moe rank={_rank}] Layer {layer_idx}: "
                    f"forward + backward from kt_weight_path (.kt files)",
                    flush=True,
                )
            elif use_checkpoint_files:
                layers_prefix = _get_layers_prefix(model.config)
                gate_proj, up_proj, down_proj = load_experts_from_checkpoint_files(
                    checkpoint_files=checkpoint_files,
                    sharded_metadata=sharded_metadata,
                    layers_prefix=layers_prefix,
                    moe_config=moe_config,
                    layer_idx=layer_idx,
                    block_size=_block_size,
                )
            else:
                gate_proj, up_proj, down_proj = extract_moe_weights(moe_module, moe_config)
                gate_proj = gate_proj.cpu().to(torch.bfloat16).contiguous()
                up_proj = up_proj.cpu().to(torch.bfloat16).contiguous()
                down_proj = down_proj.cpu().to(torch.bfloat16).contiguous()

        chunked_prefill_size = getattr(kt_plugin, "model_max_length", None)
        if chunked_prefill_size is None:
            chunked_prefill_size = getattr(model.config, "max_position_embeddings", 4096)

        # Only rank 0 creates KTMoEWrapper and loads weights
        if is_rank_0:
            wrapper = KTMoEWrapper(
                layer_idx=layer_idx,
                num_experts=moe_config.expert_num,
                num_experts_per_tok=moe_config.num_experts_per_tok,
                hidden_size=hidden_size,
                moe_intermediate_size=moe_config.intermediate_size,
                num_gpu_experts=0,
                cpuinfer_threads=getattr(kt_plugin, "kt_num_threads", 1),
                threadpool_count=threadpool_count,
                weight_path=kt_weight_path or "",
                chunked_prefill_size=chunked_prefill_size,
                method=kt_method,
                mode="sft",
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                max_cache_depth=getattr(kt_plugin, "kt_max_cache_depth", 2),
            )

            physical_to_logical_map = torch.arange(moe_config.expert_num, dtype=torch.int64, device="cpu")

            if use_kt_weight_path:
                print(
                    f"[kt_moe] Layer {layer_idx}: calling wrapper.load_weights() "
                    f"(C++ direct .kt load, kt_weight_path={kt_weight_path!r})",
                    flush=True,
                )
                wrapper.load_weights(physical_to_logical_map)
            else:
                print(
                    f"[kt_moe] Layer {layer_idx}: calling wrapper.load_weights_from_tensors() "
                    f"(BF16 tensor path, gate_proj shape={gate_proj.shape if gate_proj is not None else None})",
                    flush=True,
                )
                wrapper.load_weights_from_tensors(
                    gate_proj=gate_proj,
                    up_proj=up_proj,
                    down_proj=down_proj,
                    physical_to_logical_map_cpu=physical_to_logical_map,
                )

            wrapper.gate_proj = None
            wrapper.up_proj = None
            wrapper.down_proj = None

        if is_rank_0:
            print(
                f"[kt_moe INIT] Layer {layer_idx}: C++ wrapper created with "
                f"lora_rank={lora_rank}, lora_alpha={lora_alpha}, "
                f"num_experts={moe_config.expert_num}, hidden_size={hidden_size}, "
                f"intermediate_size={moe_config.intermediate_size}",
                flush=True,
            )

        # Create LoRA Experts if enabled
        lora_experts = None
        if use_lora_experts:
            lora_experts = LoRAExperts(
                num_experts=lora_expert_num,
                hidden_size=hidden_size,
                intermediate_size=lora_expert_intermediate_size,
                device="cuda",
                dtype=torch.bfloat16,
            )

        layer_wrapper = KTMoELayerWrapper(
            original_moe=moe_module,
            wrapper=wrapper,
            lora_params=None,
            moe_config=moe_config,
            hidden_size=hidden_size,
            layer_idx=layer_idx,
            lora_experts=lora_experts,
        )
        layer_wrapper._skip_lora = "SkipLoRA" in kt_method

        setattr(layer, moe_config.moe_layer_attr, layer_wrapper)
        # Base weights have been copied into the C++ kernel's internal BufferB format.
        # Do not hold a Python-side reference — it wastes ~1 GB/layer.
        del gate_proj, up_proj, down_proj

        wrappers.append(layer_wrapper)
        moe_layer_count += 1

        # Replace original expert weights with meta placeholders.
        # Experts remain in the model tree (via wrapper.experts) so PEFT can discover them.
        # Rank 0 already copied weights to C++ kernel via load_weights_from_tensors.
        _clear_original_expert_weights(moe_module, moe_config)

    logger.info(f"Wrapped {moe_layer_count} MoE layers with KTMoEWrapper")
    gc.collect()
    return wrappers


def _build_kt_plugin_from_args(model_args: Any, finetuning_args: Any | None = None) -> KTransformersPlugin:
    return KTransformersPlugin(
        enabled=True,
        kt_backend=getattr(model_args, "kt_backend", None),
        kt_num_threads=getattr(model_args, "kt_num_threads", None),
        kt_tp_enabled=getattr(model_args, "kt_tp_enabled", None),
        kt_threadpool_count=getattr(model_args, "kt_threadpool_count", None),
        kt_max_cache_depth=getattr(model_args, "kt_max_cache_depth", None),
        kt_num_gpu_experts=getattr(model_args, "kt_num_gpu_experts", None),
        kt_weight_path=getattr(model_args, "kt_weight_path", None),
        kt_expert_checkpoint_path=getattr(model_args, "kt_expert_checkpoint_path", None),
        kt_use_lora_experts=getattr(model_args, "kt_use_lora_experts", None),
        kt_lora_expert_num=getattr(model_args, "kt_lora_expert_num", None),
        kt_lora_expert_intermediate_size=getattr(model_args, "kt_lora_expert_intermediate_size", None),
        lora_rank=getattr(finetuning_args, "lora_rank", None) if finetuning_args is not None else None,
        lora_alpha=getattr(finetuning_args, "lora_alpha", None) if finetuning_args is not None else None,
        model_max_length=getattr(model_args, "model_max_length", None),
    )


def load_kt_model(
    config,
    model_args: Any | None = None,
    finetuning_args: Any | None = None,
    kt_plugin: KTransformersPlugin | None = None,
    model_name_or_path: str | None = None,
    trust_remote_code: bool | None = None,
    token: str | None = None,
    torch_dtype: torch.dtype | str | None = torch.bfloat16,
    **kwargs,
) -> nn.Module:
    """
    Load model with KTMoEWrapper backend.

    Accepts either a KTPlugin directly or model_args/finetuning_args with compatible attributes.
    """
    if not KT_KERNEL_AVAILABLE:
        raise KTAMXNotAvailableError("kt_kernel not found. Please install kt_kernel to enable KT MoE support.")

    if kt_plugin is None:
        if model_args is None:
            raise KTAMXConfigError("Either kt_plugin or model_args must be provided to load_kt_model().")
        kt_plugin = _build_kt_plugin_from_args(model_args, finetuning_args)

    if model_name_or_path is None and model_args is not None:
        model_name_or_path = getattr(model_args, "model_name_or_path", None)
    if model_name_or_path is None:
        raise KTAMXConfigError("model_name_or_path is required to load_kt_model().")

    if trust_remote_code is None and model_args is not None:
        trust_remote_code = getattr(model_args, "trust_remote_code", None)
    if token is None and model_args is not None:
        token = getattr(model_args, "hf_hub_token", None)
    cache_dir = getattr(model_args, "cache_dir", None) if model_args is not None else None
    revision = getattr(model_args, "revision", None) if model_args is not None else None

    _ = get_moe_arch_config(config)

    logger.info("Loading model with KTMoEWrapper backend")
    logger.info(
        "KT config: "
        f"kt_num_gpu_experts={getattr(kt_plugin, 'kt_num_gpu_experts', None)}, "
        f"kt_weight_path={getattr(kt_plugin, 'kt_weight_path', None)}, "
        f"kt_backend={getattr(kt_plugin, 'kt_backend', None)}, "
        f"kt_tp_enabled={getattr(kt_plugin, 'kt_tp_enabled', None)}"
    )

    from transformers import AutoModelForCausalLM
    from transformers.integrations.kt import set_kt_config, unset_kt_config

    loading_kwargs = get_kt_loading_kwargs(
        config,
        kt_plugin,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
        token=token,
    )
    if model_args is not None:
        for key in ("cache_dir", "revision"):
            value = getattr(model_args, key, None)
            if value is not None:
                loading_kwargs[key] = value

    loading_kwargs.update(kwargs)

    if getattr(kt_plugin, "kt_skip_expert_loading", None) is None:
        checkpoint_files, sharded_metadata = _resolve_checkpoint_files(
            model_name_or_path=model_name_or_path,
            cache_dir=cache_dir,
            revision=revision,
            token=token,
            trust_remote_code=trust_remote_code,
        )
        if checkpoint_files and all(f.endswith(".safetensors") for f in checkpoint_files):
            if getattr(kt_plugin, "kt_weight_path", None) is None:
                kt_plugin.kt_skip_expert_loading = True
            else:
                # kt_weight_path provides INT8 for forward, but we still need
                # checkpoint files to load BF16 weights for backward pass.
                kt_plugin.kt_skip_expert_loading = False
            kt_plugin.kt_checkpoint_files = checkpoint_files
            kt_plugin.kt_sharded_metadata = sharded_metadata
        else:
            kt_plugin.kt_skip_expert_loading = False

    set_kt_config(kt_plugin)
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **loading_kwargs)
    finally:
        unset_kt_config()

    moe_config = get_moe_arch_config(config)
    move_non_experts_to_gpu(model, moe_config, device="cuda:0")

    expert_device = get_expert_device(model, moe_config)
    logger.info(f"MoE experts on device: {expert_device}")

    wrappers = wrap_moe_layers_with_kt_wrapper(model, kt_plugin)

    model._kt_wrappers = wrappers
    model._kt_tp_enabled = bool(getattr(kt_plugin, "kt_tp_enabled", False))
    model._kt_use_lora_experts = bool(getattr(kt_plugin, "kt_use_lora_experts", False))

    logger.info("Model loaded with KTMoEWrapper backend successfully")
    return model


def get_kt_lora_params(model: nn.Module) -> list[nn.Parameter]:
    """Get all MoE LoRA parameters from KT model.

    Returns PEFT LoRA parameters from expert modules and lora_experts parameters.
    """
    params: list[nn.Parameter] = []

    wrappers = getattr(model, "_kt_wrappers", None)
    if wrappers is None:
        base_model = model
        for attr in ("base_model", "model"):
            if hasattr(base_model, attr):
                base_model = getattr(base_model, attr)
                wrappers = getattr(base_model, "_kt_wrappers", None)
                if wrappers:
                    break

    if wrappers:
        for wrapper in wrappers:
            # PEFT LoRA parameters (from _peft_lora_modules)
            peft_lora_modules = getattr(wrapper, "_peft_lora_modules", None)
            if peft_lora_modules is not None:
                for expert_loras in peft_lora_modules.values():
                    for lora_A, lora_B in expert_loras.values():
                        if hasattr(lora_A, 'weight') and lora_A.weight.requires_grad:
                            params.append(lora_A.weight)
                        if hasattr(lora_B, 'weight') and lora_B.weight.requires_grad:
                            params.append(lora_B.weight)
            # lora_experts parameters (separate feature)
            if getattr(wrapper, "lora_experts", None) is not None:
                params.extend(wrapper.lora_experts.parameters())

    return params


def kt_adapt_peft_lora(model: nn.Module) -> None:
    """
    Adapt PEFT LoRA on expert modules for KT kernel.

    After PEFT injects LoRA adapters onto expert Linear modules, this function:
    1. Detects PEFT LoRA presence and rank on each wrapper's experts
    2. Stores references to PEFT LoRA modules on the wrapper (for backward gradient writing)
    3. Syncs initial PEFT LoRA weights to the C++ KT kernel (rank 0 only)

    PEFT LoRA remains active and is managed by PEFT. No separate KT lora_params created.
    Optimizer updates PEFT LoRA directly, and KT kernel reads from PEFT LoRA on each forward.

    Should be called after PEFT LoRA injection and before create_optimizer.
    """
    import torch.distributed as dist

    wrappers = getattr(model, "_kt_wrappers", None)
    if wrappers is None:
        # Try unwrapping PEFT/other wrappers
        base = model
        for attr in ("base_model", "model"):
            if hasattr(base, attr):
                base = getattr(base, attr)
                wrappers = getattr(base, "_kt_wrappers", None)
                if wrappers:
                    break

    if not wrappers:
        logger.info("[kt_adapt_peft_lora] No _kt_wrappers found, skipping")
        return

    is_rank_0 = True
    if dist.is_initialized():
        is_rank_0 = dist.get_rank() == 0

    # Detect PEFT LoRA rank and lora_alpha from first wrapper's experts
    # detected_lora_rank = 0
    # detected_lora_alpha = 0.0
    # first_wrapper = wrappers[0] if wrappers else None
    # if first_wrapper is not None:
    #     experts_attr = getattr(first_wrapper, "_experts_attr", "experts")
    #     experts = getattr(first_wrapper, experts_attr, None)

    #     if experts is not None and len(experts) > 0:
    #         moe_config = first_wrapper.moe_config
    #         gate_name = moe_config.weight_names[0]
    #         first_expert = experts[0]
    #         first_gate = getattr(first_expert, gate_name, None)


    adapted_count = 0
    for wrapper in wrappers:
        moe_config = wrapper.moe_config
        layer_idx = wrapper.layer_idx
        experts_attr = getattr(wrapper, "_experts_attr", "experts")
        experts = getattr(wrapper, experts_attr, None)

        if experts is None or len(experts) == 0:
            continue

        # Collect references to PEFT LoRA modules for each expert
        # Structure: {expert_idx: {proj_name: (lora_A_module, lora_B_module)}}
        peft_lora_modules = {}
        gate_name, up_name, down_name = moe_config.weight_names

        for expert_idx, expert in enumerate(experts):
            expert_loras = {}
            for proj_name in (gate_name, up_name, down_name):
                proj = getattr(expert, proj_name, None)
                if proj is None:
                    continue
                lora_A = getattr(proj, "lora_A", None)
                lora_B = getattr(proj, "lora_B", None)
                if lora_A is not None and lora_B is not None:
                    # Get the actual Linear modules (inside ModuleDict if using adapters)
                    if isinstance(lora_A, nn.ModuleDict):
                        adapter_name = "default"
                        active = getattr(proj, "active_adapter", ["default"])
                        if isinstance(active, (list, tuple)) and active:
                            adapter_name = active[0]
                        # ModuleDict doesn't have .get(), use [] with in check
                        lora_A = lora_A[adapter_name] if adapter_name in lora_A else None
                        lora_B = lora_B[adapter_name] if adapter_name in lora_B else None
                    if lora_A is not None and lora_B is not None:
                        expert_loras[proj_name] = (lora_A, lora_B)
            if expert_loras:
                peft_lora_modules[expert_idx] = expert_loras

        # Store PEFT LoRA references on wrapper
        wrapper._peft_lora_modules = peft_lora_modules

        # SkipLoRA mode: if no LoRA found on experts, skip buffer creation
        if not peft_lora_modules:
            if getattr(wrapper, '_skip_lora', False):
                logger.info(
                    f"[kt_adapt_peft_lora] Layer {layer_idx}: SkipLoRA mode, "
                    f"no PEFT LoRA on experts — skipping LoRA buffer creation"
                )
                adapted_count += 1
                continue
            else:
                raise RuntimeError(
                    f"[kt_adapt_peft_lora] Layer {layer_idx}: No PEFT LoRA found on any expert. "
                    f"If you intend to train without expert LoRA, use a SkipLoRA backend "
                    f"(e.g., kt_backend: AMXINT8_SkipLoRA)."
                )

        # Allocate contiguous bf16 buffers and populate with initial PEFT values (all ranks)
        lora_buffers = _create_lora_view_buffers(peft_lora_modules, moe_config, torch.bfloat16)
        lora_grad_buffers = _create_lora_grad_buffers(peft_lora_modules, moe_config)

        # Rank 0: pass buffers to C++ wrapper (init_lora_weights stores them via .contiguous() no-op)
        if is_rank_0 and wrapper.wrapper is not None:
            # concat lora_buffers and lora_grad_buffers into single dict
            lora_buffers.update(lora_grad_buffers)
            wrapper.wrapper.init_lora_weights(**lora_buffers)
            logger.info(f"[kt_adapt_peft_lora] Layer {layer_idx}: synced PEFT LoRA to C++ kernel")

        # All ranks: replace PEFT weights with views into the contiguous buffers
        _replace_peft_weights_with_views(peft_lora_modules, lora_buffers, lora_grad_buffers, moe_config)

        adapted_count += 1

    # After collecting all LoRA references, shrink expert base weight parameters
    # from their original shape (e.g. [768, 2048]) to scalar (1,).
    # These base weights were already replaced with tiny-storage stride=[0] placeholders
    # by _clear_original_expert_weights(). They have correct shape but serve no purpose
    # after PEFT injection. FSDP2 broadcasts ALL non-DTensor params, and uses
    # torch.empty(param.size()) on non-rank-0 — with the original shape this wastes
    # ~28GB+. Shrinking to (1,) reduces broadcast cost to ~30KB total.
    shrunk_count = 0
    shrunk_saved_bytes = 0
    for wrapper in wrappers:
        experts_attr = getattr(wrapper, "_experts_attr", "experts")
        experts = getattr(wrapper, experts_attr, None)
        if experts is None:
            continue
        for expert in experts:
            for param_name, param in list(expert.named_parameters()):
                if param.requires_grad:
                    continue  # Skip trainable params (LoRA weights)
                try:
                    storage_bytes = param.data.untyped_storage().nbytes()
                except Exception:
                    continue
                if storage_bytes > 2:
                    continue  # Skip non-placeholder params

                # This is a tiny-storage placeholder (base weight) — replace with
                # a scalar (1,) parameter so FSDP broadcasts only 1 element.
                original_numel = param.nelement()
                parts = param_name.split(".")
                container = expert
                for p in parts[:-1]:
                    container = getattr(container, p)
                local_name = parts[-1]
                container_params = getattr(container, "_parameters", {})
                if isinstance(container_params, dict) and local_name in container_params:
                    scalar_param = nn.Parameter(
                        torch.empty(1, dtype=param.dtype, device="cpu"),
                        requires_grad=False,
                    )
                    container_params[local_name] = scalar_param
                    shrunk_count += 1
                    shrunk_saved_bytes += (original_numel - 1) * param.element_size()

    if shrunk_count > 0:
        logger.info(
            f"[kt_adapt_peft_lora] Shrunk {shrunk_count} expert base weight params "
            f"to shape (1,), FSDP broadcast savings={shrunk_saved_bytes / 1024 / 1024:.1f} MB"
        )

    logger.info(f"[kt_adapt_peft_lora] Adapted {adapted_count} layers (PEFT LoRA mode)")


def _create_lora_view_buffers(
    peft_lora_modules: dict[int, dict[str, tuple[nn.Module, nn.Module]]],
    moe_config: MOEArchConfig,
    dtype: torch.dtype = torch.bfloat16,
) -> dict[str, torch.Tensor]:
    """
    Allocate contiguous buffers and populate with initial PEFT LoRA values.

    Returns dict with gate_lora_a, gate_lora_b, up_lora_a, up_lora_b,
    down_lora_a, down_lora_b — each shape [num_experts, ...].
    """
    gate_name, up_name, down_name = moe_config.weight_names
    num_experts = moe_config.expert_num

    first_expert_loras = peft_lora_modules.get(0, {})
    if not first_expert_loras:
        raise RuntimeError("No PEFT LoRA found on expert 0")
    gate_lora = first_expert_loras.get(gate_name)
    if gate_lora is None:
        raise RuntimeError(f"No PEFT LoRA found on expert 0 {gate_name}")

    lora_rank = gate_lora[0].weight.shape[0]
    hidden_size = gate_lora[0].weight.shape[1]
    intermediate_size = gate_lora[1].weight.shape[0]

    buffers = {
        "gate_lora_a": torch.zeros(num_experts, lora_rank, hidden_size, dtype=dtype, device="cpu"),
        "gate_lora_b": torch.zeros(num_experts, intermediate_size, lora_rank, dtype=dtype, device="cpu"),
        "up_lora_a": torch.zeros(num_experts, lora_rank, hidden_size, dtype=dtype, device="cpu"),
        "up_lora_b": torch.zeros(num_experts, intermediate_size, lora_rank, dtype=dtype, device="cpu"),
        "down_lora_a": torch.zeros(num_experts, lora_rank, intermediate_size, dtype=dtype, device="cpu"),
        "down_lora_b": torch.zeros(num_experts, hidden_size, lora_rank, dtype=dtype, device="cpu"),
    }

    proj_to_keys = {
        gate_name: ("gate_lora_a", "gate_lora_b"),
        up_name: ("up_lora_a", "up_lora_b"),
        down_name: ("down_lora_a", "down_lora_b"),
    }
    for expert_idx in range(num_experts):
        expert_loras = peft_lora_modules.get(expert_idx, {})
        for proj_name, (key_a, key_b) in proj_to_keys.items():
            if proj_name in expert_loras:
                lora_A, lora_B = expert_loras[proj_name]
                buffers[key_a][expert_idx].copy_(lora_A.weight.data.to(dtype=dtype))
                buffers[key_b][expert_idx].copy_(lora_B.weight.data.to(dtype=dtype))

    return buffers

def _create_lora_grad_buffers(peft_lora_modules: dict[int, dict[str, tuple[nn.Module, nn.Module]]],moe_config: MOEArchConfig,dtype: torch.dtype = torch.bfloat16):
    gate_name, up_name, down_name = moe_config.weight_names
    num_experts = moe_config.expert_num

    first_expert_loras = peft_lora_modules.get(0, {})
    if not first_expert_loras:
        raise RuntimeError("No PEFT LoRA found on expert 0")
    gate_lora = first_expert_loras.get(gate_name)
    if gate_lora is None:
        raise RuntimeError(f"No PEFT LoRA found on expert 0 {gate_name}")

    lora_rank = gate_lora[0].weight.shape[0]
    hidden_size = gate_lora[0].weight.shape[1]
    intermediate_size = gate_lora[1].weight.shape[0]
    
    buffers = {
        "grad_gate_lora_a": torch.zeros(num_experts, lora_rank, hidden_size, dtype=dtype, device="cpu"),
        "grad_gate_lora_b": torch.zeros(num_experts, intermediate_size, lora_rank, dtype=dtype, device="cpu"),
        "grad_up_lora_a": torch.zeros(num_experts, lora_rank, hidden_size, dtype=dtype, device="cpu"),
        "grad_up_lora_b": torch.zeros(num_experts, intermediate_size, lora_rank, dtype=dtype, device="cpu"),
        "grad_down_lora_a": torch.zeros(num_experts, lora_rank, intermediate_size, dtype=dtype, device="cpu"),
        "grad_down_lora_b": torch.zeros(num_experts, hidden_size, lora_rank, dtype=dtype, device="cpu"),
    }

    return buffers


def _replace_peft_weights_with_views(
    peft_lora_modules: dict[int, dict[str, tuple[nn.Module, nn.Module]]],
    buffers: dict[str, torch.Tensor],
    grad_buffers: dict[str, torch.Tensor],
    moe_config: MOEArchConfig,
) -> None:
    """
    Replace each PEFT LoRA module's .weight with a view into the contiguous buffer.

    After this, optimizer.step() updates the buffer in-place via the view —
    no copy needed to sync with C++.
    """
    gate_name, up_name, down_name = moe_config.weight_names
    num_experts = moe_config.expert_num

    proj_to_keys = {
        gate_name: ("gate_lora_a", "gate_lora_b"),
        up_name: ("up_lora_a", "up_lora_b"),
        down_name: ("down_lora_a", "down_lora_b"),
    }

    _replaced = 0
    _first_logged = False
    for expert_idx in range(num_experts):
        expert_loras = peft_lora_modules.get(expert_idx, {})
        for proj_name, (key_a, key_b) in proj_to_keys.items():
            if proj_name not in expert_loras:
                continue
            lora_A, lora_B = expert_loras[proj_name]

            # Log before/after for first replacement to verify .data assignment
            if not _first_logged:
                _old_id_a = id(lora_A.weight)
                _old_ptr_a = lora_A.weight.data_ptr()

            # Use .data assignment to keep the same Parameter objects.
            # This preserves optimizer references (which point to these objects).
            # Creating new nn.Parameter() would break the optimizer link.
            lora_A.weight.data = buffers[key_a][expert_idx]
            lora_B.weight.data = buffers[key_b][expert_idx]
            lora_A.weight.requires_grad_(True)
            lora_B.weight.requires_grad_(True)
            lora_A.weight.grad = grad_buffers["grad_"+key_a][expert_idx]
            lora_B.weight.grad = grad_buffers["grad_"+key_b][expert_idx]

            if not _first_logged:
                _new_id_a = id(lora_A.weight)
                _new_ptr_a = lora_A.weight.data_ptr()
                _buf_ptr_a = buffers[key_a][expert_idx].data_ptr()
                _has_grad = lora_A.weight.grad is not None
                logger.info(
                    "[_replace_peft_weights_with_views] first param: "
                    "id %s->%s (same=%s) data_ptr %s->%s buf_ptr=%s (match=%s) "
                    "has_grad=%s requires_grad=%s shape=%s",
                    _old_id_a, _new_id_a, _old_id_a == _new_id_a,
                    _old_ptr_a, _new_ptr_a, _buf_ptr_a, _new_ptr_a == _buf_ptr_a,
                    _has_grad, lora_A.weight.requires_grad, tuple(lora_A.weight.shape),
                )
                _first_logged = True
            _replaced += 1

    logger.info("[_replace_peft_weights_with_views] replaced %d param pairs", _replaced)

def update_kt_lora_pointers(model: nn.Module):
    """Mark KT wrapper LoRA pointers as dirty after optimizer.step()."""
    wrappers = getattr(model, "_kt_wrappers", None)
    if wrappers is None:
        base_model = model
        for attr in ("base_model", "model"):
            if hasattr(base_model, attr):
                base_model = getattr(base_model, attr)
                wrappers = getattr(base_model, "_kt_wrappers", None)
                if wrappers:
                    break

    if wrappers:
        for wrapper in wrappers:
            wrapper._lora_pointers_dirty = True


def sync_kt_lora_gradients(model: nn.Module) -> None:
    """
    Synchronize KT-managed LoRA gradients across ranks.

    KT computes expert LoRA gradients only on rank 0 (gather/scatter path). This function broadcasts the
    per-layer contiguous grad buffers from rank 0 to all ranks so that:
      - gradient clipping sees identical grads on every rank
      - optimizer.step() applies identical updates
    """
    import torch.distributed as dist

    if not (dist.is_initialized() and dist.get_world_size() > 1):
        return

    world_size = dist.get_world_size()
    if world_size <= 1:
        return

    params = get_kt_lora_params(model)
    if not params:
        return

    for param in params:
        if param.grad is not None:
            # Move grad to the same device as the parameter for all-reduce
            # Then move back to CPU
            original_device = param.grad.device
            if original_device.type == "cpu":
                # All-reduce on CPU might be slow; consider using a GPU buffer
                grad_gpu = param.grad.cuda()
                dist.all_reduce(grad_gpu, op=dist.ReduceOp.SUM)
                grad_gpu.div_(world_size)
                param.grad.copy_(grad_gpu.cpu())
            else:
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                param.grad.div_(world_size)
            # print(dist.get_rank(), "[SYNC KT LORA GRAD]", param)


def save_lora_experts_to_adapter(model: nn.Module, output_dir: str) -> None:
    """
    Save LoRA Experts weights to adapter file by merging with existing Attention LoRA.
    """
    import os
    from safetensors import safe_open
    from safetensors.torch import save_file

    wrappers = getattr(model, "_kt_wrappers", [])
    if not wrappers:
        base_model = model
        for attr in ["base_model", "model"]:
            if hasattr(base_model, attr):
                base_model = getattr(base_model, attr)
                wrappers = getattr(base_model, "_kt_wrappers", [])
                if wrappers:
                    break
    if not wrappers:
        logger.warning("No KT wrappers found, skipping LoRA Experts saving")
        return

    adapter_file = os.path.join(output_dir, "adapter_model.safetensors")
    if not os.path.exists(adapter_file):
        adapter_file_bin = os.path.join(output_dir, "adapter_model.bin")
        if os.path.exists(adapter_file_bin):
            state_dict = torch.load(adapter_file_bin, map_location="cpu", weights_only=True)
        else:
            logger.warning(f"No existing adapter file found at {output_dir}, creating new one")
            state_dict = {}
    else:
        state_dict = {}
        with safe_open(adapter_file, framework="pt") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)

    lora_expert_count = 0
    for wrapper in wrappers:
        if wrapper.lora_experts is None:
            continue

        layer_idx = wrapper.layer_idx
        for expert_idx, expert in enumerate(wrapper.lora_experts.experts):
            base_key = f"base_model.model.model.layers.{layer_idx}.mlp.lora_experts.{expert_idx}"
            state_dict[f"{base_key}.le_gate.weight"] = expert.le_gate.weight.data.cpu().clone()
            state_dict[f"{base_key}.le_up.weight"] = expert.le_up.weight.data.cpu().clone()
            state_dict[f"{base_key}.le_down.weight"] = expert.le_down.weight.data.cpu().clone()
            lora_expert_count += 3

        logger.debug(f"Added LoRA Experts for layer {layer_idx} ({len(wrapper.lora_experts.experts)} experts)")

    output_file = os.path.join(output_dir, "adapter_model.safetensors")
    save_file(state_dict, output_file, metadata={"format": "pt"})

    logger.info(
        f"Saved LoRA Experts to {output_file}: "
        f"{len(wrappers)} layers, {lora_expert_count} LoRA Expert tensors added, "
        f"{len(state_dict)} total tensors"
    )


def save_kt_moe_to_adapter(model: nn.Module, output_dir: str) -> None:
    """
    Unified function to save KT MoE weights to adapter file.
    Note: Per-expert PEFT LoRA is saved by PEFT directly, not here.
    This function only handles lora_experts (a separate feature).
    """
    print(f"[save_kt_moe] called, model type={type(model).__name__}, output_dir={output_dir}", flush=True)
    wrappers = getattr(model, "_kt_wrappers", [])
    print(f"[save_kt_moe] direct _kt_wrappers: {len(wrappers) if wrappers else 'None/empty'}", flush=True)
    if not wrappers:
        base_model = model
        for attr in ["base_model", "model"]:
            if hasattr(base_model, attr):
                base_model = getattr(base_model, attr)
                print(f"[save_kt_moe] trying {attr} -> {type(base_model).__name__}", flush=True)
                wrappers = getattr(base_model, "_kt_wrappers", [])
                if wrappers:
                    print(f"[save_kt_moe] found {len(wrappers)} wrappers on {attr}", flush=True)
                    break
    if not wrappers:
        print("[save_kt_moe] No KT wrappers found anywhere, skipping", flush=True)
        return

    has_lora_experts = any(w.lora_experts is not None for w in wrappers)
    le_info = [(i, w.layer_idx, w.lora_experts is not None) for i, w in enumerate(wrappers)]
    print(f"[save_kt_moe] has_lora_experts={has_lora_experts}, wrappers={le_info[:5]}...", flush=True)

    if has_lora_experts:
        save_lora_experts_to_adapter(model, output_dir)
    else:
        print("[save_kt_moe] No lora_experts in KT wrappers", flush=True)


def load_lora_experts_from_adapter(model: nn.Module, adapter_path: str) -> None:
    """
    Load LoRA Experts weights from adapter file into KT wrappers.
    """
    import os
    import re
    from safetensors import safe_open

    wrappers = getattr(model, "_kt_wrappers", [])
    if not wrappers:
        base_model = model
        for attr in ["base_model", "model"]:
            if hasattr(base_model, attr):
                base_model = getattr(base_model, attr)
                wrappers = getattr(base_model, "_kt_wrappers", [])
                if wrappers:
                    break
    if not wrappers:
        logger.warning("No KT wrappers found, skipping LoRA Experts loading")
        return

    wrapper_map = {w.layer_idx: w for w in wrappers if w.lora_experts is not None}
    if not wrapper_map:
        logger.warning("No LoRA Experts found in KT wrappers, skipping")
        return

    # Prefer dedicated lora_experts file, fallback to adapter file
    adapter_file = os.path.join(adapter_path, "lora_experts.safetensors")
    if not os.path.exists(adapter_file):
        adapter_file = os.path.join(adapter_path, "adapter_model.safetensors")
        if not os.path.exists(adapter_file):
            adapter_file = os.path.join(adapter_path, "adapter_model.bin")
            if not os.path.exists(adapter_file):
                logger.warning(f"No lora_experts or adapter file found at {adapter_path}")
                return

    logger.info(f"Loading LoRA Experts from {adapter_file}")

    lora_expert_pattern = re.compile(
        r"base_model\.model\.model\.layers\.(\d+)\.mlp\.lora_experts\.(\d+)\.(le_gate|le_up|le_down)\.weight"
    )

    layer_weights = {}
    with safe_open(adapter_file, framework="pt") as f:
        for key in f.keys():
            match = lora_expert_pattern.match(key)
            if match:
                layer_idx = int(match.group(1))
                expert_idx = int(match.group(2))
                proj_name = match.group(3)
                layer_weights.setdefault(layer_idx, {}).setdefault(expert_idx, {})[proj_name] = f.get_tensor(key)

    loaded_count = 0
    for layer_idx, experts_dict in layer_weights.items():
        if layer_idx not in wrapper_map:
            logger.warning(f"No LoRA Experts for layer {layer_idx}, skipping")
            continue

        wrapper = wrapper_map[layer_idx]
        for expert_idx, proj_dict in experts_dict.items():
            if expert_idx >= len(wrapper.lora_experts.experts):
                continue
            expert = wrapper.lora_experts.experts[expert_idx]
            if "le_gate" in proj_dict:
                expert.le_gate.weight.data.copy_(proj_dict["le_gate"].to(expert.le_gate.weight.device))
            if "le_up" in proj_dict:
                expert.le_up.weight.data.copy_(proj_dict["le_up"].to(expert.le_up.weight.device))
            if "le_down" in proj_dict:
                expert.le_down.weight.data.copy_(proj_dict["le_down"].to(expert.le_down.weight.device))
            loaded_count += 1

    logger.info(f"Loaded LoRA Experts for {loaded_count} experts from {adapter_path}")


def load_kt_moe_from_adapter(model: nn.Module, adapter_path: str) -> None:
    """
    Unified function to load KT MoE weights from adapter file.
    Note: Per-expert PEFT LoRA is loaded by PEFT directly, not here.
    This function only handles lora_experts (a separate feature).
    """
    wrappers = getattr(model, "_kt_wrappers", [])
    if not wrappers:
        base_model = model
        for attr in ["base_model", "model"]:
            if hasattr(base_model, attr):
                base_model = getattr(base_model, attr)
                wrappers = getattr(base_model, "_kt_wrappers", [])
                if wrappers:
                    break
    if not wrappers:
        logger.warning("No KT wrappers found, skipping KT MoE loading")
        return

    has_lora_experts = any(w.lora_experts is not None for w in wrappers)

    if has_lora_experts:
        load_lora_experts_from_adapter(model, adapter_path)
    else:
        logger.info("No lora_experts in KT wrappers (PEFT LoRA is loaded by PEFT directly)")

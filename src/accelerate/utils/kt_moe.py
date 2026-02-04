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
import math
import os
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

import logging as _logging

from .dataclasses import KTransformersPlugin

logger = _logging.getLogger(__name__)
KT_DEBUG = os.environ.get("ACCELERATE_KT_DEBUG", "0") == "1"

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
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False, device=device, dtype=dtype)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False, device=device, dtype=dtype)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False, device=device, dtype=dtype)
        self.act_fn = nn.SiLU()

        nn.init.zeros_(self.down_proj.weight)
        nn.init.kaiming_uniform_(self.gate_proj.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.up_proj.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


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


def load_experts_from_checkpoint_files(
    checkpoint_files: list[str],
    sharded_metadata: dict | None,
    layers_prefix: str,
    moe_config: MOEArchConfig,
    layer_idx: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not SAFETENSORS_AVAILABLE:
        raise ImportError("safetensors is required for loading experts from checkpoint files")

    if not checkpoint_files:
        raise FileNotFoundError("checkpoint_files is empty")

    weight_map = None
    base_dir = os.path.dirname(checkpoint_files[0])
    if sharded_metadata is not None:
        weight_map = sharded_metadata.get("weight_map", None)

    gate_name, up_name, down_name = moe_config.weight_names
    keys = []
    for expert_idx in range(moe_config.expert_num):
        base = f"{layers_prefix}.{layer_idx}.{moe_config.moe_layer_attr}.{moe_config.experts_attr}.{expert_idx}"
        keys.append(f"{base}.{gate_name}.weight")
        keys.append(f"{base}.{up_name}.weight")
        keys.append(f"{base}.{down_name}.weight")

    keys_by_file: dict[str, list[str]] = {}
    for key in keys:
        if weight_map is not None:
            filename = weight_map.get(key)
            if filename is None:
                continue
            file_path = os.path.join(base_dir, filename)
        else:
            file_path = checkpoint_files[0]
        keys_by_file.setdefault(file_path, []).append(key)

    tensor_map: dict[str, torch.Tensor] = {}
    for file_path, file_keys in keys_by_file.items():
        with safe_open(file_path, framework="pt") as f:
            for key in file_keys:
                if key in f.keys():
                    tensor_map[key] = f.get_tensor(key)

    gate_weights = []
    up_weights = []
    down_weights = []
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

    gate_proj = torch.stack(gate_weights, dim=0).cpu().to(torch.bfloat16).contiguous()
    up_proj = torch.stack(up_weights, dim=0).cpu().to(torch.bfloat16).contiguous()
    down_proj = torch.stack(down_weights, dim=0).cpu().to(torch.bfloat16).contiguous()
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
        peft_lora_modules: dict[int, dict[str, tuple[nn.Module, nn.Module]]] | None,
        lora_ref: torch.Tensor,
        hidden_size: int,
        num_experts_per_tok: int,
        layer_idx: int,
        training: bool,
        train_lora: bool,
        precomputed_output: torch.Tensor | None = None,
        weight_names: tuple[str, str, str] | None = None,
    ) -> torch.Tensor:
        original_device = hidden_states.device
        original_dtype = hidden_states.dtype
        batch_size, seq_len, _ = hidden_states.shape
        qlen = batch_size * seq_len

        import torch.distributed as dist
        dist_on = dist.is_initialized() and dist.get_world_size() > 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist_on else 1

        ctx.use_broadcast = wrapper is None

        if precomputed_output is not None:
            output = precomputed_output
        elif dist_on:
            # ---- Data-parallel gather/scatter path ----
            # Each rank has its own batch. Gather all on rank 0, compute, scatter back.
            # Both batch_size and seq_len may differ across ranks.

            # 1. Exchange qlen from each rank
            local_qlen_t = torch.tensor([qlen], device=original_device, dtype=torch.int64)
            all_qlen_t = [torch.empty(1, device=original_device, dtype=torch.int64) for _ in range(world_size)]
            dist.all_gather(all_qlen_t, local_qlen_t)
            qlen_max = max(q.item() for q in all_qlen_t)

            # 2. Flatten everything to 1D [qlen, ...] and pad to [qlen_max, ...]
            #    CRITICAL: hidden_states must be flattened BEFORE padding so that
            #    token positions align with topk_ids/topk_weights (also flat).
            def _pad_flat(t, target_len, cur_len):
                """Pad a [cur_len, ...] tensor to [target_len, ...]."""
                if cur_len == target_len:
                    return t.contiguous()
                pad_shape = list(t.shape)
                pad_shape[0] = target_len - cur_len
                return torch.cat([t, torch.zeros(pad_shape, device=t.device, dtype=t.dtype)], dim=0).contiguous()

            hs_flat = hidden_states.view(qlen, hidden_size)           # [qlen, H]
            hs_padded = _pad_flat(hs_flat, qlen_max, qlen)            # [qlen_max, H]
            ids_padded = _pad_flat(topk_ids, qlen_max, qlen)          # [qlen_max, K]
            wts_padded = _pad_flat(topk_weights, qlen_max, qlen)      # [qlen_max, K]

            # 3. Gather on rank 0
            if rank == 0:
                gathered_hs = [torch.empty_like(hs_padded) for _ in range(world_size)]
                gathered_ids = [torch.empty_like(ids_padded) for _ in range(world_size)]
                gathered_wts = [torch.empty_like(wts_padded) for _ in range(world_size)]
            else:
                gathered_hs = gathered_ids = gathered_wts = None

            dist.gather(hs_padded, gathered_hs, dst=0)
            dist.gather(ids_padded, gathered_ids, dst=0)
            dist.gather(wts_padded, gathered_wts, dst=0)

            # 4. Rank 0: run KT kernel on full gathered batch
            if rank == 0:
                all_hs = torch.cat(gathered_hs, dim=0)   # [qlen_max*W, H]
                all_ids = torch.cat(gathered_ids, dim=0)  # [qlen_max*W, K]
                all_wts = torch.cat(gathered_wts, dim=0)  # [qlen_max*W, K]
                total_qlen = qlen_max * world_size

                all_output = wrapper.forward_sft(
                    hidden_states=all_hs,
                    expert_ids=all_ids,
                    weights=all_wts,
                    save_for_backward=training,
                    output_device=original_device,
                )
                # all_output: [total_qlen, H] → split into per-rank chunks of qlen_max
                all_output = all_output.to(dtype=original_dtype)
                scatter_list = list(all_output.view(world_size, qlen_max, hidden_size).unbind(0))
                scatter_list = [c.contiguous() for c in scatter_list]
            else:
                scatter_list = None

            # 5. Scatter back and trim to local qlen
            output_padded = torch.empty(qlen_max, hidden_size, device=original_device, dtype=original_dtype)
            dist.scatter(output_padded, scatter_list, src=0)
            output = output_padded[:qlen].view(batch_size, seq_len, hidden_size)
        elif wrapper is not None:
            # ---- Single-GPU path ----
            input_flat = hidden_states.view(qlen, hidden_size)
            expert_ids = topk_ids.view(qlen, num_experts_per_tok)
            weights = topk_weights.view(qlen, num_experts_per_tok)

            output = wrapper.forward_sft(
                hidden_states=input_flat,
                expert_ids=expert_ids,
                weights=weights,
                save_for_backward=training,
                output_device=original_device,
            )
            output = output.view(batch_size, seq_len, hidden_size).to(dtype=original_dtype)
        else:
            output = torch.empty(
                batch_size, seq_len, hidden_size, device=original_device, dtype=original_dtype
            )

        ctx.wrapper = wrapper
        ctx.peft_lora_modules = peft_lora_modules
        ctx.hidden_size = hidden_size
        ctx.qlen = qlen
        ctx.batch_size = batch_size
        ctx.seq_len = seq_len
        ctx.original_device = original_device
        ctx.original_dtype = original_dtype
        ctx.layer_idx = layer_idx
        ctx.train_lora = train_lora
        ctx.weights_shape = topk_weights.shape
        ctx.weights_dtype = topk_weights.dtype
        ctx.weights_device = topk_weights.device
        ctx.dist_on = dist_on
        ctx.world_size = world_size
        ctx.num_experts_per_tok = num_experts_per_tok
        ctx.weight_names = weight_names

        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        qlen = ctx.qlen
        hidden_size = ctx.hidden_size
        batch_size = ctx.batch_size
        seq_len = ctx.seq_len
        dist_on = ctx.dist_on
        world_size = ctx.world_size
        num_experts_per_tok = ctx.num_experts_per_tok

        import torch.distributed as dist
        rank = dist.get_rank() if dist.is_initialized() else 0

        if dist_on:
            # ---- Data-parallel gather/scatter backward ----
            # Both batch_size and seq_len may differ across ranks.

            # 1. Exchange qlen from each rank
            local_qlen_t = torch.tensor([qlen], device=ctx.original_device, dtype=torch.int64)
            all_qlen_t = [torch.empty(1, device=ctx.original_device, dtype=torch.int64) for _ in range(world_size)]
            dist.all_gather(all_qlen_t, local_qlen_t)
            qlen_max = max(q.item() for q in all_qlen_t)

            # 2. Flatten grad_output to [qlen, H] and pad to [qlen_max, H]
            def _pad_flat(t, target_len, cur_len):
                if cur_len == target_len:
                    return t.contiguous()
                pad_shape = list(t.shape)
                pad_shape[0] = target_len - cur_len
                return torch.cat([t, torch.zeros(pad_shape, device=t.device, dtype=t.dtype)], dim=0).contiguous()

            grad_out_flat = grad_output.view(qlen, hidden_size)
            grad_out_padded = _pad_flat(grad_out_flat, qlen_max, qlen)  # [qlen_max, H]

            # 3. Gather on rank 0
            if rank == 0:
                gathered_go = [torch.empty_like(grad_out_padded) for _ in range(world_size)]
            else:
                gathered_go = None
            dist.gather(grad_out_padded, gathered_go, dst=0)

            # 4. Rank 0: run backward on full gathered batch
            if rank == 0:
                all_go = torch.cat(gathered_go, dim=0)  # [qlen_max*W, H]
                total_qlen = qlen_max * world_size

                # C++ kernel always computes and returns grad_loras regardless of lora_params
                backward_out = ctx.wrapper.backward(
                    all_go,
                    output_device=ctx.original_device,
                )
                if isinstance(backward_out, tuple) and len(backward_out) == 2:
                    all_grad_input, all_grad_weights = backward_out
                    grad_loras = None
                elif isinstance(backward_out, tuple) and len(backward_out) == 3:
                    all_grad_input, grad_loras, all_grad_weights = backward_out
                else:
                    raise ValueError("KTMoEWrapper.backward returned unexpected format.")

                # all_grad_input: [total_qlen, H], all_grad_weights: [total_qlen, K]
                all_grad_input = all_grad_input.to(dtype=ctx.original_dtype)
                all_grad_weights = all_grad_weights.to(dtype=torch.bfloat16)

                scatter_gi = list(all_grad_input.view(world_size, qlen_max, -1).unbind(0))
                scatter_gi = [c.contiguous() for c in scatter_gi]
                scatter_gw = list(all_grad_weights.view(world_size, qlen_max, -1).unbind(0))
                scatter_gw = [c.contiguous() for c in scatter_gw]
            else:
                scatter_gi = None
                scatter_gw = None
                grad_loras = None

            # 5. Scatter back and trim to local qlen
            gi_padded = torch.empty(qlen_max, hidden_size, device=ctx.original_device, dtype=ctx.original_dtype)
            gw_padded = torch.empty(qlen_max, num_experts_per_tok, device=ctx.weights_device, dtype=torch.bfloat16)
            dist.scatter(gi_padded, scatter_gi, src=0)
            dist.scatter(gw_padded, scatter_gw, src=0)
            grad_input = gi_padded[:qlen].view(batch_size, seq_len, hidden_size)
            grad_weights = gw_padded[:qlen].view(ctx.weights_shape).to(dtype=ctx.weights_dtype)

        elif not ctx.use_broadcast:
            # ---- Single-GPU path ----
            grad_output_flat = grad_output.view(qlen, hidden_size)
            # C++ kernel always computes and returns grad_loras regardless of lora_params
            backward_out = ctx.wrapper.backward(
                grad_output_flat,
                output_device=ctx.original_device,
            )
            if isinstance(backward_out, tuple) and len(backward_out) == 2:
                grad_input, grad_weights = backward_out
                grad_loras = None
            elif isinstance(backward_out, tuple) and len(backward_out) == 3:
                grad_input, grad_loras, grad_weights = backward_out
            else:
                raise ValueError("KTMoEWrapper.backward returned unexpected format.")
            grad_input = grad_input.view(batch_size, seq_len, hidden_size).to(dtype=ctx.original_dtype)
            grad_weights = grad_weights.to(dtype=torch.bfloat16)
        else:
            # No wrapper, no dist — shouldn't happen in normal flow
            grad_input = torch.zeros(batch_size, seq_len, hidden_size, device=ctx.original_device, dtype=ctx.original_dtype)
            grad_weights = torch.zeros(ctx.weights_shape, device=ctx.weights_device, dtype=ctx.weights_dtype)
            grad_loras = None

        # LoRA gradients: only rank 0 needs them (only rank 0 has KT wrapper).
        # No broadcast needed — non-rank-0 optimizer skips params with grad=None.
        # PEFT LoRA: write gradients to PEFT modules directly.
        if ctx.train_lora and ctx.peft_lora_modules is not None and grad_loras is not None and rank == 0:
            # Map from C++ kernel key prefixes to weight_names indices
            # weight_names = (gate_name, up_name, down_name), e.g., ("gate_proj", "up_proj", "down_proj")
            gate_name, up_name, down_name = ctx.weight_names

            # Helper to write gradient to PEFT LoRA module
            def _write_grad_to_peft(grad_key: str, proj_name: str, is_lora_b: bool):
                """Write stacked gradient tensor to individual PEFT LoRA modules."""
                grad_tensor = grad_loras.get(grad_key) or grad_loras.get(f"grad_{grad_key}")
                if grad_tensor is None:
                    return

                # grad_tensor shape: [num_experts, ...], split by expert
                num_experts = grad_tensor.shape[0]
                for expert_idx in range(num_experts):
                    expert_loras = ctx.peft_lora_modules.get(expert_idx, {})
                    lora_pair = expert_loras.get(proj_name)
                    if lora_pair is None:
                        continue

                    lora_module = lora_pair[1] if is_lora_b else lora_pair[0]  # (lora_A, lora_B)
                    if not hasattr(lora_module, 'weight') or not lora_module.weight.requires_grad:
                        continue

                    param = lora_module.weight
                    expert_grad = grad_tensor[expert_idx].clone().to(dtype=param.dtype, device=param.device)

                    # Scale gradient by world_size (same as before)
                    if world_size > 1:
                        expert_grad /= world_size

                    # Accumulate gradient (supports gradient_accumulation_steps).
                    # C++ grad_lora buffers are now zeroed before each backward call,
                    # so we need to accumulate across micro-batches here.
                    if param.grad is None:
                        param.grad = expert_grad
                    else:
                        param.grad = param.grad + expert_grad

            # Write gradients for all projections
            _write_grad_to_peft("gate_lora_a", gate_name, is_lora_b=False)
            _write_grad_to_peft("gate_lora_b", gate_name, is_lora_b=True)
            _write_grad_to_peft("up_lora_a", up_name, is_lora_b=False)
            _write_grad_to_peft("up_lora_b", up_name, is_lora_b=True)
            _write_grad_to_peft("down_lora_a", down_name, is_lora_b=False)
            _write_grad_to_peft("down_lora_b", down_name, is_lora_b=True)

        return grad_input, None, grad_weights, None, None, None, None, None, None, None, None, None, None


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
        lora_experts: "LoRAExperts | None" = None,  # Deprecated, ignored
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
        has_gpu_components = self.shared_experts is not None or self.lora_experts is not None
        # Only ask KT kernel to save forward caches if a backward pass will actually run.
        save_for_backward = (
            self.training
            and torch.is_grad_enabled()
            and (hidden_states.requires_grad or topk_weights.requires_grad or train_lora)
        )
        if KT_DEBUG:
            logger.warning(
                "[KT DEBUG] rank %s KTMoELayerWrapper.forward layer=%s train_lora=%s has_gpu_components=%s "
                "save_for_backward=%s",
                rank,
                self.layer_idx,
                train_lora,
                has_gpu_components,
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

        # Overlap: rank 0 submits CPU expert work, all ranks compute GPU shared_experts concurrently.
        # In dist mode, all ranks MUST enter _forward_with_overlap together because it uses
        # collectives (all_gather, gather, scatter) that require all-rank participation.
        use_overlap = has_gpu_components and save_for_backward and (dist_on or self.wrapper is not None)
        if KT_DEBUG:
            logger.warning(
                "[KT DEBUG] rank %s KTMoELayerWrapper.forward layer=%s use_overlap=%s",
                rank,
                self.layer_idx,
                use_overlap,
            )

        def compute_gpu_output() -> torch.Tensor | None:
            gpu_output = None
            if self.shared_experts is not None:
                if KT_DEBUG:
                    logger.warning(
                        "[KT DEBUG] rank %s KTMoELayerWrapper.forward shared_experts start layer=%s",
                        rank,
                        self.layer_idx,
                    )
                gpu_output = self.shared_experts(hidden_states)
                if KT_DEBUG:
                    logger.warning(
                        "[KT DEBUG] rank %s KTMoELayerWrapper.forward shared_experts done layer=%s",
                        rank,
                        self.layer_idx,
                    )
            if self.lora_experts is not None:
                if KT_DEBUG:
                    logger.warning(
                        "[KT DEBUG] rank %s KTMoELayerWrapper.forward lora_experts start layer=%s",
                        rank,
                        self.layer_idx,
                    )
                lora_out = self.lora_experts(hidden_states)
                if KT_DEBUG:
                    logger.warning(
                        "[KT DEBUG] rank %s KTMoELayerWrapper.forward lora_experts done layer=%s",
                        rank,
                        self.layer_idx,
                    )
                gpu_output = lora_out if gpu_output is None else gpu_output + lora_out
            return gpu_output

        precomputed_output = None
        gpu_output = None
        # In dist mode, all ranks must participate in overlap (gather/scatter collectives).
        # In single-GPU mode, only rank 0 (which has wrapper) uses overlap.
        if use_overlap and (dist_on or self.wrapper is not None):
            if KT_DEBUG:
                logger.warning(
                    "[KT DEBUG] rank %s KTMoELayerWrapper.forward overlap path start layer=%s",
                    rank,
                    self.layer_idx,
                )
            precomputed_output, gpu_output = self._forward_with_overlap(
                hidden_states,
                topk_ids,
                topk_weights,
                compute_gpu_output,
                save_for_backward,
            )
            if KT_DEBUG:
                logger.warning(
                    "[KT DEBUG] rank %s KTMoELayerWrapper.forward overlap path done layer=%s precomputed=%s %s",
                    rank,
                    self.layer_idx,
                    tuple(precomputed_output.shape),
                    precomputed_output.dtype,
                )
        else:
            gpu_output = compute_gpu_output()

        lora_ref = hidden_states.new_empty(())
        if train_lora and self._peft_lora_modules:
            # Get a PEFT LoRA parameter for autograd tracking
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
            self._peft_lora_modules,  # Pass PEFT LoRA modules instead of lora_params
            lora_ref,
            self.hidden_size,
            self.moe_config.num_experts_per_tok,
            self.layer_idx,
            save_for_backward,
            train_lora,
            precomputed_output,
            self.moe_config.weight_names,  # (gate_name, up_name, down_name) for grad mapping
        )

        if gpu_output is not None:
            if KT_DEBUG:
                logger.warning(
                    "[KT DEBUG] rank %s KTMoELayerWrapper.forward add gpu_output layer=%s",
                    rank,
                    self.layer_idx,
                )
            moe_output = moe_output + gpu_output

        if KT_DEBUG:
            logger.warning(
                "[KT DEBUG] rank %s KTMoELayerWrapper.forward done layer=%s output=%s %s",
                rank,
                self.layer_idx,
                tuple(moe_output.shape),
                moe_output.dtype,
            )
        return moe_output

    def _compute_routing(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        router = getattr(self, self._router_attr)
        if self.router_type == "deepseek_gate":
            router_output = router(hidden_states)
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

    def _forward_with_overlap(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        compute_gpu_output: Any,
        save_for_backward: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        import torch.distributed as dist

        batch_size, seq_len, _ = hidden_states.shape
        original_device = hidden_states.device
        original_dtype = hidden_states.dtype

        dist_on = dist.is_initialized() and dist.get_world_size() > 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist_on else 1

        qlen = batch_size * seq_len

        if dist_on:
            # ---- Gather inputs from all ranks before submitting CPU work ----
            local_qlen_t = torch.tensor([qlen], device=original_device, dtype=torch.int64)
            all_qlen_t = [torch.empty(1, device=original_device, dtype=torch.int64) for _ in range(world_size)]
            dist.all_gather(all_qlen_t, local_qlen_t)
            qlen_max = max(q.item() for q in all_qlen_t)

            def _pad_flat(t, target_len, cur_len):
                if cur_len == target_len:
                    return t.contiguous()
                pad_shape = list(t.shape)
                pad_shape[0] = target_len - cur_len
                return torch.cat([t, torch.zeros(pad_shape, device=t.device, dtype=t.dtype)], dim=0).contiguous()

            hs_flat = hidden_states.view(qlen, self.hidden_size)      # [qlen, H]
            hs_padded = _pad_flat(hs_flat, qlen_max, qlen)            # [qlen_max, H]
            ids_padded = _pad_flat(topk_ids, qlen_max, qlen)
            wts_padded = _pad_flat(topk_weights, qlen_max, qlen)

            # Differentiable all_gather for hidden_states (needed by both CPU experts and shared_experts)
            from torch.distributed.nn.functional import all_gather as diff_all_gather
            all_hs_list = diff_all_gather(hs_padded)  # list of [qlen_max, H] with autograd
            all_hs = torch.cat(all_hs_list, dim=0)    # [qlen_max*W, H]
            total_qlen = qlen_max * world_size

            # Gather ids/wts to rank 0 only (no grad needed, only rank 0 uses them for CPU experts)
            if rank == 0:
                gathered_ids = [torch.empty_like(ids_padded) for _ in range(world_size)]
                gathered_wts = [torch.empty_like(wts_padded) for _ in range(world_size)]
            else:
                gathered_ids = gathered_wts = None

            dist.gather(ids_padded, gathered_ids, dst=0)
            dist.gather(wts_padded, gathered_wts, dst=0)

            # Rank 0: submit async CPU expert work on gathered full batch
            if rank == 0:
                all_ids = torch.cat(gathered_ids, dim=0)  # [qlen_max*W, K]
                all_wts = torch.cat(gathered_wts, dim=0)  # [qlen_max*W, K]
                self.wrapper.submit_forward_sft(
                    all_hs.detach(), all_ids, all_wts, save_for_backward=save_for_backward
                )

            # All ranks compute shared_experts on the SAME gathered input concurrently
            # with CPU expert work. FSDP2-wrapped shared_experts requires all ranks to
            # participate. Identical input ensures identical output, eliminating precision
            # differences from per-rank input divergence.
            gpu_output = None
            if self.shared_experts is not None:
                all_shared_out = self.shared_experts(
                    all_hs.view(1, total_qlen, self.hidden_size)
                )
                all_shared_out = all_shared_out.to(dtype=original_dtype)
                # Each rank takes its own slice
                all_shared_out = all_shared_out.view(world_size, qlen_max, self.hidden_size)
                gpu_output = all_shared_out[rank, :qlen].view(batch_size, seq_len, self.hidden_size)

            if self.lora_experts is not None:
                lora_out = self.lora_experts(hidden_states)
                gpu_output = lora_out if gpu_output is None else gpu_output + lora_out

            # Rank 0: sync CPU result and scatter back
            if rank == 0:
                cpu_output_gpu = self.wrapper.sync_forward_sft(output_device=original_device)
                # cpu_output_gpu: [total_qlen, H] → split per rank
                cpu_output_gpu = cpu_output_gpu.to(dtype=original_dtype)
                scatter_list = list(cpu_output_gpu.view(world_size, qlen_max, self.hidden_size).unbind(0))
                scatter_list = [c.contiguous() for c in scatter_list]
            else:
                scatter_list = None

            output_padded = torch.empty(qlen_max, self.hidden_size, device=original_device, dtype=original_dtype)
            dist.scatter(output_padded, scatter_list, src=0)
            precomputed_output = output_padded[:qlen].view(batch_size, seq_len, self.hidden_size)
        else:
            # ---- Single-GPU overlap path ----
            input_flat = hidden_states.view(qlen, self.hidden_size)
            expert_ids = topk_ids.view(qlen, self.moe_config.num_experts_per_tok)
            weights = topk_weights.view(qlen, self.moe_config.num_experts_per_tok)

            self.wrapper.submit_forward_sft(input_flat, expert_ids, weights, save_for_backward=save_for_backward)
            gpu_output = compute_gpu_output()
            cpu_output_gpu = self.wrapper.sync_forward_sft(output_device=original_device)
            precomputed_output = cpu_output_gpu.view(batch_size, seq_len, self.hidden_size).to(dtype=original_dtype)

        return precomputed_output, gpu_output

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
    kt_backend = getattr(kt_plugin, "kt_backend", "AMXBF16")
    kt_method = kt_backend_map.get(kt_backend, "AMXBF16_SFT")

    if "SkipLoRA" in kt_method:
        logger.info(f"Using SkipLoRA backend: {kt_method} (MoE LoRA gradients will be skipped)")

    threadpool_count = getattr(kt_plugin, "kt_threadpool_count", 1) if getattr(kt_plugin, "kt_tp_enabled", False) else 1

    kt_weight_path = getattr(kt_plugin, "kt_weight_path", None)
    use_kt_weight_path = kt_weight_path is not None
    if use_kt_weight_path:
        logger.info(f"Loading INT8 weights from kt_weight_path: {kt_weight_path}")

    checkpoint_files = getattr(kt_plugin, "kt_checkpoint_files", None)
    sharded_metadata = getattr(kt_plugin, "kt_sharded_metadata", None)
    use_checkpoint_files = bool(checkpoint_files) and not use_kt_weight_path
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

    model_container, layers = _get_model_container_and_layers(model, purpose="wrapping")

    for layer_idx, layer in enumerate(layers):
        moe_module = get_moe_module(layer, moe_config)
        if moe_module is None:
            continue

        logger.info(
            f"Wrapping MoE layer {layer_idx} with KTMoEWrapper "
            f"(method={kt_method}, tp={threadpool_count})"
        )

        # Only rank 0 loads weights and initializes KT kernel
        gate_proj, up_proj, down_proj = None, None, None
        wrapper = None

        if is_rank_0:
            if use_kt_weight_path:
                if checkpoint_files:
                    layers_prefix = _get_layers_prefix(model.config)
                    logger.info(
                        f"  Layer {layer_idx}: loading BF16 from checkpoint files for backward, "
                        f"INT8 forward from kt_weight_path={kt_weight_path!r}"
                    )
                    gate_proj, up_proj, down_proj = load_experts_from_checkpoint_files(
                        checkpoint_files=checkpoint_files,
                        sharded_metadata=sharded_metadata,
                        layers_prefix=layers_prefix,
                        moe_config=moe_config,
                        layer_idx=layer_idx,
                    )
                else:
                    logger.warning(
                        f"  Layer {layer_idx}: no checkpoint files available for BF16 backward weights! "
                        f"Falling back to extract_moe_weights (may be empty with cpu_ram_efficient_loading)"
                    )
                    gate_proj, up_proj, down_proj = extract_moe_weights(moe_module, moe_config)
                    gate_proj = gate_proj.cpu().to(torch.bfloat16).contiguous()
                    up_proj = up_proj.cpu().to(torch.bfloat16).contiguous()
                    down_proj = down_proj.cpu().to(torch.bfloat16).contiguous()
            elif use_checkpoint_files:
                layers_prefix = _get_layers_prefix(model.config)
                gate_proj, up_proj, down_proj = load_experts_from_checkpoint_files(
                    checkpoint_files=checkpoint_files,
                    sharded_metadata=sharded_metadata,
                    layers_prefix=layers_prefix,
                    moe_config=moe_config,
                    layer_idx=layer_idx,
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
                wrapper._bf16_gate_proj = gate_proj
                wrapper._bf16_up_proj = up_proj
                wrapper._bf16_down_proj = down_proj
                print(
                    f"[kt_moe] Layer {layer_idx}: calling wrapper.load_weights() "
                    f"(pre-quantized path, kt_weight_path={kt_weight_path!r})",
                    flush=True,
                )
                wrapper.load_weights(physical_to_logical_map)
                wrapper._bf16_gate_proj = None
                wrapper._bf16_up_proj = None
                wrapper._bf16_down_proj = None
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

        layer_wrapper = KTMoELayerWrapper(
            original_moe=moe_module,
            wrapper=wrapper,
            lora_params=None,
            moe_config=moe_config,
            hidden_size=hidden_size,
            layer_idx=layer_idx,
        )

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

        # Allocate contiguous bf16 buffers and populate with initial PEFT values (all ranks)
        lora_buffers = _create_lora_view_buffers(peft_lora_modules, moe_config, torch.bfloat16)

        # Rank 0: pass buffers to C++ wrapper (init_lora_weights stores them via .contiguous() no-op)
        if is_rank_0 and wrapper.wrapper is not None:
            wrapper.wrapper.init_lora_weights(**lora_buffers)
            logger.info(f"[kt_adapt_peft_lora] Layer {layer_idx}: synced PEFT LoRA to C++ kernel")

        # All ranks: replace PEFT weights with views into the contiguous buffers
        _replace_peft_weights_with_views(peft_lora_modules, lora_buffers, moe_config)

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


def _collect_peft_lora_tensors(
    peft_lora_modules: dict[int, dict[str, tuple[nn.Module, nn.Module]]],
    moe_config: MOEArchConfig,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    """
    Collect PEFT LoRA weights into stacked tensors for C++ kernel.

    Args:
        peft_lora_modules: {expert_idx: {proj_name: (lora_A, lora_B)}}
        moe_config: MoE architecture config
        dtype: Target dtype

    Returns:
        Dict with gate_lora_a, gate_lora_b, up_lora_a, up_lora_b, down_lora_a, down_lora_b tensors
    """
    gate_name, up_name, down_name = moe_config.weight_names
    num_experts = moe_config.expert_num

    # Get shape info from first expert
    first_expert_loras = peft_lora_modules.get(0, {})
    if not first_expert_loras:
        raise RuntimeError("No PEFT LoRA found on expert 0")

    gate_lora = first_expert_loras.get(gate_name)
    if gate_lora is None:
        raise RuntimeError(f"No PEFT LoRA found on expert 0 {gate_name}")

    lora_A_weight = gate_lora[0].weight  # [rank, in_features]
    lora_rank = lora_A_weight.shape[0]
    hidden_size = lora_A_weight.shape[1]
    intermediate_size = gate_lora[1].weight.shape[0]  # [out_features, rank]

    # Collect all experts' LoRA weights
    gate_lora_a_list, gate_lora_b_list = [], []
    up_lora_a_list, up_lora_b_list = [], []
    down_lora_a_list, down_lora_b_list = [], []

    for expert_idx in range(num_experts):
        expert_loras = peft_lora_modules.get(expert_idx, {})

        # Gate proj
        if gate_name in expert_loras:
            lora_A, lora_B = expert_loras[gate_name]
            gate_lora_a_list.append(lora_A.weight.data.to(dtype=dtype).cpu())
            gate_lora_b_list.append(lora_B.weight.data.to(dtype=dtype).cpu())
        else:
            gate_lora_a_list.append(torch.zeros(lora_rank, hidden_size, dtype=dtype))
            gate_lora_b_list.append(torch.zeros(intermediate_size, lora_rank, dtype=dtype))

        # Up proj
        if up_name in expert_loras:
            lora_A, lora_B = expert_loras[up_name]
            up_lora_a_list.append(lora_A.weight.data.to(dtype=dtype).cpu())
            up_lora_b_list.append(lora_B.weight.data.to(dtype=dtype).cpu())
        else:
            up_lora_a_list.append(torch.zeros(lora_rank, hidden_size, dtype=dtype))
            up_lora_b_list.append(torch.zeros(intermediate_size, lora_rank, dtype=dtype))

        # Down proj
        if down_name in expert_loras:
            lora_A, lora_B = expert_loras[down_name]
            down_lora_a_list.append(lora_A.weight.data.to(dtype=dtype).cpu())
            down_lora_b_list.append(lora_B.weight.data.to(dtype=dtype).cpu())
        else:
            down_lora_a_list.append(torch.zeros(lora_rank, intermediate_size, dtype=dtype))
            down_lora_b_list.append(torch.zeros(hidden_size, lora_rank, dtype=dtype))

    return {
        "gate_lora_a": torch.stack(gate_lora_a_list, dim=0).contiguous(),
        "gate_lora_b": torch.stack(gate_lora_b_list, dim=0).contiguous(),
        "up_lora_a": torch.stack(up_lora_a_list, dim=0).contiguous(),
        "up_lora_b": torch.stack(up_lora_b_list, dim=0).contiguous(),
        "down_lora_a": torch.stack(down_lora_a_list, dim=0).contiguous(),
        "down_lora_b": torch.stack(down_lora_b_list, dim=0).contiguous(),
    }


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


def _replace_peft_weights_with_views(
    peft_lora_modules: dict[int, dict[str, tuple[nn.Module, nn.Module]]],
    buffers: dict[str, torch.Tensor],
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

    for expert_idx in range(num_experts):
        expert_loras = peft_lora_modules.get(expert_idx, {})
        for proj_name, (key_a, key_b) in proj_to_keys.items():
            if proj_name not in expert_loras:
                continue
            lora_A, lora_B = expert_loras[proj_name]
            lora_A.weight = nn.Parameter(buffers[key_a][expert_idx], requires_grad=True)
            lora_B.weight = nn.Parameter(buffers[key_b][expert_idx], requires_grad=True)


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



def sync_kt_lora_gradients(model: nn.Module):
    """
    Synchronize KT LoRA parameter gradients across distributed ranks.

    In FSDP2 training, KT LoRA params are marked as ignored (not sharded), so their
    gradients are not automatically synchronized. This function performs an all-reduce
    on the gradients to ensure consistent updates across all ranks.
    """
    import torch.distributed as dist

    if not dist.is_initialized():
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
            state_dict[f"{base_key}.gate_proj.weight"] = expert.gate_proj.weight.data.cpu().clone()
            state_dict[f"{base_key}.up_proj.weight"] = expert.up_proj.weight.data.cpu().clone()
            state_dict[f"{base_key}.down_proj.weight"] = expert.down_proj.weight.data.cpu().clone()
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
    wrappers = getattr(model, "_kt_wrappers", [])
    if not wrappers:
        logger.warning("No KT wrappers found, skipping KT MoE saving")
        return

    has_lora_experts = any(w.lora_experts is not None for w in wrappers)

    if has_lora_experts:
        save_lora_experts_to_adapter(model, output_dir)
    else:
        logger.info("No lora_experts in KT wrappers (PEFT LoRA is saved by PEFT directly)")


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

    adapter_file = os.path.join(adapter_path, "adapter_model.safetensors")
    if not os.path.exists(adapter_file):
        adapter_file = os.path.join(adapter_path, "adapter_model.bin")
        if not os.path.exists(adapter_file):
            logger.warning(f"No adapter file found at {adapter_path}")
            return

    logger.info(f"Loading LoRA Experts from {adapter_file}")

    lora_expert_pattern = re.compile(
        r"base_model\.model\.model\.layers\.(\d+)\.mlp\.lora_experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight"
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
            if "gate_proj" in proj_dict:
                expert.gate_proj.weight.data.copy_(proj_dict["gate_proj"].to(expert.gate_proj.weight.device))
            if "up_proj" in proj_dict:
                expert.up_proj.weight.data.copy_(proj_dict["up_proj"].to(expert.up_proj.weight.device))
            if "down_proj" in proj_dict:
                expert.down_proj.weight.data.copy_(proj_dict["down_proj"].to(expert.down_proj.weight.device))
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

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

    with _maybe_zero3_gathered_parameters(gather_params):
        for proj, container, param_name, weight_param in _iter_weight_params():
            original_device = weight_param.device
            original_dtype = weight_param.dtype
            new_param = nn.Parameter(
                torch.empty(0, device=original_device, dtype=original_dtype),
                requires_grad=False,
            )

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

    logger.debug("Cleared original expert weights for MoE module")


# =============================================================================
# LoRA Initialization
# =============================================================================


def _get_peft_lora_weights(module: nn.Module) -> tuple[torch.Tensor, torch.Tensor] | None:
    """
    Extract LoRA A and B weights from a PEFT-wrapped module.

    PEFT wraps modules with lora_A and lora_B submodules. This function extracts
    the weight tensors from these submodules.

    Returns:
        Tuple of (lora_A_weight, lora_B_weight) or None if not PEFT-wrapped.
    """
    # Check for PEFT LoRA structure
    lora_A = getattr(module, "lora_A", None)
    lora_B = getattr(module, "lora_B", None)

    if lora_A is None or lora_B is None:
        return None

    # PEFT stores adapters in a ModuleDict, default adapter is "default"
    if isinstance(lora_A, nn.ModuleDict):
        logger.info(f"[_get_peft_lora_weights] lora_A keys={list(lora_A.keys())}")
        lora_A = lora_A["default"] if "default" in lora_A else None
        lora_B = lora_B["default"] if "default" in lora_B else None

    if lora_A is None or lora_B is None:
        return None

    # Get the weight tensors
    if hasattr(lora_A, "weight"):
        logger.info(f"[_get_peft_lora_weights] lora_A.weight shape={lora_A.weight.shape}, lora_B.weight shape={lora_B.weight.shape}")
        return lora_A.weight.data, lora_B.weight.data

    return None


def extract_peft_lora_from_experts(
    experts: nn.ModuleList,
    moe_config: MOEArchConfig,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[dict[str, nn.Parameter] | None, int | None]:
    """
    Extract LoRA weights from PEFT-wrapped expert modules.

    Args:
        experts: ModuleList of expert modules (potentially PEFT-wrapped)
        moe_config: MoE architecture configuration
        dtype: Target dtype for LoRA weights

    Returns:
        Tuple of (lora_params dict, lora_rank) or (None, None) if no PEFT LoRA found.
    """
    gate_name, up_name, down_name = moe_config.weight_names
    num_experts = len(experts)

    # Check first expert to see if PEFT LoRA is applied
    first_expert = experts[0]
    gate_proj = getattr(first_expert, gate_name, None)
    if gate_proj is None:
        return None, None

    gate_lora = _get_peft_lora_weights(gate_proj)
    if gate_lora is None:
        logger.debug("No PEFT LoRA found on expert gate_proj, will use KT native LoRA init")
        return None, None

    # Determine LoRA rank from first expert
    lora_A, lora_B = gate_lora
    lora_rank = lora_A.shape[0]  # lora_A shape: [rank, in_features]
    hidden_size = lora_A.shape[1]
    intermediate_size = lora_B.shape[0]  # lora_B shape: [out_features, rank]

    logger.info(f"Extracting PEFT LoRA from experts: rank={lora_rank}, hidden={hidden_size}, intermediate={intermediate_size}")

    # Initialize tensors to collect LoRA weights
    gate_lora_a_list = []
    gate_lora_b_list = []
    up_lora_a_list = []
    up_lora_b_list = []
    down_lora_a_list = []
    down_lora_b_list = []

    for expert_idx, expert in enumerate(experts):
        # Gate proj
        gate_proj = getattr(expert, gate_name)
        gate_lora = _get_peft_lora_weights(gate_proj)
        if gate_lora is not None:
            gate_lora_a_list.append(gate_lora[0].to(dtype=dtype).cpu().contiguous())
            gate_lora_b_list.append(gate_lora[1].to(dtype=dtype).cpu().contiguous())
        else:
            # Fallback: zero init
            gate_lora_a_list.append(torch.zeros(lora_rank, hidden_size, dtype=dtype))
            gate_lora_b_list.append(torch.zeros(intermediate_size, lora_rank, dtype=dtype))

        # Up proj
        up_proj = getattr(expert, up_name)
        up_lora = _get_peft_lora_weights(up_proj)
        if up_lora is not None:
            up_lora_a_list.append(up_lora[0].to(dtype=dtype).cpu().contiguous())
            up_lora_b_list.append(up_lora[1].to(dtype=dtype).cpu().contiguous())
        else:
            up_lora_a_list.append(torch.zeros(lora_rank, hidden_size, dtype=dtype))
            up_lora_b_list.append(torch.zeros(intermediate_size, lora_rank, dtype=dtype))

        # Down proj
        down_proj = getattr(expert, down_name)
        down_lora = _get_peft_lora_weights(down_proj)
        if down_lora is not None:
            down_lora_a_list.append(down_lora[0].to(dtype=dtype).cpu().contiguous())
            down_lora_b_list.append(down_lora[1].to(dtype=dtype).cpu().contiguous())
        else:
            down_lora_a_list.append(torch.zeros(lora_rank, intermediate_size, dtype=dtype))
            down_lora_b_list.append(torch.zeros(hidden_size, lora_rank, dtype=dtype))

    # Stack into [num_experts, ...] tensors and ensure bf16 + contiguous
    lora_params = {
        "gate_lora_a": nn.Parameter(torch.stack(gate_lora_a_list, dim=0).to(dtype=dtype).contiguous()),
        "gate_lora_b": nn.Parameter(torch.stack(gate_lora_b_list, dim=0).to(dtype=dtype).contiguous()),
        "up_lora_a": nn.Parameter(torch.stack(up_lora_a_list, dim=0).to(dtype=dtype).contiguous()),
        "up_lora_b": nn.Parameter(torch.stack(up_lora_b_list, dim=0).to(dtype=dtype).contiguous()),
        "down_lora_a": nn.Parameter(torch.stack(down_lora_a_list, dim=0).to(dtype=dtype).contiguous()),
        "down_lora_b": nn.Parameter(torch.stack(down_lora_b_list, dim=0).to(dtype=dtype).contiguous()),
    }

    # Debug: print shapes
    logger.info(f"[PEFT LoRA DEBUG] Extracted shapes:")
    logger.info(f"  gate_lora_a: {lora_params['gate_lora_a'].shape}")
    logger.info(f"  gate_lora_b: {lora_params['gate_lora_b'].shape}")
    logger.info(f"  up_lora_a: {lora_params['up_lora_a'].shape}")
    logger.info(f"  up_lora_b: {lora_params['up_lora_b'].shape}")
    logger.info(f"  down_lora_a: {lora_params['down_lora_a'].shape}")
    logger.info(f"  down_lora_b: {lora_params['down_lora_b'].shape}")
    logger.info(f"  lora_rank={lora_rank}, hidden_size={hidden_size}, intermediate_size={intermediate_size}")

    logger.info(f"Extracted PEFT LoRA from {num_experts} experts (dtype={dtype})")
    return lora_params, lora_rank


def disable_peft_lora_on_experts(experts: nn.ModuleList, moe_config: MOEArchConfig) -> int:
    """
    Disable PEFT LoRA on expert modules after extraction.

    This prevents the original PEFT LoRA parameters from being trained
    (since KT will manage its own copy of the LoRA weights).
    Replaces parameters with empty tensors to free memory and reduce state_dict size.

    Returns:
        Number of LoRA modules disabled.
    """
    gate_name, up_name, down_name = moe_config.weight_names
    disabled_count = 0

    for expert in experts:
        for proj_name in (gate_name, up_name, down_name):
            proj = getattr(expert, proj_name, None)
            if proj is None:
                continue

            # Check for PEFT LoRA structure
            lora_A = getattr(proj, "lora_A", None)
            lora_B = getattr(proj, "lora_B", None)

            if lora_A is None or lora_B is None:
                continue

            # Replace with empty tensors to free memory
            if isinstance(lora_A, nn.ModuleDict):
                for adapter_lora in lora_A.values():
                    if hasattr(adapter_lora, "weight"):
                        orig_dtype = adapter_lora.weight.dtype
                        adapter_lora.weight = nn.Parameter(
                            torch.empty(0, dtype=orig_dtype, device="cpu"),
                            requires_grad=False
                        )
                        disabled_count += 1
            elif hasattr(lora_A, "weight"):
                orig_dtype = lora_A.weight.dtype
                lora_A.weight = nn.Parameter(
                    torch.empty(0, dtype=orig_dtype, device="cpu"),
                    requires_grad=False
                )
                disabled_count += 1

            if isinstance(lora_B, nn.ModuleDict):
                for adapter_lora in lora_B.values():
                    if hasattr(adapter_lora, "weight"):
                        orig_dtype = adapter_lora.weight.dtype
                        adapter_lora.weight = nn.Parameter(
                            torch.empty(0, dtype=orig_dtype, device="cpu"),
                            requires_grad=False
                        )
                        disabled_count += 1
            elif hasattr(lora_B, "weight"):
                orig_dtype = lora_B.weight.dtype
                lora_B.weight = nn.Parameter(
                    torch.empty(0, dtype=orig_dtype, device="cpu"),
                    requires_grad=False
                )
                disabled_count += 1

    if disabled_count > 0:
        logger.info(f"Replaced {disabled_count} PEFT LoRA parameters with empty tensors (KT manages LoRA)")

    return disabled_count


def create_lora_params(
    expert_num: int,
    hidden_size: int,
    intermediate_size: int,
    lora_rank: int,
    lora_alpha: float,
    device: str = "cpu",
    dtype: torch.dtype = torch.bfloat16,
) -> dict[str, nn.Parameter]:
    """Create LoRA parameters for MoE layer (KT native initialization)."""
    gate_lora_a = torch.zeros(expert_num, lora_rank, hidden_size, dtype=dtype, device=device)
    gate_lora_b = torch.zeros(expert_num, intermediate_size, lora_rank, dtype=dtype, device=device)

    up_lora_a = torch.zeros(expert_num, lora_rank, hidden_size, dtype=dtype, device=device)
    up_lora_b = torch.zeros(expert_num, intermediate_size, lora_rank, dtype=dtype, device=device)

    down_lora_a = torch.zeros(expert_num, lora_rank, intermediate_size, dtype=dtype, device=device)
    down_lora_b = torch.zeros(expert_num, hidden_size, lora_rank, dtype=dtype, device=device)

    for tensor in [gate_lora_a, up_lora_a, down_lora_a]:
        nn.init.kaiming_uniform_(tensor, a=math.sqrt(5))

    return {
        "gate_lora_a": nn.Parameter(gate_lora_a),
        "gate_lora_b": nn.Parameter(gate_lora_b),
        "up_lora_a": nn.Parameter(up_lora_a),
        "up_lora_b": nn.Parameter(up_lora_b),
        "down_lora_a": nn.Parameter(down_lora_a),
        "down_lora_b": nn.Parameter(down_lora_b),
    }


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
        lora_params: dict[str, nn.Parameter] | None,
        lora_ref: torch.Tensor,
        hidden_size: int,
        num_experts_per_tok: int,
        layer_idx: int,
        training: bool,
        train_lora: bool,
        precomputed_output: torch.Tensor | None = None,
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
        ctx.lora_params = lora_params
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

                lora_params_for_backward = ctx.lora_params if ctx.train_lora else None
                backward_out = ctx.wrapper.backward(
                    all_go,
                    lora_params=lora_params_for_backward,
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
            lora_params_for_backward = ctx.lora_params if ctx.train_lora else None
            backward_out = ctx.wrapper.backward(
                grad_output_flat,
                lora_params=lora_params_for_backward,
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
        if ctx.train_lora and ctx.lora_params is not None and grad_loras is not None and rank == 0:
            for key, param in ctx.lora_params.items():
                if not param.requires_grad:
                    continue
                grad_tensor = grad_loras.get(key) or grad_loras.get(f"grad_{key}")
                if grad_tensor is None:
                    continue
                grad_cloned = grad_tensor.clone().to(dtype=param.dtype, device=param.device)
                # KT backward runs on rank 0 with gathered data from all ranks,
                # so the gradient is a sum over all ranks' contributions.
                # FSDP-managed params get all-reduce averaged gradients (divided by world_size).
                # Divide by world_size to keep MoE LoRA on the same scale as FSDP params.
                if world_size > 1:
                    grad_cloned /= world_size
                if param.grad is None:
                    param.grad = grad_cloned
                else:
                    param.grad = param.grad + grad_cloned

        return grad_input, None, grad_weights, None, None, None, None, None, None, None, None, None


# =============================================================================
# KTMoE Layer Wrapper
# =============================================================================


class KTMoELayerWrapper(nn.Module):
    """Wrapper for MoE layer using KTMoEWrapper."""

    def __init__(
        self,
        original_moe: nn.Module,
        wrapper: Any,
        lora_params: dict[str, nn.Parameter] | None,
        moe_config: MOEArchConfig,
        hidden_size: int,
        layer_idx: int,
        lora_experts: LoRAExperts | None = None,
    ):
        super().__init__()
        self._is_kt_moe_wrapper = True

        self.wrapper = wrapper
        self.moe_config = moe_config
        self.hidden_size = hidden_size
        self.layer_idx = layer_idx
        self.router_type = moe_config.router_type

        self.lora_experts = lora_experts

        if lora_experts is not None:
            self._dummy_lora_params = lora_params
            self.lora_params = None
        else:
            self._dummy_lora_params = None
            self.lora_params = nn.ParameterDict(lora_params) if lora_params else None

        self.router = getattr(original_moe, moe_config.router_attr)

        if moe_config.has_shared_experts and hasattr(original_moe, "shared_experts"):
            self.shared_experts = original_moe.shared_experts
        else:
            self.shared_experts = None

        self._lora_pointers_dirty = False  # Initially false because init_lora_weights was just called
        # Store original lora_params tensors separately to prevent _apply from touching them
        self._original_lora_tensors = None
        if lora_params is not None:
            self._original_lora_tensors = {
                k: v.data for k, v in lora_params.items()
            }

    def _apply(self, fn, recurse=True):
        # Temporarily detach lora_params from module to prevent super()._apply() from touching them
        # lora_params is a ParameterDict (nn.Module subclass), so it's stored in _modules not _parameters
        saved_lora_params = None
        saved_dummy_lora_params = None

        if self.lora_params is not None:
            # Remove from _modules to prevent recursion into ParameterDict
            saved_lora_params = self.lora_params
            self._modules.pop('lora_params', None)

        if self._dummy_lora_params is not None:
            # _dummy_lora_params is a plain dict, might be in _parameters or just an attribute
            saved_dummy_lora_params = self._dummy_lora_params
            self._parameters.pop('_dummy_lora_params', None)

        result = super()._apply(fn, recurse)

        # Restore lora_params - tensor references are preserved, no pointer update needed
        if saved_lora_params is not None:
            self._modules['lora_params'] = saved_lora_params
            # Pointers are NOT dirty because we kept the same tensor memory

        if saved_dummy_lora_params is not None:
            self._dummy_lora_params = saved_dummy_lora_params

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

        train_lora = self.lora_params is not None and any(p.requires_grad for p in self.lora_params.values())
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
        if train_lora and self.lora_params is not None:
            for p in self.lora_params.values():
                if p.requires_grad:
                    lora_ref = p
                    break

        moe_output = KTMoEFunction.apply(
            hidden_states,
            topk_ids,
            topk_weights,
            self.wrapper,
            dict(self.lora_params) if self.lora_params else None,
            lora_ref,
            self.hidden_size,
            self.moe_config.num_experts_per_tok,
            self.layer_idx,
            save_for_backward,
            train_lora,
            precomputed_output,
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
        if self.router_type == "deepseek_gate":
            router_output = self.router(hidden_states)
            if len(router_output) == 2:
                topk_ids, topk_weights = router_output
            else:
                topk_ids, topk_weights = router_output[0], router_output[1]
            if topk_weights.is_floating_point():
                topk_weights = topk_weights.to(torch.bfloat16)
            return topk_ids, topk_weights

        router_logits = self.router(hidden_states.view(-1, self.hidden_size))
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
        # Skip if wrapper is None (non-rank-0 processes)
        if self.wrapper is None:
            return
        # Skip if wrapper is not properly initialized (weights not loaded or LoRA not initialized)
        if not getattr(self.wrapper, "_weights_loaded", False):
            logger.warning(f"Layer {self.layer_idx}: Skipping update_lora_pointers - weights not loaded")
            return
        if not getattr(self.wrapper, "_lora_initialized", False):
            logger.warning(f"Layer {self.layer_idx}: Skipping update_lora_pointers - LoRA not initialized")
            return

        # if self.lora_params is not None and self.lora_experts is None:
        #     # Check if the wrapper's internal tensors are still valid (same data as our lora_params)
        #     # If they are the same, we don't need to update anything
        #     wrapper_gate_lora_a = getattr(self.wrapper, "gate_lora_a", None)
        #     our_gate_lora_a = self.lora_params["gate_lora_a"].data

        #     if wrapper_gate_lora_a is not None and wrapper_gate_lora_a.data_ptr() == our_gate_lora_a.data_ptr():
        #         # Tensors are the same, no update needed
        #         print(f"[update_lora_pointers] Layer {self.layer_idx}: Tensors unchanged, skipping update", flush=True)
        #         return

        #     # Validate shapes match wrapper's expected dimensions
        #     num_experts = self.wrapper.num_experts
        #     lora_rank = self.wrapper.lora_rank
        #     hidden_size = self.wrapper.hidden_size
        #     intermediate_size = self.wrapper.moe_intermediate_size

        #     print(f"[update_lora_pointers] Layer {self.layer_idx}: wrapper dimensions - "
        #           f"num_experts={num_experts}, lora_rank={lora_rank}, "
        #           f"hidden_size={hidden_size}, intermediate_size={intermediate_size}", flush=True)

        #     expected_shapes = {
        #         "gate_lora_a": (num_experts, lora_rank, hidden_size),
        #         "gate_lora_b": (num_experts, intermediate_size, lora_rank),
        #         "up_lora_a": (num_experts, lora_rank, hidden_size),
        #         "up_lora_b": (num_experts, intermediate_size, lora_rank),
        #         "down_lora_a": (num_experts, lora_rank, intermediate_size),
        #         "down_lora_b": (num_experts, hidden_size, lora_rank),
        #     }

        #     # Ensure all tensors are on CPU, bf16, contiguous, and have correct shapes
        #     for key in expected_shapes:
        #         param = self.lora_params[key]
        #         expected = expected_shapes[key]
        #         actual = tuple(param.data.shape)

        #         print(f"[update_lora_pointers] Layer {self.layer_idx}: {key} - "
        #               f"expected={expected}, actual={actual}, "
        #               f"dtype={param.data.dtype}, device={param.data.device}, "
        #               f"contiguous={param.data.is_contiguous()}", flush=True)

        #         if actual != expected:
        #             raise ValueError(
        #                 f"Layer {self.layer_idx}: LoRA param '{key}' shape mismatch. "
        #                 f"Expected {expected}, got {actual}. "
        #                 f"num_experts={num_experts}, lora_rank={lora_rank}, "
        #                 f"hidden_size={hidden_size}, intermediate_size={intermediate_size}"
        #             )

        #         if param.data.device.type != "cpu":
        #             param.data = param.data.to("cpu")
        #         if param.data.dtype != torch.bfloat16:
        #             param.data = param.data.to(torch.bfloat16)
        #         if not param.data.is_contiguous():
        #             param.data = param.data.contiguous()

        #     # IMPORTANT: Call init_lora_weights instead of update_lora_weights
        #     # because the wrapper's internal tensor references have changed
        #     # and we need to re-register them properly
        #     print(f"[update_lora_pointers] Layer {self.layer_idx}: Calling wrapper.init_lora_weights()", flush=True)
        #     self.wrapper.init_lora_weights(
        #         gate_lora_a=self.lora_params["gate_lora_a"].data,
        #         gate_lora_b=self.lora_params["gate_lora_b"].data,
        #         up_lora_a=self.lora_params["up_lora_a"].data,
        #         up_lora_b=self.lora_params["up_lora_b"].data,
        #         down_lora_a=self.lora_params["down_lora_a"].data,
        #         down_lora_b=self.lora_params["down_lora_b"].data,
        #     )
            # print(f"[update_lora_pointers] Layer {self.layer_idx}: Done", flush=True)

        # Sync latest lora_params data to wrapper's internal tensors so that
        # update_lora_weights() passes the optimizer-updated values to C++ kernel.
        if self.lora_params is not None:
            for key, param in self.lora_params.items():
                wrapper_attr = getattr(self.wrapper, key, None)
                if wrapper_attr is not None:
                    wrapper_attr.copy_(param.data)

        if KT_DEBUG:
            import torch.distributed as dist
            rank = dist.get_rank() if dist.is_initialized() else 0
            if self.lora_params is not None:
                for key, param in self.lora_params.items():
                    d = param.data
                    g = param.grad
                    logger.warning(
                        "\033[35m[KT DEBUG] rank %s update_lora_pointers layer=%s %s "
                        "shape=%s dtype=%s device=%s requires_grad=%s "
                        "data: norm=%.6f mean=%.6f abs_max=%.6f abs_min=%.6f\033[0m",
                        rank, self.layer_idx, key,
                        tuple(d.shape), d.dtype, d.device, param.requires_grad,
                        d.float().norm().item(),
                        d.float().mean().item(),
                        d.float().abs().max().item(),
                        d.float().abs().min().item(),
                    )
                    if g is not None:
                        logger.warning(
                            "\033[33m[KT DEBUG] rank %s update_lora_pointers layer=%s %s "
                            "grad: norm=%.6f mean=%.6f abs_max=%.6f abs_min=%.6f "
                            "dtype=%s device=%s\033[0m",
                            rank, self.layer_idx, key,
                            g.float().norm().item(),
                            g.float().mean().item(),
                            g.float().abs().max().item(),
                            g.float().abs().min().item(),
                            g.dtype, g.device,
                        )
                    else:
                        logger.warning(
                            "\033[31m[KT DEBUG] rank %s update_lora_pointers layer=%s %s "
                            "grad=None\033[0m",
                            rank, self.layer_idx, key,
                        )
            else:
                logger.warning(
                    "\033[35m[KT DEBUG] rank %s update_lora_pointers layer=%s lora_params=None\033[0m",
                    rank, self.layer_idx,
                )

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

    Expects `kt_plugin` to provide KT settings and LoRA settings.
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

    lora_rank = getattr(kt_plugin, "lora_rank", None)
    lora_alpha = getattr(kt_plugin, "lora_alpha", None)
    # lora_rank/lora_alpha can be None if PEFT LoRA is already applied to experts
    # We'll validate this later when we check for PEFT LoRA

    use_lora_experts = getattr(kt_plugin, "kt_use_lora_experts", False)

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

    if use_lora_experts:
        if getattr(kt_plugin, "kt_lora_expert_num", None) is None or getattr(
            kt_plugin, "kt_lora_expert_intermediate_size", None
        ) is None:
            raise KTAMXConfigError(
                "KTPlugin requires kt_lora_expert_num and kt_lora_expert_intermediate_size when kt_use_lora_experts is True."
            )
        logger.info(
            "Using LoRA Experts mode: "
            f"{getattr(kt_plugin, 'kt_lora_expert_num', None)} experts, "
            f"intermediate_size={getattr(kt_plugin, 'kt_lora_expert_intermediate_size', None)}"
        )

    model_container, layers = _get_model_container_and_layers(model, purpose="wrapping")

    for layer_idx, layer in enumerate(layers):
        moe_module = get_moe_module(layer, moe_config)
        if moe_module is None:
            continue

        mode_str = "LoRA Experts" if use_lora_experts else "per-expert LoRA"
        logger.info(
            f"Wrapping MoE layer {layer_idx} with KTMoEWrapper "
            f"(method={kt_method}, tp={threadpool_count}, mode={mode_str})"
        )

        # Only rank 0 loads weights and initializes KT kernel
        gate_proj, up_proj, down_proj = None, None, None
        wrapper = None

        if is_rank_0:
            if use_kt_weight_path:
                int8_weights = load_experts_from_kt_weight_path(
                    kt_weight_path=kt_weight_path,
                    layer_idx=layer_idx,
                    num_experts=moe_config.expert_num,
                    hidden_size=hidden_size,
                    intermediate_size=moe_config.intermediate_size,
                )
                gate_proj = int8_weights.gate_proj
                up_proj = int8_weights.up_proj
                down_proj = int8_weights.down_proj
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

        use_skip_lora = "SkipLoRA" in kt_method

        # Try to extract PEFT LoRA from experts first (only on first layer, share detection result)
        peft_lora_params = None
        peft_lora_rank = None
        if layer_idx == 0 or not hasattr(kt_plugin, "_peft_lora_detected"):
            experts = getattr(moe_module, moe_config.experts_attr, None)
            if experts is not None and len(experts) > 0:
                # Check first expert for PEFT LoRA
                first_expert = experts[0]
                gate_name = moe_config.weight_names[0]
                first_gate_proj = getattr(first_expert, gate_name, None)

                peft_lora_params, peft_lora_rank = extract_peft_lora_from_experts(
                    experts, moe_config, dtype=torch.bfloat16
                )
                kt_plugin._peft_lora_detected = peft_lora_rank is not None
                if peft_lora_rank is not None:
                    kt_plugin._peft_lora_rank = peft_lora_rank
                    logger.info(f"Layer {layer_idx}: Detected PEFT-initialized LoRA (rank={peft_lora_rank})")
            if dist.is_initialized():
                # Sync PEFT LoRA detection across ranks to keep KT behavior consistent.
                detected_tensor = torch.tensor(
                    [1 if getattr(kt_plugin, "_peft_lora_detected", False) else 0], dtype=torch.int32, device="cpu"
                )
                dist.broadcast(detected_tensor, src=0)
                kt_plugin._peft_lora_detected = bool(detected_tensor.item())
                if kt_plugin._peft_lora_detected:
                    rank_tensor = torch.tensor(
                        [getattr(kt_plugin, "_peft_lora_rank", 0)], dtype=torch.int32, device="cpu"
                    )
                    dist.broadcast(rank_tensor, src=0)
                    kt_plugin._peft_lora_rank = int(rank_tensor.item())

        # Use PEFT LoRA if detected, otherwise create KT native LoRA
        use_peft_lora = getattr(kt_plugin, "_peft_lora_detected", False) and not use_skip_lora
        if layer_idx == 0:
            rank = dist.get_rank() if dist.is_initialized() else 0
            logger.info(
                f"Rank {rank}: peft_lora_detected={getattr(kt_plugin, '_peft_lora_detected', False)}, "
                f"use_peft_lora={use_peft_lora}"
            )

        if use_lora_experts:
            lora_experts = LoRAExperts(
                num_experts=getattr(kt_plugin, "kt_lora_expert_num", None),
                hidden_size=hidden_size,
                intermediate_size=getattr(kt_plugin, "kt_lora_expert_intermediate_size", None),
                device="cuda",
                dtype=torch.bfloat16,
            )

            if use_skip_lora:
                dummy_lora_rank = 1
                lora_params = create_lora_params(
                    expert_num=moe_config.expert_num,
                    hidden_size=hidden_size,
                    intermediate_size=moe_config.intermediate_size,
                    lora_rank=dummy_lora_rank,
                    lora_alpha=1.0,
                )
                for param in lora_params.values():
                    param.requires_grad = False
                wrapper_lora_rank = dummy_lora_rank
                wrapper_lora_alpha = 1.0
                logger.info(
                    f"  Layer {layer_idx}: LoRA Experts + SkipLoRA mode (per-expert LoRA frozen)"
                )
            else:
                # LoRA Experts mode with per-expert LoRA - requires lora_rank and lora_alpha
                if lora_rank is None or lora_alpha is None:
                    raise KTAMXConfigError(
                        "KTPlugin requires lora_rank and lora_alpha for LoRA Experts + LoRA mode."
                    )
                lora_params = create_lora_params(
                    expert_num=moe_config.expert_num,
                    hidden_size=hidden_size,
                    intermediate_size=moe_config.intermediate_size,
                    lora_rank=lora_rank,
                    lora_alpha=lora_alpha,
                )
                wrapper_lora_rank = lora_rank
                wrapper_lora_alpha = lora_alpha
                logger.info(f"  Layer {layer_idx}: LoRA Experts + LoRA mode (both trained)")
        else:
            lora_experts = None

            if use_skip_lora:
                dummy_lora_rank = 1
                lora_params = create_lora_params(
                    expert_num=moe_config.expert_num,
                    hidden_size=hidden_size,
                    intermediate_size=moe_config.intermediate_size,
                    lora_rank=dummy_lora_rank,
                    lora_alpha=1.0,
                )
                for param in lora_params.values():
                    param.requires_grad = False
                wrapper_lora_rank = dummy_lora_rank
                wrapper_lora_alpha = 1.0
                logger.info(f"  Layer {layer_idx}: SkipLoRA mode (MoE frozen)")
            elif use_peft_lora:
                # Extract PEFT LoRA for this layer
                experts = getattr(moe_module, moe_config.experts_attr, None)
                lora_params, extracted_rank = extract_peft_lora_from_experts(
                    experts, moe_config, dtype=torch.bfloat16
                )
                if lora_params is None:
                    # Fallback if extraction fails for this layer
                    lora_params = create_lora_params(
                        expert_num=moe_config.expert_num,
                        hidden_size=hidden_size,
                        intermediate_size=moe_config.intermediate_size,
                        lora_rank=lora_rank,
                        lora_alpha=lora_alpha,
                    )
                    wrapper_lora_rank = lora_rank
                    wrapper_lora_alpha = lora_alpha
                else:
                    wrapper_lora_rank = extracted_rank
                    # Use lora_alpha from config, or default to lora_rank (common PEFT default)
                    wrapper_lora_alpha = lora_alpha if lora_alpha is not None else float(extracted_rank)
                    # Disable original PEFT LoRA params to prevent double training
                    disabled_count = disable_peft_lora_on_experts(experts, moe_config)
                    rank = dist.get_rank() if dist.is_initialized() else 0
                    logger.info(
                        f"\033[35mRank {rank}: disabled PEFT LoRA params={disabled_count}\033[0m"
                    )

                    # Debug: print extracted lora_params shapes
                    print(f"[PEFT LoRA] Layer {layer_idx}: Extracted lora_params shapes:", flush=True)
                    for k, v in lora_params.items():
                        print(f"  {k}: {v.shape}, dtype={v.dtype}", flush=True)
                    print(f"  moe_config: expert_num={moe_config.expert_num}, "
                          f"intermediate_size={moe_config.intermediate_size}", flush=True)
                    print(f"  model hidden_size={hidden_size}", flush=True)

                logger.info(f"  Layer {layer_idx}: PEFT LoRA mode (rank={wrapper_lora_rank}, alpha={wrapper_lora_alpha})")
            else:
                # KT native LoRA mode - requires lora_rank and lora_alpha
                if lora_rank is None or lora_alpha is None:
                    raise KTAMXConfigError(
                        "KTPlugin requires lora_rank and lora_alpha when PEFT LoRA is not applied to experts. "
                        "Either configure lora_rank/lora_alpha in kt_config, or apply PEFT LoRA to expert modules first."
                    )
                lora_params = create_lora_params(
                    expert_num=moe_config.expert_num,
                    hidden_size=hidden_size,
                    intermediate_size=moe_config.intermediate_size,
                    lora_rank=lora_rank,
                    lora_alpha=lora_alpha,
                )
                wrapper_lora_rank = lora_rank
                wrapper_lora_alpha = lora_alpha
                logger.info(f"  Layer {layer_idx}: KT native LoRA mode (rank={lora_rank})")

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
                weight_path="",
                chunked_prefill_size=chunked_prefill_size,
                method=kt_method,
                mode="sft",
                lora_rank=wrapper_lora_rank,
                lora_alpha=wrapper_lora_alpha,
                max_cache_depth=8,
            )

            physical_to_logical_map = torch.arange(moe_config.expert_num, dtype=torch.int64, device="cpu")

            wrapper.load_weights_from_tensors(
                gate_proj=gate_proj,
                up_proj=up_proj,
                down_proj=down_proj,
                physical_to_logical_map_cpu=physical_to_logical_map,
            )

            if lora_params is not None:
                print(f"[init_lora_weights] Layer {layer_idx}: Calling init_lora_weights", flush=True)
                print(f"  wrapper dimensions: num_experts={wrapper.num_experts}, "
                      f"lora_rank={wrapper.lora_rank}, hidden_size={wrapper.hidden_size}, "
                      f"intermediate_size={wrapper.moe_intermediate_size}", flush=True)
                for k, v in lora_params.items():
                    print(f"  {k}: shape={v.shape}, dtype={v.dtype}, "
                          f"device={v.device}, contiguous={v.data.is_contiguous()}", flush=True)
                wrapper.init_lora_weights(
                    gate_lora_a=lora_params["gate_lora_a"].data,
                    gate_lora_b=lora_params["gate_lora_b"].data,
                    up_lora_a=lora_params["up_lora_a"].data,
                    up_lora_b=lora_params["up_lora_b"].data,
                    down_lora_a=lora_params["down_lora_a"].data,
                    down_lora_b=lora_params["down_lora_b"].data,
                )
                print(f"[init_lora_weights] Layer {layer_idx}: Done", flush=True)

        layer_wrapper = KTMoELayerWrapper(
            original_moe=moe_module,
            wrapper=wrapper,
            lora_params=lora_params,
            moe_config=moe_config,
            hidden_size=hidden_size,
            layer_idx=layer_idx,
            lora_experts=lora_experts,
        )

        setattr(layer, moe_config.moe_layer_attr, layer_wrapper)
        if is_rank_0:
            layer_wrapper._base_weights = (gate_proj, up_proj, down_proj)

        wrappers.append(layer_wrapper)
        moe_layer_count += 1

        # _clear_original_expert_weights(moe_module, moe_config)

    mode_str = "LoRA Experts" if use_lora_experts else "per-expert LoRA"
    logger.info(f"Wrapped {moe_layer_count} MoE layers with KTMoEWrapper ({mode_str} mode)")
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

    if getattr(kt_plugin, "kt_skip_expert_loading", None) is None and getattr(kt_plugin, "kt_weight_path", None) is None:
        checkpoint_files, sharded_metadata = _resolve_checkpoint_files(
            model_name_or_path=model_name_or_path,
            cache_dir=cache_dir,
            revision=revision,
            token=token,
            trust_remote_code=trust_remote_code,
        )
        if checkpoint_files and all(f.endswith(".safetensors") for f in checkpoint_files):
            kt_plugin.kt_skip_expert_loading = True
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

    moe_lora_params = {}
    for wrapper in wrappers:
        if getattr(wrapper, "lora_params", None) is not None:
            moe_lora_params[wrapper.layer_idx] = dict(wrapper.lora_params)
    model._kt_moe_lora_params = moe_lora_params
    model._kt_use_lora_experts = bool(getattr(kt_plugin, "kt_use_lora_experts", False))

    logger.info("Model loaded with KTMoEWrapper backend successfully")
    return model


def get_kt_lora_params(model: nn.Module) -> list[nn.Parameter]:
    """Get all MoE LoRA parameters from KT model."""
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
            if wrapper.lora_params is not None:
                params.extend(wrapper.lora_params.values())
            if wrapper.lora_experts is not None:
                params.extend(wrapper.lora_experts.parameters())

    return params


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


def load_moe_lora_from_adapter(model: nn.Module, adapter_path: str):
    """
    Load MoE LoRA weights from PEFT adapter into KT wrappers.
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
                    logger.info(f"Found _kt_wrappers on unwrapped model ({attr})")
                    break
    if not wrappers:
        logger.warning("No KT wrappers found, skipping MoE LoRA loading")
        return

    wrapper_map = {w.layer_idx: w for w in wrappers if w.lora_params is not None}
    if not wrapper_map:
        logger.warning("No KT wrappers with per-expert LoRA found, skipping MoE LoRA loading")
        return

    adapter_file = os.path.join(adapter_path, "adapter_model.safetensors")
    if not os.path.exists(adapter_file):
        adapter_file = os.path.join(adapter_path, "adapter_model.bin")
        if not os.path.exists(adapter_file):
            logger.warning(f"No adapter file found at {adapter_path}")
            return

    logger.info(f"Loading MoE LoRA from {adapter_file}")

    moe_pattern_kt = re.compile(
        r"base_model\.model\.model\.layers\.(\d+)\.mlp\.original_moe\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.lora_(A|B)\.weight"
    )
    moe_pattern_peft = re.compile(
        r"base_model\.model\.model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.lora_(A|B)\.weight"
    )

    layer_weights = {}
    matched_kt_format = 0
    matched_peft_format = 0

    with safe_open(adapter_file, framework="pt") as f:
        for key in f.keys():
            match = moe_pattern_kt.match(key)
            if match:
                matched_kt_format += 1
            else:
                match = moe_pattern_peft.match(key)
                if match:
                    matched_peft_format += 1
            if match:
                layer_idx = int(match.group(1))
                expert_idx = int(match.group(2))
                proj_name = match.group(3)
                ab = match.group(4)

                layer_weights.setdefault(layer_idx, {}).setdefault(expert_idx, {}).setdefault(proj_name, {})
                tensor = f.get_tensor(key)
                layer_weights[layer_idx][expert_idx][proj_name][ab] = tensor

    loaded_count = 0
    for layer_idx, experts_dict in layer_weights.items():
        if layer_idx not in wrapper_map:
            logger.warning(f"No KT wrapper for layer {layer_idx}, skipping")
            continue

        wrapper = wrapper_map[layer_idx]
        num_experts = wrapper.moe_config.expert_num
        lora_rank = wrapper.lora_params["gate_lora_a"].shape[1]
        hidden_size = wrapper.hidden_size
        intermediate_size = wrapper.moe_config.intermediate_size

        gate_lora_a = torch.zeros(num_experts, lora_rank, hidden_size, dtype=torch.bfloat16)
        gate_lora_b = torch.zeros(num_experts, intermediate_size, lora_rank, dtype=torch.bfloat16)
        up_lora_a = torch.zeros(num_experts, lora_rank, hidden_size, dtype=torch.bfloat16)
        up_lora_b = torch.zeros(num_experts, intermediate_size, lora_rank, dtype=torch.bfloat16)
        down_lora_a = torch.zeros(num_experts, lora_rank, intermediate_size, dtype=torch.bfloat16)
        down_lora_b = torch.zeros(num_experts, hidden_size, lora_rank, dtype=torch.bfloat16)

        for expert_idx, proj_dict in experts_dict.items():
            if expert_idx >= num_experts:
                continue
            for proj_name, ab_dict in proj_dict.items():
                if "A" in ab_dict:
                    a_tensor = ab_dict["A"].to(torch.bfloat16)
                    if proj_name == "gate_proj":
                        gate_lora_a[expert_idx] = a_tensor
                    elif proj_name == "up_proj":
                        up_lora_a[expert_idx] = a_tensor
                    elif proj_name == "down_proj":
                        down_lora_a[expert_idx] = a_tensor
                if "B" in ab_dict:
                    b_tensor = ab_dict["B"].to(torch.bfloat16)
                    if proj_name == "gate_proj":
                        gate_lora_b[expert_idx] = b_tensor
                    elif proj_name == "up_proj":
                        up_lora_b[expert_idx] = b_tensor
                    elif proj_name == "down_proj":
                        down_lora_b[expert_idx] = b_tensor

        device = wrapper.lora_params["gate_lora_a"].device
        wrapper.lora_params["gate_lora_a"].data.copy_(gate_lora_a.to(device))
        wrapper.lora_params["gate_lora_b"].data.copy_(gate_lora_b.to(device))
        wrapper.lora_params["up_lora_a"].data.copy_(up_lora_a.to(device))
        wrapper.lora_params["up_lora_b"].data.copy_(up_lora_b.to(device))
        wrapper.lora_params["down_lora_a"].data.copy_(down_lora_a.to(device))
        wrapper.lora_params["down_lora_b"].data.copy_(down_lora_b.to(device))

        loaded_count += 1
        logger.debug(f"Loaded MoE LoRA for layer {layer_idx} ({len(experts_dict)} experts)")

    update_kt_lora_pointers(model)

    logger.info(
        f"Loaded MoE LoRA into {loaded_count} KT wrappers from {adapter_path} "
        f"(matched {matched_kt_format} KT-format keys, {matched_peft_format} PEFT-format keys)"
    )


def save_moe_lora_to_adapter(model: nn.Module, output_dir: str) -> None:
    """
    Save MoE LoRA weights to adapter file by merging with existing Attention LoRA.
    """
    import os
    from safetensors import safe_open
    from safetensors.torch import save_file

    wrappers = getattr(model, "_kt_wrappers", [])
    if not wrappers:
        logger.warning("No KT wrappers found, skipping MoE LoRA saving")
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

    moe_lora_count = 0
    for wrapper in wrappers:
        if wrapper.lora_params is None:
            continue

        layer_idx = wrapper.layer_idx
        num_experts = wrapper.moe_config.expert_num

        gate_lora_a = wrapper.lora_params["gate_lora_a"].data.cpu()
        gate_lora_b = wrapper.lora_params["gate_lora_b"].data.cpu()
        up_lora_a = wrapper.lora_params["up_lora_a"].data.cpu()
        up_lora_b = wrapper.lora_params["up_lora_b"].data.cpu()
        down_lora_a = wrapper.lora_params["down_lora_a"].data.cpu()
        down_lora_b = wrapper.lora_params["down_lora_b"].data.cpu()

        for expert_idx in range(num_experts):
            base_key = f"base_model.model.model.layers.{layer_idx}.mlp.original_moe.experts.{expert_idx}"
            state_dict[f"{base_key}.gate_proj.lora_A.weight"] = gate_lora_a[expert_idx].clone()
            state_dict[f"{base_key}.gate_proj.lora_B.weight"] = gate_lora_b[expert_idx].clone()
            state_dict[f"{base_key}.up_proj.lora_A.weight"] = up_lora_a[expert_idx].clone()
            state_dict[f"{base_key}.up_proj.lora_B.weight"] = up_lora_b[expert_idx].clone()
            state_dict[f"{base_key}.down_proj.lora_A.weight"] = down_lora_a[expert_idx].clone()
            state_dict[f"{base_key}.down_proj.lora_B.weight"] = down_lora_b[expert_idx].clone()
            moe_lora_count += 6

        logger.debug(f"Added MoE LoRA for layer {layer_idx} ({num_experts} experts)")

    output_file = os.path.join(output_dir, "adapter_model.safetensors")
    save_file(state_dict, output_file, metadata={"format": "pt"})

    logger.info(
        f"Saved MoE LoRA to {output_file}: "
        f"{len(wrappers)} layers, {moe_lora_count} MoE LoRA tensors added, "
        f"{len(state_dict)} total tensors"
    )


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
    """
    wrappers = getattr(model, "_kt_wrappers", [])
    if not wrappers:
        logger.warning("No KT wrappers found, skipping KT MoE saving")
        return

    has_lora_experts = any(w.lora_experts is not None for w in wrappers)
    has_lora_params = any(w.lora_params is not None for w in wrappers)

    if has_lora_experts:
        save_lora_experts_to_adapter(model, output_dir)
    elif has_lora_params:
        save_moe_lora_to_adapter(model, output_dir)
    else:
        logger.warning("No trainable KT MoE parameters found, skipping saving")


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
    has_lora_params = any(w.lora_params is not None for w in wrappers)

    if has_lora_experts:
        load_lora_experts_from_adapter(model, adapter_path)
    elif has_lora_params:
        load_moe_lora_from_adapter(model, adapter_path)
    else:
        logger.warning("No trainable KT MoE parameters found, skipping loading")

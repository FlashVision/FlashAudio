"""LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning of audio models.

Injects trainable low-rank matrices into existing linear layers
while freezing the original weights, reducing trainable parameters.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Set

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """Linear layer with LoRA adaptation.

    Replaces W*x with (W + alpha/r * B*A)*x where A and B are low-rank matrices.

    Args:
        original: The original nn.Linear layer.
        rank: LoRA rank (r).
        alpha: LoRA scaling factor.
        dropout: Dropout probability for LoRA layers.
    """

    def __init__(
        self,
        original: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.original = original
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_features = original.in_features
        out_features = original.out_features

        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.original.weight.requires_grad_(False)
        if self.original.bias is not None:
            self.original.bias.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_output = self.original(x)
        lora_output = F.linear(self.lora_dropout(x), self.lora_A)
        lora_output = F.linear(lora_output, self.lora_B)
        return base_output + lora_output * self.scaling

    def merge(self) -> nn.Linear:
        """Merge LoRA weights into the original linear layer."""
        merged = nn.Linear(
            self.original.in_features,
            self.original.out_features,
            bias=self.original.bias is not None,
        )
        merged.weight.data = self.original.weight.data + (
            self.lora_B @ self.lora_A * self.scaling
        )
        if self.original.bias is not None:
            merged.bias.data = self.original.bias.data
        return merged


def apply_lora(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    target_modules: Optional[Set[str]] = None,
) -> nn.Module:
    """Apply LoRA to a model's linear layers.

    Args:
        model: The model to modify.
        rank: LoRA rank.
        alpha: LoRA scaling factor.
        dropout: Dropout rate.
        target_modules: Set of module name patterns to target.
            Defaults to common attention projections.

    Returns:
        The modified model with LoRA layers.
    """
    if target_modules is None:
        target_modules = {"q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2", "query", "value"}

    lora_count = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            name_parts = name.split(".")
            if any(target in name_parts[-1] for target in target_modules):
                parent_name = ".".join(name_parts[:-1])
                child_name = name_parts[-1]

                parent = model
                if parent_name:
                    for part in parent_name.split("."):
                        parent = getattr(parent, part)

                lora_layer = LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout)
                setattr(parent, child_name, lora_layer)
                lora_count += 1

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  LoRA applied to {lora_count} layers")
    print(f"  Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    return model


def merge_lora_weights(model: nn.Module) -> nn.Module:
    """Merge all LoRA weights into the base model.

    After merging, the model has no LoRA overhead and can be
    exported/deployed normally.

    Args:
        model: Model with LoRA layers.

    Returns:
        Model with merged weights.
    """
    merged_count = 0
    for name, module in list(model.named_modules()):
        if isinstance(module, LoRALinear):
            name_parts = name.split(".")
            parent_name = ".".join(name_parts[:-1])
            child_name = name_parts[-1]

            parent = model
            if parent_name:
                for part in parent_name.split("."):
                    parent = getattr(parent, part)

            merged_linear = module.merge()
            setattr(parent, child_name, merged_linear)
            merged_count += 1

    print(f"  Merged {merged_count} LoRA layers into base model")
    return model


def get_lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Extract only LoRA parameters from the model.

    Args:
        model: Model with LoRA layers.

    Returns:
        State dict containing only LoRA weights.
    """
    lora_state = {}
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            lora_state[name] = param.data.clone()
    return lora_state

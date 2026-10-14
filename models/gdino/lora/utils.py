"""The two helpers this repository uses from the fork's ``lora_utils``.

Copied rather than imported: the fork's file also carries a detectron2
checkpointer and a task-memory dependency that nothing here needs, and it lives
in the same untracked layer as the modules below -- see this package's
``__init__`` for why that matters.

Both functions are verbatim from
``groundingdino/util/lora_utils.py`` (sha256 dac8d1e1640cef28); only the two
imports at the top of this file differ, because the classes they name are now
siblings.
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn

from .layer import LoRALayer
from .multi_head_attention import MultiHeadAttention


def mark_only_lora_as_trainable(model: nn.Module, bias: str = 'none') -> None:
    for n, p in model.named_parameters():
        if 'lora_' not in n:
            p.requires_grad = False
    if bias == 'none':
        return
    elif bias == 'all':
        for n, p in model.named_parameters():
            if 'bias' in n:
                p.requires_grad = True
    elif bias == 'lora_only':
        for m in model.modules():
            if isinstance(m, LoRALayer) and \
                hasattr(m, 'bias') and \
                m.bias is not None:
                    m.bias.requires_grad = True
    else:
        raise NotImplementedError


@torch.no_grad()
def apply_custom_attention(module, attention_module_names):
    assert attention_module_names is not None

    for attn_module_name in attention_module_names:
        original_self_attn = copy.deepcopy(getattr(module, attn_module_name))

        setattr(module, attn_module_name, MultiHeadAttention(original_self_attn.embed_dim, original_self_attn.num_heads))
        # clone weights from original self_attn
        qkv_weight = original_self_attn.in_proj_weight.clone()
        qkv_bias = original_self_attn.in_proj_bias.clone()
        o_w = original_self_attn.out_proj.weight.clone()
        o_b = original_self_attn.out_proj.bias.clone()
        # chunk qkv_weight and qkv_bias
        q_w, k_w, v_w = torch.chunk(qkv_weight, 3, dim=0)
        q_b, k_b, v_b = torch.chunk(qkv_bias, 3, dim=0)
        # assign weights and biases to new self_attn
        attn_module = getattr(module, attn_module_name)
        attn_module.linear_q.weight = torch.nn.Parameter(q_w)
        attn_module.linear_k.weight = torch.nn.Parameter(k_w)
        attn_module.linear_v.weight = torch.nn.Parameter(v_w)
        attn_module.linear_o.weight = torch.nn.Parameter(o_w)
        attn_module.linear_q.bias = torch.nn.Parameter(q_b)
        attn_module.linear_k.bias = torch.nn.Parameter(k_b)
        attn_module.linear_v.bias = torch.nn.Parameter(v_b)
        attn_module.linear_o.bias = torch.nn.Parameter(o_b)

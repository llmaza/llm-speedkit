from __future__ import annotations

from typing import Dict, Tuple

def infer_kv_params_from_hf_config(model_id: str) -> Dict[str, int]:
    """
    Infer (num_layers, num_kv_heads, head_dim) from HF AutoConfig.
    Works for most LLaMA-like configs; has safe fallbacks.
    """
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(model_id)

    # layers
    num_layers = int(getattr(cfg, "num_hidden_layers", getattr(cfg, "n_layer", 0)))
    if num_layers <= 0:
        raise ValueError("Could not infer num_layers from model config (num_hidden_layers/n_layer missing).")

    # heads
    num_attn_heads = int(getattr(cfg, "num_attention_heads", getattr(cfg, "n_head", 0)))
    if num_attn_heads <= 0:
        raise ValueError("Could not infer num_attention_heads from model config (num_attention_heads/n_head missing).")

    num_kv_heads = getattr(cfg, "num_key_value_heads", None)
    if num_kv_heads is None:
        # common fallback: MHA => kv_heads == attn_heads
        num_kv_heads = num_attn_heads
    num_kv_heads = int(num_kv_heads)

    # hidden size / head dim
    hidden_size = int(getattr(cfg, "hidden_size", getattr(cfg, "n_embd", 0)))
    if hidden_size <= 0:
        raise ValueError("Could not infer hidden_size from model config (hidden_size/n_embd missing).")

    head_dim = hidden_size // num_attn_heads
    if head_dim <= 0:
        raise ValueError("Invalid inferred head_dim (hidden_size // num_attention_heads).")

    return {
        "num_layers": num_layers,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
    }

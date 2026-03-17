from typing import Any, Dict, List, Tuple
import torch
import torch.nn.functional as F
from tabpfn.architectures.base.attention.full_attention import MultiHeadAttention
import einops
import inspect


# # 1. Define the rotation logic
# def _rotate_half(x):
#     x1, x2 = x.chunk(2, dim=-1)
#     return torch.cat((-x2, x1), dim=-1)


# def _apply_rope(x, d_k):
#     # x shape: [batch, seq_len, nhead, d_k]
#     n = x.shape[1]
#     device = x.device

#     # Standard RoPE frequencies
#     inv_freq = 1.0 / (10000 ** (torch.arange(0, d_k, 2).float().to(device) / d_k))
#     t = torch.arange(n, device=device).type_as(inv_freq)
#     freqs = torch.outer(t, inv_freq)
#     emb = torch.cat((freqs, freqs), dim=-1)

#     # [1, seq_len, 1, d_k] for broadcasting
#     cos, sin = emb.cos()[None, :, None, :], emb.sin()[None, :, None, :]
#     return (x * cos) + (_rotate_half(x) * sin)


# # 2. Define the new logic
# def rope_compute_heads_wrapper(q, k, v, kv, qkv, dropout_p=None, softmax_scale=None):
#     # Step A: Let the existing static logic unbind the tensors
#     # We use the class name to call the original static method
#     if qkv is not None:
#         q, k, v = qkv.unbind(dim=-3)
#     elif kv is not None:
#         k, v = kv.unbind(dim=-3)

#     # Step B: Apply RoPE to the unbundled Q and K
#     d_k = q.shape[-1]
#     q = _apply_rope(q, d_k)
#     k = _apply_rope(k, d_k)

#     # Step C: Call original static method with our rotated tensors
#     # We set qkv and kv to None so the original method uses our provided q, k, v
#     return MultiHeadAttention.compute_attention_heads(
#         q=q, k=k, v=v, kv=None, qkv=None, dropout_p=dropout_p, softmax_scale=softmax_scale
#     )

def _rotate_half(x):
    """Sépare le tenseur en deux et applique la rotation de base."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _compute_rope_embeddings(f_per_ch, d_k, device):
    """Calcule les composantes cosinus et sinus pour la rotation."""
    inv_freq = 1.0 / (10000 ** (torch.arange(0, d_k, 2, device=device).float() / d_k))
    t_pos = torch.arange(f_per_ch, device=device).float()
    freqs = torch.outer(t_pos, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)

    # Reshape pour broadcasting : [1 (seq), 1 (ch), f_per_ch, 1 (heads), d_k]
    cos = emb.cos().view(1, 1, -1, 1, d_k)
    sin = emb.sin().view(1, 1, -1, 1, d_k)
    return cos, sin


# --- 2. Logique de transformation des tenseurs ---


def _apply_channel_rope(q_feat, k_feat, num_channels):
    """Applique le RoPE spécifiquement sur la structure multi-channel."""
    s, f, h, d_k = q_feat.shape
    assert f % num_channels == 0, f"Feature length {f} must be divisible by num_channels {num_channels}"
    f_per_ch = f // num_channels

    # 1. Passage en mode multi-channel
    q_feat = q_feat.view(s, num_channels, f_per_ch, h, d_k)
    k_feat = k_feat.view(s, num_channels, f_per_ch, h, d_k)

    # 2. Application de la rotation
    cos, sin = _compute_rope_embeddings(f_per_ch, d_k, q_feat.device)
    q_feat = (q_feat * cos) + (_rotate_half(q_feat) * sin)
    k_feat = (k_feat * cos) + (_rotate_half(k_feat) * sin)

    # 3. Retour à la forme plate
    return q_feat.view(s, f, h, d_k), k_feat.view(s, f, h, d_k)


# --- 3. Wrapper principal (le Patch) ---


def rope_compute_heads_wrapper(
    q,
    k,
    v,
    kv,
    qkv,
    dropout_p=None,
    softmax_scale=None,
    time_points=50,
    num_channels=1,
    original_func=None,
    **kwargs
):
    """
    Wrapper patché pour MultiHeadAttention.compute_attention_heads.
    Filtre les appels pour n'appliquer le RoPE que sur les features.
    """
    # A. Détection du contexte (évite de toucher aux items)
    frame = inspect.currentframe().f_back
    caller_self = frame.f_locals.get("self", None)
    if not getattr(caller_self, "is_feature_attn", False):
        return original_func(q, k, v, kv, qkv, dropout_p, softmax_scale, **kwargs)


    # B. Extraction des tenseurs (Unpack)
    if qkv is not None:
        q, k, v = qkv.unbind(dim=-3)
    elif kv is not None:
        k, v = kv.unbind(dim=-3)

    # C. Découpage Features / Label
    # q shape: [Seq, Features, Heads, D_k]
    q_feat, q_label = q[:, :time_points], q[:, time_points:]
    k_feat, k_label = k[:, :time_points], k[:, time_points:]

    # D. Application RoPE
    q_feat, k_feat = _apply_channel_rope(q_feat, k_feat, num_channels)

    # E. Re-assemblage final
    q_final = torch.cat([q_feat, q_label], dim=1)
    k_final = torch.cat([k_feat, k_label], dim=1)

    # F. Appel à la fonction originale (évite la récursion)
    return original_func(
        q=q_final, k=k_final, v=v, kv=None, qkv=None, dropout_p=dropout_p, softmax_scale=softmax_scale, **kwargs
    )

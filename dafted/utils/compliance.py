from typing import Tuple, Union, MutableMapping, Optional, Callable, Dict, Any

import torch
from omegaconf import DictConfig
from torch import nn

import dafted.models.transformer
from dafted.models.baselines import (
    AvgConcatMLP,
    ConcatMLP,
    ConcatMLPDecoupling,
    ConcatMLPDecoupling2FTs,
    FlatConcatMLP,
    IdentityEncoder,
    IRENEModel,
    MMCLEncoder,
)
# from dafted.models.tabpfn_encoder import TabPFNEncoder
import tabpfn


def check_model_encoder(encoder: nn.Module, hparams: MutableMapping[str, Any]) -> Tuple[int, bool]:
    if isinstance(encoder, nn.TransformerEncoder):  # Native PyTorch `TransformerEncoder`
        nhead = encoder.layers[0].self_attn.num_heads
    elif isinstance(encoder, dafted.models.transformer.FT_Transformer):  # dafted submodule `Transformer`
        nhead = hparams.model.encoder.attention_n_heads
        separate_modality = bool(hparams.model.encoder.n_cross_blocks or hparams.model.encoder.n_bidirectional_blocks)
        print(f"Separate modality is {separate_modality}")
    elif isinstance(encoder, dafted.models.baselines.ConcatMLPDecoupling):  # dafted submodule `MLP`
        nhead = 1
        separate_modality = True
        assert not hparams.cls_token, "ConcatMLP does not support cls_token"
    elif isinstance(encoder, dafted.models.baselines.ConcatMLPDecoupling2FTs):  # dafted submodule `MLP`
        nhead = 1
        separate_modality = True
        assert not hparams.cls_token, "ConcatMLP does not support cls_token"
    elif isinstance(encoder, dafted.models.baselines.FlatConcatMLP):  # dafted submodule `FlatConcatMLP`
        nhead = 1
        separate_modality = True
    elif isinstance(encoder, dafted.models.baselines.AvgConcatMLP):  # dafted submodule `AvgConcatMLP`
        nhead = 1
        separate_modality = True
    elif isinstance(encoder, dafted.models.baselines.ConcatMLP):  # dafted submodule `ConcatMLP`
        nhead = 1
        separate_modality = True
    elif isinstance(encoder, dafted.models.baselines.IdentityEncoder):  # dafted submodule `IdentityEncoder`
        separate_modality = False
    elif isinstance(encoder, dafted.models.baselines.MMCLEncoder):  # dafted submodule `MMCLEncoder (BoB)`
        nhead = 1
        separate_modality = True
        assert not hparams.task.contrastive_loss, "MMCLEncoder does not support contrastive loss"
    elif isinstance(encoder, dafted.models.baselines.IRENEModel):  # dafted submodule `IRENEModel`
        # nhead = hparams.model.encoder.attention_n_heads
        separate_modality = True
        assert not (
            hparams.task.contrastive_loss or hparams.task.inter_sample_loss
        ), "IRENEModel does not support contrastive or inter-sample loss"
    elif isinstance(encoder, dafted.models.transformer.FT_Interleaved):  # dafted submodule `FT_Interleaved`
        nhead = hparams.model.encoder.attention_n_heads
        separate_modality = True
    elif isinstance(encoder, dafted.models.alternatives_transformer.FT_Transformer_2UniFTs):  # dafted submodule `Transformer`
        nhead = hparams.model.encoder.attention_n_heads
        separate_modality = False
    elif isinstance(encoder, dafted.models.transformer.FT_Alignment):  # dafted submodule `FT_Alignment`
        nhead = hparams.model.encoder.attention_n_heads
        separate_modality = False
    elif isinstance(encoder, dafted.models.alternatives_transformer.FT_Alignment_2UniFTs):  # dafted submodule `FT_Alignment`
        nhead = hparams.model.encoder.attention_n_heads
        separate_modality = False
    elif isinstance(encoder, dafted.models.transformer.FT_Interleaved_Alignment):  # dafted submodule `FT_Alignment`
        nhead = hparams.model.encoder.attention_n_heads
        separate_modality = False
    elif isinstance(encoder, dafted.models.transformer.FT_DiffInterleaved_Alignment):  # dafted submodule `FT_DiffAlignment`
        nhead = hparams.model.encoder.attention_n_heads
        separate_modality = False
    elif isinstance(encoder, dafted.models.tabpfn_encoder.TabPFNEncoder):  # dafted submodule `FT_Alignment`
        nhead = 1
        separate_modality = True
    elif isinstance(
        encoder, dafted.models.alternatives_transformer.FT_Alignment_2UniFTs_BiDirectional
    ):  # dafted submodule `FT_Alignment_BiDirectional`
        nhead = hparams.model.encoder.attention_n_heads
        separate_modality = False
    elif isinstance(
        encoder, dafted.models.alternatives_transformer.FT_Alignment_2UniFTs_CrossAtt
    ):  # dafted submodule `FT_Alignment`
        nhead = hparams.model.encoder.attention_n_heads
        separate_modality = False
    elif isinstance(encoder, dafted.models.transformer.FT_Interleaved_2UniFTs):  # dafted submodule `FT_Interleaved`
        nhead = hparams.model.encoder.attention_n_heads
        separate_modality = False
    elif isinstance(
        encoder, dafted.models.transformer.DAFTED_encoder
    ):  # dafted submodule `FT_Interleaved`
        nhead = hparams.model.encoder.attention_n_heads
        separate_modality = False
    elif isinstance(
        encoder, dafted.models.alternatives_transformer.FT_Interleaved_2UniFTs_nodecoupling
    ):  # dafted submodule `FT_Interleaved`
        nhead = hparams.model.encoder.attention_n_heads
        separate_modality = False
    # elif isinstance(
    #     encoder, dafted.models.transformer.FT_Interleaved_Inverted
    # ):  # dafted submodule `FT_Interleaved`
    #     nhead = hparams.model.encoder.attention_n_heads
    #     separate_modality = False
    elif isinstance(
        encoder, tabpfn.TabPFNClassifier
    ):
        nhead = 1
        separate_modality = False
    else:
        raise NotImplementedError(
            "To instantiate the cardiac multimodal representation task, it is necessary to determine the number of "
            f"attention heads. However, this is not implemented for the requested encoder configuration: "
            f"'{encoder.__class__.__name__}'. Either change the configuration, or implement the introspection "
            f"of the number of attention heads for your configuration above this warning."
        )
    return nhead, separate_modality

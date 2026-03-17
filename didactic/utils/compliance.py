from typing import Tuple, Union, MutableMapping, Optional, Callable, Dict, Any

import torch
from omegaconf import DictConfig
from torch import nn

import didactic.models.transformer
from didactic.models.baselines import (
    AvgConcatMLP,
    ConcatMLP,
    ConcatMLPDecoupling,
    ConcatMLPDecoupling2FTs,
    FlatConcatMLP,
    IdentityEncoder,
    IRENEModel,
    MMCLEncoder,
)
# from didactic.models.tabpfn_encoder import TabPFNEncoder
import tabpfn


def check_model_encoder(encoder: nn.Module, hparams: MutableMapping[str, Any]) -> Tuple[int, bool]:
    if isinstance(encoder, nn.TransformerEncoder):  # Native PyTorch `TransformerEncoder`
        nhead = encoder.layers[0].self_attn.num_heads
    elif isinstance(encoder, didactic.models.transformer.FT_Transformer):  # didactic submodule `Transformer`
        nhead = hparams.model.encoder.attention_n_heads
        separate_modality = bool(hparams.model.encoder.n_cross_blocks or hparams.model.encoder.n_bidirectional_blocks)
        print(f"Separate modality is {separate_modality}")
    elif isinstance(encoder, didactic.models.baselines.ConcatMLPDecoupling):  # didactic submodule `MLP`
        nhead = 1
        separate_modality = True
        assert not hparams.cls_token, "ConcatMLP does not support cls_token"
    elif isinstance(encoder, didactic.models.baselines.ConcatMLPDecoupling2FTs):  # didactic submodule `MLP`
        nhead = 1
        separate_modality = True
        assert not hparams.cls_token, "ConcatMLP does not support cls_token"
    elif isinstance(encoder, didactic.models.baselines.FlatConcatMLP):  # didactic submodule `FlatConcatMLP`
        nhead = 1
        separate_modality = True
    elif isinstance(encoder, didactic.models.baselines.AvgConcatMLP):  # didactic submodule `AvgConcatMLP`
        nhead = 1
        separate_modality = True
    elif isinstance(encoder, didactic.models.baselines.ConcatMLP):  # didactic submodule `ConcatMLP`
        nhead = 1
        separate_modality = True
    elif isinstance(encoder, didactic.models.baselines.IdentityEncoder):  # didactic submodule `IdentityEncoder`
        separate_modality = False
    elif isinstance(encoder, didactic.models.baselines.MMCLEncoder):  # didactic submodule `MMCLEncoder (BoB)`
        nhead = 1
        separate_modality = True
        assert not hparams.task.contrastive_loss, "MMCLEncoder does not support contrastive loss"
    elif isinstance(encoder, didactic.models.baselines.IRENEModel):  # didactic submodule `IRENEModel`
        # nhead = hparams.model.encoder.attention_n_heads
        separate_modality = True
        assert not (
            hparams.task.contrastive_loss or hparams.task.inter_sample_loss
        ), "IRENEModel does not support contrastive or inter-sample loss"
    elif isinstance(encoder, didactic.models.transformer.FT_Interleaved):  # didactic submodule `FT_Interleaved`
        nhead = hparams.model.encoder.attention_n_heads
        separate_modality = True
    elif isinstance(encoder, didactic.models.alternatives_transformer.FT_Transformer_2UniFTs):  # didactic submodule `Transformer`
        nhead = hparams.model.encoder.attention_n_heads
        separate_modality = False
    elif isinstance(encoder, didactic.models.transformer.FT_Alignment):  # didactic submodule `FT_Alignment`
        nhead = hparams.model.encoder.attention_n_heads
        separate_modality = False
    elif isinstance(encoder, didactic.models.alternatives_transformer.FT_Alignment_2UniFTs):  # didactic submodule `FT_Alignment`
        nhead = hparams.model.encoder.attention_n_heads
        separate_modality = False
    elif isinstance(encoder, didactic.models.transformer.FT_Interleaved_Alignment):  # didactic submodule `FT_Alignment`
        nhead = hparams.model.encoder.attention_n_heads
        separate_modality = False
    elif isinstance(encoder, didactic.models.transformer.FT_Interleaved_Alignment_NoCross):  # didactic submodule `FT_Alignment`
        nhead = hparams.model.encoder.attention_n_heads
        separate_modality = False
    elif isinstance(encoder, didactic.models.transformer.FT_DiffInterleaved_Alignment):  # didactic submodule `FT_DiffAlignment`
        nhead = hparams.model.encoder.attention_n_heads
        separate_modality = False
    elif isinstance(encoder, didactic.models.tabpfn_encoder.TabPFNEncoder):  # didactic submodule `FT_Alignment`
        nhead = 1
        separate_modality = True
    elif isinstance(
        encoder, didactic.models.alternatives_transformer.FT_Alignment_2UniFTs_BiDirectional
    ):  # didactic submodule `FT_Alignment_BiDirectional`
        nhead = hparams.model.encoder.attention_n_heads
        separate_modality = False
    elif isinstance(
        encoder, didactic.models.alternatives_transformer.FT_Alignment_2UniFTs_CrossAtt
    ):  # didactic submodule `FT_Alignment`
        nhead = hparams.model.encoder.attention_n_heads
        separate_modality = False
    elif isinstance(encoder, didactic.models.transformer.FT_Interleaved_2UniFTs):  # didactic submodule `FT_Interleaved`
        nhead = hparams.model.encoder.attention_n_heads
        separate_modality = False
    elif isinstance(
        encoder, didactic.models.transformer.FT_Interleaved_2UniFTs_Inverted
    ):  # didactic submodule `FT_Interleaved`
        nhead = hparams.model.encoder.attention_n_heads
        separate_modality = False
    elif isinstance(
        encoder, didactic.models.transformer.FT_Interleaved_2UniFTs_nosubmodule
    ):  # didactic submodule `FT_Interleaved`
        nhead = hparams.model.encoder.attention_n_heads
        separate_modality = False
    elif isinstance(
        encoder, didactic.models.alternatives_transformer.FT_Interleaved_2UniFTs_nodecoupling
    ):  # didactic submodule `FT_Interleaved`
        nhead = hparams.model.encoder.attention_n_heads
        separate_modality = False
    # elif isinstance(
    #     encoder, didactic.models.transformer.FT_Interleaved_Inverted
    # ):  # didactic submodule `FT_Interleaved`
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

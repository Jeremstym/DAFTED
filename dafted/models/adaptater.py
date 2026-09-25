"""
Implementation of LoRA (LoRA: Low-Rank Adaptation of Large Language Models: https://arxiv.org/abs/2106.09685)
Codes are modified from (https://github.com/microsoft/LoRA/blob/main/loralib/layers.py)
This example takes inspiration from LoKRD code (https://arxiv.org/abs/2404.17184)
"""
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALayer(nn.Module):
    """
    Base lora class
    """

    def __init__(
        self,
        r,
        lora_alpha,
    ):
        super().__init__()
        self.r = r
        self.lora_alpha = lora_alpha
        # Mark the weight as unmerged
        self.merged = False

    def reset_parameters(self):
        raise NotImplementedError

    def train(self, mode: bool = True):
        raise NotImplementedError

    def eval(self):
        raise NotImplementedError


class LoRALinear(LoRALayer):
    def __init__(self, r, lora_alpha, linear_layer):
        """
        LoRA class for nn.Linear class
        :param r: low rank dimension
        :param lora_alpha: scaling factor
        :param linear_layer: target nn.Linear layer for applying Lora
        """
        super().__init__(r, lora_alpha)
        self.linear = linear_layer

        in_features = self.linear.in_features
        out_features = self.linear.out_features

        # Lora configuration
        self.lora_A = nn.Parameter(self.linear.weight.new_zeros((r, in_features)))
        self.lora_B = nn.Parameter(self.linear.weight.new_zeros((out_features, r)))
        self.scaling = self.lora_alpha / self.r
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def train(self, mode: bool = True):
        self.linear.train(mode)
        if self.merged:
            self.linear.weight.data -= (self.lora_B @ self.lora_A) * self.scaling
            self.merged = False

    def eval(self):
        self.linear.eval()
        if not self.merged:
            self.linear.weight.data += (self.lora_B @ self.lora_A) * self.scaling
            self.merged = True

    def forward(self, x):
        if not self.merged:
            result = F.linear(x, self.linear.weight, bias=self.linear.bias)
            out = x @ self.lora_A.T @ self.lora_B.T
            result += out
            return result
        else:
            return F.linear(x, self.linear.weight, bias=self.linear.bias)


class MultiLoRAMultiheadSelfAttenion(LoRALayer):
    def __init__(self, r, lora_alpha, multihead_attention, num_task):
        """
        LoRA class for nn.MultiheadAttention class
        :param r: low rank dimension
        :param lora_alpha: scaling factor
        :param multihead_attention: target nn.MultiheadAttention layer for applying Lora
        """
        super().__init__(r, lora_alpha)
        self.multihead_attention = multihead_attention
        self.num_task = num_task
        self.one_hot_task = torch.eye(num_task).cuda()

        d_token = self.l_attention.linear_proj.d_token
        # num_heads = self.multihead_attention.num_heads

        # Lora configuration
        self.lora_A_list = nn.ParamList(
            [
                nn.Parameter(self.multihead_attention.linear_proj.W_q.weight.new_zeros((r, d_token)))
                for task in range(num_task)
            ]
        )
        self.lora_B_list = nn.ParamList(
            [
                nn.Parameter(self.multihead_attention.linear_proj.W_q.weight.new_zeros((d_token, r)))
                for task in range(num_task)
            ]
        )
        self.scaling = self.lora_alpha / self.r
        self.reset_parameters()

    def reset_parameters(self):
        for task in range(self.num_task):
            nn.init.kaiming_uniform_(self.lora_A_list[task], a=math.sqrt(5))
            nn.init.zeros_(self.lora_B_list[task])

    def train(self, mode: bool = True):
        self.multihead_attention.train(mode)
        # if self.merged:
        #     self.multihead_attention.in_proj_weight.data -= (self.lora_A @ self.lora_B) * self.scaling
        #     self.merged = False

    def eval(self):
        self.multihead_attention.eval()
        # if not self.merged:
        #     self.multihead_attention.in_proj_weight.data += (self.lora_A @ self.lora_B) * self.scaling
        #     self.merged = True

    def mhsa_forward(self, query, key_value, w_q_weight, w_q_bias, w_k, w_v, attn_mask=None):
        q = F.linear(query, w_q.weight, w_q.bias)
        k = F.linear(key_value, w_k.weight, w_k.bias)
        v = F.linear(key_value, w_v.weight, w_v.bias)

        output = F.scaled_dot_product_attention(q, k, v, attn_mask)
        return output

    def forward(self, query, key, value, key_padding_mask=None):
        weight_shape = self.multihead_attention.linear_proj.W_q.weight.shape
        weight_stack = torch.stack(
            [(self.lora_B_list[task] @ self.lora_A_list[task]).view(weight_shape) for task in range(self.num_task)]
        )
        batch_size = query.shape[0]
        agg_weights = self.multihead_attention.linear_proj.W_q.weight + torch.sum(
            torch.mul(weight_stack.unsqueeze(0), self.one_hot_task.view(batch_size, -1, 1, 1)), dim=1
        )

        output = self.mhsa_forward(
            query,
            key_value,
            agg_weights,
            self.multihead_attention.linear_proj.W_q.bias,
            self.multihead_attention.linear_proj.W_k,
            self.multihead_attention.linear_proj.W_v,
            key_padding_mask,
        )
        # results_stack = []
        # for task in range(self.num_task):
        #     result = self.multihead_attention(query, key, value, key_padding_mask, need_weights, attn_mask)
        #     out = (query @ self.lora_A_list[task].T @ self.lora_B_list[task].T)
        #     result = (result[0] + out, *result[1:])
        #     results_stack.append(result)
        # result = self.multihead_attention(query, key, value, key_padding_mask, need_weights, attn_mask)
        # out = (query @ self.lora_A.T @ self.lora_B.T)
        # result = (result[0] + out, *result[1:])
        # return result
        # else:
        # return self.multihead_attention(query, key, value, key_padding_mask, need_weights, attn_mask)


class AdapterWrapperFT_Transformer(nn.Module):
    def __init__(self, encoder, adapter_class, gamma, lora_alpha):
        super().__init__()
        self.lora = encoder
        self.add_multi_adapter(adapter_class, gamma, lora_alpha)
        # self.model_frozen = False
        self.freeze_model(True)

    def add_multi_adapter(self, adapter_class, gamma, lora_alpha):
        """
        Add adapter to resnets
        :param adapter_class: class for adapter
        """
        # Add adapter input convolution.
        # target_conv = self.resnet.conv1
        # adapter = adapter_class(
        #     r=gamma,
        #     lora_alpha=lora_alpha,
        #     conv_layer=target_conv
        # )
        # adapter = adapter_class(r=gamma, lora_alpha=lora_alpha, conv_layer=target_conv,)

        # setattr(self.resnet, "conv1", adapter)

        for layer in self.lora.blocks:
            target_layer = layer["ffn"].linear_first
            adapter = adapter_class(r=gamma, lora_alpha=lora_alpha, linear_layer=target_layer)
            setattr(layer["ffn"], "linear_first", adapter)
            target_layer = layer["ffn"].linear_second
            adapter = adapter_class(r=gamma, lora_alpha=lora_alpha, linear_layer=target_layer)
            setattr(layer["ffn"], "linear_second", adapter)

    def forward(self, x):
        return self.lora(x)

    def freeze_model(self, freeze=True):  #
        """Freezes all weights of the encoder."""
        if freeze:  # 只更新lora, 非fc中的bias, 以及bn
            # First freeze/ unfreeze all encoder weights
            for n, p in self.named_parameters():
                if "lora_" not in n:
                    p.requires_grad = False
                else:
                    p.requires_grad = True
            # for n, p in self.named_parameters():
            #     if 'linear_first.bias' in n or "linear_second.bias" in n:
            #         p.requires_grad = True
            # elif "bn" in n:
            #     p.requires_grad = True
        else:
            # Unfreeze
            for n, p in self.named_parameters():
                p.requires_grad = True
        self.model_frozen = freeze


class AdapterWrapperFT_Interleaved(nn.Module):
    def __init__(self, encoder, adapter_class, gamma, lora_alpha):
        super().__init__()
        self.lora = encoder
        self.add_multi_adapter(adapter_class, gamma, lora_alpha)
        # self.model_frozen = False
        self.freeze_model(True)

    def add_multi_adapter(self, adapter_class, gamma, lora_alpha):
        """
        Add adapter to resnets
        :param adapter_class: class for adapter
        """
        # Add adapter input convolution.
        # target_conv = self.resnet.conv1
        # adapter = adapter_class(
        #     r=gamma,
        #     lora_alpha=lora_alpha,
        #     conv_layer=target_conv
        # )
        # adapter = adapter_class(r=gamma, lora_alpha=lora_alpha, conv_layer=target_conv,)

        # setattr(self.resnet, "conv1", adapter)

        for layer in self.lora.blocks:
            target_layer = layer["ffn"].linear_first
            adapter = adapter_class(r=gamma, lora_alpha=lora_alpha, linear_layer=target_layer)
            setattr(layer["ffn"], "linear_first", adapter)
            target_layer = layer["ffn"].linear_second
            adapter = adapter_class(r=gamma, lora_alpha=lora_alpha, linear_layer=target_layer)
            setattr(layer["ffn"], "linear_second", adapter)

    def forward(self, x_tab, x_ts):
        return self.lora(x_tab, x_ts)

    def freeze_model(self, freeze=True):  #
        """Freezes all weights of the encoder."""
        if freeze:  # 只更新lora, 非fc中的bias, 以及bn
            # First freeze/ unfreeze all encoder weights
            for n, p in self.named_parameters():
                if "attention" in n or "ffn" in n:
                    p.requires_grad = False
                else:
                    p.requires_grad = True
                if "lora_" in n:
                    p.requires_grad = True
                if "cross_attention" in n or "cross_ffn" in n:
                    p.requires_grad = True
            # for n, p in self.named_parameters():
            #     if 'linear_first.bias' in n or "linear_second.bias" in n:
            #         p.requires_grad = True
            # elif "bn" in n:
            #     p.requires_grad = True
        else:
            # Unfreeze
            for n, p in self.named_parameters():
                p.requires_grad = True
        self.model_frozen = freeze

    def adapter_state_dict(self):
        """
        Save only adapter parts
        """
        state_dict = self.state_dict()
        adapter_dict = OrderedDict()

        for name, param in state_dict.items():
            if "lora_" in name:
                adapter_dict[name] = param
            elif "bn" in name:
                adapter_dict[name] = param
            elif "bias" in name:
                if "fc" not in name:
                    adapter_dict[name] = param
        return adapter_dict


class AdapterWrapperFT_Transformer_CrossAtt(nn.Module):
    def __init__(self, encoder, adapter_class, gamma, lora_alpha):
        super().__init__()
        self.lora = encoder
        self.add_multi_adapter(adapter_class, gamma, lora_alpha)
        # self.model_frozen = False
        self.freeze_model(True)

    def add_multi_adapter(self, adapter_class, gamma, lora_alpha):
        """
        Add adapter to resnets
        :param adapter_class: class for adapter
        """
        # Add adapter input convolution.
        # target_conv = self.resnet.conv1
        # adapter = adapter_class(
        #     r=gamma,
        #     lora_alpha=lora_alpha,
        #     conv_layer=target_conv
        # )
        # adapter = adapter_class(r=gamma, lora_alpha=lora_alpha, conv_layer=target_conv,)

        # setattr(self.resnet, "conv1", adapter)

        for layer in self.lora.blocks:
            if "l_ffn" in layer:
                target_layer = layer["l_ffn"].linear_first
                adapter = adapter_class(r=gamma, lora_alpha=lora_alpha, linear_layer=target_layer)
                setattr(layer["l_ffn"], "linear_first", adapter)
                target_layer = layer["l_ffn"].linear_second
                adapter = adapter_class(r=gamma, lora_alpha=lora_alpha, linear_layer=target_layer)
                setattr(layer["l_ffn"], "linear_second", adapter)
            if "r_ffn" in layer:
                target_layer = layer["r_ffn"].linear_first
                adapter = adapter_class(r=gamma, lora_alpha=lora_alpha, linear_layer=target_layer)
                setattr(layer["r_ffn"], "linear_first", adapter)
                target_layer = layer["r_ffn"].linear_second
                adapter = adapter_class(r=gamma, lora_alpha=lora_alpha, linear_layer=target_layer)
                setattr(layer["r_ffn"], "linear_second", adapter)

    def forward(self, x_tab, x_ts):
        return self.lora(x_tab, x_ts)

    def freeze_model(self, freeze=True):  #
        """Freezes all weights of the encoder."""
        if freeze:  # 只更新lora, 非fc中的bias, 以及bn
            # First freeze/ unfreeze all encoder weights
            for n, p in self.named_parameters():
                if any([x in n for x in ["blocks.0", "blocks.1", "blocks.2"]]):
                    if "lora_" not in n:
                        p.requires_grad = False
                    else:
                        p.requires_grad = True
                # for n, p in self.named_parameters():
                #     if any([x in n for x in ["blocks.0", "blocks.1", "blocks.2"]]):
                # if 'linear_first.bias' in n or "linear_second.bias" in n:
                #     if "fc" not in n:
                #     p.requires_grad = True
                # elif "bn" in n:
                #     p.requires_grad = True

                else:
                    p.requires_grad = True
        else:
            # Unfreeze
            for n, p in self.named_parameters():
                p.requires_grad = True
        self.model_frozen = freeze


class AdapterWrapperShuffleNet(nn.Module):
    def __init__(self, transformer_model, adapter_class, num_task, gamma, lora_alpha):
        super().__init__()
        self.transformer = transformer_model
        self.add_multi_adapter(adapter_class, num_task, gamma, lora_alpha)
        self.model_frozen = True
        self.freeze_model(True)

    def add_multi_adapter(self, adapter_class, num_task, gamma, lora_alpha):
        """
        Add adapter to resnets
        :param adapter_class: class for adapter
        """
        # Add adapter input convolution.
        target_attention = self.transformer.blocks[0].l_attention
        adapter = adapter_class(r=gamma, lora_alpha=lora_alpha, conv_layer=target_attention, num_task=num_task)

        # if target_attention.groups == 1:
        #     # setattr(target_conv, "conv1", adapter)
        #     setattr(self.transformer.blocks[0].l_attention, "tab_attention", adapter)

        # Add adapter for transformer blocks
        target_layers = self.transformer.blocks

        for th, layer in enumerate(target_layers[: self.transformer.n_self_blocks]):
            # print('layer:', layer)
            if th == len(target_layers) - 1:
                adapter = adapter_class(r=gamma, lora_alpha=lora_alpha, conv_layer=layer.conv, num_task=num_task)
                if layer.l_attention.groups == 1:
                    setattr(layer, "lora_attention", adapter)
                break
            for bottleneck_layer in layer:
                # print('bottleneck_layer:', bottleneck_layer)
                if hasattr(bottleneck_layer, "branch1"):
                    for each_branch in [bottleneck_layer.branch1, bottleneck_layer.branch2]:
                        # print('each_branch:', each_branch)
                        for each_conv in each_branch:
                            # print('each_conv:', each_conv)
                            # bound_method = mmcv_conv_forward.__get__(each_conv, each_conv.__class__)
                            # assert each_conv.__class__ == ConvModule
                            # setattr(each_conv, 'forward', bound_method)

                            adapter = adapter_class(
                                r=gamma, lora_alpha=lora_alpha, conv_layer=each_conv.conv, num_task=num_task
                            )
                            if each_conv.conv.groups == 1:
                                setattr(each_conv, "conv", adapter)
                else:
                    for each_branch in [bottleneck_layer.branch2]:
                        # print('each_branch:', each_branch)
                        for each_conv in each_branch:
                            # print('each_conv:', each_conv)
                            # bound_method = mmcv_conv_forward.__get__(each_conv, each_conv.__class__)
                            # assert each_conv.__class__ == ConvModule
                            # setattr(each_conv, 'forward', bound_method)

                            adapter = adapter_class(
                                r=gamma, lora_alpha=lora_alpha, conv_layer=each_conv.conv, num_task=num_task
                            )
                            if each_conv.conv.groups == 1:
                                setattr(each_conv, "conv", adapter)

        # raise ValueError()

    def calculate_training_parameter_ratio(self):
        def count_parameters(model, grad):
            return sum(p.numel() for p in model.parameters() if p.requires_grad == grad)

        trainable_param_num = count_parameters(self.transformer, True)
        other_param_num = count_parameters(self.transformer, False)
        print("Non-trainable parameters:", other_param_num)
        print("Trainable parameters:", trainable_param_num)

        ratio = trainable_param_num / other_param_num
        final_ratio = ratio / (1 - ratio)
        print("Ratio:", final_ratio)

        return final_ratio

    def forward(self, x, alphas=None):
        return self.transformer(x, alphas=alphas)

    def freeze_model(self, freeze=True):  #
        """Freezes all weights of the model."""
        if freeze:  # 只更新lora, 非fc中的bias, 以及bn
            # First freeze/ unfreeze all model weights
            for n, p in self.named_parameters():
                if "lora_" not in n:
                    p.requires_grad = False
                else:
                    p.requires_grad = True
            for n, p in self.named_parameters():
                if "bias" in n:
                    if "fc" not in n:
                        p.requires_grad = True
                elif "bn" in n:
                    p.requires_grad = True
        else:
            # Unfreeze
            for n, p in self.named_parameters():
                p.requires_grad = True
        self.model_frozen = freeze

    def adapter_state_dict(self):
        """
        Save only adapter parts
        """
        state_dict = self.state_dict()
        adapter_dict = OrderedDict()

        for name, param in state_dict.items():
            if "lora_" in name:
                adapter_dict[name] = param
            elif "bn" in name:
                adapter_dict[name] = param
            elif "bias" in name:
                if "fc" not in name:
                    adapter_dict[name] = param
        return adapter_dict


# def create_pretrained_classifier(classifier_name):
#     """
#     create pretrained classifier from classifier_name and load publicly available model weights.

#     :param classifier_name: a string classifier name. (only xtab supported for now)
#     :return: a nn.Module class transformer.
#     """
#     if classifier_name == "xtab":
#         from xtab.models import XTab
#         classifier = XTab()
#         classifier.load_from_pretrained("xtab-base")
#     else:
#         raise ValueError(f"classifier_name is not supported for : {classifier_name}")

#     return classifier

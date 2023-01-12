"""
Backbone modules.
"""
import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List
from util.misc import NestedTensor, is_main_process
from .position_encoding import build_position_encoding


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """
    def __init__(self, n, eps=1e-5):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))
        self.eps = eps

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d,
              self)._load_from_state_dict(state_dict, prefix, local_metadata,
                                          strict, missing_keys,
                                          unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = self.eps
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):
    def __init__(self, backbone: nn.Module, train_backbone: bool,
                 return_interm_layers: bool):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        if return_interm_layers:
            # return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
            return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"}
            self.strides = [8, 16, 32]
            self.num_channels = [512, 1024, 2048]
        else:
            return_layers = {'layer4': "0"}
            self.strides = [32]
            self.num_channels = [2048]
        self.body = IntermediateLayerGetter(backbone,
                                            return_layers=return_layers)

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)

        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(),
                                 size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)

        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str, train_backbone: bool,
                 return_interm_layers: bool, dilation: bool):
        norm_layer = FrozenBatchNorm2d
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(),
            norm_layer=norm_layer)
        assert name not in ('resnet18',
                            'resnet34'), "number of channels are hard coded"
        super().__init__(backbone, train_backbone, return_interm_layers)
        if dilation:
            self.strides[-1] = self.strides[-1] // 2


class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BottleneckBlock, self).__init__()
        inter_planes = out_planes * 4
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes,
                               inter_planes,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(inter_planes)
        self.conv2 = nn.Conv2d(inter_planes,
                               out_planes,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.droprate = dropRate

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out,
                            p=self.droprate,
                            inplace=False,
                            training=self.training)
        out = self.conv2(self.relu(self.bn2(out)))
        if self.droprate > 0:
            out = F.dropout(out,
                            p=self.droprate,
                            inplace=False,
                            training=self.training)
        return torch.cat([x, out], 1)




class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding, self_attn=False):
        super().__init__(backbone, position_embedding)
        self.strides = backbone.strides
        self.num_channels = backbone.num_channels
        # self attention
        self.self_attn = self_attn
        in_channels = [512, 1024, 2048]

        self.avg_pool256 = nn.AdaptiveAvgPool3d((256, None, None))
        self.avg_pool512 = nn.AdaptiveAvgPool3d((512, None, None))
        self.avg_pool1024 = nn.AdaptiveAvgPool3d((1024, None, None))
        self.avg_pool2048 = nn.AdaptiveAvgPool3d((2048, None, None))
        
        if self.self_attn:
            d_model, n_heads, dropout = 256, 4, 0.1
            self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
            self.norm1 = nn.LayerNorm(d_model)
            input_proj_list = []
            for in_channel in in_channels:
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(in_channel, d_model, kernel_size=1),
                        nn.GroupNorm(32, d_model),
                    ))
            self.input_proj = nn.ModuleList(input_proj_list)

    def forward(self, tensor_list: NestedTensor):
        if tensor_list.tensors.ndim == 5:
            b, f, c, h, w = tensor_list.tensors.size()
            tensor_list.tensors = tensor_list.tensors.view(-1, c, h, w)

        xs = self[0](tensor_list)
        # print(b, f, c, h, w)
        out: List[NestedTensor] = []
        pos = []

        for i, (name, x) in enumerate(sorted(xs.items())):
            c, w, h = x.tensors.shape[-3:]
            xis = []
            if c == 512:
                cnt = 0
            elif c == 1024:
                cnt = 1
            else:
                cnt = 2
            x.tensors = self.input_proj[cnt](x.tensors)
            c, w, h = x.tensors.shape[-3:]

            x.tensors = x.tensors.reshape(b, f, c, w, h)

            # inter-video fusion
            for i in range(f // 2):
                x1 = x.tensors[:, i, :, :, :].flatten(2).permute(2, 0, 1)
                x2 = x.tensors[:, i + f // 2, :, :, :].flatten(2).permute(2, 0, 1)
                
                x_i = self.attn(x2, x1, x1)[0]
                x_i = self.norm1(x_i)
                x_i = x1 + x_i
                xis.append(x_i)

            # intra-video fusion
            xx = self.attn(xis[2], xis[1], xis[0])[0]
            xx = xis[0] + xx
            xx = xx.permute(1, 2, 0).reshape(b, c, w, h)
            x.tensors = xx
            out.append(x)

        # position encoding
        for x in out:
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    return_interm_layers = args.masks or (args.num_feature_levels > 1)
    backbone = Backbone(args.backbone, train_backbone, return_interm_layers,
                        args.dilation)
    model = Joiner(backbone, position_embedding, args.self_attn)
    return model

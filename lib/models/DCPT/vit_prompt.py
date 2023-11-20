from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import Mlp, DropPath, trunc_normal_, lecun_normal_
from .utils import combine_tokens, token2feature, feature2token
from lib.models.layers.patch_embed import PatchEmbed
from .vit import VisionTransformer

class ConvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size, stride, padding, bias=True, isuseBN=False):
        super(ConvBlock, self).__init__()
        self.isuseBN = isuseBN
        self.conv = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias=bias)
        if self.isuseBN:
            self.bn = nn.BatchNorm2d(output_size)
        self.act = torch.nn.PReLU()

    def forward(self, x):
        out = self.conv(x)
        if self.isuseBN:
            out = self.bn(out)
        out = self.act(out)
        return out

class DCEBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size, stride, padding, bias=True):
        super(DCEBlock, self).__init__()
        codedim=output_size//2
        self.conv_Encoder = ConvBlock(input_size, codedim, 3, 1, 1,isuseBN=False)
        self.conv_Offset = ConvBlock(codedim, codedim, 3, 1, 1,isuseBN=False)
        self.conv_Decoder = ConvBlock(codedim, output_size, 3, 1, 1,isuseBN=False)

    def forward(self, x):
        code= self.conv_Encoder(x)
        offset = self.conv_Offset(code)
        code_lighten = code+offset
        out = self.conv_Decoder(code_lighten)
        return out

class DCUBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size, stride, padding, bias=True):
        super(DCUBlock, self).__init__()
        codedim=output_size//2
        self.conv_Encoder = ConvBlock(input_size, codedim, 3, 1, 1,isuseBN=False)
        self.conv_Offset = ConvBlock(codedim, codedim, 3, 1, 1,isuseBN=False)
        self.conv_Decoder = ConvBlock(codedim, output_size, 3, 1, 1,isuseBN=False)

    def forward(self, x):
        code= self.conv_Encoder(x)
        offset = self.conv_Offset(code)
        code_lighten = code-offset
        out = self.conv_Decoder(code_lighten)
        return out

class FusionLayer(nn.Module):
    def __init__(self, inchannel, outchannel, reduction=16):
        super(FusionLayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(inchannel, inchannel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(inchannel // reduction, inchannel, bias=False),
            nn.Sigmoid()
        )
        self.outlayer = ConvBlock(inchannel, outchannel, 1, 1, 0, bias=True)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        y = x * y.expand_as(x)
        y = y + x
        y = self.outlayer(y)
        return y


class LBP(torch.nn.Module):
    def __init__(self, input_size, hide_channel, output_size, kernel_size, stride, padding):
        super(LBP, self).__init__()
        self.conv0 = nn.Conv2d(in_channels=input_size,out_channels=hide_channel, kernel_size=1, stride=1, padding=0)
        self.fusion = FusionLayer(hide_channel,hide_channel)
        self.conv1_1 = DCEBlock(hide_channel, hide_channel, kernel_size, stride, padding, bias=True)
        self.conv2 = DCUBlock(hide_channel, hide_channel, kernel_size, stride, padding, bias=True)
        self.conv3 = DCEBlock(hide_channel, hide_channel, kernel_size, stride, padding, bias=True)
        self.local_weight1_1 = ConvBlock(hide_channel, hide_channel, kernel_size=1, stride=1, padding=0, bias=True)
        self.local_weight2_1 = ConvBlock(hide_channel, hide_channel, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv4 = nn.Conv2d(in_channels=hide_channel,out_channels=output_size, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.conv0(x)
        x=self.fusion(x)
        hr = self.conv1_1(x)
        lr = self.conv2(hr)
        residue = self.local_weight1_1(x) - lr
        h_residue = self.conv3(residue)
        hr_weight = self.local_weight2_1(hr)
        return self.conv4(hr_weight + h_residue)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, return_attention=False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        if return_attention:
            return x, attn
        return x

class Gate_Feature(nn.Module):
    def __init__(self, NUM = 256, GATE_INIT = 10, NUM_TOKENS = 256):
        super().__init__()
        self.num = NUM
        gate_logit = (torch.ones(NUM) * GATE_INIT)
        self.num_tokens = NUM_TOKENS
        self.gate_logit = nn.Parameter(gate_logit)
    def forward(self,xin,xout):
        gate = self.gate_logit.sigmoid()
        gate = gate.unsqueeze(0).unsqueeze(-1).repeat(xin.size(0),1,xin.size(2))
        # convex combinate input and output prompt representations of current block via learnable gate
        prompt_out = gate * xout
        prompt_in = 1 * xin
        xout = prompt_out + prompt_in
        return xout


class Gate_Prompt(nn.Module):
    def __init__(self, NUM = 1, GATE_INIT = 10, NUM_TOKENS = 256):
        super().__init__()
        gate_logit = -(torch.ones(NUM) * GATE_INIT)
        self.num_tokens = NUM_TOKENS
        self.gate_logit = nn.Parameter(gate_logit)
    def forward(self,xin,xout):
        gate = self.gate_logit.sigmoid()
        prompt_in = xin
        # current block's output prompt representation
        prompt_out = xout
        # convex combinate input and output prompt representations of current block via learnable gate
        xout = (1 - gate) * prompt_out + gate * prompt_in
        return xout


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention=False):
        if return_attention:
            feat, attn = self.attn(self.norm1(x), True)
            x = x + self.drop_path(feat)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x, attn
        else:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x


class VisionTransformerP(VisionTransformer):
    """ Vision Transformer
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    Includes distillation token & head support for `DeiT: Data-efficient Image Transformers`
        - https://arxiv.org/abs/2012.12877
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init='',search_size=None, template_size=None,
                 new_patch_size=None, prompt_type=None):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            weight_init: (str): weight init scheme
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        # num_patches = self.patch_embed.num_patches

        '''patch_embed_prompt'''
        self.patch_embed_prompt = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        # self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        '''
        prompt parameters
        '''
        H, W = search_size
        new_P_H, new_P_W = H // new_patch_size, W // new_patch_size
        self.num_patches_search=new_P_H * new_P_W
        H, W = template_size
        new_P_H, new_P_W = H // new_patch_size, W // new_patch_size
        self.num_patches_template=new_P_H * new_P_W
        """add here, no need use backbone.finetune_track """
        self.pos_embed_z = nn.Parameter(torch.zeros(1, self.num_patches_template, embed_dim))
        self.pos_embed_x = nn.Parameter(torch.zeros(1, self.num_patches_search, embed_dim))

        self.prompt_type = prompt_type
        '''prompt parameters'''
        if self.prompt_type in ['DCPT']:
            prompt_blocks = []
            block_nums = depth
            for i in range(block_nums):
                prompt_blocks.append(LBP(input_size=768, hide_channel=64, output_size=768, kernel_size=3, stride=1, padding=1))
            self.prompt_blocks = nn.Sequential(*prompt_blocks)

            prompt_norms = []
            for i in range(block_nums):
                prompt_norms.append(norm_layer(embed_dim))
            self.prompt_norms = nn.Sequential(*prompt_norms)

            prompt_gates = []
            for i in range(block_nums-1):
                prompt_gates.append(Gate_Prompt(GATE_INIT = 10, NUM_TOKENS = 256))
            self.prompt_gates = nn.Sequential(*prompt_gates)

            prompt_feature_gates = []
            for i in range(block_nums):
                prompt_feature_gates.append(Gate_Feature(GATE_INIT = 10, NUM_TOKENS = 256))
            self.prompt_feature_gates = nn.Sequential(*prompt_feature_gates)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        self.init_weights(weight_init)
        self.cnt = 0

    def forward_features(self, z, x, mask_z=None, mask_x=None,
                         return_last_attn=False
                         ):
        self.cnt = self.cnt + 1
        B, H, W = x.shape[0], x.shape[2], x.shape[3]

        z = self.patch_embed(z)
        x = self.patch_embed(x)
        feature_prev = x


        # '''input prompt:
        # by adding to rgb tokens
        # '''
        if self.prompt_type in ['DCPT']:
            z_feat = token2feature(self.prompt_norms[0](z))
            x_feat = token2feature(self.prompt_norms[0](x))
            z_feat = self.prompt_blocks[0](z_feat)
            x_feat = self.prompt_blocks[0](x_feat)
            z_dte = feature2token(z_feat)
            x_dte = feature2token(x_feat)

            x_prev = x_dte

            #feature gates
            x = self.prompt_feature_gates[0](feature_prev, x_dte)
        

        # attention mask handling
        # B, H, W
        if mask_z is not None and mask_x is not None:
            mask_z = F.interpolate(mask_z[None].float(), scale_factor=1. / self.patch_size).to(torch.bool)[0]
            mask_z = mask_z.flatten(1).unsqueeze(-1)

            mask_x = F.interpolate(mask_x[None].float(), scale_factor=1. / self.patch_size).to(torch.bool)[0]
            mask_x = mask_x.flatten(1).unsqueeze(-1)

            mask_x = combine_tokens(mask_z, mask_x, mode=self.cat_mode)
            mask_x = mask_x.squeeze(-1)

        if self.add_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            cls_tokens = cls_tokens + self.cls_pos_embed

        z += self.pos_embed_z
        x += self.pos_embed_x

        if self.add_sep_seg:
            x += self.search_segment_pos_embed
            z += self.template_segment_pos_embed

        x = combine_tokens(z, x, mode=self.cat_mode)
        if self.add_cls_token:
            x = torch.cat([cls_tokens, x], dim=1)

        x = self.pos_drop(x)

        lens_z = self.pos_embed_z.shape[1]
        lens_x = self.pos_embed_x.shape[1]

        for i, blk in enumerate(self.blocks):
            '''
            add parameters prompt from 1th layer
            '''
            if i >= 1:
                if self.prompt_type in ['DCPT']:
                    x_ori = x
                    feature_prev = x_ori[:, lens_z:, :]
                    # prompt
                    x = self.prompt_norms[i](x)
                    z_tokens = x[:, :lens_z, :]
                    x_tokens = x[:, lens_z:, :]
                    z_feat = token2feature(z_tokens)
                    x_feat = token2feature(x_tokens)

                    z_feat = self.prompt_blocks[i](z_feat)
                    x_feat = self.prompt_blocks[i](x_feat)
                    z = feature2token(z_feat)
                    x = feature2token(x_feat)
                    #prompt gates
                    x = self.prompt_gates[i-1](x_prev, x)
                    x_prev = x
                    #feature gates
                    x = self.prompt_feature_gates[i](feature_prev, x)
                    x = combine_tokens(x_ori[:, :lens_z, :], x, mode=self.cat_mode)

            x = blk(x)

        x = self.norm(x)
        aux_dict = {"attn": None}
        return x, aux_dict

    def forward(self, z, x, ce_template_mask=None, ce_keep_rate=None,
                tnc_keep_rate=None,
                return_last_attn=False):

        x, aux_dict = self.forward_features(z, x)

        return x, aux_dict


def _create_vision_transformer(pretrained=False, **kwargs):
    model = VisionTransformerP(**kwargs)

    if pretrained:
        if 'npz' in pretrained:
            model.load_pretrained(pretrained, prefix='')
        else:
            checkpoint = torch.load(pretrained, map_location="cpu")
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
            print('Load pretrained DCPT without CE from: ' + pretrained)
            print(f"missing_keys: {missing_keys}")
            print(f"unexpected_keys: {unexpected_keys}")

    return model


def vit_base_patch16_224_prompt(pretrained=False, **kwargs):
    """
    ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(pretrained=pretrained, **model_kwargs)
    return model


import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.update import BasicMultiUpdateBlock_no_mono_disp, BasicMultiUpdateBlock_mix2
from core.geometry import Combined_Geo_Encoding_Volume
from core.submodule import *
from core.refinement import REMP
from core.warp import disp_warp
import matplotlib.pyplot as plt
import time
from torchvision import transforms


import functools
import math
try:
    autocast = torch.cuda.amp.autocast
except:
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Depth-Anything-V2-list_intermediate_level"))


from depth_anything_v2.dpt import (
    DepthAnythingV2,
    DepthAnythingV2_decoder,
    IntermediateDPTHead,
    IntermediateDPTFeatureHead,
)


# Project ViT-S patch tokens into the teacher-assistant feature space.
class StudentToTAProjection(nn.Module):
    """
    Project ViT-S tokens from 384 dimensions into the
    704-dimensional Teacher Assistant representation space.
    """

    def __init__(
        self,
        num_layers=4,
        student_dim=384,
        ta_dim=704,
    ):
        super().__init__()

        self.projections = nn.ModuleList([
            nn.Linear(
                student_dim,
                ta_dim,
                bias=False,
            )
            for _ in range(num_layers)
        ])

    def forward(self, student_features):
        """
        student_features:
            list of four tuples:
                patch_tokens: [B, N, 384]
                cls_token:    [B, 384]

        returns:
            projected patch tokens:
                list of four [B, N, 704] tensors
        """

        projected_features = []

        for projection, layer_features in zip(
            self.projections,
            student_features,
        ):
            patch_tokens, cls_token = layer_features

            projected_patch_tokens = projection(
                patch_tokens
            )

            projected_features.append(
                projected_patch_tokens
            )

        return projected_features

# Build intermediate teacher-assistant features from student and teacher tokens.
class IntermediateCrossAttentionEncoder(nn.Module):
    """
    Builds four intermediate encoder layers by fusing corresponding
    ViT-S and ViT-L intermediate representations.

    Input format for each layer:
        student_features[i] = (
            student_patch_tokens,   # [B, N, 384]
            student_cls_token       # [B, 384]
        )

        teacher_features[i] = (
            teacher_patch_tokens,   # [B, N, 1024]
            teacher_cls_token       # [B, 1024]
        )

    Output format:
        fused_features[i] = (
            fused_patch_tokens,     # [B, N, 704]
            fused_cls_token         # [B, 704]
        )
    """

    def __init__(
        self,
        num_layers=4,
        student_dim=384,
        teacher_dim=1024,
        fusion_dim=704,
        num_heads=11,
        ffn_ratio=4.0,
        dropout=0.0,
    ):
        super().__init__()

        self.num_layers = num_layers
        self.fusion_dim = fusion_dim

        self.fusion_blocks = nn.ModuleList([
            CrossAttentionFusionBlock(
                student_dim=student_dim,
                teacher_dim=teacher_dim,
                fusion_dim=fusion_dim,
                num_heads=num_heads,
                ffn_ratio=ffn_ratio,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

    def forward(
        self,
        student_features,
        teacher_features,
    ):
        if len(student_features) != self.num_layers:
            raise ValueError(
                f"Expected {self.num_layers} student layers, "
                f"got {len(student_features)}"
            )

        if len(teacher_features) != self.num_layers:
            raise ValueError(
                f"Expected {self.num_layers} teacher layers, "
                f"got {len(teacher_features)}"
            )

        fused_features = []

        for layer_idx, (
            fusion_block,
            student_layer,
            teacher_layer,
        ) in enumerate(zip(
            self.fusion_blocks,
            student_features,
            teacher_features,
        )):
            student_patch_tokens, student_cls_token = student_layer
            teacher_patch_tokens, teacher_cls_token = teacher_layer


            if student_patch_tokens.shape[:2] != \
                    teacher_patch_tokens.shape[:2]:
                raise ValueError(
                    f"Layer {layer_idx}: patch-token mismatch. "
                    f"Student: {student_patch_tokens.shape}, "
                    f"teacher: {teacher_patch_tokens.shape}"
                )

            if student_cls_token.ndim == 2:
                student_cls_token = student_cls_token.unsqueeze(1)

            if teacher_cls_token.ndim == 2:
                teacher_cls_token = teacher_cls_token.unsqueeze(1)

            student_all_tokens = torch.cat(
                [
                    student_cls_token,
                    student_patch_tokens,
                ],
                dim=1,
            )

            teacher_all_tokens = torch.cat(
                [
                    teacher_cls_token,
                    teacher_patch_tokens,
                ],
                dim=1,
            )

            fused_all_tokens = fusion_block(
                student_all_tokens,
                teacher_all_tokens,
            )

            fused_cls_token = fused_all_tokens[:, 0]
            fused_patch_tokens = fused_all_tokens[:, 1:]

            fused_features.append(
                (
                    fused_patch_tokens,
                    fused_cls_token,
                )
            )

        return fused_features


# Cross-attention transfers teacher information; self-attention refines fused tokens.
class CrossAttentionFusionBlock(nn.Module):
    """
    1. Student tokens attend to teacher tokens.
    2. The resulting fused tokens attend to each other.
    3. A per-token FFN is applied.

    Inputs:
        student_tokens: [B, L, student_dim]
        teacher_tokens: [B, L, teacher_dim]

    Output:
        fused_tokens: [B, L, fusion_dim]
    """

    def __init__(
        self,
        student_dim=384,
        teacher_dim=1024,
        fusion_dim=704,
        num_heads=11,
        ffn_ratio=4.0,
        dropout=0.0,
    ):
        super().__init__()

        if fusion_dim % num_heads != 0:
            raise ValueError(
                f"fusion_dim={fusion_dim} must be divisible by "
                f"num_heads={num_heads}"
            )

        self.student_query_projection = nn.Linear(
            student_dim,
            fusion_dim,
            bias=True,
        )

        self.teacher_key_projection = nn.Linear(
            teacher_dim,
            fusion_dim,
            bias=True,
        )

        self.teacher_value_projection = nn.Linear(
            teacher_dim,
            fusion_dim,
            bias=True,
        )

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.self_attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm_cross = nn.LayerNorm(fusion_dim)
        self.norm_self = nn.LayerNorm(fusion_dim)
        self.norm_ffn = nn.LayerNorm(fusion_dim)

        hidden_dim = int(fusion_dim * ffn_ratio)

        self.ffn = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, fusion_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        student_tokens,
        teacher_tokens,
    ):
        """
        student_tokens: [B, L, 384]
        teacher_tokens: [B, L, 1024]

        returns:
            fused_tokens: [B, L, 704]
        """

        if student_tokens.ndim != 3:
            raise ValueError(
                f"Student tokens must have shape [B, L, C], "
                f"got {student_tokens.shape}"
            )

        if teacher_tokens.ndim != 3:
            raise ValueError(
                f"Teacher tokens must have shape [B, L, C], "
                f"got {teacher_tokens.shape}"
            )

        if student_tokens.shape[:2] != teacher_tokens.shape[:2]:
            raise ValueError(
                "Student and teacher batch/token dimensions must match. "
                f"Student: {student_tokens.shape}, "
                f"teacher: {teacher_tokens.shape}"
            )

        query = self.student_query_projection(student_tokens)
        key = self.teacher_key_projection(teacher_tokens)
        value = self.teacher_value_projection(teacher_tokens)

        cross_output, _ = self.cross_attention(
            query=query,
            key=key,
            value=value,
            need_weights=False,
        )

        fused_tokens = self.norm_cross(
            query + cross_output
        )

        self_output, _ = self.self_attention(
            query=fused_tokens,
            key=fused_tokens,
            value=fused_tokens,
            need_weights=False,
        )

        fused_tokens = self.norm_self(
            fused_tokens + self_output
        )

        fused_tokens = self.norm_ffn(
            fused_tokens + self.ffn(fused_tokens)
        )

        return fused_tokens
class StudentEncoderToTeacherProjection(nn.Module):
    def __init__(self, student_dim=384, teacher_dim=1024):
        super().__init__()

        self.projections = nn.ModuleList([
            nn.Linear(student_dim, teacher_dim, bias=False)
            for _ in range(4)
        ])

    def forward(self, features):
        projected_features = []

        for proj, feat in zip(self.projections, features):
            projected_features.append(proj(feat))

        return projected_features
    
class StudentToTeacherProjection(nn.Module):
    def __init__(self, student_channels=64, teacher_channels=256):
        super().__init__()

        self.proj4 = nn.Conv2d(
            student_channels,
            teacher_channels,
            kernel_size=1,
            bias=False,
        )

        self.proj8 = nn.Conv2d(
            student_channels,
            teacher_channels,
            kernel_size=1,
            bias=False,
        )

        self.proj16 = nn.Conv2d(
            student_channels,
            teacher_channels,
            kernel_size=1,
            bias=False,
        )

    def forward(self, features):
        feat4, feat8, feat16 = features

        feat4 = self.proj4(feat4)
        feat8 = self.proj8(feat8)
        feat16 = self.proj16(feat16)

        return [feat4, feat8, feat16]
def load_depth_anything_branch(encoder_name, mono_model_configs, ROOT):
    model = DepthAnythingV2(**mono_model_configs[encoder_name])
    decoder_model = DepthAnythingV2_decoder(**mono_model_configs[encoder_name])

    ckpt = ROOT / "pretrained" / f"depth_anything_v2_{encoder_name}.pth"
    state = torch.load(str(ckpt), map_location="cpu")

    model.load_state_dict(state, strict=True)
    decoder_model.load_state_dict(state, strict=False)

    mono_encoder = model.pretrained
    mono_decoder = model.depth_head
    feat_decoder = decoder_model.depth_head

    del model, decoder_model, state
    return mono_encoder, mono_decoder, feat_decoder

    
class UpsampleBlock(nn.Module):
    def __init__(self, c0, c1):
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(c0, c1, kernel_size=2, stride=2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.merge = nn.Sequential(
            nn.Conv2d(c1 * 2, c1, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c1, c1, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c1, c1, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x, skip):
        x = self.up(x)

        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(
                x,
                size=skip.shape[-2:],
                mode="bilinear",
                align_corners=False
            )

        x = torch.cat([x, skip], dim=1)

        return self.merge(x)

class ResBlock(nn.Module):
    def __init__(self, c0, dilation=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(c0, c0, 3, 1, dilation, dilation),
            nn.LeakyReLU(0.2),
            nn.Conv2d(c0, c0, 3, 1, dilation, dilation),
        )
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, input):
        x = self.conv(input)
        x = x + input
        x = self.relu(x)
        return x
def warp_and_aggregate(hyp, left, right):
    scale = left.size(3) // hyp.size(3)
   
    assert scale == 4

    d_expand = disp_up_hit(hyp[:, :1], hyp[:, 1:2], hyp[:, 2:3], scale, tile_expand=True)
    d_range = torch.arange(right.size(3), device=right.device)
    d_range = d_range.view(1, 1, 1, -1) - d_expand
    d_range = d_range.repeat(1, right.size(1), 1, 1)

    cost = [torch.sum(torch.abs(left), dim=1, keepdim=True)]
    for offset in [1, 0, -1]:
        index_float = d_range + offset
        index_long = torch.floor(index_float).long()
        index_left = torch.clip(index_long, min=0, max=right.size(3) - 1)
        index_right = torch.clip(index_long + 1, min=0, max=right.size(3) - 1)
        index_weight = index_float - index_left

        right_warp_left = torch.gather(right, dim=-1, index=index_left.long())
        right_warp_right = torch.gather(right, dim=-1, index=index_right.long())
        right_warp = right_warp_left + index_weight * (
            right_warp_right - right_warp_left
        )
        cost.append(torch.sum(torch.abs(left - right_warp), dim=1, keepdim=True))
    cost = torch.cat(cost, dim=1)

    n, c, h, w = cost.size()
    cost = cost.reshape(n, c, h // scale, scale, w // scale, scale)
    cost = cost.permute(0, 3, 5, 1, 2, 4)
    cost = cost.reshape(n, scale * scale * c, h // scale, w // scale)
    return cost

@functools.lru_cache()
@torch.no_grad()
def make_warp_coef(scale, device):
    center = (scale - 1) / 2
    index = torch.arange(scale, device=device) - center
    coef_y, coef_x = torch.meshgrid(index, index)
    coef_x = coef_x.reshape(1, -1, 1, 1)
    coef_y = coef_y.reshape(1, -1, 1, 1)
    return coef_x, coef_y


class SameConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        return same_padding_conv(x, self.weight, self.bias, self.stride)

def same_padding_conv(x, w, b, s):
    out_h = math.ceil(x.size(2) / s[0])
    out_w = math.ceil(x.size(3) / s[1])

    pad_h = max((out_h - 1) * s[0] + w.size(2) - x.size(2), 0)
    pad_w = max((out_w - 1) * s[1] + w.size(3) - x.size(3), 0)
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom))
    x = F.conv2d(x, w, b, stride=s)
    return x
def make_cost_volume_v2(left, right, max_disp):
    d_range = torch.arange(max_disp, device=left.device)
    d_range = d_range.view(1, 1, -1, 1, 1)

    x_index = torch.arange(left.size(3), device=left.device)
    x_index = x_index.view(1, 1, 1, 1, -1)

    x_index = torch.clip(4 * x_index - d_range + 1, 0, right.size(3) - 1).repeat(
        right.size(0), right.size(1), 1, right.size(2), 1
    )
    right = torch.gather(
        right.unsqueeze(2).repeat(1, 1, max_disp, 1, 1), dim=-1, index=x_index
    )

    return left.unsqueeze(2) - right


def disp_up_hit(d, dx, dy, scale, tile_expand):
    n, _, h, w = d.size()
    coef_x, coef_y = make_warp_coef(scale, d.device)

    if tile_expand:
        d = d + coef_x * dx + coef_y * dy
    else:
        d = d * scale + coef_x * dx * 4 + coef_y * dy * 4

    d = d.reshape(n, 1, scale, scale, h, w)
    d = d.permute(0, 1, 4, 2, 5, 3)
    d = d.reshape(n, 1, h * scale, w * scale)
    return d

def hyp_up(hyp, scale=1, tile_scale=1):
    if scale != 1:
        d = disp_up_hit(hyp[:, :1], hyp[:, 1:2], hyp[:, 2:3], scale, tile_expand=False)
        p = F.interpolate(hyp[:, 1:], scale_factor=scale)
        hyp = torch.cat((d, p), dim=1)
    if tile_scale != 1:
        d = disp_up_hit(hyp[:, :1], hyp[:, 1:2], hyp[:, 2:3], tile_scale, tile_expand=True)
        p = F.interpolate(hyp[:, 1:], scale_factor=tile_scale)
        hyp = torch.cat((d, p), dim=1)
    return hyp  

def disp_upsample_using_gradients(d, dx, dy, scale, tile_expand=True):
    n, _, h, w = d.size()
    coef_x, coef_y = make_warp_coef(scale, d.device)

    if tile_expand:
        d = d + coef_x * dx + coef_y * dy
    else:
        d = d * scale + coef_x * dx * 4 + coef_y * dy * 4

    d = d.reshape(n, 1, scale, scale, h, w)
    d = d.permute(0, 1, 4, 2, 5, 3)
    d = d.reshape(n, 1, h * scale, w * scale)
    return d

class RefinementNet_Disp_Next(nn.Module):
    def __init__(self, cin, cres, dilations):
        super().__init__()
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(cin + 3, cres, 1),
            nn.LeakyReLU(0.2),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(cres, cres, 3, 1, 1),
            nn.LeakyReLU(0.2),
        )
        self.res_block = []
        for d in dilations:
            self.res_block += [ResBlock(cres, d)]
        self.res_block = nn.Sequential(*self.res_block)
        self.convn = nn.Conv2d(cres, 3, 3, 1, 1)

    def forward(self, hpy, left):
        x = torch.cat((left, hpy), dim=1)
        x = self.conv1x1(x)
        x = self.conv1(x)
        x = self.res_block(x)
        x = self.convn(x)
        return hpy + x
    
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.block(x))
    
class Mono_Dip_predict(nn.Module):
    def __init__(self, img_feat_channels, disp_channels=1):
        super().__init__()
        in_channels = img_feat_channels + disp_channels

        self.input_conv = nn.Sequential(

            nn.Conv2d(in_channels, 16, kernel_size=1),

            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )

        self.res_blocks = nn.Sequential(
            ResidualBlock(16),
            ResidualBlock(16),
            ResidualBlock(16)

            
        )

        self.output_conv = nn.Conv2d(16, 1, kernel_size=3, padding=1)

    def forward(self, img_feats, stereo_disp):
        x = torch.cat([img_feats, stereo_disp], dim=1)

        x = self.input_conv(x)
        x = self.res_blocks(x)
        x=self.output_conv(x)
        return x
    

class MultiScaleFusion(nn.Module):
    def __init__(self, in_channels_list):
        super().__init__()
        self.blocks = nn.ModuleList()

        c_prev = in_channels_list[0]
        for c_skip in in_channels_list[1:]:
            c_in = c_prev + c_skip
            block = nn.Sequential(
                nn.Conv2d(c_in, c_in, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(c_in, c_in, 3, padding=1),
                nn.ReLU(inplace=True),
            )
            self.blocks.append(block)
            c_prev = c_in

        self.total_channels = c_prev

        self.final_conv = nn.Sequential(
            nn.Conv2d(self.total_channels, self.total_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.total_channels, self.total_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, feats, disp_size=None):
        x = feats[0]

        for i, f in enumerate(feats[1:]):
            x = F.interpolate(x, size=f.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, f], dim=1)

        return x
    
class DispFullResRefine(nn.Module):
    def __init__(self, in_ch=1, feat_ch=32, num_blocks=3):
        super().__init__()

        self.in_conv = nn.Sequential(
            nn.Conv2d(in_ch, feat_ch, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        blocks = []
        for _ in range(num_blocks):
            blocks.append(ResBlock(feat_ch))
        self.res_blocks = nn.Sequential(*blocks)

        self.out_conv = nn.Conv2d(feat_ch, 1, 3, padding=1)

    def forward(self, disp):
        x = self.in_conv(disp)
        x = self.res_blocks(x)
        residual = self.out_conv(x)

        return disp + residual
def get_cur_disp_range_samples(cur_disp, ndisp, disp_interval_pixel, shape):

    w = float(cur_disp.shape[2])
    cur_disp_min = (cur_disp - ndisp / 2 * disp_interval_pixel).clamp(min=0.0)
    cur_disp_max = (cur_disp_min + (ndisp - 1) * disp_interval_pixel).clamp(max=w)

    assert cur_disp.shape == torch.Size(shape), "cur_disp:{}, input shape:{}".format(cur_disp.shape, shape)
    new_interval = (cur_disp_max - cur_disp_min) / (ndisp - 1)

    disp_range_samples = cur_disp_min.unsqueeze(1) + (torch.arange(0, ndisp, device=cur_disp.device,
                                                                    dtype=cur_disp.dtype,
                                                                    requires_grad=False).reshape(1, -1, 1,
                                                                                                1) * new_interval.unsqueeze(1))
    return disp_range_samples
            
def compute_scale_shift(monocular_depth, gt_depth, mask=None):
    """
    计算 monocular depth 和 ground truth depth 之间的 scale 和 shift.
    
    参数:
    monocular_depth (torch.Tensor): 单目深度图，形状为 (H, W) 或 (N, H, W)
    gt_depth (torch.Tensor): ground truth 深度图，形状为 (H, W) 或 (N, H, W)
    mask (torch.Tensor, optional): 有效区域的掩码，形状为 (H, W) 或 (N, H, W)
    
    返回:
    scale (float): 计算得到的 scale
    shift (float): 计算得到的 shift
    """
    
    flattened_depth_maps = monocular_depth.clone().view(-1).contiguous()
    sorted_depth_maps, _ = torch.sort(flattened_depth_maps)
    percentile_10_index = int(0.2 * len(sorted_depth_maps))
    threshold_10_percent = sorted_depth_maps[percentile_10_index]

    if mask is None:
        mask = (gt_depth > 0) & (monocular_depth > 1e-2) & (monocular_depth > threshold_10_percent)
        
        
    monocular_depth_flat = monocular_depth[mask]
    gt_depth_flat = gt_depth[mask]
    
    X = torch.stack([monocular_depth_flat, torch.ones_like(monocular_depth_flat)], dim=1)
    y = gt_depth_flat
    
    A = torch.matmul(X.t(), X) + 1e-6 * torch.eye(2, device=X.device)
    b = torch.matmul(X.t(), y)
    
    
    if not torch.isfinite(A).all() or not torch.isfinite(b).all():
        print(
            "[WARNING compute_scale_shift] NaN/Inf detected.",
            "N =", monocular_depth_flat.numel()
        )
        return 1.0, 0.0
    
    
    try:
        params = torch.linalg.solve(A, b)

    except torch._C._LinAlgError:
        print(
            "[WARNING compute_scale_shift] Singular matrix.",
            "N =", monocular_depth_flat.numel(),
            "mono_min =", monocular_depth_flat.min().item(),
            "mono_max =", monocular_depth_flat.max().item(),
            "mono_std =", monocular_depth_flat.std().item(),
            "gt_min =", gt_depth_flat.min().item(),
            "gt_max =", gt_depth_flat.max().item(),
        )

        return 1.0, 0.0
    
    scale, shift = params[0].item(), params[1].item()
    
    
    if not (
        torch.isfinite(params).all()
    ):
        print("[WARNING compute_scale_shift] Invalid scale/shift.")
        return 1.0, 0.0

    
    return scale, shift


class hourglass_4x(nn.Module):
    def __init__(self, in_channels):
        super(hourglass_4x, self).__init__()

        self.conv1 = BasicConv(in_channels, in_channels, is_3d=True, bn=True, relu=True, kernel_size=3,
                                                padding=1, stride=1, dilation=1)
    def forward(self, x):
        conv = self.conv1(x)

        return conv
    
class hourglass_8x(nn.Module):
    def __init__(self, in_channels):
        super(hourglass_8x, self).__init__()

        self.conv1 = BasicConv(in_channels, in_channels, is_3d=True, bn=True, relu=True, kernel_size=3,
                                                padding=1, stride=1, dilation=1)
    def forward(self, x):

        conv = self.conv1(x)

        return conv


class hourglass_16x(nn.Module):
    def __init__(self, in_channels):
        super(hourglass_16x, self).__init__()

        self.conv1 = BasicConv(in_channels, in_channels, is_3d=True, bn=True, relu=True, kernel_size=3,
                                                padding=1, stride=1, dilation=1)

    def forward(self, x):

        conv = self.conv1(x)

        return conv    
    

class Feat_transfer_cnet(nn.Module):
    def __init__(self, output_dim):
        super().__init__()

        self.res_16x = nn.Conv2d(192 + 192, output_dim[0], kernel_size=3, padding=1)
        self.res_8x  = nn.Conv2d(64 + 96,   output_dim[1], kernel_size=3, padding=1)
        self.res_4x  = nn.Conv2d(48 + 48,   output_dim[2], kernel_size=3, padding=1)

    def forward(self, features, stem_x_list):
        feat_16x = self.res_16x(torch.cat((features[2], stem_x_list[0]), 1))
        feat_8x  = self.res_8x(torch.cat((features[1], stem_x_list[1]), 1))
        feat_4x  = self.res_4x(torch.cat((features[0], stem_x_list[2]), 1))

        return [
            [feat_4x, feat_4x],
            [feat_8x, feat_8x],
            [feat_16x, feat_16x],
        ]

class Feat_transfer(nn.Module):
    def __init__(self, dim_list):
        super(Feat_transfer, self).__init__()
        self.conv4x = nn.Sequential(
            nn.Conv2d(in_channels=int(48+dim_list[0]), out_channels=48, kernel_size=5, stride=1, padding=2),
            nn.InstanceNorm2d(48), nn.ReLU()
            )
        self.conv8x = nn.Sequential(
            nn.Conv2d(in_channels=int(64+dim_list[0]), out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.InstanceNorm2d(64), nn.ReLU()
            )
        self.conv16x = nn.Sequential(
            nn.Conv2d(in_channels=int(dim_list[0]), out_channels=192, kernel_size=5, stride=1, padding=2),
            nn.InstanceNorm2d(192), nn.ReLU()
            )

        self.conv_up_16x = nn.ConvTranspose2d(192,
                                64,
                                kernel_size=3,
                                padding=1,
                                output_padding=1,
                                stride=2,
                                bias=False)
        self.conv_up_8x = nn.ConvTranspose2d(64,
                                48,
                                kernel_size=3,
                                padding=1,
                                output_padding=1,
                                stride=2,
                                bias=False)
        
        self.res_16x = nn.Conv2d(dim_list[0], 192, kernel_size=1, padding=0, stride=1)
        self.res_8x = nn.Conv2d(dim_list[0], 64, kernel_size=1, padding=0, stride=1)
        self.res_4x = nn.Conv2d(dim_list[0], 48, kernel_size=1, padding=0, stride=1)


    def forward(self, features):
        features_mono_list = []

        feat_16x = self.conv16x(features[2]) + self.res_16x(features[2])
        feat_16x_up = self.conv_up_16x(feat_16x)
        feat_8x = self.conv8x(torch.cat((features[1], feat_16x_up), 1)) + self.res_8x(features[1])
        feat_8x_up = self.conv_up_8x(feat_8x)
        feat_4x = self.conv4x(torch.cat((features[0], feat_8x_up), 1)) + self.res_4x(features[0])
        features_mono_list.append(feat_4x)
        features_mono_list.append(feat_8x)
        features_mono_list.append(feat_16x)

        return features_mono_list


class UncertaintyPredictor(nn.Module):
    def __init__(self, img_feat_channels, disp_channels=1):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(img_feat_channels + disp_channels, 32, 3, padding=1),
            
            nn.ReLU(inplace=True),

            ResidualBlock(32),
            ResidualBlock(32),

            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 1, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, img_feats, disp):
        x = torch.cat([img_feats, disp], dim=1)
        return self.net(x)    

    
class DispPixelMLP(nn.Module):
    def __init__(self, feat_ch, hidden_ch=64):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(feat_ch + 1, hidden_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_ch, hidden_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.mlp = nn.Sequential(
            nn.Conv2d(hidden_ch, hidden_ch, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_ch, 1, 1)
        )

    def forward(self, feat, disp_next):
        """
        feat      : [B, C, H, W]
        disp_next : [B, 1, H, W]

        return:
            out_map  : [B, 1, H, W]
            out_flat : [B, 1, H*W]
        """

        if disp_next.shape[-2:] != feat.shape[-2:]:
            print("SHAPE MISMATHC")
            disp_next = F.interpolate(
                disp_next,
                size=feat.shape[-2:],
                mode="bilinear",
                align_corners=False
            )

        x = torch.cat([feat, disp_next], dim=1)

        x = self.cnn(x)
        Delta_O = self.mlp(x)


        return Delta_O    
    
    
# Stereo model with Stage 2 teacher-assistant distillation.
class URS(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.ndisps = [15, 13]
        self.disp_interval_pixel = [2, 1]
        context_dims = args.hidden_dims[::-1]
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        self.normalize = transforms.Compose([
            transforms.Normalize(mean=mean, std=std) 
        ])

        self.intermediate_layer_idx = {
            'vits': [2, 5, 8, 11],
            'vitb': [2, 5, 8, 11], 
            'vitl': [4, 11, 17, 23], 
            'vitg': [9, 19, 29, 39]
        }
        mono_model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
        dim_list_ = mono_model_configs[self.args.encoder_s]['features']
        dim_list = []
        dim_list.append(dim_list_)

        self.update_block_list = nn.ModuleList([BasicMultiUpdateBlock_no_mono_disp(self.args, hidden_dims=args.hidden_dims, ndisps=self.ndisps), BasicMultiUpdateBlock_no_mono_disp(self.args, hidden_dims=args.hidden_dims, ndisps=self.ndisps), BasicMultiUpdateBlock_no_mono_disp(self.args, hidden_dims=args.hidden_dims, ndisps=self.ndisps)])

        self.context_zqr_convs = nn.ModuleList([nn.Conv2d(context_dims[i], context_dims[i]*3, 3, padding=3//2) for i in range(self.args.n_gru_layers)])

        
        self.feat_transfer = Feat_transfer(dim_list)
       
        self.feat_transfer_cnet = Feat_transfer_cnet(output_dim=args.hidden_dims)


        self.stem_2 = nn.Sequential(
            BasicConv_IN(3, 32, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(32), nn.ReLU()
            )
        self.stem_4 = nn.Sequential(
            BasicConv_IN(32, 48, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(48, 48, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(48), nn.ReLU()
            )

        self.stem_8 = nn.Sequential(
            BasicConv_IN(48, 96, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(96, 96, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(96), nn.ReLU()
            )

        self.stem_16 = nn.Sequential(
            BasicConv_IN(96, 192, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(192, 192, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(192), nn.ReLU()
            )

        self.spx = nn.Sequential(nn.ConvTranspose2d(2*32, 9, kernel_size=4, stride=2, padding=1),)
        self.spx_2 = Conv2x_IN(24, 32, True)
        self.spx_4 = nn.Sequential(
            BasicConv_IN(96, 24, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(24, 24, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(24), nn.ReLU()
            )
        self.spx_16 = nn.ModuleList([
            BasicConv_IN(192, 96, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(96, 96, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(96), nn.ReLU()
            ])

        self.spx_2_gru = Conv2x(32, 32, True)
        self.spx_gru = nn.Sequential(nn.ConvTranspose2d(2*32, 9, kernel_size=4, stride=2, padding=1),)
        self.spx_gru_2x = nn.ModuleList([Conv2x(32, 96, True, keep_concat=False), Conv2x(32, 48, True, keep_concat=False)])
        self.spx_gru_list = nn.ModuleList([nn.Conv2d(96, 9, kernel_size=1, stride=1, padding=0), nn.Conv2d(48, 9, kernel_size=1, stride=1, padding=0)])


        self.conv_list = nn.ModuleList([BasicConv_IN(192+192, 96, kernel_size=3, padding=1, stride=1),
                                        BasicConv_IN(64+96, 96, kernel_size=3, padding=1, stride=1),
                                        BasicConv_IN(48+48, 96, kernel_size=3, padding=1, stride=1)])
        self.desc = nn.Conv2d(96, 96, kernel_size=1, padding=0, stride=1)

        self.corr_stem = BasicConv(8, 8, is_3d=True, kernel_size=3, stride=1, padding=1)
        self.corr_feature_att_list = nn.ModuleList([FeatureAtt(8, 192+192), FeatureAtt(8, 64+96), FeatureAtt(8, 96)])
        self.cost_agg = nn.ModuleList([hourglass_16x(8), hourglass_8x(8), hourglass_4x(8)])
        self.classifier = nn.Conv3d(8, 1, 3, 1, 1, bias=False)


        # Load the pretrained teacher and student branches.
        self.teacher_encoder, self.teacher_mono_decoder, self.teacher_feat_decoder = \
            load_depth_anything_branch(args.teacher_encoder, mono_model_configs, ROOT)


        self.mono_encoder, self.mono_decoder, self.feat_decoder = \
            load_depth_anything_branch(args.student_encoder, mono_model_configs, ROOT)


        self.update_block_mix_stereo = BasicMultiUpdateBlock_mix2(self.args, hidden_dims=args.hidden_dims, ndisps=self.ndisps)
        self.update_block_mix_mono = BasicMultiUpdateBlock_mix2(self.args, hidden_dims=args.hidden_dims, ndisps=self.ndisps)
        self.spx_gru_2x_new = nn.ModuleList([
                    Conv2x(32, 32, True, keep_concat=False),
                    Conv2x(32, 32, True, keep_concat=False),
                Conv2x(32, 32, True, keep_concat=False)
                ])
        self.spx_gru_list_new = nn.ModuleList([
            nn.ConvTranspose2d(32, 9, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(32, 9, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(32, 9, kernel_size=4, stride=2, padding=1)
        ])


        self.mono_disparity_predictor= Mono_Dip_predict(
            img_feat_channels=24,
            disp_channels=1
        )


        self.uncertainty_predictor8 = UncertaintyPredictor(
            img_feat_channels=64,
            disp_channels=1
        )

        self.uncertainty_predictor4 = UncertaintyPredictor(
            img_feat_channels=48,
            disp_channels=1
        )

        self.disp_pixel_mlp_8x = DispPixelMLP(feat_ch=64)
        self.disp_pixel_mlp_4x = DispPixelMLP(feat_ch=48)
        

        # Reuse the Stage 1 teacher assistant as a frozen distillation target.
        self.intermediate_encoder = IntermediateCrossAttentionEncoder(
            num_layers=4,
            student_dim=384,
            teacher_dim=1024,
            fusion_dim=704,
            num_heads=11,
            ffn_ratio=4.0,
            dropout=0.0,
        )

        self.intermediate_mono_decoder = IntermediateDPTHead(
            in_channels=704,
            features=64,
            out_channels=(48, 96, 192, 384),
            use_bn=False,
            use_clstoken=True,
        )

        self.intermediate_feat_decoder = IntermediateDPTFeatureHead(
            in_channels=704,
            features=64,
            out_channels=(48, 96, 192, 384),
            use_bn=False,
            use_clstoken=True,
        )

        self.student_to_ta_projection = StudentToTAProjection(
            num_layers=4,
            student_dim=384,
            ta_dim=704,
        )
        

        # Stage 2: freeze teacher/assistant; train the selected student modules.

        self.teacher_encoder.requires_grad_(False)
        self.teacher_mono_decoder.requires_grad_(False)
        self.teacher_feat_decoder.requires_grad_(False)

        self.intermediate_encoder.requires_grad_(False)
        self.intermediate_mono_decoder.requires_grad_(False)
        self.intermediate_feat_decoder.requires_grad_(False)

        self.mono_encoder.requires_grad_(True)

        self.student_to_ta_projection.requires_grad_(True)

        self.mono_decoder.requires_grad_(False)
        self.feat_decoder.requires_grad_(True)

    # Extract stereo, monocular, and feature-distillation outputs.
    def infer_intermediate_branch(
        self,
        image1,
        image2,
    ):
        batch_size, _, height_ori, width_ori = image1.shape

        image = torch.cat(
            [image1, image2],
            dim=0,
        )


        image_half = F.interpolate(
        image,
        size=(width_ori // 2, width_ori // 2),
        mode="bilinear",
        align_corners=True,
        )

        image_half = (
        image_half / 255.0
        ).contiguous()

        image_half = self.normalize(image_half)

        resized_image = F.interpolate(
        image_half,
        scale_factor=14 / 16,
        mode="bilinear",
        align_corners=True,
        )
        patch_h = resized_image.shape[-2] // 14
        patch_w = resized_image.shape[-1] // 14

        student_features = self.mono_encoder.get_intermediate_layers(
            resized_image,
            self.intermediate_layer_idx[
                self.args.student_encoder
            ],
            return_class_token=True,
        )

        with torch.no_grad():
            teacher_features = (
                self.teacher_encoder.get_intermediate_layers(
                    resized_image,
                    self.intermediate_layer_idx[
                        self.args.teacher_encoder
                    ],
                    return_class_token=True,
                )
            )

        # Detach student tokens to prevent gradients through the TA target branch.
        student_features_detached = [
            (
                patch_tokens.detach(),
                cls_token.detach(),
            )
            for patch_tokens, cls_token in student_features
        ]

        # Compute teacher-assistant features without gradients.
        with torch.no_grad():
            ta_features = self.intermediate_encoder(
                student_features_detached,
                teacher_features,
            )


        depth_mono = self.mono_decoder(
                student_features,
                patch_h,
                patch_w,
            )
        decoder_features = self.feat_decoder(
            student_features,
            patch_h,
            patch_w,
        )


        # Learn the projection from student tokens to TA feature dimensions.
        projected_student_features = (
            self.student_to_ta_projection(
                student_features
            )
        )

        # Detach TA patch tokens used as feature-distillation targets.
        ta_patch_features = [
            patch_tokens.detach()
            for patch_tokens, cls_token in ta_features
        ]

        depth_mono = F.relu(
            depth_mono
        )

        depth_mono = F.interpolate(
            depth_mono,
            size=(
                height_ori // 2,
                width_ori // 2,
            ),
            mode="bilinear",
            align_corners=False,
        )

        (
            features_4x,
            features_8x,
            features_16x,
            features_32x,
        ) = decoder_features

        features_left = [
            F.interpolate(
                features_4x[:batch_size],
                size=(
                    height_ori // 4,
                    width_ori // 4,
                ),
                mode="bilinear",
                align_corners=False,
            ),
            F.interpolate(
                features_8x[:batch_size],
                size=(
                    height_ori // 8,
                    width_ori // 8,
                ),
                mode="bilinear",
                align_corners=False,
            ),
            F.interpolate(
                features_16x[:batch_size],
                size=(
                    height_ori // 16,
                    width_ori // 16,
                ),
                mode="bilinear",
                align_corners=False,
            ),
        ]

        features_right = [
            F.interpolate(
                features_4x[batch_size:],
                size=(
                    height_ori // 4,
                    width_ori // 4,
                ),
                mode="bilinear",
                align_corners=False,
            ),
            F.interpolate(
                features_8x[batch_size:],
                size=(
                    height_ori // 8,
                    width_ori // 8,
                ),
                mode="bilinear",
                align_corners=False,
            ),
            F.interpolate(
                features_16x[batch_size:],
                size=(
                    height_ori // 16,
                    width_ori // 16,
                ),
                mode="bilinear",
                align_corners=False,
            ),
        ]

        intermediate_patch_features = [
            patch_tokens
            for patch_tokens, cls_token in ta_features
        ]

        intermediate_features_left = [
            feature[:batch_size]
            for feature in intermediate_patch_features
        ]

        intermediate_features_right = [
            feature[batch_size:]
            for feature in intermediate_patch_features
        ]
        student_features_left = [
            feature[:batch_size]
            for feature in projected_student_features
        ]

        student_features_right = [
            feature[batch_size:]
            for feature in projected_student_features
        ]

        ta_features_left = [
            feature[:batch_size]
            for feature in ta_patch_features
        ]

        ta_features_right = [
            feature[batch_size:]
            for feature in ta_patch_features
        ]
        depth_left = depth_mono[:batch_size]
        depth_right = depth_mono[batch_size:]

        return (
            depth_left,
            depth_right,
            features_left,
            features_right,
            intermediate_features_left,
            intermediate_features_right,
            student_features_left,
            student_features_right,
            ta_features_left,
            ta_features_right,

        )


    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
            if isinstance(m, nn.SyncBatchNorm):
                m.eval()

    # Convex disparity upsampling from the recurrent update features.
    def upsample_disp(self, disp, mask_feat_4, stem_2x):

        xspx = self.spx_2_gru(mask_feat_4, stem_2x)
        spx_pred = self.spx_gru(xspx)
        spx_pred = F.softmax(spx_pred, 1)
        up_disp = context_upsample(disp*4., spx_pred).unsqueeze(1)

        return up_disp
    def upsample_disp_2x_to_full(self, disp, mask_feat, stem, i):

        xspx = self.spx_gru_2x_new[i](mask_feat, stem)
        spx_pred = self.spx_gru_list_new[i](xspx)
        spx_pred = F.softmax(spx_pred, dim=1)

        up_disp = context_upsample_2x(disp * 2., spx_pred).unsqueeze(1)

        return up_disp

    def upsample_disp_x2(self, disp, mask_feat, stem, i):


        xspx = self.spx_gru_2x[i](mask_feat, stem)
        spx_pred = self.spx_gru_list[i](xspx)
        spx_pred = F.softmax(spx_pred, 1)
        up_disp = context_upsample_2x(disp*2., spx_pred).unsqueeze(1)

        return up_disp

    # Coarse-to-fine stereo forward pass.
    def forward(self, image1, image2, iters=4,  test_mode=False):
        """ Estimate disparity between pair of frames """
        

        n, c, h, w = image1.size()
        with torch.autocast(device_type="cuda", dtype=torch.float32):
                (
                    depth_mono,
                    depth_mono_right,
                    features_mono_left,
                    features_mono_right,
                    intermediate_features_left,
                    intermediate_features_right,
                    student_features_left,
                    student_features_right,
                    ta_features_left,
                    ta_features_right,
                ) = self.infer_intermediate_branch(
                    image1,
                    image2,
                )

            
        size = (image1.shape[-2], image1.shape[-1])
        disp_mono_4x = F.interpolate(depth_mono, size=tuple(x // 4 for x in size), mode='bilinear', align_corners=False)
        disp_mono_8x = F.interpolate(depth_mono, size=tuple(x // 8 for x in size), mode='bilinear', align_corners=False)
        disp_mono_16x = F.interpolate(depth_mono, size=tuple(x // 16 for x in size), mode='bilinear', align_corners=False)
        disp_mono_list = [disp_mono_16x, disp_mono_8x, disp_mono_4x]
        
        features_left = self.feat_transfer(features_mono_left)
        features_right = self.feat_transfer(features_mono_right)
        
        
        feat_left_4 = features_left[0]
        feat_left_8 = features_left[1]
        
        
        image = torch.cat([image1,image2], 0)
        stem_2x = self.stem_2(image)
        stem_4x = self.stem_4(stem_2x)
        stem_8x = self.stem_8(stem_4x)
        stem_16x = self.stem_16(stem_8x)
        stem_2x, stem_2y = torch.chunk(stem_2x, 2, dim=0)
        stem_4x, stem_4y = torch.chunk(stem_4x, 2, dim=0)
        stem_8x, stem_8y = torch.chunk(stem_8x, 2, dim=0)
        stem_16x, stem_16y = torch.chunk(stem_16x, 2, dim=0)


        stem_x_list = [stem_16x, stem_8x, stem_4x, stem_2x]
        stem_y_list = [stem_16y, stem_8y, stem_4y, stem_2y]
        if not test_mode:
            iters_list = [iters//4, iters//4, iters//2]
        else:
            iters_list = [1,1,2]


        cnet_list = self.feat_transfer_cnet(features_left, stem_x_list)

        net_list = [torch.tanh(x[0]) for x in cnet_list]
        inp_list = [torch.relu(x[1]) for x in cnet_list]
        inp_list = [torch.relu(x) for x in inp_list]
        inp_list = [list(conv(i).split(split_size=conv.out_channels//3, dim=1)) for i,conv in zip(inp_list, self.context_zqr_convs)]
        net_list_mono = [x.clone() for x in net_list]
        
        disp_preds = []
               
                
        predicted_uncertainty8_list = []
        predicted_uncertainty4_list=[]
        
         
        # Disparity refinement proceeds from 1/16 to 1/8 to 1/4 resolution.
        for i in range(3):

            feat_left = torch.cat((features_left[-1-i], stem_x_list[i]), 1)
            feat_right = torch.cat((features_right[-1-i], stem_y_list[i]), 1)

            match_left = self.desc(self.conv_list[i](feat_left))
            match_right = self.desc(self.conv_list[i](feat_right))


            # Initialize disparity using the coarse full cost volume.
            if i==0:
                disp_range_samples = None
                gwc_volume = build_gwc_volume(match_left, match_right, self.args.max_disp//16, 8)
                gwc_volume = self.corr_stem(gwc_volume)
                gwc_volume = self.corr_feature_att_list[i](gwc_volume, feat_left)
                geo_encoding_volume = self.cost_agg[i](gwc_volume)  

                prob = F.softmax(self.classifier(geo_encoding_volume).squeeze(1), dim=1)
                init_disp = disparity_regression(prob, self.args.max_disp//16)
                cur_disp = init_disp
                del prob

            # Refine using shifted local cost volumes at the finer scales.
            else:
                
                if i==1:
                    feat_left_i=feat_left_8
                    

                    predicted_uncertainty_8 = self.uncertainty_predictor8(feat_left_8, disp_next)

                    predicted_uncertainty8_list.append(predicted_uncertainty_8)
                      
                    Delta_O = self.disp_pixel_mlp_8x(feat_left_i, disp_next)
                    Delta_O = predicted_uncertainty_8 * Delta_O
                    disp_center_8x = disp_next + Delta_O
                    disp_center = disp_next + Delta_O
                    disp_range_samples = get_cur_disp_range_samples(cur_disp=disp_center.squeeze(1),
                                                    ndisp=self.ndisps[i-1],
                                                    disp_interval_pixel=self.disp_interval_pixel[i-1],
                                                    shape=[match_left.shape[0], match_left.shape[2], match_left.shape[3]]) 

                    gwc_volume = build_gwc_volume_selective(match_left, match_right, disp_range_samples, self.ndisps[i-1], 8)  
                    
                elif i==2:
                    feat_left_i=feat_left_4

                    
                    predicted_uncertainty_4 = self.uncertainty_predictor4(feat_left_4, disp_next)

                    predicted_uncertainty4_list.append(predicted_uncertainty_4)
              
                    
                    Delta_O = self.disp_pixel_mlp_4x(feat_left_i, disp_next)
                    Delta_O = predicted_uncertainty_4 * Delta_O
                    disp_center_4x = disp_next + Delta_O

                    disp_center = disp_next + Delta_O
                    disp_range_samples = get_cur_disp_range_samples(cur_disp=disp_center.squeeze(1),
                                ndisp=self.ndisps[i-1],
                                disp_interval_pixel=self.disp_interval_pixel[i-1],
                                shape=[match_left.shape[0], match_left.shape[2], match_left.shape[3]]) 

                    gwc_volume = build_gwc_volume_selective(match_left, match_right, disp_range_samples, self.ndisps[i-1], 8)
                    

                gwc_volume = self.corr_stem(gwc_volume)
                gwc_volume = self.corr_feature_att_list[i](gwc_volume, feat_left)
                geo_encoding_volume = self.cost_agg[i](gwc_volume)  
                cur_disp = disp_next  
            del gwc_volume

            geo_block = Combined_Geo_Encoding_Volume
            geo_fn = geo_block(match_left.float(), match_right.float(), geo_encoding_volume.float(), radius=self.args.corr_radius[i])
            b, c, h, w = match_left.shape
            coords = torch.arange(w).float().to(match_left.device).reshape(1,1,w,1).repeat(b, h, 1, 1).contiguous()
            disp = cur_disp


            for itr in range(iters_list[i]):
                disp = disp.detach()

                if i==0:
                    geo_feat = geo_fn(disp, coords)
                else:
                    geo_feat = geo_fn(disp, coords, self.ndisps[i-1])

                bs, _, _, _ = disp.shape
                for b in range(bs):
                    with torch.autocast(device_type='cuda', dtype=torch.float32): 
                        scale, shift = compute_scale_shift(disp_mono_list[i][b].clone().squeeze(1).to(torch.float32), disp[b].clone().squeeze(1).to(torch.float32))
                    disp_mono_list[i][b] = scale * disp_mono_list[i][b] + shift

                if i==2:
                    if itr < int(iters_list[2]-1):
                        net_list, mask_feat, delta_disp = self.update_block_list[i](net_list, inp_list, geo_feat, disp,iter16=(self.args.n_gru_layers==3 and i>=0), iter08=(self.args.n_gru_layers>=2 and i>=1), iter04=(self.args.n_gru_layers>=1 and i>=2))
                        disp_4=disp+delta_disp
    

                    # Apply mono-stereo mutual refinement in the final 1/4-scale iteration.
                    else:
                        warped_right_mono = disp_warp(feat_right, disp_mono_list[i].clone().to(feat_right.dtype))[0]
                        flaw_mono = warped_right_mono - feat_left

                        warped_right_stereo = disp_warp(feat_right, disp.clone().to(feat_right.dtype))[0]  
                        flaw_stereo = warped_right_stereo - feat_left 
                        geo_feat_mono = geo_fn(disp_mono_list[i], coords, self.ndisps[i-1])  

                        net_list, mask_feat, delta_disp = self.update_block_mix_stereo(net_list, inp_list, flaw_stereo, disp, geo_feat, flaw_mono, disp_mono_list[i], geo_feat_mono, iter16=False, iter08=False)
                        net_list_mono, mask_feat_4_mono, delta_disp_mono = self.update_block_mix_mono(net_list_mono, inp_list, flaw_mono, disp_mono_list[i], geo_feat_mono, flaw_stereo, disp, geo_feat, iter16=False, iter08=False)

                        disp_mono_list[i] = disp_mono_list[i] + delta_disp_mono
                        disp_mono_4x_up = self.upsample_disp(disp_mono_list[i], mask_feat_4_mono, stem_2x)
                        disp_preds.append(disp_mono_4x_up)

                if i==0:
                    net_list, mask_feat, delta_disp = self.update_block_list[i](net_list, inp_list, geo_feat, disp, iter16=(self.args.n_gru_layers==3 and i>=0), iter08=(self.args.n_gru_layers>=2 and i>=1), iter04=(self.args.n_gru_layers>=1 and i>=2)) 
                    
                    
                if i==1:
                    net_list, mask_feat, delta_disp = self.update_block_list[i](net_list, inp_list, geo_feat, disp, iter16=(self.args.n_gru_layers==3 and i>=0), iter08=(self.args.n_gru_layers>=2 and i>=1), iter04=(self.args.n_gru_layers>=1 and i>=2))
                    disp_8=disp+delta_disp
                    
             
                disp = disp + delta_disp

                if test_mode and itr < iters_list[2]-1:
                    continue
                if i < 2: 
                    disp_up = F.interpolate(disp * (2**(4-i)), size=size, mode='bilinear', align_corners=True)
                    disp_preds.append(disp_up)
                else:
                    disp_up = self.upsample_disp(disp, mask_feat, stem_2x)

           
                    disp_preds.append(disp_up)

            disp_next = self.upsample_disp_x2(disp, mask_feat, stem_x_list[i+1], i) if i < 2 else None


        if test_mode:
            return disp_up
        
        init_disp_up = F.interpolate(init_disp * 16, size=size, mode='bilinear', align_corners=True)
        disp_center_8x = F.interpolate(disp_center_8x * 8, size=size, mode='bilinear', align_corners=True)
        disp_center_4x = F.interpolate(disp_center_4x * 4, size=size, mode='bilinear', align_corners=True)
        

        return (
            init_disp_up,
            disp_preds,
            depth_mono,
            
            predicted_uncertainty8_list,
            predicted_uncertainty4_list,
            disp_8,
            disp_4,
            disp_center_4x,
            disp_center_8x,
            student_features_left,
            student_features_right,
            ta_features_left,
            ta_features_right,


        )
    


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import ToTensor, Lambda, Compose
import numpy as np
from einops import rearrange, repeat
from timm.models.layers import DropPath
from functools import partial

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

# PointWiseConv: 1x1 convolution with BatchNorm and PReLU
class PointWiseConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.pointwise = nn.Conv2d(in_c, out_c, kernel_size=1, stride=1, padding=0)
        self.bn = nn.BatchNorm2d(out_c)
        self.prelu = nn.PReLU(out_c)

    def forward(self, x):
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.prelu(x)
        return x

# MDConv: Mixed Depthwise Convolution with multiple kernel sizes
class MDConv(nn.Module):
    def __init__(self, channels, kernel_size, split_out_channels, stride):
        super(MDConv, self).__init__()
        self.num_groups = len(kernel_size)
        self.split_channels = split_out_channels
        self.mixed_depthwise_conv = nn.ModuleList()
        for i in range(self.num_groups):
            self.mixed_depthwise_conv.append(nn.Conv2d(
                self.split_channels[i],
                self.split_channels[i],
                kernel_size[i],
                stride=stride,
                padding=kernel_size[i]//2,
                groups=self.split_channels[i],
                bias=False
            ))
        self.bn = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU(channels)

    def forward(self, x):
        if self.num_groups == 1:
            return self.mixed_depthwise_conv[0](x)
        x_split = torch.split(x, self.split_channels, dim=1)
        x = [conv(t) for conv, t in zip(self.mixed_depthwise_conv, x_split)]
        x = torch.cat(x, dim=1)
        x = self.bn(x)
        x = self.prelu(x)
        return x

# MixConvBlock: Combines PointWiseConv and MDConv
class MixConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernels, splits):
        super().__init__()
        self.pw1 = PointWiseConv(in_ch, out_ch)
        self.dw = MDConv(out_ch, kernels, splits, stride=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pw1(x)
        x = self.dw(x)
        x = self.relu(x)
        return x

# InitialFeatureExtractor: Extracts features from input image
class InitialFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = MixConvBlock(3, 64, [3,5,7,9], [16,16,16,16])
        self.block2 = MixConvBlock(64, 64, [3,5,7], [32,16,16])
        self.block3 = MixConvBlock(64, 64, [3,5], [32,32])
        self.block4 = MixConvBlock(64, 128, [3,5], [64,64])
        self.maxpool = nn.MaxPool2d(2,2)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.maxpool(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.maxpool(x)
        return x

# CoordAtt: Coordinate Attention module
class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.ReLU()
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        out = identity * a_h * a_w
        return out

# GCAM: Global Context-Aware Module
class GCAM(nn.Module):
    def __init__(self, in_ch=128, num_classes=8):
        super().__init__()
        self.block1 = MixConvBlock(128, 256, [3,5,7,9], [64,64,64,64])
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, groups=1, bias=False)
        self.bn2 = nn.BatchNorm2d(256)
        self.prelu2 = nn.PReLU(256)
        self.coordatt = CoordAtt(256, 256)
        self.block2 = MixConvBlock(256, 512, [3,5], [256,256])
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.prelu2(self.bn2(self.conv2(x)))
        x = self.coordatt(x)
        x = self.block2(x)
        x = self.gap(x).view(x.size(0), -1)
        out = self.fc(x)
        return out

# PatchSplit2x2: Splits feature map into 4 patches
class PatchSplit2x2(nn.Module):
    def forward(self, x):
        B, C, H, W = x.shape
        h_half, w_half = H // 2, W // 2
        tl = x[:, :, :h_half, :w_half]
        tr = x[:, :, :h_half, w_half:]
        bl = x[:, :, h_half:, :w_half]
        br = x[:, :, h_half:, w_half:]
        return torch.cat([tl, tr, bl, br], dim=0)

# LayerNorm2d: Layer normalization for 2D feature maps
class LayerNorm2d(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2)

# Mlp: Multi-layer Perceptron for feature transformation
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0., channels_first=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.channels_first = channels_first
        if channels_first:
            self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1)
            self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1)
        else:
            self.fc1 = nn.Linear(in_features, hidden_features)
            self.fc2 = nn.Linear(hidden_features, out_features)
        self.act = act_layer()
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# SS2D: Selective Scan 2D module
class SS2D(nn.Module):
    def __init__(self, dim, expansion=2):
        super().__init__()
        hidden = dim * expansion
        self.proj_in = nn.Conv2d(dim, hidden, kernel_size=1)
        self.dwconv = nn.Conv2d(hidden, hidden, kernel_size=3, stride=1, padding=1, groups=hidden)
        self.act = nn.GELU()
        self.proj_out = nn.Conv2d(hidden, dim, kernel_size=1)

    def forward(self, x):
        residual = x
        x = self.proj_in(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.proj_out(x)
        return x + residual

# LocalBranch: Local feature extraction with depthwise and pointwise convolution
class LocalBranch(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim),
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )

    def forward(self, x):
        return self.conv(x)

# FER_VSS_Block: Feature Extraction and Selective Scan Block
class FER_VSS_Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.local_branch = LocalBranch(dim)
        self.global_branch = SS2D(dim)
        self.alpha = nn.Parameter(torch.ones(1))
        self.norm2 = nn.BatchNorm2d(dim)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, dim * 4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim * 4, dim, kernel_size=1)
        )

    def forward(self, x):
        x1 = self.norm1(x)
        local_feat = self.local_branch(x1)
        global_feat = self.global_branch(x1)
        mixed = self.alpha * global_feat + (1 - self.alpha) * local_feat
        x = x + mixed
        x2 = self.norm2(x)
        x = x + self.mlp(x2)
        return x

# HFIM: Hierarchical Feature Integration Module
class HFIM(nn.Module):
    def __init__(self, in_channels=128, num_regions=4, dropout=0.5, dropout2=0.6, eps=1e-6):
        super().__init__()
        self.C = in_channels
        self.R = num_regions
        self.eps = eps
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.alpha = nn.Sequential(
            nn.Linear(self.C, 1),
            nn.Sigmoid()
        )
        self.beta = nn.Sequential(
            nn.Linear(self.C * 2, 1),
            nn.Sigmoid()
        )
        self.classifi = nn.Linear(self.C * 2, 8)
        self.dropout = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout2)

    def forward(self, regions):
        if isinstance(regions, list) or isinstance(regions, tuple):
            regions = torch.stack(regions, dim=1)
        B, R, C, H, W = regions.shape
        assert R == self.R and C == self.C, f"expected R={self.R}, C={self.C}, got {R},{C}"
        vs = []
        alphas = []
        for r in range(R):
            f = regions[:, r, :, :, :]
            v = self.gap(f).view(B, C)
            vs.append(v)
            a = self.alpha(self.dropout(v))
            alphas.append(a)
        vs_stack = torch.stack(vs, dim=2)
        alphas_stack = torch.stack(alphas, dim=2)
        weighted = vs_stack * alphas_stack
        num = weighted.sum(dim=2)
        denom = alphas_stack.sum(dim=2)
        vm1 = num.div(denom + self.eps)
        betas = []
        for r in range(R):
            vi = vs[r]
            cat = torch.cat([vi, vm1], dim=1)
            b = self.beta(self.dropout(cat))
            betas.append(b)
            vs[r] = cat
        cascadeVs_stack = torch.stack(vs, dim=2)
        betas_stack = torch.stack(betas, dim=2)
        weights = betas_stack * alphas_stack
        weighted2 = cascadeVs_stack * weights
        num2 = weighted2.sum(dim=2)
        denom2 = weights.sum(dim=2)
        output = num2.div(denom2 + self.eps)
        output = self.dropout2(output)
        output = self.classifi(output)
        return output

# Full Mamba Model
class full_mamba(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()
        self.initExtractor = InitialFeatureExtractor()
        self.gcam = GCAM(in_ch=128, num_classes=num_classes)
        self.batch = PatchSplit2x2()
        self.hfim = HFIM(in_channels=128, num_regions=4)
        self.vss_block1 = FER_VSS_Block(128)
        self.vss_block2 = FER_VSS_Block(128)
        self.vss_block3 = FER_VSS_Block(128)
        self.vss_block4 = FER_VSS_Block(128)

    def forward(self, x):
        x_feat = self.initExtractor(x)
        x1 = self.gcam(x_feat)
        patches = self.batch(x_feat)
        x2, x3, x4, x5 = patches.chunk(4, dim=0)
        x2 = self.vss_block1(x2)
        x3 = self.vss_block2(x3)
        x4 = self.vss_block3(x4)
        x5 = self.vss_block4(x5)
        x6 = self.hfim([x2, x3, x4, x5])
        x_out = x1 + x6
        return x_out


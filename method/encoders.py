import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional


def _norm_layer(kind : str, num_channels : int) -> nn.Module :
    """
    Returns different normalization layers based on `kind`.
      - "instance" : nn.InstanceNorm2d(affine=True)
      - "batch"    : nn.BatchNorm2d
      - "none"     : nn.Identity (no normalization) 
    """
    kind = kind.lower()
    if kind == 'instance' :
        return nn.InstanceNorm2d(num_channels, affine= True)
    elif kind == 'batch' : 
        return nn.BatchNorm2d(num_channels)
    elif kind == 'none' : 
        return nn.Identity()
    else : 
        return ValueError(f"Unsupported norm kind: {kind!r}")
    


class L2Norm(nn.Module):
    def forward(self, x):  # channel-wise unit vectors
        return F.normalize(x, p=2, dim=1, eps=1e-6)



class AddCoords(nn.Module):
    
    def __init__(self):
        super().__init__()
        pass

    def forward(self,x):
        b,c,h,w = x.shape
        yy = torch.linspace(-1,1,steps = h,device=x.device).view(1,1,h,1).expand(b,1,h,w)
        xx = torch.linspace(-1,1,steps=w,device=x.device).view(1,1,1,w).expand(b,1,h,w)
        return torch.cat([x,xx,yy])
    


class ResidualBlock(nn.Module):
    """
    Standard 2×(3×3) residual block.
    Supports optional downsampling via stride=2 on the first conv.
    Uses ReLU nonlinearity and a configurable normalization kind.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        norm_kind: str = "none",
    ):
        super().__init__()
        # If no normalization, keep bias=True; otherwise bias=False is customary.
        use_bias = (norm_kind == "none")

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=use_bias)
        self.norm1 = _norm_layer(norm_kind, out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=use_bias)
        self.norm2 = _norm_layer(norm_kind, out_channels)

        # Projection for residual if shape changes
        if stride != 1 or in_channels != out_channels:
            self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                                  stride=stride, bias=True)
        else:
            self.proj = nn.Identity()

        if hasattr(self.norm2, 'weight'):
            nn.init.zeros_(self.norm2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.proj(x)

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        out = out + identity
        out = self.relu(out)
        return out

class PreActResidualBlock(nn.Module) : 
    """
    Pre-activation ResNet block:
      norm → ReLU → Conv → norm → ReLU → Conv → (+ ECA) → add skip
    """
    def __init__(self, in_channels, out_channels, kernel_size = 3,stride = 1, norm_kind = 'none' , dilation : int = 1) :

        super.__init__()
        use_bias = (norm_kind == 'none')
        self.norm1 = _norm_layer(norm_kind, in_channels)
        self.norm2 = _norm_layer(norm_kind,out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size,stride = stride, 
                               padding = dilation, dilation = dilation, bias = use_bias)
        self.conv2 = nn.Conv2d(out_channels,out_channels,kernel_size=kernel_size,
                               stride = 1, padding = 1, bias = use_bias)
        
        self.need_proj = (stride != 1) or (in_channels != out_channels) 
        self.proj = nn.Conv2d(in_channels, out_channels, 1, stride = stride, bias = True) if self.need_proj else nn.Identity()

        # if hasattr(self.norm2, 'weight'):
        #     nn.init.zeros_(self.norm2.weight)

    def forward(self,x) : 

        out = self.relu(self.norm1(x))
        skip = self.proj(out) if self.need_proj else x

        out = self.conv1(out)
        out = self.relu(self.norm2(out))
        out = self.conv2(out)
        
        res = out + skip

        return res
    


class _BaseEncoder(nn.Module):
    """
    Shared backbone:
      - Stem: 7×7 conv, stride 2  → 1/2 resolution, 32 channels
      - Stage 1: 2 residual blocks at 1/2 res (C=32)
      - Stage 2: 2 residual blocks at 1/4 res (C=64), with downsample on the first block
    Norm kind differs between Matching (instance) and Context (none).
    """
    def __init__(self, norm_kind: str):
        super().__init__()
        use_bias = (norm_kind == "none")

        # Stem: 7x7 conv stride 2 → output channels 32
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3, bias=use_bias),
            _norm_layer(norm_kind, 32),
            nn.ReLU(inplace=True),
        )

        # Stage 1 @ 1/2 resolution: two residual blocks, C=32
        self.layer1 = nn.Sequential(
            ResidualBlock(32, 32, stride=1, norm_kind=norm_kind),
            ResidualBlock(32, 32, stride=1, norm_kind=norm_kind),
        )

        # Stage 2 @ 1/4 resolution: downsample then another block, C=64
        self.layer2 = nn.Sequential(
            ResidualBlock(32, 64, stride=2, norm_kind=norm_kind),  # downsample to 1/4
            ResidualBlock(64, 64, stride=1, norm_kind=norm_kind),
        )

        # Kaiming init for convs
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        # InstanceNorm2d with affine=True will have weight/bias
        if isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)) and getattr(m, "affine", False):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns the final feature map at 1/4 input resolution with 64 channels.
        Input:  (N, 3, H, W)
        Output: (N, 64, H/4, W/4)
        """
        x = self.stem(x)     # -> (N, 32, H/2, W/2)
        x = self.layer1(x)   # -> (N, 32, H/2, W/2)
        x = self.layer2(x)   # -> (N, 64, H/4, W/4)
        return x


class MatchingEncoder(_BaseEncoder):
    """
    Matching feature network (uses InstanceNorm).
    """
    def __init__(self):
        super().__init__(norm_kind="instance")

    @staticmethod
    def build_two_level_pyramid(feat_1_4: torch.Tensor) -> List[torch.Tensor]:
        """
        Two-level feature pyramid from matching features:
          - Level 0: original 1/4 res (C=64)
          - Level 1: average-pooled with 4×4 kernel & stride 4 → ~1/16 res
        """
        lvl0 = feat_1_4
        lvl1 = F.avg_pool2d(feat_1_4, kernel_size=4, stride=4, padding=0)
        return [lvl0, lvl1]


class ContextEncoder(_BaseEncoder):
    """
    Context feature network (no normalization).
    """
    def __init__(self):
        super().__init__(norm_kind="none")



if __name__ == "__main__":
    enc_match = MatchingEncoder()
    enc_ctx = ContextEncoder()

    x = torch.randn(2, 3, 256, 256)  # batch of images

    f_match_1_4 = enc_match(x)       # (2, 64, 64, 64)
    pyramid = MatchingEncoder.build_two_level_pyramid(f_match_1_4)
    # pyramid[0]: (2, 64, 64, 64)  -> 1/4 res
    # pyramid[1]: (2, 64, 16, 16)  -> 1/16 res via avg pooling

    f_ctx_1_4 = enc_ctx(x)           # (2, 64, 64, 64)
    print(f_match_1_4.shape, pyramid[1].shape, f_ctx_1_4.shape)

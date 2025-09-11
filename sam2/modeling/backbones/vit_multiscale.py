#把任意 HF ViT（如 DINOv3 / I-JEPA）从单尺度 patch 网格转成四个尺度，接口与 SAM2 的 FPN 对齐。
import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformers import AutoModel
except Exception:  # transformers might not be installed in minimal env
    AutoModel = None


def _split_cls_and_patches(tokens: torch.Tensor):
    """Return (patch_tokens, has_cls).
    Accept either [B, N, C] (pure patches) or [B, 1+N, C] (cls+patches) or cls+reg+k+patches.
    We conservatively detect first token as CLS if (S-1) is a perfect square.
    """
    B, S, C = tokens.shape
    s = int(math.isqrt(S))
    if s * s == S:
        return tokens, False
    # try remove 1
    S1 = S - 1
    s1 = int(math.isqrt(S1))
    if s1 * s1 == S1:
        return tokens[:, 1:], True
    # try remove upto 16 registry tokens (CLS + reg*k)
    for k in range(1, 17):
        Sm = S - 1 - k
        if Sm <= 0:
            break
        sm = int(math.isqrt(Sm))
        if sm * sm == Sm:
            return tokens[:, 1 + k :], True
    raise ValueError(f"Cannot parse token layout S={S}")


class DepthwiseSeparable(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.dw = nn.Conv2d(ch, ch, 3, 1, 1, groups=ch, bias=False)
        self.pw = nn.Conv2d(ch, ch, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(ch)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.bn(self.pw(self.dw(x))))


class ViTTrunkMultiScale(nn.Module):
    """
    Wrap a single-scale ViT (DINOv3 / I-JEPA / generic HF ViT) and synthesize 4 pyramid levels:
      F4  (stride 4)
      F8  (stride 8)
      F16 (stride 16)  -- base patch grid
      F32 (stride 32)

    Returned list order: [F4, F8, F16, F32].
    channel_list is stored in *low→high* order to match existing FpnNeck assertion:
        channel_list == backbone_channel_list (which enumerates from lowest spatial resolution to highest)
    Therefore: channel_list = [C32, C16, C8, C4].
    """

    def __init__(
        self,
        pretrained: str,
        encoder_type: str = "dino",  # for future conditional logic if needed
        out_dims: Optional[List[int]] = None,  # [C32,C16,C8,C4]
        upsample_mode: str = "bilinear",  # bilinear | deconv
        refine_highres: bool = True,
        freeze_vit: bool = False,
        force_dtype: Optional[str] = None,  # one of None|bf16|fp16|fp32
        verbose: bool = False,
    ):
        super().__init__()
        assert AutoModel is not None, "transformers not available. Please install transformers package."
        self.vit = AutoModel.from_pretrained(pretrained, trust_remote_code=True)
        hidden_size = getattr(self.vit.config, "hidden_size")
        assert hidden_size is not None, "ViT config missing hidden_size"

        self.encoder_type = encoder_type
        self.hidden_size = hidden_size
        self.upsample_mode = upsample_mode
        self.refine_highres = refine_highres
        self.freeze_vit = freeze_vit
        self.force_dtype = force_dtype
        self.verbose = verbose

        if out_dims is None:
            # default keep all same
            out_dims = [hidden_size, hidden_size, hidden_size, hidden_size]
        assert len(out_dims) == 4, "out_dims must be length 4: [C32,C16,C8,C4]"
        self.c32, self.c16, self.c8, self.c4 = out_dims
        # Provide channel_list low→high for FpnNeck compatibility
        self.channel_list = [self.c32, self.c16, self.c8, self.c4]

        # Projections from base F16 hidden -> each needed channel
        self.proj16 = self._make_proj(hidden_size, self.c16)
        self.proj32 = self._make_proj(self.c16, self.c32)
        self.proj8 = self._make_proj(self.c16, self.c8)
        self.proj4 = self._make_proj(self.c8, self.c4)

        if self.refine_highres:
            self.refine8 = DepthwiseSeparable(self.c8)
            self.refine4 = DepthwiseSeparable(self.c4)
        else:
            self.refine8 = nn.Identity()
            self.refine4 = nn.Identity()

        if self.freeze_vit:
            for p in self.vit.parameters():
                p.requires_grad_(False)

        # Flag to print shapes only once
        self._printed = False

    @staticmethod
    def _make_proj(cin, cout):
        if cin == cout:
            return nn.Identity()
        return nn.Sequential(nn.Conv2d(cin, cout, 1, bias=False), nn.BatchNorm2d(cout), nn.GELU())

    def _upsample(self, x, scale=2):
        if self.upsample_mode == "deconv":
            # allocate on first call
            key = f"deconv_{x.shape[1]}"
            if not hasattr(self, key):
                setattr(
                    self,
                    key,
                    nn.ConvTranspose2d(x.shape[1], x.shape[1], kernel_size=2, stride=2),
                )
            layer = getattr(self, key)
            return layer(x)
        # bilinear path
        return F.interpolate(x, scale_factor=scale, mode="bilinear", align_corners=False)

    def forward(self, pixel_values: torch.Tensor):
        B, C, H, W = pixel_values.shape
        patch = getattr(self.vit.config, "patch_size", 16)
        assert H % patch == 0 and W % patch == 0, f"Input shape ({H},{W}) not divisible by patch size {patch}"
        # HF ViT expects keyword pixel_values
        with torch.set_grad_enabled(not self.freeze_vit):
            vit_out = self.vit(pixel_values=pixel_values, return_dict=True)
            tokens = vit_out.last_hidden_state  # [B, S(+cls), D]

        patches, _ = _split_cls_and_patches(tokens)
        N = patches.shape[1]
        h16 = int(math.isqrt(N))
        assert h16 * h16 == N, f"Patch tokens not square: {N}"
        feat16 = patches.transpose(1, 2).reshape(B, self.hidden_size, h16, h16)
        f16 = self.proj16(feat16)  # (B, C16, h16, h16)

        # F32 (downsample)
        f32 = F.conv2d(f16, weight=self._get_downsample_weight(self.c16, self.c32), stride=2, padding=1)
        # F8 (upsample)
        f8 = self._upsample(f16, 2)
        f8 = self.proj8(f8)
        # F4 (from f8)
        f4 = self._upsample(f8, 2)
        f4 = self.proj4(f4)

        if self.refine_highres:
            f8 = self.refine8(f8)
            f4 = self.refine4(f4)

        if self.force_dtype:
            tgt = {
                "bf16": torch.bfloat16,
                "fp16": torch.float16,
                "fp32": torch.float32,
            }.get(self.force_dtype.lower())
            if tgt is None:
                raise ValueError(f"Unsupported force_dtype {self.force_dtype}")
            f4, f8, f16, f32 = [t.to(tgt) for t in (f4, f8, f16, f32)]

        if self.verbose and not self._printed:
            print(
                f"[ViTTrunkMultiScale] Synthesized shapes: F4={tuple(f4.shape)}, F8={tuple(f8.shape)}, F16={tuple(f16.shape)}, F32={tuple(f32.shape)}"
            )
            self._printed = True

        # Order required by ImageEncoder -> FpnNeck pipeline: high resolution first
        # Our channel_list is low→high so neck.assert still passes
        return [f4, f8, f16, f32]

    def _get_downsample_weight(self, cin, cout):
        # Create or cache a simple conv weight (3x3 depthwise+pointwise collapsed into single conv). Simplicity first.
        key = f"_dw_weight_{cin}_{cout}"
        if not hasattr(self, key):
            layer = nn.Conv2d(cin, cout, 3, 2, 1, bias=False)
            nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
            setattr(self, key, layer)
        return getattr(self, key).weight

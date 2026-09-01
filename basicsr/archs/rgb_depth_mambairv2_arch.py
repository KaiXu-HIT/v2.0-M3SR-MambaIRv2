import torch
from torch import nn

from basicsr.archs.mambairv2_arch import MambaIRv2
from basicsr.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class RGBDepthMambaIRv2(MambaIRv2):
    """Stage-1 B1: minimal early RGB-depth fusion before MambaIRv2.

    The original RGB shallow convolution is retained. A matching depth
    convolution and a 1x1 projection fuse the concatenated features; every
    subsequent MambaIRv2 backbone and reconstruction layer is unchanged.
    """

    def __init__(self, depth_in_chans=1, **kwargs):
        super().__init__(**kwargs)
        self.depth_in_chans = depth_in_chans
        self.depth_conv = nn.Conv2d(depth_in_chans, self.embed_dim, 3, 1, 1)
        self.rgb_depth_fusion = nn.Conv2d(self.embed_dim * 2, self.embed_dim, 1, 1)

    @staticmethod
    def _mirror_pad_to_window(x, target_h, target_w):
        x = torch.cat([x, torch.flip(x, [2])], 2)[:, :, :target_h, :]
        return torch.cat([x, torch.flip(x, [3])], 3)[:, :, :, :target_w]

    def forward(self, rgb, depth):
        if depth is None:
            raise ValueError('RGBDepthMambaIRv2 requires a depth tensor.')
        if rgb.ndim != 4 or depth.ndim != 4:
            raise ValueError('RGB and depth inputs must both be BCHW tensors.')
        if rgb.shape[0] != depth.shape[0] or rgb.shape[-2:] != depth.shape[-2:]:
            raise ValueError(f'RGB/depth batch or spatial mismatch: {rgb.shape} vs {depth.shape}.')
        if depth.shape[1] != self.depth_in_chans:
            raise ValueError(f'Expected {self.depth_in_chans} depth channel(s), got {depth.shape[1]}.')

        h_ori, w_ori = rgb.shape[-2:]
        h = ((h_ori + self.window_size - 1) // self.window_size) * self.window_size
        w = ((w_ori + self.window_size - 1) // self.window_size) * self.window_size
        rgb = self._mirror_pad_to_window(rgb, h, w)
        depth = self._mirror_pad_to_window(depth, h, w)

        self.mean = self.mean.type_as(rgb)
        rgb = (rgb - self.mean) * self.img_range
        depth = depth.type_as(rgb)

        attn_mask = self.calculate_mask([h, w]).to(rgb.device)
        params = {'attn_mask': attn_mask, 'rpi_sa': self.relative_position_index_SA}

        rgb_feature = self.conv_first(rgb)
        depth_feature = self.depth_conv(depth)
        fused = self.rgb_depth_fusion(torch.cat([rgb_feature, depth_feature], dim=1))
        body = self.conv_after_body(self.forward_features(fused, params)) + fused

        if self.upsampler == 'pixelshuffle':
            output = self.conv_last(self.upsample(self.conv_before_upsample(body)))
        elif self.upsampler == 'pixelshuffledirect':
            output = self.upsample(body)
        elif self.upsampler == 'nearest+conv':
            output = self.conv_before_upsample(body)
            output = self.lrelu(self.conv_up1(torch.nn.functional.interpolate(output, scale_factor=2, mode='nearest')))
            output = self.lrelu(self.conv_up2(torch.nn.functional.interpolate(output, scale_factor=2, mode='nearest')))
            output = self.conv_last(self.lrelu(self.conv_hr(output)))
        else:
            output = rgb + self.conv_last(body)

        output = output / self.img_range + self.mean
        return output[..., :h_ori * self.upscale, :w_ori * self.upscale]

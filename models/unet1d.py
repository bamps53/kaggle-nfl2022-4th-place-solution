import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.losses import MaskedBCEWithLogitsLoss
from models.mixup import SeparateMultiMixup


class SEModule(nn.Module):
    def __init__(self, channel, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class SingleConv(nn.Module):
    """(convolution => [BN] => ReLU)"""

    def __init__(self, in_channels, out_channels, kernel_size, bias=False, norm=nn.BatchNorm1d, se=False, res=False):
        super().__init__()
        self.res = res
        if se:
            non_linearity = SEModule(out_channels)
        else:
            non_linearity = nn.ReLU(inplace=True)
        self.single_conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2, bias=bias),
            norm(out_channels),
            non_linearity
        )

    def forward(self, x):
        if self.res:
            return x + self.single_conv(x)
        else:
            return self.single_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, norm=nn.BatchNorm1d, se=False, res=False):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(scale_factor),
            SingleConv(in_channels, out_channels, kernel_size=kernel_size, norm=norm, se=se, res=res)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, kernel_size, scale_factor=2, norm=nn.BatchNorm1d):
        super().__init__()
        self.up = nn.Upsample(scale_factor=scale_factor, mode='linear', align_corners=True)
        self.conv = SingleConv(in_channels, out_channels, kernel_size, norm=norm)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diff = x2.size()[2] - x1.size()[2]
        assert diff == 0
        x1 = F.pad(x1, [diff // 2, diff - diff // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1):
        super(OutConv, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2, bias=True)

    def forward(self, x):
        return self.conv(x)


class ImageFeatUNet1dGlobal(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        in_player_channels = model_cfg.in_player_channels
        in_pair_channels = model_cfg.in_pair_channels
        in_global_channels = model_cfg.in_global_channels
        in_image_channels = model_cfg.in_image_channels
        hidden_channels = model_cfg.hidden_channels
        kernel_size = model_cfg.kernel_size
        pos_ratio = model_cfg.pos_ratio
        n_classes = 1
        scale_factor = 2
        drop_rate = model_cfg.drop_rate

        self.player_conv = nn.Sequential(
            SingleConv(in_player_channels, hidden_channels, kernel_size=1, bias=True),
            SingleConv(hidden_channels, hidden_channels, kernel_size, se=True)
        )
        self.pair_conv = nn.Sequential(
            SingleConv(in_pair_channels, hidden_channels, kernel_size=1, bias=True),
            SingleConv(hidden_channels, hidden_channels, kernel_size, se=True)
        )
        self.global_conv = nn.Sequential(
            SingleConv(in_global_channels, hidden_channels, kernel_size=1, bias=True),
            SingleConv(hidden_channels, hidden_channels, kernel_size, se=True)
        )
        self.image_pair_conv = nn.Sequential(
            SingleConv(in_image_channels, in_image_channels // 2, kernel_size=1, bias=True),
            SingleConv(in_image_channels // 2, hidden_channels * 2, kernel_size, se=True)
        )
        n_channels = hidden_channels * 6

        unit = 64
        unit2 = 128
        unit4 = 256
        unit8 = 512

        self.inc = SingleConv(n_channels, unit, kernel_size)
        self.down1 = Down(unit, unit2, kernel_size, scale_factor)
        self.down2 = Down(unit2, unit4, kernel_size, scale_factor)
        self.down3 = Down(unit4, unit8 // 2, kernel_size, scale_factor)
        self.up1 = Up(unit8, unit4 // 2, kernel_size, scale_factor)
        self.up2 = Up(unit4, unit2 // 2, kernel_size, scale_factor)
        self.up3 = Up(unit2, unit, kernel_size, scale_factor)
        self.cls = OutConv(unit, n_classes, kernel_size)
        self.cls.bias = nn.Parameter(
            torch.tensor([math.log(pos_ratio / (1 - pos_ratio))]))

        self.dropout = nn.Dropout(drop_rate)
        self.cls_loss = MaskedBCEWithLogitsLoss()

        self.do_mixup = False
        if model_cfg.mix_beta > 0:
            self.do_mixup = True
            self.mixup = SeparateMultiMixup(mix_beta=model_cfg.mix_beta)

    def forward(self, inputs, fold=None):
        pair_feats = self.pair_conv(inputs['pair_feats'])  # b, c, t
        p1_feats = self.player_conv(inputs['p1_feats'])  # b, c, t
        p2_feats = self.player_conv(inputs['p2_feats'])  # b, c, t
        global_feats = self.global_conv(inputs['global_feats'])  # b, c, t
        if fold is not None:
            img_feats = self.image_pair_conv(inputs[f'image_features_fold{fold}'])  # b, c, t
        else:
            img_feats = self.image_pair_conv(inputs['image_features'])  # b, c, t
        avg_feats = (p1_feats + p2_feats) / 2.0
        corr_feats = p1_feats * p2_feats

        feats = torch.cat([pair_feats, avg_feats, corr_feats, global_feats, img_feats], dim=1)

        if self.training and self.do_mixup:
            mixed_inputs = self.mixup(
                x=feats,
                cls_labels=inputs['inter'],
                cls_masks=inputs['masks'],
            )
            feats = mixed_inputs['x']

        x1 = self.inc(feats)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = x4
        x = self.up1(x, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.dropout(x)
        inter_logits = self.cls(x).squeeze(1)
        outputs = {
            'inter': inter_logits,
        }
        if self.training and self.do_mixup:
            mixed_inputs.pop('x')
            outputs.update(mixed_inputs)
        return outputs

    def get_loss(self, outputs, inputs):
        if self.training and self.do_mixup:
            cls_loss1 = self.cls_loss(outputs['inter'], outputs['cls_labels1'],
                                      outputs['cls_masks1'], outputs['coeffs'])
            cls_loss2 = self.cls_loss(outputs['inter'], outputs['cls_labels2'],
                                      outputs['cls_masks2'], 1 - outputs['coeffs'])
            cls_loss = cls_loss1 + cls_loss2
        else:
            cls_loss = self.cls_loss(outputs['inter'], inputs['inter'], inputs['masks'])

        loss_dict = {
            'loss': cls_loss,
            'inter': cls_loss,
        }
        return loss_dict


class ImageFeatUNet1dGroundGlobalLN(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        in_player_channels = model_cfg.in_player_channels
        in_global_channels = model_cfg.in_global_channels
        in_image_channels = model_cfg.in_image_channels
        hidden_channels = model_cfg.hidden_channels
        kernel_size = model_cfg.kernel_size
        pos_ratio = model_cfg.pos_ratio
        n_classes = 1
        scale_factor = 2
        drop_rate = model_cfg.drop_rate
        self.duration = model_cfg.max_len

        def create_layer_norm(channel, length):
            return nn.LayerNorm([channel, length])

        self.player_conv = nn.Sequential(
            SingleConv(in_player_channels, hidden_channels, kernel_size=1, bias=True,
                       norm=partial(create_layer_norm, length=self.duration)),
            SingleConv(hidden_channels, hidden_channels, kernel_size, se=True,
                       norm=partial(create_layer_norm, length=self.duration))
        )
        self.global_conv = nn.Sequential(
            SingleConv(in_global_channels, hidden_channels, kernel_size=1, bias=True,
                       norm=partial(create_layer_norm, length=self.duration)),
            SingleConv(hidden_channels, hidden_channels, kernel_size, se=True,
                       norm=partial(create_layer_norm, length=self.duration))
        )
        self.image_pair_conv = nn.Sequential(
            SingleConv(in_image_channels, in_image_channels // 2, kernel_size=1,
                       bias=True, norm=partial(create_layer_norm, length=self.duration)),
            SingleConv(in_image_channels // 2, hidden_channels * 2, kernel_size,
                       se=True, norm=partial(create_layer_norm, length=self.duration))
        )
        n_channels = hidden_channels * 3

        unit = 64
        unit2 = 128
        unit4 = 256
        unit8 = 512

        self.inc = SingleConv(n_channels, unit, kernel_size, norm=partial(create_layer_norm, length=self.duration))
        self.down1 = Down(unit, unit2, kernel_size, scale_factor,
                          norm=partial(create_layer_norm, length=self.duration//2))
        self.down2 = Down(unit2, unit4, kernel_size, scale_factor,
                          norm=partial(create_layer_norm, length=self.duration//4))
        self.down3 = Down(unit4, unit8 // 2, kernel_size, scale_factor,
                          norm=partial(create_layer_norm, length=self.duration//8))
        self.up1 = Up(unit8, unit4 // 2, kernel_size, scale_factor,
                      norm=partial(create_layer_norm, length=self.duration//4))
        self.up2 = Up(unit4, unit2 // 2, kernel_size, scale_factor,
                      norm=partial(create_layer_norm, length=self.duration//2))
        self.up3 = Up(unit2, unit, kernel_size, scale_factor, norm=partial(create_layer_norm, length=self.duration))
        self.cls = OutConv(unit, n_classes, kernel_size)
        self.cls.bias = nn.Parameter(
            torch.tensor([math.log(pos_ratio / (1 - pos_ratio))]))

        self.dropout = nn.Dropout(drop_rate)
        self.cls_loss = MaskedBCEWithLogitsLoss()

        self.do_mixup = False
        if model_cfg.mix_beta > 0:
            self.do_mixup = True
            self.mixup = SeparateMultiMixup(mix_beta=model_cfg.mix_beta)

    def forward(self, inputs, fold=None):
        feats = self.player_conv(inputs['p1_feats'])  # b, c, t
        if fold is not None:
            img_feats = self.image_pair_conv(inputs[f'image_features_fold{fold}'])  # b, c, t
        else:
            img_feats = self.image_pair_conv(inputs['image_features'])  # b, c, t
        feats = torch.cat([feats, img_feats], dim=1)
        if self.training and self.do_mixup:
            mixed_inputs = self.mixup(
                x=feats,
                cls_labels=inputs['ground'],
                cls_masks=inputs['masks'],
            )
            feats = mixed_inputs['x']

        x1 = self.inc(feats)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = x4
        x = self.up1(x, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.dropout(x)
        ground_logits = self.cls(x).squeeze(1)
        outputs = {
            'ground': ground_logits,
        }
        if self.training and self.do_mixup:
            mixed_inputs.pop('x')
            outputs.update(mixed_inputs)
        return outputs

    def get_loss(self, outputs, inputs):
        if self.training and self.do_mixup:
            cls_loss1 = self.cls_loss(outputs['ground'], outputs['cls_labels1'],
                                      outputs['cls_masks1'], outputs['coeffs'])
            cls_loss2 = self.cls_loss(outputs['ground'], outputs['cls_labels2'],
                                      outputs['cls_masks2'], 1 - outputs['coeffs'])
            cls_loss = cls_loss1 + cls_loss2
        else:
            cls_loss = self.cls_loss(outputs['ground'], inputs['ground'], inputs['masks'])

        loss_dict = {
            'loss': cls_loss,
            'ground': cls_loss,
        }
        return loss_dict

from einops import rearrange
import torch
import torch.nn as nn
from torchvision.ops import roi_align

from models.utils import transform_rois

from .yolox_models import YOLOPAFPN
from .losses import MaskedBCEWithLogitsLoss
from .common import Transformer


def build_yolox_backbone(pretrained_path=None, model_name='yolox_m', keep_deep=False):
    if model_name == 'yolox_s':
        depth = 0.33
        width = 0.50
    elif model_name == 'yolox_m':
        depth = 0.67
        width = 0.75
    elif model_name == 'yolox_l':
        depth = 1.00
        width = 1.00
    elif model_name == 'yolox_x':
        depth = 1.33
        width = 1.25
    else:
        raise NotImplementedError()

    act = 'silu'
    in_channels = [256, 512, 1024]
    backbone = YOLOPAFPN(depth, width, in_channels=in_channels, act=act, keep_deep=keep_deep)
    if pretrained_path is not None:
        print('load yolox weights from', pretrained_path)
        ckpt = torch.load(pretrained_path, map_location='cpu')
        backbone.load_state_dict(ckpt)
    return backbone


class BaseConv(nn.Module):
    """A Conv2d -> Batchnorm -> silu block"""

    def __init__(
        self, in_channels, out_channels, ksize, stride, groups=1, bias=False
    ):
        super().__init__()
        # same padding
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class TrackAnyTransformerYoloxHMGlobal(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        self.model_cfg = model_cfg
        self.backbone = build_yolox_backbone(model_cfg.pretrained_path, model_cfg.model_name, keep_deep=True)

        self.dist_activation = model_cfg.dist_activation
        self.down_ratio = model_cfg.down_ratio
        self.roi_size = model_cfg.roi_size
        self.num_channels = model_cfg.num_channels

        self.hm_conv2d = BaseConv(1, model_cfg.stem_channels, 3, 1)

        self.gc_pool = nn.AdaptiveAvgPool2d(1)
        self.gc_linear = nn.Sequential(
            nn.Linear(model_cfg.context_channels, self.num_channels),
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.shared_linear = nn.Sequential(
            nn.Linear(self.num_channels * (self.roi_size ** 2 + 1) * 2, self.num_channels * 2),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.transformer = Transformer(self.num_channels * 2, depth=3, heads=8, dim_head=64, mlp_dim=512, dropout=0.1)

        self.inter_linear = nn.Sequential(
            nn.Linear(self.num_channels * 2, self.num_channels * 2),
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.g_linear = nn.Sequential(
            nn.Linear(self.num_channels * 2, self.num_channels * 2),
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.dist_linear = nn.Linear(3, self.num_channels*3, bias=True)
        if self.dist_activation == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif self.dist_activation == 'sigmoid':
            self.act = nn.Sigmoid()
        else:
            self.act = nn.Identity()

        self.ext_g_linear = nn.Sequential(
            nn.Linear(model_cfg.num_track_features, self.num_channels),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(self.num_channels, self.num_channels),
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.ext_any_linear = nn.Sequential(
            nn.Linear(model_cfg.num_track_features, self.num_channels),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(self.num_channels, self.num_channels),
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.ext_inter_linear = nn.Sequential(
            nn.Linear(model_cfg.num_track_features*2 + model_cfg.num_pair_features, self.num_channels),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(self.num_channels, self.num_channels),
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.inter_dropout = nn.Dropout(0.1)
        self.inter_classifier = nn.Sequential(
            nn.Linear(self.num_channels*3, self.num_channels),
            nn.ReLU(),
            nn.Linear(self.num_channels, 1, bias=True),
        )

        self.g_dropout = nn.Dropout(0.1)
        self.g_classifier = nn.Sequential(
            nn.Linear(self.num_channels*3, self.num_channels),
            nn.ReLU(),
            nn.Linear(self.num_channels, 1, bias=True),
        )

        self.any_dropout = nn.Dropout(0.1)
        self.any_classifier = nn.Sequential(
            nn.Linear(self.num_channels*3, self.num_channels),
            nn.ReLU(),
            nn.Linear(self.num_channels, 1, bias=True),
        )

        self.cls_loss = MaskedBCEWithLogitsLoss(model_cfg.label_smoothing)
        self.inter_weight = model_cfg.inter_weight
        self.ground_weight = model_cfg.ground_weight
        self.any_weight = model_cfg.any_weight

    def forward(self, inputs):
        images = []
        rois = []
        heatmaps = []
        masks = []
        for view in ['Endzone', "Sideline"]:
            images.append(inputs[f'{view}_image'])
            rois.append(inputs[f'{view}_rois'])  # should be (b, t, n, 4) maybe has null, int xyxy format
            masks.append(inputs[f'{view}_mask'])
            heatmaps.append(inputs[f'{view}_heatmap'])

        g_masks = torch.cat(masks, dim=1).sum(1) > 0
        n = g_masks.shape[1]
        masks1 = g_masks[:, :, None].expand(-1, -1, n)
        masks2 = g_masks[:, None, :].expand(-1, n, -1)
        inter_masks = masks1 & masks2
        if not self.model_cfg.add_self_contact_as_False:
            for i in range(n):
                inter_masks[:, i, i] = False  # mask out self-self pair

        images = torch.cat(images, dim=0)
        heatmaps = torch.cat(heatmaps, dim=0)
        rois = torch.cat(rois, dim=0)

        coords_dist = inputs['distance']

        # 余り離れているものはlossを取らない（閾値はデフォルトかなり大きい値なので変えたときのみ発動）
        inter_masks = inter_masks & (coords_dist < self.model_cfg.dist_th)

        end_coords = inputs[f'Endzone_coords'].squeeze(1)
        end_coords_dist = torch.cdist(end_coords, end_coords)  # b, n, n

        side_coords = inputs[f'Sideline_coords'].squeeze(1)
        side_coords_dist = torch.cdist(side_coords, side_coords)  # b, n, n

        distances = torch.stack([coords_dist, end_coords_dist, side_coords_dist], -1)  # b, n, n, 3

        images = images.squeeze(1)  # t dim
        heatmaps = heatmaps.squeeze(1)  # t dim
        b, c, h, w = images.shape
        mask_feats = self.hm_conv2d(heatmaps)
        output = self.backbone(images, mask_feats)
        x = output[0]
        context = output[-1]
        n = rois.shape[2]
        rois = rois.reshape(-1, n, 4)
        rois = transform_rois(rois)  # convert to shape=(b*t*n, 5) and rescale

        # crop_image = roi_align(images, rois, output_size=(128, 128), spatial_scale=1.0, aligned=True) # (b*n, c, h, w)
        # from torchvision.utils import save_image
        # save_image(crop_image, 'temp.png', normalize=True)
        # import ipdb; ipdb.set_trace()

        # xはFPNの1/8スケールの特徴マップ
        x = roi_align(x, rois, output_size=(self.roi_size, self.roi_size),
                      spatial_scale=1.0/self.down_ratio, aligned=True)  # (b*n, c, h, w)
        x = rearrange(x, '(b n) c h w -> b n (h w c)', b=b, n=n)
        context = self.gc_pool(context).squeeze(-1).squeeze(-1)
        context = self.gc_linear(context)
        context = context[:, None].expand(-1, n, -1)
        x = torch.cat([x, context], dim=2)
        end_x, side_x = torch.split(x, b//2)
        x = torch.cat([end_x, side_x], dim=2)
        x = self.shared_linear(x)  # (b*n, c)
        x = self.transformer(x, distances, attn_mask=inter_masks)
        x = rearrange(x, 'b n c -> (b n) c')
        player_x = self.inter_linear(x)  # (b*n, c)
        player_x = player_x.reshape(b//2, n, -1)  # ここまでは各playerごとの特徴

        g_x = self.g_linear(x)  # (b*n, c)
        g_x = g_x.reshape(b//2, n, -1)

        # endとsideをconcatして、全プレイヤー間の組み合わせで要素積を取る
        inter_x = torch.einsum('bnc,bmc->bnmc', player_x, player_x)

        # 各プレイヤーごとのtracking特徴のペアを作りconcat->Linear
        inter_track_x = inputs['track']
        inter_track_x1 = inter_track_x[:, :, None].expand(-1, -1, n, -1)
        inter_track_x2 = inter_track_x[:, None, :].expand(-1, n, -1, -1)
        pair_track_x = inputs['pair']
        cross_track_x = torch.cat([inter_track_x1, inter_track_x2, pair_track_x], dim=3)
        cross_track_x = self.ext_inter_linear(cross_track_x)  # (b, n, n, c)
        inter_x = torch.cat([inter_x, cross_track_x], dim=3)

        dist_features = self.dist_linear(distances)  # b, n, n, c
        dist_features = self.act(dist_features)  # b, n, n, c

        inter_x *= dist_features  # b, n, n, c

        inter_x = self.inter_dropout(inter_x)
        inter_logits = self.inter_classifier(inter_x).squeeze(3)  # (b, n, 1)

        g_track_x = self.ext_g_linear(inputs['track'])
        g_x = torch.cat([g_x, g_track_x], dim=2)
        g_x = self.g_dropout(g_x)
        g_logits = self.g_classifier(g_x).squeeze(2)  # (b, n, 1)

        any_track_x = self.ext_any_linear(inputs['track'])
        any_x = torch.cat([player_x, any_track_x], dim=2)
        any_logits = self.any_classifier(any_x).squeeze(2)  # (b, n, 1)

        outputs = {
            'ground': g_logits,
            'ground_masks': g_masks,
            'any_masks': g_masks,
            'any': any_logits,
            'any_masks': g_masks,
            'inter': inter_logits,
            'inter_masks': inter_masks,
        }
        if self.model_cfg.return_feats:
            outputs['g_x'] = g_x
            outputs['any_x'] = any_x
            outputs['inter_x'] = inter_x
        return outputs

    def get_loss(self, outputs, inputs):
        inter_loss = self.cls_loss(outputs['inter'], inputs['inter'], outputs['inter_masks'])
        ground_loss = self.cls_loss(outputs['ground'], inputs['ground'], outputs['ground_masks'])
        inputs['any'] = (inputs['inter'].sum(dim=1) > 0).float()
        any_loss = self.cls_loss(outputs['any'], inputs['any'], outputs['any_masks'])

        loss = inter_loss * self.inter_weight + ground_loss * self.ground_weight + any_loss * self.any_weight
        loss_dict = {
            'loss': loss,
            'inter': inter_loss,
            'ground': ground_loss,
            'any': any_loss,
        }
        return loss_dict


class TrackAnyTransformerYoloxHMGlobalFixedMask(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        self.model_cfg = model_cfg
        self.backbone = build_yolox_backbone(model_cfg.pretrained_path, model_cfg.model_name, keep_deep=True)

        self.dist_activation = model_cfg.dist_activation
        self.down_ratio = model_cfg.down_ratio
        self.roi_size = model_cfg.roi_size
        self.num_channels = model_cfg.num_channels

        self.hm_conv2d = BaseConv(1, model_cfg.stem_channels, 3, 1)

        self.gc_pool = nn.AdaptiveAvgPool2d(1)
        self.gc_linear = nn.Sequential(
            nn.Linear(model_cfg.context_channels, self.num_channels),
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.shared_linear = nn.Sequential(
            nn.Linear(self.num_channels * (self.roi_size ** 2 + 1) * 2, self.num_channels * 2),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.transformer = Transformer(self.num_channels * 2, depth=3, heads=8, dim_head=64, mlp_dim=512, dropout=0.1)

        self.inter_linear = nn.Sequential(
            nn.Linear(self.num_channels * 2, self.num_channels * 2),
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.g_linear = nn.Sequential(
            nn.Linear(self.num_channels * 2, self.num_channels * 2),
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.dist_linear = nn.Linear(3, self.num_channels*3, bias=True)
        if self.dist_activation == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif self.dist_activation == 'sigmoid':
            self.act = nn.Sigmoid()
        else:
            self.act = nn.Identity()

        self.ext_g_linear = nn.Sequential(
            nn.Linear(model_cfg.num_track_features, self.num_channels),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(self.num_channels, self.num_channels),
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.ext_any_linear = nn.Sequential(
            nn.Linear(model_cfg.num_track_features, self.num_channels),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(self.num_channels, self.num_channels),
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.ext_inter_linear = nn.Sequential(
            nn.Linear(31, self.num_channels),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(self.num_channels, self.num_channels),
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.inter_dropout = nn.Dropout(0.1)
        self.inter_classifier = nn.Sequential(
            nn.Linear(self.num_channels*3, self.num_channels),
            nn.ReLU(),
            nn.Linear(self.num_channels, 1, bias=True),
        )

        self.g_dropout = nn.Dropout(0.1)
        self.g_classifier = nn.Sequential(
            nn.Linear(self.num_channels*3, self.num_channels),
            nn.ReLU(),
            nn.Linear(self.num_channels, 1, bias=True),
        )

        self.any_dropout = nn.Dropout(0.1)
        self.any_classifier = nn.Sequential(
            nn.Linear(self.num_channels*3, self.num_channels),
            nn.ReLU(),
            nn.Linear(self.num_channels, 1, bias=True),
        )

        self.cls_loss = MaskedBCEWithLogitsLoss(model_cfg.label_smoothing)
        self.inter_weight = model_cfg.inter_weight
        self.ground_weight = model_cfg.ground_weight
        self.any_weight = model_cfg.any_weight

    def forward(self, inputs):
        images = []
        rois = []
        heatmaps = []
        masks = []
        for view in ['Endzone', "Sideline"]:
            images.append(inputs[f'{view}_image'])
            rois.append(inputs[f'{view}_rois'])  # should be (b, t, n, 4) maybe has null, int xyxy format
            masks.append(inputs[f'{view}_mask'])
            heatmaps.append(inputs[f'{view}_heatmap'])

        player_masks = torch.cat(masks, dim=1).sum(1) > 0  # どちらか一つでもあればOK
        n = player_masks.shape[1]
        player_masks1 = player_masks[:, :, None].expand(-1, -1, n)
        player_masks2 = player_masks[:, None, :].expand(-1, n, -1)
        pair_masks = player_masks1 & player_masks2  # ペアのplayer両方がendかsideかのboxがある（片方がendだけで片方がsideだけもあり得る）

        images = torch.cat(images, dim=0)
        heatmaps = torch.cat(heatmaps, dim=0)
        rois = torch.cat(rois, dim=0)

        coords_dist = inputs['distance']

        end_coords = inputs[f'Endzone_coords'].squeeze(1)
        end_coords_dist = torch.cdist(end_coords, end_coords)  # b, n, n

        side_coords = inputs[f'Sideline_coords'].squeeze(1)
        side_coords_dist = torch.cdist(side_coords, side_coords)  # b, n, n

        distances = torch.stack([coords_dist, end_coords_dist, side_coords_dist], -1)  # b, n, n, 3

        images = images.squeeze(1)  # t dim
        heatmaps = heatmaps.squeeze(1)  # t dim
        b, c, h, w = images.shape
        mask_feats = self.hm_conv2d(heatmaps)
        output = self.backbone(images, mask_feats)
        x = output[0]
        context = output[-1]
        n = rois.shape[2]
        rois = rois.reshape(-1, n, 4)
        rois = transform_rois(rois)  # convert to shape=(b*t*n, 5) and rescale

        # crop_image = roi_align(images, rois, output_size=(128, 128), spatial_scale=1.0, aligned=True) # (b*n, c, h, w)
        # from torchvision.utils import save_image
        # save_image(crop_image, 'temp.png', normalize=True)
        # import ipdb; ipdb.set_trace()

        # xはFPNの1/8スケールの特徴マップ
        x = roi_align(x, rois, output_size=(self.roi_size, self.roi_size),
                      spatial_scale=1.0/self.down_ratio, aligned=True)  # (b*n, c, h, w)
        x = rearrange(x, '(b n) c h w -> b n (h w c)', b=b, n=n)
        context = self.gc_pool(context).squeeze(-1).squeeze(-1)
        context = self.gc_linear(context)
        context = context[:, None].expand(-1, n, -1)
        x = torch.cat([x, context], dim=2)
        end_x, side_x = torch.split(x, b//2)
        x = torch.cat([end_x, side_x], dim=2)
        x = self.shared_linear(x)  # (b*n, c)
        x = self.transformer(x, distances, attn_mask=pair_masks)
        x = rearrange(x, 'b n c -> (b n) c')
        player_x = self.inter_linear(x)  # (b*n, c)
        player_x = player_x.reshape(b//2, n, -1)  # ここまでは各playerごとの特徴

        g_x = self.g_linear(x)  # (b*n, c)
        g_x = g_x.reshape(b//2, n, -1)

        # endとsideをconcatして、全プレイヤー間の組み合わせで要素積を取る
        inter_x = torch.einsum('bnc,bmc->bnmc', player_x, player_x)

        # 各プレイヤーごとのtracking特徴のペアを作りconcat->Linear
        inter_track_x = inputs['track']
        inter_track_x1 = inter_track_x[:, :, None].expand(-1, -1, n, -1)
        inter_track_x2 = inter_track_x[:, None, :].expand(-1, n, -1, -1)
        pair_track_x = inputs['pair']
        cross_track_x = torch.cat([inter_track_x1, inter_track_x2, pair_track_x], dim=3)
        cross_track_x = self.ext_inter_linear(cross_track_x)  # (b, n, n, c)
        inter_x = torch.cat([inter_x, cross_track_x], dim=3)

        dist_features = self.dist_linear(distances)  # b, n, n, c
        dist_features = self.act(dist_features)  # b, n, n, c

        inter_x *= dist_features  # b, n, n, c

        inter_x = self.inter_dropout(inter_x)
        inter_logits = self.inter_classifier(inter_x).squeeze(3)  # (b, n, 1)

        g_track_x = self.ext_g_linear(inputs['track'])
        g_x = torch.cat([g_x, g_track_x], dim=2)
        g_x = self.g_dropout(g_x)
        g_logits = self.g_classifier(g_x).squeeze(2)  # (b, n, 1)

        any_track_x = self.ext_any_linear(inputs['track'])
        any_x = torch.cat([player_x, any_track_x], dim=2)
        any_logits = self.any_classifier(any_x).squeeze(2)  # (b, n, 1)

        # 余り離れているものはlossを取らない（閾値はデフォルトかなり大きい値なので変えたときのみ発動）
        loss_masks = coords_dist < self.model_cfg.dist_th
        # 自分同士のcontactををFalseとして加えるかどうか
        if not self.model_cfg.add_self_contact_as_False:
            for i in range(n):
                loss_masks[:, i, i] = False  # mask out self-self pair

        outputs = {
            'ground': g_logits,
            'ground_masks': player_masks,
            'any_masks': player_masks,
            'any': any_logits,
            'any_masks': player_masks,
            'inter': inter_logits,
            'inter_masks': loss_masks,
        }
        if self.model_cfg.return_feats:
            outputs['g_x'] = g_x
            outputs['any_x'] = any_x
            outputs['inter_x'] = inter_x
        return outputs

    def get_loss(self, outputs, inputs):
        inter_loss = self.cls_loss(outputs['inter'], inputs['inter'], outputs['inter_masks'])
        ground_loss = self.cls_loss(outputs['ground'], inputs['ground'], outputs['ground_masks'])
        inputs['any'] = (inputs['inter'].sum(dim=1) > 0).float()
        any_loss = self.cls_loss(outputs['any'], inputs['any'], outputs['any_masks'])

        loss = inter_loss * self.inter_weight + ground_loss * self.ground_weight + any_loss * self.any_weight
        loss_dict = {
            'loss': loss,
            'inter': inter_loss,
            'ground': ground_loss,
            'any': any_loss,
        }
        return loss_dict

import torch
import torch.nn as nn
from torchvision.ops import roi_align

from models.utils import transform_rois

from .yolox_models import YOLOPAFPN
from .losses import MaskedBCEWithLogitsLoss


def build_yolox_backbone(pretrained_path=None, model_name='yolox_m'):
    if model_name == 'yolox_s':
        depth = 0.33
        width = 0.50
    elif model_name == 'yolox_m':
        depth = 0.67
        width = 0.75
    elif model_name == 'yolox_x':
        depth = 1.33
        width = 1.25
    else:
        raise NotImplementedError()

    act = 'silu'
    in_channels = [256, 512, 1024]
    backbone = YOLOPAFPN(depth, width, in_channels=in_channels, act=act)
    if pretrained_path is not None:
        print('load yolox weights from', pretrained_path)
        ckpt = torch.load(pretrained_path, map_location='cpu')
        backbone.load_state_dict(ckpt)
    return backbone


class YOLOX_FPN_EXT(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        self.model_cfg = model_cfg
        self.backbone = build_yolox_backbone(model_cfg.pretrained_path, model_cfg.model_name)

        self.dist_activation = model_cfg.dist_activation
        self.down_ratio = model_cfg.down_ratio
        self.roi_size = 5
        self.num_channels = model_cfg.num_channels
        self.inter_conv2d = nn.Sequential(
            nn.Conv2d(in_channels=self.num_channels, out_channels=self.num_channels, kernel_size=3, stride=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(in_channels=self.num_channels, out_channels=self.num_channels, kernel_size=3, stride=1),
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.g_conv2d = nn.Sequential(
            nn.Conv2d(in_channels=self.num_channels, out_channels=self.num_channels, kernel_size=3, stride=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(in_channels=self.num_channels, out_channels=self.num_channels, kernel_size=3, stride=1),
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

        self.ext_inter_linear = nn.Sequential(
            nn.Linear(model_cfg.num_track_features*2, self.num_channels),
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
        self.cls_loss = MaskedBCEWithLogitsLoss(model_cfg.label_smoothing)
        self.inter_weight = model_cfg.inter_weight
        self.ground_weight = model_cfg.ground_weight

    def forward(self, inputs):
        images = []
        rois = []
        masks = []
        for view in ['Endzone', "Sideline"]:
            images.append(inputs[f'{view}_image'])
            rois.append(inputs[f'{view}_rois'])  # should be (b, t, n, 4) maybe has null, int xyxy format
            masks.append(inputs[f'{view}_mask'])
        g_masks = torch.cat(masks, dim=1).sum(1) > 0
        n = g_masks.shape[1]
        masks1 = g_masks[:, :, None].expand(-1, -1, n)
        masks2 = g_masks[:, None, :].expand(-1, n, -1)
        inter_masks = masks1 & masks2
        for i in range(n):
            inter_masks[:, i, i] = False  # mask out self-self pair

        images = torch.cat(images, dim=0)
        rois = torch.cat(rois, dim=0)

        images = images.squeeze(1)  # t dim
        b, c, h, w = images.shape
        x = self.backbone(images)[0]
        n = rois.shape[2]
        rois = rois.reshape(-1, n, 4)
        rois = transform_rois(rois)  # convert to shape=(b*t*n, 5) and rescale

        # crop_image = roi_align(images, rois, output_size=(128, 128), spatial_scale=1.0, aligned=True) # (b*n, c, h, w)
        # from torchvision.utils import save_image
        # save_image(crop_image, 'tmep.png', normalize=True)
        # import ipdb; ipdb.set_trace()

        # xはFPNの1/8スケールの特徴マップ
        x = roi_align(x, rois, output_size=(self.roi_size, self.roi_size),
                      spatial_scale=1.0/self.down_ratio, aligned=True)  # (b*n, c, h, w)
        inter_x = self.inter_conv2d(x).squeeze(-1).squeeze(-1).squeeze(-1)  # (b*n, c)
        inter_x = inter_x.reshape(b, n, -1)  # ここまでは各playerごとの特徴

        g_x = self.g_conv2d(x).squeeze(-1).squeeze(-1).squeeze(-1)  # (b*n, c)
        g_x = g_x.reshape(b, n, -1)

        # 各プレイヤーごとのtracking特徴のペアを作りconcat->Linear
        inter_track_x = inputs['track']
        inter_track_x1 = inter_track_x[:, :, None].expand(-1, -1, n, -1)
        inter_track_x2 = inter_track_x[:, None, :].expand(-1, n, -1, -1)
        cross_track_x = torch.cat([inter_track_x1, inter_track_x2], dim=3)
        cross_track_x = self.ext_inter_linear(cross_track_x)  # (b, n, n, c)

        # endとsideをconcatして、全プレイヤー間の組み合わせで要素積を取る
        inter_end_x, inter_side_x = torch.split(inter_x, b//2)
        inter_x = torch.cat([inter_end_x, inter_side_x], dim=2)
        inter_x = torch.einsum('bnc,bmc->bnmc', inter_x, inter_x)
        # tracking特徴も足しとく
        inter_x = torch.cat([inter_x, cross_track_x], dim=3)

        coords = inputs['track'][:, :, :2]  # b, n, 2
        coords_dist = torch.cdist(coords, coords)  # b, n, n

        # 余り離れているものはlossを取らない（閾値はデフォルトかなり大きい値なので変えたときのみ発動）
        inter_masks = inter_masks & (coords_dist < self.model_cfg.dist_th)

        end_coords = inputs[f'Endzone_coords'].squeeze(1)
        end_coords_dist = torch.cdist(end_coords, end_coords)  # b, n, n

        side_coords = inputs[f'Sideline_coords'].squeeze(1)
        side_coords_dist = torch.cdist(side_coords, side_coords)  # b, n, n

        dist_features = torch.stack([coords_dist, end_coords_dist, side_coords_dist], -1)  # b, n, n, 3
        dist_features = self.dist_linear(dist_features)  # b, n, n, c
        dist_features = self.act(dist_features)  # b, n, n, c

        inter_x *= dist_features  # b, n, n, c

        inter_x = self.inter_dropout(inter_x)
        inter_logits = self.inter_classifier(inter_x).squeeze(3)  # (b, n, 1)

        g_track_x = self.ext_g_linear(inputs['track'])
        g_end_x, g_side_x = torch.split(g_x, b//2)
        g_x = torch.cat([g_end_x, g_side_x, g_track_x], dim=2)
        g_x = self.g_dropout(g_x)
        g_logits = self.g_classifier(g_x).squeeze(2)  # (b, n, 1)

        outputs = {
            'ground': g_logits,
            'ground_masks': g_masks,
            'inter': inter_logits,
            'inter_masks': inter_masks,
        }
        if self.model_cfg.return_feats:
            outputs['g_x'] = g_x
            outputs['inter_x'] = inter_x
        return outputs

    def get_loss(self, outputs, inputs):
        inter_loss = self.cls_loss(outputs['inter'], inputs['inter'], outputs['inter_masks'])
        ground_loss = self.cls_loss(outputs['ground'], inputs['ground'], outputs['ground_masks'])

        loss = inter_loss * self.inter_weight + ground_loss * self.ground_weight
        loss_dict = {
            'loss': loss,
            'inter': inter_loss,
            'ground': ground_loss,
        }
        return loss_dict

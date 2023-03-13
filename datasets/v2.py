import albumentations as A
import random
import os
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from datasets import video_transforms

from .common import setup_df, normalize_img, to_torch_tensor, load_cv2_image, load_cv2_image, load_pickle, load_pil_image

VIEWS = ['Endzone', 'Sideline']
GROUND_ID = -1


def get_unique_ids_dict(label_df):
    unique_ids_dict = label_df.groupby(['game_play', 'frame'], observed=True)['nfl_player_id_1'].unique().to_dict()

    for k, v in unique_ids_dict.items():
        v = sorted(v)
        if len(v) < 22:
            v = v + [-1] * (22 - len(v))  # padding
        unique_ids_dict[k] = v
    return unique_ids_dict


def load_adj_frames(base_image_dir, game_play, view, frame, k=5):
    image_dir = f'{base_image_dir}/{game_play}_{view}/'
    imgs = []
    assert (k % 2 == 1)
    half_k = (k - 1) // 2
    assert(frame - half_k > 0)
    img_paths = []
    for i in range(-half_k, half_k+1):
        frame_id = frame + i
        img_path = os.path.join(image_dir, f'{frame_id:06}.jpg')
        if os.path.exists(img_path):
            img_paths.append(img_path)
        else:
            return [None] * k
    return [load_pil_image(img_path) for img_path in img_paths]


def get_inter_labels(inter_contacts, unique_ids):
    label_matrix = np.zeros((len(unique_ids), len(unique_ids)))
    for id1, id2 in inter_contacts:
        idx1 = unique_ids.index(id1)
        idx2 = unique_ids.index(id2)
        label_matrix[idx1, idx2] = 1
        label_matrix[idx2, idx1] = 1
    return label_matrix


def get_distance_matrix(distance_labels, unique_ids):
    distance_matrix = np.zeros((len(unique_ids), len(unique_ids)))
    for id1, id2, distance in distance_labels:
        idx1 = unique_ids.index(id1)
        idx2 = unique_ids.index(id2)
        distance_matrix[idx1, idx2] = distance / 100.0
        distance_matrix[idx2, idx1] = distance / 100.0
    return distance_matrix


def get_g_labels(ground_contacts, unique_ids):
    g_labels = np.zeros((len(unique_ids),))
    for id1 in ground_contacts:
        g_labels[unique_ids.index(id1)] = 1
    return g_labels


def get_track_feats(track_feats, unique_ids, num_track_features):
    ordered_track_feats = np.zeros((len(unique_ids), num_track_features))
    track_ids = track_feats[:, -1].astype(int).tolist()
    for i, id1 in enumerate(track_ids):
        ordered_track_feats[unique_ids.index(id1)] = track_feats[i, :-1]
    return ordered_track_feats


def get_pair_feats(pair_feats, unique_ids, num_pair_features):
    pair_feats_matrix = np.zeros((len(unique_ids), len(unique_ids), num_pair_features))
    for i, (id1, id2) in enumerate(pair_feats[:, :2]):
        idx1 = unique_ids.index(id1)
        idx2 = unique_ids.index(id2)
        pair_feats_matrix[idx1, idx2, :] = pair_feats[i, 2:]
        pair_feats_matrix[idx2, idx1, :] = pair_feats[i, 2:]
    return pair_feats_matrix


HELMET_SCALE = 16  # trainのheight, widthのmedianくらい


def draw_heatmap(bboxes, image_size, original_image_size):
    bboxes = bboxes.copy()
    # rescale bbox
    h, w = image_size
    org_h, org_w = original_image_size
    bboxes[:, 0] = bboxes[:, 0] * w / org_w
    bboxes[:, 1] = bboxes[:, 1] * h / org_h
    bboxes[:, 2] = bboxes[:, 2] * w / org_w
    bboxes[:, 3] = bboxes[:, 3] * h / org_h

    cxs = bboxes[:, 0] + bboxes[:, 2] / 2
    cys = bboxes[:, 1] + bboxes[:, 3] / 2
    rad = int((bboxes[:, 2:].max(1).mean() / 4).round())  # hyper param

    heatmaps = [
        draw_centernet_label(h, w, (cx, cy), down_ratio=1, radius=rad, k=1)
        for cx, cy in zip(cxs, cys)
    ]
    return heatmaps
    # all_heatmap = np.max(np.stack(heatmaps, axis=0), axis=0)


def draw_all_heatmap(bboxes, image_size, original_image_size):
    bboxes = bboxes.copy()
    # rescale bbox
    h, w = image_size
    org_h, org_w = original_image_size
    bboxes[:, 0] = bboxes[:, 0] * w / org_w
    bboxes[:, 1] = bboxes[:, 1] * h / org_h
    bboxes[:, 2] = bboxes[:, 2] * w / org_w
    bboxes[:, 3] = bboxes[:, 3] * h / org_h

    rad = int((bboxes[:, 2:].max(1).mean() / 4).round())  # hyper param
    heatmap = draw_centernet_labels(h, w, bboxes[:, :2], down_ratio=2, radius=rad, k=1)
    return heatmap[:, :, None]  # ch axis


def get_bboxes(frame_helmets, unique_ids, image_size, original_image_size, roi_size=None, normalize_coords=False):
    label_bboxes = np.zeros((len(unique_ids), 4), np.float32)
    label_coords = np.zeros((len(unique_ids), 2), np.float32)
    if frame_helmets is None:
        h, w = image_size
        heatmap = np.zeros((h//2, w//2, 1))
        return label_bboxes, label_coords, heatmap
    helmet_ids = frame_helmets[:, -1].tolist()
    bboxes = frame_helmets[:, :-1].astype(np.float32).copy()
    cxs = bboxes[:, 0] + bboxes[:, 2] / 2
    cys = bboxes[:, 1] + bboxes[:, 3] / 2

    heatmap = draw_all_heatmap(bboxes, image_size, original_image_size)
    # resized_heatmaps = draw_heatmap(bboxes, image_size, original_image_size)
    # all_heatmap = np.max(np.stack(resized_heatmaps, axis=0), axis=0)
    # cv2.imwrite(f'hm_all.png', (heatmap*255).astype(np.uint8))
    # import ipdb; ipdb.set_trace()

    if normalize_coords:
        # このフレームのヘルメットの大きさが平均と比べてどうか
        frame_helmet_scale = bboxes[:, 2:].max(1).mean() / HELMET_SCALE

        # 画像サイズで正規化
        normalized_bboxes = bboxes.copy()
        normalized_bboxes[:, [0, 2]] /= original_image_size[1]
        normalized_bboxes[:, [1, 3]] /= original_image_size[0]

        normalized_cx = normalized_bboxes[:, 0] + normalized_bboxes[:, 2] / 2
        normalized_cy = normalized_bboxes[:, 1] + normalized_bboxes[:, 3] / 2
        normalized_cx = normalized_cx / frame_helmet_scale
        normalized_cy = normalized_cy / frame_helmet_scale
    else:
        mean_wh = bboxes[:, 2:].max(1).mean()
        normalized_cx = cxs / mean_wh
        normalized_cy = cys / mean_wh

    if roi_size == 'dynamic':
        size = bboxes[:, 2:].max(axis=1)
        bboxes[:, 0] = cxs - size * 3
        bboxes[:, 1] = cys - size
        bboxes[:, 2] = cxs + size * 3
        bboxes[:, 3] = cys + size * 8

    elif roi_size is not None:
        height, width = roi_size
        height = float(height)
        width = float(width)
        bboxes[:, 0] = cxs - width / 2
        bboxes[:, 1] = cys - height / 2
        bboxes[:, 2] = cxs + width / 2
        bboxes[:, 3] = cys + height / 2
    else:
        bboxes[:, 2:] += bboxes[:, :2]  # xyxy

    # rescale bbox
    h, w = image_size
    org_h, org_w = original_image_size
    bboxes[:, 0] = bboxes[:, 0] * w / org_w
    bboxes[:, 1] = bboxes[:, 1] * h / org_h
    bboxes[:, 2] = bboxes[:, 2] * w / org_w
    bboxes[:, 3] = bboxes[:, 3] * h / org_h

    for i, unique_id in enumerate(unique_ids):
        if unique_id in helmet_ids:
            idx = helmet_ids.index(unique_id)
            label_bboxes[i] = bboxes[idx]
            label_coords[i, 0] = normalized_cx[idx]
            label_coords[i, 1] = normalized_cy[idx]
    return label_bboxes, label_coords, heatmap

# from CenterNet


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

# from CenterNet


def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def draw_centernet_label(img_h, img_w, center, down_ratio, radius=2, k=1):
    label_h, label_w = img_h // down_ratio, img_w // down_ratio
    heatmap = np.zeros((label_h, label_w))
    if center is None:
        return heatmap
    cx, cy = center
    cx /= down_ratio
    cy /= down_ratio
    heatmap = draw_umich_gaussian(heatmap, (cx, cy), radius, k)
    return heatmap


def draw_centernet_labels(img_h, img_w, centers, down_ratio, radius=2, k=1):
    label_h, label_w = img_h // down_ratio, img_w // down_ratio
    heatmap = np.zeros((label_h, label_w), np.float32)
    if centers is None:
        return heatmap
    for center in centers:
        cx, cy = center
        cx /= down_ratio
        cy /= down_ratio
        heatmap = draw_umich_gaussian(heatmap, (cx, cy), radius, k)
    return heatmap


def preprocess_bboxes_for_aug(bboxes, image_size):
    """albumentationははみ出しているbboxを受け付けないのでclipする"""
    h_lim = image_size[0]
    w_lim = image_size[1]

    bboxes[:, 0] = np.clip(bboxes[:, 0], 0, w_lim)
    bboxes[:, 1] = np.clip(bboxes[:, 1], 0, h_lim)
    bboxes[:, 2] = np.clip(bboxes[:, 2], 0, w_lim)
    bboxes[:, 3] = np.clip(bboxes[:, 3], 0, h_lim)
    is_valid = bboxes.sum(axis=1) > 0
    bboxes_with_index = np.concatenate([bboxes, np.arange(len(bboxes))[:, None]], axis=1)
    bboxes_with_index = bboxes_with_index[is_valid]

    return bboxes_with_index


def reorder_augmented_bboxes(original_bboxes, augmented_bboxes):
    """augしたbboxは順番が変わっているもしくは消滅している可能性があるので、正しい位置に戻す"""
    bboxes = np.zeros_like(original_bboxes)  # 消滅したものがそのまま残っているとこまるのでzeros_like
    for bb in augmented_bboxes:
        bboxes[int(bb[-1]), :4] = np.array(bb[:4])
    return bboxes


def preprocess_image_sequence(images):
    images = [to_torch_tensor(normalize_img(img)) for img in images]
    return torch.stack(images, dim=0)


def preprocess_mask_sequence(masks):
    masks = [to_torch_tensor(img) for img in masks]
    return torch.stack(masks, dim=0)


class TrainDataset(Dataset):
    def __init__(self, cfg, fold, mode="train"):

        self.cfg = cfg
        self.mode = mode
        self.df = setup_df(cfg.df_path, fold, mode)

        id_path = f'{cfg.data_dir}/{cfg.unique_ids_dict_name}_{mode}_fold{fold}.pkl'
        distance_path = f'{cfg.data_dir}/{cfg.distance_dict_name}_{mode}_fold{fold}.pkl'
        inter_contact_path = f'{cfg.data_dir}/{cfg.inter_contact_dict_name}_{mode}_fold{fold}.pkl'
        ground_contact_path = f'{cfg.data_dir}/{cfg.ground_contact_dict_name}_{mode}_fold{fold}.pkl'
        helmet_dict_path = f'{cfg.data_dir}/{cfg.helmet_dict_name}_{mode}_fold{fold}.pkl'
        track_dict_path = f'{cfg.data_dir}/{cfg.track_dict_name}_{mode}_fold{fold}.pkl'
        pair_dict_path = f'{cfg.data_dir}/{cfg.pair_dict_name}_{mode}_fold{fold}.pkl'

        self.unique_ids_dict = load_pickle(id_path)
        self.distance_dict = load_pickle(distance_path)
        self.inter_contact_dict = load_pickle(inter_contact_path)
        self.ground_contact_dict = load_pickle(ground_contact_path)
        self.helmet_dict = load_pickle(helmet_dict_path)
        self.track_dict = load_pickle(track_dict_path)
        self.pair_dict = load_pickle(pair_dict_path)

        self.image_size = self.cfg.image_size
        self.original_image_size = self.cfg.original_image_size
        self.roi_size = self.cfg.roi_size
        self.normalize_coords = self.cfg.normalize_coords

        self.duration = self.cfg.duration
        assert self.duration % 2 == 1, f'duration has to be odd number.'
        self.half_duration = (self.duration - 1) // 2

        height, width = cfg.image_size
        self.resizer = video_transforms.Resize((height, width))

        self.enable_hflip = cfg.enable_hflip
        self.video_transforms = cfg.transforms
        self.image_transforms = cfg.image_transforms

        # self.heatmap_type = cfg.heatmap_type

    def __len__(self):
        return len(self.df)

    def _get_item(self, game_play, frame):
        unique_ids = self.unique_ids_dict.get((game_play, frame), None).copy()
        distance_labels = self.distance_dict.get((game_play, frame), None).copy()
        track_feats = self.track_dict.get((game_play, frame), None).copy()
        pair_feats = self.pair_dict.get((game_play, frame), None).copy()
        inter_contacts = self.inter_contact_dict.get((game_play, frame), []).copy()
        ground_contacts = self.ground_contact_dict.get((game_play, frame), []).copy()
        assert unique_ids is not None
        assert distance_labels is not None
        assert track_feats is not None
        assert pair_feats is not None

        # labels
        inputs = {}
        inputs['unique_ids'] = np.array(unique_ids)
        inputs['distance'] = get_distance_matrix(distance_labels, unique_ids)
        inputs['inter'] = get_inter_labels(inter_contacts, unique_ids)
        inputs['ground'] = get_g_labels(ground_contacts, unique_ids)
        inputs['track'] = get_track_feats(track_feats, unique_ids, self.cfg.num_track_features)
        inputs['pair'] = get_pair_feats(pair_feats, unique_ids, self.cfg.num_pair_features)

        if self.cfg.enable_frame_noise and (self.mode == 'train'):
            frame = frame + random.randint(-5, 5)

        # helmets
        for view in VIEWS:
            inputs[view + '_rois'] = []
            inputs[view + '_coords'] = []
            inputs[view + '_mask'] = []
            inputs[view + '_heatmap'] = []
            for i in range(-self.half_duration, self.half_duration+1):
                frame_id = frame + i
                frame_helmets = self.helmet_dict.get((game_play, frame_id, view), None)
                bboxes, coords, heatmap = get_bboxes(frame_helmets, unique_ids, self.image_size,
                                                     self.original_image_size, self.roi_size, self.normalize_coords)
                masks = bboxes.sum(axis=1) > 0  # has bbox flag
                # print('helmet:', frame_helmets)
                # print('bboxes:', bboxes[masks])
                inputs[view + '_rois'].append(bboxes)
                inputs[view + '_coords'].append(coords)
                inputs[view + '_mask'].append(masks)
                inputs[view + '_heatmap'].append(heatmap)
            inputs[view + '_rois'] = np.stack(inputs[view + '_rois'], axis=0)
            inputs[view + '_coords'] = np.stack(inputs[view + '_coords'], axis=0)
            inputs[view + '_mask'] = np.stack(inputs[view + '_mask'], axis=0).astype(np.uint8)
            inputs[view + '_heatmap'] = np.stack(inputs[view + '_heatmap'], axis=0)

        # images
        for view in VIEWS:
            images = load_adj_frames(self.cfg.image_dir, game_play, view, frame, k=self.duration)
            if any([img is None for img in images]):
                # print(f'frame{frame} may not exist')
                random_idx = random.randint(0, len(self)-1)
                return self.__getitem__(random_idx)
            images = self.video_transforms(images)
            images = [np.asarray(img) for img in images]

            if self.image_transforms is not None:
                seq_len = len(images)
                aug_inputs = {}

                def zero2blank(i):
                    if i == 0:
                        return ""
                    else:
                        return i
                aug_inputs.update({f'image{zero2blank(i)}': image for i, image in enumerate(images)})
                view_heatmap = inputs[view + '_heatmap']
                aug_inputs.update({f'mask{zero2blank(i)}': mask for i, mask in enumerate(view_heatmap)})
                aug_inputs.update({f'bboxes{zero2blank(i)}': preprocess_bboxes_for_aug(
                    bboxes, self.image_size) for i, bboxes in enumerate(inputs[view + '_rois'])})
                augmented = self.image_transforms(**aug_inputs)
                augmented_seq_images = [augmented[f'image{zero2blank(i)}'] for i in range(seq_len)]
                augmented_seq_masks = [augmented[f'mask{zero2blank(i)}'] for i in range(seq_len)]
                augmented_seq_bboxes = [augmented[f'bboxes{zero2blank(i)}'] for i in range(seq_len)]
                augmented_seq_bboxes = [reorder_augmented_bboxes(bboxes, augmented_bboxes) for bboxes, augmented_bboxes in zip(
                    inputs[view + '_rois'], augmented_seq_bboxes)]
                inputs[view + "_image"] = preprocess_image_sequence(augmented_seq_images)
                inputs[view + "_heatmap"] = preprocess_mask_sequence(augmented_seq_masks)
                inputs[view + '_rois'] = np.stack(augmented_seq_bboxes)
            else:
                inputs[view + "_image"] = preprocess_image_sequence(images)

        inputs['game_play'] = game_play
        inputs['frame'] = frame
        return inputs

    def __getitem__(self, idx):
        frame_row = self.df.iloc[idx]
        return self._get_item(frame_row.game_play, frame_row.frame)


def get_unique_ids_dict_by_step(label_df):
    unique_ids_dict = label_df.groupby(['game_play', 'step'], observed=True)['nfl_player_id_1'].unique().to_dict()

    for k, v in unique_ids_dict.items():
        v = sorted(v)
        if len(v) < 22:
            v = v + [-1] * (22 - len(v))  # padding
        unique_ids_dict[k] = v
    return unique_ids_dict


def load_image(image_path, image_size):
    img0 = load_cv2_image(image_path)
    img0 = cv2.resize(img0, (image_size[1], image_size[0]))
    # img0 = normalize_img(img0)  # .astype(np.float16)
    # img0 = to_torch_tensor(img0)
    return img0


def convert_to_tensor(img0):
    img0 = normalize_img(img0)  # .astype(np.float16)
    img0 = to_torch_tensor(img0)
    return img0


class ValidDataFrameDataset(Dataset):
    def __init__(self, cfg, fold, mode="train", use_all_frames=False, is_flip=False):
        self.cfg = cfg
        self.image_dir = cfg.image_dir
        self.is_flip = is_flip
        self.image_size = cfg.image_size
        if self.is_flip:
            self.flipper = A.Compose([A.HorizontalFlip(p=1.0), ], bbox_params=A.BboxParams(format='pascal_voc'))

        if use_all_frames:
            df_path = cfg.df_path.replace('sample', 'frame_sample')
            distance_dict_name = 'frame_' + cfg.distance_dict_name
            track_dict_name = 'frame_' + cfg.track_dict_name
            pair_dict_name = 'frame_' + cfg.pair_dict_name
            helmet_dict_name = cfg.helmet_dict_name
        else:
            df_path = cfg.df_path
            distance_dict_name = cfg.distance_dict_name
            track_dict_name = cfg.track_dict_name
            pair_dict_name = cfg.pair_dict_name
            helmet_dict_name = cfg.helmet_dict_name

        self.sample_df = setup_df(df_path, fold, mode)
        if fold == -1:
            self.distance_dict = {}
            self.helmet_dict = {}
            self.track_dict = {}
            self.pair_dict = {}
            for i in range(4):
                print('load data dict for fold', i)
                self.distance_dict.update(load_pickle(f'{cfg.data_dir}/{distance_dict_name}_{mode}_fold{i}.pkl'))
                self.helmet_dict.update(load_pickle(f'{cfg.data_dir}/{helmet_dict_name}_{mode}_fold{i}.pkl'))
                self.track_dict.update(load_pickle(f'{cfg.data_dir}/{track_dict_name}_{mode}_fold{i}.pkl'))
                self.pair_dict.update(load_pickle(f'{cfg.data_dir}/{pair_dict_name}_{mode}_fold{i}.pkl'))
        else:
            distance_path = f'{cfg.data_dir}/{distance_dict_name}_{mode}_fold{fold}.pkl'
            helmet_dict_path = f'{cfg.data_dir}/{helmet_dict_name}_{mode}_fold{fold}.pkl'
            track_dict_path = f'{cfg.data_dir}/{track_dict_name}_{mode}_fold{fold}.pkl'
            pair_dict_path = f'{cfg.data_dir}/{pair_dict_name}_{mode}_fold{fold}.pkl'
            self.distance_dict = load_pickle(distance_path)
            self.helmet_dict = load_pickle(helmet_dict_path)
            self.track_dict = load_pickle(track_dict_path)
            self.pair_dict = load_pickle(pair_dict_path)

        self.df = setup_df(cfg.label_df_path, fold, mode)
        self.pairs_gb = self.df.groupby(['game_play', 'step'], observed=True)['nfl_player_id_1', 'nfl_player_id_2']
        self.unique_id_dict = get_unique_ids_dict_by_step(self.df)

    def __len__(self):
        return len(self.sample_df)

    def __getitem__(self, idx):
        row = self.sample_df.iloc[idx]
        game_play = row['game_play']
        step = row['step']
        frame = row['frame']
        unique_ids = self.unique_id_dict[(game_play, step)]
        distance_labels = self.distance_dict.get((game_play, frame), None).copy()
        track_feats = self.track_dict.get((game_play, frame), None).copy()
        pair_feats = self.pair_dict.get((game_play, frame), None).copy()

        inputs = {}
        inputs['distance'] = get_distance_matrix(distance_labels, unique_ids)
        inputs['track'] = get_track_feats(track_feats, unique_ids, self.cfg.num_track_features)
        inputs['pair'] = get_pair_feats(pair_feats, unique_ids, self.cfg.num_pair_features)

        for view in VIEWS:
            frame_helmets = self.helmet_dict.get((game_play, frame, view), None)
            bboxes, coords, heatmap = get_bboxes(frame_helmets, unique_ids, self.cfg.image_size,
                                                 self.cfg.original_image_size, self.cfg.roi_size, self.cfg.normalize_coords)
            masks = bboxes.sum(axis=1) > 0  # has bbox flag
            inputs[view + '_rois'] = torch.tensor(bboxes[None].astype(np.float32))  # add seq dim
            inputs[view + '_coords'] = torch.tensor(coords[None].astype(np.float32))  # add seq dim
            inputs[view + '_mask'] = torch.tensor(masks[None].astype(np.float32))  # add seq dim
            inputs[view + '_heatmap'] = to_torch_tensor(heatmap.astype(np.float32))[None]  # add seq dim

            img_path = f'{self.image_dir}/{game_play}_{view}/{frame:06}.jpg'
            inputs[view + "_image"] = load_image(img_path, self.cfg.image_size)
            inputs[view + "_image"] = convert_to_tensor(inputs[view + "_image"])[None]  # add seq dim
            inputs[view + '_rois']

            if self.is_flip:
                height, width = self.image_size
                inputs[view + "_image"] = inputs[view + "_image"].flip(-1)
                inputs[view + "_heatmap"] = inputs[view + "_heatmap"].flip(-1)
                invalid_index = inputs[view + '_rois'][0].sum(-1) == 0
                x_max = width - inputs[view + '_rois'][0, :, 0]
                x_min = width - inputs[view + '_rois'][0, :, 2]
                inputs[view + '_rois'][0, :, 0] = x_min
                inputs[view + '_rois'][0, :, 2] = x_max
                inputs[view + '_rois'][0, invalid_index, :] = 0
                # coordsは距離計算にしか使ってないのでflipしなくてもいいはず

        inputs['game_play'] = game_play
        inputs['step'] = step.astype(np.int32)
        inputs['frame'] = frame.astype(np.int32)
        inputs['unique_ids'] = np.array(unique_ids).astype(np.int32)
        return inputs


class TestDataFrameDataset(Dataset):
    def __init__(self, cfg, mode="test", use_all_frames=False, game_play=None):
        fold = -1
        self.cfg = cfg
        self.image_dir = cfg.image_dir

        if use_all_frames:
            df_path = cfg.df_path.replace('sample', 'frame_sample')
            distance_dict_name = 'frame_' + cfg.distance_dict_name
            track_dict_name = 'frame_' + cfg.track_dict_name
            pair_dict_name = 'frame_' + cfg.pair_dict_name
            helmet_dict_name = cfg.helmet_dict_name
        else:
            df_path = cfg.df_path
            distance_dict_name = cfg.distance_dict_name
            track_dict_name = cfg.track_dict_name
            pair_dict_name = cfg.pair_dict_name
            helmet_dict_name = cfg.helmet_dict_name

        self.sample_df = setup_df(df_path, fold, mode)
        distance_path = f'{cfg.data_dir}/{distance_dict_name}.pkl'
        helmet_dict_path = f'{cfg.data_dir}/{helmet_dict_name}.pkl'
        track_dict_path = f'{cfg.data_dir}/{track_dict_name}.pkl'
        pair_dict_path = f'{cfg.data_dir}/{pair_dict_name}.pkl'
        self.distance_dict = load_pickle(distance_path)
        self.helmet_dict = load_pickle(helmet_dict_path)
        self.track_dict = load_pickle(track_dict_path)
        self.pair_dict = load_pickle(pair_dict_path)

        self.df = setup_df(cfg.label_df_path, fold, mode)
        self.pairs_gb = self.df.groupby(['game_play', 'step'], observed=True)['nfl_player_id_1', 'nfl_player_id_2']
        self.unique_id_dict = get_unique_ids_dict_by_step(self.df)

        self.sample_by_game_play = False
        if game_play is not None:
            self.sample_by_game_play = True
            self.game_sample_df = self.sample_df.query('game_play == @game_play').reset_index(drop=True)

    def set_game_play(self, game_play):
        self.sample_by_game_play = True
        self.game_sample_df = self.sample_df.query('game_play == @game_play').reset_index(drop=True)

    def __len__(self):
        if self.sample_by_game_play:
            return len(self.game_sample_df)
        else:
            return len(self.sample_df)

    def __getitem__(self, idx):
        if self.sample_by_game_play:
            row = self.game_sample_df.iloc[idx]
        else:
            row = self.sample_df.iloc[idx]
        game_play = row['game_play']
        step = row['step']
        frame = row['frame']
        unique_ids = self.unique_id_dict[(game_play, step)]
        distance_labels = self.distance_dict.get((game_play, frame), None).copy()
        track_feats = self.track_dict.get((game_play, frame), None).copy()
        pair_feats = self.pair_dict.get((game_play, frame), None).copy()

        inputs = {}
        inputs['distance'] = get_distance_matrix(distance_labels, unique_ids)
        inputs['track'] = get_track_feats(track_feats, unique_ids, self.cfg.num_track_features)
        inputs['pair'] = get_pair_feats(pair_feats, unique_ids, self.cfg.num_pair_features)

        for view in VIEWS:
            frame_helmets = self.helmet_dict.get((game_play, frame, view), None)
            bboxes, coords, heatmap = get_bboxes(frame_helmets, unique_ids, self.cfg.image_size,
                                                 self.cfg.original_image_size, self.cfg.roi_size, self.cfg.normalize_coords)
            masks = bboxes.sum(axis=1) > 0  # has bbox flag
            inputs[view + '_rois'] = torch.tensor(bboxes[None].astype(np.float32))  # add seq dim
            inputs[view + '_coords'] = torch.tensor(coords[None].astype(np.float32))  # add seq dim
            inputs[view + '_mask'] = torch.tensor(masks[None].astype(np.float32))  # add seq dim
            inputs[view + '_heatmap'] = to_torch_tensor(heatmap.astype(np.float32))[None]  # add seq dim

            img_path = f'{self.image_dir}/{game_play}_{view}/{frame:06}.jpg'
            inputs[view + "_image"] = load_image(img_path, self.cfg.image_size)
            inputs[view + "_image"] = convert_to_tensor(inputs[view + "_image"])[None]  # add seq dim

        inputs['game_play'] = game_play
        inputs['step'] = step.astype(np.int32)
        inputs['frame'] = frame.astype(np.int32)
        inputs['unique_ids'] = np.array(unique_ids).astype(np.int32)
        return inputs


def get_train_dataloader(cfg, fold):
    dataset = TrainDataset(cfg, fold=fold, mode="train")
    train_dataloader = DataLoader(
        dataset,
        sampler=None,
        shuffle=True,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=False,
        drop_last=True
    )
    return train_dataloader


def get_val_dataframe_dataloader(cfg, fold, use_all_frames=False, is_flip=False):
    dataset = ValidDataFrameDataset(cfg, fold=fold, mode="valid", use_all_frames=use_all_frames, is_flip=is_flip)
    val_dataloader = DataLoader(
        dataset,
        sampler=None,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=False,
        drop_last=False
    )
    return val_dataloader


def get_test_dataframe_dataloader(cfg, use_all_frames=False, game_play=None):
    dataset = TestDataFrameDataset(cfg, mode="test", use_all_frames=use_all_frames, game_play=game_play)
    val_dataloader = DataLoader(
        dataset,
        sampler=None,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=False,
        drop_last=False
    )
    return val_dataloader

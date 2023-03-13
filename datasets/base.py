import random
import os
import pandas as pd
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from datasets import video_transforms

from .common import setup_df, normalize_img, to_torch_tensor, load_cv2_image, load_cv2_image, load_pickle, load_pil_image

VIEWS = ['Endzone', 'Sideline']
GROUND_ID = 0


def get_unique_ids_dict(label_df):
    unique_ids_dict = label_df.groupby(['game_play', 'frame'])['nfl_player_id_1'].unique().to_dict()

    for k, v in unique_ids_dict.items():
        v = sorted(v)
        if len(v) < 22:
            v = v + [-1] * (22 - len(v))  # padding
        unique_ids_dict[k] = v
    return unique_ids_dict


def get_inter_contact_dict(label_df):
    gb = label_df.query('contact == 1 & nfl_player_id_2 > 0').groupby(
        ['game_play', 'frame'])[['nfl_player_id_1', 'nfl_player_id_2']]
    contact_dict = {k: v[['nfl_player_id_1', 'nfl_player_id_2']].values for k, v in gb}
    return contact_dict


def get_ground_contact_dict(label_df):
    gb = label_df.query('contact == 1 & nfl_player_id_2 == 0').groupby(
        ['game_play', 'frame'])[['nfl_player_id_1', 'nfl_player_id_2']]
    contact_dict = {k: v['nfl_player_id_1'].values for k, v in gb}
    return contact_dict


def get_helmet_dict(helmet_df):
    cols = ['nfl_player_id', 'left', 'width', 'top', 'height']
    gb = helmet_df.groupby(['game_play', 'frame', 'view'])
    helmet_dict = {k: v[cols] for k, v in gb}
    return helmet_dict


def get_labels_for_frame(label_df, game_play, frame):
    selected = label_df.query('game_play == @game_play & frame == @frame')
    return selected


def load_single_frame(base_image_dir, game_play, view, frame):
    image_dir = f'{base_image_dir}/{game_play}_{view}/'
    img_path = os.path.join(image_dir, f'{frame:06}.jpg')
    if os.path.exists(img_path):
        return load_pil_image(img_path)
    else:
        return None


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
    return label_matrix.copy()


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


HELMET_SCALE = 16  # trainのheight, widthのmedianくらい


def get_bboxes(frame_helmets, unique_ids, image_size, original_image_size, roi_size=None, normalize_coords=False):
    label_bboxes = np.zeros((len(unique_ids), 4), np.float32)
    label_coords = np.zeros((len(unique_ids), 2), np.float32)
    if frame_helmets is None:
        return label_bboxes, label_coords
    helmet_ids = frame_helmets[:, -1].tolist()
    bboxes = frame_helmets[:, :-1].astype(np.float32).copy()
    cx = bboxes[:, 0] + bboxes[:, 2] / 2
    cy = bboxes[:, 1] + bboxes[:, 3] / 2

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
        normalized_cx = cx / mean_wh
        normalized_cy = cy / mean_wh

    if roi_size == 'dynamic':
        size = bboxes[:, 2:].max(axis=1)
        bboxes[:, 0] = cx - size * 3
        bboxes[:, 1] = cy - size
        bboxes[:, 2] = cx + size * 3
        bboxes[:, 3] = cy + size * 8

    elif roi_size is not None:
        height, width = roi_size
        height = float(height)
        width = float(width)
        bboxes[:, 0] = cx - width / 2
        bboxes[:, 1] = cy - height / 2
        bboxes[:, 2] = cx + width / 2
        bboxes[:, 3] = cy + height / 2
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
    return label_bboxes, label_coords

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


class TrainDataset(Dataset):
    def __init__(self, cfg, fold, mode="train"):

        self.cfg = cfg
        self.mode = mode
        self.df = setup_df(cfg.df_path, fold, mode)

        id_path = f'../input/{cfg.unique_ids_dict_name}_{mode}_fold{fold}.pkl'
        inter_contact_path = f'../input/{cfg.inter_contact_dict_name}_{mode}_fold{fold}.pkl'
        ground_contact_path = f'../input/{cfg.ground_contact_dict_name}_{mode}_fold{fold}.pkl'
        helmet_dict_path = f'../input/{cfg.helmet_dict_name}_{mode}_fold{fold}.pkl'
        track_dict_path = f'../input/{cfg.track_dict_name}_{mode}_fold{fold}.pkl'

        print('load unique_ids dict from:', id_path)
        print('load inter_contact dict from:', inter_contact_path)
        print('load ground_contact dict from:', ground_contact_path)
        print('load helmet_dict dict from:', helmet_dict_path)
        print('load track_dict dict from:', track_dict_path)

        self.unique_ids_dict = load_pickle(id_path)
        self.inter_contact_dict = load_pickle(inter_contact_path)
        self.ground_contact_dict = load_pickle(ground_contact_path)
        self.helmet_dict = load_pickle(helmet_dict_path)
        self.track_dict = load_pickle(track_dict_path)

        self.image_size = self.cfg.image_size
        self.original_image_size = self.cfg.original_image_size
        self.roi_size = self.cfg.roi_size
        self.normalize_coords = self.cfg.normalize_coords
        self.num_track_features = self.cfg.num_track_features

        self.duration = self.cfg.duration
        assert self.duration % 2 == 1, f'duration has to be odd number.'
        self.half_duration = (self.duration - 1) // 2

        height, width = cfg.image_size
        self.resizer = video_transforms.Resize((height, width))

        self.enable_hflip = cfg.enable_hflip
        if cfg.transforms == None:
            self.transforms = video_transforms.Compose([
                video_transforms.ColorJitter(brightness=0.2, contrast=0.1),
                video_transforms.Resize((height, width)),
            ])
        else:
            self.transforms = cfg.transforms

        self.image_transforms = cfg.image_transforms

    def __len__(self):
        return len(self.df)

    def _get_item(self, game_play, frame):
        unique_ids = self.unique_ids_dict.get((game_play, frame), None).copy()
        assert unique_ids is not None
        inter_contacts = self.inter_contact_dict.get((game_play, frame), []).copy()
        ground_contacts = self.ground_contact_dict.get((game_play, frame), []).copy()
        track_feats = self.track_dict.get((game_play, frame), []).copy()

        # labels
        inputs = {}
        inputs['unique_ids'] = np.array(unique_ids)
        inputs['inter'] = get_inter_labels(inter_contacts, unique_ids)
        inputs['ground'] = get_g_labels(ground_contacts, unique_ids)
        inputs['track'] = get_track_feats(track_feats, unique_ids, self.num_track_features)

        if self.cfg.fix_coords_scale:
            inputs['track'][:, 0] *= 120 / 100
            inputs['track'][:, 1] *= 53.3 / 100

        if self.cfg.enable_frame_noise and (self.mode == 'train'):
            frame = frame + random.randint(-5, 5)

        # helmets
        for view in VIEWS:
            inputs[view + '_rois'] = []
            inputs[view + '_coords'] = []
            inputs[view + '_mask'] = []
            for i in range(-self.half_duration, self.half_duration+1):
                frame_id = frame + i
                frame_helmets = self.helmet_dict.get((game_play, frame_id, view), None)
                bboxes, coords = get_bboxes(frame_helmets, unique_ids, self.image_size,
                                            self.original_image_size, self.roi_size, self.normalize_coords)
                masks = bboxes.sum(axis=1) > 0  # has bbox flag
                # print('helmet:', frame_helmets)
                # print('bboxes:', bboxes[masks])
                inputs[view + '_rois'].append(bboxes)
                inputs[view + '_coords'].append(coords)
                inputs[view + '_mask'].append(masks)
            inputs[view + '_rois'] = np.stack(inputs[view + '_rois'], axis=0)
            inputs[view + '_coords'] = np.stack(inputs[view + '_coords'], axis=0)
            inputs[view + '_mask'] = np.stack(inputs[view + '_mask'], axis=0).astype(np.uint8)

        # images
        for view in VIEWS:
            images = load_adj_frames(self.cfg.image_dir, game_play, view, frame, k=self.duration)
            if any([img is None for img in images]):
                # print(f'frame{frame} may not exist')
                random_idx = random.randint(0, len(self)-1)
                return self.__getitem__(random_idx)
            images = self.transforms(images)
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
                aug_inputs.update({f'bboxes{zero2blank(i)}': preprocess_bboxes_for_aug(
                    bboxes, self.image_size) for i, bboxes in enumerate(inputs[view + '_rois'])})
                augmented = self.image_transforms(**aug_inputs)
                augmented_seq_images = [augmented[f'image{zero2blank(i)}'] for i in range(seq_len)]
                augmented_seq_bboxes = [augmented[f'bboxes{zero2blank(i)}'] for i in range(seq_len)]
                augmented_seq_bboxes = [reorder_augmented_bboxes(bboxes, augmented_bboxes) for bboxes, augmented_bboxes in zip(
                    inputs[view + '_rois'], augmented_seq_bboxes)]
                inputs[view + "_image"] = preprocess_image_sequence(augmented_seq_images)
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
    unique_ids_dict = label_df.groupby(['game_play', 'step'])['nfl_player_id_1'].unique().to_dict()

    for k, v in unique_ids_dict.items():
        v = sorted(v)
        if len(v) < 22:
            v = v + [-1] * (22 - len(v))  # padding
        unique_ids_dict[k] = v
    return unique_ids_dict


def calc_distance(x1, y1, x2, y2):
    return np.sqrt(np.square(x2 - x1) + np.square(y2 - y1))


def merge_tracking(df, tr, use_cols=["x_position", "y_position", ]):
    key_cols = ["nfl_player_id", "step", "game_play"]
    use_cols = [c for c in use_cols if c in tr.columns]

    dst = pd.merge(
        df,
        tr[key_cols + use_cols].rename(columns={c: c + "_1" for c in use_cols}),
        left_on=["nfl_player_id_1", "step", "game_play"],
        right_on=key_cols,
        how="left"
    ).drop("nfl_player_id", axis=1)
    dst = pd.merge(
        dst,
        tr[key_cols + use_cols].rename(columns={c: c + "_2" for c in use_cols}),
        left_on=["nfl_player_id_2", "step", "game_play"],
        right_on=key_cols,
        how="left"
    ).drop("nfl_player_id", axis=1)

    dst["distance"] = calc_distance(
        dst["x_position_1"],
        dst["y_position_1"],
        dst["x_position_2"],
        dst["y_position_2"])

    return dst


class ValidDataFrameDataset(Dataset):
    def __init__(self, cfg, fold, mode="train", use_all_frames=None):
        self.cfg = cfg
        self.image_dir = cfg.image_dir
        self.num_track_features = self.cfg.num_track_features
        use_all_frames = use_all_frames or cfg.use_all_frames
        if use_all_frames:
            print('use all frames.')
            self.sample_df = setup_df(cfg.sample_df_path, fold, mode)
        else:
            print('use only exact frame at the steps.')
            self.sample_df = setup_df(cfg.sample_df_by_step_path, fold, mode)

        if fold == -1:
            self.helmet_dict = {}
            self.track_dict = {}
            for i in range(4):
                print('load helmet and track dict for fold', i)
                self.helmet_dict.update(load_pickle(f'../input/{cfg.helmet_dict_name}_valid_fold{i}.pkl'))
                self.track_dict.update(load_pickle(f'../input/{cfg.track_dict_by_step_name}_valid_fold{i}.pkl'))
        else:
            self.helmet_dict = load_pickle(f'../input/{cfg.helmet_dict_name}_{mode}_fold{fold}.pkl')
            self.track_dict = load_pickle(f'../input/{cfg.track_dict_by_step_name}_{mode}_fold{fold}.pkl')
        self.df = setup_df(cfg.label_df_path, fold, mode)
        track_df = pd.read_csv(cfg.tracking_df_path)
        self.df = merge_tracking(self.df, track_df, ["x_position", "y_position", ])
        self.pairs_gb = self.df.groupby(['game_play', 'step'])['nfl_player_id_1', 'nfl_player_id_2']

        self.unique_id_dict = get_unique_ids_dict_by_step(self.df)
        # id_gb = df.groupby(['game_play', 'step'])['nfl_player_id_1']
        # self.unique_id_dict = {k: v.unique().tolist() for k, v in id_gb}

    def __len__(self):
        return len(self.sample_df)

    def __getitem__(self, idx):
        row = self.sample_df.iloc[idx]
        game_play = row['game_play']
        step = row['step']
        frame = row['frame']

        inputs = {}
        unique_ids = self.unique_id_dict[(game_play, step)]

        track_feats = self.track_dict[(game_play, step)]
        track_feats = get_track_feats(track_feats, unique_ids, self.num_track_features)
        inputs['track'] = torch.tensor(track_feats.astype(np.float32))
        if self.cfg.fix_coords_scale:
            inputs['track'][:, 0] *= 120 / 100
            inputs['track'][:, 1] *= 53.3 / 100
        for view in VIEWS:
            frame_helmets = self.helmet_dict[(game_play, frame, view)]
            bboxes, coords = get_bboxes(frame_helmets, unique_ids, self.cfg.image_size,
                                        self.cfg.original_image_size, self.cfg.roi_size, self.cfg.normalize_coords)
            masks = bboxes.sum(axis=1) > 0  # has bbox flag
            inputs[view + '_rois'] = torch.tensor(bboxes[None].astype(np.float32))  # add seq dim
            inputs[view + '_coords'] = torch.tensor(coords[None].astype(np.float32))  # add seq dim
            inputs[view + '_mask'] = torch.tensor(masks[None].astype(np.float32))  # add seq dim

            img_path = f'{self.image_dir}/{game_play}_{view}/{frame:06}.jpg'
            inputs[view + "_image"] = load_image(img_path, self.cfg.image_size)[None]  # add seq dim
        inputs['game_play'] = game_play
        inputs['step'] = step.astype(np.int32)
        inputs['frame'] = frame.astype(np.int32)
        inputs['unique_ids'] = np.array(unique_ids).astype(np.int32)
        return inputs


def zero2blank(i):
    if i == 0:
        return ""
    else:
        return i


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


def load_image(image_path, image_size):
    img0 = load_cv2_image(image_path)
    img0 = cv2.resize(img0, (image_size[1], image_size[0]))
    img0 = normalize_img(img0)  # .astype(np.float16)
    img0 = to_torch_tensor(img0)
    return img0


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


def get_val_dataloader(cfg, fold):
    dataset = TrainDataset(cfg, fold=fold, mode="valid")
    val_dataloader = DataLoader(
        dataset,
        sampler=None,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=False,
        drop_last=False
    )
    return val_dataloader


def get_val_dataframe_dataloader(cfg, fold, use_all_frames=None):
    dataset = ValidDataFrameDataset(cfg, fold=fold, mode="valid", use_all_frames=use_all_frames)
    val_dataloader = DataLoader(
        dataset,
        sampler=None,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=False,
        drop_last=False
    )
    return val_dataloader

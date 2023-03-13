from tqdm.auto import tqdm
from torch.cuda.amp import autocast
from exp048 import (
    cfg, get_model)
from datasets.common import load_cv2_image, normalize_img, to_torch_tensor
from datasets.base import get_bboxes
import torch.nn as nn
import torch
import pandas as pd
import numpy as np
import cv2
import os
import glob
import sys  # isort: skip
# sys.path.append('../input/timmmaster') # isort: skip
sys.path.insert(0, '../input/nfl2022-setup/kaggle-nfl2022')  # isort: skip


cfg.model.pretrained_path = None

VIEWS = ['Endzone', 'Sideline']
BASE_DIR = '/temp/test_frames/'
# BASE_DIR = '../input/train_frames/'


def step_to_frame(step):
    step0_frame = 300
    fps_frame = 59.95
    fps_step = 10
    return int(round(step * fps_frame / fps_step + step0_frame))


def cast_player_id(df):
    # RAM消費を減らしたいので、Gを-1に置換して整数で持つ。
    if "nfl_player_id_2" in df.columns:
        df.loc[df["nfl_player_id_2"] == "G", "nfl_player_id_2"] = "-1"

    for c in ["nfl_player_id", "nfl_player_id_1", "nfl_player_id_2"]:
        if c in df.columns:
            df[c] = df[c].astype(np.int32)

    return df


def expand_contact_id(df):
    """
    Splits out contact_id into seperate columns.
    """
    df["game_play"] = df["contact_id"].str[:12]
    df["game"] = df["game_play"].str[:5].astype(int)
    df["step"] = df["contact_id"].str.split("_").str[-3].astype("int")
    df["nfl_player_id_1"] = df["contact_id"].str.split("_").str[-2]
    df["nfl_player_id_2"] = df["contact_id"].str.split("_").str[-1]
    return cast_player_id(df)


class Models(nn.Module):
    def __init__(self, weight_paths):
        super().__init__()
        self.models = [get_model(cfg, p) for p in sorted(weight_paths)]
        [m.eval() for m in self.models]

        self.output_keys = ['ground', 'ground_masks', 'inter', 'inter_masks']

    def ensemble(self, inputs):
        with torch.no_grad():
            outputs = [m(inputs) for m in self.models]
        ensemble_output = {}
        ensemble_output['ground_masks'] = outputs[0]['ground_masks']
        ensemble_output['ground'] = torch.mean(torch.stack([output['ground'] for output in outputs], dim=0), dim=0)
        ensemble_output['inter_masks'] = outputs[0]['inter_masks']
        ensemble_output['inter'] = torch.mean(torch.stack([output['inter'] for output in outputs], dim=0), dim=0)
        return ensemble_output

    def forward(self, inputs, fold=-1):
        if fold == -1:
            # print('ensemble')
            return self.ensemble(inputs)
        else:
            # print('single model', fold)
            with torch.no_grad():
                outputs = self.models[fold](inputs)
            return outputs


def load_image(image_path, image_size):
    img0 = load_cv2_image(image_path)
    img0 = cv2.resize(img0, (image_size[1], image_size[0]))
    img0 = normalize_img(img0).astype(np.float16)
    img0 = to_torch_tensor(img0)
    return img0


def load_helmet(helmet_gb, helmet_keys, key):
    if key in helmet_keys:
        helmets = helmet_gb.get_group(key)
    else:
        helmets = None
    return helmets


def load_track_dict(track_path):
    track_df = pd.read_csv(track_path, parse_dates=["datetime"])
    track_df['x_position'] /= 120
    track_df['y_position'] /= 53.3
    track_df['sin_direction'] = track_df['direction'].map(lambda x: np.sin(x * np.pi / 180))
    track_df['cos_direction'] = track_df['direction'].map(lambda x: np.cos(x * np.pi / 180))
    track_df['sin_orientation'] = track_df['orientation'].map(lambda x: np.sin(x * np.pi / 180))
    track_df['cos_orientation'] = track_df['orientation'].map(lambda x: np.cos(x * np.pi / 180))
    track_df = track_df[['game_play',  'nfl_player_id', 'step', 'x_position', 'y_position', 'speed',
                         'distance', 'acceleration', 'sa', 'sin_direction', 'cos_direction', 'sin_orientation', 'cos_orientation']]
    track_df['frame'] = track_df['step'].map(step_to_frame)

    gb_cols = ['game_play', 'frame']
    track_cols = ['x_position', 'y_position', 'speed', 'distance',
                  'acceleration', 'sa', 'sin_direction', 'cos_direction',
                  'sin_orientation', 'cos_orientation', 'nfl_player_id']
    track_gb = track_df.groupby(gb_cols)
    track_dict = {k: v[track_cols].values for k, v in track_gb}
    return track_dict


def get_track_feats(track_feats, unique_ids):
    ordered_track_feats = np.zeros((len(unique_ids), 10))
    track_ids = track_feats[:, -1].astype(int).tolist()
    for i, id1 in enumerate(track_ids):
        ordered_track_feats[unique_ids.index(id1)] = track_feats[i, :-1]
    return ordered_track_feats


def both_image_exists(game_play, frame):
    for view in VIEWS:
        img_path = f'{BASE_DIR}/{game_play}_{view}/{frame:06}.jpg'
        if not os.path.exists(img_path):
            return False
    return True


def get_preds_and_masks(outputs, unique_ids, pairs):
    ground_preds = outputs['ground'].sigmoid().cpu().numpy()[0]
    ground_masks = outputs['ground_masks'].cpu().numpy()[0]
    inter_preds = outputs['inter'].sigmoid().cpu().numpy()[0]
    inter_masks = outputs['inter_masks'].cpu().numpy()[0]

    preds = []
    masks = []
    for id1, id2 in pairs:
        idx1 = unique_ids.index(id1)
        if id2 == -1:
            if ground_masks[idx1]:
                preds.append(ground_preds[idx1])
                masks.append(True)
            else:
                preds.append(0.0)
                masks.append(False)
        else:
            idx2 = unique_ids.index(id2)
            if (inter_masks[idx1, idx2]):
                # print(abs(inter_preds[idx1, idx2] - inter_preds[idx2, idx1]))
                preds.append((inter_preds[idx1, idx2] + inter_preds[idx2, idx1]) / 2.0)
                masks.append(True)
            else:
                preds.append(0.0)
                masks.append(False)
    return preds, masks


def inference(sub_df, helmet_df, track_dict, models):

    helmet_gb = helmet_df.groupby(['game_play', 'view', 'frame'])
    helmet_keys = helmet_gb.groups.keys()

    dfs = []
    for (game_play, frame), frame_df in tqdm(sub_df.groupby(['game_play', 'frame'])):
        if not both_image_exists(game_play, frame):
            # endとsideどちらかのフレームがなかったらスキップ
            frame_df['preds'] = 0.0
            frame_df['masks'] = False
            dfs.append(frame_df)
            continue

        inputs = {}
        unique_ids = frame_df['nfl_player_id_1'].values.tolist()

        track_feats = track_dict[(game_play, frame)]
        track_feats = get_track_feats(track_feats, unique_ids)
        inputs['track'] = torch.tensor(track_feats[None].astype(np.float32)).cuda()

        success = True
        for view in VIEWS:
            helmet_df = load_helmet(helmet_gb, helmet_keys, (game_play, view, frame))
            if helmet_df is None:
                success = False
                break

            frame_helmets = helmet_df[['left', 'top', 'width', 'height', 'nfl_player_id']].values
            bboxes, coords = get_bboxes(frame_helmets, unique_ids, cfg.train.image_size,
                                        cfg.train.original_image_size, cfg.train.roi_size, cfg.train.normalize_coords)
            masks = bboxes.sum(axis=1) > 0  # has bbox flag
            inputs[view + '_rois'] = torch.tensor(bboxes[None, None].astype(np.float32)).cuda()  # add batch and seq dim
            inputs[view + '_coords'] = torch.tensor(coords[None, None].astype(np.float32)
                                                    ).cuda()  # add batch and seq dim
            inputs[view + '_mask'] = torch.tensor(masks[None, None].astype(np.float32)).cuda()  # add batch and seq dim

            img_path = f'{BASE_DIR}/{game_play}_{view}/{frame:06}.jpg'
            inputs[view + "_image"] = load_image(img_path, cfg.train.image_size)[None,
                                                                                 None].cuda()  # add batch and seq dim

        if not success:
            # endとsideどちらかのヘルメットがなかったらスキップ
            frame_df['preds'] = 0.0
            frame_df['masks'] = False
            dfs.append(frame_df)
            continue

        with torch.no_grad() and autocast(cfg.mixed_precision):
            outputs = models(inputs, fold=-1)

        pairs = frame_df[['nfl_player_id_1', 'nfl_player_id_2']].values
        output_preds, output_masks = get_preds_and_masks(outputs, unique_ids, pairs)

        frame_df['preds'] = output_preds
        frame_df['masks'] = output_masks

        dfs.append(frame_df)
    return pd.concat(dfs)


def main():
    # weight_paths = sorted(
    #     glob.glob(f'./output/exp048_both_ext_blur_dynamic_normalize_coords_fix_frame_noise/last_fold*'))
    weight_paths = sorted(glob.glob(f'/kaggle/input/nfl-exp048/last_fold*'))
    models = Models(weight_paths)

    test_path = '../input/nfl-player-contact-detection/sample_submission.csv'
    test_helmet_path = '../input/nfl-player-contact-detection/test_baseline_helmets.csv'
    test_tracking_path = "../input/nfl-player-contact-detection/test_player_tracking.csv"

    sub = pd.read_csv(test_path)
    sub = expand_contact_id(sub)
    sub['frame'] = sub['step'].map(step_to_frame)

    test_helmet_df = pd.read_csv(test_helmet_path)
    test_helmet_df['game_play'] = test_helmet_df['game_key'].astype(
        str) + '_' + test_helmet_df['play_id'].astype(str).str.zfill(6)

    test_track_dict = load_track_dict(test_tracking_path)

    sub = inference(sub, test_helmet_df, test_track_dict, models)
    sub.to_csv(f'exp048_preds.csv', index=False)


if __name__ == '__main__':
    main()

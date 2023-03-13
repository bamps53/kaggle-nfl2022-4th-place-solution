import pickle
import numpy as np
import pandas as pd
from datasets.common import setup_df

VIEWS = ['Endzone', 'Sideline']


def get_track_dict(track_df, no_sa=False):
    gb_cols = ['game_play', 'frame']
    if no_sa:
        track_cols = ['x_position', 'y_position', 'speed', 'distance',
                      'acceleration', 'sin_direction', 'cos_direction',
                      'sin_orientation', 'cos_orientation', 'nfl_player_id']
    else:
        track_cols = ['x_position', 'y_position', 'speed', 'distance',
                      'acceleration', 'sa', 'sin_direction', 'cos_direction',
                      'sin_orientation', 'cos_orientation', 'nfl_player_id']

    track_cols
    gb = track_df.groupby(gb_cols)
    track_dict = {k: v[track_cols].values for k, v in gb}
    return track_dict


def get_track_dict_by_step(track_df, no_sa=False):
    gb_cols = ['game_play', 'step']
    if no_sa:
        track_cols = ['x_position', 'y_position', 'speed', 'distance',
                      'acceleration', 'sin_direction', 'cos_direction',
                      'sin_orientation', 'cos_orientation', 'nfl_player_id']
    else:
        track_cols = ['x_position', 'y_position', 'speed', 'distance',
                      'acceleration', 'sa', 'sin_direction', 'cos_direction',
                      'sin_orientation', 'cos_orientation', 'nfl_player_id']

    track_cols
    gb = track_df.groupby(gb_cols)
    track_dict = {k: v[track_cols].values for k, v in gb}
    return track_dict


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
    cols = ['left', 'top', 'width', 'height', 'nfl_player_id']
    gb = helmet_df.groupby(['game_play', 'frame', 'view'])
    helmet_dict = {k: v[cols].values for k, v in gb}
    return helmet_dict


df_path = '../input/folds.csv'
label_df_path = '../input/train_labels_with_folds.csv'
helmet_df_path = '../input/train_helmets_with_folds.csv'
tracking_df_path = '../input/train_tracking_with_folds.csv'

for fold in range(4):
    for mode in ['train', 'valid']:
        print('start', mode, fold)
        df = setup_df(df_path, fold, mode)
        label_df = setup_df(label_df_path, fold, mode)
        helmet_df = setup_df(helmet_df_path, fold, mode)
        tracking_df = setup_df(tracking_df_path, fold, mode)

        unique_ids_dict = get_unique_ids_dict(label_df)
        inter_contact_dict = get_inter_contact_dict(label_df)
        ground_contact_dict = get_ground_contact_dict(label_df)
        helmet_dict = get_helmet_dict(helmet_df)
        track_dict = get_track_dict(tracking_df)
        track_dict_by_step = get_track_dict_by_step(tracking_df)
        track_dict_no_sa = get_track_dict(tracking_df, no_sa=True)
        track_dict_no_sa_by_step = get_track_dict_by_step(tracking_df, no_sa=True)

        with open(f'../input/unique_ids_dict_{mode}_fold{fold}.pkl', 'wb') as f:
            pickle.dump(unique_ids_dict, f)
        with open(f'../input/inter_contact_dict_{mode}_fold{fold}.pkl', 'wb') as f:
            pickle.dump(inter_contact_dict, f)
        with open(f'../input/ground_contact_dict_{mode}_fold{fold}.pkl', 'wb') as f:
            pickle.dump(ground_contact_dict, f)
        with open(f'../input/helmet_dict_{mode}_fold{fold}.pkl', 'wb') as f:
            pickle.dump(helmet_dict, f)
        with open(f'../input/track_dict_{mode}_fold{fold}.pkl', 'wb') as f:
            pickle.dump(track_dict, f)
        with open(f'../input/track_dict_by_step_{mode}_fold{fold}.pkl', 'wb') as f:
            pickle.dump(track_dict_by_step, f)
        with open(f'../input/track_dict_no_sa_{mode}_fold{fold}.pkl', 'wb') as f:
            pickle.dump(track_dict_no_sa, f)
        with open(f'../input/track_dict_no_sa_by_step_{mode}_fold{fold}.pkl', 'wb') as f:
            pickle.dump(track_dict_no_sa_by_step, f)

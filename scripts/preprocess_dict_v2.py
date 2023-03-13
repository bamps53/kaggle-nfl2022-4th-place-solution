import pickle
from datasets.common import setup_df
import argparse
import pandas as pd
import numpy as np


GROUND_ID = -1


def angle_diff(s1, s2):
    diff = s1 - s2
    return np.abs((diff + 180) % 360 - 180) / 180


def calc_distance(x1, y1, x2, y2):
    return np.sqrt(np.square(x2 - x1) + np.square(y2 - y1))


def add_pair_features(df):
    df["distance"] = calc_distance(
        df["x_position_1"],
        df["y_position_1"],
        df["x_position_2"],
        df["y_position_2"])
    df["different_team"] = (df["team_1"] != df["team_2"]).astype(float) - 0.5
    df["anglediff_dir1_dir2"] = angle_diff(df["direction_1"], df["direction_2"])
    df["anglediff_ori1_ori2"] = angle_diff(df["orientation_1"], df["orientation_2"])
    df["anglediff_dir1_ori1"] = angle_diff(df["direction_1"], df["orientation_1"])
    df["anglediff_dir2_ori2"] = angle_diff(df["direction_2"], df["orientation_2"])

    df['speed_diff'] = (df['speed_1'] - df['speed_2']).abs()
    df['acceleration_diff'] = (df['acceleration_1'] - df['acceleration_2']).abs()
    df['sa_diff'] = (df['sa_1'] - df['sa_2']).abs()
    df['x_position_diff'] = (df['x_position_1'] - df['x_position_2']).abs()
    df['y_position_diff'] = (df['y_position_1'] - df['y_position_2']).abs()
    return df


track_cols = ['team', 'x_position', 'y_position',
              'speed', 'direction', 'orientation', 'acceleration', 'sa',
              'sin_direction', 'cos_direction', 'sin_orientation', 'cos_orientation']

pair_cols = ['distance', 'different_team', 'anglediff_dir1_dir2',
             'anglediff_ori1_ori2', 'anglediff_dir1_ori1', 'anglediff_dir2_ori2',
             'speed_diff', 'acceleration_diff', 'sa_diff', 'x_position_diff',
             'y_position_diff']

single_cols = ['game_play', 'nfl_player_id', 'step', 'x_position', 'y_position',
               'speed', 'distance', 'acceleration', 'sa', 'sin_direction', 'cos_direction',
               'sin_orientation', 'cos_orientation']


VIEWS = ['Endzone', 'Sideline']
GROUND_ID = -1


def get_unique_ids_dict(label_df):
    unique_ids_dict = label_df.groupby(['game_play', 'frame'])['nfl_player_id_1'].unique().to_dict()

    for k, v in unique_ids_dict.items():
        v = sorted(v)
        if len(v) < 22:
            v = v + [-1] * (22 - len(v))  # padding
        unique_ids_dict[k] = v
    return unique_ids_dict


def get_inter_contact_dict(label_df):
    gb = label_df.query('contact == 1 & nfl_player_id_2 != @GROUND_ID').groupby(
        ['game_play', 'frame'])[['nfl_player_id_1', 'nfl_player_id_2', ]]
    contact_dict = {k: v[['nfl_player_id_1', 'nfl_player_id_2']].values for k, v in gb}
    return contact_dict


def get_distance_dict(label_df):
    gb = label_df.query('nfl_player_id_2 != @GROUND_ID').groupby(['game_play', 'frame'])
    distance_dict = {k: v[['nfl_player_id_1', 'nfl_player_id_2', 'distance']].values for k, v in gb}
    return distance_dict


def get_ground_contact_dict(label_df):
    gb = label_df.query(
        'contact == 1 & nfl_player_id_2 == @GROUND_ID').groupby(['game_play', 'frame'])[['nfl_player_id_1', 'nfl_player_id_2']]
    contact_dict = {k: v['nfl_player_id_1'].values for k, v in gb}
    return contact_dict


def get_helmet_dict(helmet_df):
    cols = ['left', 'top', 'width', 'height', 'nfl_player_id']
    gb = helmet_df.groupby(['game_play', 'frame', 'view'])
    helmet_dict = {k: v[cols].values for k, v in gb}
    return helmet_dict


def get_track_dict(track_df):
    gb_cols = ['game_play', 'frame']
    track_cols = ['x_position', 'y_position', 'speed', 'distance',
                  'acceleration', 'sa', 'sin_direction', 'cos_direction',
                  'sin_orientation', 'cos_orientation', 'nfl_player_id']
    gb = track_df.groupby(gb_cols)
    track_dict = {k: v[track_cols].values for k, v in gb}
    return track_dict


def get_pair_dict(pair_df):
    gb_cols = ['game_play', 'frame']
    pair_cols = ['nfl_player_id_1', 'nfl_player_id_2', 'distance', 'different_team', 'anglediff_dir1_dir2',
                 'anglediff_ori1_ori2', 'anglediff_dir1_ori1', 'anglediff_dir2_ori2',
                 'speed_diff', 'acceleration_diff', 'sa_diff', 'x_position_diff',
                 'y_position_diff']
    gb = pair_df.query('nfl_player_id_2 != @GROUND_ID').groupby(gb_cols)
    distance_dict = {k: v[pair_cols].values for k, v in gb}
    return distance_dict


def save_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def merge_tracking(df, tr, use_cols):
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

    return dst


def save_train_data_dict(save_dir, mode, fold=-1):
    label_df_path = f'{save_dir}/label_df.csv'
    helmet_df_path = f'{save_dir}/helmet_df.csv'
    track_df_path = f'{save_dir}/track_df.csv'

    label_df = setup_df(label_df_path, fold, mode)

    unique_ids_dict = get_unique_ids_dict(label_df)
    distance_dict = get_distance_dict(label_df)
    inter_contact_dict = get_inter_contact_dict(label_df)
    ground_contact_dict = get_ground_contact_dict(label_df)
    save_pickle(unique_ids_dict, f'{save_dir}/unique_ids_dict_{mode}_fold{fold}.pkl')
    save_pickle(distance_dict, f'{save_dir}/distance_dict_{mode}_fold{fold}.pkl')
    save_pickle(inter_contact_dict, f'{save_dir}/inter_contact_dict_{mode}_fold{fold}.pkl')
    save_pickle(ground_contact_dict, f'{save_dir}/ground_contact_dict_{mode}_fold{fold}.pkl')

    helmet_df = setup_df(helmet_df_path, fold, mode)
    helmet_dict = get_helmet_dict(helmet_df)
    save_pickle(helmet_dict, f'{save_dir}/helmet_dict_{mode}_fold{fold}.pkl')

    track_df = setup_df(track_df_path, fold, mode)
    track_dict = get_track_dict(track_df)
    save_pickle(track_dict, f'{save_dir}/track_dict_{mode}_fold{fold}.pkl')

    pair_df = merge_tracking(label_df, track_df, track_cols)
    pair_df = add_pair_features(pair_df)
    pair_df = pair_df.query('nfl_player_id_2 != @GROUND_ID')
    pair_dict = get_pair_dict(pair_df)
    save_pickle(pair_dict, f'{save_dir}/pair_dict_{mode}_fold{fold}.pkl')


def save_test_data_dict(save_dir):
    label_df_path = f'{save_dir}/label_df.csv'
    frame_label_df_path = f'{save_dir}/image_label_df.csv'
    helmet_df_path = f'{save_dir}/helmet_df.csv'
    track_df_path = f'{save_dir}/track_df.csv'
    frame_track_df_path = f'{save_dir}/frame_track_df.csv'

    label_df = pd.read_csv(label_df_path)

    unique_ids_dict = get_unique_ids_dict(label_df)
    distance_dict = get_distance_dict(label_df)
    save_pickle(unique_ids_dict, f'{save_dir}/unique_ids_dict.pkl')
    save_pickle(distance_dict, f'{save_dir}/distance_dict.pkl')

    helmet_df = pd.read_csv(helmet_df_path)
    helmet_dict = get_helmet_dict(helmet_df)
    save_pickle(helmet_dict, f'{save_dir}/helmet_dict.pkl')

    track_df = pd.read_csv(track_df_path)
    track_dict = get_track_dict(track_df)
    save_pickle(track_dict, f'{save_dir}/track_dict.pkl')

    pair_df = merge_tracking(label_df, track_df, track_cols)
    pair_df = add_pair_features(pair_df)
    pair_df = pair_df.query('nfl_player_id_2 != @GROUND_ID')
    pair_dict = get_pair_dict(pair_df)
    save_pickle(pair_dict, f'{save_dir}/pair_dict.pkl')


def main(is_train=False):
    if is_train:
        save_dir = '../input/preprocessed_data'
        for fold in range(4):
            for mode in ['train', 'valid']:
                print('start', mode, fold)
                save_train_data_dict(save_dir, mode, fold)
    else:
        save_dir = './test_preprocessed_data'
        save_test_data_dict(save_dir)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--is_train", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.is_train)

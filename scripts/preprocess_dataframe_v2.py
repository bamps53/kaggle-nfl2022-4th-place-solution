import argparse
import pandas as pd
import os
import numpy as np

from utils.common import reduce_mem_usage


GROUND_ID = -1


def add_time_to_helmet(helmet_df, meta_df):
    start_times = meta_df[["game_play", "start_time"]].drop_duplicates()
    start_times["start_time"] = pd.to_datetime(start_times["start_time"])

    helmet_df = pd.merge(helmet_df,
                         start_times,
                         on="game_play",
                         how="left")

    fps = 59.94
    helmet_df["datetime"] = helmet_df["start_time"] + \
        pd.to_timedelta(helmet_df["frame"] * (1 / fps), unit="s")
    helmet_df["datetime"] = pd.to_datetime(helmet_df["datetime"], utc=True)
    helmet_df["datetime_ngs"] = pd.DatetimeIndex(
        helmet_df["datetime"] + pd.to_timedelta(50, "ms")).floor("100ms").values
    helmet_df["datetime_ngs"] = pd.to_datetime(helmet_df["datetime_ngs"], utc=True)
    helmet_df = helmet_df.drop(['game_key', 'play_id', 'video', 'player_label',  'start_time', 'datetime'], axis=1)
    return helmet_df


def add_time_to_labels(label_df):
    label_df["datetime_ngs"] = pd.to_datetime(label_df["datetime"], utc=True)
    label_df = label_df.drop(['contact_id', 'datetime'], axis=1)
    return label_df


def check_frame_exists(base_image_dir, game_play, view, frame, k=5):
    image_dir = f'{base_image_dir}/{game_play}_{view}/'
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
            # print(f'{img_path} not exists.')
            return False
    return True


def select_only_has_image_labels(label_df, step_frame_info, image_dir):
    image_exists = []
    for game_play, datetime_ngs, step, frame in step_frame_info.values:
        for view in ['Endzone', 'Sideline']:
            has_image = check_frame_exists(image_dir, game_play, view, frame, k=1)
            image_exists.append((game_play, datetime_ngs, step, frame, view, has_image))

    image_df = pd.DataFrame(image_exists, columns=['game_play', 'datetime_ngs', 'step', 'frame', 'view', 'has_image'])
    end_df = image_df.query('view == "Endzone"').rename(columns={'has_image': 'has_end_image'}).drop('view', axis=1)
    side_df = image_df.query('view == "Sideline"').rename(columns={'has_image': 'has_side_image'}).drop('view', axis=1)
    image_df = end_df.merge(side_df)

    label_df = label_df.merge(image_df)
    label_df = label_df.query('has_end_image == True & has_side_image == True')
    label_df = label_df.reset_index(drop=True).drop(['datetime_ngs', 'has_end_image', 'has_side_image'], axis=1)
    return label_df


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


def calc_distance(x1, y1, x2, y2):
    return np.sqrt(np.square(x2 - x1) + np.square(y2 - y1))


def add_distance(label_df, track_df):
    label_df = merge_tracking(label_df, track_df, use_cols=['x_position', 'y_position'])
    label_df['distance'] = calc_distance(
        label_df['x_position_1'],
        label_df['y_position_1'],
        label_df['x_position_2'],
        label_df['y_position_2'],
    )
    label_df = label_df.drop(['x_position_1', 'y_position_1', 'x_position_2', 'y_position_2'], axis=1)
    return label_df


def preprocess_track_df(track_df):
    track_df['x_position'] /= 100
    track_df['y_position'] /= 100
    track_df['speed'] /= 10
    track_df['acceleration'] /= 10
    track_df['sa'] /= 10

    track_df['sin_direction'] = track_df['direction'].map(lambda x: np.sin(x * np.pi / 180))
    track_df['cos_direction'] = track_df['direction'].map(lambda x: np.cos(x * np.pi / 180))
    track_df['sin_orientation'] = track_df['orientation'].map(lambda x: np.sin(x * np.pi / 180))
    track_df['cos_orientation'] = track_df['orientation'].map(lambda x: np.cos(x * np.pi / 180))
    return track_df


def add_fold(df, fold_df):
    df['game'] = df['game_play'].map(lambda x: int(x.split('_')[0]))
    df = df.merge(fold_df)
    df = df.drop('game', axis=1)
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
    return df


def add_datetime(df, track_df):
    df = pd.merge(df,
                  track_df[["step", "game_play", "datetime"]].drop_duplicates(),
                  on=["game_play", "step"], how="left")
    return df


def main(is_train=False):
    input_dir = '../input/nfl-player-contact-detection'
    if is_train:
        phase = 'train'
        image_dir = '../input/train_frames/'
        label_path = f'{input_dir}/train_labels.csv'
        helmet_path = f'{input_dir}/train_baseline_helmets.csv'
        meta_path = f'{input_dir}/train_video_metadata.csv'
        track_path = f'{input_dir}/train_player_tracking.csv'
        save_dir = '../input/preprocessed_data'
        fold_df = pd.read_csv("../input/nfl-game-fold/game_fold.csv")
    else:
        phase = 'test'
        image_dir = '/temp/test_frames'
        label_path = f'{input_dir}/sample_submission.csv'
        helmet_path = f'{input_dir}/test_baseline_helmets.csv'
        meta_path = f'{input_dir}/test_video_metadata.csv'
        track_path = f'{input_dir}/test_player_tracking.csv'
        save_dir = './test_preprocessed_data'
    os.makedirs(save_dir, exist_ok=True)

    helmet_df = pd.read_csv(helmet_path)
    helmet_df = helmet_df.query('view != "Endzone2"').reset_index(drop=True)
    meta_df = pd.read_csv(meta_path)
    track_df = pd.read_csv(track_path)

    label_df = pd.read_csv(label_path)
    if not is_train:
        label_df = expand_contact_id(label_df)
        label_df = add_datetime(label_df, track_df)
        label_df['nfl_player_id_1'] = label_df['nfl_player_id_1'].astype(int)
    label_df['nfl_player_id_2'] = label_df['nfl_player_id_2'].replace('G', f'{GROUND_ID}').astype(int)

    if is_train:
        # add folds
        label_df = add_fold(label_df, fold_df)
        helmet_df = add_fold(helmet_df, fold_df)
        track_df = add_fold(track_df, fold_df)

    label_df = add_time_to_labels(label_df)
    helmet_df = add_time_to_helmet(helmet_df, meta_df)

    frame_info = helmet_df[['game_play', 'frame', 'datetime_ngs']].drop_duplicates()
    step_info = label_df[['game_play', 'datetime_ngs', 'step']].drop_duplicates()
    step_frame_info = step_info.merge(frame_info)

    label_df = label_df.drop('datetime_ngs', axis=1)
    image_label_df = select_only_has_image_labels(label_df, step_frame_info, image_dir)
    helmet_df = helmet_df.drop('datetime_ngs', axis=1)

    label_df = add_distance(label_df, track_df)  # original scale
    image_label_df = add_distance(image_label_df, track_df)

    label_df = label_df.merge(image_label_df.groupby(['game_play', 'step'])[
                              'frame'].mean().astype(int).rename('frame').reset_index())

    track_df = preprocess_track_df(track_df)
    frame_track_df = track_df.merge(
        image_label_df[['game_play', 'step', 'frame']].drop_duplicates(), how='inner')  # add frame
    track_df = track_df.merge(label_df[['game_play', 'step', 'frame']].drop_duplicates(), how='inner')  # add frame

    label_df = reduce_mem_usage(label_df)
    image_label_df = reduce_mem_usage(image_label_df)
    helmet_df = reduce_mem_usage(helmet_df)
    track_df = reduce_mem_usage(track_df)
    frame_track_df = reduce_mem_usage(frame_track_df)

    label_df.to_csv(f'{save_dir}/label_df.csv', index=False)
    image_label_df.to_csv(f'{save_dir}/image_label_df.csv', index=False)
    helmet_df.to_csv(f'{save_dir}/helmet_df.csv', index=False)
    track_df.to_csv(f'{save_dir}/track_df.csv', index=False)
    frame_track_df.to_csv(f'{save_dir}/frame_track_df.csv', index=False)

    if is_train:
        sample_df = label_df[['game_play', 'step', 'fold', 'frame']].drop_duplicates().reset_index(drop=True)
        frame_sample_df = image_label_df[['game_play', 'step', 'fold', 'frame']
                                         ].drop_duplicates().reset_index(drop=True)
    else:
        sample_df = label_df[['game_play', 'step', 'frame']].drop_duplicates().reset_index(drop=True)
        frame_sample_df = image_label_df[['game_play', 'step', 'frame']].drop_duplicates().reset_index(drop=True)

    sample_df.to_csv(f'{save_dir}/sample_df.csv')
    frame_sample_df.to_csv(f'{save_dir}/frame_sample_df.csv')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--is_train", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.is_train)

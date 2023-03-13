import numpy as np
import pandas as pd


def step_to_frame(df):
    step0_frame = 300
    fps_frame = 59.94
    fps_step = 10
    step_max = 200
    convert_dict = {}
    for step in range(step_max):
        convert_dict[step] = step * fps_frame / fps_step + step0_frame
    df['frame'] = df['step'].map(convert_dict)
    df['frame'] = df['frame'].round().astype(int)
    return df


unique_data_cols = ['game_play', 'step', 'frame', 'fold']
label_cols = ['game_play', 'step', 'nfl_player_id_1',
              'nfl_player_id_2', 'contact', 'frame', 'fold']
tr_helmets_cols = ['game_play', 'view', 'frame',
                   'nfl_player_id', 'player_label', 'left', 'width', 'top', 'height',
                   'fold']


def preprocess_tracking_df(track_df):
    track_df['x_position'] /= 120
    track_df['y_position'] /= 53.3
    track_df['sin_direction'] = track_df['direction'].map(lambda x: np.sin(x * np.pi / 180))
    track_df['cos_direction'] = track_df['direction'].map(lambda x: np.cos(x * np.pi / 180))
    track_df['sin_orientation'] = track_df['orientation'].map(lambda x: np.sin(x * np.pi / 180))
    track_df['cos_orientation'] = track_df['orientation'].map(lambda x: np.cos(x * np.pi / 180))
    track_df = track_df[['game_play', 'nfl_player_id', 'step', 'x_position', 'y_position', 'speed',
                         'distance', 'acceleration', 'sa', 'sin_direction', 'cos_direction', 'sin_orientation', 'cos_orientation']]
    return track_df


if __name__ == "__main__":
    print('load dataframes...')
    train_labels = pd.read_csv('../input/nfl-player-contact-detection/train_labels.csv')
    train_helmets = pd.read_csv('../input/nfl-player-contact-detection/train_baseline_helmets.csv')
    train_meta = pd.read_csv('../input/nfl-player-contact-detection/train_video_metadata.csv')
    train_tracking = pd.read_csv(
        "../input/nfl-player-contact-detection/train_player_tracking.csv", parse_dates=["datetime"])
    folds = pd.read_csv("../input/nfl-game-fold/game_fold.csv")

    print('preprocess dataframes...')
    train_labels['nfl_player_id_2'] = train_labels['nfl_player_id_2'].replace('G', 0).astype(int)
    train_labels = step_to_frame(train_labels)
    train_labels['game'] = train_labels['game_play'].map(lambda x: int(x.split('_')[0]))
    unique_data = train_labels[['game', 'game_play', 'step', 'frame']].drop_duplicates().reset_index(drop=True)
    train_tracking = preprocess_tracking_df(train_tracking)

    print('merge dataframes...')
    unique_data = unique_data.merge(folds)
    train_labels = train_labels.merge(folds)
    train_helmets = train_helmets.merge(folds.rename(columns={'game': 'game_key'}))
    train_tracking = train_labels.query('nfl_player_id_2 == 0').rename(
        columns={'nfl_player_id_1': 'nfl_player_id'}).merge(train_tracking, how='left')

    print('save dataframes...')
    unique_data[unique_data_cols].to_csv('../input/folds.csv', index=False)
    train_labels[label_cols].to_csv('../input/train_labels_with_folds.csv', index=False)
    train_tracking.to_csv('../input/train_tracking_with_folds.csv', index=False)
    train_helmets[tr_helmets_cols].to_csv('../input/train_helmets_with_folds.csv', index=False)

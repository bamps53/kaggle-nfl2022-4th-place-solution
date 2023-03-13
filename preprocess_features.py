import argparse
import gc
import glob
import os
import pickle

import numpy as np
import pandas as pd
from feature_engineering.kalmen_filter import expand_helmet_smooth
from feature_engineering.merge_for_seq_model import make_features_for_seq_model
from feature_engineering.point_set_matching import match_p2p_with_cache
from sklearn.preprocessing import StandardScaler
from utils.general import timer
from utils.nfl import (TRACK_COLS, TRAIN_COLS, Config,
                       ModelSize, expand_contact_id,
                       expand_helmet, read_csv_with_cache)

pd.options.mode.chained_assignment = None  # default='warn'


def add_fold(df, fold_df):
    df['game'] = df['game_play'].map(lambda x: int(x.split('_')[0]))
    df = df.merge(fold_df)
    df = df.drop('game', axis=1)
    return df


def select_features(df):
    count = 0
    id_cols = [
        'game_play',
        'step',
        'nfl_player_id_1',
        'nfl_player_id_2',
        'contact',
        'datetime_ngs',
        'fold'
    ]
    unused_cols = [
        'team_1',
        'position_1',
        'team_2',
        'position_2',
    ]

    exclude_keys = ['cnn', 'camaro', 'shift', 'diff', 'roll', 'window', 'past', 'future', 'step0', 'step', 'start', 'move',
                    'bbox_center_Sideline_distance_',
                    'bbox_center_Endzone_distance_',
                    'bbox_center_y_Sideline_1_',
                    'bbox_center_y_Endzone_1_',
                    ]
    # exclude_keys = ['cnn', 'camaro']
    use_cols = []
    for c in df.columns.tolist():
        if c in unused_cols or c in id_cols:
            continue
        exclude = False
        for k in exclude_keys:
            if k in c:
                exclude = True
                break
        if not exclude:
            count += 1
            # print(c)
            use_cols.append(c)
    print(count)

    p1_cols = [c for c in use_cols if c.endswith('1') or '_1_' in c]
    p2_cols = [c for c in use_cols if c.endswith('2') or '_2_' in c]
    pair_cols = [c for c in use_cols if c not in p1_cols and c not in p2_cols]

    p1_additional_cols = [
        'x_position_offset_on_img_Side',
        'y_position_offset_on_img_Side',
        'x_rel_position_offset_on_img_Side',
        'y_rel_position_offset_on_img_Side',
        'p2p_registration_residual_Side',
        'p2p_registration_residual_frame_Side',
        'x_position_offset_on_img_End',
        'y_position_offset_on_img_End',
        'x_rel_position_offset_on_img_End',
        'y_rel_position_offset_on_img_End',
        'p2p_registration_residual_End',
        'p2p_registration_residual_frame_End',
    ]

    global_cols = [
        'width_Sideline_mean',
        'height_Sideline_mean',
        'Sideline_count',
        'aspect_Sideline_mean',
        'width_Endzone_mean',
        'height_Endzone_mean',
        'Endzone_count',
        'aspect_Endzone_mean',
        'x_position_mean',
        'y_position_mean',
        'speed_mean',
        'acceleration_mean',
        'sa_mean',
        'distance_team2team',
        'distance_mean_in_play',
    ]

    pair_cols = [
        'bbox_iou_Sideline',
        'bbox_iou_Endzone',
        'dx',
        'distance',
        'mean_distance_around_player_full_pair',
        'std_distance_around_player_full_pair',
        'idxmin_distance_aronud_player_full_pair',
        'bbox_center_Sideline_distance',
        'bbox_center_Endzone_distance',
        'mean_distance_around_player_pair',
        'min_distance_around_player_pair',
        'std_distance_around_player_pair',
        'idxmin_distance_aronud_player_pair',
        'distance_ratio_distance_to_min_pair_distance',
        'bbox_x_std_overlap_Sideline',
        'bbox_y_std_overlap_Sideline',
        'bbox_x_std_overlap_Endzone',
        'bbox_y_std_overlap_Endzone',
        'angle_dxdy'
    ]

    print('p1_cols:', len(p1_cols + p1_additional_cols))
    print('p2_cols:', len(p2_cols))
    print('pair_cols:', len(pair_cols))
    print('global_cols:', len(global_cols))

    use_cols = p1_cols + p1_additional_cols + p2_cols + pair_cols + global_cols
    len(use_cols)

    col_info = {
        "id": id_cols,
        "p1": p1_cols,
        "p1_additional": p1_additional_cols,
        "p2": p2_cols,
        "pair": pair_cols,
        "global": global_cols,
    }

    keep_cols = id_cols + use_cols
    df = df[keep_cols]
    return df, col_info


def preprocess_df(df, use_cols, scaler=None):
    if scaler is None:
        scaler = StandardScaler()
        X_num = scaler.fit_transform(df[use_cols])
        X_num = np.nan_to_num(X_num, posinf=0, neginf=0)
        df[use_cols] = X_num
        return df, scaler
    else:
        X_num = scaler.transform(df[use_cols])  # TODO: infでも大丈夫？
        df[use_cols] = np.nan_to_num(X_num, posinf=0, neginf=0)
        return df


def train(cfg: Config, inference_only=False):
    save_dir = f'output/{cfg.EXP_NAME}'
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(cfg.CACHE, exist_ok=True)

    if inference_only:
        with open(f'../input/features-for-seq-model/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open(f'../input/features-for-seq-model/col_info.pkl', 'rb') as f:
            col_info = pickle.load(f)
        use_cols = col_info['p1'] + col_info['p1_additional'] + col_info['p2'] + col_info['pair'] + col_info['global']
        return scaler, use_cols

    with timer("load file"):
        tr_tracking = read_csv_with_cache("train_player_tracking.csv", cfg.INPUT, cfg.CACHE, usecols=TRACK_COLS)
        train_df = read_csv_with_cache("train_labels.csv", cfg.INPUT, cfg.CACHE, usecols=TRAIN_COLS)
        tr_helmets = read_csv_with_cache("train_baseline_helmets.csv", cfg.HELMET_DIR, cfg.CACHE)
        tr_meta = pd.read_csv(os.path.join(cfg.INPUT, "train_video_metadata.csv"),
                              parse_dates=["start_time", "end_time", "snap_time"])
        tr_regist = match_p2p_with_cache(os.path.join(cfg.CACHE, "train_registration.f"),
                                         tracking=tr_tracking, helmets=tr_helmets, meta=tr_meta)
        fold_df = pd.read_csv('../input/nfl-game-fold/game_fold.csv')

    with timer("assign helmet metadata"):
        train_df = expand_helmet(cfg, train_df, "train")
        train_df = expand_helmet_smooth(cfg, train_df, "train")
        del tr_helmets, tr_meta
        gc.collect()

    train_feature_df = make_features_for_seq_model(train_df, tr_tracking, tr_regist)
    del tr_tracking, tr_regist
    gc.collect()

    train_feature_df = add_fold(train_feature_df, fold_df)
    train_feature_df, col_info = select_features(train_feature_df)
    use_cols = col_info['p1'] + col_info['p1_additional'] + col_info['p2'] + col_info['pair'] + col_info['global']
    train_feature_df, scaler = preprocess_df(train_feature_df, use_cols)
    train_feature_df.to_feather(f'{save_dir}/preprocessed_feature_df_for_seq_model.f')

    with open(f'{save_dir}/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open(f'{save_dir}/col_info.pkl', 'wb') as f:
        pickle.dump(col_info, f)
    return scaler, use_cols


def inference(cfg: Config, scaler, use_cols):
    with timer("load file"):
        test_tracking = read_csv_with_cache(
            "test_player_tracking.csv", cfg.INPUT, cfg.CACHE, usecols=TRACK_COLS)

        sub = read_csv_with_cache("sample_submission.csv", cfg.INPUT, cfg.CACHE)
        test_df = expand_contact_id(sub)
        test_df = pd.merge(test_df,
                           test_tracking[["step", "game_play", "datetime"]].drop_duplicates(),
                           on=["game_play", "step"], how="left")
        test_df = expand_helmet(cfg, test_df, "test")
        test_df = expand_helmet_smooth(cfg, test_df, "test")

        te_helmets = read_csv_with_cache("test_baseline_helmets.csv", cfg.HELMET_DIR, cfg.CACHE)
        te_meta = pd.read_csv(os.path.join(cfg.INPUT, "test_video_metadata.csv"),
                              parse_dates=["start_time", "end_time", "snap_time"])
        test_regist = match_p2p_with_cache(os.path.join(cfg.CACHE, "test_registration.f"),
                                           tracking=test_tracking, helmets=te_helmets, meta=te_meta)

    def _make_features_per_game(game_play, game_test_df, game_test_tracking, game_test_regist, cnn_df_dict):
        with timer("make features(test)"):
            game_test_feature_df = make_features_for_seq_model(game_test_df, game_test_tracking, game_test_regist)
            game_test_feature_df = preprocess_df(game_test_feature_df, use_cols, scaler)
            game_test_feature_df.to_feather(f'{game_play}_preprocessed_feature_df_for_seq_model.f')

    game_plays = test_df['game_play'].unique()
    game_test_gb = test_df.groupby(['game_play'])
    game_test_tracking_gb = test_tracking.groupby(['game_play'])
    game_test_regist_gb = test_regist.groupby(['game_play'])
    for game_play in game_plays:
        game_test_df = game_test_gb.get_group(game_play)
        game_test_tracking = game_test_tracking_gb.get_group(game_play)
        game_test_regist = game_test_regist_gb.get_group(game_play)
        _make_features_per_game(game_play, game_test_df, game_test_tracking, game_test_regist, cnn_df_dict={})
    df = pd.concat([pd.read_feather(p)
                   for p in glob.glob('*_preprocessed_feature_df_for_seq_model.f')]).reset_index(drop=True)
    df.to_feather('test_preprocessed_feature_df_for_seq_model.f')


def main(args):
    cfg = Config(
        EXP_NAME='features_for_seq_model',
        PRETRAINED_MODEL_PATH=args.lgbm_path,
        CAMARO_DF1_PATH=args.camaro1_path,
        CAMARO_DF1_ANY_PATH=args.camaro1_any_path,
        CAMARO_DF2_PATH=args.camaro2_path,
        CAMARO_DF2_ANY_PATH=args.camaro2_any_path,
        KMAT_END_DF_PATH=args.kmat_end_path,
        KMAT_SIDE_DF_PATH=args.kmat_side_path,
        KMAT_END_MAP_DF_PATH=args.kmat_end_map_path,
        KMAT_SIDE_MAP_DF_PATH=args.kmat_side_map_path,
        MODEL_SIZE=ModelSize.HUGE,
        ENABLE_MULTIPROCESS=args.enable_multiprocess,
        DEBUG=args.debug,
        RAW_OUTPUT=args.raw_output,
    )
    scaler, use_cols = train(cfg, args.inference_only)
    inference(cfg, scaler, use_cols)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", "-db", action="store_true")
    parser.add_argument("--inference_only", "-i", action="store_true")
    parser.add_argument("--validate_only", "-v", action="store_true")
    parser.add_argument("--lgbm_path", "-l", default="", type=str)
    parser.add_argument("--camaro1_path", "-c1", default="", type=str)
    parser.add_argument("--camaro1_any_path", "-c1a", default="", type=str)
    parser.add_argument("--camaro2_path", "-c2", default="", type=str)
    parser.add_argument("--camaro2_any_path", "-c2a", default="", type=str)
    parser.add_argument("--kmat_end_path", "-e", default="", type=str)
    parser.add_argument("--kmat_side_path", "-s", default="", type=str)
    parser.add_argument("--kmat_end_map_path", "-em", default="", type=str)
    parser.add_argument("--kmat_side_map_path", "-sm", default="", type=str)
    parser.add_argument("--enable_multiprocess", "-m", action='store_true')
    parser.add_argument("--raw_output", "-r", action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)

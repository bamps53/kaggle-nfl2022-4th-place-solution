import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset

from .common import setup_df

VIEWS = ['Endzone', 'Sideline']
GROUND_ID = -1


class TrainNewSeqTrackDataset(Dataset):
    def __init__(self, cfg, fold, mode="train"):
        self.fold = fold
        self.id_cols = [
            'game_play',
            'step',
            'datetime_ngs',
        ]
        self.label_col = 'contact'
        self.pair_cols = cfg.pair_cols
        self.player1_cols = cfg.player1_cols
        self.player2_cols = cfg.player2_cols
        self.global_cols = cfg.global_cols

        self.cfg = cfg
        self.mode = mode
        self.df = setup_df(cfg.label_df_path, fold, mode)
        self.sample_df = setup_df(cfg.df_path, fold, mode)[['game_play', 'step']].drop_duplicates()
        self.df = pd.merge(self.df, self.sample_df)
        if cfg.ground_only:
            print('only ground data.')
            self.df = self.df.query('nfl_player_id_2 == @GROUND_ID')
        else:
            print('only inter data.')
            self.df = self.df.query('nfl_player_id_2 != @GROUND_ID')
        self.samples = self.df[['game_play', 'nfl_player_id_1', 'nfl_player_id_2']
                               ].drop_duplicates().reset_index(drop=True)
        self.gb = self.df.groupby(['game_play', 'nfl_player_id_1', 'nfl_player_id_2'])
        self.max_len = self.cfg.max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        key = tuple(self.samples.iloc[idx].values)
        pair_df = self.gb.get_group(key)
        is_ground = key[-1] <= 0  # 0 or -1
        seq_len = len(pair_df)

        inputs = {}
        game_play, p1, p2 = key
        inputs['game_play'] = game_play
        inputs['nfl_player_id_1'] = p1
        inputs['nfl_player_id_2'] = p2

        inputs['is_ground'] = is_ground
        inputs['p1_feats'] = np.zeros((len(self.player1_cols), self.max_len), dtype=np.float32)
        inputs['p1_feats'][:, :seq_len] = pair_df[self.player1_cols].astype(np.float32).values.T
        if self.cfg.ground_only:
            inputs['ground'] = np.zeros((self.max_len,), dtype=np.float32)
            inputs['ground'][:seq_len] = pair_df['contact'].astype(np.float32).values
        else:
            inputs['p2_feats'] = np.zeros((len(self.player2_cols), self.max_len), dtype=np.float32)
            inputs['p2_feats'][:, :seq_len] = pair_df[self.player2_cols].astype(np.float32).values.T
            inputs['pair_feats'] = np.zeros((len(self.pair_cols), self.max_len), dtype=np.float32)
            inputs['pair_feats'][:, :seq_len] = pair_df[self.pair_cols].astype(np.float32).values.T
            inputs['inter'] = np.zeros((self.max_len,), dtype=np.float32)
            inputs['inter'][:seq_len] = pair_df['contact'].astype(np.float32).values

        inputs['global_feats'] = np.zeros((len(self.global_cols), self.max_len), dtype=np.float32)
        inputs['global_feats'][:, :seq_len] = pair_df[self.global_cols].astype(np.float32).values.T
        inputs['masks'] = np.zeros((self.max_len,), dtype=np.bool)
        inputs['masks'][:seq_len] = True

        if self.cfg.image_feature_dir is not None:
            inputs['image_features'] = np.zeros((576, self.max_len), dtype=np.float32)
            game_play, p1, p2 = key
            feature_path = f"{self.cfg.image_feature_dir}_fold{self.fold}/{game_play}_p1_{p1}_p2_{p2}.npy"
            image_features = np.load(feature_path)
            inputs['image_features'][:, :seq_len] = image_features.T
            if np.isnan(inputs['image_features'][:, :seq_len]).any():
                assert False, print('features contain nan')

        if self.cfg.image_feature_dir2 is not None:
            inputs['image_features2'] = np.zeros((576, self.max_len), dtype=np.float32)
            game_play, p1, p2 = key
            feature_path = f"{self.cfg.image_feature_dir2}_fold{self.fold}/{game_play}_p1_{p1}_p2_{p2}.npy"
            image_features2 = np.load(feature_path)
            inputs['image_features2'][:, :seq_len] = image_features2.T
            if np.isnan(inputs['image_features2'][:, :seq_len]).any():
                assert False, print('features contain nan')
        return inputs


class TestSeqTrackDataset(Dataset):
    def __init__(self, cfg):
        self.id_cols = [
            'game_play',
            'step',
            'datetime_ngs',
        ]
        self.label_col = 'contact'
        self.pair_cols = cfg.pair_cols
        self.player1_cols = cfg.player1_cols
        self.player2_cols = cfg.player2_cols
        self.global_cols = cfg.global_cols

        self.cfg = cfg
        self.df = setup_df(cfg.label_df_path, fold=-1, mode='test')
        self.sample_df = setup_df(cfg.df_path, fold=-1, mode='test')[['game_play', 'step']].drop_duplicates()
        # self.df = pd.read_csv(cfg.label_df_path)
        # self.sample_df = pd.read_csv(cfg.df_path)[['game_play', 'step']].drop_duplicates()
        self.df = pd.merge(self.df, self.sample_df)
        if cfg.ground_only:
            print('only ground data.')
            self.df = self.df.query('nfl_player_id_2 == @GROUND_ID')
        else:
            print('only inter data.')
            self.df = self.df.query('nfl_player_id_2 != @GROUND_ID')
        self.samples = self.df[['game_play', 'nfl_player_id_1', 'nfl_player_id_2']
                               ].drop_duplicates().reset_index(drop=True)
        self.gb = self.df.groupby(['game_play', 'nfl_player_id_1', 'nfl_player_id_2'])
        self.max_len = self.cfg.max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        key = tuple(self.samples.iloc[idx].values)
        pair_df = self.gb.get_group(key)
        is_ground = key[-1] <= 0  # 0 or -1
        seq_len = len(pair_df)

        inputs = {}
        game_play, p1, p2 = key
        inputs['game_play'] = game_play
        inputs['nfl_player_id_1'] = p1
        inputs['nfl_player_id_2'] = p2

        inputs['is_ground'] = is_ground
        inputs['p1_feats'] = np.zeros((len(self.player1_cols), self.max_len), dtype=np.float32)
        inputs['p1_feats'][:, :seq_len] = pair_df[self.player1_cols].astype(np.float32).values.T
        if self.cfg.ground_only:
            inputs['ground'] = np.zeros((self.max_len,), dtype=np.float32)
            inputs['ground'][:seq_len] = pair_df['contact'].astype(np.float32).values
        else:
            inputs['p2_feats'] = np.zeros((len(self.player2_cols), self.max_len), dtype=np.float32)
            inputs['p2_feats'][:, :seq_len] = pair_df[self.player2_cols].astype(np.float32).values.T
            inputs['pair_feats'] = np.zeros((len(self.pair_cols), self.max_len), dtype=np.float32)
            inputs['pair_feats'][:, :seq_len] = pair_df[self.pair_cols].astype(np.float32).values.T
            inputs['inter'] = np.zeros((self.max_len,), dtype=np.float32)
            inputs['inter'][:seq_len] = pair_df['contact'].astype(np.float32).values

        inputs['global_feats'] = np.zeros((len(self.global_cols), self.max_len), dtype=np.float32)
        inputs['global_feats'][:, :seq_len] = pair_df[self.global_cols].astype(np.float32).values.T
        inputs['masks'] = np.zeros((self.max_len,), dtype=np.bool)
        inputs['masks'][:seq_len] = True

        if self.cfg.image_feature_dir is not None:
            for fold in range(4):
                inputs[f'image_features_fold{fold}'] = np.zeros((576, self.max_len), dtype=np.float32)
                game_play, p1, p2 = key
                feature_path = f"{self.cfg.image_feature_dir}/features_fold{fold}/{game_play}_p1_{p1}_p2_{p2}.npy"
                image_features = np.load(feature_path)
                inputs[f'image_features_fold{fold}'][:, :seq_len] = image_features.T

        return inputs


def get_train_new_seq_track_dataloader(cfg, fold):
    dataset = TrainNewSeqTrackDataset(cfg, fold=fold, mode="train")
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


def get_val_new_seq_track_dataloader(cfg, fold):
    dataset = TrainNewSeqTrackDataset(cfg, fold=fold, mode="valid")
    val_dataloader = DataLoader(
        dataset,
        sampler=None,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=False,
        drop_last=False
    )
    return val_dataloader


def get_test_seq_track_dataloader(cfg):
    dataset = TestSeqTrackDataset(cfg)
    test_dataloader = DataLoader(
        dataset,
        sampler=None,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=False,
        drop_last=False
    )
    return test_dataloader

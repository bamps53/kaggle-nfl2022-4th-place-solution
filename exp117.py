from collections import defaultdict
import glob
import shutil
import sys
import warnings
from copy import deepcopy
import argparse
import gc
import os

import albumentations as A
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import matthews_corrcoef
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
try:
    from timm.scheduler import CosineLRScheduler
except:
    print('timm is not installed.')

from metric.metric import evaluate_pred_df

try:
    # import training only modules
    import wandb
except:
    print('wandb is not installed.')

from configs.v2 import cfg
from models.v2 import TrackAnyTransformerYoloxHMGlobal as Net


from optimizers import get_optimizer
from utils.debugger import set_debugger
from utils.common import set_seed, create_checkpoint, resume_checkpoint, batch_to_device, log_results
from utils.ema import ModelEmaV2
from datasets.v2 import get_train_dataloader, get_val_dataframe_dataloader, get_test_dataframe_dataloader
from datasets import video_transforms

warnings.simplefilter(action='ignore', category=FutureWarning)

FPS = 59.94
HEIGHT, WIDTH = 704, 1280
ORIGINAL_HEIGHT, ORIGINAL_WIDTH = 720, 1280
DURATION = 1
GROUND_ID = -1

cfg = deepcopy(cfg)
cfg.project = 'kaggle-nfl2022'
cfg.exp_name = 'exp117_hm_transformer_gc'
cfg.exp_id = cfg.exp_name.split('_')[0]
cfg.output_dir = f'output/{cfg.exp_name}'
cfg.debug = False

cfg.train.df_path = '../input/preprocessed_data/sample_df.csv'
cfg.train.unique_ids_dict_name = 'unique_ids_dict'
cfg.train.distance_dict_name = 'distance_dict'
cfg.train.inter_contact_dict_name = 'inter_contact_dict'
cfg.train.ground_contact_dict_name = 'ground_contact_dict'
cfg.train.helmet_dict_name = 'helmet_dict'
cfg.train.track_dict_name = 'track_dict'
cfg.train.pair_dict_name = 'pair_dict'
cfg.train.data_dir = '../input/preprocessed_data'
cfg.train.image_dir = '../input/train_frames'
cfg.train.num_track_features = 10
cfg.train.num_pair_features = 11

cfg.train.duration = DURATION
cfg.train.batch_size = 4
cfg.train.num_workers = 4
cfg.train.down_ratio = 8
cfg.train.roi_size = 'dynamic'
cfg.train.image_size = (HEIGHT, WIDTH)
cfg.train.original_image_size = (ORIGINAL_HEIGHT, ORIGINAL_WIDTH)
cfg.train.transforms = video_transforms.Compose([
    video_transforms.Resize((HEIGHT, WIDTH)),
])
cfg.train.image_transforms = A.Compose([
    # A.RandomCrop(width=WIDTH, height=HEIGHT),
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(p=1.0, rotate_limit=5),
    A.RandomBrightnessContrast(p=1.0),
    A.AdvancedBlur(),
], bbox_params=A.BboxParams(format='pascal_voc'))
cfg.train.normalize_coords = True
cfg.train.fix_coords_scale = True
cfg.train.enable_frame_noise = True

cfg.valid.df_path = '../input/preprocessed_data/sample_df.csv'
cfg.valid.label_df_path = '../input/preprocessed_data/label_df.csv'
cfg.valid.unique_ids_dict_name = 'unique_ids_dict'
cfg.valid.distance_dict_name = 'distance_dict'
cfg.valid.inter_contact_dict_name = 'inter_contact_dict'
cfg.valid.ground_contact_dict_name = 'ground_contact_dict'
cfg.valid.helmet_dict_name = 'helmet_dict'
cfg.valid.track_dict_name = 'track_dict'
cfg.valid.pair_dict_name = 'pair_dict'
cfg.valid.data_dir = '../input/preprocessed_data'
cfg.valid.image_dir = '../input/train_frames'
cfg.valid.num_track_features = 10
cfg.valid.num_pair_features = 11

cfg.valid.duration = DURATION
cfg.valid.batch_size = 4
cfg.valid.num_workers = 4
cfg.valid.down_ratio = 8
cfg.valid.roi_size = 'dynamic'
cfg.valid.image_size = (HEIGHT, WIDTH)
cfg.valid.original_image_size = (ORIGINAL_HEIGHT, ORIGINAL_WIDTH)
cfg.valid.transforms = video_transforms.Compose([
    video_transforms.Resize((HEIGHT, WIDTH)),
])
cfg.valid.image_transforms = None
cfg.valid.normalize_coords = True
cfg.valid.fix_coords_scale = True

cfg.test = deepcopy(cfg.valid)  # copy valid
cfg.test.num_workers = 0
cfg.test.data_dir = './test_preprocessed_data'
cfg.test.df_path = f'{cfg.test.data_dir}/sample_df.csv'
cfg.test.label_df_path = f'{cfg.test.data_dir}/label_df.csv'
cfg.test.image_dir = '/temp/test_frames'

cfg.model.inter_weight = 0.5
cfg.model.ground_weight = 0.5
cfg.model.any_weight = 0.5  # 足して1にならんけどまあいっか
cfg.model.pretrained_path = '../input/full_yolox_m_backbone.pth'
cfg.model.duration = DURATION
cfg.model.down_ratio = 8
cfg.model.label_smoothing = 0.0
cfg.model.return_feats = True

# others
cfg.seed = 42
cfg.device = 'cuda'
cfg.lr = 1.0e-4
cfg.wd = 1.0e-4
cfg.min_lr = 5.0e-5
cfg.warmup_lr = 1.0e-5
cfg.warmup_epochs = 3
cfg.warmup = 1
cfg.epochs = 10
cfg.eval_intervals = 1
cfg.mixed_precision = True
cfg.ema_start_epoch = 1


def get_model(cfg, weight_path=None):
    model = Net(cfg.model)
    if cfg.model.resume_exp is not None:
        weight_path = os.path.join(
            cfg.root, 'output', cfg.model.resume_exp, f'best_fold{cfg.fold}.pth')
    if weight_path is not None:
        state_dict = torch.load(weight_path, map_location='cpu')
        epoch = state_dict['epoch']
        model_key = 'model_ema'
        if model_key not in state_dict.keys():
            model_key = 'model'
            print(f'load epoch {epoch} model from {weight_path}')
        else:
            print(f'load epoch {epoch} ema model from {weight_path}')

        model.load_state_dict(state_dict[model_key])

    return model.to(cfg.device)


def save_val_results(targets, preds, save_path):
    num_classes = targets.shape[1]
    df = pd.DataFrame()
    for c in range(num_classes):
        df[f'target_{c}'] = targets[:, c]
        df[f'pred_{c}'] = preds[:, c]
    df.to_csv(save_path, index=False)


OUTPUT_KEYS = ['inter', 'ground', 'any']


def train(cfg, fold):
    os.makedirs(str(cfg.output_dir + "/"), exist_ok=True)
    cfg.fold = fold
    mode = 'disabled' if cfg.debug else None
    wandb.init(project=cfg.project,
               name=f'{cfg.exp_name}_fold{fold}', config=cfg, reinit=True, mode=mode)
    set_seed(cfg.seed)
    train_dataloader = get_train_dataloader(cfg.train, fold)
    valid_dataloader = get_val_dataframe_dataloader(cfg.valid, fold)
    model = get_model(cfg)

    # if cfg.model.grad_checkpointing:
    #     model.set_grad_checkpointing(enable=True)

    # setup exponential moving average of model weights, SWA could be used here too
    model_ema = ModelEmaV2(model, decay=0.999)

    optimizer = get_optimizer(model, cfg)
    steps_per_epoch = len(train_dataloader)
    scheduler = CosineLRScheduler(
        optimizer,
        t_initial=cfg.epochs*steps_per_epoch,
        lr_min=cfg.min_lr,
        warmup_lr_init=cfg.warmup_lr,
        warmup_t=cfg.warmup_epochs*steps_per_epoch,
        k_decay=1.0,
    )

    scaler = GradScaler(enabled=cfg.mixed_precision)
    init_epoch = 0
    best_val_score = 0
    ckpt_path = f"{cfg.output_dir}/last_fold{fold}.pth"
    if cfg.resume and os.path.exists(ckpt_path):
        model, optimizer, init_epoch, best_val_score, scheduler, scaler, model_ema = resume_checkpoint(
            f"{cfg.output_dir}/last_fold{fold}.pth",
            model,
            optimizer,
            scheduler,
            scaler,
            model_ema
        )

    cfg.curr_step = 0
    i = init_epoch * steps_per_epoch

    optimizer.zero_grad()
    for epoch in range(init_epoch, cfg.epochs):
        set_seed(cfg.seed + epoch)

        cfg.curr_epoch = epoch

        progress_bar = tqdm(range(len(train_dataloader)),
                            leave=False,  dynamic_ncols=True)
        tr_it = iter(train_dataloader)

        train_outputs = defaultdict(list)
        gc.collect()

        # ==== TRAIN LOOP
        for itr in progress_bar:
            i += 1
            cfg.curr_step += cfg.train.batch_size

            model.train()
            torch.set_grad_enabled(True)

            inputs = next(tr_it)
            inputs = batch_to_device(inputs, cfg.device, cfg.mixed_precision)

            optimizer.zero_grad()
            with autocast(enabled=cfg.mixed_precision):
                outputs = model(inputs)
                loss_dict = model.get_loss(outputs, inputs)
                loss = loss_dict['loss']

            for key in OUTPUT_KEYS:
                train_outputs[key + "_loss"].append(loss_dict[key].item())
                train_outputs[key + "_labels"].append(inputs[key].cpu().numpy())
                train_outputs[key + "_preds"].append(outputs[key].sigmoid().detach().cpu().numpy())
                train_outputs[key + "_masks"].append(outputs[key + '_masks'].cpu().numpy())

            if torch.isfinite(loss):
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            if model_ema is not None:
                model_ema.update(model)

            if scheduler is not None:
                scheduler.step(i)

            NUM_RECENT = 10
            THRESHOLD = 0.3
            text = f"step:{i} "
            for key in OUTPUT_KEYS:
                key_labels = np.concatenate(train_outputs[key + "_labels"][-NUM_RECENT:], axis=0).reshape(-1)
                key_preds = np.concatenate(train_outputs[key + "_preds"][-NUM_RECENT:], axis=0).reshape(-1)
                key_masks = np.concatenate(train_outputs[key + "_masks"][-NUM_RECENT:], axis=0).reshape(-1)
                key_mcc = matthews_corrcoef(key_labels,
                                            key_preds > THRESHOLD,
                                            sample_weight=key_masks)
                key_loss = np.mean(train_outputs[key + "_loss"][-NUM_RECENT:])
                text += f"{key}_loss: {key_loss:.4f} {key}_mcc@0.3: {key_mcc:.3f} "

            lr = optimizer.param_groups[0]['lr']
            text += f"lr:{lr:.6}"
            progress_bar.set_description(text)

        all_train_outputs = {}
        for key in OUTPUT_KEYS:
            all_train_outputs[key + '_loss'] = np.mean(train_outputs[key + "_loss"][-NUM_RECENT:])  # lossは直近のもの
            all_train_outputs[key + '_labels'] = np.concatenate(train_outputs[key + "_labels"], axis=0).reshape(-1)
            all_train_outputs[key + '_preds'] = np.concatenate(train_outputs[key + "_preds"], axis=0).reshape(-1)
            all_train_outputs[key + '_masks'] = np.concatenate(train_outputs[key + "_masks"], axis=0).reshape(-1)

            all_train_outputs[key + '_mcc'] = matthews_corrcoef(all_train_outputs[key + '_labels'],
                                                                all_train_outputs[key + '_preds'] > THRESHOLD,
                                                                sample_weight=all_train_outputs[key + '_masks'])

        all_labels = np.concatenate([
            all_train_outputs['inter_labels'],
            all_train_outputs['ground_labels']])
        all_preds = np.concatenate([
            all_train_outputs['inter_preds'],
            all_train_outputs['ground_preds']])
        all_masks = np.concatenate([
            all_train_outputs['inter_masks'],
            all_train_outputs['ground_masks']])
        score = matthews_corrcoef(all_labels, all_preds > THRESHOLD, sample_weight=all_masks)

        if (epoch % cfg.eval_intervals == 0) or (epoch > 30):
            if model_ema is not None:
                val_results = full_validate(cfg, fold, model_ema.module, valid_dataloader)
            else:
                val_results = full_validate(cfg, fold, model, valid_dataloader)
        else:
            val_results = {}
        lr = optimizer.param_groups[0]['lr']

        all_results = {
            'epoch': epoch,
            'lr': lr,
        }
        train_results = {'score': score}
        for key in OUTPUT_KEYS:
            train_results[key + "_loss"] = all_train_outputs[key + '_loss']
            train_results[key + "_score"] = all_train_outputs[key + '_mcc']
        log_results(all_results, train_results, val_results)

        val_score = val_results.get('score', 0.0)
        if best_val_score < val_score:
            best_val_score = val_score
            checkpoint = create_checkpoint(
                model, optimizer, epoch, scheduler=scheduler, scaler=scaler, score=best_val_score,
                model_ema=model_ema
            )
            torch.save(checkpoint, f"{cfg.output_dir}/best_fold{fold}.pth")
            shutil.copy(f'{cfg.output_dir}/val_preds_df_fold{fold}.csv',
                        f'{cfg.output_dir}/best_val_preds_df_fold{fold}.csv')
            shutil.copy(f'{cfg.output_dir}/val_any_preds_df_fold{fold}.csv',
                        f'{cfg.output_dir}/best_val_any_preds_df_fold{fold}.csv')

        checkpoint = create_checkpoint(
            model, optimizer, epoch, scheduler=scheduler, scaler=scaler, model_ema=model_ema)
        torch.save(checkpoint, f"{cfg.output_dir}/last_fold{fold}.pth")

        if epoch == 4:  # magic
            torch.save(checkpoint, f"{cfg.output_dir}/epoch4_fold{fold}.pth")
            shutil.copy(f'{cfg.output_dir}/val_preds_df_fold{fold}.csv',
                        f'{cfg.output_dir}/epoch4_val_preds_df_fold{fold}.csv')
            shutil.copy(f'{cfg.output_dir}/val_any_preds_df_fold{fold}.csv',
                        f'{cfg.output_dir}/epoch4_val_any_preds_df_fold{fold}.csv')


def get_preds_and_masks(inputs, outputs, train_pairs_gb, folds_features=False):
    dfs = []
    any_dfs = []
    all_features = []
    num_samples = len(inputs['frame'])

    batch_game_plays = inputs['game_play']
    batch_steps = inputs['step'].cpu().numpy().astype(int)
    batch_frames = inputs['frame'].cpu().numpy().astype(int)
    batch_unique_ids = inputs['unique_ids'].cpu().numpy().astype(int)
    batch_ground_preds = outputs['ground'].sigmoid().cpu().numpy()
    batch_ground_masks = outputs['ground_masks'].cpu().numpy()
    batch_inter_preds = outputs['inter'].sigmoid().cpu().numpy()
    batch_inter_masks = outputs['inter_masks'].cpu().numpy()
    batch_any_preds = outputs['any'].sigmoid().cpu().numpy()

    if folds_features:
        batch_inter_x = np.stack([outputs[f'inter_x_fold{i}'].cpu().numpy() for i in range(4)], axis=-1)
        batch_g_x = np.stack([outputs[f'g_x_fold{i}'].cpu().numpy() for i in range(4)], axis=-1)
    else:
        batch_inter_x = outputs['inter_x'].cpu().numpy()
        batch_g_x = outputs['g_x'].cpu().numpy()

    for idx in range(num_samples):
        game_play = batch_game_plays[idx]
        step = batch_steps[idx]
        frame = batch_frames[idx]
        pairs = train_pairs_gb.get_group((game_play, step)).values
        unique_ids = batch_unique_ids[idx].tolist()
        ground_masks = batch_ground_masks[idx]
        ground_preds = batch_ground_preds[idx]
        inter_masks = batch_inter_masks[idx]
        inter_preds = batch_inter_preds[idx]
        any_preds = batch_any_preds[idx]

        inter_x = batch_inter_x[idx]
        g_x = batch_g_x[idx]

        preds_list = []
        masks_list = []
        feats_list = []
        any_preds_list = []
        any_masks_list = []
        any_ids_list = []

        for id1, id2 in pairs:
            idx1 = unique_ids.index(id1)
            if (id2 == -1) | (id2 == 0):
                if ground_masks[idx1]:
                    preds_list.append(ground_preds[idx1])
                    masks_list.append(True)
                    feats_list.append(g_x[idx1])
                    any_preds_list.append(any_preds[idx1])
                    any_masks_list.append(True)
                    any_ids_list.append(id1)
                else:
                    preds_list.append(0.0)
                    masks_list.append(False)
                    feats_list.append(np.zeros_like(g_x[idx1]))
                    any_preds_list.append(0.0)
                    any_masks_list.append(False)
                    any_ids_list.append(id1)
            else:
                idx2 = unique_ids.index(id2)
                if (inter_masks[idx1, idx2]):
                    # print(abs(inter_preds[idx1, idx2] - inter_preds[idx2, idx1]))
                    preds_list.append((inter_preds[idx1, idx2] + inter_preds[idx2, idx1]) / 2.0)
                    masks_list.append(True)
                    feats_list.append((inter_x[idx1, idx2] + inter_x[idx2, idx1]) / 2.0)
                else:
                    preds_list.append(0.0)
                    masks_list.append(False)
                    feats_list.append(np.zeros_like(inter_x[idx1, idx2]))
        df = pd.DataFrame(data={'preds': preds_list, "masks": masks_list})
        df['game_play'] = game_play
        df['step'] = step
        df['frame'] = frame
        df["nfl_player_id_1"] = pairs[:, 0]
        df["nfl_player_id_2"] = pairs[:, 1]
        dfs.append(df)

        any_df = pd.DataFrame(data={'preds': any_preds_list, "masks": any_masks_list, "nfl_player_id_1": any_ids_list})
        any_df['game_play'] = game_play
        any_df['step'] = step
        any_df['frame'] = frame
        any_dfs.append(any_df)

        all_features.append(np.stack(feats_list))
    df = pd.concat(dfs).reset_index(drop=True)
    any_df = pd.concat(any_dfs).reset_index(drop=True)
    all_features = np.concatenate(all_features)
    return df, any_df, all_features


def full_validate(cfg, fold, model=None, test_dataloader=None, use_all_frames=False):
    if model is None:
        weight_path = f"{cfg.output_dir}/best_fold{fold}.pth"
        model = get_model(cfg, weight_path)
    model.eval()
    torch.set_grad_enabled(False)

    if test_dataloader is None:
        test_dataloader = get_val_dataframe_dataloader(cfg.valid, fold, use_all_frames)

    pairs_gb = test_dataloader.dataset.pairs_gb

    dfs = []
    any_dfs = []
    all_features = []
    for inputs in tqdm(test_dataloader):
        inputs = batch_to_device(inputs, cfg.device, cfg.mixed_precision)
        with torch.no_grad() and autocast(cfg.mixed_precision):
            outputs = model(inputs)
        df, any_df, features = get_preds_and_masks(inputs, outputs, pairs_gb)
        all_features.append(features)
        dfs.append(df)
        any_dfs.append(any_df)
    pred_df = pd.concat(dfs).reset_index(drop=True)
    any_pred_df = pd.concat(any_dfs).reset_index(drop=True)
    all_features = np.concatenate(all_features)
    if not use_all_frames:
        np.save(f'{cfg.output_dir}/val_features_fold{fold}.npy', all_features)
        pred_df.to_csv(f'{cfg.output_dir}/val_preds_df_fold{fold}.csv', index=False)
        any_pred_df.to_csv(f'{cfg.output_dir}/val_any_preds_df_fold{fold}.csv', index=False)
        return evaluate_pred_df(test_dataloader.dataset.df, pred_df, ground_id=GROUND_ID)
    else:
        pred_df.to_csv(f'{cfg.output_dir}/val_frame_preds_df_fold{fold}.csv', index=False)
        any_pred_df.to_csv(f'{cfg.output_dir}/val_frame_any_preds_df_fold{fold}.csv', index=False)


class Models(nn.Module):
    def __init__(self, weight_paths):
        super().__init__()
        self.models = [get_model(cfg, p) for p in sorted(weight_paths)]
        [m.eval() for m in self.models]

    def ensemble(self, inputs):
        with torch.no_grad():
            outputs = [m(inputs) for m in self.models]
        ensemble_output = {}
        ensemble_output['ground_masks'] = outputs[0]['ground_masks']
        ensemble_output['ground'] = torch.mean(torch.stack([output['ground'] for output in outputs], dim=0), dim=0)
        ensemble_output['inter_masks'] = outputs[0]['inter_masks']
        ensemble_output['inter'] = torch.mean(torch.stack([output['inter'] for output in outputs], dim=0), dim=0)
        ensemble_output['any'] = torch.mean(torch.stack([output['any'] for output in outputs], dim=0), dim=0)
        for i, output in enumerate(outputs):
            ensemble_output[f'inter_x_fold{i}'] = output['inter_x']
            ensemble_output[f'g_x_fold{i}'] = output['g_x']

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


def inference(cfg, model_dir, use_all_frames=False):
    cfg.model.pretrained_path = None
    weight_paths = sorted(glob.glob(f'{model_dir}/best_fold*'))
    model = Models(weight_paths)

    torch.set_grad_enabled(False)

    test_dataloader = get_test_dataframe_dataloader(cfg.test, use_all_frames)

    pairs_gb = test_dataloader.dataset.pairs_gb

    dfs = []
    any_dfs = []
    # all_features = []
    for inputs in tqdm(test_dataloader):
        inputs = batch_to_device(inputs, cfg.device, cfg.mixed_precision)
        with torch.no_grad() and autocast(cfg.mixed_precision):
            outputs = model(inputs)
        df, any_df, features = get_preds_and_masks(inputs, outputs, pairs_gb, folds_features=True)
        # all_features.append(features)
        dfs.append(df)
        any_dfs.append(any_df)
    pred_df = pd.concat(dfs).reset_index(drop=True)
    any_pred_df = pd.concat(any_dfs).reset_index(drop=True)
    # all_features = np.concatenate(all_features)
    pred_df.to_csv(f'{cfg.exp_id}_preds_df.csv', index=False)
    any_pred_df.to_csv(f'{cfg.exp_id}_any_preds_df.csv', index=False)

    # save_dir = f'{cfg.test.data_dir}/features'
    # os.makedirs(save_dir, exist_ok=True)
    # df = pd.read_csv(f'{cfg.test.data_dir}/test_preds_df.csv').reset_index()

    # # to include skipped step by missing videos
    # test_df = pd.read_csv(cfg.test.label_df_path)
    # df = test_df[['game_play', 'step', 'nfl_player_id_1', 'nfl_player_id_2']].merge(
    #     df, on=['game_play', 'step', 'nfl_player_id_1', 'nfl_player_id_2'], how='left')

    # n_channels = all_features.shape[1]
    # for (game_play, p1, p2), group_df in tqdm(df.groupby(['game_play', 'nfl_player_id_1', 'nfl_player_id_2'])):
    #     group_df = group_df.sort_values('step')
    #     not_null = group_df['index'].notnull()
    #     indices = group_df['index'][not_null].astype(int).values
    #     filled_features = np.zeros((len(group_df), n_channels, 4))
    #     filled_features[not_null] = all_features[indices]
    #     file_name = f"{save_dir}/{game_play}_p1_{p1}_p2_{p2}.npy"
    #     np.save(file_name, filled_features)


def inference_v2(cfg, model_dir, use_all_frames=False):
    """game_playごとに推論して特徴保存"""
    cfg.model.pretrained_path = None
    weight_paths = sorted(glob.glob(f'{model_dir}/best_fold*'))
    model = Models(weight_paths)

    torch.set_grad_enabled(False)

    test_dataloader = get_test_dataframe_dataloader(cfg.test, use_all_frames)

    pairs_gb = test_dataloader.dataset.pairs_gb

    game_plays = test_dataloader.dataset.sample_df['game_play'].unique()
    test_df = pd.read_csv(cfg.test.label_df_path)

    for game_play in game_plays:
        test_dataloader.dataset.set_game_play(game_play)

        dfs = []
        any_dfs = []
        all_features = []
        for inputs in tqdm(test_dataloader):
            inputs = batch_to_device(inputs, cfg.device, cfg.mixed_precision)
            with torch.no_grad() and autocast(cfg.mixed_precision):
                outputs = model(inputs)
            df, any_df, features = get_preds_and_masks(inputs, outputs, pairs_gb, folds_features=True)
            all_features.append(features)
            dfs.append(df)
            any_dfs.append(any_df)
        pred_df = pd.concat(dfs).reset_index(drop=True)
        any_pred_df = pd.concat(any_dfs).reset_index(drop=True)
        all_features = np.concatenate(all_features)
        pred_df.to_csv(f'{cfg.exp_id}_preds_df_{game_play}.csv', index=False)
        any_pred_df.to_csv(f'{cfg.exp_id}_any_preds_df_{game_play}.csv', index=False)

        save_dir = f'{cfg.test.data_dir}/exp117_features'
        os.makedirs(save_dir, exist_ok=True)

        game_df = test_df.query('game_play == @game_play').reset_index(drop=True)
        # to include skipped step by missing videos
        pred_df = pred_df.reset_index()
        pred_df = game_df[['game_play', 'step', 'nfl_player_id_1', 'nfl_player_id_2']].merge(
            pred_df, on=['game_play', 'step', 'nfl_player_id_1', 'nfl_player_id_2'], how='left')

        n_channels = all_features.shape[1]
        for (game_play, p1, p2), group_df in tqdm(pred_df.groupby(['game_play', 'nfl_player_id_1', 'nfl_player_id_2'])):
            group_df = group_df.sort_values('step')
            not_null = group_df['index'].notnull()
            indices = group_df['index'][not_null].astype(int).values
            filled_features = np.zeros((len(group_df), n_channels, 4))
            filled_features[not_null] = all_features[indices]
            file_name = f"{save_dir}/{game_play}_p1_{p1}_p2_{p2}.npy"
            np.save(file_name, filled_features)


def extract_feats_for_all_train_data(cfg, fold, model=None, test_dataloader=None, use_all_frames=None, is_flip=False):
    if model is None:
        weight_path = f"{cfg.output_dir}/best_fold{fold}.pth"
        model = get_model(cfg, weight_path)
    model.eval()
    torch.set_grad_enabled(False)

    if test_dataloader is None:
        test_dataloader = get_val_dataframe_dataloader(cfg.valid, fold=-1, use_all_frames=False, is_flip=is_flip)

    pairs_gb = test_dataloader.dataset.pairs_gb

    dfs = []
    all_features = []
    for inputs in tqdm(test_dataloader):
        inputs = batch_to_device(inputs, cfg.device, cfg.mixed_precision)
        with torch.no_grad() and autocast(cfg.mixed_precision):
            outputs = model(inputs)
        df, _, features = get_preds_and_masks(inputs, outputs, pairs_gb)
        dfs.append(df)
        all_features.append(features)
    pred_df = pd.concat(dfs).reset_index(drop=True)
    all_features = np.concatenate(all_features)
    if is_flip:
        pred_df.to_csv(f'{cfg.output_dir}/train_val_flipped_preds_df_fold{fold}.csv', index=False)
        np.save(f'{cfg.output_dir}/train_val_flipped_features_fold{fold}.npy', all_features)
    else:
        pred_df.to_csv(f'{cfg.output_dir}/train_val_preds_df_fold{fold}.csv', index=False)
        np.save(f'{cfg.output_dir}/train_val_features_fold{fold}.npy', all_features)


def merge(train_df, fold):
    save_dir = f'{cfg.output_dir}/features_fold{fold}'
    os.makedirs(save_dir, exist_ok=True)
    df = pd.read_csv(f'{cfg.output_dir}/train_val_preds_df_fold{fold}.csv').reset_index()
    df = train_df[['game_play', 'step', 'nfl_player_id_1', 'nfl_player_id_2']].merge(
        df, on=['game_play', 'step', 'nfl_player_id_1', 'nfl_player_id_2'], how='left')
    features = np.load(f'{cfg.output_dir}/train_val_features_fold{fold}.npy')
    n_channels = features.shape[1]
    for (game_play, p1, p2), group_df in tqdm(df.groupby(['game_play', 'nfl_player_id_1', 'nfl_player_id_2'])):
        group_df = group_df.sort_values('step')
        not_null = group_df['index'].notnull()
        indices = group_df['index'][not_null].astype(int).values
        filled_features = np.zeros((len(group_df), n_channels))
        filled_features[not_null] = features[indices]
        file_name = f"{save_dir}/{game_play}_p1_{p1}_p2_{p2}.npy"
        np.save(file_name, filled_features)


def flipped_merge(train_df, fold):
    save_dir = f'{cfg.output_dir}/features_fold{fold}'
    os.makedirs(save_dir, exist_ok=True)
    df = pd.read_csv(f'{cfg.output_dir}/train_val_flipped_preds_df_fold{fold}.csv').reset_index()
    df = train_df[['game_play', 'step', 'nfl_player_id_1', 'nfl_player_id_2']].merge(
        df, on=['game_play', 'step', 'nfl_player_id_1', 'nfl_player_id_2'], how='left')
    features = np.load(f'{cfg.output_dir}/train_val_flipped_features_fold{fold}.npy')
    n_channels = features.shape[1]
    for (game_play, p1, p2), group_df in tqdm(df.groupby(['game_play', 'nfl_player_id_1', 'nfl_player_id_2'])):
        group_df = group_df.sort_values('step')
        not_null = group_df['index'].notnull()
        indices = group_df['index'][not_null].astype(int).values
        filled_features = np.zeros((len(group_df), n_channels))
        filled_features[not_null] = features[indices]
        file_name = f"{save_dir}/{game_play}_p1_{p1}_p2_{p2}_flip.npy"
        np.save(file_name, filled_features)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="./", type=str)
    parser.add_argument("--device_id", "-d", default="0", type=str)
    parser.add_argument("--start_fold", "-s", default=0, type=int)
    parser.add_argument("--end_fold", "-e", default=4, type=int)
    parser.add_argument("--validate", "-v", action="store_true")
    parser.add_argument("--extract", "-x", action="store_true")
    parser.add_argument("--merge", "-m", action="store_true")
    parser.add_argument("--infer", "-i", action="store_true")
    parser.add_argument("--debug", "-db", action="store_true")
    parser.add_argument("--resume", "-r", action="store_true")
    parser.add_argument("--model_dir", default="./", type=str)
    parser.add_argument("--all_frames", "-a", action="store_true")
    parser.add_argument("--flip", "-f", action="store_true")

    return parser.parse_args()


def update_cfg(cfg, args, fold):
    if args.debug:
        cfg.debug = True
        cfg.train.num_workers = 4 if not cfg.debug else 0
        cfg.valid.num_workers = 4 if not cfg.debug else 0
        set_debugger()

    cfg.fold = fold

    if args.resume:
        cfg.resume = True

    cfg.root = args.root

    cfg.output_dir = os.path.join(args.root, cfg.output_dir)

    if cfg.model.resume_exp is not None:
        cfg.model.pretrained_path = os.path.join(
            cfg.root, 'output', cfg.model.resume_exp, f'best_fold{cfg.fold}.pth')

    return cfg


if __name__ == "__main__":
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id)

    if args.infer:
        inference_v2(cfg, args.model_dir, use_all_frames=args.all_frames)
        sys.exit()

    for fold in range(args.start_fold, args.end_fold):
        cfg = update_cfg(cfg, args, fold)
        if args.validate:
            full_validate(cfg, fold, use_all_frames=args.all_frames)
        elif args.extract:
            extract_feats_for_all_train_data(cfg, fold, is_flip=args.flip)
            train_df = pd.read_feather(cfg.valid.label_df_path)
            if args.flip:
                flipped_merge(train_df, fold)
            else:
                merge(train_df, fold)
        elif args.merge:
            train_df = pd.read_feather(cfg.valid.label_df_path)
            merge(train_df, fold)
        else:
            train(cfg, fold)

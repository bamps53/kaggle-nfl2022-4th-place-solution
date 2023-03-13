import glob
import shutil
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
from metric.metric import search_best_threshold

try:
    # import training only modules
    import wandb
except:
    print('wandb is not installed.')

from configs.base import cfg
from models.unet1d import ImageFeatUNet1dGlobal as Net

from optimizers import get_optimizer
from utils.debugger import set_debugger
from utils.common import set_seed, create_checkpoint, resume_checkpoint, batch_to_device, log_results
from utils.ema import ModelEmaV2
from datasets.v3_seq import get_train_new_seq_track_dataloader as get_train_dataloader
from datasets.v3_seq import get_val_new_seq_track_dataloader as get_val_dataloader
from datasets.v3_seq import get_test_seq_track_dataloader as get_test_dataloader

warnings.simplefilter(action='ignore', category=FutureWarning)

FPS = 59.94
HEIGHT, WIDTH = 352, 640
ORIGINAL_HEIGHT, ORIGINAL_WIDTH = 720, 1280
MAX_LEN = 176

col_info = {'id': ['game_play',
                   'step',
                   'nfl_player_id_1',
                   'nfl_player_id_2',
                   'contact',
                   'datetime_ngs'],
            'p1': ['width_Sideline_1',
                   'height_Sideline_1',
                   'bbox_center_x_Sideline_1',
                   'bbox_center_y_Sideline_1',
                   'bbox_center_x_std_Sideline_1',
                   'bbox_center_y_std_Sideline_1',
                   'width_Endzone_1',
                   'height_Endzone_1',
                   'bbox_center_x_Endzone_1',
                   'bbox_center_y_Endzone_1',
                   'bbox_center_x_std_Endzone_1',
                   'bbox_center_y_std_Endzone_1',
                   'bbox_smooth_velx_Sideline_1',
                   'bbox_smooth_vely_Sideline_1',
                   'bbox_smooth_x_Sideline_1',
                   'bbox_smooth_y_Sideline_1',
                   'bbox_smooth_outlier_Sideline_1',
                   'bbox_smooth_accx_Sideline_1',
                   'bbox_smooth_accy_Sideline_1',
                   'bbox_smooth_velx_Endzone_1',
                   'bbox_smooth_vely_Endzone_1',
                   'bbox_smooth_x_Endzone_1',
                   'bbox_smooth_y_Endzone_1',
                   'bbox_smooth_outlier_Endzone_1',
                   'bbox_smooth_accx_Endzone_1',
                   'bbox_smooth_accy_Endzone_1',
                   'x_position_1',
                   'y_position_1',
                   'speed_1',
                   'distance_1',
                   'direction_1',
                   'orientation_1',
                   'acceleration_1',
                   'sa_1',
                   'mean_distance_around_player_full_1',
                   'std_distance_around_player_full_1',
                   'idxmin_distance_aronud_player_full_1',
                   'distance_1st_1',
                   'distance_2nd_1',
                   'bbox_smooth_y_Sideline_1_dist',
                   'bbox_smooth_y_Endzone_1_dist',
                   'x_position_team_mean_1',
                   'y_position_team_mean_1',
                   'speed_team_mean_1',
                   'acceleration_team_mean_1',
                   'sa_team_mean_1',
                   'sa_player_mean_1',
                   'sa_player_max_1',
                   'acceleration_player_mean_1',
                   'acceleration_player_max_1',
                   'speed_player_mean_1',
                   'speed_player_max_1',
                   'mean_distance_around_player_1',
                   'min_distance_around_player_1',
                   'std_distance_around_player_1',
                   'idxmin_distance_aronud_player_1',
                   'aspect_Sideline_1',
                   'aspect_Endzone_1',
                   'distance_from_mean_1',
                   'distance_ratio_distance_to_min_distance_around_player_1',
                   'distance_of_interceptor_1',
                   'angle_interceptor_1',
                   'nfl_player_id_interceptor_1'],
            'p1_additional': ['x_position_offset_on_img_Side',
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
                              'p2p_registration_residual_frame_End'],
            'p2': ['width_Sideline_2',
                   'height_Sideline_2',
                   'bbox_center_x_Sideline_2',
                   'bbox_center_y_Sideline_2',
                   'bbox_center_x_std_Sideline_2',
                   'bbox_center_y_std_Sideline_2',
                   'width_Endzone_2',
                   'height_Endzone_2',
                   'bbox_center_x_Endzone_2',
                   'bbox_center_y_Endzone_2',
                   'bbox_center_x_std_Endzone_2',
                   'bbox_center_y_std_Endzone_2',
                   'bbox_smooth_velx_Sideline_2',
                   'bbox_smooth_vely_Sideline_2',
                   'bbox_smooth_x_Sideline_2',
                   'bbox_smooth_y_Sideline_2',
                   'bbox_smooth_outlier_Sideline_2',
                   'bbox_smooth_accx_Sideline_2',
                   'bbox_smooth_accy_Sideline_2',
                   'bbox_smooth_velx_Endzone_2',
                   'bbox_smooth_vely_Endzone_2',
                   'bbox_smooth_x_Endzone_2',
                   'bbox_smooth_y_Endzone_2',
                   'bbox_smooth_outlier_Endzone_2',
                   'bbox_smooth_accx_Endzone_2',
                   'bbox_smooth_accy_Endzone_2',
                   'x_position_2',
                   'y_position_2',
                   'speed_2',
                   'distance_2',
                   'direction_2',
                   'orientation_2',
                   'acceleration_2',
                   'sa_2',
                   'mean_distance_around_player_full_2',
                   'std_distance_around_player_full_2',
                   'idxmin_distance_aronud_player_full_2',
                   'distance_1st_2',
                   'distance_2nd_2',
                   'bbox_smooth_y_Sideline_2_dist',
                   'bbox_smooth_y_Endzone_2_dist',
                   'x_position_team_mean_2',
                   'y_position_team_mean_2',
                   'speed_team_mean_2',
                   'acceleration_team_mean_2',
                   'sa_team_mean_2',
                   'sa_player_mean_2',
                   'sa_player_max_2',
                   'acceleration_player_mean_2',
                   'acceleration_player_max_2',
                   'speed_player_mean_2',
                   'speed_player_max_2',
                   'mean_distance_around_player_2',
                   'min_distance_around_player_2',
                   'std_distance_around_player_2',
                   'idxmin_distance_aronud_player_2',
                   'aspect_Sideline_2',
                   'aspect_Endzone_2',
                   'distance_from_mean_2',
                   'distance_ratio_distance_to_min_distance_around_player_2',
                   'distance_of_interceptor_2',
                   'angle_interceptor_2',
                   'nfl_player_id_interceptor_2'],
            'pair': ['bbox_iou_Sideline',
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
                     'angle_dxdy'],
            'global': ['width_Sideline_mean',
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
                       'distance_mean_in_play']}

cfg = deepcopy(cfg)
cfg.project = 'kaggle-nfl2022'
cfg.exp_name = 'exp184_stage2_from_exp139_inter_all_feats_v3'
cfg.output_dir = f'output/{cfg.exp_name}'
cfg.debug = False

cfg.train.df_path = '../input/preprocessed_data/sample_df.csv'
cfg.train.label_df_path = 'output/features_for_seq_model/preprocessed_feature_df_for_seq_model.f'
cfg.train.batch_size = 32
cfg.train.num_workers = 4 if not cfg.debug else 0
cfg.train.max_len = MAX_LEN
cfg.train.image_feature_dir = 'output/exp139_hm_transformer_gc_more_aug_fixed_mask/features'
cfg.train.pair_cols = col_info['pair']
cfg.train.player1_cols = col_info['p1']
cfg.train.player2_cols = col_info['p2']
cfg.train.global_cols = col_info['global']

cfg.valid.df_path = '../input/preprocessed_data/sample_df.csv'
cfg.valid.label_df_path = 'output/features_for_seq_model/preprocessed_feature_df_for_seq_model.f'
cfg.valid.batch_size = 32
cfg.valid.num_workers = 4 if not cfg.debug else 0
cfg.valid.max_len = MAX_LEN
cfg.valid.image_feature_dir = 'output/exp139_hm_transformer_gc_more_aug_fixed_mask/features'
cfg.valid.pair_cols = col_info['pair']
cfg.valid.player1_cols = col_info['p1']
cfg.valid.player2_cols = col_info['p2']
cfg.valid.global_cols = col_info['global']


cfg.test.df_path = './test_preprocessed_data/sample_df.csv'
cfg.test.label_df_path = '../nfl-player-contact-detection/test_preprocessed_feature_df_for_seq_model.f'
cfg.test.batch_size = 32
cfg.test.num_workers = 0
cfg.test.max_len = MAX_LEN
cfg.test.include_ground = False
cfg.test.ground_only = False
cfg.test.image_feature_dir = './test_preprocessed_data'
cfg.test.pair_cols = col_info['pair']
cfg.test.player1_cols = col_info['p1']
cfg.test.player2_cols = col_info['p2']
cfg.test.global_cols = col_info['global']

cfg.model.inter_weight = 0.5
cfg.model.ground_weight = 0.5
cfg.model.max_len = MAX_LEN
cfg.model.in_player_channels = len(col_info['p1'])
cfg.model.in_pair_channels = len(col_info['pair'])
cfg.model.in_global_channels = len(col_info['global'])
cfg.model.hidden_channels = 128
cfg.model.kernel_size = 7
cfg.model.bilinear = True
cfg.model.drop_rate = 0.1
cfg.model.pos_ratio = 0.005
cfg.model.in_image_channels = 576

# others
cfg.seed = 42
cfg.device = 'cuda'
cfg.lr = 1.0e-3
cfg.wd = 1.0e-3
cfg.min_lr = 1.0e-4
cfg.warmup_lr = 1.0e-4
cfg.warmup_epochs = 1
cfg.warmup = 1
cfg.epochs = 10
cfg.eval_intervals = 1
cfg.mixed_precision = False
cfg.ema_start_epoch = 1


class Models(nn.Module):
    def __init__(self, weight_paths):
        super().__init__()
        self.models = [get_model(cfg, p) for p in sorted(weight_paths)]
        [m.eval() for m in self.models]

    def ensemble(self, inputs):
        with torch.no_grad():
            outputs = [m(inputs, fold) for fold, m in enumerate(self.models)]
        ensemble_output = {}
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


def check_helmet_exists(inputs):
    num_side = (inputs['Sideline_rois'].sum(-1) > 0).sum()
    num_end = (inputs['Endzone_rois'].sum(-1) > 0).sum()
    return (num_side + num_end) > 0


def train(cfg, fold):
    os.makedirs(str(cfg.output_dir + "/"), exist_ok=True)
    cfg.fold = fold
    mode = 'disabled' if cfg.debug else None
    wandb.init(project=cfg.project,
               name=f'{cfg.exp_name}_fold{fold}', config=cfg, reinit=True, mode=mode)
    set_seed(cfg.seed)
    train_dataloader = get_train_dataloader(cfg.train, fold)
    test_dataloader = get_val_dataloader(cfg.valid, fold)

    # data = next(iter(train_dataloader))
    # import ipdb; ipdb.set_trace()
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

        all_losses = []
        all_labels = []
        all_preds = []
        gc.collect()

        # ==== TRAIN LOOP
        for j, itr in enumerate(progress_bar):
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

            # eval by only center step
            masks = inputs['masks'].bool()
            labels = inputs['inter'][masks].cpu().numpy()
            preds = outputs['inter'][masks].sigmoid().detach().cpu().numpy()

            all_losses.append(loss_dict['inter'].item())
            all_labels.append(labels)
            all_preds.append(preds)

            if torch.isfinite(loss):
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            if model_ema is not None:
                model_ema.update(model)

            if scheduler is not None:
                scheduler.step(i)

            recent_labels = np.concatenate(all_labels[-10:], axis=0).reshape(-1)
            recent_preds = np.concatenate(all_preds[-10:], axis=0).reshape(-1)
            mcc = matthews_corrcoef(recent_labels > 0.3, recent_preds > 0.3)
            pred_mean = recent_preds.mean()
            target_mean = recent_labels.mean()

            avg_loss = np.mean(all_losses[-10:])
            lr = optimizer.param_groups[0]['lr']
            progress_bar.set_description(
                f"step:{i} loss: {avg_loss:.4f} mcc@0.3: {mcc:.3f} target_mean: {target_mean:.3f} pred_mean: {pred_mean:.3f} lr:{lr:.6}")

        all_labels = np.concatenate(all_labels, axis=0).reshape(-1)
        all_preds = np.concatenate(all_preds, axis=0).reshape(-1)
        score = matthews_corrcoef(all_labels > 0.3, all_preds > 0.3)

        if (epoch % cfg.eval_intervals == 0) or (epoch > 30):
            if model_ema is not None:
                val_results = validate(cfg, fold, model_ema.module, test_dataloader)
            else:
                val_results = validate(cfg, fold, model, test_dataloader)
        else:
            val_results = {}
        lr = optimizer.param_groups[0]['lr']

        all_results = {
            'epoch': epoch,
            'lr': lr,
        }
        train_results = {
            'loss': avg_loss,
            'inter_score': score,
            'score': score,
        }
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

        checkpoint = create_checkpoint(
            model, optimizer, epoch, scheduler=scheduler, scaler=scaler, model_ema=model_ema)
        torch.save(checkpoint, f"{cfg.output_dir}/last_fold{fold}.pth")


def validate(cfg, fold, model=None, test_dataloader=None):

    if model is None:
        weight_path = f"{cfg.output_dir}/best_fold{fold}.pth"
        model = get_model(cfg, weight_path)
    model.eval()
    torch.set_grad_enabled(False)

    if test_dataloader is None:
        test_dataloader = get_val_dataloader(cfg.valid, fold)

    df = test_dataloader.dataset.df

    dfs = []

    for i, inputs in enumerate(tqdm(test_dataloader)):
        inputs = batch_to_device(inputs, cfg.device)
        with autocast(cfg.mixed_precision):
            outputs = model(inputs)

        batch_masks = inputs['masks'].bool().cpu().numpy()
        batch_labels = inputs['inter'].cpu().numpy()
        batch_preds = outputs['inter'].sigmoid().detach().cpu().numpy()
        players1 = inputs['nfl_player_id_1'].cpu().numpy()
        players2 = inputs['nfl_player_id_2'].cpu().numpy()

        for b in range(len(batch_masks)):
            masks = batch_masks[b]
            labels = batch_labels[b][masks]
            preds = batch_preds[b][masks]
            game_play = inputs['game_play'][b]
            p1 = players1[b]
            p2 = players2[b]
            df = pd.DataFrame({'contact': labels, 'preds': preds})
            df = df.reset_index().rename(columns={'index': 'step'})
            df['game_play'] = game_play
            df['nfl_player_id_1'] = p1
            df['nfl_player_id_2'] = p2
            dfs.append(df)

    df = pd.concat(dfs).reset_index(drop=True)
    df.to_csv(f'{cfg.output_dir}/val_preds_df_fold{fold}.csv', index=False)

    best_th = search_best_threshold(df.contact.values, df.preds.values)
    score = matthews_corrcoef(df.contact,  df.preds > best_th)
    print(f'total score={score:.3f} best_th={best_th:.3f}')

    results = {
        'score': score,
    }
    return results


def inference(cfg):
    weight_paths = sorted(glob.glob(f'{cfg.model_dir}/best_fold*'))
    model = Models(weight_paths)
    torch.set_grad_enabled(False)

    test_dataloader = get_test_dataloader(cfg.test)

    dfs = []
    for i, inputs in enumerate(tqdm(test_dataloader)):
        inputs = batch_to_device(inputs, cfg.device)
        with autocast(cfg.mixed_precision):
            outputs = model(inputs)

        batch_masks = inputs['masks'].bool().cpu().numpy()
        batch_preds = outputs['inter'].sigmoid().detach().cpu().numpy()
        players1 = inputs['nfl_player_id_1'].cpu().numpy()
        players2 = inputs['nfl_player_id_2'].cpu().numpy()

        for b in range(len(batch_masks)):
            masks = batch_masks[b]
            preds = batch_preds[b][masks]
            game_play = inputs['game_play'][b]
            p1 = players1[b]
            p2 = players2[b]
            df = pd.DataFrame({'preds': preds})
            df = df.reset_index().rename(columns={'index': 'step'})
            df['game_play'] = game_play
            df['nfl_player_id_1'] = p1
            df['nfl_player_id_2'] = p2
            dfs.append(df)

    df = pd.concat(dfs).reset_index(drop=True)
    df.to_csv(f'./data/inter_preds_df.csv', index=False)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="./", type=str)
    parser.add_argument("--device_id", "-d", default="0", type=str)
    parser.add_argument("--start_fold", "-s", default=0, type=int)
    parser.add_argument("--end_fold", "-e", default=4, type=int)
    parser.add_argument("--validate", "-v", action="store_true")
    parser.add_argument("--infer", "-i", action="store_true")
    parser.add_argument("--debug", "-db", action="store_true")
    parser.add_argument("--resume", "-r", action="store_true")
    parser.add_argument("--model_dir", "-m", default="./", type=str)
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

    cfg.model_dir = args.model_dir

    return cfg


if __name__ == "__main__":
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id)

    if args.infer:
        cfg = update_cfg(cfg, args, fold=-1)
        inference(cfg)
    else:
        for fold in range(args.start_fold, args.end_fold):
            cfg = update_cfg(cfg, args, fold)
            if args.validate:
                validate(cfg, fold)
            else:
                train(cfg, fold)

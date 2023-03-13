import warnings
from copy import deepcopy
import argparse
import gc
import os

import albumentations as A
import numpy as np
import pandas as pd
import torch
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

from configs.base import cfg
from models.fpn import YOLOX_FPN_EXT as Net


from optimizers import get_optimizer
from utils.debugger import set_debugger
from utils.common import set_seed, create_checkpoint, resume_checkpoint, batch_to_device,  nms, log_results
from utils.ema import ModelEmaV2
from datasets.base import get_train_dataloader, get_val_dataloader, get_val_dataframe_dataloader
from datasets import video_transforms

warnings.simplefilter(action='ignore', category=FutureWarning)

FPS = 59.94
HEIGHT, WIDTH = 704, 1280
ORIGINAL_HEIGHT, ORIGINAL_WIDTH = 720, 1280
DURATION = 1

cfg = deepcopy(cfg)
cfg.project = 'kaggle-nfl2022'
cfg.exp_name = 'exp048_both_ext_blur_dynamic_normalize_coords_fix_frame_noise'
cfg.output_dir = f'output/{cfg.exp_name}'
cfg.debug = False

cfg.train.df_path = '../input/folds.csv'
cfg.train.label_df_path = '../input/train_labels_with_folds.csv'
cfg.train.helmet_df_path = '../input/train_helmets_with_folds.csv'
cfg.train.tracking_df_path = '../input/train_tracking_with_folds.csv'
cfg.train.image_dir = '../input/train_frames'
cfg.train.duration = DURATION
cfg.train.batch_size = 4
cfg.train.num_workers = 4 if not cfg.debug else 0
cfg.train.down_ratio = 8
cfg.train.roi_size = 'dynamic'
cfg.train.image_size = (HEIGHT, WIDTH)
cfg.train.original_image_size = (ORIGINAL_HEIGHT, ORIGINAL_WIDTH)
cfg.train.transforms = video_transforms.Compose([
    # video_transforms.RandomHorizontalFlip(),
    # video_transforms.RandomRotation(10),
    # video_transforms.ColorJitter(brightness=0.2, contrast=0.1),
    # video_transforms.RandomCrop((int(HEIGHT*0.8), int(WIDTH*0.8))),
    video_transforms.Resize((HEIGHT, WIDTH)),
])
cfg.train.image_transforms = A.Compose([
    # A.RandomCrop(width=WIDTH, height=HEIGHT),
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(p=1.0, rotate_limit=5),
    A.RandomBrightnessContrast(p=1.0),
    A.AdvancedBlur(),
], bbox_params=A.BboxParams(format='pascal_voc'))
cfg.train.enable_frame_noise = True
cfg.train.normalize_coords = True

cfg.valid.df_path = '../input/folds.csv'
cfg.valid.label_df_path = '../input/train_labels_with_folds.csv'
cfg.valid.helmet_df_path = '../input/train_helmets_with_folds.csv'
cfg.valid.tracking_df_path = '../input/train_tracking_with_folds.csv'

cfg.valid.unique_ids_dict_name = 'unique_ids_dict'
cfg.valid.inter_contact_dict_name = 'inter_contact_dict'
cfg.valid.ground_contact_dict_name = 'ground_contact_dict'
cfg.valid.helmet_dict_name = 'helmet_dict'
cfg.valid.track_dict_name = 'track_dict'
cfg.valid.track_dict_by_step_name = 'track_dict_by_step'

cfg.valid.image_dir = '../input/train_frames'
cfg.valid.duration = DURATION
cfg.valid.batch_size = 4
cfg.valid.num_workers = 4 if not cfg.debug else 0
cfg.valid.down_ratio = 8
cfg.valid.roi_size = 'dynamic'
cfg.valid.image_size = (HEIGHT, WIDTH)
cfg.valid.original_image_size = (ORIGINAL_HEIGHT, ORIGINAL_WIDTH)
cfg.valid.transforms = video_transforms.Compose([
    # video_transforms.RandomHorizontalFlip(),
    # video_transforms.RandomRotation(10),
    # video_transforms.ColorJitter(brightness=0.2, contrast=0.1),
    # video_transforms.RandomCrop((int(HEIGHT*0.8), int(WIDTH*0.8))),
    video_transforms.Resize((HEIGHT, WIDTH)),
])
cfg.valid.image_transforms = None
cfg.valid.normalize_coords = True

cfg.model.inter_weight = 0.5
cfg.model.ground_weight = 0.5
cfg.model.pretrained_path = '../input/cut_yolox_m_backbone.pth'
cfg.model.duration = DURATION
cfg.model.down_ratio = 8

# others
cfg.seed = 42
cfg.device = 'cuda'
cfg.lr = 1.0e-4
cfg.wd = 1.0e-4
cfg.min_lr = 5.0e-5
cfg.warmup_lr = 1.0e-5
cfg.warmup_epochs = 3
cfg.warmup = 1
cfg.epochs = 15
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


def train(cfg, fold):
    os.makedirs(str(cfg.output_dir + "/"), exist_ok=True)
    cfg.fold = fold
    mode = 'disabled' if cfg.debug else None
    wandb.init(project=cfg.project,
               name=f'{cfg.exp_name}_fold{fold}', config=cfg, reinit=True, mode=mode)
    set_seed(cfg.seed)
    train_dataloader = get_train_dataloader(cfg.train, fold)
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

        inter_losses = []
        ground_losses = []
        inter_labels = []
        ground_labels = []
        inter_preds = []
        ground_preds = []
        inter_masks = []
        ground_masks = []
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

            inter_losses.append(loss_dict['inter'].item())
            ground_losses.append(loss_dict['ground'].item())
            inter_labels.append(inputs['inter'].cpu().numpy())
            ground_labels.append(inputs['ground'].cpu().numpy())
            inter_preds.append(
                outputs['inter'].sigmoid().detach().cpu().numpy())
            ground_preds.append(
                outputs['ground'].sigmoid().detach().cpu().numpy())
            inter_masks.append(outputs['inter_masks'].cpu().numpy())
            ground_masks.append(outputs['ground_masks'].cpu().numpy())

            if torch.isfinite(loss):
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            if model_ema is not None:
                model_ema.update(model)

            if scheduler is not None:
                scheduler.step(i)

            recent_inter_labels = np.concatenate(
                inter_labels[-10:], axis=0).reshape(-1)
            recent_inter_preds = np.concatenate(
                inter_preds[-10:], axis=0).reshape(-1)
            recent_ground_labels = np.concatenate(
                ground_labels[-10:], axis=0).reshape(-1)
            recent_ground_preds = np.concatenate(
                ground_preds[-10:], axis=0).reshape(-1)
            recent_inter_masks = np.concatenate(
                inter_masks[-10:], axis=0).reshape(-1)
            recent_ground_masks = np.concatenate(
                ground_masks[-10:], axis=0).reshape(-1)
            inter_mcc = matthews_corrcoef(
                recent_inter_labels, recent_inter_preds > 0.3, sample_weight=recent_inter_masks)
            ground_mcc = matthews_corrcoef(
                recent_ground_labels, recent_ground_preds > 0.3, sample_weight=recent_ground_masks)

            avg_inter_loss = np.mean(inter_losses[-10:])
            avg_ground_loss = np.mean(ground_losses[-10:])
            lr = optimizer.param_groups[0]['lr']
            progress_bar.set_description(
                f"step:{i} i_loss: {avg_inter_loss:.4f} g_loss: {avg_ground_loss:.4f} i_mcc@0.3: {inter_mcc:.3f} g_mcc@0.3: {ground_mcc:.3f} lr:{lr:.6}")

        inter_labels = np.concatenate(inter_labels, axis=0).reshape(-1)
        inter_preds = np.concatenate(inter_preds, axis=0).reshape(-1)
        ground_labels = np.concatenate(ground_labels, axis=0).reshape(-1)
        ground_preds = np.concatenate(ground_preds, axis=0).reshape(-1)
        inter_masks = np.concatenate(inter_masks, axis=0).reshape(-1)
        ground_masks = np.concatenate(ground_masks, axis=0).reshape(-1)
        all_labels = np.concatenate([inter_labels, ground_labels])
        all_preds = np.concatenate([inter_preds, ground_preds])
        all_masks = np.concatenate([inter_masks, ground_masks])

        inter_score = matthews_corrcoef(
            inter_labels, inter_preds > 0.3, sample_weight=inter_masks)
        ground_score = matthews_corrcoef(
            ground_labels, ground_preds > 0.3, sample_weight=ground_masks)
        score = matthews_corrcoef(
            all_labels, all_preds > 0.3, sample_weight=all_masks)

        if (epoch % cfg.eval_intervals == 0) or (epoch > 30):
            if model_ema is not None:
                val_results = validate(cfg, fold, model_ema.module)
            else:
                val_results = validate(cfg, fold, model)
        else:
            val_results = {}
        lr = optimizer.param_groups[0]['lr']

        all_results = {
            'epoch': epoch,
            'lr': lr,
        }
        train_results = {
            'inter_loss': avg_inter_loss,
            'ground_loss': avg_ground_loss,
            'inter_score': inter_score,
            'ground_score': ground_score,
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

        checkpoint = create_checkpoint(
            model, optimizer, epoch, scheduler=scheduler, scaler=scaler, model_ema=model_ema)
        torch.save(checkpoint, f"{cfg.output_dir}/last_fold{fold}.pth")


def validate(cfg, fold, model=None, test_dataloader=None):

    if model is None:
        model = get_model(cfg)
        weight_path = f"{cfg.output_dir}/best_fold{fold}.pth"
        model.load_state_dict(torch.load(weight_path)['model'])
        print('load model from', weight_path)
    model.eval()
    torch.set_grad_enabled(False)

    if test_dataloader is None:
        test_dataloader = get_val_dataloader(cfg.valid, fold)

    unique_ids = []
    inter_losses = []
    ground_losses = []
    inter_labels = []
    ground_labels = []
    inter_preds = []
    ground_preds = []
    inter_masks = []
    ground_masks = []

    for i, inputs in enumerate(tqdm(test_dataloader)):
        inputs = batch_to_device(inputs, cfg.device)

        with autocast(cfg.mixed_precision):
            outputs = model(inputs)
            loss_dict = model.get_loss(outputs, inputs)

        inter_losses.append(loss_dict['inter'].item())
        ground_losses.append(loss_dict['ground'].item())
        inter_labels.append(inputs['inter'].cpu().numpy())
        ground_labels.append(inputs['ground'].cpu().numpy())
        inter_preds.append(outputs['inter'].sigmoid().detach().cpu().numpy())
        ground_preds.append(outputs['ground'].sigmoid().detach().cpu().numpy())
        inter_masks.append(outputs['inter_masks'].cpu().numpy())
        ground_masks.append(outputs['ground_masks'].cpu().numpy())
        unique_ids.append(inputs['unique_ids'].cpu().numpy())

    unique_ids = np.concatenate(unique_ids, axis=0)
    inter_labels = np.concatenate(inter_labels, axis=0)
    inter_preds = np.concatenate(inter_preds, axis=0)
    inter_masks = np.concatenate(inter_masks, axis=0)
    ground_labels = np.concatenate(ground_labels, axis=0)
    ground_preds = np.concatenate(ground_preds, axis=0)
    ground_masks = np.concatenate(ground_masks, axis=0)
    epoch = cfg.curr_epoch
    np.save(
        f"{cfg.output_dir}/val_inter_labels_fold{cfg.fold}_epoch{epoch}.npy", inter_labels)
    np.save(
        f"{cfg.output_dir}/val_inter_preds_fold{cfg.fold}_epoch{epoch}.npy", inter_preds)
    np.save(f"{cfg.output_dir}/val_unique_ids_fold{cfg.fold}.npy", unique_ids)

    inter_labels = inter_labels.reshape(-1)
    inter_preds = inter_preds.reshape(-1)
    inter_masks = inter_masks.reshape(-1)
    inter_score02 = matthews_corrcoef(
        inter_labels, inter_preds > 0.2, sample_weight=inter_masks)
    inter_score03 = matthews_corrcoef(
        inter_labels, inter_preds > 0.3, sample_weight=inter_masks)
    inter_score04 = matthews_corrcoef(
        inter_labels, inter_preds > 0.4, sample_weight=inter_masks)

    ground_labels = ground_labels.reshape(-1)
    ground_preds = ground_preds.reshape(-1)
    ground_masks = ground_masks.reshape(-1)
    ground_score02 = matthews_corrcoef(
        ground_labels, ground_preds > 0.2, sample_weight=ground_masks)
    ground_score03 = matthews_corrcoef(
        ground_labels, ground_preds > 0.3, sample_weight=ground_masks)
    ground_score04 = matthews_corrcoef(
        ground_labels, ground_preds > 0.4, sample_weight=ground_masks)

    all_labels = np.concatenate([inter_labels, ground_labels])
    all_preds = np.concatenate([inter_preds, ground_preds])
    all_masks = np.concatenate([inter_masks, ground_masks])

    score = matthews_corrcoef(
        all_labels, all_preds > 0.3, sample_weight=all_masks)

    results = {
        'score': score,
        'inter_score02': inter_score02,
        'inter_score03': inter_score03,
        'inter_score04': inter_score04,
        'ground_score02': ground_score02,
        'ground_score03': ground_score03,
        'ground_score04': ground_score04,
    }
    return results


def get_preds_and_masks(inputs, outputs, train_pairs_gb):
    dfs = []
    num_samples = len(inputs['frame'])

    batch_game_plays = inputs['game_play']
    batch_steps = inputs['step'].cpu().numpy().astype(int)
    batch_frames = inputs['frame'].cpu().numpy().astype(int)
    batch_unique_ids = inputs['unique_ids'].cpu().numpy().astype(int)
    batch_ground_preds = outputs['ground'].sigmoid().cpu().numpy()
    batch_ground_masks = outputs['ground_masks'].cpu().numpy()
    batch_inter_preds = outputs['inter'].sigmoid().cpu().numpy()
    batch_inter_masks = outputs['inter_masks'].cpu().numpy()

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

        preds = []
        masks = []
        for id1, id2 in pairs:
            idx1 = unique_ids.index(id1)
            if (id2 == -1) | (id2 == 0):
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
        df = pd.DataFrame(data={'preds': preds, "masks": masks})
        df['game_play'] = game_play
        df['step'] = step
        df['frame'] = frame
        df["nfl_player_id_1"] = pairs[:, 0]
        df["nfl_player_id_2"] = pairs[:, 1]
        dfs.append(df)
    return pd.concat(dfs).reset_index(drop=True)


def full_validate(cfg, fold, model=None, test_dataloader=None):
    if model is None:
        weight_path = f"{cfg.output_dir}/best_fold{fold}.pth"
        model = get_model(cfg, weight_path)
    model.eval()
    torch.set_grad_enabled(False)

    if test_dataloader is None:
        test_dataloader = get_val_dataframe_dataloader(cfg.valid, fold)

    pairs_gb = test_dataloader.dataset.pairs_gb

    dfs = []
    for inputs in tqdm(test_dataloader):
        inputs = batch_to_device(inputs, cfg.device, cfg.mixed_precision)
        with torch.no_grad() and autocast(cfg.mixed_precision):
            outputs = model(inputs)
        df = get_preds_and_masks(inputs, outputs, pairs_gb)
        dfs.append(df)
    pred_df = pd.concat(dfs).reset_index(drop=True)
    pred_df.to_csv(f'{cfg.output_dir}/val_preds_df_fold{fold}.csv', index=False)
    return evaluate_pred_df(test_dataloader.dataset.df, pred_df)


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
    for fold in range(args.start_fold, args.end_fold):
        cfg = update_cfg(cfg, args, fold)
        if args.validate:
            full_validate(cfg, fold)
            # validate(cfg, fold)
        elif args.infer:
            inference(cfg)
        else:
            train(cfg, fold)

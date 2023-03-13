from copy import deepcopy
from types import SimpleNamespace

cfg = SimpleNamespace(**{})

cfg.train = SimpleNamespace(**{})
cfg.train.roi_size = None
cfg.train.image_transforms = None
cfg.train.unique_ids_dict_name = 'unique_ids_dict'
cfg.train.inter_contact_dict_name = 'inter_contact_dict'
cfg.train.ground_contact_dict_name = 'ground_contact_dict'
cfg.train.track_dict_name = 'track_dict'
cfg.train.helmet_dict_name = 'helmet_dict'
cfg.train.enable_frame_noise = False
cfg.train.enable_hflip = False
cfg.train.normalize_coords = False
cfg.train.fix_coords_scale = False
cfg.train.include_ground = False
cfg.train.ground_only = False
cfg.train.image_feature_dir = None
cfg.train.holdout_fold = None
cfg.train.num_track_features = 10
cfg.train.data_dir = '../input/preprocessed_data'
cfg.train.image_feature_dir2 = None

cfg.valid = SimpleNamespace(**{})
cfg.valid.roi_size = None
cfg.valid.image_transforms = None
cfg.valid.sample_df_path = '../input/train_sample_df.csv'
cfg.valid.sample_df_by_step_path = '../input/train_sample_df_by_step.csv'
cfg.valid.use_all_frames = False
cfg.valid.unique_ids_dict_name = 'unique_ids_dict'
cfg.valid.inter_contact_dict_name = 'inter_contact_dict'
cfg.valid.ground_contact_dict_name = 'ground_contact_dict'
cfg.valid.track_dict_name = 'track_dict'
cfg.valid.track_dict_by_step_name = 'track_dict_by_step'
cfg.valid.helmet_dict_name = 'helmet_dict'
cfg.valid.enable_frame_noise = False
cfg.valid.enable_hflip = False
cfg.valid.normalize_coords = False
cfg.valid.fix_coords_scale = False
cfg.valid.include_ground = False
cfg.valid.ground_only = False
cfg.valid.image_feature_dir = None
cfg.valid.holdout_fold = None
cfg.valid.num_track_features = 10
cfg.valid.data_dir = '../input/preprocessed_data'
cfg.valid.image_feature_dir2 = None

cfg.test = SimpleNamespace(**{})
cfg.test.roi_size = None
cfg.test.image_transforms = None
cfg.test.df_path = '../input/sample_submission.csv'
cfg.test.helmet_df_path = '../input/test_baseline_helmets.csv'
cfg.test.tracking_df_path = '../input/test_player_tracking.csv'
cfg.test.enable_frame_noise = False
cfg.test.enable_hflip = False
cfg.test.normalize_coords = False
cfg.test.fix_coords_scale = False
cfg.test.include_ground = False
cfg.test.ground_only = False
cfg.test.image_feature_dir = None
cfg.test.image_feature_dir2 = None

cfg.model = SimpleNamespace(**{})
cfg.model.resume_exp = None
cfg.model.roi_size = 3
cfg.model.concat_views = True
cfg.model.dist_activation = None
cfg.model.pretrained = True
cfg.model.num_channels = 192
cfg.model.stem_channels = 48
cfg.model.context_channels = 768

cfg.model.drop_rate = 0.1
cfg.model.dist_th = 100000.0
cfg.model.model_name = 'yolox_m'
cfg.model.mix_beta = 0.0
cfg.model.return_feats = False
cfg.model.label_smoothing = 0.0
cfg.model.num_track_features = 10
cfg.model.num_pair_features = 11
cfg.model.add_self_contact_as_False = False
cfg.resume = False

cfg.optimizer = 'adamw'

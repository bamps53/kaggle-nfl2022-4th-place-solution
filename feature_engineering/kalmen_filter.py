import math
import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.general import reduce_dtype
from utils.nfl import read_csv_with_cache


class KalmanFilter:
    def __init__(self,
                 num_sensor=2,
                 timestep=0.02,
                 acc_band=1,  # change of xx pix/s during timestep. (unit can be changed)
                 sensor_errors=[5, 5],  # measurement error(pix)
                 initial_state=[0, 0, 0, 0],  # x pix, vx pix/s, y pix, vy pix/s
                 clip_dist=20,
                 outlier_dist=100,

                 ):
        if len(sensor_errors) != num_sensor:
            raise Exception("len(sensor_errors) must be same as the num_sensor")
        self.state_dim = 4
        self.num_sensor = num_sensor
        self.timestep = timestep
        self.acc_band = acc_band
        self.sensor_errors = sensor_errors
        self.initial_state = np.array(initial_state)  # .reshape(2,1)
        self.base_clip_dist = clip_dist
        self.base_outlier_dist = outlier_dist
        self.initial_cov_ratio = 100
        self.current_cov_ratio = self.initial_cov_ratio
        self.initialize_matrix()

    def initialize_matrix(self):
        # Motion Model
        self.A = np.array([[1, self.timestep, 0, 0],  # x, vx, y, vy
                           [0, 1, 0, 0],
                           [0, 0, 1, self.timestep],
                           [0, 0, 0, 1]])
        # Jacobian of Observation Model
        self.C = np.array([[1, 0, 0, 0],
                           [0, 0, 1, 0]])  # x and y
        self.base_Q = self.get_cov_mat_self()  # Covariance Matrix for motion
        self.base_R = self.get_cov_mat_sensor()  # Covariance Matrix for observation
        self.Q = self.base_Q.copy() * self.initial_cov_ratio
        self.R = self.base_R.copy() * self.initial_cov_ratio
        self.outlier_dist = self.base_outlier_dist * self.initial_cov_ratio
        self.clip_dist = self.base_clip_dist * self.initial_cov_ratio
        self.x = self.initial_state
        self.P = np.eye(self.state_dim)

    def get_cov_mat_sensor(self):
        R = np.diag([e**2 for e in self.sensor_errors])
        return R

    def get_cov_mat_self(self):
        #Q = np.diag([self.acc_band**2, self.acc_band**2])
        Q = np.diag([0, self.acc_band**2, 0, self.acc_band**2])
        return Q

    def update_cov(self):
        reduction_ratio = 2
        self.current_cov_ratio = self.current_cov_ratio / reduction_ratio
        if self.current_cov_ratio > 1:
            self.Q = self.Q / reduction_ratio
            self.R = self.R / reduction_ratio
            self.outlier_dist = self.outlier_dist / reduction_ratio
            self.clip_dist = self.clip_dist / reduction_ratio
        else:
            self.Q = self.base_Q
            self.R = self.base_R
            self.outlier_dist = self.base_outlier_dist
            self.clip_dist = self.base_clip_dist

    def prior_estimate(self):
        """
        prior prediction of state
        and covariant matrix of prior prediction
        """
        x_prior = np.matmul(self.A, self.x)  # 現在のstateと運動モデルから次のstateを予測
        P_prior = np.matmul(self.A, np.matmul(self.P, self.A.T)) + self.Q  # motionの共分散にここまでの状態量の共分散を上乗せ、事前予測の共分散を得る
        return x_prior, P_prior

    def update(self, x_prior, P_prior, sensor_vals):
        """
        update
        - kalman gain
        - state
        - cov matrix
        """
        S = np.matmul(self.C, np.matmul(P_prior, self.C.T)) + self.R
        K = np.matmul(np.matmul(P_prior, self.C.T), np.linalg.inv(S))
        x = x_prior + np.matmul(K, sensor_vals - np.matmul(self.C, x_prior))
        P = np.matmul((np.eye(self.state_dim) - np.matmul(K, self.C)), P_prior)
        self.update_cov()
        return x, P

    def clip_by_dist(self, pos_prior, pos):
        delta = pos - pos_prior
        dist = math.sqrt(delta[0]**2 + delta[1]**2)
        # dist = np.sqrt((delta**2).sum())
        is_outlier = dist > self.outlier_dist
        if dist > self.clip_dist:
            pos = delta * min(dist, self.clip_dist) / (dist + 1e-12) + pos_prior
        if is_outlier:
            self.outlier_dist = self.outlier_dist + self.base_clip_dist
            self.clip_dist = self.clip_dist + self.base_clip_dist
        else:
            self.outlier_dist = max(self.outlier_dist - self.clip_dist, self.base_outlier_dist)
            self.clip_dist = max(self.clip_dist - self.clip_dist, self.base_clip_dist)
        return pos, is_outlier

    def __call__(self, sensor_vals):
        x_prior, P_prior = self.prior_estimate()
        if np.isnan(sensor_vals[0]):
            self.x, self.P = x_prior, P_prior
        else:
            sensor_vals_clip, is_outlier = self.clip_by_dist(x_prior[::2], sensor_vals)
            if is_outlier:
                self.x, self.P = x_prior, P_prior
            else:
                self.x, self.P = self.update(x_prior, P_prior, sensor_vals_clip)
        return self.x

    def smooth_step(self, p_prior, p, x_prior, x):
        C = np.matmul(p, self.A.T).dot(np.linalg.inv(p_prior))
        x_smooth = x + C.dot(self.x - x_prior)  # .squeeze()
        p_smooth = p + C.dot(self.P - p_prior).dot(C.T)
        return x_smooth, p_smooth

    def smoother(self, sequence_sensor_vals):
        x_priors = [self.x]  # initial_value
        p_priors = [self.P]
        xs = [self.x]
        ps = [self.P]
        is_outlier_predicted = [0]
        for sensor_vals in sequence_sensor_vals[:]:  # [1:]
            x_prior, P_prior = self.prior_estimate()
            if np.isnan(sensor_vals[0]):
                self.x, self.P = x_prior, P_prior
                is_outlier_predicted += [1]
            else:
                sensor_vals_clip, is_outlier = self.clip_by_dist(x_prior[::2], sensor_vals)
                if is_outlier:
                    self.x, self.P = x_prior, P_prior
                    is_outlier_predicted += [1]
                else:
                    self.x, self.P = self.update(x_prior, P_prior, sensor_vals_clip)
                    is_outlier_predicted += [0]
            # self.x, self.P = self.update(x_prior, P_prior, sensor_vals)
            x_priors.append(x_prior)
            p_priors.append(P_prior)
            xs.append(self.x)
            ps.append(self.P)
        xs_smooth = [self.x]
        ps_smooth = [self.P]
        for i in reversed(range(len(sequence_sensor_vals))):  # -1
            self.x, self.P = self.smooth_step(p_priors[i + 1], ps[i], x_priors[i + 1], xs[i])
            xs_smooth.append(self.x)
            ps_smooth.append(self.P)
        xs_smooth = xs_smooth[::-1]
        ps_smooth = ps_smooth[::-1]
        return xs_smooth[1:], is_outlier_predicted[1:]


def apply_smoother_split(df_single_player,
                         num_split_threshold_no_detect=20,  # outlierも線引きしたほうがいいかもな。
                         draw_output=False):
    xy_sequence = df_single_player[["cx", "cy"]].values
    frame_no_detected = df_single_player["frame"].values
    frame_no_to_xy = {fn: xy for fn, xy in zip(frame_no_detected, xy_sequence)}
    frame_no_min = frame_no_detected.min()
    frame_no_max = frame_no_detected.max()
    frame_range_all = np.arange(frame_no_min, frame_no_max + 1)
    xy_sequence_all = np.array([frame_no_to_xy[fn] if fn in frame_no_detected else [
                               np.nan, np.nan] for fn in frame_range_all])
    split_frame_range = []
    split_xy_sequence = []
    first_time_detected = 0
    last_time_detected = 0
    unseen_duration = 0
    missed = False
    for idx, f_no in enumerate(frame_range_all):
        if f_no in frame_no_detected:
            last_time_detected = idx
            if missed:
                first_time_detected = idx
            missed = False
            unseen_duration = 0
        else:
            unseen_duration += 1
            if missed:
                continue
            if unseen_duration >= num_split_threshold_no_detect:
                missed = True
                split_xy_sequence += [xy_sequence_all[first_time_detected:last_time_detected + 1]]
                split_frame_range += [frame_range_all[first_time_detected:last_time_detected + 1]]
            continue
    split_xy_sequence += [xy_sequence_all[first_time_detected:]]
    split_frame_range += [frame_range_all[first_time_detected:]]
    split_smoothed_pos = []
    split_smoothed_vel = []
    split_outlier_predicted = []
    for xy_sequence, frame_range in zip(split_xy_sequence, split_frame_range):
        initial_xyz = xy_sequence[~np.isnan(xy_sequence[:, 0])][:3].mean(axis=0)
        kf = KalmanFilter(num_sensor=2,
                          timestep=1 / 60,
                          acc_band=25,  # change of N pix/s during timestep
                          sensor_errors=[5, 5],  # measurement error(pix)
                          initial_state=[initial_xyz[0], 0, initial_xyz[1], 0],  # x pix, vx pix/s, y pix, vy pix/s
                          clip_dist=20,
                          outlier_dist=100,
                          )

        # kalman smoother
        kf.initialize_matrix()
        smoothed, outlier_predicted = kf.smoother(xy_sequence)
        split_smoothed_pos.append(np.array(smoothed)[:, [0, 2]])
        split_smoothed_vel.append(np.array(smoothed)[:, [1, 3]])
        split_outlier_predicted.append(np.array(outlier_predicted))

    smoothed_pos = np.concatenate(split_smoothed_pos, axis=0)
    smoothed_vel = np.concatenate(split_smoothed_vel, axis=0)
    outlier_predicted = np.concatenate(split_outlier_predicted, axis=0)
    frame_range = np.concatenate(split_frame_range, axis=0)
    xy_sequence = np.concatenate(split_xy_sequence, axis=0)

    if draw_output:
        # plt.figure(figsize=[5,15])
        figs, axes = plt.subplots(1, 2, figsize=[15, 3.5])
        axes[0].scatter(xy_sequence[:, 0], xy_sequence[:, 1], c=np.arange(
            len(xy_sequence)), s=35, vmin=0, vmax=len(xy_sequence))
        axes[0].plot(xy_sequence[:, 0], xy_sequence[:, 1])
        axes[0].invert_yaxis()
        axes[0].grid()
        axes[0].set_title(f" no postprocess")

        axes[1].scatter(smoothed_pos[:, 0], smoothed_pos[:, 1], c=np.arange(
            len(smoothed_pos)), s=35, vmin=0, vmax=len(smoothed_pos))
        axes[1].plot(smoothed_pos[:, 0], smoothed_pos[:, 1])
        axes[1].invert_yaxis()
        axes[1].grid()
        axes[1].set_title(f" kalman smooth")
        plt.show()
    return smoothed_pos, smoothed_vel, outlier_predicted, frame_range


def run_smoother_on_helmets(phase="train", path=None, num_split_threshold_no_detect=10):
    if path is not None:
        if os.path.exists(path):
            df_smoothed = pd.read_csv(path)
            if 'Unnamed: 0' in df_smoothed.columns:
                df_smoothed = df_smoothed.drop(columns=['Unnamed: 0'])
            return df_smoothed

    draw_output = False

    df_helmet = pd.read_csv(f"../input/nfl-player-contact-detection/{phase}_baseline_helmets.csv")
    df_helmet["cx"] = df_helmet["left"] + df_helmet["width"] / 2
    df_helmet["cy"] = df_helmet["top"] + df_helmet["height"] / 2

    df_smoothed = []
    print("start kalman smooth.")
    for game_play in tqdm(df_helmet["game_play"].unique()):
        #print("GamePlay: ", game_play)
        # print(f"\r GamePlay {game_play}",end="")
        for view in ["Sideline", "Endzone"]:
            df_gp = df_helmet.query("game_play == @game_play and view == @view")
            smoothed_positions = []
            df_smoothed_velocities = []
            df_smoothed_gp = []
            vel_column_names = []
            for p_idx, nfl_player_id in enumerate(df_gp["nfl_player_id"].unique()):
                df_single_player = df_gp[df_gp["nfl_player_id"] == nfl_player_id]
                smoothed_pos, smoothed_vel, outlier_predicted, frame_range = apply_smoother_split(
                    df_single_player, num_split_threshold_no_detect=num_split_threshold_no_detect, draw_output=draw_output)

                df_vel_player = pd.DataFrame(smoothed_vel, columns=[
                                             f"bbox_smooth_velx", f"bbox_smooth_vely"], index=frame_range)
                df_vel_player[["bbox_smooth_x", "bbox_smooth_y"]] = smoothed_pos
                df_vel_player["bbox_smooth_outlier"] = outlier_predicted
                df_vel_player.loc[frame_range[:-1], [f"bbox_smooth_accx",
                                                     f"bbox_smooth_accy"]] = smoothed_vel[1:] - smoothed_vel[:-1]
                df_vel_player["nfl_player_id"] = nfl_player_id
                df_smoothed_gp += [df_vel_player]

            df_smoothed_gp = pd.concat(df_smoothed_gp, axis=0).reset_index().rename(columns={"index": "frame"})
            df_smoothed_gp["view"] = view
            df_smoothed_gp["game_play"] = game_play

            df_smoothed.append(df_smoothed_gp)
    df_smoothed = pd.concat(df_smoothed, axis=0).reset_index(drop=True)
    df_smoothed[["bbox_smooth_x", "bbox_smooth_y"]] = df_smoothed[["bbox_smooth_x", "bbox_smooth_y"]].astype(np.float32)
    df_smoothed[["bbox_smooth_velx", "bbox_smooth_vely"]] = df_smoothed[[
        "bbox_smooth_velx", "bbox_smooth_vely"]].astype(np.float32)
    df_smoothed[["bbox_smooth_accx", "bbox_smooth_accy"]] = df_smoothed[[
        "bbox_smooth_accx", "bbox_smooth_accy"]].astype(np.float32)

    if path is not None:
        df_smoothed.to_csv(path, index=False)
    return df_smoothed


def merge_helmet_smooth(df, helmet, meta):
    start_times = meta[["game_play", "start_time"]].drop_duplicates()
    start_times["start_time"] = pd.to_datetime(start_times["start_time"])

    helmet = pd.merge(helmet,
                      start_times,
                      on="game_play",
                      how="left")

    fps = 59.94
    helmet["datetime"] = helmet["start_time"] + pd.to_timedelta(helmet["frame"] * (1 / fps), unit="s")
    helmet["datetime"] = pd.to_datetime(helmet["datetime"], utc=True)
    helmet["datetime_ngs"] = pd.DatetimeIndex(helmet["datetime"] + pd.to_timedelta(50, "ms")).floor("100ms").values
    helmet["datetime_ngs"] = pd.to_datetime(helmet["datetime_ngs"], utc=True)
    # df["datetime_ngs"] = pd.to_datetime(df["datetime"], utc=True)

    feature_cols = ["bbox_smooth_velx", "bbox_smooth_vely",
                    "bbox_smooth_x", "bbox_smooth_y",
                    "bbox_smooth_outlier",
                    "bbox_smooth_accx", "bbox_smooth_accy"]

    helmet_agg = helmet.groupby(["datetime_ngs", "nfl_player_id", "view"]).agg({
        c: "mean" for c in feature_cols}).reset_index()

    for view in ["Sideline", "Endzone"]:
        helmet_ = helmet_agg[helmet_agg["view"] == view].drop("view", axis=1)

        for postfix in ["_1", "_2"]:
            column_renames = {c: f"{c}_{view}{postfix}" for c in feature_cols}
            column_renames["nfl_player_id"] = f"nfl_player_id{postfix}"
            df = pd.merge(
                df,
                helmet_.rename(columns=column_renames),
                on=["datetime_ngs", f"nfl_player_id{postfix}"],
                how="left"
            )

    try:
        del df["datetime"]
    except BaseException:
        pass
    return reduce_dtype(df)


def add_bbox_features_smooth(df):
    """
    bboxの外れ値をsmoothしたボックスとの位置差分で求める
    """
    for view in ["Sideline", "Endzone"]:
        for side in [1, 2]:
            x = df[f"bbox_smooth_x_{view}_{side}"] - df[f'bbox_center_x_{view}_{side}']
            y = df[f"bbox_smooth_y_{view}_{side}"] - df[f'bbox_center_y_{view}_{side}']
            df[f"bbox_smooth_y_{view}_{side}_dist"] = x**2 + y**2
    return df


def expand_helmet_smooth(cfg, df, phase="train"):
    if phase == "train":
        helmet = run_smoother_on_helmets(phase="train",
                                         path=None,
                                         num_split_threshold_no_detect=10)
    else:
        helmet = run_smoother_on_helmets(phase="test",
                                         path=None,
                                         num_split_threshold_no_detect=10)

    meta = read_csv_with_cache(f"{phase}_video_metadata.csv", cfg.INPUT, cfg.CACHE)
    df = merge_helmet_smooth(df, helmet, meta)
    return df

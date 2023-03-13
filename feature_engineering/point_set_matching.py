"""
points set matching.
run 1 play at once.
"""
import os
import time

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from utils.general import reduce_dtype


def visualize(list_targets, list_sources):
    for t, s in zip(list_targets[::50], list_sources[::50]):
        plt.figure(figsize=(8, 5))
        words = np.arange(len(t))
        plt.scatter(t[:, 0], t[:, 1], c="blue", s=50, alpha=0.5)  # , c=np.arange(len(t)))#, cmap="gray")#color="blue")
        for number, [x, y] in enumerate(t):
            plt.text(x, y + .8, number, fontsize=9, c="blue")
            # plt.annotate(number, (x, y), c="blue")
        plt.scatter(s[:, 0], s[:, 1], c="red", s=50, alpha=0.5)  # , c=np.arange(len(s)))#, cmap="gray")#color="red")
        for number, [x, y] in enumerate(s):
            plt.text(x, y - 1.75, number, fontsize=9, c="red")
            # plt.annotate(number, (x, y), c="red")
        # plt.text(s[:,0]+.3, s[:,1]+.3, words, fontsize=9)
        plt.grid()
        plt.title("tracking X-Y  VS  transformed image coord")
        plt.show()


def prepare_matching_dataframe(game_play, tr_tracking, helmets, meta, view="Sideline", fps=59.94, only_center_of_step=True):
    tr_tracking = tr_tracking.query("game_play == @game_play").copy()
    gp_helms = helmets.query("game_play == @game_play").copy()

    start_time = meta.query("game_play == @game_play and view == @view")["start_time"].values[0]

    gp_helms["datetime"] = pd.to_timedelta(gp_helms["frame"] * (1 / fps), unit="s") + start_time
    gp_helms["datetime"] = pd.to_datetime(gp_helms["datetime"], utc=True)
    gp_helms["datetime_ngs"] = pd.DatetimeIndex(gp_helms["datetime"] + pd.to_timedelta(50, "ms")).floor("100ms").values
    gp_helms["datetime_ngs"] = pd.to_datetime(gp_helms["datetime_ngs"], utc=True)
    gp_helms["delta_from_round_val"] = (gp_helms["datetime"] - gp_helms["datetime_ngs"]).dt.total_seconds()

    tr_tracking["datetime_ngs"] = pd.to_datetime(tr_tracking["datetime"], utc=True)
    gp_helms = gp_helms.merge(
        tr_tracking[["datetime_ngs", "step", "x_position", "y_position", "nfl_player_id"]],
        left_on=["datetime_ngs", "nfl_player_id"],
        right_on=["datetime_ngs", "nfl_player_id"],
        how="left",
    )
    gp_helms["center_frame_of_step"] = np.abs(gp_helms["delta_from_round_val"])
    gp_helms["center_frame_of_step"] = gp_helms["center_frame_of_step"].values == gp_helms.groupby(
        "datetime_ngs")["center_frame_of_step"].transform("min").values
    # 複数minimumが存在するケースもあるので気を付ける(あとでグループ平均とるなど)
    if only_center_of_step:
        gp_helms = gp_helms[gp_helms["center_frame_of_step"]].drop(columns=["center_frame_of_step"])
    # 追加した。ok?
    gp_helms = gp_helms[gp_helms["view"] == view]

    return gp_helms


def batch_points2points_fitting(targets, sources, padded, num_iter=6, l2_reg=0.1, init_rot=0):

    def get_transmatrix(kx, ky, rz, tx, ty):
        """
        k : log(zoom ratio).
        rz : rotation.
        tx : x offset.
        ty : z offset
        shape [batch]

        returns:
            transmatrix with shape [batch, 3, 3]
        """
        exp_kx = tf.math.exp(kx)
        exp_ky = tf.math.exp(ky)

        sin = tf.math.sin(rz)
        cos = tf.math.cos(rz)
        mat = tf.stack([[exp_kx * cos, -exp_ky * sin, 1 * tx],
                        [exp_kx * sin, exp_ky * cos, 1 * ty],
                        [tf.zeros_like(kx), tf.zeros_like(kx), tf.ones_like(kx)]])
        mat = tf.transpose(mat, [2, 0, 1])
        return mat

    def transform_points(transmatrix, points):
        x, y = tf.split(points, 2, axis=-1)
        xyones = tf.concat([x, y, tf.ones_like(x)], axis=-1)
        trans_points = tf.matmul(xyones, transmatrix, transpose_b=True)[..., :2]
        return trans_points

    def get_derivative_at(kx, ky, rz, tx, ty, points):
        dev = 1e-5
        original = transform_points(get_transmatrix(kx, ky, rz, tx, ty), points)
        dxy_dkx = (transform_points(get_transmatrix(kx + dev, ky, rz, tx, ty), points) - original) / dev
        dxy_dky = (transform_points(get_transmatrix(kx, ky + dev, rz, tx, ty), points) - original) / dev
        dxy_drz = (transform_points(get_transmatrix(kx, ky, rz + dev, tx, ty), points) - original) / dev
        dxy_dtx = (transform_points(get_transmatrix(kx, ky, rz, tx + dev, ty), points) - original) / dev
        dxy_dty = (transform_points(get_transmatrix(kx, ky, rz, tx, ty + dev), points) - original) / dev
        return original, dxy_dkx, dxy_dky, dxy_drz, dxy_dtx, dxy_dty

    # initial_values
    batch, num_points = tf.unstack(tf.shape(targets))[:2]
    kx = 0.0 * tf.ones((batch), tf.float32)  # zoom ratio = exp(k)
    ky = 0.0 * tf.ones((batch), tf.float32)  # zoom ratio = exp(k)
    rz = init_rot * tf.ones((batch), tf.float32)
    tx = 0.0 * tf.ones((batch), tf.float32)
    ty = 0.0 * tf.ones((batch), tf.float32)

    source_origin = sources  # tf.stop_gradient(sources)
    for i in range(num_iter):
        currents, dxy_dkx, dxy_dky, dxy_rz, dxy_dtx, dxy_dty = get_derivative_at(kx, ky, rz, tx, ty, source_origin)
        b = tf.reshape((targets - currents) * padded, [batch, num_points * 2, 1])  # xy flatten
        a = tf.stack([dxy_dkx, dxy_dky, dxy_rz, dxy_dtx, dxy_dty], axis=-1)
        a = a * padded[..., tf.newaxis]
        a = tf.reshape(a, [batch, num_points * 2, 5])
        updates = tf.linalg.lstsq(a, b, l2_regularizer=l2_reg, fast=True)  # batch, 5, 1
        kx = tf.maximum(kx + updates[:, 0, 0], -10)
        ky = tf.maximum(ky + updates[:, 1, 0], -10)
        rz = rz + updates[:, 2, 0]
        tx = tx + updates[:, 3, 0]
        ty = ty + updates[:, 4, 0]

    trans_matrix = get_transmatrix(kx, ky, rz, tx, ty)
    trans_sources = transform_points(trans_matrix, sources)
    try:
        trans_targets = transform_points(tf.linalg.inv(trans_matrix), targets)
    except BaseException:
        trans_targets = targets
    residuals_points = tf.reduce_sum(((targets - trans_sources) * padded)**2, axis=2)
    residuals = tf.reduce_sum(residuals_points, axis=1)
    return trans_sources, trans_targets, trans_matrix, kx, ky, rz, tx, ty, residuals_points, residuals


def batch_points2points_fitting_highdof(targets, sources, padded, num_iter=6, l2_reg=0.1, init_rot=0):

    def get_transmatrix(kx, ky, rz, tx, ty, trape):
        """
        k : log(zoom ratio).
        rz : rotation.
        tx : x offset.
        ty : z offset
        shape [batch]

        returns:
            transmatrix with shape [batch, 3, 3]
        """
        exp_kx = tf.math.exp(kx)
        exp_ky = tf.math.exp(ky)
        trape = trape  # trapezoid
        sin = tf.math.sin(rz)
        cos = tf.math.cos(rz)
        mat = tf.stack([[exp_kx * cos, trape * exp_kx * cos - exp_ky * sin, 1 * tx],
                        [exp_kx * sin, trape * exp_kx * sin + exp_ky * cos, 1 * ty],
                        [tf.zeros_like(kx), tf.zeros_like(kx), tf.ones_like(kx)]])
        mat = tf.transpose(mat, [2, 0, 1])
        return mat

    def transform_points(transmatrix, points):
        x, y = tf.split(points, 2, axis=-1)
        xyones = tf.concat([x, y, tf.ones_like(x)], axis=-1)
        trans_points = tf.matmul(xyones, transmatrix, transpose_b=True)[..., :2]
        return trans_points

    def get_derivative_at(kx, ky, rz, tx, ty, trape, points):
        dev = 1e-5
        original = transform_points(get_transmatrix(kx, ky, rz, tx, ty, trape), points)
        dxy_dkx = (transform_points(get_transmatrix(kx + dev, ky, rz, tx, ty, trape), points) - original) / dev
        dxy_dky = (transform_points(get_transmatrix(kx, ky + dev, rz, tx, ty, trape), points) - original) / dev
        dxy_drz = (transform_points(get_transmatrix(kx, ky, rz + dev, tx, ty, trape), points) - original) / dev
        dxy_dtx = (transform_points(get_transmatrix(kx, ky, rz, tx + dev, ty, trape), points) - original) / dev
        dxy_dty = (transform_points(get_transmatrix(kx, ky, rz, tx, ty + dev, trape), points) - original) / dev
        dxy_dttrape = (transform_points(get_transmatrix(kx, ky, rz, tx, ty, trape + dev), points) - original) / dev
        return original, dxy_dkx, dxy_dky, dxy_drz, dxy_dtx, dxy_dty, dxy_dttrape

    # initial_values
    batch, num_points = tf.unstack(tf.shape(targets))[:2]
    kx = 0.0 * tf.ones((batch), tf.float32)  # zoom ratio = exp(k)
    ky = 0.0 * tf.ones((batch), tf.float32)  # zoom ratio = exp(k)
    rz = init_rot * tf.ones((batch), tf.float32)
    tx = 0.0 * tf.ones((batch), tf.float32)
    ty = 0.0 * tf.ones((batch), tf.float32)
    trape = 0.0 * tf.ones((batch), tf.float32)

    source_origin = sources  # tf.stop_gradient(sources)
    for i in range(num_iter):
        currents, dxy_dkx, dxy_dky, dxy_rz, dxy_dtx, dxy_dty, dxy_dttrape = get_derivative_at(
            kx, ky, rz, tx, ty, trape, source_origin)
        b = tf.reshape((targets - currents) * padded, [batch, num_points * 2, 1])  # xy flatten
        a = tf.stack([dxy_dkx, dxy_dky, dxy_rz, dxy_dtx, dxy_dty, dxy_dttrape], axis=-1)
        a = a * padded[..., tf.newaxis]
        a = tf.reshape(a, [batch, num_points * 2, 6])
        updates = tf.linalg.lstsq(a, b, l2_regularizer=l2_reg, fast=True)  # batch, 6, 1
        kx = tf.maximum(kx + updates[:, 0, 0], -10)
        ky = tf.maximum(ky + updates[:, 1, 0], -10)
        rz = rz + updates[:, 2, 0]
        tx = tx + updates[:, 3, 0]
        ty = ty + updates[:, 4, 0]
        trape = tf.clip_by_value(trape + updates[:, 5, 0], -0.25, 0.25)

    trans_matrix = get_transmatrix(kx, ky, rz, tx, ty, trape)
    trans_sources = transform_points(trans_matrix, sources)
    try:
        trans_targets = transform_points(tf.linalg.inv(trans_matrix), targets)
    except BaseException:
        trans_targets = targets
    residuals_points = tf.reduce_sum(((targets - trans_sources) * padded)**2, axis=2)
    residuals = tf.reduce_sum(residuals_points, axis=1)
    return trans_sources, trans_targets, trans_matrix, kx, ky, rz, tx, ty, residuals_points, residuals


def possible_misassignment(trans_sources, targets, padded, padded_player_ids, top_n=2):
    """
    batch, num_player, 2(x,y)
    """
    batch, num_player, _ = tf.unstack(tf.shape(targets))
    dists = tf.reduce_sum((targets[:, :, tf.newaxis, :] - trans_sources[:, tf.newaxis, :, :])**2, axis=3)
    padded_area = 1 - padded[:, :, tf.newaxis, 0] * padded[:, tf.newaxis, :, 0]
    current_asign = tf.eye(num_player, dtype=dists.dtype)[tf.newaxis, :, :]
    dists = dists + (padded_area + current_asign) * 1e7
    argsort = tf.argsort(dists, axis=2)[:, :, :top_n]
    possible_residuals = tf.gather(dists, argsort, batch_dims=2)
    possible_player_ids = tf.gather(padded_player_ids, argsort, batch_dims=1)
    return possible_residuals, possible_player_ids


def pad_for_batch(list_tf_targets, list_tf_sources, list_tf_player_ids):
    num_points = [points.shape[0] for points in list_tf_targets]
    max_num = max(num_points)
    num_pads = [max_num - num for num in num_points]
    list_tf_targets = [tf.pad(points, [[0, pad_num], [0, 0]]) for points, pad_num in zip(list_tf_targets, num_pads)]
    list_tf_sources = [tf.pad(points, [[0, pad_num], [0, 0]]) for points, pad_num in zip(list_tf_sources, num_pads)]
    list_tf_player_ids = [tf.pad(ids, [[0, pad_num]], constant_values=-2)
                          for ids, pad_num in zip(list_tf_player_ids, num_pads)]
    tf_targets = tf.stack(list_tf_targets)
    tf_sources = tf.stack(list_tf_sources)
    tf_player_ids = tf.stack(list_tf_player_ids)
    tf_pad_bools = tf.cast(tf.stack([tf.range(max_num) < num for num in num_points]),
                           tf_targets.dtype)[:, :, tf.newaxis]
    return tf_targets, tf_sources, tf_player_ids, tf_pad_bools, num_points


def depad_for_batch(tf_items, num_points):
    list_np_items = [[points[:num] for points, num in zip(val.numpy(), num_points)] for val in tf_items]
    #list_tf_sources = [points[:num] for points, num in zip(tf_sources.numpy(), num_points)]
    return list_np_items


def p2p_registration_features(tracking, helmets, meta, view_some_results=False, num_possible_assign=2):
    """
    num_possible_assign:
    アサインメントミスの可能性のあるプレイヤを抽出する。
    具体的には現在アサインされている以外に近いプレイヤを抽出する。同時にもしそのプレイヤであった場合の残差も出す(減残差と比較することで、あとでマスク的に使う)
    """
    S = time.time()
    outputs = []
    for game_play in tracking["game_play"].unique():
        for view in ["Sideline", "Endzone"]:
            helm_merged = prepare_matching_dataframe(
                game_play,
                tracking,
                helmets,
                meta,
                view,
                fps=59.94,
                only_center_of_step=True)
            helm_merged["box_x"] = helm_merged["left"] + helm_merged["width"] / 2
            helm_merged["box_y"] = helm_merged["top"] + helm_merged["height"] / 2

            scale_rate = 10
            list_targets = []
            list_sources = []
            list_flip_target = []
            frame_nos = []
            player_ids = []
            df_out = []
            for frame, _df in helm_merged.groupby("frame"):

                box_x = _df["box_x"].values
                box_y = _df["box_y"].values
                box_x_mean = box_x.mean()
                box_y_mean = box_y.mean()

                pos_x = _df["x_position"].values
                pos_y = _df["y_position"].values
                pos_x_mean = pos_x.mean()
                pos_y_mean = pos_y.mean()
                if np.isnan(box_x_mean) or np.isnan(pos_x_mean):
                    continue

                adjusted_bx = (box_x - box_x_mean) / scale_rate
                adjusted_by = (box_y - box_y_mean) / scale_rate

                adjusted_px = pos_x - pos_x_mean
                adjusted_py = pos_y - pos_y_mean

                list_targets += [tf.cast(tf.stack([adjusted_px, adjusted_py], axis=-1), tf.float32)]
                list_sources += [tf.cast(tf.stack([adjusted_bx, adjusted_by], axis=-1), tf.float32)]

                list_flip_target += [tf.cast(tf.stack([-adjusted_px, adjusted_py], axis=-1), tf.float32)]

                frame_nos += [frame] * len(box_x)
                player_ids += [tf.cast(_df["nfl_player_id"].values, tf.int32)]

                _df["average_box_size"] = (np.sqrt(_df["width"] * _df["height"])).mean()
                df_out.append(_df[["game_play", "view", "frame", "step", "nfl_player_id", "average_box_size"]].copy())

            tf_targets, tf_sources, tf_player_ids, tf_pad_bools, num_points = pad_for_batch(
                list_targets, list_sources, player_ids)
            tf_flip_targets, tf_sources, _, tf_pad_bools, num_points = pad_for_batch(
                list_flip_target, list_sources, player_ids)
            # print(num_points)

            with tf.device('/CPU:0'):  # gpu少し怪しい。
                #result_flip = batch_points2points_fitting(tf_flip_targets, tf_sources, tf_pad_bools, num_iter=30, l2_reg=100., init_rot=0)
                #result_flip_180 = batch_points2points_fitting(tf_flip_targets, tf_sources, tf_pad_bools, num_iter=30, l2_reg=100., init_rot=3.14)
                result_flip = batch_points2points_fitting_highdof(
                    tf_flip_targets, tf_sources, tf_pad_bools, num_iter=30, l2_reg=100., init_rot=0)
                result_flip_180 = batch_points2points_fitting_highdof(
                    tf_flip_targets, tf_sources, tf_pad_bools, num_iter=30, l2_reg=100., init_rot=3.14)

            # take smaller (better fit)
            result_flip_stack = [tf.stack([r1, r2], axis=-1) for r1, r2 in zip(result_flip, result_flip_180)]
            better = tf.argmin(result_flip_stack[-1], axis=-1)  # last index is residual
            result_flip = [tf.gather(result, better, batch_dims=1, axis=-1) for result in result_flip_stack]

            flip_residual_median = np.median(result_flip[-1])
            print(f"{game_play}, {view}, residual flip: {flip_residual_median}")

            result = result_flip
            view_target = tf_flip_targets

            trans_sources, trans_targets, trans_matrix, kx, ky, rz, tx, ty, residual_points, residual = result

            possible_residuals, possible_player_ids = possible_misassignment(
                trans_sources, tf_flip_targets, tf_pad_bools, tf_player_ids, top_n=num_possible_assign)

            list_targets, list_sources, list_residual, possible_residuals, possible_player_ids = depad_for_batch([view_target, trans_sources, residual_points, possible_residuals, possible_player_ids],
                                                                                                                 num_points)  # リストで好きなだけ渡せるようにするか。残渣も各ポイントでほしいし。
            """

            for st, [possible, current, ppids] in enumerate(zip(possible_residuals, list_residual, possible_player_ids)):
                sabun = np.array(possible)[:,0] - np.array(current).reshape(-1)
                if sabun.min() < 0:
                    print(st, "-------------------------")
                    print(possible)
                    print(current)
                    print(ppids)
                #if
                #print(possible_residuals[0], possible_player_ids[0])
                #print(residual_points[0])
                #raise Exception()
            """

            tiled_resisual = []
            for r, num_p in zip(residual.numpy(), num_points):
                tiled_resisual += [r] * num_p

            df_out = pd.concat(df_out, axis=0)
            df_out[["x_position_offset_on_img", "y_position_offset_on_img"]] = (
                np.concatenate(list_sources, axis=0) - np.concatenate(list_targets, axis=0)) * scale_rate
            df_out["p2p_registration_residual"] = np.concatenate(list_residual, axis=0)
            df_out["p2p_registration_residual_frame"] = tiled_resisual
            df_out[[f"p2p_registration_possible_player_{i}" for i in range(
                num_possible_assign)]] = np.concatenate(possible_player_ids, axis=0)
            df_out[[f"p2p_registration_possible_residual_{i}" for i in range(
                num_possible_assign)]] = np.concatenate(possible_residuals, axis=0)

            outputs.append(df_out)

            if view_some_results:
                visualize(list_targets, list_sources)

        df_p2p_regist = pd.concat(outputs, axis=0)
        print(time.time() - S)
    return df_p2p_regist


def match_p2p_with_cache(cache_path, **kwargs):
    if not os.path.exists(cache_path):
        df = p2p_registration_features(**kwargs)
        df = df.reset_index(drop=True)
        df.to_feather(cache_path)
    return pd.read_feather(cache_path)


def add_p2p_matching_features(df, regist_df):
    regist_df["x_rel_position_offset_on_img"] = regist_df["x_position_offset_on_img"] / \
        (regist_df["average_box_size"] + 1e-7)
    regist_df["y_rel_position_offset_on_img"] = regist_df["y_position_offset_on_img"] / \
        (regist_df["average_box_size"] + 1e-7)

    merge_columns = [
        "game_play",
        "step",
        "nfl_player_id",
        "x_position_offset_on_img",
        "y_position_offset_on_img",
        "x_rel_position_offset_on_img",
        "y_rel_position_offset_on_img",
        "p2p_registration_residual",
        "p2p_registration_residual_frame"
    ]
    df = pd.merge(df, regist_df.loc[regist_df["view"] == "Sideline", merge_columns], how="left",
                  left_on=["game_play", "step", "nfl_player_id_1"],
                  right_on=["game_play", "step", "nfl_player_id"])
    df.rename(columns={"x_position_offset_on_img": "x_position_offset_on_img_Side",
                       "y_position_offset_on_img": "y_position_offset_on_img_Side",
                       "x_rel_position_offset_on_img": "x_rel_position_offset_on_img_Side",
                       "y_rel_position_offset_on_img": "y_rel_position_offset_on_img_Side",
                       "p2p_registration_residual": "p2p_registration_residual_Side",
                       "p2p_registration_residual_frame": "p2p_registration_residual_frame_Side",
                       }, inplace=True)
    df.drop(columns=["nfl_player_id"], inplace=True)

    df = pd.merge(df, regist_df.loc[regist_df["view"] == "Endzone", merge_columns], how="left",
                  left_on=["game_play", "step", "nfl_player_id_1"],
                  right_on=["game_play", "step", "nfl_player_id"])
    df.rename(columns={"x_position_offset_on_img": "x_position_offset_on_img_End",
                       "y_position_offset_on_img": "y_position_offset_on_img_End",
                       "x_rel_position_offset_on_img": "x_rel_position_offset_on_img_End",
                       "y_rel_position_offset_on_img": "y_rel_position_offset_on_img_End",
                       "p2p_registration_residual": "p2p_registration_residual_End",
                       "p2p_registration_residual_frame": "p2p_registration_residual_frame_End",
                       }, inplace=True)
    df.drop(columns=["nfl_player_id"], inplace=True)
    return reduce_dtype(df)

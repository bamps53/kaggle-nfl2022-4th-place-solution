from sklearn.metrics import matthews_corrcoef
from scipy.optimize import minimize


def search_best_threshold(y_true, y_pred):
    def func(x_list):
        score = matthews_corrcoef(y_true, y_pred > x_list[0])
        return -score

    x0 = [0.5]
    result = minimize(func, x0,  method="nelder-mead")

    return result.x[0]


def binarize_pred(y_pred, threshold, threshold2, threshold2_mask):
    return ~threshold2_mask * (y_pred > threshold) + \
        threshold2_mask * (y_pred > threshold2)


def search_best_threshold_pair(y_true, y_pred, is_ground):
    def func(x_list):
        score = matthews_corrcoef(y_true, binarize_pred(
            y_pred, x_list[0], x_list[1], is_ground))
        return -score

    x0 = [0.5, 0.5]
    result = minimize(func, x0, method="nelder-mead")

    return result.x[0], result.x[1]


def evaluate_pred_df(val_df, pred_df, ground_id=0):
    val_results = {}
    pred_df = val_df.merge(pred_df.drop('step', axis=1), on=[
                           'game_play', 'frame', 'nfl_player_id_1', 'nfl_player_id_2'], how='left')

    is_ground = (pred_df['nfl_player_id_2'] == ground_id).values
    th0, th1 = search_best_threshold_pair(pred_df.contact.values, pred_df.preds.values, is_ground)
    score = matthews_corrcoef(pred_df.contact.values,  binarize_pred(pred_df.preds.values, th0, th1, is_ground))
    # print(f'total score={score:.3f} best_th_play={th0:.3f}  best_th_ground={th1:.3f}')
    val_results['inter_th'] = th0
    val_results['ground_th'] = th1
    val_results['total_score'] = score
    val_results['score'] = score

    pred_g_df = pred_df.query('nfl_player_id_2 == @ground_id')
    score = matthews_corrcoef(pred_g_df.contact.values,  pred_g_df.preds.values > th0)
    # print(f'ground score={score:.3f} best_th={th0:.3f}')
    val_results['ground_score'] = score

    pred_inter_df = pred_df.query('nfl_player_id_2 != @ground_id')
    score = matthews_corrcoef(pred_inter_df.contact.values,  pred_inter_df.preds.values > th1)
    # print(f'inter score={score:.3f} best_th={th1:.3f}')
    val_results['inter_score'] = score

    pred_valid_df = pred_df.query('masks == True')
    is_ground = (pred_valid_df['nfl_player_id_2'] == ground_id).values
    th0, th1 = search_best_threshold_pair(pred_valid_df.contact.values, pred_valid_df.preds.values, is_ground)
    score = matthews_corrcoef(pred_valid_df.contact.values,  binarize_pred(
        pred_valid_df.preds.values, th0, th1, is_ground))
    # print(f'mask score={score:.3f} best_th_play={th0:.3f}  best_th_ground={th1:.3f}')
    val_results['mask_inter_th'] = th0
    val_results['mask_ground_th'] = th1
    val_results['mask_score'] = score

    return val_results

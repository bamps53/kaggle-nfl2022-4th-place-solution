import torch


def transform_rois(rois):
    rois_with_batch_dim = []
    for idx, roi in enumerate(rois):
        b_dim = torch.ones(len(roi), 1).to(roi) * idx
        roi = torch.cat([b_dim, roi], dim=1)
        rois_with_batch_dim.append(roi)
    return torch.cat(rois_with_batch_dim, dim=0)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedBCEWithLogitsLoss(nn.Module):
    def __init__(self, label_smoothing=0.0):
        super().__init__()
        self.bce = torch.nn.BCEWithLogitsLoss(reduction='none')
        self.label_smoothing = label_smoothing

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, pad_mask: torch.Tensor, mixup_coeffs=None):
        pad_mask = pad_mask.bool()
        if pad_mask.sum() == 0:
            return torch.tensor(0.0).to(y_pred)

        y_pred = y_pred[pad_mask]
        y_true = y_true[pad_mask]
        if self.label_smoothing > 0:
            y_true = y_true.clip(self.label_smoothing, 1-self.label_smoothing)
        loss = self.bce(y_pred, y_true)

        if mixup_coeffs is not None:
            mixup_coeffs = mixup_coeffs[:, None].expand_as(pad_mask)
            mixup_coeffs = mixup_coeffs[pad_mask]
            loss = loss * mixup_coeffs

        return loss.mean()

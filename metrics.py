import torch
import numpy as np


def dice_coefficient(pred_mask: torch.Tensor, true_mask: torch.Tensor, smooth: float = 1e-6) -> float:
    pred_flat = pred_mask.reshape(-1).float()
    true_flat = true_mask.reshape(-1).float()
    intersection = (pred_flat * true_flat).sum()
    return float((2 * intersection + smooth) / (pred_flat.sum() + true_flat.sum() + smooth))


def iou_score(pred_mask: torch.Tensor, true_mask: torch.Tensor, smooth: float = 1e-6) -> float:
    pred_flat = pred_mask.reshape(-1).float()
    true_flat = true_mask.reshape(-1).float()
    intersection = (pred_flat * true_flat).sum()
    union = pred_flat.sum() + true_flat.sum() - intersection
    return float((intersection + smooth) / (union + smooth))


def pixel_accuracy(pred_mask: torch.Tensor, true_mask: torch.Tensor) -> float:
    return float((pred_mask == true_mask).float().mean())


def sensitivity(pred_mask: torch.Tensor, true_mask: torch.Tensor, smooth: float = 1e-6) -> float:
    tp = (pred_mask * true_mask).sum().float()
    fn = ((1 - pred_mask) * true_mask).sum().float()
    return float((tp + smooth) / (tp + fn + smooth))


def specificity(pred_mask: torch.Tensor, true_mask: torch.Tensor, smooth: float = 1e-6) -> float:
    tn = ((1 - pred_mask) * (1 - true_mask)).sum().float()
    fp = (pred_mask * (1 - true_mask)).sum().float()
    return float((tn + smooth) / (tn + fp + smooth))


def hausdorff_distance(pred_mask: np.ndarray, true_mask: np.ndarray) -> float:
    pred_pts = np.argwhere(pred_mask)
    true_pts = np.argwhere(true_mask)

    if len(pred_pts) == 0 or len(true_pts) == 0:
        return 0.0

    from scipy.spatial.distance import cdist

    dists = cdist(pred_pts, true_pts, metric="euclidean")
    hd_pred = dists.min(axis=1)
    hd_true = dists.min(axis=0)
    all_hd = np.concatenate([hd_pred, hd_true])
    return float(np.percentile(all_hd, 95))


def evaluate_batch(logits: torch.Tensor, targets: torch.Tensor) -> dict:
    preds = logits.argmax(dim=1)

    metrics = {"dice": [], "iou": [], "accuracy": [], "sensitivity": [], "specificity": []}

    for pred, target in zip(preds, targets):
        metrics["dice"].append(dice_coefficient(pred, target))
        metrics["iou"].append(iou_score(pred, target))
        metrics["accuracy"].append(pixel_accuracy(pred, target))
        metrics["sensitivity"].append(sensitivity(pred, target))
        metrics["specificity"].append(specificity(pred, target))

    return {k: float(np.mean(v)) for k, v in metrics.items()}

"""losses, Dice/IoU metrics, and the training loop."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def dice_loss(logits, target, eps: float = 1e-7):
    probs = torch.sigmoid(logits)
    intersection = (probs * target).sum(dim=(1, 2, 3))
    union = probs.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2 * intersection + eps) / (union + eps)
    return 1 - dice.mean()


def bce_loss(logits, target):
    return F.binary_cross_entropy_with_logits(logits, target)


def combined_loss(logits, target):
    return bce_loss(logits, target) + dice_loss(logits, target)


LOSS_FNS = {"bce": bce_loss, "dice": dice_loss, "bce_dice": combined_loss}


def dice_coefficient(logits, target, threshold: float = 0.5, eps: float = 1e-7) -> float:
    preds = (torch.sigmoid(logits) > threshold).float()
    intersection = (preds * target).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2 * intersection + eps) / (union + eps)
    return dice.mean().item()


def iou_score(logits, target, threshold: float = 0.5, eps: float = 1e-7) -> float:
    preds = (torch.sigmoid(logits) > threshold).float()
    intersection = (preds * target).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) - intersection
    iou = (intersection + eps) / (union + eps)
    return iou.mean().item()


def run_epoch(model, loader: DataLoader, loss_name: str, device, optimizer=None) -> dict:
    """One pass over `loader`. Trains if `optimizer` is given, else evaluates."""
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    loss_fn = LOSS_FNS[loss_name]

    total_loss, total_dice, total_iou, n_batches = 0.0, 0.0, 0.0, 0
    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
            total_dice += dice_coefficient(logits, y)
            total_iou += iou_score(logits, y)
            n_batches += 1
    return {
        "loss": total_loss / n_batches,
        "dice": total_dice / n_batches,
        "iou": total_iou / n_batches,
    }


def train_model(model, train_loader, val_loader, device, epochs: int = 15,
                 lr: float = 1e-3, loss_name: str = "bce_dice", verbose: bool = True) -> dict:
    """Full training loop. Returns a history dict of per-epoch metrics."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = {"train_loss": [], "val_loss": [], "train_dice": [], "val_dice": [],
               "train_iou": [], "val_iou": []}
    for epoch in range(1, epochs + 1):
        tr = run_epoch(model, train_loader, loss_name, device, optimizer=optimizer)
        va = run_epoch(model, val_loader, loss_name, device, optimizer=None)
        for k, v in tr.items():
            history[f"train_{k}"].append(v)
        for k, v in va.items():
            history[f"val_{k}"].append(v)
        if verbose:
            print(f"Epoch {epoch:2d}/{epochs} | train loss {tr['loss']:.4f} dice {tr['dice']:.3f} "
                  f"| val loss {va['loss']:.4f} dice {va['dice']:.3f} iou {va['iou']:.3f}")
    return history


def loss_ablation(model_factory, train_loader, val_loader, test_loader, device,
                   epochs: int = 6, lr: float = 1e-3, seed: int = 42) -> dict:
    """Extension: train identical models from scratch with each of
    the three losses and report which gives the best VALIDATION Dice.
    Test-set Dice/IoU are also included (same
    bce_dice metric for all three, for a fair comparison) as a secondary,
    generalisation-focused check, but validation Dice is the headline
    number this function is built to answer.

    `model_factory` is a zero-arg callable that returns a fresh, untrained
    model (e.g. `lambda: UNet(base=16)`) so each loss starts from the same
    initial conditions rather than continuing from a previous run.
    """
    import numpy as np

    results = {}
    for loss_name in ["bce", "dice", "bce_dice"]:
        np.random.seed(seed)
        torch.manual_seed(seed)
        model = model_factory().to(device)
        history = train_model(model, train_loader, val_loader, device, epochs=epochs,
                               lr=lr, loss_name=loss_name, verbose=False)
        test_metrics = run_epoch(model, test_loader, "bce_dice", device, optimizer=None)
        results[loss_name] = {
            "history": history,
            "val_dice": history["val_dice"][-1],  
            "val_iou": history["val_iou"][-1],
            "test_dice": test_metrics["dice"],
            "test_iou": test_metrics["iou"],
        }
        print(f"[loss_ablation] {loss_name:10s} -> val dice {history['val_dice'][-1]:.4f}, "
              f"val iou {history['val_iou'][-1]:.4f} | test dice {test_metrics['dice']:.4f}, "
              f"iou {test_metrics['iou']:.4f}")
    return results

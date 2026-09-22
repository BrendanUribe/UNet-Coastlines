"""Losses for joint segmentation and boundary detection.

What changed and why
--------------------
The original was::

    total_loss = seg_loss + edge_loss

with ``class_weights = [0.075, 2.109, 0.816]`` and ``pos_weight = 31.2``
hardcoded and no record of how they were derived.  Three problems:

* **The 1:1 task weighting was an unexamined assumption.**  The two terms do
  not live on the same scale - the original run had edge logits spanning
  -41.7..+2.7 while segmentation class means sat at 3.35/6.79/-8.77.  Task
  weights are now *learned* through homoscedastic uncertainty (Kendall, Gal &
  Cipolla, CVPR 2018, arXiv:1705.07115), which replaces a magic constant with
  a parameter you can report.

* **Cross-entropy alone is the wrong objective for a one-pixel-wide
  structure.**  Segmentation adds a soft Dice term (Milletari et al., V-Net,
  3DV 2016, arXiv:1606.04797); CE + Dice is the nnU-Net default (Isensee et
  al., Nature Methods 2021) and a defensible baseline.

* **Class weights had no provenance.**  They are now computed from the actual
  dataset by ``compute_class_weights.py`` and passed in.  Nothing is baked in.

The boundary term uses class-balanced cross-entropy with the positive weight
estimated *per batch* from the labels, which is the weighting scheme HED
introduced for exactly this problem (Xie & Tu, ICCV 2015, arXiv:1504.06375),
plus per-channel validity masks so unobservable boundaries (a coast under
cloud, the unlit limb) are neither rewarded nor punished.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_dice_loss(logits, target, num_classes, ignore_mask=None, eps=1.0):
    """Multi-class soft Dice on softmax probabilities.

    ``ignore_mask`` is ``True`` where a pixel should not contribute.
    """
    probs = logits.softmax(dim=1)
    onehot = F.one_hot(target.clamp(min=0), num_classes).permute(0, 3, 1, 2).float()

    if ignore_mask is not None:
        keep = (~ignore_mask).unsqueeze(1).float()
        probs = probs * keep
        onehot = onehot * keep

    dims = (0, 2, 3)
    inter = (probs * onehot).sum(dims)
    denom = probs.sum(dims) + onehot.sum(dims)
    dice = (2 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()


def balanced_bce(logits, target, valid, max_pos_weight=100.0):
    """Class-balanced BCE over valid pixels only, weighted **per channel**.

    The positive weight is estimated from the batch rather than hardcoded, so
    it tracks the data instead of going stale when the dataset is regenerated.

    It must be estimated per channel, because the three boundary types differ
    in density by more than an order of magnitude - on a representative set,
    coastline is ~2.6% of valid pixels, limb ~0.66%, terminator ~0.36%, which
    call for weights of roughly 37, 151 and 279. Pooling them into a single
    scalar gives ~81 for all three: coastline is over-weighted 2.2x and is
    driven to over-predict (high recall, poor precision, thick blobby edges
    rather than thin lines), while the terminator is under-weighted.
    """
    valid = valid.float()
    dims = (0, 2, 3) if logits.dim() == 4 else tuple(range(logits.dim()))

    n_valid = valid.sum(dims).clamp(min=1.0)
    n_pos = (target * valid).sum(dims).clamp(min=1.0)
    pos_weight = ((n_valid - n_pos) / n_pos).clamp(max=max_pos_weight)
    if logits.dim() == 4:
        pos_weight = pos_weight.view(-1, 1, 1)   # broadcast over (B, C, H, W)

    raw = F.binary_cross_entropy_with_logits(
        logits, target, reduction="none", pos_weight=pos_weight
    )
    # Normalise by total valid pixels so the scale matches the pooled version.
    return (raw * valid).sum() / valid.sum().clamp(min=1.0)


class CombinedLoss(nn.Module):
    """Segmentation + boundary loss with learned task weighting.

    Parameters
    ----------
    num_classes
        Segmentation class count.
    class_weights
        Optional per-class weights for the cross-entropy term.  Produce them
        with ``compute_class_weights.py``; ``None`` means unweighted.
    dice_weight
        Relative weight of Dice against cross-entropy inside the segmentation
        term.  1.0 gives the nnU-Net-style equal mix.
    deep_supervision_weight
        Weight applied to the mean of the HED side outputs.
    learn_task_weights
        Learn the segmentation/boundary balance via homoscedastic uncertainty.
        Set ``False`` and use ``static_edge_weight`` for an ablation.
    ignore_index
        Segmentation label to exclude (``-1`` disables).

    The learned weighting optimises
    ``L = sum_i exp(-s_i) * L_i + s_i`` where ``s_i = log(sigma_i^2)``.  The
    ``+ s_i`` term is what stops the model from driving both weights to zero.

    Note that the reported total can go **negative** once the task losses fall
    below the regularisation term.  That is expected for this objective and is
    not a divergence; compare the per-task ``seg`` and ``edge`` figures, which
    stay positive, when judging whether training is progressing.
    """

    def __init__(
        self,
        num_classes: int = 5,
        class_weights=None,
        dice_weight: float = 1.0,
        deep_supervision_weight: float = 0.4,
        learn_task_weights: bool = True,
        static_edge_weight: float = 1.0,
        ignore_index: int = -1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.dice_weight = dice_weight
        self.deep_supervision_weight = deep_supervision_weight
        self.learn_task_weights = learn_task_weights
        self.static_edge_weight = static_edge_weight
        self.ignore_index = ignore_index

        if class_weights is None:
            self.register_buffer("class_weights", None)
        else:
            self.register_buffer(
                "class_weights", torch.as_tensor(class_weights, dtype=torch.float32)
            )

        # log(sigma^2) per task, initialised at 0 -> both weights start at 1.
        self.log_var = nn.Parameter(torch.zeros(2), requires_grad=learn_task_weights)

    def forward(self, seg_out, edge_out, seg_label, edge_label, edge_valid, aux=()):
        ignore = (
            seg_label == self.ignore_index
            if self.ignore_index >= 0
            else torch.zeros_like(seg_label, dtype=torch.bool)
        )

        ce = F.cross_entropy(
            seg_out,
            seg_label,
            weight=self.class_weights,
            ignore_index=self.ignore_index if self.ignore_index >= 0 else -100,
        )
        dice = soft_dice_loss(
            seg_out, seg_label, self.num_classes,
            ignore_mask=ignore if ignore.any() else None,
        )
        seg_loss = ce + self.dice_weight * dice

        edge_loss = balanced_bce(edge_out, edge_label, edge_valid)
        if aux:
            aux_loss = torch.stack(
                [balanced_bce(a, edge_label, edge_valid) for a in aux]
            ).mean()
            edge_loss = edge_loss + self.deep_supervision_weight * aux_loss

        if self.learn_task_weights:
            precision = torch.exp(-self.log_var)
            total = (
                precision[0] * seg_loss + self.log_var[0]
                + precision[1] * edge_loss + self.log_var[1]
            )
        else:
            total = seg_loss + self.static_edge_weight * edge_loss

        return total, {
            "total": total.detach(),
            "seg": seg_loss.detach(),
            "seg_ce": ce.detach(),
            "seg_dice": dice.detach(),
            "edge": edge_loss.detach(),
            "w_seg": torch.exp(-self.log_var[0]).detach(),
            "w_edge": torch.exp(-self.log_var[1]).detach(),
        }

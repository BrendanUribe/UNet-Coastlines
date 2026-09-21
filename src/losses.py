import torch
import torch.nn as nn

def dice_loss(pred_prob, target, smooth=1e-6):
    intersection = (pred_prob * target).sum()  # how much overlaps with the target
    return 1 - (2. * intersection + smooth) / (pred_prob.sum() + target.sum() + smooth)

def edge_loss_single(edge_out_single, edge_label, cloud_label, pos_weight=5.6):
    edge_criterion = nn.BCEWithLogitsLoss(
        reduction='none',
        pos_weight=torch.tensor([pos_weight]).to(edge_out_single.device)
    )
    edge_loss_raw = edge_criterion(edge_out_single.squeeze(1), edge_label)

    edge_prob = torch.sigmoid(edge_out_single.squeeze(1))
    valid_mask = (cloud_label == 0).float()

    bce = (edge_loss_raw * valid_mask).sum() / valid_mask.sum().clamp(min=1)
    dice = dice_loss(edge_prob * valid_mask, edge_label * valid_mask)

    return bce + dice

def combined_loss(seg_out, edge_outputs, seg_label, edge_label, cloud_label):
    # sqrt-inverse-frequency weights computed from the REAL dataset, background excluded
    # (water 43.7% / land 17.4% / cloud 38.9%, via audit_dataset_labels.py) - much milder
    # than the original 28x-imbalanced weights, since most of that imbalance turned out to
    # be background space rather than a real difference between water/land/cloud
    class_weights = torch.tensor([0.824, 1.304, 0.873]).to(seg_out.device)
    # ignore_index=-100 skips background/space pixels entirely (see coastline_dataset.py's
    # IGNORE_INDEX) instead of counting them as "water" - they were diluting the water class's
    # gradient signal with millions of trivial black pixels instead of real ocean examples
    seg_criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100)
    seg_loss = seg_criterion(seg_out, seg_label)

    # edge_outputs is [edge_d3_up, edge_d2_up, edge_d1, edge_fused] (see UNet.py forward) -
    # grade each scale's prediction against the same ground truth and average them, so every
    # decoder stage actually gets trained on edges, not just the final fused output
    edge_loss = sum(edge_loss_single(e, edge_label, cloud_label) for e in edge_outputs) / len(edge_outputs)

    total_loss = seg_loss + edge_loss

    return total_loss, seg_loss, edge_loss

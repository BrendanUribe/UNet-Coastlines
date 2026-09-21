import torch
import torch.nn as nn

def combined_loss(seg_out, edge_out, seg_label, edge_label, cloud_label):
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

    edge_criterion = nn.BCEWithLogitsLoss(
        reduction='none',
        pos_weight=torch.tensor([31.2]).to(edge_out.device)
    )
    edge_loss_raw = edge_criterion(edge_out.squeeze(1), edge_label)

    valid_mask = (cloud_label == 0).float()
    edge_loss = (edge_loss_raw * valid_mask).sum() / valid_mask.sum().clamp(min=1)

    total_loss = seg_loss + edge_loss

    return total_loss, seg_loss, edge_loss
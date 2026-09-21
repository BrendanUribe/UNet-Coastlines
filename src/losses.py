import torch
import torch.nn as nn

def combined_loss(seg_out, edge_out, seg_label, edge_label, cloud_label):
    # NOTE: these were sqrt-inverse-frequency weights computed from raw water/land/cloud
    # pixel counts (~87%/4%/9%). The label audit showed most of that "water" mass is actually
    # black background/space, not ocean (now excluded via ignore_index below) - so these
    # weights are still calibrated against the WRONG frequencies and should be recomputed
    # from audit_dataset_labels.py once it reports frequency excluding background pixels.
    class_weights = torch.tensor([0.312, 1.657, 1.031]).to(seg_out.device)
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
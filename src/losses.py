import torch
import torch.nn as nn

def combined_loss(seg_out, edge_out, seg_label, edge_label, cloud_label):
    class_weights = torch.tensor([0.075, 2.109, 0.816]).to(seg_out.device)
    seg_criterion = nn.CrossEntropyLoss(weight=class_weights)
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
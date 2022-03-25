import torch
from torch import nn

from utils import intersection_over_union
from params import S, B, C

class YoloLoss(nn.Module):
    """
    YOLO loss function comparing predicted bounding boxes with actual. Requires B = 2 for the tensor optimizations to work.
    """
    def __init__(self):
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss(reduction='sum')
        self.lambda_coord = 5
        self.lambda_noobj = 0.5

    def forward(self, predictions, labels):
        """
        Calculates the loss between predicted and actual bounding boxes.

        Args:
            predictions: Predicted bounding boxes, has shape (batch_size, S * S * (C + B * 5)).
            labels: Actual bounding boxes, has shape (batch_size, S, S, C + 5). Each box contains class probabilities, confidence, x, y, width, height information, in order.
        
        Returns:
            Loss calculated from differences in x, y, width, height, confidence, and class probabilities.
        """

        predictions = predictions.reshape(-1, S, S, C + B * 5)

        IOU_b1 = intersection_over_union(
            predictions[..., C+1:C+5], labels[..., C+1:C+5]
        )
        IOU_b2 = intersection_over_union(
            predictions[..., C+6:C+10], labels[..., C+1:C+5]
        )
        IOUs = torch.cat([IOU_b1.unsqueeze(0), IOU_b2.unsqueeze(0)], dim=0)
        _, best_box = torch.max(IOUs, dim=0)
        exists_obj = labels[..., 0].unsqueeze(-1)

        # Box loss
        obj_predictions = exists_obj * (
            best_box * predictions[..., C+6:C+10] +
            (1 - best_box) * predictions[..., C+1:C+5]
        )
        obj_labels = exists_obj * labels[..., C+1:C+5]

        obj_predictions[..., 2:4] = torch.sqrt(torch.abs(obj_predictions[..., 2:4] + 1e-6))
        obj_labels[..., 2:4] = torch.sqrt(obj_labels[..., 2:4])

        box_loss = self.mse(
            torch.flatten(obj_predictions, end_dim=-2),
            torch.flatten(obj_labels, end_dim=-2)
        )

        # Confidence loss
        doubled_exists_box = torch.cat([exists_obj.unsqueeze(0), exists_obj.unsqueeze(0)], dim=0)
        pred_conf = torch.cat([predictions[..., C:C+1].unsqueeze(0), predictions[..., C+5:C+6].unsqueeze(0)], dim=0)
        obj_conf_loss = self.mse(
            torch.flatten(doubled_exists_box * pred_conf),
            torch.flatten(doubled_exists_box * IOUs)
        )
        noobj_conf_loss = self.mse(
            torch.flatten((1 - doubled_exists_box) * pred_conf),
            torch.flatten((1 - doubled_exists_box) * IOUs)
        )

        # Class loss
        class_loss = self.mse(
            torch.flatten(exists_obj * predictions[..., :C], end_dim=-2),
            torch.flatten(exists_obj * labels[..., :C], end_dim=-2)
        )

        # Total loss
        loss = (
            self.lambda_coord * box_loss +
            obj_conf_loss +
            self.lambda_noobj * noobj_conf_loss +
            class_loss
        )
        return loss
import torch
from torch import nn

from utils import intersection_over_union
from params import S, B, C, losses

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
            predictions: Predicted bounding boxes, has shape (batch_size, S, S, C + B * 5).
            labels: Actual bounding boxes, has shape (batch_size, S, S, C + 5). Each box contains class probabilities, confidence, x, y, width, height information, in order.
        
        Returns:
            Loss calculated from differences in x, y, width, height, confidence, and class probabilities.
        """

        pred_box1 = predictions[..., C+1:C+5]
        pred_box2 = predictions[..., C+6:C+10]
        true_box = labels[..., C+1:C+5]
        pred_box1[..., 2:4] *= S
        pred_box2[..., 2:4] *= S
        true_box[..., 2:4] *= S
        iou1 = intersection_over_union(
            pred_box1, true_box
        )
        iou2 = intersection_over_union(
            pred_box2, true_box
        )
        pred_box1[..., 2:4] /= S
        pred_box2[..., 2:4] /= S
        true_box[..., 2:4] /= S
        _, best_box = torch.max(torch.cat([iou1.unsqueeze(0), iou2.unsqueeze(0)], dim=0), dim=0)
        exists_obj = labels[..., C].unsqueeze(-1)

        # Box loss
        obj_predictions = exists_obj * (
            best_box * predictions[..., C+6:C+10] +
            (1 - best_box) * predictions[..., C+1:C+5]
        )
        obj_labels = exists_obj * labels[..., C+1:C+5]

        obj_predictions[..., 2:4] = exists_obj * torch.sign(obj_predictions[..., 2:4]) * \
            torch.sqrt(torch.abs(obj_predictions[..., 2:4] + 1e-6))
        obj_labels[..., 2:4] = torch.sqrt(obj_labels[..., 2:4])

        box_loss = self.mse(
            torch.flatten(obj_predictions, end_dim=-2),
            torch.flatten(obj_labels, end_dim=-2)
        )

        # Confidence loss
        obj_predictions = exists_obj * (
            best_box * predictions[..., C+5:C+6] +
            (1 - best_box) * predictions[..., C:C+1]
        )
        obj_labels = exists_obj * (
            best_box * iou2 +
            (1 - best_box) * iou1
        )

        obj_conf_loss = self.mse(
            torch.flatten(obj_predictions, end_dim=-2),
            torch.flatten(obj_labels, end_dim=-2)
        )

        noobj_conf_loss = 0

        noobj_predictions = (1 - exists_obj) * predictions[..., C:C+1]
        noobj_labels = (1 - exists_obj) * iou1
        noobj_conf_loss += self.mse(
            torch.flatten(noobj_predictions, end_dim=-2),
            torch.flatten(noobj_labels, end_dim=-2)
        )
        noobj_predictions = (1 - exists_obj) * predictions[..., C+5:C+6]
        noobj_labels = (1 - exists_obj) * iou2
        noobj_conf_loss += self.mse(
            torch.flatten(noobj_predictions, end_dim=-2),
            torch.flatten(noobj_labels, end_dim=-2)
        )

        # Class loss
        obj_predictions = exists_obj * predictions[..., :C]
        obj_labels = exists_obj * labels[..., :C]
        class_loss = self.mse(
            torch.flatten(obj_predictions, end_dim=-2),
            torch.flatten(obj_labels, end_dim=-2)
        )
        # obj_predictions = predictions[..., :C]
        # obj_labels = labels[..., :C]
        # class_loss = torch.sum(
        #     -torch.log(torch.flatten(obj_predictions, end_dim=2)) *
        #     torch.flatten(obj_labels, end_dim=-2),
        #     dim=1
        # )
        # class_loss = torch.sum(class_loss) / torch.sum(class_loss != 0)

        # Total loss
        box_loss *= self.lambda_coord
        noobj_conf_loss *= self.lambda_noobj
        loss = 0
        if 'box' in losses:
            loss += box_loss
        if 'obj_conf' in losses:
            loss += obj_conf_loss
        if 'noobj_conf' in losses:
            loss += noobj_conf_loss
        if 'class' in losses:
            loss += class_loss
        return loss, box_loss, obj_conf_loss, noobj_conf_loss, class_loss

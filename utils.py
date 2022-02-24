import torch
import numpy as np
from collections import Counter
from PIL import Image, ImageDraw
from tqdm import tqdm

from params import S, B, C

def intersection_over_union(boxes_preds, boxes_labels):
    Aw, Ah = boxes_preds[..., 2:3], boxes_preds[..., 3:4]
    Bw, Bh = boxes_labels[..., 2:3], boxes_labels[..., 3:4]

    Ax1, Ay1 = boxes_preds[..., 0:1] - Aw / 2, boxes_preds[..., 1:2] - Ah / 2
    Bx1, By1 = boxes_labels[..., 0:1] - Bw / 2, boxes_labels[..., 1:2] - Bh / 2
    Ax2, Ay2 = boxes_preds[..., 0:1] + Aw / 2, boxes_preds[..., 1:2] + Ah / 2
    Bx2, By2 = boxes_labels[..., 0:1] + Bw / 2, boxes_labels[..., 1:2] + Bh / 2

    x1 = torch.max(Ax1, Bx1)
    y1 = torch.max(Ay1, By1)
    x2 = torch.min(Ax2, Bx2)
    y2 = torch.min(Ay2, By2)

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    union = abs((Ax2 - Ax1) * (Ay2 - Ay1)) + abs((Bx2 - Bx1) * (By2 - By1)) - intersection

    return intersection / (union + 1e-6)


def non_max_suppression(bboxes, iou_threshold, threshold):
    bboxes = [box for box in bboxes if box[1] > threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    ret = []

    while bboxes:
        chosen_box = bboxes.pop(0)
        bboxes = [
            box for box in bboxes
            if box[0] != chosen_box[0] or
            intersection_over_union(
                torch.tensor(chosen_box[2:]), torch.tensor(box[2:])
            ) < iou_threshold
        ]
        ret.append(chosen_box)

    return ret


def mean_average_precision(pred_boxes, true_boxes, iou_threshold=0.5):
    average_precisions = []
    for c in range(C):
        detections = []
        ground_truths = []

        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)

        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)

        amount_bboxes = Counter([gt[0] for gt in ground_truths])
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)

        detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)
        
        if total_true_bboxes == 0:
            continue

        for detection_idx, detection in enumerate(detections):
            ground_truth_img = [bbox for bbox in ground_truths if bbox[0] == detection[0]]
            best_iou = 0
            for idx, gt in enumerate(ground_truth_img):
                iou = intersection_over_union(torch.tensor(detection[3:]), torch.tensor(gt[3:]))
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou > iou_threshold:
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1
            else:
                FP[detection_idx] = 1

        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bboxes + 1e-6)
        precisions = torch.divide(TP_cumsum, (TP_cumsum + FP_cumsum + 1e-6))
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        average_precisions.append(torch.trapz(precisions, recalls))

    return sum(average_precisions) / len(average_precisions)


def plot_image(image, boxes):
    im = (np.array(image) * 255).astype(np.uint8)
    im = Image.fromarray(im)
    draw = ImageDraw.Draw(im)
    for box in boxes:
        box = box[2:]
        x, y = box[0] - box[2] / 2, box[1] - box[3] / 2
        w, h = box[2], box[3]
        x, y = x * im.width, y * im.height
        w, h = w * im.width, h * im.height
        draw.rectangle([(x, y), (x + w, y + h)], outline=(255, 255, 255))
    im.show()


def get_bboxes(loader, model, iou_threshold, threshold):
    all_pred_boxes = []
    all_true_boxes = []

    model.eval()
    train_idx = 0

    loop = tqdm(loader, desc='get bboxes', leave=False)
    for _, (x, labels) in enumerate(loop):
        with torch.no_grad():
            predictions = model(x)

        batch_size = x.shape[0]
        true_bboxes = cellboxes_to_boxes(labels)
        bboxes = cellboxes_to_boxes(predictions)

        for idx in range(batch_size):
            nms_boxes = non_max_suppression(
                bboxes[idx],
                iou_threshold=iou_threshold,
                threshold=threshold,
            )

            for nms_box in nms_boxes:
                all_pred_boxes.append([train_idx] + nms_box)

            for box in true_bboxes[idx]:
                if box[1] > threshold:
                    all_true_boxes.append([train_idx] + box)

            train_idx += 1

    model.train()
    return all_pred_boxes, all_true_boxes


def convert_cellboxes(predictions):
    predictions = predictions.to('cpu')
    batch_size = predictions.shape[0]
    predictions = predictions.reshape(batch_size, S, S, C + B * 5)

    bboxes1 = predictions[..., C+1:C+5]
    bboxes2 = predictions[..., C+6:C+10]
    scores = torch.cat((predictions[..., C].unsqueeze(0), predictions[..., C+5].unsqueeze(0)), dim=0)
    best_box = scores.argmax(0).unsqueeze(-1)
    best_boxes = bboxes1 * (1 - best_box) + best_box * bboxes2

    cell_indices = torch.arange(S).repeat(batch_size, S, 1).unsqueeze(-1)
    x = 1 / S * (best_boxes[..., :1] + cell_indices)
    y = 1 / S * (best_boxes[..., 1:2] + cell_indices.permute(0, 2, 1, 3))  # needs S = 7
    wh = 1 / S * best_boxes[..., 2:4]
    converted_bboxes = torch.cat((x, y, wh), dim=-1)
    predicted_class = predictions[..., :C].argmax(-1).unsqueeze(-1)
    best_confidence = torch.max(predictions[..., C], predictions[..., C+5]).unsqueeze(-1)

    converted_preds = torch.cat((predicted_class, best_confidence, converted_bboxes), dim=-1)
    return converted_preds


def cellboxes_to_boxes(out):
    converted_pred = convert_cellboxes(out).reshape(out.shape[0], S * S, -1)
    converted_pred[..., 0] = converted_pred[..., 0].long()
    all_bboxes = []

    for i in range(out.shape[0]):
        bboxes = []
        for bbox_idx in range(S * S):
            bboxes.append([x.item() for x in converted_pred[i, bbox_idx, :]])
        all_bboxes.append(bboxes)

    return all_bboxes


def save_checkpoint(state, filename="saves/noname.pth.tar"):
    torch.save(state, filename)
    print('Saved checkpoint.')


def load_checkpoint(checkpoint, model, optimizer):
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    print('Loaded checkpoint.')
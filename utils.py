import torch
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

from params import S, B, C

def intersection_over_union(boxA, boxB):
    """
    Calculates IoU between two boxes.

    Args:
        boxA: First box. Format for x, y, width, height doesn't matter, can be fraction of image or pixel count.
        boxB: Second box.
    
    Returns:
        The IoU between boxA and boxB.
    """

    Aw, Ah = boxA[..., 2:3], boxA[..., 3:4]
    Bw, Bh = boxB[..., 2:3], boxB[..., 3:4]

    Ax1, Ay1 = boxA[..., 0:1] - Aw / 2, boxA[..., 1:2] - Ah / 2
    Bx1, By1 = boxB[..., 0:1] - Bw / 2, boxB[..., 1:2] - Bh / 2
    Ax2, Ay2 = boxA[..., 0:1] + Aw / 2, boxA[..., 1:2] + Ah / 2
    Bx2, By2 = boxB[..., 0:1] + Bw / 2, boxB[..., 1:2] + Bh / 2

    x1 = torch.max(Ax1, Bx1)
    y1 = torch.max(Ay1, By1)
    x2 = torch.min(Ax2, Bx2)
    y2 = torch.min(Ay2, By2)

    intersection = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    union = abs((Ax2 - Ax1) * (Ay2 - Ay1)) + abs((Bx2 - Bx1) * (By2 - By1)) - intersection

    return intersection / (union + 1e-6)


def non_max_suppression(bboxes, iou_threshold, conf_threshold):
    """
    Filters out bounding boxes using non-max suppression.

    Args:
        bboxes: Bounding boxes to be filtered. Each box should contain confidence, class, x, y, width, height information, in order.
        iou_threshold: Minimum IoU between selected and remaining bounding box to remove it.
        conf_threshold: Minimum confidence required to keep bounding box.

    Returns:
        A list of remaining bounding boxes.
    """

    bboxes = [box for box in bboxes if box[1] > conf_threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    ret = []

    while len(bboxes) > 0:
        chosen_box = bboxes.pop(0)
        bboxes = [
            box for box in bboxes if
                box[0] != chosen_box[0] or
                intersection_over_union(torch.tensor(chosen_box[2:]), torch.tensor(box[2:])) < iou_threshold
        ]
        ret.append(chosen_box)

    return ret


def mean_average_precision(pred_boxes, true_boxes, iou_threshold=0.5):
    """
    Calculates the mean average precision between predicted and true bounding boxes.

    Args:
        pred_boxes: A list of predicted bounding boxes. Each box should contain image index, confidence, class, x, y, width, height information, in order.
        true_boxes: Ground truth bounding boxes with the same information format as pred_boxes.
        iou_threshold: Minimum IoU for a predicted bounding box to be resposible for a true box.
    
    Returns:
        The mean average precision, calculated with an approximated under-curve area without smoothing.
    """

    average_precisions = []
    for c in range(C):
        predicted_cboxes = []
        true_cboxes = []

        for box in pred_boxes:
            if box[1] == c:
                predicted_cboxes.append(box)

        for box in true_boxes:
            if box[1] == c:
                true_cboxes.append(box)

        img_boxes = {}
        for box in true_boxes:
            if box[0] in img_boxes:
                img_boxes[box[0]] += 1
            else:
                img_boxes[box[0]] = 1
        for box, count in img_boxes:
            img_boxes[box] = torch.zeros(count)

        predicted_cboxes.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros((len(predicted_cboxes)))
        FP = torch.zeros((len(predicted_cboxes)))

        if len(true_cboxes) == 0:
            continue

        for i, predicted_cbox in enumerate(predicted_cboxes):
            img_idx = predicted_cbox[0]
            img_true_cboxes = [box for box in true_cboxes if box[0] == img_idx]
            best_iou = 0
            best_idx = 0
            for j, true_cbox in enumerate(img_true_cboxes):
                iou = intersection_over_union(torch.tensor(predicted_cbox[3:]), torch.tensor(true_cbox[3:]))
                if iou > best_iou:
                    best_iou = iou
                    best_idx = j
            
            if best_iou > iou_threshold:
                if img_boxes[img_idx][best_idx] == 0:
                    TP[i] = 1
                    img_boxes[img_idx][best_idx] = 1
                else:
                    FP[i] = 1
            else:
                FP[i] = 1

        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (len(true_cboxes) + 1e-6)
        precisions = torch.divide(TP_cumsum, (TP_cumsum + FP_cumsum + 1e-6))

        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))

        average_precisions.append(torch.trapz(precisions, recalls))

    return sum(average_precisions) / len(average_precisions)


def plot_image(image, boxes):
    """
    Plots the image with bounding boxes and labels, shape names, and confidence.

    Args:
        image: The image to be plotted. All values should be between 0 and 1.
        boxes: A list of bounding boxes including their confidence and class prediction. Input should contain confidence, class, x, y, width, height information, in order. x, y, width, height values should be fractions of image width and height, bounded between 0 and 1.
    
    Returns:
        Nothing. Will plot and show the image with bounding boxes.
    """

    shape_names = ['circle', 'triangle', 'square', 'pentagon', 'hexagon', 'heptagon', 'octagon']
    im = (np.array(image) * 255).astype(np.uint8)
    im = Image.fromarray(im)
    draw = ImageDraw.Draw(im)
    for box in boxes:
        conf = box[0]
        shape = int(box[1])
        box = box[2:]
        x, y = box[0] - box[2] / 2, box[1] - box[3] / 2
        w, h = box[2], box[3]
        x, y = x * im.width, y * im.height
        w, h = w * im.width, h * im.height
        draw.rectangle([(x, y), (x + w, y + h)], outline=(255, 255, 255))
        draw.text((x + 5, y), str(shape) + " " + str(shape_names[shape]), fill=(255, 255, 255))
        draw.text((x + w - 20, y), str(conf), fill=(255, 255, 255))
    im.show()


def get_bboxes(loader, model, iou_threshold, conf_threshold):
    """
    Calculates predicted and true bounding boxes from dataloader. Runs non-maxmimum suppression to filter out predicted bounding boxes. Used to create list of all bounding boxes for mean average precision calculation.

    Args:
        loader: Pytorch dataloader containing images and labels.
        model: CNN used to predict bounding boxes from images.
        iou_threshold: Passed to non_max_suppression() as argument.
        conf_threshold: Passed to non_max_suppression() as argument.

    Returns:
        Two lists of bounding boxes, one for predicted and one for true. Each box contains image index, confidence, class, x, y, width, height, in order.
    """

    pred_boxes = []
    true_boxes = []

    model.eval()

    img_idx = 0

    loop = tqdm(loader, leave=False)
    for _, (img, labels) in enumerate(loop):
        with torch.no_grad():
            predictions = model(img)

        batch_size = img.shape[0]
        true_img_boxes = predictions_to_bboxes(labels)
        pred_img_boxes = predictions_to_bboxes(predictions)

        for i in range(batch_size):
            for cx in range(S):
                for cy in range(S):
                    for j in range(B):
                        pred_cboxes = non_max_suppression(pred_img_boxes[i, cx, cy, j], iou_threshold, conf_threshold)
                        true_cboxes = true_img_boxes[i, cx, cy, j]

                        for box in pred_cboxes:
                            pred_boxes.append([img_idx] + box)

                        for box in true_cboxes:
                            true_boxes.append([img_idx] + box)

            img_idx += 1
    model.train()

    return pred_boxes, true_boxes


def predictions_to_bboxes(predictions, S=S, B=B, C=C):
    """
    Converts CNN output to bounding boxes.

    Args:
        predictions: Output of CNN, has shape (batch_size, S * S * (C + B * 5))
    
    Returns:
        A pytorch tensor containing bounding box info of shape (batch_size, S, S, B, 6). Each box contains confidence, class, x, y, width, height information, in order. x, y, width, height are all fractions of the width or height of the image, so they're bounded between 0 and 1. Class is the class with the maximum probability over all C class probability predictions.
    """

    predictions = predictions.to('cpu')

    batch_size = predictions.shape[0]
    predictions = predictions.reshape(batch_size, S, S, C + B * 5)

    ret = torch.empty((batch_size, S, S, B, 6))
    for cx in range(S):
        for cy in range(S):
            cell = predictions[:, cx, cy, :]
            for i in range(B):
                idx_offset = C + 5 * i
                pred_class = torch.argmax(cell[:, :C], dim=1).unsqueeze(-1)
                conf = cell[:, idx_offset].unsqueeze(-1)
                x = 1 / S * (cell[:, idx_offset + 1].unsqueeze(-1) + cx)
                y = 1 / S * (cell[:, idx_offset + 2].unsqueeze(-1) + cy)
                w = cell[:, idx_offset + 3].unsqueeze(-1)
                h = cell[:, idx_offset + 4].unsqueeze(-1)

                box = torch.cat((conf, pred_class, x, y, w, h), dim=-1)
                print(box.shape)
                ret[:, cx, cy, i, :] = box

    return ret


def labels_to_bboxes(labels, S=S, B=B, C=C):
    """
    Converts ground truth labels to bounding boxes.

    Args:
        labels: Ground truth labels, has shape (batch_size, S * S * )
        predictions: Output of CNN, has shape (batch_size, S * S * (C + B * 5))
    
    Returns:
        A pytorch tensor containing bounding box info of shape (batch_size, S, S, B, 6). Each box contains confidence, class, x, y, width, height information, in order. x, y, width, height are all fractions of the width or height of the image, so they're bounded between 0 and 1. Class is the class with the maximum probability over all C class probability predictions.
    """

    predictions = predictions.to('cpu')

    batch_size = predictions.shape[0]
    predictions = predictions.reshape(batch_size, S, S, C + B * 5)

    ret = torch.empty((batch_size, S, S, B, 6))
    for cx in range(S):
        for cy in range(S):
            cell = predictions[:, cx, cy, :]
            for i in range(B):
                idx_offset = C + 5 * i
                pred_class = torch.argmax(cell[:, :C], dim=1).unsqueeze(-1)
                conf = cell[:, idx_offset].unsqueeze(-1)
                x = 1 / S * (cell[:, idx_offset + 1].unsqueeze(-1) + cx)
                y = 1 / S * (cell[:, idx_offset + 2].unsqueeze(-1) + cy)
                w = cell[:, idx_offset + 3].unsqueeze(-1)
                h = cell[:, idx_offset + 4].unsqueeze(-1)

                box = torch.cat((conf, pred_class, x, y, w, h), dim=-1)
                print(box.shape)
                ret[:, cx, cy, i, :] = box

    return ret


def save_checkpoint(state, filename="saves/noname.pth.tar"):
    torch.save(state, filename)
    print('Saved checkpoint.')


def load_checkpoint(checkpoint, model, optimizer):
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    print('Loaded checkpoint.')
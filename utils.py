import torch
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from os.path import exists

from load_config import p


class Losses:
    """
    Class storing individual components of Yolo loss function across multiple loss calculations.
    """
    def __init__(self):
        self.loss = []
        self.box_loss = []
        self.obj_conf_loss = []
        self.noobj_conf_loss = []
        self.class_loss = []

    def append(self, loss, box_loss, obj_conf_loss, noobj_conf_loss, class_loss):
        self.loss.append(loss)
        self.box_loss.append(box_loss)
        self.obj_conf_loss.append(obj_conf_loss)
        self.noobj_conf_loss.append(noobj_conf_loss)
        self.class_loss.append(class_loss)

    def means(self):
        mean_loss = sum(self.loss) / len(self.loss)
        mean_box_loss = sum(self.box_loss) / len(self.box_loss)
        mean_obj_conf_loss = sum(self.obj_conf_loss) / len(self.obj_conf_loss)
        mean_noobj_conf_loss = sum(self.noobj_conf_loss) / len(self.noobj_conf_loss)
        mean_class_loss = sum(self.class_loss) / len(self.class_loss)
        return mean_loss, mean_box_loss, mean_obj_conf_loss, mean_noobj_conf_loss, mean_class_loss


def intersection_over_union(box1, box2):
    """
    Calculates IoU between two boxes.

    Args:
        box1: First box. Format for x, y, width, height doesn't matter, can be fraction of image or pixel count.
        box2: Second box.
    
    Returns:
        The IoU between boxA and boxB.
    """

    if not torch.is_tensor(box1):
        box1 = torch.tensor(box1)
    if not torch.is_tensor(box2):
        box2 = torch.tensor(box2)
    
    w1, h1 = box1[..., 2:3], box1[..., 3:4]
    w2, h2 = box2[..., 2:3], box2[..., 3:4]

    x1_left, y1_lower = box1[..., 0:1] - w1 / 2, box1[..., 1:2] - h1 / 2
    x2_left, y2_lower = box2[..., 0:1] - w2 / 2, box2[..., 1:2] - h2 / 2
    x1_right, y1_higher = box1[..., 0:1] + w1 / 2, box1[..., 1:2] + h1 / 2
    x2_right, y2_higher = box2[..., 0:1] + w2 / 2, box2[..., 1:2] + h2 / 2

    x1 = torch.max(x1_left, x2_left)
    y1 = torch.max(y1_lower, y2_lower)
    x2 = torch.min(x1_right, x2_right)
    y2 = torch.min(y1_higher, y2_higher)

    intersection = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    union = abs((x1_right - x1_left) * (y1_higher - y1_lower)) + \
        abs((x2_right - x2_left) * (y2_higher - y2_lower)) - intersection

    return intersection / (union + 1e-6)


def non_max_suppression(bboxes, iou_threshold, conf_threshold):
    """
    Filters out bounding boxes using non-max suppression.

    Args:
        bboxes: Bounding boxes to be filtered. Each box should contain confidence, class, x, y, width, height
        information, in order.
        iou_threshold: Minimum IoU between selected and remaining bounding box to remove it.
        conf_threshold: Minimum confidence required to keep bounding box.

    Returns:
        A list of remaining bounding boxes.
    """

    bboxes = [box for box in bboxes if box[0] > conf_threshold]
    bboxes = sorted(bboxes, key=lambda x: x[0], reverse=True)
    ret = []

    while len(bboxes) > 0:
        chosen_box = bboxes.pop(0)
        bboxes = [
            box for box in bboxes if
            box[1] != chosen_box[1] or
            intersection_over_union(chosen_box[2:].clone().detach(), box[2:].clone().detach()) < iou_threshold
        ]
        ret.append(chosen_box)

    return ret


def mean_average_precision(pred_boxes, true_boxes, iou_threshold=0.5, plot_curve=False, use_class=True):
    """
    Calculates the mean average precision between predicted and true bounding boxes.

    Args:
        pred_boxes: A list of predicted bounding boxes. Each box should contain image index, confidence, class, x, y,
        width, height information, in order.
        true_boxes: Ground truth bounding boxes with the same information format as pred_boxes.
        iou_threshold: Minimum IoU for a predicted bounding box to be responsible for a true box.
        plot_curve: Display the precision-recall curve.
        use_class: Whether to consider classes when calculating mAP. When false, calculations can assign any predicted
        bounding box to a ground truth box, regardless of the predicted class associated with it.
    
    Returns:
        The mean average precision, calculated with an approximated under-curve area without smoothing.
    """

    average_precisions = []
    for c in range(p.C if use_class else 1):
        pred_class_boxes = []
        true_class_boxes = []

        for box in pred_boxes:
            if box[2] == c or not use_class:
                pred_class_boxes.append(box)

        for box in true_boxes:
            if box[2] == c or not use_class:
                true_class_boxes.append(box)

        img_boxes = {}
        for box in true_class_boxes:
            if box[0] in img_boxes:
                img_boxes[box[0]] += 1
            else:
                img_boxes[box[0]] = 1
        for img in img_boxes:
            img_boxes[img] = torch.zeros(img_boxes[img])

        pred_class_boxes.sort(key=lambda x: x[2], reverse=True)
        true_positive = torch.zeros((len(pred_class_boxes)))
        false_positive = torch.zeros((len(pred_class_boxes)))

        if len(true_class_boxes) == 0:
            continue

        for i, pred_box in enumerate(pred_class_boxes):
            img_idx = pred_box[0]
            true_img_boxes = [box for box in true_class_boxes if box[0] == img_idx]
            best_iou = 0
            best_idx = 0
            for j, true_box in enumerate(true_img_boxes):
                iou = intersection_over_union(torch.tensor(pred_box[3:]), torch.tensor(true_box[3:]))
                if iou > best_iou:
                    best_iou = iou
                    best_idx = j
            
            if best_iou > iou_threshold:
                if img_boxes[img_idx][best_idx] == 0:
                    true_positive[i] = 1
                    img_boxes[img_idx][best_idx] = 1
                else:
                    false_positive[i] = 1
            else:
                false_positive[i] = 1

        true_positive_cumsum = torch.cumsum(true_positive, dim=0)
        false_positive_cumsum = torch.cumsum(false_positive, dim=0)
        recalls = true_positive_cumsum / (len(true_class_boxes) + 1e-6)
        precisions = torch.divide(true_positive_cumsum, (true_positive_cumsum + false_positive_cumsum + 1e-6))

        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))

        if plot_curve:
            plt.plot(recalls.detach().cpu().numpy(), precisions.detach().cpu().numpy(), label=str(c))

        average_precisions.append(torch.trapz(precisions, recalls))

    if plot_curve:
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.show()

    return sum(average_precisions) / len(average_precisions)


def plot_image(image, boxes):
    """
    Plots the image with bounding boxes and labels, shape names, and confidence.

    Args:
        image: The image to be plotted. All values should be between 0 and 1.
        boxes: A list of bounding boxes including their confidence and class prediction. Input should contain
        confidence, class, x, y, width, height information, in order. x, y, width, height values should be fractions
        of image width and height, bounded between 0 and 1.
    
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
        draw.rectangle(((x, y), (x + w, y + h)), outline=(255, 255, 255))
        draw.text((x + 5, y), str(shape) + " " + str(shape_names[shape]), fill=(255, 255, 255))
        draw.text((x + w - 30, y), "{:.3f}".format(conf), fill=(255, 255, 255))
    im.show()


def load_losses():
    _, _, losses = load_predictions()
    return losses


def get_bboxes(model, iou_threshold, conf_threshold):
    """
    Calculates predicted and true bounding boxes from dataloader. Runs non-maximum suppression to filter out predicted
    bounding boxes. Used to create list of all bounding boxes for mean average precision calculation.

    Args:
        model: CNN used to predict bounding boxes from images.
        iou_threshold: Passed to non_max_suppression() as argument.
        conf_threshold: Passed to non_max_suppression() as argument.

    Returns:
        Two lists of bounding boxes, one for predicted and one for true. Each box contains image index, confidence,
        class, x, y, width, height, in order.
    """

    pred_boxes = []
    true_boxes = []
    model.eval()
    img_idx = 0

    predictions, labels, _ = load_predictions()
  
    dataset_size = predictions.shape[0]
    pred_batch_boxes = predictions_to_bboxes(torch.from_numpy(predictions))
    true_batch_boxes = labels_to_bboxes(torch.from_numpy(labels))

    for i in range(dataset_size):
        pred_img_boxes = pred_batch_boxes[i, :].reshape(-1, 6)
        pred_img_boxes = non_max_suppression(pred_img_boxes, iou_threshold, conf_threshold)
        for box in pred_img_boxes:
            pred_boxes.append([img_idx] + box.tolist())

        true_img_boxes = true_batch_boxes[i, :].reshape(-1, 6)
        for box in true_img_boxes:
            if box[0] == 1:
                true_boxes.append([img_idx] + box.tolist())

        img_idx += 1

    return pred_boxes, true_boxes


def predictions_to_bboxes(predictions):
    """
    Converts CNN output to bounding boxes.

    Args:
        predictions: Output of CNN, has shape (batch_size, S, S, C + B * 5).
    
    Returns:
        A pytorch tensor containing bounding box info of shape (batch_size, S, S, B, 6). Each box contains confidence,
        class, x, y, width, height information, in order. x, y, width, height are all fractions of the width or height
        of the image, so they're bounded between 0 and 1. Class is the class with the maximum probability over all C
        class probability predictions.
    """

    predictions = predictions.to('cpu')

    batch_size = predictions.shape[0]
    ret = torch.empty((batch_size, p.S, p.S, p.B, 6))
    for cx in range(p.S):
        for cy in range(p.S):
            cell = predictions[:, cx, cy, :]
            for i in range(p.B):
                idx_offset = p.C + 5 * i
                pred_class = torch.argmax(cell[:, :p.C], dim=1).unsqueeze(-1)
                conf = cell[:, idx_offset].unsqueeze(-1)
                x = 1 / p.S * (cell[:, idx_offset+1].unsqueeze(-1) + cx)
                y = 1 / p.S * (cell[:, idx_offset+2].unsqueeze(-1) + cy)
                w = cell[:, idx_offset+3].unsqueeze(-1)
                h = cell[:, idx_offset+4].unsqueeze(-1)

                box = torch.cat((conf, pred_class, x, y, w, h), dim=-1)
                ret[:, cx, cy, i, :] = box

    return ret


def labels_to_bboxes(labels):
    """
    Converts ground truth labels to bounding boxes.

    Args:
        labels: Ground truth labels, has shape (batch_size, S, S, C + 5)
    
    Returns:
        A pytorch tensor containing bounding box info of shape (batch_size, S, S, 6). Each box contains confidence,
        class, x, y, width, height information, in order. x, y, width, height are all fractions of the width or height
        of the image, so they're bounded between 0 and 1. Class is the class with the maximum probability over all C
        class probability predictions.
    """

    labels = labels.to('cpu')

    batch_size = labels.shape[0]
    ret = torch.empty((batch_size, p.S, p.S, 6))
    for cx in range(p.S):
        for cy in range(p.S):
            cell = labels[:, cx, cy, :]
            pred_class = torch.argmax(cell[:, :p.C], dim=1).unsqueeze(-1)
            conf = cell[:, p.C].unsqueeze(-1)
            x = 1 / p.S * (cell[:, p.C+1].unsqueeze(-1) + cx) * conf
            y = 1 / p.S * (cell[:,  p.C+2].unsqueeze(-1) + cy) * conf
            w = cell[:, p.C+3].unsqueeze(-1)
            h = cell[:, p.C+4].unsqueeze(-1)

            box = torch.cat((conf, pred_class, x, y, w, h), dim=-1)
            ret[:, cx, cy, :] = box

    return ret


def save_checkpoint(state, filename="saves/noname.pth.tar"):
    torch.save(state, filename)
    print('Saved checkpoint.')


def load_checkpoint(checkpoint, model, optimizer):
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    print('Loaded checkpoint.')


def save_predictions(dataloader, model, loss_fn):
    """
    Saves unfiltered output of YOLO model along with losses and the ground truth labels corresponding
    to each prediction.

    Args:
        dataloader: Pytorch dataloader containing images and labels.
        model: CNN used to predict bounding boxes from images.
        loss_fn: Loss function used to calculate loss between predictions and labels.
    
    Returns:
        Nothing. Saves predictions, losses, and labels to .npz file.
    """

    model.eval()
    losses = Losses()
    predictions = torch.empty(0).to(p.device)
    labels = torch.empty(0).to(p.device)
    for _, (x, y) in enumerate(dataloader):
        x, y = x.to(p.device), y.to(p.device)
        with torch.no_grad():
            out = model(x)
        loss, box_loss, obj_conf_loss, noobj_conf_loss, class_loss = loss_fn(out, y)

        losses.append(loss.item(), box_loss.item(), obj_conf_loss.item(), noobj_conf_loss.item(), class_loss.item())
        predictions = torch.cat((predictions, out), dim=0)
        labels = torch.cat((labels, y), dim=0)

    losses = np.array(losses.means())
    predictions = predictions.cpu().numpy()
    labels = labels.cpu().numpy()

    np.savez(p.predictions_filepath, predictions=predictions, losses=losses, labels=labels)
    model.train()
    if p.verbose:
        print('Saved predictions, losses, and labels in %s.' % p.predictions_filepath)


def load_predictions():
    if not exists(p.predictions_filepath):
        print('ERROR: Missing predictions save file. Run save_predictions(loader, model, loss_fn) first.')
        return
    data = np.load(p.predictions_filepath)
    return data['predictions'], data['labels'], data['losses']

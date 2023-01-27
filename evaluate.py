import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

from run import get_transform
from utils import (
    mean_average_precision,
    get_bboxes,
    load_checkpoint,
    save_predictions,
    predictions_to_bboxes,
    non_max_suppression,
    plot_image,
    load_losses
)

from dataset import Dataset
from model import Yolo
from loss import YoloLoss

import config
from config import S, C


def visualize(model, dataloader):
    for x, _ in dataloader:
        x = x.to(config.device)
        pred_batch_boxes = predictions_to_bboxes(model(x))
        for i in range(config.batch_size):
            pred_img_boxes = pred_batch_boxes[i, :].reshape(-1, 6)
            pred_img_boxes = non_max_suppression(pred_img_boxes, iou_threshold=0.5, conf_threshold=0.4)
            pred_img_boxes = [box.tolist() for box in pred_img_boxes]
            input('>')
            plot_image(x[i].permute(1, 2, 0).to('cpu'), pred_img_boxes)


def plot_class_heatmap(model, dataloader, n):
    """
    Plots pre-NMS class heatmap, displaying which cells in the grid have higher confidence and probabilities for
    each class.

    Args:
        model: Pytorch YOLO model.
        dataloader: Pytorch dataloader used to train model.
        n: Number of images to analyze.
    """

    fig, axs = plt.subplots(2 * n, 6)
    it = iter(dataloader)
    for i in range(n):
        x, y = next(it)
        x = x[0:1, ...]
        y = y[0:1, ...]
        bboxes = []
        x, y = x.to(config.device), y.to(config.device)
        with torch.no_grad():
            out = model(x)

        img = np.moveaxis(x[0].numpy(), 0, -1)
        
        pred_batch_boxes = predictions_to_bboxes(out)
        pred_img_boxes = pred_batch_boxes[0, :].reshape(-1, 6)
        for box in pred_img_boxes:
            bboxes.append(box.tolist())

        bboxes = non_max_suppression(pred_img_boxes, iou_threshold=0.5, conf_threshold=0.4)
        bboxes_to_heatmap(bboxes, axs, i)

        axs[2 * i, C].imshow(img)
        axs[2 * i, C].axis('off')
        axs[2 * i + 1, C].axis('off')
    
    fig.tight_layout()
    plt.show()


def bboxes_to_heatmap(bboxes, axs, i):
    for c in range(C):
        heatmap = np.zeros((S, S))
        for box in bboxes:
            if box[1] == c:
                sx = int((box[2] + box[4] / 2) * S)
                sy = int((box[3] + box[5] / 2) * S)
                if 0 <= sx < S and 0 <= sy < S:
                    heatmap[sx, sy] += box[0]
        axs[2 * i + 1, c].imshow(heatmap, cmap='binary')
        axs[2 * i + 1, c].axis('off')


def main():
    print(f'Load file: {config.load_model_file}')

    transform = get_transform()

    dataset = Dataset(
        'shape', 'test.csv',
        transform=transform,
    )
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        shuffle=True,
        drop_last=True
    )

    print('Created datasets and dataloaders.')

    model = Yolo().to(config.device)
    if config.optimizer['name'] == 'adam':
        optim = torch.optim.Adam(model.parameters(), lr=config.optimizer['learning_rate'],
                                 weight_decay=config.optimizer['weight_decay'])
    elif config.optimizer['name'] == 'sgd':
        optim = torch.optim.SGD(model.parameters(), lr=config.optimizer['learning_rate'], momentum=config.optimizer['momentum'],
                                weight_decay=config.optimizer['weight_decay'])
    else:
        print('Invalid optimizer.')
        optim = None
    loss_fn = YoloLoss()

    load_checkpoint(torch.load(config.load_model_file, map_location=torch.device(config.device)), model, optim)

    # Save predictions
    save_predictions(dataloader, model, loss_fn)

    # Calculate mAP and mean loss
    conf_threshold = 0.1
    while conf_threshold <= 0.5:
        print(conf_threshold)
        mean_loss = load_losses()[0]
        pred_boxes, target_boxes = get_bboxes(
            model, iou_threshold=0.5, conf_threshold=conf_threshold
        )
        mean_avg_prec = mean_average_precision(
            pred_boxes, target_boxes, iou_threshold=0.5, plot_curve=False
        )
        classless_mean_avg_prec = mean_average_precision(
            pred_boxes, target_boxes, iou_threshold=0.5, plot_curve=False, use_class=False
        )
        print(f'mAP: {mean_avg_prec}')
        print(f'No-class mAP: {classless_mean_avg_prec}')
        print(f'Mean loss: {mean_loss}')
        print()

        conf_threshold += 0.05

    # Visualize predictions
    # visualize(model, dataloader)

    # Visualize class heatmap
    # plot_class_heatmap(model, dataloader, 3)

if __name__ == '__main__':
    main()

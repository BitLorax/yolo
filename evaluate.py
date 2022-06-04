import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# from pytorch_grad_cam import EigenCAM
# from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
# from pytorch_grad_cam.utils.image import show_cam_on_image

from utils import (
    mean_average_precision,
    get_bboxes,
    load_checkpoint,
    save_predictions,
    predictions_to_bboxes,
    non_max_suppression,
    plot_image,
    get_losses
)

from dataset import Dataset
from model import Yolo
from loss import YoloLoss
from params import *


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, im, bboxes):
        for t in self.transforms:
            im, bboxes = t(im), bboxes
        return im, bboxes

if selected_dataset == 'voc':
    transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor()])
elif selected_dataset[0:5] == 'shape':
    transform = Compose([transforms.ToTensor()])
else:
    print('Invalid dataset configuration.')


def visualize(dataloader):
    for x, _ in dataloader:
        x = x.to(device)
        pred_batch_boxes = predictions_to_bboxes(model(x))
        for i in range(batch_size):
            pred_img_boxes = pred_batch_boxes[i, :].reshape(-1, 6)
            pred_img_boxes = non_max_suppression(pred_img_boxes, iou_threshold=0.5, conf_threshold=0.4)
            pred_img_boxes = [box.tolist() for box in pred_img_boxes]
            input('>')
            plot_image(x[i].permute(1, 2, 0).to('cpu'), pred_img_boxes)


def plot_class_heatmap(dataloader, n):
    """
    Plots pre-NMS class heatmap, displaying which cells in the grid have higher confidence and probabilities for each class.

    Args:
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
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            out = model(x)

        img = np.moveaxis(x[0].numpy(), 0, -1)
        
        pred_batch_boxes = predictions_to_bboxes(out)
        pred_img_boxes = pred_batch_boxes[0, :].reshape(-1, 6)
        for box in pred_img_boxes:
            bboxes.append(box.tolist())

        for c in range(C):
            heatmap = np.zeros((S, S))
            for box in bboxes:
                if box[1] == c:
                    sx = (int)((box[2] + box[4] / 2) * S)
                    sy = (int)((box[3] + box[5] / 2) * S)
                    if sx >= 0 and sx < S and sy >= 0 and sy < S:
                        heatmap[sx, sy] += box[0]
            axs[2 * i, c].imshow(heatmap, cmap='binary')
            axs[2 * i, c].axis('off')
        
        bboxes = non_max_suppression(pred_img_boxes, iou_threshold=0.5, conf_threshold=0.4)
        for c in range(C):
            heatmap = np.zeros((S, S))
            for box in bboxes:
                if box[1] == c:
                    sx = (int)((box[2] + box[4] / 2) * S)
                    sy = (int)((box[3] + box[5] / 2) * S)
                    if sx >= 0 and sx < S and sy >= 0 and sy < S:
                        heatmap[sx, sy] += box[0]
            axs[2 * i + 1, c].imshow(heatmap, cmap='binary')
            axs[2 * i + 1, c].axis('off')

        axs[2 * i, C].imshow(img)
        axs[2 * i, C].axis('off')
        axs[2 * i + 1, C].axis('off')
    
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    print(f'Running on dataset: {selected_dataset}')
    print(f'Load file: {load_model_file}')
    print(f'Data from: {test_data_csv}')

    dataset = Dataset(
        selected_dataset,
        test_data_csv,
        transform=transform,
    )
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
        drop_last=True
    )

    print('Created datasets and dataloaders.')

    model = Yolo().to(device)
    if optimizer == 'adam':
        optim = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer == 'sgd':
        optim = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    else:
        print('Invalid optimizer.')
        optim = None
    loss_fn = YoloLoss()

    load_checkpoint(torch.load(load_model_filepath, map_location=torch.device(device)), model, optim)

    # Save predictions
    save_predictions(dataloader, model, loss_fn)

    # Calculate mAP and mean loss
    conf_threshold = 0.1
    while conf_threshold <= 0.5:
        print(conf_threshold)
        mean_loss = get_losses()[0]
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
    # visualize(dataloader)

    # Visualize class heatmap
    # plot_class_heatmap(dataloader, 3)

    # Visualize Grad-CAM (NOT WORKING)
    # for name, layer in model.named_modules():
    #     print(name, layer)
    # target_layers = [model.darknet]
    # cam = EigenCAM(model, target_layers=target_layers, use_cuda=False)

    # x, y = next(iter(dataloader))
    # x = x[0:1, ...]
    # y = y[0:1, ...]

    # grayscale_cam = cam(input_tensor=x)
    # grayscale_cam = grayscale_cam[0, :]
    # vis = show_cam_on_image(x[0], grayscale_cam, use_rgb=True)

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from utils import (
    mean_average_precision,
    get_bboxes,
    load_checkpoint,

    predictions_to_bboxes,
    non_max_suppression,
    plot_image
)
import utils_old

from dataset import Dataset
from model import Yolo
# from loss import YoloLoss
from params import *

from loss_old import YoloLoss

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, im, bboxes):
        for t in self.transforms:
            im, bboxes = t(im), bboxes
        return im, bboxes

if selected_dataset == 'voc':
    transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor()])
elif selected_dataset == 'shape':
    transform = Compose([transforms.ToTensor()])
else:
    print('Invalid dataset configuration.')


def visualize(dataloader):
    for x, _ in dataloader:
        x = x.to(device)
        bboxes = predictions_to_bboxes(model(x))
        bboxes = non_max_suppression(bboxes[0], iou_threshold=0.5, conf_threshold=0.4)
        input()
        plot_image(x[0].permute(1, 2, 0).to('cpu'), bboxes)


if __name__ == '__main__':
    print(f'Running on dataset: {selected_dataset}')
    print(f'Load file: {load_model_file}')
    print(f'Data from: {data_csv}')
    print()
    input()

    dataset = Dataset(
        selected_dataset,
        data_csv,
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

    load_checkpoint(torch.load(load_model_file, map_location=torch.device(device)), model, optim)

    pred_boxes, target_boxes, mean_loss = get_bboxes(
        dataloader, model, iou_threshold=0.5, conf_threshold=0.4, get_loss=True, loss_fn=loss_fn
    )
    mean_avg_prec = mean_average_precision(
        pred_boxes, target_boxes, iou_threshold=0.5, plot_curve=False
    )
    print(f'mAP: {mean_avg_prec}')
    print(f'old mAP: {utils_old.mean_average_precision(pred_boxes, target_boxes, iou_threshold=0.5)}')
    print(f'Mean loss: {mean_loss}')

    # print('Beginning visualization.')
    # visualize(dataloader)
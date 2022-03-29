
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np 

from dataset import Dataset
from model import Yolo
from loss import YoloLoss
from params import *

from utils import labels_to_bboxes, predictions_to_bboxes
from utils_old import convert_cellboxes


def test_predictions_to_bboxes():
    batch_size = 2
    S = 2
    B = 2
    C = 3

    print(f'batch_size: {batch_size}')
    print(f'S: {S}')
    print(f'B: {B}')
    print(f'C: {C}')

    predictions = [
        [
            [
                [0.1, 0.2, 0.3, 0.9, 0.1, 0.1, 0.5, 0.5, 0.9, 0.1, 0.1, 0.5, 0.5],
                [0.1, 0.2, 0.3, 0.9, 0.1, 0.1, 0.5, 0.5, 0.9, 0.1, 0.1, 0.5, 0.5]
            ],
            [
                [0.1, 0.2, 0.3, 0.9, 0.1, 0.1, 0.5, 0.5, 0.9, 0.1, 0.1, 0.5, 0.5],
                [0.1, 0.2, 0.3, 0.9, 0.1, 0.1, 0.5, 0.5, 0.9, 0.1, 0.1, 0.5, 0.5]
            ]
        ],
        [
            [
                [0.5, 0.4, 0.3, 0.8, 0.6, 0.6, 0.2, 0.2, 0.8, 0.6, 0.6, 0.2, 0.2],
                [0.5, 0.4, 0.3, 0.8, 0.6, 0.6, 0.2, 0.2, 0.8, 0.6, 0.6, 0.2, 0.2]
            ],
            [
                [0.5, 0.4, 0.3, 0.8, 0.6, 0.6, 0.2, 0.2, 0.8, 0.6, 0.6, 0.2, 0.2],
                [0.5, 0.4, 0.3, 0.8, 0.6, 0.6, 0.2, 0.2, 0.8, 0.6, 0.6, 0.2, 0.2]
            ]
        ]
    ]
    predictions = torch.tensor(predictions)

    bboxes = predictions_to_bboxes(predictions, S=S, B=B, C=C)
    print(bboxes)
    print(bboxes.shape)


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, im, bboxes):
        for t in self.transforms:
            im, bboxes = t(im), bboxes
        return im, bboxes

def test_loss():
    torch.manual_seed(0)
    np.random.seed(0)

    model = Yolo().to(device)
    loss_fn = YoloLoss()
    if selected_dataset == 'voc':
        transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor()])
    elif selected_dataset == 'shape':
        transform = Compose([transforms.ToTensor()])
    else:
        print('Invalid dataset configuration.')
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

    x, y = next(iter(dataloader))   
    x = x[0:1, ...]
    y = y[0:1, ...]
    out = model(x)
    loss = loss_fn(out, y)
    print(loss.item())

def test_prediction_conversion():
    torch.manual_seed(0)
    np.random.seed(0)

    model = Yolo().to(device)
    if selected_dataset == 'voc':
        transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor()])
    elif selected_dataset == 'shape':
        transform = Compose([transforms.ToTensor()])
    else:
        print('Invalid dataset configuration.')
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

    x, y = next(iter(dataloader))   
    x = x[0:1, ...]
    y = y[0:1, ...]
    out = model(x)
    out_boxes_1 = predictions_to_bboxes(out)
    out_boxes_2 = convert_cellboxes(out)
    print(out_boxes_1[0, 0, ...])
    print(out_boxes_2[0, 0, ...])


if __name__ == '__main__':
    test_prediction_conversion()

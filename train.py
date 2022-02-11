import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from tqdm import tqdm

from utils import (
    mean_average_precision,
    get_bboxes,
    save_checkpoint,
    load_checkpoint,

    cellboxes_to_boxes,
    non_max_suppression,
    plot_image
)

from dataset import VOCDataset
from model import Yolo
from loss import YoloLoss
from params import *

seed = 123
torch.manual_seed(seed)

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, im, bboxes):
        for t in self.transforms:
            im, bboxes = t(im), bboxes
        return im, bboxes

if dataset == 'voc':
    transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor()])
elif dataset == 'shape':
    transform = Compose([transforms.ToTensor()])
else:
    print('Invalid dataset configuration.')


def train(train_loader, model, optim, loss_fn):
    loop = tqdm(train_loader, leave=True)
    mean_loss = []

    for _, (x, y) in enumerate(loop):
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = loss_fn(out, y)
        mean_loss.append(loss.item())
        for param in model.parameters():
            param.grad = None
        loss.backward()
        optim.step()

        loop.set_postfix()
    
    print(f'Mean loss: {sum(mean_loss)/len(mean_loss)}')


def visualize(train_loader):
    for x, _ in train_loader:
        x = x.to(device)
        bboxes = cellboxes_to_boxes(model(x))
        bboxes = non_max_suppression(bboxes[0], iou_threshold=0.5, threshold=0.4)
        plot_image(x[0].permute(1, 2, 0).to('cpu'), bboxes)


if __name__ == '__main__':
    print(f'Running on dataset: {dataset}')
    print(f'Save file: {load_model_file}')
    print(f'Training from: {train_csv}')
    print()

    train_dataset = VOCDataset(
        dataset,
        train_csv,
        transform=transform,
    )
    test_dataset = VOCDataset(
        dataset,
        'test.csv',
        transform=transform,
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
        drop_last=True
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
        drop_last=True
    )

    print('Created datasets and dataloaders.')

    model = Yolo().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn = YoloLoss()

    if load_model:
        load_checkpoint(torch.load(load_model_file), model, optim)
    
    print('Created model, optimizer, and loss function.')

    for epoch in range(epochs):
        print(f'Epoch: {epoch}')

        train(train_loader, model, optim, loss_fn)

        if visualize_preds:
            visualize(train_loader)
        pred_boxes, target_boxes = get_bboxes(
            train_loader, model, iou_threshold=0.5, threshold=0.4
        )
        mean_avg_prec = mean_average_precision(
            pred_boxes, target_boxes, iou_threshold=0.5
        )
        print(f'Train mAP: {mean_avg_prec}')

        if mean_avg_prec > 0.9 or (epoch + 1) % 5 == 0:
            checkpoint = {
                'state_dict': model.state_dict(),
                'optimizer': optim.state_dict()
            }
            save_checkpoint(checkpoint, filename=load_model_file)
            import time
            time.sleep(10)

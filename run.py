import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import os

from tqdm import tqdm

from utils import (
    get_bboxes,
    mean_average_precision,
    save_checkpoint,
    load_checkpoint,
)

from dataset import Dataset
from model import Yolo
from loss import YoloLoss
from params import *

import wandb


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


def train(dataloader, model, optim, loss_fn):
    loop = tqdm(dataloader, leave=True)
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
    
    if enable_wandb:
        wandb.log({"loss": loss.item()})
    print(f'Training loss: {sum(mean_loss) / len(mean_loss)}')


def test(dataloader, model, loss_fn):
    pred_boxes, target_boxes, mean_loss = get_bboxes(
        dataloader, model, iou_threshold=0.5, conf_threshold=0.4, get_loss=True, loss_fn=loss_fn
    )
    mean_avg_prec = mean_average_precision(
        pred_boxes, target_boxes, iou_threshold=0.5, plot_curve=False
    )

    if enable_wandb:
        wandb.log({"validation loss": mean_loss})
        wandb.log({"mAP": mean_avg_prec})

    print(f'Validation loss: {mean_loss}')
    print(f'mAP: {mean_avg_prec}')


if __name__ == '__main__':
    print(f'Config id: {config_id}')
    print()
    print(f'Running on dataset: {selected_dataset}')
    print(f'Save file: {save_model_file}')
    print(f'Epochs: {epochs}')
    if resume_run:
        print('Resuming previous run. ')
        print(f'Load file: {load_model_file}')
    print()
    input()

    config = {
    'learning_rate': learning_rate,
    'epochs': epochs,
    'batch_size': batch_size,
    'weight_decay': weight_decay,
    'optimizer': optimizer
    }
    if optimizer == 'sgd':
        config['momentum'] = momentum
    if enable_wandb:
        if resume_run:
            wandb.init(project='yolo', entity='willjhliang', config=config, id=resume_run_id, resume='must')
        else:
            wandb.init(project='yolo', entity='willjhliang', config=config)

    train_dataset = Dataset(
        selected_dataset,
        train_data_csv,
        transform=transform,
    )
    test_dataset = Dataset(
        selected_dataset,
        test_data_csv,
        transform=transform,
    )
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
        drop_last=True
    )
    test_dataloader = DataLoader(
        dataset=test_dataset,
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

    if resume_run:
        load_checkpoint(torch.load(load_model_file), model, optim)
    
    print('Created model, optimizer, and loss function.')

    if not os.path.exists('saves'):
        os.makedirs('saves')

    for epoch in range(epochs):
        print(f'Epoch: {epoch}')

        train(train_dataloader, model, optim, loss_fn)
        test(test_dataloader, model, loss_fn)

        checkpoint = {
            'state_dict': model.state_dict(),
            'optimizer': optim.state_dict()
        }
        save_checkpoint(checkpoint, filename=save_model_file)

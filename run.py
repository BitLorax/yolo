import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import os

from tqdm import tqdm

from utils import (
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
            wandb.watch(model)

        loop.set_postfix()
    
    print(f'Mean loss: {sum(mean_loss) / len(mean_loss)}')


if __name__ == '__main__':
    print(f'Running on dataset: {selected_dataset}')
    print(f'Save file: {save_model_file}')
    print(f'Data from: {data_csv}')
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

    seed = 123
    torch.manual_seed(seed)

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
        optim = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)  # REPALCE WITH SGD
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

        train(dataloader, model, optim, loss_fn)

        checkpoint = {
            'state_dict': model.state_dict(),
            'optimizer': optim.state_dict()
        }
        save_checkpoint(checkpoint, filename=save_model_file)

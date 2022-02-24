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

        wandb.log({"loss": loss.item()})
        wandb.watch(model)

        loop.set_postfix()
    
    print(f'Mean loss: {sum(mean_loss) / len(mean_loss)}')


def visualize(dataloader):
    for x, _ in dataloader:
        x = x.to(device)
        bboxes = cellboxes_to_boxes(model(x))
        bboxes = non_max_suppression(bboxes[0], iou_threshold=0.5, threshold=0.4)
        plot_image(x[0].permute(1, 2, 0).to('cpu'), bboxes)


if __name__ == '__main__':
    config = {
    'learning_rate': learning_rate,
    'epochs': epochs,
    'batch_size': batch_size,
    'weight_decay': weight_decay,
    'optimizer': optimizer
    }
    if optimizer == 'sgd':
        config['momentum'] = momentum
    wandb.init(project='yolo', entity='willjhliang', config=config)

    seed = 123
    torch.manual_seed(seed)

    print(f'Running on dataset: {selected_dataset}')
    print(f'Save file: {load_model_file}')
    print(f'Data from: {data_csv}')
    print()

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

    if load_model:
        load_checkpoint(torch.load(load_model_file), model, optim)
    
    print('Created model, optimizer, and loss function.')

    for epoch in range(epochs):
        print(f'Epoch: {epoch}')

        train(dataloader, model, optim, loss_fn)


        if visualize_preds:
            visualize(dataloader)
        
        if (epoch + 1) % 5 == 0:
            pred_boxes, target_boxes = get_bboxes(
                dataloader, model, iou_threshold=0.5, threshold=0.4
            )
            mean_avg_prec = mean_average_precision(
                pred_boxes, target_boxes, iou_threshold=0.5
            )
            print(f'mAP: {mean_avg_prec}')

            if mean_avg_prec > 0.9:
                checkpoint = {
                    'state_dict': model.state_dict(),
                    'optimizer': optim.state_dict()
                }
                save_checkpoint(checkpoint, filename=load_model_file)
                import time
                time.sleep(10)
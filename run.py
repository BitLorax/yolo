import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import os
import shutil

from tqdm import tqdm

from utils import (
    get_bboxes,
    get_losses,
    mean_average_precision,
    save_checkpoint,
    load_checkpoint,
    save_predictions,
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
elif selected_dataset[0:5] == 'shape':
    transform = Compose([transforms.ToTensor()])
else:
    print('Invalid dataset configuration.')


def train(dataloader, model, optim, loss_fn, epoch):
    if verbose:
        loop = tqdm(dataloader, leave=True)
    else:
        loop = dataloader
    mean_loss = []
    mean_box_loss = []
    mean_obj_conf_loss = []
    mean_noobj_conf_loss = []
    mean_class_loss = []

    for _, (x, y) in enumerate(loop):
        x, y = x.to(device), y.to(device)
        out = model(x)
        # if (int)(epoch / 10) % 2 == 0:  # alternate confidence-only and all loss every 10 epochs
        #     conf_only = False
        # else:
        #     conf_only = True
        # loss, box_loss, obj_conf_loss, noobj_conf_loss, class_loss = loss_fn(out, y, conf_only)
        loss, box_loss, obj_conf_loss, noobj_conf_loss, class_loss = loss_fn(out, y)

        mean_loss.append(loss.item())
        mean_box_loss.append(box_loss.item())
        mean_obj_conf_loss.append(obj_conf_loss.item())
        mean_noobj_conf_loss.append(noobj_conf_loss.item())
        mean_class_loss.append(class_loss.item())

        for param in model.parameters():
            param.grad = None
        loss.backward()
        optim.step()
    
    mean_loss = sum(mean_loss) / len(mean_loss)
    mean_box_loss = sum(mean_box_loss) / len(mean_box_loss)
    mean_obj_conf_loss = sum(mean_obj_conf_loss) / len(mean_obj_conf_loss)
    mean_noobj_conf_loss = sum(mean_noobj_conf_loss) / len(mean_noobj_conf_loss)
    mean_class_loss = sum(mean_class_loss) / len(mean_class_loss)
    return mean_loss, mean_box_loss, mean_obj_conf_loss, mean_noobj_conf_loss, mean_class_loss


def test(dataloader, model, loss_fn):
    save_predictions(dataloader, model, loss_fn)
    losses = get_losses()
    mAPs = {}
    conf_threshold = 0.1
    while conf_threshold <= 0.5:
        pred_boxes, target_boxes = get_bboxes(
            model, iou_threshold=0.5, conf_threshold=conf_threshold
        )
        mAP = mean_average_precision(
            pred_boxes, target_boxes, iou_threshold=0.5, plot_curve=False
        )
        mAPs['mAP_' + "{:.2f}".format(conf_threshold)] = mAP

        conf_threshold += 0.05

    return losses[0], losses[1], losses[2], losses[3], losses[4], mAPs


if __name__ == '__main__':
    print(f'Running on dataset: {selected_dataset}')
    print(f'Save file: {save_model_file}')
    print(f'Epochs: {epochs}')
    if resume_run:
        print('Resuming previous run. ')
        print(f'Load file: {load_model_file}')
    print()

    config = {
        'learning_rate': learning_rate,
        'epochs': epochs,
        'batch_size': batch_size,
        'weight_decay': weight_decay,
        'optimizer': optimizer
    }
    if optimizer == 'sgd':
        config['momentum'] = momentum

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

    if verbose:
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

    if verbose:
        print('Created model, optimizer, and loss function.')

    if enable_wandb:
        if resume_run and resume_run_id != '':
            wandb.init(project='yolo', entity='willjhliang', config=config, id=resume_run_id, resume='must')
        else:
            wandb.init(project='yolo', entity='willjhliang', config=config)
        wandb.watch(model, log_freq=10*len(train_dataloader))

    if resume_run:
        load_checkpoint(torch.load(load_model_filepath), model, optim)

    if not os.path.exists('saves'):
        os.makedirs('saves')

    for epoch in range(epochs):
        print(f'Epoch: {epoch}')

        loss, box_loss, obj_conf_loss, noobj_conf_loss, class_loss = train(train_dataloader, model, optim, loss_fn, epoch)
        val_loss, val_box_loss, val_obj_conf_loss, val_noobj_conf_loss, val_class_loss, mAPs = test(test_dataloader, model, loss_fn)
        max_mAP = max(mAPs.values())
        
        print(f'Training loss: {loss}')
        print(f'Validation loss: {val_loss}')
        print(f'Max mAP: {max_mAP}')
        if enable_wandb:
            log = {
                "loss": loss,
                "box_loss": box_loss,
                "obj_conf_loss": obj_conf_loss,
                "noobj_conf_loss": noobj_conf_loss,
                "class_loss": class_loss,
                "val_loss": val_loss,
                "val_box_loss": val_box_loss,
                "val_obj_conf_loss": val_obj_conf_loss,
                "val_noobj_conf_loss": val_noobj_conf_loss,
                "val_class_loss": val_class_loss,
                "mAP": max_mAP,
            }
            log.update(mAPs)
            wandb.log(log)

        checkpoint = {
            'state_dict': model.state_dict(),
            'optimizer': optim.state_dict()
        }
        save_checkpoint(checkpoint, filename=save_model_filepath)
        if os.path.isdir('drive/MyDrive/'):
            shutil.copy(save_model_filepath, 'drive/MyDrive/model.pth.tar')
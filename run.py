import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import os
import shutil

from tqdm import tqdm

from utils import (
    Losses,
    get_bboxes,
    load_losses,
    mean_average_precision,
    save_checkpoint,
    load_checkpoint,
    save_predictions,
)

from dataset import Dataset
from model import Yolo
from loss import YoloLoss
from config import epochs, batch_size, optimizer, learning_rate, momentum, weight_decay, resume_run, resume_run_id, save_model_file, load_model_file, num_workers, pin_memory, device, enable_wandb

import wandb


class Compose(object):
    def __init__(self, t):
        self.transforms = t
    
    def __call__(self, im, bboxes):
        for t in self.transforms:
            im, bboxes = t(im), bboxes
        return im, bboxes


def get_transform():
    return Compose([transforms.ToTensor()])


def train(dataloader, model, optim, loss_fn):
    loop = tqdm(dataloader, leave=True)
    losses = Losses()

    for _, (x, y) in enumerate(loop):
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss, box_loss, obj_conf_loss, noobj_conf_loss, class_loss = loss_fn(out, y)
        losses.append(loss.item(), box_loss.item(), obj_conf_loss.item(), noobj_conf_loss.item(), class_loss.item())

        for param in model.parameters():
            param.grad = None
        loss.backward()
        optim.step()
    
    return losses.means()


def test(dataloader, model, loss_fn):
    save_predictions(dataloader, model, loss_fn)
    losses = load_losses()
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


def main():
    print(f'Save file: %s' % save_model_file)
    print(f'Epochs: %d' % epochs)
    if resume_run:
        print('Resuming previous run. ')
        print(f'Load file: %s' % load_model_file)
    print()

    config = {
        'epochs': epochs,
        'batch_size': batch_size,
        'optimizer': optimizer,
        'learning_rate': learning_rate,
        'weight_decay': weight_decay
    }
    if optimizer == 'sgd':
        config['momentum'] = momentum

    transform = get_transform()
    train_dataset = Dataset(
        'shape', 'train.csv',
        transform=transform,
    )
    test_dataset = Dataset(
        'shape', 'test.csv',
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
        optim = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 weight_decay=weight_decay)
    elif optimizer == 'sgd':
        optim = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum,
                                weight_decay=weight_decay)
    else:
        print('ERROR: Invalid optimizer.')
        return
    loss_fn = YoloLoss()

    print('Created model, optimizer, and loss function.')

    if enable_wandb:
        if resume_run and resume_run_id is not None:
            if resume_run_id is None:
                print('ERROR: Resume run enabled by ID is not specified.')
                return
            else:
                wandb.init(project='yolo', entity='willjhliang', config=config, id=resume_run_id, resume='must')
        else:
            wandb.init(project='yolo', entity='willjhliang', config=config)
        wandb.watch(model, log_freq=10*len(train_dataloader))

    if resume_run:
        load_checkpoint(torch.load(load_model_file), model, optim)

    if not os.path.exists('saves'):
        os.makedirs('saves')

    for epoch in range(epochs):
        print(f'Epoch: {epoch}')

        loss, box_loss, obj_conf_loss, noobj_conf_loss, class_loss = train(train_dataloader, model, optim, loss_fn)
        val_loss, val_box_loss, val_obj_conf_loss, val_noobj_conf_loss, val_class_loss, mAPs = test(test_dataloader,
                                                                                                    model, loss_fn)
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
        save_checkpoint(checkpoint, filename=save_model_file)
        if os.path.isdir('drive/MyDrive/'):
            shutil.copy(save_model_file, 'drive/MyDrive/model.pth.tar')


if __name__ == '__main__':
    main()

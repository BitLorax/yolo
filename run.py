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
import config

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
        x, y = x.to(config.device), y.to(config.device)
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
    print(f'Save file: %s' % config.save_model_file)
    print(f'Epochs: %d' % config.epochs)
    if config.resume_run:
        print('Resuming previous run. ')
        print(f'Load file: %s' % config.load_model_file)
    print()

    wandb_config = {
        'epochs': config.epochs,
        'batch_size': config.batch_size,
        'optimizer': config.optimizer,
        'learning_rate': config.learning_rate,
        'weight_decay': config.weight_decay
    }
    if config.optimizer == 'sgd':
        wandb_config['momentum'] = config.momentum

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
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        shuffle=True,
        drop_last=True
    )
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        shuffle=True,
        drop_last=True
    )

    print('Created datasets and dataloaders.')

    model = Yolo().to(config.device)
    if config.optimizer == 'adam':
        optim = torch.optim.Adam(model.parameters(), lr=config.learning_rate,
                                 weight_decay=config.weight_decay)
    elif config.optimizer == 'sgd':
        optim = torch.optim.SGD(model.parameters(), lr=config.learning_rate, momentum=config.momentum,
                                weight_decay=config.weight_decay)
    else:
        print('ERROR: Invalid optimizer.')
        return
    loss_fn = YoloLoss()

    print('Created model, optimizer, and loss function.')

    if config.enable_wandb:
        if config.resume_run and config.resume_run_id is not None:
            if config.resume_run_id is None:
                print('ERROR: Resume run enabled by ID is not specified.')
                return
            else:
                wandb.init(project='yolo', entity='willjhliang', config=wandb_config, id=config.resume_run_id, resume='must')
        else:
            wandb.init(project='yolo', entity='willjhliang', config=wandb_config)
        wandb.watch(model, log_freq=10*len(train_dataloader))

    if config.resume_run:
        load_checkpoint(torch.load(config.load_model_file), model, optim)

    if not os.path.exists('saves'):
        os.makedirs('saves')

    for epoch in range(config.epochs):
        print(f'Epoch: {epoch}')

        loss, box_loss, obj_conf_loss, noobj_conf_loss, class_loss = train(train_dataloader, model, optim, loss_fn)
        val_loss, val_box_loss, val_obj_conf_loss, val_noobj_conf_loss, val_class_loss, mAPs = test(test_dataloader, model, loss_fn)
        max_mAP = max(mAPs.values())
        
        print(f'Training loss: {loss}')
        print(f'Validation loss: {val_loss}')
        print(f'Max mAP: {max_mAP}')
        if config.enable_wandb:
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
        save_checkpoint(checkpoint, filename=config.save_model_file)
        if os.path.isdir('drive/MyDrive/'):
            shutil.copy(config.save_model_file, 'drive/MyDrive/model.pth.tar')


if __name__ == '__main__':
    main()

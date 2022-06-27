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
from load_config import p

import wandb


class Compose(object):
    def __init__(self, t):
        self.transforms = t
    
    def __call__(self, im, bboxes):
        for t in self.transforms:
            im, bboxes = t(im), bboxes
        return im, bboxes


def get_transform():
    if p.selected_dataset.name == 'voc':
        return Compose([transforms.Resize((448, 448)), transforms.ToTensor()])
    elif p.selected_dataset.name[0:5] == 'shape':
        return Compose([transforms.ToTensor()])
    else:
        print('Invalid dataset configuration.')
        return None


def train(dataloader, model, optim, loss_fn):
    if p.verbose:
        loop = tqdm(dataloader, leave=True)
    else:
        loop = dataloader
    losses = Losses()

    for _, (x, y) in enumerate(loop):
        x, y = x.to(p.device), y.to(p.device)
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
    print(f'Running on dataset: %s' % p.selected_dataset.name)
    print(f'Save file: %s' % p.save_model_file)
    print(f'Epochs: %d' % p.epochs)
    if p.resume_run:
        print('Resuming previous run. ')
        print(f'Load file: %s' % p.load_model_file)
    print()

    config = {
        'learning_rate': p.optimizer.learning_rate,
        'epochs': p.epochs,
        'batch_size': p.batch_size,
        'weight_decay': p.optimizer.weight_decay,
        'optimizer': p.optimizer.name
    }
    if p.optimizer.name == 'sgd':
        config['momentum'] = p.optimizer.momentum

    transform = get_transform()
    train_dataset = Dataset(
        p.selected_dataset.name,
        p.selected_dataset.train_data_csv,
        transform=transform,
    )
    test_dataset = Dataset(
        p.selected_dataset.name,
        p.selected_dataset.test_data_csv,
        transform=transform,
    )
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=p.batch_size,
        num_workers=p.num_workers,
        pin_memory=p.pin_memory,
        shuffle=True,
        drop_last=True
    )
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=p.batch_size,
        num_workers=p.num_workers,
        pin_memory=p.pin_memory,
        shuffle=True,
        drop_last=True
    )

    if p.verbose:
        print('Created datasets and dataloaders.')

    model = Yolo().to(p.device)
    if p.optimizer.name == 'adam':
        optim = torch.optim.Adam(model.parameters(), lr=p.optimizer.learning_rate,
                                 weight_decay=p.optimizer.weight_decay)
    elif p.optimizer.name == 'sgd':
        optim = torch.optim.SGD(model.parameters(), lr=p.optimizer.learning_rate, momentum=p.optimizer.momentum,
                                weight_decay=p.optimizer.weight_decay)
    else:
        print('Invalid optimizer.')
        optim = None
    loss_fn = YoloLoss()

    if p.verbose:
        print('Created model, optimizer, and loss function.')

    if p.enable_wandb:
        if p.resume_run and p.resume_run_id is not None:
            wandb.init(project='yolo', entity='willjhliang', config=config, id=p.resume_run_id, resume='must')
        else:
            wandb.init(project='yolo', entity='willjhliang', config=config)
        wandb.watch(model, log_freq=10*len(train_dataloader))

    if p.resume_run:
        load_checkpoint(torch.load(p.load_model_filepath), model, optim)

    if not os.path.exists('saves'):
        os.makedirs('saves')

    for epoch in range(p.epochs):
        print(f'Epoch: {epoch}')

        loss, box_loss, obj_conf_loss, noobj_conf_loss, class_loss = train(train_dataloader, model, optim, loss_fn)
        val_loss, val_box_loss, val_obj_conf_loss, val_noobj_conf_loss, val_class_loss, mAPs = test(test_dataloader,
                                                                                                    model, loss_fn)
        max_mAP = max(mAPs.values())
        
        print(f'Training loss: {loss}')
        print(f'Validation loss: {val_loss}')
        print(f'Max mAP: {max_mAP}')
        if p.enable_wandb:
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
        save_checkpoint(checkpoint, filename=p.save_model_filepath)
        if os.path.isdir('drive/MyDrive/'):
            shutil.copy(p.save_model_filepath, 'drive/MyDrive/model.pth.tar')


if __name__ == '__main__':
    main()

import torch
from torch.utils.data import Dataset

import os
from PIL import Image
import pandas as pd

from params import S, B, C

class Dataset(Dataset):
    """
    YOLO dataset loader. Loads images and bounding boxes, edits bounding box format to have shape (S, S, C + 5) by assigning bounding boxes to associated grid cell and modifying x and y to be fraction of cell length. Each resulting box contains class probabilities, confidence (1 if box exists, 0 otherwise), x, y, width, height, in order.
    """

    def __init__(self, dataset, csv_file, transform=None):
        fileloc = 'dataset/' + dataset + '/'
        self.annotations = pd.read_csv(fileloc + csv_file)
        self.transform = transform

        self.im_dir = fileloc + 'images'
        self.label_dir = fileloc + 'labels'
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        im_path = os.path.join(self.im_dir, self.annotations.iloc[idx, 0])
        im = Image.open(im_path)
        label_path = os.path.join(self.label_dir, self.annotations.iloc[idx, 1])
        boxes = []
        with open(label_path) as f:
            for label in f.readlines():
                label = label.strip().split()
                c = int(label[0])
                x, y, w, h = [float(i) for i in label[1:]]
                boxes.append([c, x, y, w, h])
            boxes = torch.tensor(boxes)
            if self.transform:
                im, boxes = self.transform(im, boxes)
            
            labels = torch.zeros((S, S, C + 5))
            for box in boxes:
                class_label, x, y, w, h = box.tolist()
                class_label = int(class_label)

                cx = int(S * x)
                cy = int(S * y)
                x = S * x - cx
                y = S * y - cy

                labels[cx, cy] = 0
                labels[cx, cy, class_label] = 1
                labels[cx, cy, C] = 1
                labels[cx, cy, C+1] = x
                labels[cx, cy, C+2] = y
                labels[cx, cy, C+3] = w
                labels[cx, cy, C+4] = h
                
            return im, labels
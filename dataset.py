import torch
from torch.utils.data import Dataset

import os
from PIL import Image
import pandas as pd

from params import S, B, C

class Dataset(Dataset):
    def __init__(self, dataset, csv_file, transform=None):
        fileloc = 'dataset/' + dataset + '/'
        self.annotations = pd.read_csv(fileloc + csv_file)
        self.transform = transform

        self.im_dir = fileloc + 'images'
        self.label_dir = fileloc + 'labels'
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        label_path = os.path.join(self.label_dir, self.annotations.iloc[idx, 1])
        boxes = []
        with open(label_path) as f:
            for label in f.readlines():
                label = label.strip().split()
                class_label = int(label[0])
                x, y, w, h = [float(x) for x in label[1:]]
                boxes.append([class_label, x, y, w, h])
            im_path = os.path.join(self.im_dir, self.annotations.iloc[idx, 0])
            im = Image.open(im_path)

            boxes = torch.tensor(boxes)
            if self.transform:
                im, boxes = self.transform(im, boxes)
            
            label_matrix = torch.zeros((S, S, C + 5 * B))  # C + 5 * B to work with util functions
            for box in boxes:
                class_label, x, y, w, h = box.tolist()
                class_label = int(class_label)
                i, j = int(S * y), int(S * x)
                x_cell, y_cell = S * x - j, S * y - i
                w_cell, h_cell = w * S, h * S

                if label_matrix[i, j, C] == 0:
                    label_matrix[i, j, C] = 1
                    box_coordinates = torch.tensor([x_cell, y_cell, w_cell, h_cell])
                    label_matrix[i, j, C+1:C+5] = box_coordinates
                    label_matrix[i, j, class_label] = 1
                
            return im, label_matrix
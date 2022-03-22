
from PIL import Image
import numpy as np
from utils import plot_image

filename = '0001'
dataset = 'shape'
with Image.open('dataset/' + dataset + '/images/' + filename + '.jpg') as im, open('dataset/' + dataset + '/labels/' + filename + '.txt') as data:
    im = np.array(im).astype('float64')
    im /= 255
    boxes = data.readlines()
    boxes = [i.split() for i in boxes]
    for i in range(len(boxes)):
        boxes[i] = [1.0] + [float(j) for j in boxes[i]]
    plot_image(im, boxes)
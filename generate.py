
import numpy as np
from PIL import Image, ImageDraw
import random
import os
from tqdm import tqdm

WIDTH = 448
HEIGHT = 448

SHAPES = [0, 3, 4, 5, 6]


def overlaps(boxA, boxB):
    Aw, Ah = boxA[2:3], boxA[3:4]
    Bw, Bh = boxB[2:3], boxB[3:4]

    Ax1, Ay1 = boxA[0:1] - Aw / 2, boxA[1:2] - Ah / 2
    Bx1, By1 = boxB[0:1] - Bw / 2, boxB[1:2] - Bh / 2
    Ax2, Ay2 = boxA[0:1] + Aw / 2, boxA[1:2] + Ah / 2
    Bx2, By2 = boxB[0:1] + Bw / 2, boxB[1:2] + Bh / 2

    x1 = np.maximum(Ax1, Bx1)
    y1 = np.maximum(Ay1, By1)
    x2 = np.minimum(Ax2, Bx2)
    y2 = np.minimum(Ay2, By2)

    intersection = np.clip((x2 - x1), 0, None) * np.clip((y2 - y1), 0, None)
    if intersection / (Aw * Ah) > 0.3:
        return True
    if intersection / (Bw * Bh) > 0.3:
        return True
    
    return False


def generate_shape(data):
    im = Image.fromarray(np.zeros((WIDTH, HEIGHT, 4), dtype='uint8'), 'RGBA')
    draw = ImageDraw.Draw(im)
    color = (random.randint(0, 5) * 40 + 50, random.randint(0, 5) * 40 + 50, random.randint(0, 5) * 40 + 50)
    shape_idx = random.randint(0, len(SHAPES) - 1)
    shape = SHAPES[shape_idx]

    valid = False
    while not valid:
        s = random.randint(int(min(WIDTH, HEIGHT) * 0.1), int(min(WIDTH, HEIGHT) * 0.4))
        x = random.randint(0, WIDTH - s - 1)
        y = random.randint(0, HEIGHT - s - 1)
        valid = True
        for box in data:
            box = np.array(box[1:])
            ss = s / WIDTH
            prop_box = np.array([x / WIDTH + ss / 2, y / HEIGHT + ss / 2, ss, ss])
            if overlaps(box, prop_box):
                valid = False

    if shape == 0:
        draw.ellipse([(x, y), (x + s, y + s)], fill=color)
    else:
        # rotation = random.randint(0, 360)
        rotation = 0
        r = (int)(s / 2)
        draw.regular_polygon((x + r, y + r, r), shape, rotation=rotation, fill=color)

    s = s / WIDTH
    x = x / WIDTH + s / 2
    y = y / HEIGHT + s / 2

    return im, shape_idx, x, y, s


def generate_image():
    num_shapes = random.randint(1, 3)
    ret = Image.fromarray(np.zeros((WIDTH, HEIGHT, 4), dtype='uint8'), 'RGBA')
    data = []
    for _ in range(num_shapes):
        res, shape, x, y, s = generate_shape(data)
        ret.paste(res, (0, 0), res)
        data.append([shape, x, y, s, s])

    return ret, data


def generate_dataset(start_idx, end_idx, csv):
    for i in tqdm(range(start_idx, end_idx)):
        im, data = generate_image()
        im = im.convert('RGB')
        filename = str(i).zfill(4)
        im.save(image_dir + '/' + filename + '.png')
        with open(label_dir + '/' + filename + '.txt', 'w') as f:
            for i in data:
                f.write(' '.join(map(str, i)))
                f.write('\n')
        csv.write(filename + '.png,' + filename + '.txt\n')


dataset_name = 'shape_outline_norot'
image_dir = 'dataset/' + dataset_name + '/images'
label_dir = 'dataset/' + dataset_name + '/labels'

if not os.path.exists(image_dir):
    os.makedirs(image_dir)
if not os.path.exists(label_dir):
    os.makedirs(label_dir)

with open('dataset/' + dataset_name + '/train.csv', 'w') as csv:
    generate_dataset(0, 9000, csv)
with open('dataset/' + dataset_name + '/test.csv', 'w') as csv:
    generate_dataset(9000, 10000, csv)

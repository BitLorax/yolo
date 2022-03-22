
import numpy as np
from PIL import Image, ImageDraw
import random
import os
from tqdm import tqdm

WIDTH = 448
HEIGHT = 448

SHAPES = [0, 3, 4, 5, 6]

def generate_shape():
    im = Image.fromarray(np.zeros((WIDTH, HEIGHT, 4), dtype='uint8'), 'RGBA')
    draw = ImageDraw.Draw(im)
    color = (random.randint(0, 5) * 40 + 50, random.randint(0, 5) * 40 + 50, random.randint(0, 5) * 40 + 50)
    shape_idx = random.randint(0, len(SHAPES) - 1)
    shape = SHAPES[shape_idx]

    s = random.randint(int(min(WIDTH, HEIGHT) * 0.05), int(min(WIDTH, HEIGHT) * 0.3))
    x = random.randint(0, WIDTH - s - 1)
    y = random.randint(0, HEIGHT - s - 1)

    if shape == 0:
        draw.ellipse([(x, y), (x + s, y + s)], fill=color)
    else:
        rotation = random.randint(0, 360)
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
        res, shape, x, y, s = generate_shape()
        ret.paste(res, (0, 0), res)
        data.append([shape, x, y, s, s])

    filtered_data = []
    pix = ret.load()
    for line in data:
        x, y, w, h = line[1] * WIDTH, line[2] * HEIGHT, line[3] * WIDTH, line[4] * HEIGHT
        x -= w / 2
        y -= h / 2
        x, y, w, h = int(x), int(y), int(w), int(h)
        c = pix[x, y]
        valid = False
        for i in range(x, x + w):
            for j in range(y, y + h):
                if pix[i, j] != c:
                    valid = True
        if valid:
            filtered_data.append(line)

    return ret, filtered_data

if not os.path.exists('dataset/shape/images'):
    os.makedirs('dataset/shape/images')
if not os.path.exists('dataset/shape/labels'):
    os.makedirs('dataset/shape/labels')

with open('dataset/shape/train.csv', 'w') as csv:
    for i in tqdm(range(10000)):
        im, data = generate_image()
        im = im.convert('RGB')
        filename = str(i).zfill(4)
        im.save('dataset/shape/images/' + filename + '.jpg')
        with open('dataset/shape/labels/' + filename + '.txt', 'w') as f:
            for i in data:
                f.write(' '.join(map(str, i)))
                f.write('\n')
        csv.write(filename + '.jpg,' + filename + '.txt\n')


import numpy as np
from PIL import Image, ImageDraw
import random

WIDTH = 448
HEIGHT = 448

SHAPES = [0, 3, 4, 5, 6]

def generate_shape(used_colors, data):
    im = Image.fromarray(np.zeros((WIDTH, HEIGHT, 4), dtype='uint8'), 'RGBA')
    draw = ImageDraw.Draw(im)
    color = (random.randint(0, 3) * 85, random.randint(0, 3) * 85, random.randint(0, 3) * 85)
    while color == (0, 0, 0) or color in used_colors:
        color = (random.randint(0, 3) * 85, random.randint(0, 3) * 85, random.randint(0, 3) * 85)
    shape = random.choice(SHAPES)

    s = random.randint(int(min(WIDTH, HEIGHT) * 0.2), int(min(WIDTH, HEIGHT) * 0.6))
    x = random.randint(0, WIDTH - s - 1)
    y = random.randint(0, HEIGHT - s - 1)
    while not valid([x, y, s], data):
        s = random.randint(int(min(WIDTH, HEIGHT) * 0.2), int(min(WIDTH, HEIGHT) * 0.6))
        x = random.randint(0, WIDTH - s - 1)
        y = random.randint(0, HEIGHT - s - 1)

    if shape == 0:
        draw.ellipse([(x, y), (x + s, y + s)], fill=color)
    else:
        rotation = random.randint(0, 360)
        r = (int)(s / 2)
        draw.regular_polygon((x + r, y + r, r), shape, rotation=rotation, fill=color)

    return im, shape, x, y, s, color

def generate_image():
    num_shapes = random.randint(1, 3)
    ret = Image.fromarray(np.zeros((WIDTH, HEIGHT, 4), dtype='uint8'), 'RGBA')
    data = []
    used_colors = []
    for _ in range(num_shapes):
        res, shape, x, y, s, color = generate_shape(used_colors, data)
        ret.paste(res, (0, 0), res)
        data.append([shape, x, y, s, s])
        used_colors.append(color)

    return ret, data

def valid(box, data):
    for shape in data:
        if contained([box[0], box[1], box[0] + box[2], box[1] + box[2]], [shape[1], shape[2], shape[1] + shape[3], shape[2] + shape[3]]):
            return False
    return True

def contained(box1, box2):
    if (box1[0] >= box2[0] and box1[1] >= box2[1] and box1[2] <= box2[2] and box1[3] <= box2[3]):
        return True
    if (box2[0] >= box1[0] and box2[1] >= box1[1] and box2[2] <= box1[2] and box2[3] <= box1[3]):
        return True
    return False

for i in range(10000):
    im, data = generate_image()
    im = im.convert('RGB')
    filename = 'data/shape-' + str(i).zfill(4)
    im.save(filename + '.png')
    with open(filename + '.txt', 'w') as f:
        for i in data:
            f.write(' '.join(map(str, i)))
            f.write('\n')

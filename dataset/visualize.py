
import numpy as np
from PIL import Image, ImageDraw

filename = input('Filename: ')
with Image.open(filename + '.png') as im, open(filename + '.txt') as data:
    draw = ImageDraw.Draw(im)
    for line in data:
        line = line.split(' ')
        x, y, w, h = int(line[1]), int(line[2]), int(line[3]), int(line[4])
        draw.rectangle([(x, y), (x + w, y + h)], outline=(255, 255, 255))
    im.show()
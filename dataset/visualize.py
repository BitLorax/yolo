
from PIL import Image, ImageDraw

filename = '0000'
dataset = 'shape'
with Image.open('dataset/' + dataset + '/images/' + filename + '.jpg') as im, open('dataset/' + dataset + '/labels/' + filename + '.txt') as data:
    draw = ImageDraw.Draw(im)
    for line in data:
        line = line.split(' ')
        line = [float(x) for x in line]
        w = line[3] * im.width
        h = line[4] * im.height
        x = line[1] * im.width - w / 2
        y = line[2] * im.height - h / 2
        draw.rectangle([(x, y), (x + w, y + h)], outline=(255, 255, 255))
    im.show()
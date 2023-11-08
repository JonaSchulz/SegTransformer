import torch
from PIL import Image
import os
import numpy as np
from loss import BoundaryCELoss


def split_background(label, background_label=3):
    for i, column in enumerate(label):
        for j, row in enumerate(column):
            if label[i, j] == background_label:
                label[i, j] = 5
            else:
                break
    return label


def remap(label, mapping):
    for i, column in enumerate(label):
        for j, row in enumerate(column):
            label[i, j] = mapping[label[i, j]]


def binary(label):
    return (label[:, :-1] != label[:, 1:]).astype("uint8")


# mapping = {80: 1, 160: 2, 255: 3}
# mapping = {5: 0, 0: 1, 1: 2, 4: 3, 2: 4, 3: 5}
mapping = {0: 0, 1: 51, 2: 102, 3: 153, 4: 204, 5: 255}

img_dir = "data/test_label_square_3L"
target_dir = "data/test_label_square_3L"
#test_label_dir = "data/test_label"
#image_dir = "data/Layer_Masks"
loss = BoundaryCELoss()

for img_file in os.listdir(img_dir):
    with Image.open(os.path.join(img_dir, img_file)) as img:
        img = np.array(img)
        img[img == 3] = 0
        img[img == 5] = 0
        img[img == 4] = 3
        #for i in range(4):
        #    square = img[:, i*256:(i+1)*256]
        #    square = Image.fromarray(square)
        #    square.save(os.path.join(target_dir, f"{img_file.split('.')[0]}_{i}.png"))

        #print(target.shape)
        #break
        #print(img[:, 0])
        #remap(img, mapping)
        #img = img[:, :, 0]
        #for i, val in mapping.items():
        #    img[img == i] = val
        #img = split_background(img)
        #img = np.array(img).swapaxes(0, 1)
        #img = binary(img)
        #img = img.swapaxes(0, 1)
        #img = np.concatenate((img, np.zeros((1, img.shape[1]), dtype="uint8")))
        target = Image.fromarray(img)
        target.save(os.path.join(target_dir, img_file))

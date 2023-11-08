import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch import nn
from PIL import Image
import os


class SplitBackground:
    def __init__(self, background_label=3):
        self.background_label = background_label

    def __call__(self, label):
        for i, column in enumerate(label):
            for j, row in enumerate(column):
                if label[i, j] == self.background_label:
                    label[i, j] = 5
                else:
                    break
        return label


class VariedRange(nn.Module):
    def __init__(self, device="cuda"):
        super().__init__()
        self.device = device

    def forward(self, x):
        return x * (0.8 + torch.rand(1).to(self.device) / 2.5)


class VariedSeqLength(nn.Module):
    def __init__(self, max_cutoff):
        super().__init__()
        self.keep = False
        self.cutoff = 0
        self.max_cutoff = max_cutoff

    def forward(self, x):
        if not self.keep:
            self.cutoff = torch.randint(0, self.max_cutoff, (1,))
            self.keep = True
        else:
            self.keep = False
        return x[:, :-self.cutoff, :]


class OCT_Dataset(Dataset):
    def __init__(self, image_dir, label_dir=None, image_transform=None, label_transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_files = os.listdir(image_dir)
        if label_dir:
            self.label_files = os.listdir(label_dir)
        self.image_transform = image_transform
        self.label_transform = label_transform
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, item):
        image_file = os.path.join(self.image_dir, self.image_files[item])
        if self.label_dir is not None:
            label_file = os.path.join(self.label_dir, self.label_files[item])

        with Image.open(image_file) as img:
            img = img.convert("L")
            img = self.to_tensor(img)
        if self.image_transform:
            img = self.image_transform(img)
        img = torch.squeeze(img).permute(1, 0)

        if self.label_dir is not None:
            with Image.open(label_file) as label_image:
                label = torch.unsqueeze(torch.from_numpy(np.asarray(label_image)), 0)
                if self.label_transform:
                    label = self.label_transform(label)
                label = torch.squeeze(label).permute(1, 0)
                label = label.to(torch.long)

            return img, label

        return img


class OCT_Flipped_Dataset(Dataset):
    def __init__(self, image_dir, label_dir=None, image_transform=None, label_transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_files = os.listdir(image_dir)
        if label_dir:
            self.label_files = os.listdir(label_dir)
        self.image_transform = image_transform
        self.label_transform = label_transform
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, item):
        image_file = os.path.join(self.image_dir, self.image_files[item])
        if self.label_dir is not None:
            label_file = os.path.join(self.label_dir, self.label_files[item])

        with Image.open(image_file) as img:
            img = img.convert("L")
            img = self.to_tensor(img)
        if self.image_transform:
            img = self.image_transform(img)
        img = torch.squeeze(img)

        if self.label_dir is not None:
            with Image.open(label_file) as label_image:
                label = torch.unsqueeze(torch.from_numpy(np.asarray(label_image)), 0)
                if self.label_transform:
                    label = self.label_transform(label)
                label = torch.squeeze(label)
                label = label.to(torch.long)

            return img, label

        return img


class OCT_Column_Dataset(Dataset):
    def __init__(self, image_dir, image_transform=None):
        self.image_dir = image_dir
        self.image_files = os.listdir(image_dir)
        self.image_transform = image_transform
        self.to_tensor = transforms.ToTensor()

        image_file = os.path.join(self.image_dir, self.image_files[0])
        with Image.open(image_file) as img:
            img = img.convert("L")
            img = self.to_tensor(img)
        if self.image_transform:
            img = self.image_transform(img)
        self.image_width = img.shape[-1]

    def __len__(self):
        return len(self.image_files) * self.image_width

    def __getitem__(self, item):
        image_index = item // self.image_width
        column_index = item % self.image_width
        image_file = os.path.join(self.image_dir, self.image_files[image_index])

        with Image.open(image_file) as img:
            img = img.convert("L")
            img = self.to_tensor(img)
        if self.image_transform:
            img = self.image_transform(img)
        img = torch.squeeze(img).permute(1, 0)[column_index]

        return img

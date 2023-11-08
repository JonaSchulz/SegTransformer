import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
import numpy as np
from model import PositionalEncoding
from model import SegTransformer
from dataset import OCT_Dataset, SplitBackground, VariedRange, VariedSeqLength

out = torch.randn(68, 2, 16)
x = torch.randn(68, 2, 16)
tgt = torch.randn(68, 2, 16)


fig, ax = plt.subplots(3)
ax[0].imshow(out.cpu().permute(2, 1, 0).numpy()[:, 0, :])
ax[1].imshow(x.cpu().permute(2, 1, 0).numpy()[:, 0, :])
ax[2].imshow(tgt.cpu().permute(2, 1, 0).numpy()[:, 0, :])
plt.savefig("visualization.png")
plt.close()


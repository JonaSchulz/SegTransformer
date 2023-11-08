import torch
from torch import nn
from model import RelPosTransformer


if __name__ == "__main__":
    model = RelPosTransformer(16).to("cuda")

    x = torch.randn(10, 2, 16).to("cuda")
    tgt = torch.randn(1, 2, 16).to("cuda")
    y = model(x, tgt)
    print(y.shape)








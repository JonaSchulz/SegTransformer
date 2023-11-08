import torch
from torch import nn


class Sparse_AE_Encoder(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, d_model // 8),
            nn.ReLU(),
            nn.Linear(d_model // 8, d_model // 16)
        )

        #self.encoder = nn.Sequential(
        #    nn.Linear(d_model, d_model),
        #    nn.ReLU(),
        #    nn.Linear(d_model, d_model),
        #    nn.ReLU(),
        #    nn.Linear(d_model, d_model),
        #    nn.ReLU(),
        #    nn.Linear(d_model, d_model)
        #)

    def forward(self, x):
        return self.encoder(x)


class Sparse_AE_Decoder(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.decoder = nn.Sequential(
            nn.Linear(d_model // 16, d_model // 8),
            nn.ReLU(),
            nn.Linear(d_model // 8, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model)
        )

    def forward(self, x):
        return self.decoder(x)


class Sparse_AE(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.encoder = Sparse_AE_Encoder(d_model)
        self.decoder = Sparse_AE_Decoder(d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.encoder(x)
        #out = self.relu(out)
        out = self.decoder(out)

        return out

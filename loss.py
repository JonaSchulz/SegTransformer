import torch
from torch import nn
from torch.nn.functional import normalize


class BoundaryCELoss(nn.Module):
    def __init__(self, boundary_factor=1.0):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(reduction="none")

    def forward(self, pred: torch.Tensor, gt_label: torch.Tensor):
        loss = self.ce_loss(pred, gt_label)
        boundary_map = (gt_label[:, :, :-1] != gt_label[:, :, 1:])
        boundary_map = torch.cat(boundary_map, torch.zeros())
        pass


class TopologyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
            x is a labelmap output
        """
        return torch.mean(self.relu(x[:, :, :-1] - x[:, :, 1:]).to(float))


class SegLoss(nn.Module):
    def __init__(self, loss_weight: float, loss_type="ce"):
        super().__init__()
        self.loss_weight = loss_weight
        self.loss_type = loss_type
        if loss_type == "ce":
            self.main_loss = nn.CrossEntropyLoss()
            # self.main_loss = BoundaryCELoss()
        elif loss_type == "l2":
            self.main_loss = nn.MSELoss()
        self.topology_loss = TopologyLoss()

    def forward(self, pred: torch.Tensor, gt_label: torch.Tensor):
        """
        Args:
            pred: Tensor, shape
                if ce: [batch_size, n_classes, seq_len, embedding_dim]
                if l2: [batch_size, 1, seq_len, embedding_dim]
            gt_label: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        if self.loss_type == "ce":
            pred_label = torch.argmax(pred, 1)
        elif self.loss_type == "l2":
            pred = torch.squeeze(pred)
            pred_label = pred
            gt_label = gt_label.to(torch.float32)
        main_loss = self.main_loss(pred, gt_label)
        topology_loss = self.topology_loss(pred_label)

        return {"SegLoss": main_loss + self.loss_weight * topology_loss,
                "MainLoss": main_loss,
                "TopologyLoss": topology_loss}


class AELoss(nn.Module):
    def __init__(self, loss_weight):
        super().__init__()
        self.mse = nn.MSELoss()
        self.loss_weight = loss_weight

    def forward(self, x, gt, model):
        sparse_loss = 0
        out = x
        gt = gt.to(torch.float32)

        for layer in model.encoder.children():
            out = layer(out)
            sparse_loss += torch.mean(torch.abs(out))

        return self.mse(x, gt) + self.loss_weight * sparse_loss

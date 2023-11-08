from argparse import ArgumentParser
import json
import os
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from torchvision.datasets import MNIST
import torchvision.transforms as t

from dataset import OCT_Dataset, OCT_Column_Dataset
from loss import AELoss
from ae_model import Sparse_AE

parser = ArgumentParser()
parser.add_argument("--config", default=None, type=str, dest="config")
args_parser = parser.parse_args()

with open(args_parser.config) as json_stream:
    config = json.load(json_stream)

run_root = config["run_root"]
device = "cuda"
data_root = config["data_root"]
train_image_dataset = config["train_image_dataset"]
train_label_dataset = config["train_label_dataset"]
test_image_dataset = config["test_image_dataset"]
test_label_dataset = config["test_label_dataset"]
batch_size = config["batch_size"]
d_model = config["d_model"]
n_classes = config["n_classes"]
epochs = config["epochs"]
test_freq = config["test_freq"]

pred_dir = os.path.join(run_root, "ae_image_preds")
if not os.path.exists(pred_dir):
    os.makedirs(pred_dir)


def show_outputs(data_loader, model):
    model.eval()
    with torch.no_grad():
        for i, img in enumerate(data_loader):
            fig, ax = plt.subplots(2)
            img = img[0].to(device)
            #img = torch.randn(275, 64).to(device)
            pred = model(img)
            img = img.to("cpu")
            pred = pred.to("cpu")
            #img = img.permute(1, 0).to("cpu")
            #pred = pred.permute(1, 0).to("cpu")
            ax[0].imshow(img)
            ax[1].imshow(pred)
            fig.savefig(os.path.join(pred_dir, f"out_{i}.png"))
            plt.close()


def test_outputs(model, data_loader):
    output = torch.zeros(100, 64, dtype=torch.float32)
    for i in range(100):
        input = torch.randn(64).to(device)
        output[i] = model(input)
    print("Single inputs:")
    print(f"Variance: {torch.var(output, 0)[10]}")
    print(output[:, 10])

    input = torch.randn(100, 64).to(device)
    output = model(input)
    print("Batch:")
    print(f"Variance: {torch.var(output, 0)[10]}")
    print(output[:, 10])

    for img in data_loader:
        img = img[0].to(device)
        output = model(img)
        break
    print("Image:")
    print(f"Variance: {torch.var(output, 0)[10]}")
    print(output[:, 10])


def train(data_loader, model, loss_fn, optimizer, writer=None, epoch=None):
    model.train()

    for i, img in enumerate(data_loader):
        img = img.to(device)
        #label = label.to(device)

        optimizer.zero_grad()
        pred = model(img)
        loss = loss_fn(pred, img, model)
        loss.backward()
        optimizer.step()

        if writer is not None:
            writer.add_scalar("AELoss/train", loss, epoch * len(data_loader) + i)


def test(data_loader, model, loss_fn, writer=None, epoch=None):
    average_loss = 0
    model.eval()

    with torch.no_grad():
        for i, img in enumerate(data_loader):
            img = img.to(device)
            #label = label.to(device)

            pred = model(img)
            loss = loss_fn(pred, img, model)
            average_loss += loss

            if writer is not None:
                writer.add_scalar("AELoss/test", loss, epoch * len(data_loader) + i)

    average_loss /= (i + 1)
    print(f"Average Loss: {average_loss}")


# --------------------------------------------------
# prepare train and test datasets:
image_transform = torch.nn.Sequential(
    transforms.Resize(d_model)
)

label_transform = torch.nn.Sequential(
    transforms.Resize(d_model)
)

train_data = OCT_Column_Dataset(image_dir=os.path.join(data_root, train_image_dataset),
                                image_transform=image_transform)

test_data = OCT_Column_Dataset(image_dir=os.path.join(data_root, test_image_dataset),
                               image_transform=image_transform)

visual_data = OCT_Dataset(image_dir=os.path.join(data_root, train_image_dataset),
                          image_transform=image_transform)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
visual_loader = DataLoader(visual_data, batch_size=batch_size, shuffle=False)

# --------------------------------------------------

model = Sparse_AE(d_model).to(device)
loss_fn = AELoss(1).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

show_outputs(visual_loader, model)
# test_outputs(model, visual_loader)
for epoch in range(epochs):
    train(train_loader, model, loss_fn, optimizer, epoch=epoch)
    if not (epoch + 1) % test_freq:
        print(f"\nTest Epoch: {epoch}:")
        test(test_loader, model, loss_fn, epoch=epoch)
        show_outputs(visual_loader, model)


import os
import json
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from torchmetrics.classification import MulticlassJaccardIndex

from dataset import OCT_Dataset, OCT_Flipped_Dataset, VariedRange, VariedSeqLength
from model import SegTransformerEncoder, SegTransformer, OutputLayer
from loss import SegLoss

parser = ArgumentParser()
parser.add_argument("--config", default=None, type=str, dest="config")
parser.add_argument("--mode", default="train", type=str, dest="mode")
parser.add_argument("--load", default=None, type=str, dest="load")
args_parser = parser.parse_args()

with open(args_parser.config) as json_stream:
    config = json.load(json_stream)

mode = args_parser.mode
load_model_path = args_parser.load

run_root = config["run_root"]
device = "cuda"
data_root = config["data_root"]
train_image_dataset = config["train_image_dataset"]
train_label_dataset = config["train_label_dataset"]
test_image_dataset = config["test_image_dataset"]
test_label_dataset = config["test_label_dataset"]
dataset_type = config["dataset_type"] if "dataset_type" in config else "OCT_Dataset"
image_augmentation = config["image_augmentation"] if "image_augmentation" in config else []
transformer_config = config["transformer"] if "transformer" in config else None
positional_encoding = config["positional_encoding"] if "positional_encoding" in config else True
batch_size = config["batch_size"]
d_model = config["d_model"]
input_norm = config["input_norm"]
n_classes = config["n_classes"]
loss_type = config["loss_type"]
loss_weight = config["loss_weight"]
lr = config["lr"]
optimizer = config["optimizer"]
epochs = config["epochs"]
test_freq = config["test_freq"]
out_layer_config = config["output_layer"]
plot_dist = config["plot_dist"]
resize = config["resize"] if "resize" in config else d_model

if not os.path.exists(run_root):
    os.makedirs(run_root)

if plot_dist:
    dist_dir = os.path.join(run_root, "dist")

model_dir = os.path.join(run_root, "model")
labelmap_dir = os.path.join(run_root, "labelmaps")
# dist_dir = os.path.join(run_root, "dist")
tensorboard_dir = os.path.join(run_root, "tensorboard")

if not os.path.exists(labelmap_dir):
    os.makedirs(labelmap_dir)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
# if not os.path.exists(dist_dir):
#     os.makedirs(dist_dir)
if not os.path.exists(tensorboard_dir):
    os.makedirs(tensorboard_dir)


def tensor_distribution(x, file):
    """
    Plot the distribution of a tensor of arbitrary shape as a histogram

    Args:
         x: Tensor
         file: str, file path of histogram image
    """
    x = x.reshape(-1).numpy(force=True)
    plt.hist(x)
    plt.savefig(file)
    plt.close()


def save_labelmap(labelmap, name):
    """
    Args:
         labelmap: Tensor, shape [seq_len, embedding_dim]
    """
    fig, ax = plt.subplots()
    ax.imshow(labelmap.permute(1, 0).to("cpu"))
    fig.savefig(name)
    plt.close()


def test_columnwise(data_loader, model, labelmap_dir, mode="inference"):
    model.eval()

    with torch.no_grad():
        for i, (image, label) in enumerate(data_loader):
            image = image.to(device)
            label = label.to(device)
            result = torch.zeros_like(label[0])[:-1]
            for j in range(image.shape[1] - 1):
                _image = torch.cat((image[:, j:, :], torch.zeros((image.shape[0], image.shape[1] - j, image.shape[2]), dtype=torch.float).to("cuda")))
                #_image = image[:, j:, :]
                tgt = label[:, j:, :] if mode == "train" else None
                preds = model(_image, tgt, mode=mode)
                for pred in preds:
                    labelmap = get_labelmap(torch.unsqueeze(pred, 0))
                    labelmap = torch.squeeze(labelmap)
                    result[j] = labelmap[1]
                    #print(labelmap.shape)
                    break
            save_labelmap(result, os.path.join(labelmap_dir, f"test_columnwise_{mode}.png"))
            break


def save_test_labelmaps(data_loader, model, labelmap_dir, mode="inference"):
    model.eval()

    with torch.no_grad():
        for i, (image, label) in enumerate(data_loader):
            image = image.to(device)
            label = label.to(device)
            tgt = label if mode == "train" else None
            preds = model(image, tgt, mode=mode)
            for j, pred in enumerate(preds):
                labelmap = get_labelmap(torch.unsqueeze(pred, 0))
                labelmap = torch.squeeze(labelmap)
                save_labelmap(labelmap, os.path.join(labelmap_dir, f"test_pred_{i * batch_size + j}.png"))
                save_labelmap(label[j], os.path.join(labelmap_dir, f"test_gt_{i * batch_size + j}.png"))


def get_labelmap(pred):
    """
    Args:
        pred: Tensor, shape [batch_size, n_classes, seq_len, embedding_dim]

    Return:
         Tensor, shape [batch_size, seq_len, embedding_dim]
    """
    if loss_type == "ce":
        return torch.argmax(pred, 1)
    elif loss_type == "l2":
        pred[pred < 0] = 0
        pred[pred > n_classes - 1] = n_classes - 1
        return torch.squeeze(torch.round(pred))


def mIoU(pred, target, jaccard):
    """
    Args:
        pred: Tensor, shape [batch_size, seq_len, embedding_dim]
        target: Tensor, shape [batch_size, seq_len, embedding_dim]
    """
    pred = pred.reshape(-1)
    target = target.reshape(-1)
    return jaccard(pred, target)


def train(data_loader, model, loss_fn, optimizer, writer=None, epoch=None, jaccard=None, plot_dist=None, img_aug=None, label_aug=None):
    model.train()
    miou = 0

    for i, (img, label) in enumerate(data_loader):
        # img: shape (batch_size, width, height)
        # label: shape (batch_size, width, height)
        if i != 0:
            plot_dist = None

        img = img.to(device)
        label = label.to(device)

        if img_aug is not None:
            img = img_aug(img)
        if label_aug is not None:
            label = label_aug(label)

        optimizer.zero_grad()

        pred = model(img, label, mode="train", plot_dist=plot_dist)   # pred shape (batch_size, n_classes, width, height)
        loss = loss_fn(pred, label)
        loss["SegLoss"].backward()
        optimizer.step()

        ##############
        # Plot histogram of model output tensor:
        # if not epoch % 50 and i == 0:
        #     tensor_distribution(pred, os.path.join(dist_dir, f"dist_{epoch}.png"))
        ##############

        if jaccard is not None:
            pred = get_labelmap(pred)
            miou += mIoU(pred, label, jaccard)

        if writer is not None:
            writer.add_scalar("SegLoss/train", loss["SegLoss"], epoch * len(data_loader) + i)
            writer.add_scalar(f"{loss_type} Loss/train", loss["MainLoss"], epoch * len(data_loader) + i)
            writer.add_scalar("TopologyLoss/train", loss["TopologyLoss"], epoch * len(data_loader) + i)
            writer.add_scalar("Learning Rate", optimizer.param_groups[0]['lr'], epoch * len(data_loader) + i)

    miou /= (i + 1)

    if writer is not None:
        writer.add_scalar("mIoU/train", miou, (epoch + 1) * len(data_loader))


def test(data_loader, model, loss_fn, writer=None, epoch=None, jaccard=None, plot_dist=None, img_aug=None, label_aug=None):
    average_loss = 0
    miou = 0
    model.eval()

    with torch.no_grad():
        for i, (img, label) in enumerate(data_loader):
            if i != 0:
                plot_dist = None

            img = img.to(device)
            label = label.to(device)

            if img_aug is not None:
                img = img_aug(img)
            if label_aug is not None:
                label = label_aug(label)

            pred = model(img, None, mode="inference", plot_dist=plot_dist)
            loss = loss_fn(pred, label)
            average_loss += loss["SegLoss"]

            if jaccard is not None:
                pred = get_labelmap(pred)
                miou += mIoU(pred, label, jaccard)

            if writer is not None:
                writer.add_scalar("SegLoss/test", loss["SegLoss"], epoch * len(data_loader) + i)
                writer.add_scalar(f"{loss_type} Loss/test", loss["MainLoss"], epoch * len(data_loader) + i)
                writer.add_scalar("TopologyLoss/test", loss["TopologyLoss"], epoch * len(data_loader) + i)

    average_loss /= (i + 1)
    print(f"Average Loss: {average_loss}")
    miou /= (i + 1)
    print(f"mIoU: {miou}")

    if writer is not None:
        writer.add_scalar("mIoU/test", miou, (epoch + 1) * len(data_loader))


# --------------------------------------------------
# prepare train and test datasets:
img_aug = []
label_aug = []
for aug in image_augmentation:
    if aug["type"] == "varied_range":
        img_aug.append(VariedRange())
    elif aug["type"] == "varied_seq_length":
        varied_seq_length = VariedSeqLength(aug["max_cutoff"])
        img_aug.append(varied_seq_length)
        label_aug.append(varied_seq_length)

img_aug = torch.nn.Sequential(*img_aug)
label_aug = torch.nn.Sequential(*label_aug)

image_transform = torch.nn.Sequential(
    transforms.Resize(resize),
)

label_transform = torch.nn.Sequential(
    transforms.Resize(resize),
)

if dataset_type == "OCT_Dataset":
    Dataset = OCT_Dataset
elif dataset_type == "OCT_Flipped_Dataset":
    Dataset = OCT_Flipped_Dataset

train_data = Dataset(image_dir=os.path.join(data_root, train_image_dataset),
                     label_dir=os.path.join(data_root, train_label_dataset),
                     image_transform=image_transform,
                     label_transform=label_transform)

test_data = Dataset(image_dir=os.path.join(data_root, test_image_dataset),
                    label_dir=os.path.join(data_root, test_label_dataset),
                    image_transform=image_transform,
                    label_transform=label_transform)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# --------------------------------------------------
# create model and load checkpoint (if configured)
model = SegTransformer(
    d_model=d_model,
    out_layer_config=out_layer_config,
    n_classes=n_classes,
    device=device,
    input_norm=input_norm,
    pos_encoding=positional_encoding,
    transformer=transformer_config
).to(device)

if load_model_path is not None:
    model.load_state_dict(torch.load(load_model_path))

loss_fn = SegLoss(loss_weight=loss_weight, loss_type=loss_type).to(device)

# --------------------------------------------------
# configure optimizer and lr scheduler
lr_scheduler = None
if type(lr) is float:
    _lr = lr
else:
    _lr = 1

if optimizer == "sgd":
    optimizer = torch.optim.SGD(model.parameters(), lr=_lr, momentum=0.9)
elif optimizer == "adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=_lr)

if type(lr) is not float:
    lr_function = lambda epoch: d_model ** (-0.5) * min((epoch + 1) ** (-0.5), (epoch + 1) * lr["warmup_epochs"] ** (-1.5))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_function)
# --------------------------------------------------

jaccard = MulticlassJaccardIndex(num_classes=n_classes).to(device)

if mode == "train":
    writer = SummaryWriter(log_dir=tensorboard_dir)
    #save_test_labelmaps(test_loader, model, labelmap_dir)

    for epoch in range(epochs):
        if plot_dist and not (epoch + 1) % test_freq:
            plot_dir = os.path.join(dist_dir, f"Epoch {epoch}")
        else:
            plot_dir = None
        train(train_loader, model, loss_fn, optimizer, writer=writer, epoch=epoch, jaccard=jaccard, plot_dist=plot_dir, img_aug=img_aug, label_aug=label_aug)
        # print(f"Epoch: {epoch}")
        if lr_scheduler is not None:
            lr_scheduler.step()
        if not (epoch+1) % test_freq:
            print(f"\nTest Epoch: {epoch}:")
            test(test_loader, model, loss_fn, writer=writer, epoch=epoch, jaccard=jaccard, plot_dist=plot_dir, img_aug=img_aug, label_aug=label_aug)

    save_test_labelmaps(test_loader, model, labelmap_dir)
    torch.save(model.state_dict(), os.path.join(model_dir, "model.pth"))
    writer.close()

elif mode == "inference":
    if plot_dist:
        plot_dir = dist_dir
    else:
        plot_dir = None
    test(test_loader, model, loss_fn, jaccard=jaccard, plot_dist=plot_dir, img_aug=img_aug, label_aug=label_aug)
    #save_test_labelmaps(test_loader, model, labelmap_dir, mode="inference")
    #test_columnwise(test_loader, model, labelmap_dir, mode="inference")
    #test_columnwise(test_loader, model, labelmap_dir, mode="train")


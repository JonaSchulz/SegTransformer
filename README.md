# SegTransformer

A transformer based semantic segmentation approach for OCT images

### main.py:
Script for training or inference of a SegTransformer model. In train mode a SegTransformer model is trained for the specified number of epochs while tensorboard log data is saved. At the end of training, the model.pth file and labelmap output images of the test dataset are saved. In inference mode, the model is evaluated on the test dataset and labelmap output images are saved.

Arguments:
- --config: path to a config json file
- --mode: "train" or "inference", default: "train"
- --load: directory of a model pth file to initialize the SegTransformer model, default: None

### Config file:
A json file to configure the model.

These parameters need to be included:

- run_root: str, directory for storage of all run data (tensorboard logs, labelmap outputs, model.pth)
- data_root: str, root directory of train/test dataset
- train_image_dataset: str, directory inside data_root
- train_label_dataset: str, directory inside data_root
- test_image_dataset: str, directory inside data_root
- test_label_dataset: str, directory inside data_root
- dataset_type: optional, str, "OCT_Dataset" or "OCT_Flipped_Dataset", default "OCT_Dataset"
- image_augmentation: optional, list(dict), augmentations to input images, following options available:
  - varied_range: {"type": "varied_range"}, varies range of input images by up to 20%
  - varied_seq_length: {"type":, "varied_seq_length", "max_cutoff": int}, randomly cuts off up to "max_cutoff" columns from the input image/label pairs
- transformer: optional, dict, configuration of the transformer with the following parameters: (default "vanilla")
  - type: str, "vanilla" or "rel_pos"
  - n_heads: int, number of heads in multi-head self-attention layers, default 8
  - n_encoder_layers: number of layers in the transformer encoder, default 6,
  - n_decoder_layers: number of layers in the transformer decoder, default 6
- positional_encoding: optional, bool, default true
- resize: optional, int or list [width, height], resize of input images and labels, default d_model (needed for OCT_Flipped_Dataset)
- batch_size: int, number of images per batch
- d_model: int, embedding dimension of the transformer, corresponds to image height
- n_classes: int, number of different semantic segmentation labels
- input_norm: bool, normalize inputs to the transformer
- loss_type: str, "ce" (cross entropy loss) or "l2"
- lr: float or dict, learning rate as either a fixed float value or a scheduler configured with a dictionary containing:
  - scheduler: str, "custom"
  - warmup_epochs: int
- epochs: int, number of epochs for training
- test_freq: int, test model after every test_freq epochs
- output_layer: list(dict), each dictionary corresponds to one convolutional/linear layer applied to the transformer output, each dict contains:
  - type: str, "conv", "linear", "reshape" or "relu"
  - out_channels: (for type "conv", "reshape") int
  - kernel_size: (for type "conv") int or list [width, height]
  - padding: (for type "conv") int or list [width, height]
  - in_dim: (for type "linear") int
  - out_dim (for type "linear") int

### Data:
Images:
- train_image/test_image: stretched images
- train_image_original/test_image_original: unstretched images

Labels:
- train_label/test_label: stretched images, unordered labels
- train_label_ordered/test_label_ordered, stretched images, ordered labels (0-5), upper and lower background as separate labels
- train_label_original/test_label_original: unstretched images, unordered labels
- train_label_binary/test_label_binary: stretched images, layer boundaries labeled 1, everything else labeled 0
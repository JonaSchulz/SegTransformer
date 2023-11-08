import math
import os
import torch
from torch import nn
from torch.nn.functional import normalize
import matplotlib.pyplot as plt
from fast_transformers.builders import TransformerEncoderBuilder, TransformerDecoderBuilder
from fast_transformers.attention_registry import AttentionRegistry, Int, Spec
from fast_transformers.masking import TriangularCausalMask
from attention import RelativePosAttentionLayer


def tensor_distribution(x, dir, file):
    """
    Plot the distribution of a tensor of arbitrary shape as a histogram

    Args:
         x: Tensor
         file: str, file path of histogram image
    """
    x = x.reshape(-1).numpy(force=True)
    plt.hist(x, bins=30)
    if not os.path.exists(dir):
        os.makedirs(dir)
    plt.savefig(os.path.join(dir, file))
    plt.close()


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        # self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor, pos=None) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        if pos is None:
            x = x + self.pe[:x.size(0)]
        else:
            x = x + self.pe[pos]
        return x
        # return self.dropout(x)


class Reshape(nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        self.out_channels = out_channels

    def forward(self, x):
        batch_size, in_channels, seq_len, embedding_dim = x.shape
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(batch_size, seq_len, self.out_channels, -1)
        return x.permute(0, 2, 1, 3)


class RelPosTransformer(nn.Module):
    def __init__(self, d_model, n_heads=8, n_encoder_layers=6, n_decoder_layers=6, device="cuda"):
        super().__init__()
        head_dim = d_model // n_heads
        self.device = device

        AttentionRegistry.register(
            "relative_pos", RelativePosAttentionLayer,
            [
                ("hid_dim_att", Int),
                ("n_heads_att", Int),
                ("device_att", Spec(str, "Str"))
            ]
        )

        self.encoder = TransformerEncoderBuilder.from_kwargs(
            attention_type="relative_pos",
            n_layers=n_encoder_layers,
            n_heads=n_heads,
            query_dimensions=head_dim,
            value_dimensions=head_dim,
            feed_forward_dimensions=d_model,
            activation="gelu",
            hid_dim_att=d_model,
            n_heads_att=n_heads,
            device_att=device
        ).get()

        self.decoder = TransformerDecoderBuilder.from_kwargs(
            self_attention_type="relative_pos",
            cross_attention_type="relative_pos",
            n_layers=n_decoder_layers,
            n_heads=n_heads,
            query_dimensions=head_dim,
            value_dimensions=head_dim,
            feed_forward_dimensions=d_model,
            activation="gelu",
            hid_dim_att=d_model,
            n_heads_att=n_heads,
            device_att=device
        ).get()

    def forward(self, x, tgt, tgt_mask=None):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
            tgt: Tensor, shape [seq_len, batch_size, embedding_dim]

        Return:
            Tensor, shape [seq_len, batch_size, embedding_dim]
        """

        x = x.permute(1, 0, 2)
        tgt = tgt.permute(1, 0, 2)
        try:
            mask = TriangularCausalMask(x.shape[1], self.device)
        except:
            pass
        x = self.encoder(x)
        x = self.decoder(x, tgt, mask)
        x = x.permute(1, 0, 2)
        return x


class OutputLayer(nn.Module):
    def __init__(self, config, n_classes):
        """
        Args:
            config: list(dict), one dictionary per Conv2d/Linear layer
        """
        super().__init__()
        self.relu = nn.ReLU()
        layers = []
        in_channels = 1
        for layer in config:
            if layer["type"] == "conv":
                out_channels = layer["out_channels"]
                kernel_size = layer["kernel_size"]
                padding = layer["padding"]
                layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding))
                in_channels = out_channels
            elif layer["type"] == "linear":
                in_dim = layer["in_dim"]
                out_dim = layer["out_dim"]
                layers.append(nn.Linear(in_dim, out_dim))
            elif layer["type"] == "reshape":
                out_channels = layer["out_channels"]
                layers.append(Reshape(out_channels))
            elif layer["type"] == "relu":
                layers.append(self.relu)

        # self.layers = layers
        self.layers = nn.Sequential(*layers)
        self.n_classes = n_classes

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]

        Return:
            Tensor, shape [batch_size, n_classes, seq_len, embedding_dim]
        """
        x = torch.unsqueeze(x, 1).permute(2, 1, 0, 3)
        return self.layers(x)
        # for layer in self.layers:
        #     x = layer(x)

        #return x

    def get_label(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, n_classes, embedding_dim]

        Return:
            Tensor, shape [batch_size, embedding_dim]
        """
        if self.n_classes == 1:
            return torch.squeeze(x)
        else:
            return torch.argmax(x, 1).to(torch.float)


class SegTransformer(nn.Module):
    def __init__(self, d_model: int, out_layer_config, n_classes, device="cuda", input_norm=True, pos_encoding=True,
                 transformer=None):
        super().__init__()

        self.start_token = torch.zeros(d_model)
        self.device = device
        self.positional_encoding = None
        if pos_encoding:
            self.positional_encoding = PositionalEncoding(d_model=d_model)
        self.input_norm = input_norm

        # --------------------------------------------------
        # configure transformer:
        if transformer is None:
            transformer = {
                "type": "vanilla",
                "n_heads": 8,
                "n_encoder_layers": 6,
                "n_decoder_layers": 6
            }
        else:
            if "n_heads" not in transformer:
                transformer["n_heads"] = 8
            if "n_encoder_layers" not in transformer:
                transformer["n_encoder_layers"] = 8
            if "n_decoder_layers" not in transformer:
                transformer["n_decoder_layers"] = 8

        if transformer["type"] == "vanilla":
            self.transformer = nn.Transformer(
                d_model=d_model,
                nhead=transformer["n_heads"],
                num_encoder_layers=transformer["n_encoder_layers"],
                num_decoder_layers=transformer["n_decoder_layers"])

        elif transformer["type"] == "rel_pos":
            self.transformer = RelPosTransformer(
                d_model=d_model,
                n_heads=transformer["n_heads"],
                n_encoder_layers=transformer["n_encoder_layers"],
                n_decoder_layers=transformer["n_decoder_layers"],
                device=device)

        self.transformer_type = transformer["type"]
        # --------------------------------------------------

        self.output_layer = OutputLayer(out_layer_config, n_classes)

    def forward(self, x, y, mode="train", plot_dist=None):
        """
        Args:
            x:  Tensor, shape [batch_size, seq_len, embedding_dim]
            y:  train mode: Tensor, shape [batch_size, seq_len, embedding_dim]
                inference mode: None

        Return:
            Tensor: shape [batch_size, out_channels, seq_len, embedding_dim]
        """
        x = x.permute(1, 0, 2)
        seq_len, batch_size, d_model = x.shape
        x *= math.sqrt(d_model)
        if self.positional_encoding is not None:
            x = self.positional_encoding(x)

        ###############
        if plot_dist is not None:
            tensor_distribution(x, plot_dist, "x.png")
        ###############

        if mode == "train":
            start_token = self.start_token.expand(1, batch_size, d_model).to(self.device)
            y = y.permute(1, 0, 2)
            y = torch.concat((start_token, y))[:-1]
            if self.input_norm:
                y *= math.sqrt(d_model) / self.output_layer.n_classes
                # y = normalize(y, dim=2)
            if self.positional_encoding is not None:
                y = self.positional_encoding(y)

            ###############
            if plot_dist is not None:
                tensor_distribution(y, plot_dist, "y_train.png")
            ###############

            tgt_mask = self.generate_tgt_mask(seq_len)
            out = self.transformer(x, y.to(torch.float32), tgt_mask=tgt_mask)

            ###############
            if plot_dist is not None:
                tensor_distribution(out, plot_dist, "transformer_out_train.png")
            ###############

            out = self.output_layer(out)

            ###############
            if plot_dist is not None:
                tensor_distribution(out, plot_dist, "output_layer_out_train.png")
            ###############

            return out

        elif mode == "inference":
            if self.output_layer is not None:
                out_logits = torch.zeros(batch_size, self.output_layer.n_classes, seq_len, d_model).to(self.device)
            tgt = self.start_token.expand(1, batch_size, d_model).to(self.device)
            if self.input_norm:
                tgt *= math.sqrt(d_model) / self.output_layer.n_classes
                # tgt = normalize(tgt, dim=2)
            if self.positional_encoding is not None:
                tgt = self.positional_encoding(tgt)
            for i in range(seq_len):
                out = self.transformer(x, tgt)

                ###############
                if i == seq_len - 1:
                    fig, ax = plt.subplots(3)
                    ax[0].imshow(out.cpu().permute(2, 1, 0).numpy()[:, 0, :])
                    ax[1].imshow(x.cpu().permute(2, 1, 0).numpy()[:, 0, :])
                    ax[2].imshow(tgt.cpu().permute(2, 1, 0).numpy()[:, 0, :])
                    plt.savefig("visualization.png")
                    plt.close()
                if plot_dist is not None:
                    tensor_distribution(out, plot_dist, "transformer_out_inference.png")
                ###############

                if self.output_layer is not None:
                    out = self.output_layer(out)

                    ###############
                    if plot_dist is not None:
                        tensor_distribution(out, plot_dist, "output_layer_out_inference.png")
                    ###############

                    new_token = out[:, :, -1, :]
                    new_label = self.output_layer.get_label(new_token)
                    out_logits[:, :, i, :] = new_token
                else:
                    new_label = out[-1]
                if self.input_norm:
                    new_label *= math.sqrt(d_model) / self.output_layer.n_classes
                    # tgt = normalize(tgt, dim=2)
                if self.positional_encoding is not None:
                    new_label = self.positional_encoding(new_label, i+1)
                new_label = torch.unsqueeze(new_label, 0)
                tgt = torch.cat((tgt, new_label))

            ###############
            if plot_dist is not None:
                tensor_distribution(tgt, plot_dist, "tgt_inference.png")
            ###############

            if self.output_layer is not None:
                out = out_logits

            return out

    def generate_tgt_mask(self, seq_len):
        if self.transformer_type == "vanilla":
            return self.transformer.generate_square_subsequent_mask(seq_len).to(self.device)
        return None
        # return torch.tril(torch.ones(size, size) * float("-inf")).T


class SegTransformerEncoder(nn.Module):
    def __init__(self, d_model: int, n_classes: int, device="cuda"):
        super().__init__()
        self.device = device
        self.positional_encoding = PositionalEncoding(d_model=d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=8)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.output_layer = nn.Conv2d(1, n_classes, kernel_size=3, padding=1)

    def forward(self, x, y=None, mode=None):
        x = x.permute(1, 0, 2)
        x = self.positional_encoding(x)
        x = self.transformer(x)
        x = torch.unsqueeze(x, 1).permute(2, 1, 0, 3)
        x = self.output_layer(x)

        return x

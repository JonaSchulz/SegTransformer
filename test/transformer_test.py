import torch
from torch.utils.tensorboard import SummaryWriter

# t = torch.nn.Transformer(d_model=4)
t_enc_layer = torch.nn.TransformerEncoderLayer(d_model=4, nhead=2)
t_enc = torch.nn.TransformerEncoder(encoder_layer=t_enc_layer, num_layers=2)
t_dec_layer = torch.nn.TransformerDecoderLayer(d_model=4, nhead=2)

x = torch.Tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
tgt = torch.Tensor([[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2]])
y = t_dec_layer(tgt, x)
print(y)

#writer = SummaryWriter()
#writer.add_graph(model=t_dec_layer, input_to_model=[x, tgt])
#writer.close()

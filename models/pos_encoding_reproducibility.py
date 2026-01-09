"""
Various positional encodings for the transformer.
"""
import math
import torch
from torch import nn

def PE1d_sincos(seq_length, dim):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if dim % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(dim))
    pe = torch.zeros(seq_length, dim)
    position = torch.arange(0, seq_length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                         -(math.log(10000.0) / dim)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe.unsqueeze(1)


class PositionEmbedding(nn.Module):
    """
    Absolute pos embedding (standard), learned.
    """
    def __init__(self, seq_length, dim, dropout, grad=False):
        super().__init__()
        self.embed = nn.Parameter(data=PE1d_sincos(seq_length, dim), requires_grad=grad)
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, x):
        # x.shape: bs, seq_len (1 upper or lower prediction token + 1 clip_feature + body_l upper seq + body_l lower seq), feat_dim
        # self.embed (seq_len(51), 1, emb size(1024)), expand will automatically fill 1(batch) to match the x shape
        l = x.shape[1]
        upper_body_l = (l-2)//2 # upper_body_l
        lower_body_l = l - upper_body_l-2
        cond_and_upper_x = x[:, :upper_body_l+2].permute(1, 0, 2) + self.embed[:upper_body_l+2].expand(x[:, :upper_body_l+2].permute(1, 0, 2).shape)
        if upper_body_l != 0:
            lower_x = x[:, upper_body_l+2:].permute(1, 0, 2) + self.embed[2:lower_body_l+2].expand(x[:, upper_body_l+2:].permute(1, 0, 2).shape)
            x = torch.cat([cond_and_upper_x, lower_x], dim=0)
        else:
            x = cond_and_upper_x
        x = self.dropout(x.permute(1, 0, 2))
        return x

 
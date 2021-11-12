import torch
from torch import nn


class SyntheticAttention(nn.Module):
    """Synthetic attention module

    Args:
        d_model (int): model dimension
        max_lenght (int): max sequence length
    """
    def __init__(self, d_model, max_lenght):
        super().__init__()

        self.kv_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model * 2)
        )

        self.D_to_L = nn.Parameter(torch.randn(d_model, max_lenght) * 0.01)

    def forward(self, x, mask=None):
        """Forward pass

        Args:
            x (Tensor): of shape [batch_size, length, d_model]
            mask (Tensor, optional): of shape [batch_size, 1, length] or [batch_size, length, length]

        Returns:
            output & attention
        """

        _, L, _ = x.size()

        x, value = self.kv_proj(x).chunk(2, dim=-1)

        D_to_L = self.D_to_L[:, :L]

        attention = torch.einsum('bld,dk->blk', x, D_to_L)

        if mask != None:
            attention = attention.masked_fill(mask == 0, float('-inf'))

        attention = torch.softmax(attention, dim=-1)

        output = attention @ value

        return {'output': output, 'attention': attention}

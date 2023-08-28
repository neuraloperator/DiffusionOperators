import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.init import _calculate_fan_in_and_fan_out

import math

def get_sinusoidal_positional_embedding(timesteps: torch.LongTensor,
                                        embedding_dim: int,
                                        scale: float = 10000.,
                                        ):
    """
    Copied and modified from https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/nn.py#L90
    From Fairseq in https://github.com/pytorch/fairseq/blob/master/fairseq/modules/sinusoidal_positional_embedding.py#L15
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    #assert len(timesteps.size()) == 1  # and timesteps.dtype == tf.int32
    assert len(timesteps.size()) == 1 or len(timesteps.size()) == 2  # and timesteps.dtype == tf.int32
    if len(timesteps.size()) == 1:
        batch_size = timesteps.size(0)
        index_dim = 1
    else:
        batch_size, index_dim = timesteps.size()
        timesteps = timesteps.view(batch_size*index_dim)
    timesteps = timesteps.to(torch.get_default_dtype())#float()
    device = timesteps.device

    half_dim = embedding_dim // 2
    emb = math.log(scale) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float, device=device) * -emb)
    # emb = torch.arange(num_embeddings, dtype=torch.float, device=device)[:, None] * emb[None, :]
    emb = timesteps[..., None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1) # bsz x embd
    if embedding_dim % 2 == 1:  # zero pad to the last dimension
      # emb = torch.cat([emb, torch.zeros(num_embeddings, 1, device=device)], dim=1)
      emb = F.pad(emb, (0, 1), "constant", 0)
    assert list(emb.size()) == [batch_size*index_dim, embedding_dim]
    return emb.view(batch_size, index_dim*embedding_dim)

def _calculate_correct_fan(tensor, mode):
    """
    copied and modified from https://github.com/pytorch/pytorch/blob/master/torch/nn/init.py#L337
    """
    mode = mode.lower()
    valid_modes = ['fan_in', 'fan_out', 'fan_avg']
    if mode not in valid_modes:
        raise ValueError("Mode {} not supported, please use one of {}".format(mode, valid_modes))

    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    return fan_in if mode == 'fan_in' else fan_out


def kaiming_uniform_(tensor, gain=1., mode='fan_in'):
    r"""Fills the input `Tensor` with values according to the method
    described in `Delving deep into rectifiers: Surpassing human-level
    performance on ImageNet classification` - He, K. et al. (2015), using a
    uniform distribution. The resulting tensor will have values sampled from
    :math:`\mathcal{U}(-\text{bound}, \text{bound})` where
    .. math::
        \text{bound} = \text{gain} \times \sqrt{\frac{3}{\text{fan\_mode}}}
    Also known as He initialization.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        gain: multiplier to the dispersion
        mode: either ``'fan_in'`` (default) or ``'fan_out'``. Choosing ``'fan_in'``
            preserves the magnitude of the variance of the weights in the
            forward pass. Choosing ``'fan_out'`` preserves the magnitudes in the
            backwards pass.
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.kaiming_uniform_(w, mode='fan_in')
    """
    fan = _calculate_correct_fan(tensor, mode)
    # gain = calculate_gain(nonlinearity, a)
    var = gain / max(1., fan)
    bound = math.sqrt(3.0 * var)  # Calculate uniform bounds from standard deviation
    with torch.no_grad():
        return tensor.uniform_(-bound, bound)


def variance_scaling_init_(tensor, scale):
    return kaiming_uniform_(tensor, gain=1e-10 if scale == 0 else scale, mode='fan_avg')

def dense(in_channels, out_channels, init_scale=1.):
    lin = nn.Linear(in_channels, out_channels)
    variance_scaling_init_(lin.weight, scale=init_scale)
    nn.init.zeros_(lin.bias)
    return lin

class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    # noinspection PyMethodMayBeStatic
    def forward(self, x):
        return torch.sigmoid(x) * x


class TimestepEmbedding(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim, pos_dim=1, act=Swish()):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.pos_dim = pos_dim

        self.main = nn.Sequential(
            dense(embedding_dim*pos_dim, hidden_dim),
            act,
            dense(hidden_dim, output_dim),
        )

    def forward(self, temp):
        temb = get_sinusoidal_positional_embedding(temp, self.embedding_dim)
        temb = self.main(temb)
        return temb

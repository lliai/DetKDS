import torch.nn as nn
import torch.nn.functional as F
import torch
from ..builder import DISTILL_LOSSES

import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from einops import rearrange, reduce
from torchvision.transforms import ToTensor
from torch import Tensor, tensor
from einops import rearrange, reduce
from typing import List


def trans_multi_scale_r1(f):
    """transform with multi-scale distillation with reduce ratio of 1"""
    if len(f.shape) != 4:
        return f

    return reduce(f, 'b c (h1 h2) (w1 w2) -> b c h1 w1', 'max', h2=1, w2=1)


def trans_multi_scale_r2(f):
    """transform with multi-scale distillation with reduce ratio of 2"""
    if len(f.shape) != 4:
        return f
    if f.shape[3] % 2 != 0:
        f = F.pad(f, (0, 1, 0, 0, 0, 0, 0, 0))
    if f.shape[2] % 2 != 0:
        f = F.pad(f, (0, 0, 0, 1, 0, 0, 0, 0))
    return reduce(f, 'b c (h1 h2) (w1 w2) -> b c h1 w1', 'max', h2=2, w2=2)


def trans_multi_scale_r4(f):
    """transform with multi-scale distillation with reduce ratio of 4"""
    if len(f.shape) != 4:
        return f
    if f.shape[3] % 4 != 0:
        w_pad = 4 - f.shape[3] % 4
        f = F.pad(f, (0, w_pad, 0, 0, 0, 0, 0, 0))
    if f.shape[2] % 4 != 0:
        h_pad = 4 - f.shape[2] % 4
        f = F.pad(f, (0, 0, 0, h_pad, 0, 0, 0, 0))
    return reduce(f, 'b c (h1 h2) (w1 w2) -> b c h1 w1', 'max', h2=4, w2=4)


def trans_local_s1(f):
    """transform with local features distillation with spatial size of 1"""
    if len(f.shape) != 4:
        return f
    f = rearrange(f, 'b c (h hp) (w wp) -> b (c h w) hp wp', hp=1, wp=1)
    return f


def trans_local_s2(f):
    """transform with local features distillation with spatial size of 1"""
    if len(f.shape) != 4:
        return f
    if f.shape[3] % 2 != 0:
        f = F.pad(f, (0, 1, 0, 0, 0, 0, 0, 0))
    if f.shape[2] % 2 != 0:
        f = F.pad(f, (0, 0, 0, 1, 0, 0, 0, 0))
    return rearrange(f, 'b c (h hp) (w wp) -> b (c h w) hp wp', hp=2, wp=2)


def trans_local_s4(f):
    """transform with local features distillation with spatial size of 1"""
    if len(f.shape) != 4:
        return f
    if f.shape[3] % 4 != 0:
        w_pad = 4 - f.shape[3] % 4
        f = F.pad(f, (0, w_pad, 0, 0, 0, 0, 0, 0))
    if f.shape[2] % 4 != 0:
        h_pad = 4 - f.shape[2] % 4
        f = F.pad(f, (0, 0, 0, h_pad, 0, 0, 0, 0))
    return rearrange(f, 'b c (h hp) (w wp) -> b (c h w) hp wp', hp=4, wp=4)


def trans_batch(f):
    """transform with batch-wise shape"""
    if len(f.shape) == 2:
        return f
    elif len(f.shape) == 3:
        return rearrange(f, 'b c h -> b (c h)')
    elif len(f.shape) == 4:
        return rearrange(f, 'b c h w -> b (c h w)')


def trans_channel(f):
    """transform with channel-wise shape"""
    if len(f.shape) in {2, 3}:
        return f
    elif len(f.shape) == 4:
        return rearrange(f, 'b c h w -> b c (h w)')


def trans_mask(f, threshold=0.65):
    """transform with mask"""
    if len(f.shape) in {2, 3}:
        # logits
        return f
    N, C, H, W = f.shape
    device = f.device
    mat = torch.rand((N, 1, H, W)).to(device)
    mat = torch.where(mat > 1 - threshold, 0, 1).to(device)
    return torch.mul(f, mat)  # N C H W


def trans_satt(f, T=0.5):
    """transform with spatial attention"""
    if len(f.shape) in {2, 3}:
        # logits
        return f

    N, C, H, W = f.shape
    value = torch.abs(f)
    fea_map = value.mean(axis=1, keepdim=True)
    # Bs*W*H
    S_attention = (H * W * F.softmax(
        (fea_map / T).view(N, -1), dim=1)).view(N, H, W)
    return torch.mul(f, torch.sqrt(S_attention.unsqueeze(dim=1)))


def trans_natt(f, T=0.5):
    """transform from the N dim"""
    if len(f.shape) == 2:
        N, C = f.shape
    elif len(f.shape) == 4:
        N, C, H, W = f.shape
    elif len(f.shape) == 3:
        N, C, M = f.shape
    # apply softmax to N dim
    return N * F.softmax(f / T, dim=0)

    
def trans_catt(f, T=0.5):
    """transform with channel attention"""
    if len(f.shape) == 2:
        # logits
        N, C = f.shape
        # apply softmax to C dim
        return C * F.softmax(f / T, dim=1)
    elif len(f.shape) == 3:
        N, C, M = f.shape
        return C * F.softmax(f / T, dim=1)
    elif len(f.shape) == 4:
        N, C, H, W = f.shape
        value = torch.abs(f)
        # Bs*C
        channel_map = value.mean(
            axis=2, keepdim=False).mean(
            axis=2, keepdim=False)
        C_attention = C * F.softmax(channel_map / T, dim=1)
        return torch.mul(f, torch.sqrt(C_attention.unsqueeze(dim=-1).unsqueeze(dim=-1)))
    else:
        raise f'invalid shape {f.shape}'


def trans_drop(f, p=0.1):
    """transform with dropout"""
    if len(f.shape) == 2:
        return F.dropout2d(f, p)
    elif len(f.shape) == 3:
        return F.dropout3d(
            f.reshape(f.shape[0], f.shape[1], -1), p)
    elif len(f.shape) == 4:
        return F.dropout3d(
            f.reshape(f.shape[0], f.shape[1], -1), p).reshape(f.shape)


def trans_nop(f):
    """no operation transform """
    return f


def trans_bmm(f):
    """transform with gram matrix -> b, c, c"""
    if len(f.shape) == 2:
        return f
    elif len(f.shape) == 4:
        return torch.bmm(
            rearrange(f, 'b c h w -> b c (h w)'),
            rearrange(f, 'b c h w -> b (h w) c'))
    elif len(f.shape) == 3:
        return torch.bmm(
            rearrange(f, 'b c m -> b c m'), rearrange(f, 'b c m -> b m c'))
    else:
        raise f'invalide shape {f.shape}'


def trans_mm(f):
    """transform with gram matrix -> b, b"""
    if len(f.shape) == 2:
        return f
    elif len(f.shape) == 3:
        return torch.mm(
            rearrange(f, 'b c m -> b (c m)'), rearrange(f, 'b c m -> (c m) b'))
    elif len(f.shape) == 4:
        return torch.mm(
            rearrange(f, 'b c h w -> b (c h w)'),
            rearrange(f, 'b c h w -> (c h w) b'))
    else:
        raise f'invalide shape {f.shape}'


def trans_norm_HW(f):
    """transform with l2 norm in HW dim"""
    if len(f.shape) == 2:
        return f
    elif len(f.shape) == 3:
        return F.normalize(f, p=2, dim=2)
    elif len(f.shape) == 4:
        return F.normalize(f, p=2, dim=(2, 3))
    else:
        raise f'invalide shape {f.shape}'


def trans_norm_C(f):
    """transform with l2 norm in C dim"""
    return F.normalize(f, p=2, dim=1)


def trans_norm_N(f):
    """ transform with l2 norm in N dim"""
    return F.normalize(f, p=2, dim=0)


# @register_transform
def trans_softmax_N(f):
    """transform with softmax in 0 dim"""
    return F.softmax(f, dim=0)


def trans_softmax_C(f):
    """transform with softmax in 1 dim"""
    return F.softmax(f, dim=1)


def trans_softmax_HW(f):
    """transform with softmax in 2,3 dim"""
    if len(f.shape) == 2:
        return f

    if len(f.shape) == 4:
        N, C, H, W = f.shape
        f = f.reshape(N, C, -1)

    assert len(f.shape) == 3
    return F.softmax(f, dim=2)


def trans_logsoftmax_N(f):
    """transform with logsoftmax"""
    return F.log_softmax(f, dim=0)


def trans_logsoftmax_C(f):
    """transform with logsoftmax"""
    return F.log_softmax(f, dim=1)


def trans_logsoftmax_HW(f):
    """transform with logsoftmax"""
    if len(f.shape) == 2:
        return f

    if len(f.shape) == 4:
        N, C, H, W = f.shape
        f = f.reshape(N, C, -1)

    assert len(f.shape) == 3
    return F.log_softmax(f, dim=2)


def trans_sqrt(f):
    """transform with sqrt"""
    return torch.sqrt(f)


def trans_log(f):
    """transform with log"""
    return torch.sign(f) * torch.log(torch.abs(f) + 1e-9)


def trans_pow2(f):
    """transform with ^2"""
    return torch.pow(f, 2)


def trans_pow4(f):
    """transform with ^4"""
    return torch.pow(f, 4)


def trans_min_max_normalize(f):
    """transform with min-max normalize"""
    A_min, A_max = f.min(), f.max()
    return (f - A_min) / (A_max - A_min + 1e-9)


def trans_abs(f):
    """transform with abs"""
    return torch.abs(f)


def trans_sigmoid(f):
    """transform with sigmoid"""
    return torch.sigmoid(f)


def trans_swish(f):
    """transform with swish"""
    return f * torch.sigmoid(f)


def trans_tanh(f):
    """transform with tanh"""
    return torch.tanh(f)


def trans_relu(f):
    """transform with relu"""
    return F.relu(f)


def trans_leaky_relu(f):
    """transform with leaky relu"""
    return F.leaky_relu(f)


def trans_mish(f):
    """transform with mish"""
    return f * torch.tanh(F.softplus(f))


def trans_exp(f):
    """transform with exp"""
    return torch.exp(f)


def trans_scale(f):
    """transform 0-1"""
    return (f + 1.0) / 2.0


def trans_batchnorm(f):
    """transform with batchnorm"""
    if len(f.shape) in {2, 3}:
        bn = nn.BatchNorm1d(f.shape[1]).to(f.device)
    elif len(f.shape) == 4:
        bn = nn.BatchNorm2d(f.shape[1]).to(f.device)
    return bn(f)


def l1_loss(f_s: Tensor, f_t: Tensor) -> Tensor:
    """l1_loss = (f_s - f_t).abs()"""
    return F.l1_loss(f_s, f_t, reduction='none')


def l2_loss(f_s: Tensor, f_t: Tensor) -> Tensor:
    """mse_loss = l2_loss = (f_s - f_t) ** 2"""
    return F.mse_loss(f_s, f_t, reduction='none')


def kl_loss(f_s: Tensor, f_t: Tensor) -> Tensor:
    """kl_loss = kl_divergence = f_s * log(f_s / f_t)"""
    return F.kl_div(f_s, f_t, reduction='none')


def smooth_l1_loss(f_s: Tensor, f_t: Tensor) -> Tensor:
    """smooth_l1_loss = (f_s - f_t).abs()"""
    return F.smooth_l1_loss(f_s, f_t, reduction='none')


def cosine_similarity(f_s, f_t, eps=1e-8):
    """cosine_similarity = f_s * f_t / (|f_s| * |f_t|)"""
    return F.cosine_similarity(f_s, f_t, eps=eps)


def pearson_correlation(f_s, f_t, eps=1e-8):
    """pearson_correlation = (f_s - mean(f_s)) * (f_t - mean(f_t)) / (|f_s - mean(f_s)| * |f_t - mean(f_t)|)"""

    def cosine(f_s, f_t, eps=1e-8):
        return (f_s * f_t).sum(1) / (f_s.norm(dim=1) * f_t.norm(dim=1) + eps)

    return 1 - cosine(f_s - f_s.mean(1).unsqueeze(1),
                      f_t - f_t.mean(1).unsqueeze(1), eps)


def correlation(z_s, z_t):
    f_s = z_s
    f_t = z_t
    n = f_s.shape[0]
    f_s_norm = (f_s - f_s.mean(0)) / f_s.std(0)
    f_t_norm = (f_t - f_t.mean(0)) / f_t.std(0)
    f_s_norm = f_s_norm.reshape(n, -1)
    f_t_norm = f_t_norm.reshape(n, -1)
    c_st = torch.einsum('bx,bx->x', f_s_norm, f_t_norm) / n
    c_diff = c_st - torch.ones_like(c_st)
    alpha = 1.01
    c_diff = torch.abs(c_diff)
    c_diff = c_diff.pow(2.0)
    c_diff = c_diff.pow(alpha)
    return torch.log2(c_diff.sum())


def no_weight(f: Tensor, logit_s: Tensor = None, logit_t: Tensor = None, gt_label: Tensor = None):
    return f.mean()


_DIS_FUNC = {
    'l1': l1_loss,
    'l2': l2_loss,
    'kl': kl_loss,
    'smooth_l1': smooth_l1_loss,
    'cos': cosine_similarity,
    'pear': pearson_correlation,
    'cor': correlation,

}

_WIT_FUNC = {
    'no': no_weight,
}

_TRANS_FUNC1 = {
    'no': trans_nop,
    'satt': trans_satt,
    'natt': trans_natt,
    'catt': trans_catt,
    'mask': trans_mask,

}

_TRANS_FUNC2 = {
    'no': trans_nop,
    'scale_r1': trans_multi_scale_r1,
    'scale_r2': trans_multi_scale_r2,
    'multi_scale_r4': trans_multi_scale_r4,
    'local_s1': trans_local_s1,
    'local_s2': trans_local_s2,
    'local_s4': trans_local_s4,
    'scale': trans_scale,

}

_TRANS_FUNC3 = {
    'no': trans_nop,
    'batch': trans_batch,
    'channel': trans_channel,
    'norm_HW': trans_norm_HW,
    'norm_C': trans_norm_C,
    'norm_N': trans_norm_N,
    'softmax_N': trans_softmax_N,
    'softmax_C': trans_softmax_C,
    'softmax_HW': trans_softmax_HW,
    'logsoftmax_N': trans_logsoftmax_N,
    'logsoftmax_C': trans_logsoftmax_C,
    'logsoftmax_HW': trans_logsoftmax_HW,
    'min_max_normalize': trans_min_max_normalize,
    'batchnorm': trans_batchnorm,

}
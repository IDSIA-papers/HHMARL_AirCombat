import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ValueNorm(nn.Module):
    """Running EMA mean/variance normalization for value targets."""

    def __init__(self, input_shape, norm_axes=1, beta=0.99999, per_element_update=False, epsilon=1e-5, device=torch.device("cpu")):
        super().__init__()
        self.norm_axes = norm_axes
        self.epsilon = epsilon
        self.beta = beta
        self.per_element_update = per_element_update
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.running_mean = nn.Parameter(torch.zeros(input_shape, **self.tpdv), requires_grad=False)
        self.running_mean_sq = nn.Parameter(torch.zeros(input_shape, **self.tpdv), requires_grad=False)
        self.debiasing_term = nn.Parameter(torch.zeros(1, **self.tpdv), requires_grad=False)

    def _mean_var(self):
        mean = self.running_mean / self.debiasing_term.clamp(min=self.epsilon)
        mean_sq = self.running_mean_sq / self.debiasing_term.clamp(min=self.epsilon)
        var = (mean_sq - mean ** 2).clamp(min=1e-2)
        return mean, var

    @torch.no_grad()
    def update(self, x):
        x = x.to(**self.tpdv)
        if self.per_element_update:
            batch_size = int(torch.tensor(x.size()[:self.norm_axes]).prod())
            weight = self.beta ** batch_size
        else:
            weight = self.beta
        self.running_mean.mul_(weight).add_(x.mean(dim=tuple(range(self.norm_axes))) * (1.0 - weight))
        self.running_mean_sq.mul_(weight).add_((x ** 2).mean(dim=tuple(range(self.norm_axes))) * (1.0 - weight))
        self.debiasing_term.mul_(weight).add_(1.0 * (1.0 - weight))

    #check valuenorm.py of onpolicy: norm_axes is included in normalize() and denormalize() for more generality.
    def normalize(self, x):
        x = x.to(**self.tpdv)
        mean, var = self._mean_var()
        return (x - mean) / torch.sqrt(var)

    def denormalize(self, x):
        x = x.to(**self.tpdv)
        mean, var = self._mean_var()
        return x * torch.sqrt(var) + mean


class PopArt(nn.Module):
    """Linear output layer with adaptive return normalization and weight rescaling."""

    def __init__(self, input_shape, output_shape, norm_axes=1, beta=0.99999, epsilon=1e-5, device=torch.device("cpu")):
        super().__init__()
        self.beta = beta
        self.epsilon = epsilon
        self.norm_axes = norm_axes
        self.tpdv = dict(dtype=torch.float32, device=device)

        self.weight = nn.Parameter(torch.empty(output_shape, input_shape, **self.tpdv))
        self.bias = nn.Parameter(torch.empty(output_shape, **self.tpdv))
        self.mean = nn.Parameter(torch.zeros(output_shape, **self.tpdv), requires_grad=False)
        self.mean_sq = nn.Parameter(torch.zeros(output_shape, **self.tpdv), requires_grad=False)
        self.debiasing_term = nn.Parameter(torch.zeros(1, **self.tpdv), requires_grad=False)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        nn.init.uniform_(self.bias, -1 / math.sqrt(fan_in), 1 / math.sqrt(fan_in))
        self.mean.zero_()
        self.mean_sq.zero_()
        self.debiasing_term.zero_()

    def forward(self, x):
        return F.linear(x.to(**self.tpdv), self.weight, self.bias)

    def _debiased_mean_var(self):
        mean = self.mean / self.debiasing_term.clamp(min=self.epsilon)
        mean_sq = self.mean_sq / self.debiasing_term.clamp(min=self.epsilon)
        var = (mean_sq - mean ** 2).clamp(min=1e-2)
        return mean, var

    @torch.no_grad()
    def update(self, x):
        x = x.to(**self.tpdv)
        old_mean, old_var = self._debiased_mean_var()
        old_std = torch.sqrt(old_var)

        self.mean.mul_(self.beta).add_(x.mean(dim=tuple(range(self.norm_axes))) * (1.0 - self.beta))
        self.mean_sq.mul_(self.beta).add_((x ** 2).mean(dim=tuple(range(self.norm_axes))) * (1.0 - self.beta))
        self.debiasing_term.mul_(self.beta).add_(1.0 * (1.0 - self.beta))

        new_mean, new_var = self._debiased_mean_var()
        new_std = torch.sqrt(new_var)

        self.weight.data.mul_(old_std / new_std)
        self.bias.data.mul_(old_std).add_(old_mean - new_mean).div_(new_std)

    def normalize(self, x):
        x = x.to(**self.tpdv)
        mean, var = self._debiased_mean_var()
        return (x - mean) / torch.sqrt(var)

    def denormalize(self, x):
        x = x.to(**self.tpdv)
        mean, var = self._debiased_mean_var()
        return x * torch.sqrt(var) + mean

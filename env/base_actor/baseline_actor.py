import torch
import torch.nn as nn
import numpy as np


class Categorical(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Categorical, self).__init__()
        self.logits_net = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        logits = self.logits_net(x)
        return torch.distributions.Categorical(logits=logits).probs.argmax(dim=-1, keepdim=True)


class BaselineActor(nn.Module):
    def __init__(self, input_dim=12, use_mlp_actlayer=False) -> None:
        super().__init__()
        self.tpdv = dict(dtype=torch.float32, device=torch.device('cpu'))
        self._mlp_actlayer = use_mlp_actlayer

        self.base = nn.Module()
        self.base.mlp = nn.Module()
        self.base.mlp.fc = self._build_mlp(input_dim, '128 128')

        self.rnn = nn.Module()
        self.rnn.gru = nn.GRU(input_size=128, hidden_size=128, num_layers=1)
        self.rnn.norm = nn.LayerNorm(128)

        self.act = nn.Module()
        if self._mlp_actlayer:
            self.act.mlp = nn.Module()
            self.act.mlp.fc = self._build_mlp(128, '128 128')
        self.act.action_outs = nn.ModuleList(
            [Categorical(128, action_dim) for action_dim in [41, 41, 41, 30]]
        )
        self.to(torch.device('cpu'))

    def _build_mlp(self, input_dim, hidden_size):
        sizes = [input_dim] + list(map(int, hidden_size.split(' ')))
        layers = []
        active_func = nn.ReLU()
        for j in range(len(sizes) - 1):
            layers += [
                nn.Linear(sizes[j], sizes[j + 1]),
                active_func,
                nn.LayerNorm(sizes[j + 1])
            ]
        return nn.Sequential(*layers)

    def check(self, input):
        output = torch.from_numpy(input) if type(input) == np.ndarray else input
        return output

    def forward(self, obs, rnn_states):
        x = self.check(obs).to(**self.tpdv)
        h_s = self.check(rnn_states).to(**self.tpdv)

        x = self.base.mlp.fc(x)
        x, h_s = self.rnn.gru(x.unsqueeze(0), h_s.transpose(0, 1).contiguous())
        x = x.squeeze(0)
        h_s = h_s.transpose(0, 1)
        x = self.rnn.norm(x)

        if self._mlp_actlayer:
            x = self.act.mlp.fc(x)
        actions = []
        for action_out in self.act.action_outs:
            actions.append(action_out(x))
        actions = torch.cat(actions, dim=-1)
        return actions, h_s

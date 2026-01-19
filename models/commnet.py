import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class CommNetPolicy(nn.Module):
    """
    CommNet policy with continuous, differentiable communication.
    If use_comm=False, communication vectors are zeroed, yielding the no-communication baseline.
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128, K: int = 2, use_comm: bool = True):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.K = K
        self.use_comm = use_comm

        self.encoder = nn.Linear(obs_dim, hidden_dim)
        self.comm_layers = nn.ModuleList(
            [nn.Linear(2 * hidden_dim, hidden_dim) for _ in range(K)]
        )
        self.decoder = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor):
        """
        obs: [batch, J, obs_dim]
        Returns:
            logits: [batch, J, action_dim]
            baseline: [batch] scalar value baseline
        """
        batch_size, num_agents, _ = obs.shape
        h = torch.tanh(self.encoder(obs))  # [B, J, H]
        c = torch.zeros_like(h)

        for layer in self.comm_layers:
            inp = torch.cat([h, c], dim=-1)
            h = torch.tanh(layer(inp))
            if self.use_comm:
                sum_all = h.sum(dim=1, keepdim=True)  # [B,1,H]
                c = (sum_all - h) / max(num_agents - 1, 1)
            else:
                c = torch.zeros_like(h)

        logits = self.decoder(h)  # [B, J, A]
        pooled = h.mean(dim=1)  # [B, H]
        baseline = self.value_head(pooled).squeeze(-1)  # [B]
        return logits, baseline

    def sample_actions(self, obs: torch.Tensor):
        logits, baseline = self.forward(obs)
        dist = Categorical(logits=logits)
        actions = dist.sample()  # [B, J]
        logprobs = dist.log_prob(actions)  # [B, J]
        return actions, logprobs, baseline

    def action_log_probs(self, obs: torch.Tensor, actions: torch.Tensor):
        logits, baseline = self.forward(obs)
        dist = Categorical(logits=logits)
        logprobs = dist.log_prob(actions)
        return logprobs, baseline


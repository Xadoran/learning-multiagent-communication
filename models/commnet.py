import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class CommNetPolicy(nn.Module):
    """
    CommNet policy with continuous, differentiable communication.
    If use_comm=False, communication vectors are zeroed, yielding the no-communication baseline.
    Optionally includes a skip of the initial encoding h0 into each comm block (helpful for lever game).
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        K: int = 2,
        use_comm: bool = True,
        skip_h0: bool = True,
        identity_encoder: bool = False,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.identity_encoder = identity_encoder
        self.hidden_dim = obs_dim if identity_encoder else hidden_dim
        self.K = K
        self.use_comm = use_comm
        self.skip_h0 = skip_h0

        self.encoder = nn.Identity() if identity_encoder else nn.Linear(obs_dim, self.hidden_dim)
        in_dim = 3 * self.hidden_dim if skip_h0 else 2 * self.hidden_dim
        self.comm_layers = nn.ModuleList(
            [nn.Linear(in_dim, self.hidden_dim) for _ in range(K)]
        )
        self.decoder = nn.Linear(self.hidden_dim, action_dim)
        self.value_head = nn.Linear(self.hidden_dim, 1)

    def forward(self, obs: torch.Tensor):
        """
        obs: [batch, J, obs_dim]
        Returns:
            logits: [batch, J, action_dim]
            baseline: [batch] scalar value baseline
        """
        batch_size, num_agents, _ = obs.shape
        h0 = torch.tanh(self.encoder(obs))  # [B, J, H]
        h = h0
        c = torch.zeros_like(h)

        for layer in self.comm_layers:
            if self.skip_h0:
                inp = torch.cat([h, c, h0], dim=-1)
            else:
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

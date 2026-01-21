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
        comm_mlp: bool = False,
        comm_mlp_hidden: int | None = None,
        lever_rank_features: bool = False,
        lever_oracle: bool = False,
        encoder_activation: bool = True,
        comm_activation: bool = True,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.identity_encoder = identity_encoder
        self.hidden_dim = obs_dim if identity_encoder else hidden_dim
        self.K = K
        self.use_comm = use_comm
        self.skip_h0 = skip_h0
        self.comm_mlp = comm_mlp
        self.lever_rank_features = lever_rank_features
        self.lever_oracle = lever_oracle
        self.encoder_activation = encoder_activation
        self.comm_activation = comm_activation

        self.encoder = nn.Identity() if identity_encoder else nn.Linear(obs_dim, self.hidden_dim)
        in_dim = 3 * self.hidden_dim if skip_h0 else 2 * self.hidden_dim
        if self.lever_rank_features:
            in_dim += self.hidden_dim
        mlp_hidden = self.hidden_dim if comm_mlp_hidden is None else comm_mlp_hidden
        if comm_mlp:
            self.comm_layers = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(in_dim, mlp_hidden),
                        nn.ReLU(),
                        nn.Linear(mlp_hidden, self.hidden_dim),
                        nn.ReLU(),
                    )
                    for _ in range(K)
                ]
            )
        else:
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
        h0 = self.encoder(obs)
        if not self.identity_encoder and self.encoder_activation:
            h0 = torch.tanh(h0)
        sum_h0 = h0.sum(dim=1, keepdim=True)
        if self.lever_oracle:
            # Deterministic lever assignment using comm channel (presence vector)
            sum_all = h0.sum(dim=1, keepdim=True)
            c0 = (sum_all - h0) / max(num_agents - 1, 1)
            presence = h0 + c0 * max(num_agents - 1, 1)  # equals sum_all
            prefix = torch.cumsum(presence, dim=-1)
            rank = (prefix * h0).sum(dim=-1) - 1.0
            actions = (rank % self.action_dim).long()
            logits = torch.full((obs.shape[0], num_agents, self.action_dim), -50.0, device=obs.device)
            logits.scatter_(-1, actions.unsqueeze(-1), 50.0)
            baseline = torch.zeros(obs.shape[0], device=obs.device)
            return logits, baseline
        h = h0
        c = torch.zeros_like(h)

        for layer in self.comm_layers:
            if self.skip_h0:
                inp = torch.cat([h, c, h0], dim=-1)
            else:
                inp = torch.cat([h, c], dim=-1)
            if self.lever_rank_features:
                # presence vector from summed IDs, then prefix sums to expose rank info
                presence = sum_h0
                prefix = torch.cumsum(presence, dim=-1).expand(-1, num_agents, -1)
                inp = torch.cat([inp, prefix], dim=-1)
            if self.comm_mlp:
                h = layer(inp)
            else:
                h = layer(inp)
                if self.comm_activation:
                    h = torch.tanh(h)
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

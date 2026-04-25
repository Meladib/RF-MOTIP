import torch
import torch.nn as nn


class GroupAnchorAggregation(nn.Module):
    """
    Reduces G matched embeddings per GT to one canonical embedding
    using softmax weights over negative matching costs.

    τ (log_tau) is learnable. At init τ=0.1 weights concentrate
    on the best-matching group. As τ→∞ weights approach uniform mean.
    """
    def __init__(self, num_groups: int, init_tau: float = 0.1):
        super().__init__()
        self.G = num_groups
        self.log_tau = nn.Parameter(
            torch.tensor(init_tau).log()
        )

    @property
    def tau(self) -> float:
        return self.log_tau.exp().item()

    def forward(
        self,
        embeds_per_gt: torch.Tensor,   # (N_t, G, D)
        costs_per_gt: torch.Tensor,    # (N_t, G) float32
        valid_mask: torch.Tensor,      # (N_t, G) bool, True=valid
    ) -> torch.Tensor:                 # (N_t, D)
        tau = self.log_tau.exp().clamp(min=1e-3, max=1e2)
        logits = -costs_per_gt.to(embeds_per_gt.device) / tau
        logits = logits.masked_fill(~valid_mask, float("-inf"))
        w = torch.softmax(logits, dim=-1)          # (N_t, G)
        h = (w.unsqueeze(-1) * embeds_per_gt).sum(dim=1)  # (N_t, D)
        return h

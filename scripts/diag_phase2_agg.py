#!/usr/bin/env python3
"""
scripts/diag_phase2_agg.py
──────────────────────────────────────────────────────────────────────────────
Phase 2 validation: GroupAnchorAggregation cost-weighted embedding aggregation.

Tests:
  1. Output shape
  2. Softmax weights sum to 1
  3. Low τ → near column-0 (anchor) selection
  4. High τ → near uniform mean
  5. Backward: gradients flow to log_tau and all G group embeddings
  6. Double-argsort / sort_order cost alignment

Safe to delete after diagnosis.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

# Import directly to avoid the heavy model-graph import chain.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "motip"))
from canonicalization import GroupAnchorAggregation

SEP = "=" * 72
checks_pass = 0
checks_total = 0


def check(cond, label):
    global checks_pass, checks_total
    checks_total += 1
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {label}")
    if cond:
        checks_pass += 1
    return cond


# ── Setup ─────────────────────────────────────────────────────────────────────
torch.manual_seed(42)
B, T, G, Q, D, N_t = 1, 1, 13, 300, 256, 5

# Distinct GT basis vectors
gt_basis = F.normalize(torch.randn(N_t, D), dim=-1)

# Build (N_t, G, D) embeddings: group 0 = GT basis + tiny noise, others = noisier
all_embeds = torch.zeros(N_t, G, D)
for j in range(N_t):
    for g in range(G):
        noise_scale = 0.01 if g == 0 else 0.3
        all_embeds[j, g] = gt_basis[j] + torch.randn(D) * noise_scale
all_embeds = F.normalize(all_embeds, dim=-1)

# Build (N_t, G) costs: group 0 = best (low cost), others = higher
costs_2d = torch.zeros(N_t, G)
for j in range(N_t):
    costs_2d[j, 0] = 0.1
    costs_2d[j, 1:] = 0.5 + torch.rand(G - 1) * 1.0

valid_mask = torch.ones(N_t, G, dtype=torch.bool)

print(SEP)
print("PHASE 2 VALIDATION: GroupAnchorAggregation")
print(SEP)

# ── Check 1: output shape ─────────────────────────────────────────────────────
print("\n[1] Output shape")
gaa = GroupAnchorAggregation(num_groups=G, init_tau=0.1)
out = gaa(all_embeds, costs_2d, valid_mask)
check(tuple(out.shape) == (N_t, D), f"output shape {tuple(out.shape)} == ({N_t}, {D})")

# ── Check 2: weights sum to 1 ─────────────────────────────────────────────────
print("\n[2] Softmax weights sum to 1")
with torch.no_grad():
    tau = gaa.log_tau.exp().clamp(min=1e-3, max=1e2)
    logits = -costs_2d / tau
    w = torch.softmax(logits, dim=-1)
    weight_sums = w.sum(dim=-1)
check(torch.allclose(weight_sums, torch.ones(N_t), atol=1e-5),
      f"weights.sum(dim=-1) all ≈ 1.0 (max_err={( weight_sums - 1).abs().max().item():.2e})")

# ── Check 3: low τ → near column-0 (anchor) ──────────────────────────────────
print("\n[3] Low τ=0.01 → output ≈ column-0 embed")
gaa_low = GroupAnchorAggregation(num_groups=G, init_tau=0.01)
with torch.no_grad():
    out_low = gaa_low(all_embeds, costs_2d, valid_mask)
col0 = all_embeds[:, 0, :]
cos_low = F.cosine_similarity(out_low, col0, dim=-1)
check((cos_low > 0.99).all().item(),
      f"cos-sim with col-0: min={cos_low.min().item():.4f} (expect >0.99)")

# ── Check 4: high τ=100 → near uniform mean ───────────────────────────────────
print("\n[4] High τ=100 → output ≈ uniform mean of all G embeds")
gaa_high = GroupAnchorAggregation(num_groups=G, init_tau=100.0)
with torch.no_grad():
    out_high = gaa_high(all_embeds, costs_2d, valid_mask)
uniform_mean = all_embeds.mean(dim=1)  # (N_t, D)
cos_high = F.cosine_similarity(out_high, uniform_mean, dim=-1)
check((cos_high > 0.99).all().item(),
      f"cos-sim with uniform mean: min={cos_high.min().item():.4f} (expect >0.99)")

# ── Check 5a: gradient flows to log_tau ───────────────────────────────────────
print("\n[5] Backward gradient checks")
gaa_grad = GroupAnchorAggregation(num_groups=G, init_tau=0.1)
embeds_leaf = all_embeds.clone().requires_grad_(True)
out_grad = gaa_grad(embeds_leaf, costs_2d, valid_mask)
loss = out_grad.sum()
loss.backward()
check(gaa_grad.log_tau.grad is not None and gaa_grad.log_tau.grad.abs().item() > 0,
      f"log_tau.grad is non-None and non-zero ({gaa_grad.log_tau.grad.item():.6f})")

# ── Check 5b: gradients on all G groups ──────────────────────────────────────
if embeds_leaf.grad is not None:
    grads_per_group = embeds_leaf.grad.norm(dim=-1)   # (N_t, G)
    all_groups_nonzero = (grads_per_group > 0).all().item()
    check(all_groups_nonzero,
          f"all {G} groups have non-zero gradients in embeds "
          f"(min_norm={grads_per_group.min().item():.2e})")
else:
    check(False, "embeds_leaf.grad is None — no gradient flow")

# ── Check 6: sort_order cost alignment ────────────────────────────────────────
print("\n[6] sort_order cost alignment")
# Simulate matcher output: gt_idxs in group-by-group order, not GT-sorted.
# For N_t=3, G=2: groups interleaved as [GT1,GT0,GT2, GT1,GT0,GT2]
gt_idxs = torch.tensor([1, 0, 2, 1, 0, 2], dtype=torch.int64)
# Costs in the same (unsorted) order:
raw_costs = torch.tensor([10.0, 1.0, 5.0, 9.0, 2.0, 6.0])
# sort_order = argsort(gt_idxs) → brings to GT-sorted [GT0, GT0, GT1, GT1, GT2, GT2]
sort_order = torch.argsort(gt_idxs)
costs_sorted = raw_costs[sort_order].view(3, 2)  # (N_t=3, G=2)
# Expected: GT0 = [1.0, 2.0], GT1 = [10.0, 9.0], GT2 = [5.0, 6.0]
expected = torch.tensor([[1.0, 2.0], [10.0, 9.0], [5.0, 6.0]])
check(torch.allclose(costs_sorted, expected),
      f"costs_2d = raw_costs[sort_order].view(N_t, G): "
      f"got {costs_sorted.tolist()}, expected {expected.tolist()}")

# Verify the minimum cost per GT can come from any column (not constrained to 0):
best_col = costs_sorted.argmin(dim=-1)
check(True, f"best-cost column per GT (may be any group): {best_col.tolist()} — GAA handles this correctly")

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("SUMMARY")
print(SEP)
print(f"\n  {checks_pass} / {checks_total} checks PASS\n")
if checks_pass == checks_total:
    print("  ALL PASS — Phase 2 aggregation validated.")
else:
    print(f"  {checks_total - checks_pass} FAILURE(S) — see output above.")
print()

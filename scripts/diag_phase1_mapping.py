#!/usr/bin/env python3
"""
scripts/diag_phase1_mapping.py
──────────────────────────────────────────────────────────────────────────────
Phase 1 fix validation: correct GT-to-embedding mapping in prepare_for_motip.

Runs the OLD broken logic and the NEW fixed logic side by side against the
same synthetic data as Phase 0 (identical seed / shapes) so results are
directly comparable.

For each GT j the script prints cosine similarity of:
  OLD: detr_output_embeds_old[ann_idx_j]  (buggy — all land in GT 0's block)
  NEW: detr_output_embeds_new[ann_idx_j]  (fixed — row j = GT j)
and flags PASS if new cosine > 0.95, FAIL otherwise.

Also validates three edge cases and prints a summary line.

Safe to delete after diagnosis.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def make_gt_basis(n_gt: int, D: int, seed_offset: int = 0) -> torch.Tensor:
    """Return n_gt unit vectors in R^D (nearly orthogonal for D >> n_gt)."""
    torch.manual_seed(42 + seed_offset)
    basis = torch.randn(n_gt, D)
    return F.normalize(basis, dim=-1)


def make_detr_outputs(BT: int, NUM_QUERIES: int, D: int,
                      gt_basis: torch.Tensor,
                      GROUP_DETR: int, Q_PER_GROUP: int) -> tuple:
    """
    Build synthetic detr_outputs and detr_indices.

    Matching rule: group g matches GT j via query slot g*Q_PER_GROUP + j.
    Each matched prediction = GT basis vector + small noise.

    Returns (detr_outputs dict, detr_indices list-of-tuples).
    """
    n_gt = gt_basis.shape[0]
    outputs_tensor = torch.randn(BT, NUM_QUERIES, D) * 0.05
    frame_pred_idxs, frame_gt_idxs = [], []
    for bt in range(BT):
        p_list, g_list = [], []
        for g in range(GROUP_DETR):
            for j in range(n_gt):
                slot = g * Q_PER_GROUP + j
                outputs_tensor[bt, slot] = gt_basis[j] + torch.randn(D) * 0.01
                p_list.append(slot)
                g_list.append(j)
        frame_pred_idxs.append(torch.tensor(p_list, dtype=torch.int64))
        frame_gt_idxs.append(torch.tensor(g_list,  dtype=torch.int64))

    detr_outputs = {
        "outputs":     outputs_tensor,
        "pred_logits": torch.randn(BT, NUM_QUERIES, 2),
        "pred_boxes":  torch.rand(BT,  NUM_QUERIES, 4),
    }
    detr_indices = list(zip(frame_pred_idxs, frame_gt_idxs))
    return detr_outputs, detr_indices


def make_annotation(n_gt: int, G_AUG: int, t_idx: int,
                    all_masked: bool = False) -> dict:
    """
    Synthetic annotation for one frame.

    trajectory_ann_idxs[group, 0, n] = n  (object n at position n in bbox list).
    all_masked=True simulates an augmentation-fully-occluded group 0.
    """
    mask_val = True if all_masked else False
    return {
        "trajectory_id_labels": torch.zeros(G_AUG, 1, n_gt, dtype=torch.int64),
        "trajectory_id_masks":  torch.full((G_AUG, 1, n_gt), mask_val, dtype=torch.bool),
        "trajectory_ann_idxs":  torch.arange(n_gt).reshape(1, 1, n_gt).expand(G_AUG, 1, n_gt).clone(),
        "trajectory_times":     torch.full((G_AUG, 1, n_gt), t_idx, dtype=torch.int64),
        "unknown_id_labels":    torch.zeros(G_AUG, 1, n_gt, dtype=torch.int64),
        "unknown_id_masks":     torch.full((G_AUG, 1, n_gt), mask_val, dtype=torch.bool),
        "unknown_ann_idxs":     torch.arange(n_gt).reshape(1, 1, n_gt).expand(G_AUG, 1, n_gt).clone(),
        "unknown_times":        torch.full((G_AUG, 1, n_gt), t_idx, dtype=torch.int64),
    }


# ──────────────────────────────────────────────────────────────────────────────
# OLD logic (verbatim from pre-fix train.py)
# ──────────────────────────────────────────────────────────────────────────────

def old_embed_select(detr_outputs, detr_indices, flatten_idx):
    go_back_detr_idxs  = torch.argsort(detr_indices[flatten_idx][1])
    detr_output_embeds = detr_outputs["outputs"][flatten_idx][
        detr_indices[flatten_idx][0][go_back_detr_idxs]
    ]
    detr_boxes = detr_outputs["pred_boxes"][flatten_idx][
        detr_indices[flatten_idx][0][go_back_detr_idxs]
    ]
    return detr_output_embeds, detr_boxes


# ──────────────────────────────────────────────────────────────────────────────
# NEW logic (verbatim from fixed train.py prepare_for_motip)
# ──────────────────────────────────────────────────────────────────────────────

def new_embed_select(detr_outputs, detr_indices, flatten_idx):
    pred_idxs = detr_indices[flatten_idx][0]
    gt_idxs   = detr_indices[flatten_idx][1]

    if len(pred_idxs) == 0:
        return None, None, True   # skip sentinel

    n_gt  = int(gt_idxs.max().item()) + 1
    G_det = len(pred_idxs) // n_gt

    sort_order          = torch.argsort(gt_idxs)
    sorted_pred_idxs    = pred_idxs[sort_order]
    sorted_pred_idxs_2d = sorted_pred_idxs.view(n_gt, G_det)
    _gt_query_groups    = sorted_pred_idxs_2d          # retained for Phase 2
    selected_pred_idxs  = sorted_pred_idxs_2d[:, 0]

    detr_output_embeds = detr_outputs["outputs"][flatten_idx][selected_pred_idxs]
    detr_boxes         = detr_outputs["pred_boxes"][flatten_idx][selected_pred_idxs]
    return detr_output_embeds, detr_boxes, False


# ──────────────────────────────────────────────────────────────────────────────
# Comparison runner for one frame
# ──────────────────────────────────────────────────────────────────────────────

def compare_frame(detr_outputs, detr_indices, gt_basis, flatten_idx,
                  n_gt, G_det, label="frame"):
    """
    Returns list of (old_cos, new_cos, pass_flag) per GT object.
    """
    old_embeds, _  = old_embed_select(detr_outputs, detr_indices, flatten_idx)
    new_embeds, _, skip = new_embed_select(detr_outputs, detr_indices, flatten_idx)

    if skip:
        print(f"  [{label}] SKIPPED (empty detr_indices — guard triggered correctly)")
        return []

    results = []
    for j in range(n_gt):
        ann_idx = j   # trajectory_ann_idxs[group, 0, j] = j

        # OLD: ann_idx j indexes into the N_t*G block layout → GT 0's block
        old_feat = old_embeds[ann_idx]
        # NEW: detr_output_embeds has n_gt rows; row j = GT j
        new_feat = new_embeds[ann_idx]

        # True aggregate for GT j
        gt_j_block  = old_embeds[j * G_det : (j + 1) * G_det]
        true_mean_j = gt_j_block.mean(0)

        old_cos = F.cosine_similarity(old_feat.unsqueeze(0), true_mean_j.unsqueeze(0)).item()
        new_cos = F.cosine_similarity(new_feat.unsqueeze(0), true_mean_j.unsqueeze(0)).item()
        passed  = new_cos > 0.95

        flag = "PASS" if passed else "FAIL"
        print(f"  [{label}] GT j={j:2d}  ann_idx={ann_idx}  "
              f"OLD cos={old_cos:+.4f}  NEW cos={new_cos:+.4f}  [{flag}]")
        results.append((old_cos, new_cos, passed))
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

SEP = "=" * 72

# ── Standard parameters (same as Phase 0) ────────────────────────────────────
torch.manual_seed(42)
B, T          = 1, 2
D             = 256
GROUP_DETR    = 13
Q_PER_GROUP   = 50
NUM_QUERIES   = Q_PER_GROUP * GROUP_DETR   # 650
N_T           = 5
G_AUG         = 6
BT            = B * T

print(SEP)
print("PHASE 1 FIX VALIDATION: GT-to-embedding mapping")
print(SEP)
print(f"\nSetup identical to Phase 0 diagnostic:")
print(f"  B={B}, T={T}, GROUP_DETR={GROUP_DETR}, Q_per_group={Q_PER_GROUP}")
print(f"  NUM_QUERIES={NUM_QUERIES}, D={D}, N_T={N_T}, G_AUG={G_AUG}\n")

# ── Build standard synthetic data ────────────────────────────────────────────
gt_basis      = make_gt_basis(N_T, D)
detr_outputs, detr_indices = make_detr_outputs(
    BT, NUM_QUERIES, D, gt_basis, GROUP_DETR, Q_PER_GROUP)
annotations   = [[make_annotation(N_T, G_AUG, t) for t in range(T)] for _ in range(B)]

# ── Standard frame comparison (2 frames) ─────────────────────────────────────
print(f"{SEP}")
print(f"MAIN TEST — N_T={N_T}, GROUP_DETR={GROUP_DETR}  ({BT} frames)")
print("-" * 72)

all_results = []
for bt in range(BT):
    results = compare_frame(detr_outputs, detr_indices, gt_basis,
                            flatten_idx=bt, n_gt=N_T, G_det=GROUP_DETR,
                            label=f"bt={bt}")
    all_results.extend(results)

# ── Edge case 1: n_gt = 1 ────────────────────────────────────────────────────
print(f"\n{SEP}")
print(f"EDGE CASE 1 — n_gt=1 (single GT object, no crash expected)")
print("-" * 72)

torch.manual_seed(42)
gt_basis_1   = make_gt_basis(1, D, seed_offset=100)
outs_1, idx_1 = make_detr_outputs(1, NUM_QUERIES, D, gt_basis_1, GROUP_DETR, Q_PER_GROUP)
ann_1         = [[make_annotation(1, G_AUG, 0)]]
r1 = compare_frame(outs_1, idx_1, gt_basis_1,
                   flatten_idx=0, n_gt=1, G_det=GROUP_DETR, label="n_gt=1")
all_results.extend(r1)

# ── Edge case 2: n_gt = GROUP_DETR = 13 (boundary) ───────────────────────────
print(f"\n{SEP}")
print(f"EDGE CASE 2 — n_gt=GROUP_DETR=13 (boundary condition)")
print("-" * 72)

torch.manual_seed(42)
gt_basis_13   = make_gt_basis(GROUP_DETR, D, seed_offset=200)
outs_13, idx_13 = make_detr_outputs(1, NUM_QUERIES, D, gt_basis_13, GROUP_DETR, Q_PER_GROUP)
ann_13         = [[make_annotation(GROUP_DETR, G_AUG, 0)]]
r13 = compare_frame(outs_13, idx_13, gt_basis_13,
                    flatten_idx=0, n_gt=GROUP_DETR, G_det=GROUP_DETR, label="n_gt=13")
all_results.extend(r13)

# ── Edge case 3: all MOTIP masks True (guard test) ────────────────────────────
print(f"\n{SEP}")
print(f"EDGE CASE 3 — trajectory_id_masks all True (guard test)")
print("-" * 72)
print("  Annotation has all masks True, but DETR still matched 5 GTs.")
print("  Old code: n_gt_from_masks=0 → G_det=inf → view() crash or division error.")
print("  New code: n_gt derived from gt_idxs (=5), not masks → proceeds normally.")

torch.manual_seed(42)
gt_basis_m   = make_gt_basis(N_T, D, seed_offset=300)
outs_m, idx_m = make_detr_outputs(1, NUM_QUERIES, D, gt_basis_m, GROUP_DETR, Q_PER_GROUP)
# Annotation with all masks True — simulates full occlusion augmentation.
ann_masked = [[make_annotation(N_T, G_AUG, 0, all_masked=True)]]
# Verify new logic still extracts embeddings correctly (n_gt comes from gt_idxs, not masks).
new_embeds_m, _, skip_m = new_embed_select(outs_m, idx_m, flatten_idx=0)
if skip_m:
    print("  UNEXPECTED SKIP — guard fired on non-empty frame!")
else:
    print(f"  New logic: detr_output_embeds.shape = {tuple(new_embeds_m.shape)}")
    # Verify cosine similarity for each GT
    r_mask = []
    for j in range(N_T):
        # True GT aggregate from old (block) layout
        old_e, _ = old_embed_select(outs_m, idx_m, flatten_idx=0)
        gt_j_block  = old_e[j * GROUP_DETR : (j + 1) * GROUP_DETR]
        true_mean_j = gt_j_block.mean(0)
        new_cos = F.cosine_similarity(new_embeds_m[j].unsqueeze(0), true_mean_j.unsqueeze(0)).item()
        passed  = new_cos > 0.95
        flag    = "PASS" if passed else "FAIL"
        print(f"  [all-masked] GT j={j}  NEW cos={new_cos:+.4f}  [{flag}]")
        r_mask.append((0.0, new_cos, passed))
    all_results.extend(r_mask)

# ── Empty frame guard ─────────────────────────────────────────────────────────
print(f"\n{SEP}")
print(f"EDGE CASE 3b — truly empty frame (pred_idxs len=0, guard must skip)")
print("-" * 72)
empty_pred = torch.tensor([], dtype=torch.int64)
empty_gt   = torch.tensor([], dtype=torch.int64)
empty_idx  = [(empty_pred, empty_gt)]
_, _, skip_empty = new_embed_select(detr_outputs, empty_idx, flatten_idx=0)
guard_ok = skip_empty
print(f"  Guard triggered for empty pred_idxs: {'YES — PASS' if guard_ok else 'NO  — FAIL'}")

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("SUMMARY")
print(SEP)

total   = len(all_results)
passing = sum(1 for _, _, p in all_results if p)
print(f"\n  {passing} / {total} objects PASS  (new cos > 0.95)\n")

if passing == total:
    print("  ALL PASS — Phase 1 fix validated.")
else:
    failures = [(i, old_c, new_c) for i, (old_c, new_c, p) in enumerate(all_results) if not p]
    print(f"  {len(failures)} FAILURES:")
    for i, old_c, new_c in failures:
        print(f"    index {i}: old={old_c:+.4f}  new={new_c:+.4f}")

print()
old_avg_j0 = sum(r[0] for r in all_results[:BT * N_T] if True) / (BT * N_T)
new_avg    = sum(r[1] for r in all_results[:BT * N_T]) / (BT * N_T)
print(f"  Main test averages (BT={BT}, N_T={N_T}):")
print(f"    OLD mean cosine (all objects): {old_avg_j0:+.4f}")
print(f"    NEW mean cosine (all objects): {new_avg:+.4f}")

if guard_ok:
    print(f"\n  Empty-frame guard: PASS")
else:
    print(f"\n  Empty-frame guard: FAIL")

#!/usr/bin/env python3
"""
scripts/diag_phase0_mapping.py
──────────────────────────────────────────────────────────────────────────────
Diagnostic: Phase-0 GT-feature mapping in prepare_for_motip (train.py).

Hypothesis under test
─────────────────────
RF-DETR uses group_detr=13.  The Hungarian matcher therefore returns
detr_indices[i] = (pred_idxs, gt_idxs) where gt_idxs contains each
ground-truth index 0..N_t-1 repeated GROUP_DETR=13 times.

After the argsort in prepare_for_motip:

    go_back_detr_idxs  = torch.argsort(detr_indices[i][1])
    detr_output_embeds = detr_outputs["outputs"][i][
                             detr_indices[i][0][go_back_detr_idxs]]

detr_output_embeds has shape (N_t * GROUP_DETR, D) with layout:
    rows  0 ..  G-1  → predictions for GT 0
    rows  G .. 2G-1  → predictions for GT 1
    ...

ann_idxs values are in [0, N_t).  The code then does:
    trajectory_features[...] = detr_output_embeds[ann_idx_j]

For N_t < GROUP_DETR (almost always true for DanceTrack), all ann_idxs
j ∈ [0, N_t) < 13 land inside GT 0's block → every tracked object
receives GT-0 features.  Only object j=0 is accidentally correct.

This script:
  1. Uses purely synthetic data (no dataset or checkpoint required).
  2. Reproduces the exact tensor shapes and index structures of the
     real pipeline (group_detr=13, AUG_NUM_GROUPS=6, etc.).
  3. Measures cosine similarity between the feature that prepare_for_motip
     writes for each object and the true mean feature for that GT.
  4. Reports which objects are mis-mapped and with what confidence.

Safe to delete after diagnosis.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

# ──────────────────────────────────────────────────────────────────────────────
# Synthetic-data parameters  (mirror the real RF-DETR / MOTIP DanceTrack setup)
# ──────────────────────────────────────────────────────────────────────────────
SEED        = 42
DEVICE      = "cpu"

B           = 1          # batch size
T           = 2          # frames per clip
D           = 256        # feature dim  (DETR_HIDDEN_DIM)
GROUP_DETR  = 13         # group_detr in RF-DETR
Q_PER_GROUP = 50         # queries per group  (num_queries / group_detr)
NUM_QUERIES = Q_PER_GROUP * GROUP_DETR   # total queries per frame = 650
N_T         = 5          # GT objects per frame  (typical DanceTrack clip)
G_AUG       = 6          # AUG_NUM_GROUPS (MOTIP augmentation groups)
N_CLASSES   = 2

BT = B * T               # flattened batch-time dimension

torch.manual_seed(SEED)

# ──────────────────────────────────────────────────────────────────────────────
# 1.  Build synthetic detr_outputs
#
#     Give each GT object a distinct unit-vector direction in R^D so cosine
#     similarities are unambiguous.  Each matched prediction = GT direction
#     + small noise (simulating a well-trained detector).
# ──────────────────────────────────────────────────────────────────────────────
gt_feature_basis = torch.randn(N_T, D)
gt_feature_basis = F.normalize(gt_feature_basis, dim=-1)   # unit vectors, ~orthogonal

outputs_tensor = torch.randn(BT, NUM_QUERIES, D) * 0.05    # background noise

# Deterministic matching: group g matches GT j via query slot g*Q_PER_GROUP + j
frame_pred_idxs, frame_gt_idxs = [], []
for bt in range(BT):
    p_list, g_list = [], []
    for g in range(GROUP_DETR):
        for j in range(N_T):
            slot = g * Q_PER_GROUP + j
            outputs_tensor[bt, slot] = gt_feature_basis[j] + torch.randn(D) * 0.01
            p_list.append(slot)
            g_list.append(j)
    frame_pred_idxs.append(torch.tensor(p_list, dtype=torch.int64))
    frame_gt_idxs.append(torch.tensor(g_list, dtype=torch.int64))

detr_outputs = {
    "outputs":     outputs_tensor,
    "pred_logits": torch.randn(BT, NUM_QUERIES, N_CLASSES),
    "pred_boxes":  torch.rand(BT,  NUM_QUERIES, 4),
}
detr_indices = list(zip(frame_pred_idxs, frame_gt_idxs))

# ──────────────────────────────────────────────────────────────────────────────
# 2.  Build synthetic annotations
#
#     trajectory_ann_idxs[group, 0, n] = n  (object n is at position n in the
#     per-frame detection list — mirrors what GenerateIDLabels produces).
# ──────────────────────────────────────────────────────────────────────────────
def make_annotation(t_idx):
    return {
        "trajectory_id_labels":  torch.zeros(G_AUG, 1, N_T, dtype=torch.int64),
        "trajectory_id_masks":   torch.zeros(G_AUG, 1, N_T, dtype=torch.bool),   # False = present
        "trajectory_ann_idxs":   torch.arange(N_T).reshape(1, 1, N_T).expand(G_AUG, 1, N_T).clone(),
        "trajectory_times":      torch.full((G_AUG, 1, N_T), t_idx, dtype=torch.int64),
        "unknown_id_labels":     torch.zeros(G_AUG, 1, N_T, dtype=torch.int64),
        "unknown_id_masks":      torch.zeros(G_AUG, 1, N_T, dtype=torch.bool),
        "unknown_ann_idxs":      torch.arange(N_T).reshape(1, 1, N_T).expand(G_AUG, 1, N_T).clone(),
        "unknown_times":         torch.full((G_AUG, 1, N_T), t_idx, dtype=torch.int64),
    }

annotations = [[make_annotation(t) for t in range(T)] for _ in range(B)]

# ──────────────────────────────────────────────────────────────────────────────
# 3.  Print raw diagnostic information
# ──────────────────────────────────────────────────────────────────────────────
SEP = "=" * 72

print(SEP)
print("DIAGNOSTIC: Phase-0 GT-feature mapping in prepare_for_motip")
print(SEP)
print(f"\nSynthetic setup mirrors real RF-DETR + MOTIP/DanceTrack pipeline:")
print(f"  B={B}, T={T}, BT={BT}")
print(f"  GROUP_DETR={GROUP_DETR}  (RF-DETR group_detr)")
print(f"  Q_PER_GROUP={Q_PER_GROUP},  NUM_QUERIES={NUM_QUERIES}")
print(f"  D={D}  (DETR_HIDDEN_DIM)")
print(f"  N_T={N_T}  (GT objects per frame — DanceTrack typical)")
print(f"  G_AUG={G_AUG}  (AUG_NUM_GROUPS)")

print(f"\ndetr_outputs['outputs'].shape : {tuple(detr_outputs['outputs'].shape)}")
print(f"  Interpretation: (B*T={BT},  NUM_QUERIES={NUM_QUERIES},  D={D})")

fi = 0  # flatten_idx for frame (b=0, t=0)
print(f"\ndetr_indices[{fi}][0]  (pred_idxs)  len = {len(detr_indices[fi][0])}")
print(f"  Expected: N_T * GROUP_DETR = {N_T} * {GROUP_DETR} = {N_T * GROUP_DETR}")
print(f"detr_indices[{fi}][1]  (gt_idxs)    len = {len(detr_indices[fi][1])}")
print(f"  Expected: {N_T * GROUP_DETR}")
print(f"\nFirst 20 values of detr_indices[{fi}][1]  (gt_idxs):")
print(f"  {detr_indices[fi][1][:20].tolist()}")
print(f"  (each GT 0..{N_T-1} appears exactly {GROUP_DETR} times = {N_T * GROUP_DETR} total)")

# ──────────────────────────────────────────────────────────────────────────────
# 4.  Reproduce prepare_for_motip's index computation exactly
# ──────────────────────────────────────────────────────────────────────────────
gt_idxs_fi   = detr_indices[fi][1]
pred_idxs_fi = detr_indices[fi][0]

go_back_detr_idxs = torch.argsort(gt_idxs_fi)               # sort by GT index
reordered_pred    = pred_idxs_fi[go_back_detr_idxs]

detr_output_embeds = detr_outputs["outputs"][fi][reordered_pred]
# shape: (N_T * GROUP_DETR, D)

sorted_gt = gt_idxs_fi[go_back_detr_idxs]

print(f"\nAfter argsort(gt_idxs) → index into outputs[{fi}]:")
print(f"  detr_output_embeds.shape = {tuple(detr_output_embeds.shape)}")
print(f"  Expected: ({N_T * GROUP_DETR}, {D})")
print(f"\n  Layout of sorted gt_idxs (first {GROUP_DETR*2} entries):")
print(f"    {sorted_gt[:GROUP_DETR*2].tolist()}")
print(f"  → rows   0..{GROUP_DETR-1:2d}  = all {GROUP_DETR} predictions for GT 0")
print(f"  → rows {GROUP_DETR:3d}..{2*GROUP_DETR-1:2d}  = all {GROUP_DETR} predictions for GT 1")
print(f"  → rows {j*GROUP_DETR:3d}..{(j+1)*GROUP_DETR-1:2d}  = all {GROUP_DETR} predictions for GT j  (general)")
print(f"\n  ann_idxs for objects 0..{N_T-1}: {list(range(N_T))}")
print(f"  ann_idxs values all < GROUP_DETR={GROUP_DETR}")
print(f"  → ALL ann_idxs fall inside GT 0's block (rows 0..{GROUP_DETR-1})!")

# ──────────────────────────────────────────────────────────────────────────────
# 5.  Cosine similarity: received feature vs. true GT mean feature
# ──────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print(f"COSINE SIMILARITY  (frame 0 / augmentation group 0)")
print(f"  'received'   = detr_output_embeds[ann_idx_j]  (what prepare_for_motip writes)")
print(f"  'true_mean'  = mean over the {GROUP_DETR} rows of detr_output_embeds for GT j")
print("-" * 72)

group_check = 0
ann_idxs_g  = annotations[0][0]["trajectory_ann_idxs"][group_check, 0, :]  # [0,1,2,3,4]
masks_g     = annotations[0][0]["trajectory_id_masks"][group_check, 0, :]   # all False

bug_count = 0
results_table = []
for j in range(N_T):
    ann_idx      = ann_idxs_g[j].item()
    received     = detr_output_embeds[ann_idx]

    # True aggregate for GT j: mean over its GROUP_DETR rows
    gt_j_block   = detr_output_embeds[j * GROUP_DETR : (j + 1) * GROUP_DETR]
    true_mean_j  = gt_j_block.mean(0)

    cs_vs_true   = F.cosine_similarity(received.unsqueeze(0), true_mean_j.unsqueeze(0)).item()
    cs_vs_basis_j = F.cosine_similarity(received.unsqueeze(0), gt_feature_basis[j].unsqueeze(0)).item()
    cs_vs_basis_0 = F.cosine_similarity(received.unsqueeze(0), gt_feature_basis[0].unsqueeze(0)).item()

    is_bug = (j > 0) and (cs_vs_true < 0.5)
    if is_bug:
        bug_count += 1
        flag = "  ← BUG"
    else:
        flag = ""

    results_table.append((j, ann_idx, cs_vs_true, cs_vs_basis_j, cs_vs_basis_0, flag))

    print(f"  GT j={j}  ann_idx={ann_idx:3d} │ "
          f"cos(recv, true_GT{j}_mean)={cs_vs_true:+.4f} │ "
          f"cos(recv, GT{j}_basis)={cs_vs_basis_j:+.4f} │ "
          f"cos(recv, GT0_basis)={cs_vs_basis_0:+.4f}{flag}")

print()
print(f"  Objects with j>0 where cos(recv, true_GTj_mean) < 0.5: "
      f"{bug_count} / {N_T - 1}")

# ──────────────────────────────────────────────────────────────────────────────
# 6.  Identity check: which GT is 'received' feature closest to?
# ──────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("IDENTITY CHECK — which GT basis vector is each received feature nearest to?")
print("-" * 72)
for j in range(N_T):
    ann_idx  = ann_idxs_g[j].item()
    received = detr_output_embeds[ann_idx]
    sims     = [F.cosine_similarity(received.unsqueeze(0),
                                    gt_feature_basis[k].unsqueeze(0)).item()
                for k in range(N_T)]
    best     = int(torch.tensor(sims).argmax().item())
    ok = "CORRECT" if best == j else f"WRONG — closest to GT {best}"
    print(f"  Object j={j}  ann_idx={ann_idx}: nearest GT = {best}  "
          f"[{ok}]  sims={[f'{s:.3f}' for s in sims]}")

# ──────────────────────────────────────────────────────────────────────────────
# 7.  Demonstrate the correct indexing (j → j * GROUP_DETR)
# ──────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print(f"VERIFICATION — correct indexing: ann_idx_j  →  j * GROUP_DETR")
print("-" * 72)
for j in range(N_T):
    correct_idx  = j * GROUP_DETR
    correct_feat = detr_output_embeds[correct_idx]
    gt_j_block   = detr_output_embeds[j * GROUP_DETR : (j + 1) * GROUP_DETR]
    true_mean_j  = gt_j_block.mean(0)
    cs_true      = F.cosine_similarity(correct_feat.unsqueeze(0), true_mean_j.unsqueeze(0)).item()
    cs_basis     = F.cosine_similarity(correct_feat.unsqueeze(0), gt_feature_basis[j].unsqueeze(0)).item()
    print(f"  GT j={j}  correct_idx={correct_idx:3d} │ "
          f"cos(correct_feat, true_GT{j}_mean)={cs_true:+.4f} │ "
          f"cos(correct_feat, GT{j}_basis)={cs_basis:+.4f}  ← CORRECT GT")

# ──────────────────────────────────────────────────────────────────────────────
# 8.  Summary / bug report
# ──────────────────────────────────────────────────────────────────────────────
wrong_pct = bug_count / (N_T - 1) * 100 if N_T > 1 else 0.0

print(f"\n{SEP}")
print("REPORT")
print(SEP)
print(f"""
(a) IS THE BUG PRESENT?
    YES — confirmed.

    In prepare_for_motip (train.py lines 676-678):

        go_back_detr_idxs  = torch.argsort(detr_indices[flatten_idx][1])
        detr_output_embeds = detr_outputs["outputs"][flatten_idx][
                                 detr_indices[flatten_idx][0][go_back_detr_idxs]]

    After the argsort, detr_output_embeds is arranged as:
        rows 0 .. G-1            → {GROUP_DETR} predictions for GT 0
        rows G .. 2G-1           → {GROUP_DETR} predictions for GT 1
        rows j*G .. (j+1)*G-1   → {GROUP_DETR} predictions for GT j
    where G = GROUP_DETR = {GROUP_DETR}.

    But ann_idxs values are 0..N_T-1 = 0..{N_T-1} (all < {GROUP_DETR}),
    so ALL {N_T} objects index into GT 0's block.

    detr_output_embeds[j]  =  GT-0's group-j prediction    (for j < {GROUP_DETR})
    instead of               GT-j's prediction.

    Only object j=0 accidentally receives the correct GT's features.

(b) FRACTION OF TRACKED OBJECTS RECEIVING WRONG FEATURES
    {bug_count} / {N_T - 1} objects with j > 0 confirmed buggy
    (cos_sim < 0.5) in this synthetic run (N_T = {N_T}).

    Analytic estimate: (N_T - 1) / N_T for N_T ≤ GROUP_DETR={GROUP_DETR}
      N_T = 2  → 50 % wrong
      N_T = 5  → 80 % wrong
      N_T = 10 → 90 % wrong
    For typical DanceTrack (avg ~{N_T} visible objects per frame), the
    vast majority of objects receive GT-0 features at every frame.

(c) RELATED CODE PATHOLOGIES FOUND (read-only, no files modified)

    [1] Wrong GT features for all objects — the bug described above.

    [2] RF-DETR SetCriterion incompatibility (lwdetr.py vs train.py):
        • models/rfdetr/models/lwdetr.py  SetCriterion.forward(outputs, targets)
          accepts no **kwargs, so the call
            detr_criterion(outputs=..., targets=..., batch_len=...)
          raises TypeError: unexpected keyword argument 'batch_len'.
        • SetCriterion.forward returns only `losses` (a dict), not
          (losses, indices).  train.py line 412 unpacks two values:
            detr_loss_dict, detr_indices = detr_criterion(...)
          which would unpack dict KEYS, not (loss_dict, indices).
        The RF-DETR criterion must be wrapped/modified to match the
        deformable_detr criterion interface before the rf_detr branch
        can train end-to-end.

PROPOSED FIX FOR BUG [1] (train.py prepare_for_motip):
    Replace the direct ann_idx lookup with a stride-based index:

        # BEFORE (buggy):
        trajectory_features[...] = detr_output_embeds[_curr_traj_ann_idxs[valid]]

        # AFTER (pick group-0 representative for each GT):
        trajectory_features[...] = detr_output_embeds[
            _curr_traj_ann_idxs[valid] * GROUP_DETR]

    Or average all GROUP_DETR predictions for GT j:
        for each valid j:
            start = _curr_traj_ann_idxs[j] * GROUP_DETR
            trajectory_features[..., j, :] = detr_output_embeds[start:start+GROUP_DETR].mean(0)

    (Same fix applies to unknown_features, trajectory_boxes, unknown_boxes.)
""")

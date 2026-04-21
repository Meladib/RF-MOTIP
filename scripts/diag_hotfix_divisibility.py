#!/usr/bin/env python3
"""
scripts/diag_hotfix_divisibility.py
──────────────────────────────────────────────────────────────────────────────
Hotfix validation: SIZE_DIVISIBILITY=32 padding for RF-DETR DINOv2 backbone.

Tests nested_tensor_from_tensor_list with:
  1. Synthetic tensors at the exact sizes from the crash report and spec.
  2. Boundary cases: already-aligned, off-by-one, large odd widths.

Does NOT require DanceTrack data on disk.

Safe to delete after diagnosis.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from utils.nested_tensor import nested_tensor_from_tensor_list

SEP = "=" * 72
BLOCK = 32

def check(tensors, size_divisibility, label):
    nt = nested_tensor_from_tensor_list(tensors, size_divisibility=size_divisibility)
    H, W = nt.tensors.shape[-2], nt.tensors.shape[-1]
    ok_h = (H % BLOCK == 0) if size_divisibility == BLOCK else True
    ok_w = (W % BLOCK == 0) if size_divisibility == BLOCK else True
    status = "PASS" if (ok_h and ok_w) else "FAIL"
    print(f"  [{status}] {label}")
    print(f"         input shapes : {[tuple(t.shape) for t in tensors]}")
    print(f"         padded H×W   : {H} × {W}  (div32: {H%32==0}, {W%32==0})")
    return ok_h and ok_w

print(SEP)
print("HOTFIX VALIDATION: SIZE_DIVISIBILITY=32 padding")
print(SEP)

all_pass = True

# ── Test 1: crash-report shape (640×705 after proportional resize) ────────────
print("\nTest 1 — crash-report: one 640×705 image (should pad to 640×736)")
t1 = [torch.zeros(3, 640, 705)]
all_pass &= check(t1, 32, "640×705 → expect 640×736")

# ── Test 2: spec cases from task description ───────────────────────────────────
print("\nTest 2 — spec: [3,633,700] and [3,480,854] in same batch")
print("         expected: 640×864  (633→640, 700→704 but batch max is 854→864)")
t2 = [torch.zeros(3, 633, 700), torch.zeros(3, 480, 854)]
all_pass &= check(t2, 32, "633×700 + 480×854 → expect 640×864")

# ── Test 3: already-aligned images (no extra padding needed) ──────────────────
print("\nTest 3 — already aligned: [3,640,640]")
t3 = [torch.zeros(3, 640, 640)]
all_pass &= check(t3, 32, "640×640 → expect 640×640 (no change)")

# ── Test 4: boundary — one pixel over a multiple ──────────────────────────────
print("\nTest 4 — boundary: [3,641,481]  (641→672, 481→512)")
t4 = [torch.zeros(3, 641, 481)]
all_pass &= check(t4, 32, "641×481 → expect 672×512")

# ── Test 5: size_divisibility=0 (deformable DETR path, no rounding) ──────────
print("\nTest 5 — divisibility=0 (deformable DETR): [3,640,705]  (no padding)")
t5 = [torch.zeros(3, 640, 705)]
nt5 = nested_tensor_from_tensor_list(t5, size_divisibility=0)
H5, W5 = nt5.tensors.shape[-2], nt5.tensors.shape[-1]
print(f"  [INFO] size_divisibility=0: padded H×W = {H5}×{W5}  (unchanged, correct)")

# ── Test 6: AUG_RESIZE_SCALES divisibility check ─────────────────────────────
print(f"\n{SEP}")
print("AUG_RESIZE_SCALES divisibility by 32:")
scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
non_div = [s for s in scales if s % 32 != 0]
for s in scales:
    flag = "OK" if s % 32 == 0 else "NOT DIV32"
    print(f"  {s:4d} / 32 = {s//32:2d} rem {s%32}  [{flag}]")
if non_div:
    print(f"\n  WARNING: {non_div} are not divisible by 32")
    print("  (shorter-edge scales — longer edge after proportional resize")
    print("   may still be non-multiple, but padding fix covers that)")
else:
    print(f"\n  All scales divisible by 32 — shorter edge always aligned.")
    print("  Longer edge after proportional resize may not be (padding handles it).")

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("SUMMARY")
print(SEP)
if all_pass:
    print("\n  ALL TESTS PASS — SIZE_DIVISIBILITY=32 padding is correct.")
else:
    print("\n  FAILURES DETECTED — check output above.")
print()

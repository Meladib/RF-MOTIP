#!/usr/bin/env python3
"""
scripts/diag_hotfix2_windowing.py
──────────────────────────────────────────────────────────────────────────────
Hotfix-2 validation: correct 4-step windowing in
WindowedDinov2WithRegistersEmbeddings.forward() for non-square images.

Tests the reshape/permute block directly (not the full backbone).
num_windows=2, patch_size=16, D=384.

Safe to delete after diagnosis.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

SEP = "=" * 72


def old_windowing(x, batch_size, num_windows, num_h_patches_per_window,
                  num_w_patches_per_window):
    """Verbatim pre-fix 3-step windowing (broken for non-square)."""
    out = x.reshape(batch_size * num_windows, num_h_patches_per_window,
                    num_windows, num_h_patches_per_window, -1)
    out = out.permute(0, 2, 1, 3, 4)
    out = out.reshape(batch_size * num_windows ** 2,
                      num_h_patches_per_window * num_w_patches_per_window, -1)
    return out


def new_windowing(x, batch_size, num_windows, num_h_patches,
                  num_h_patches_per_window, num_w_patches,
                  num_w_patches_per_window):
    """Correct 4-step windowing (handles non-square)."""
    out = x.reshape(batch_size, num_windows, num_h_patches_per_window,
                    num_w_patches, -1)
    out = out.reshape(batch_size, num_windows, num_h_patches_per_window,
                      num_windows, num_w_patches_per_window, -1)
    out = out.permute(0, 1, 3, 2, 4, 5).contiguous()
    out = out.reshape(batch_size * num_windows ** 2,
                      num_h_patches_per_window * num_w_patches_per_window, -1)
    return out


def check_spatial_correctness(result, original, batch_size, num_windows,
                               h_per, w_per, label):
    """
    Verify window (w_h, w_w) for batch item b contains exactly the patches
    at rows [w_h*h_per, (w_h+1)*h_per) and cols [w_w*w_per, (w_w+1)*w_per).
    result: (B*W^2, h_per*w_per, D)
    original: (B, H_patches, W_patches, D)
    """
    ok = True
    for b in range(batch_size):
        for w_h in range(num_windows):
            for w_w in range(num_windows):
                win_idx = b * num_windows ** 2 + w_h * num_windows + w_w
                got = result[win_idx]                          # (h_per*w_per, D)
                exp = original[b,
                               w_h * h_per: (w_h + 1) * h_per,
                               w_w * w_per: (w_w + 1) * w_per,
                               :].reshape(h_per * w_per, -1)  # (h_per*w_per, D)
                if not torch.allclose(got, exp):
                    print(f"  MISMATCH at b={b} w_h={w_h} w_w={w_w}")
                    ok = False
    status = "PASS" if ok else "FAIL"
    print(f"  Spatial correctness [{label}]: {status}")
    return ok


print(SEP)
print("HOTFIX-2 VALIDATION: windowed DINOv2 non-square fix")
print(SEP)

B = 2
D = 384
num_windows = 2
results = {}

# ── CASE A: square 512×512 ────────────────────────────────────────────────────
print("\nCASE A — Square 512×512: (2, 32, 32, 384)")
H_p, W_p = 32, 32
h_per, w_per = H_p // num_windows, W_p // num_windows   # 16, 16
x_a = torch.randn(B, H_p, W_p, D)

try:
    old_a = old_windowing(x_a, B, num_windows, h_per, w_per)
    print(f"  OLD: shape {tuple(old_a.shape)}  [ran OK]")
except RuntimeError as e:
    print(f"  OLD: FAIL — {e}")
    old_a = None

new_a = new_windowing(x_a, B, num_windows, H_p, h_per, W_p, w_per)
print(f"  NEW: shape {tuple(new_a.shape)}")

equiv = old_a is not None and torch.allclose(old_a, new_a)
print(f"  Square equivalence (old==new): {'PASS' if equiv else 'FAIL'}")
spatial_a = check_spatial_correctness(new_a, x_a, B, num_windows, h_per, w_per, "A")
results["A"] = equiv and spatial_a

# ── CASE B: non-square horizontal 640×736 ─────────────────────────────────────
print("\nCASE B — Non-square horizontal 640×736: (2, 40, 46, 384)")
H_p, W_p = 40, 46
h_per, w_per = H_p // num_windows, W_p // num_windows   # 20, 23
x_b = torch.randn(B, H_p, W_p, D)

try:
    old_b = old_windowing(x_b, B, num_windows, h_per, w_per)
    print(f"  OLD: shape {tuple(old_b.shape)}  [unexpected success]")
    results["OLD_B_crashed"] = False
except RuntimeError as e:
    print(f"  OLD: FAIL (as expected) — {type(e).__name__}")
    old_b = None
    results["OLD_B_crashed"] = True

new_b = new_windowing(x_b, B, num_windows, H_p, h_per, W_p, w_per)
print(f"  NEW: shape {tuple(new_b.shape)}  (expected (8, 460, 384))")
spatial_b = check_spatial_correctness(new_b, x_b, B, num_windows, h_per, w_per, "B")
results["B"] = results["OLD_B_crashed"] and spatial_b

# ── CASE C: non-square vertical 736×640 ──────────────────────────────────────
print("\nCASE C — Non-square vertical 736×640: (2, 46, 40, 384)")
H_p, W_p = 46, 40
h_per, w_per = H_p // num_windows, W_p // num_windows   # 23, 20
x_c = torch.randn(B, H_p, W_p, D)

try:
    old_c = old_windowing(x_c, B, num_windows, h_per, w_per)
    print(f"  OLD: shape {tuple(old_c.shape)}  [unexpected success]")
    results["OLD_C_crashed"] = False
except RuntimeError as e:
    print(f"  OLD: FAIL (as expected) — {type(e).__name__}")
    old_c = None
    results["OLD_C_crashed"] = True

new_c = new_windowing(x_c, B, num_windows, H_p, h_per, W_p, w_per)
print(f"  NEW: shape {tuple(new_c.shape)}  (expected (8, 460, 384))")
spatial_c = check_spatial_correctness(new_c, x_c, B, num_windows, h_per, w_per, "C")
results["C"] = results["OLD_C_crashed"] and spatial_c

# ── Task 4: size_divisibility necessity and sufficiency ───────────────────────
print(f"\n{SEP}")
print("TASK 4 — SIZE_DIVISIBILITY=32: necessity and sufficiency")
print(SEP)
patch_size = 16

print("\n  Necessity (H not divisible by 32):")
H_bad = 641
print(f"  H={H_bad}: H % 32 = {H_bad % 32}  → non-integer patches ({H_bad}/{patch_size}={H_bad/patch_size:.3f})")
print(f"  dinov2.py asserts H % (patch_size * num_windows) == 0 → fires before windowing.")
print(f"  SIZE_DIVISIBILITY=32 ensures this is caught at padding time, not at backbone time.")

print("\n  Sufficiency (H=640, W=736, both divisible by 32):")
for H, W in [(640, 736), (512, 512), (800, 864)]:
    h_p = H // patch_size
    w_p = W // patch_size
    h_per = h_p // num_windows
    w_per = w_p // num_windows
    ok = (H % 32 == 0) and (W % 32 == 0) and (h_p % num_windows == 0) and (w_p % num_windows == 0)
    print(f"  H={H}, W={W}: H_patches={h_p}, W_patches={w_p}, "
          f"h_per={h_per}, w_per={w_per}  → {'OK' if ok else 'FAIL'}")

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("SUMMARY")
print(SEP)
print(f"\n  CASE A (square 512×512):             {'PASS' if results['A'] else 'FAIL'}")
print(f"  CASE B (horizontal 640×736):         {'PASS' if results['B'] else 'FAIL'}")
print(f"  CASE C (vertical 736×640):           {'PASS' if results['C'] else 'FAIL'}")
equiv_label = "PASS" if (results.get("A") and old_a is not None and torch.allclose(old_a, new_a)) else "FAIL"
print(f"  Square equivalence (old == new):     {equiv_label}")
overall = all([results["A"], results["B"], results["C"]])
print(f"\n  {'ALL PASS' if overall else 'FAILURES DETECTED'}")
print()

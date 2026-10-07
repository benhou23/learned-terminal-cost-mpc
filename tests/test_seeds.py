"""
Training, hold-out and final evaluation seeds must not overlap.

    python tests/test_seeds.py
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

PASS = FAIL = 0

def check(name, ok, detail=""):
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")

try:
    from params import SIM_PARAMS, EVAL_HOLDOUT_SEEDS, EVAL_FINAL_SEEDS
except ImportError:
    print("ERROR: params.py not found. Run from project root.")
    sys.exit(1)

print(f"\n{'='*55}")
print("  Seed range isolation test")
print(f"{'='*55}")

train_max   = SIM_PARAMS.get("train_seed_max", 9999)
train_set   = set(range(0, train_max + 1))
holdout_set = set(EVAL_HOLDOUT_SEEDS)
final_set   = set(EVAL_FINAL_SEEDS)

# Sizes
check(f"train_seed_max = {train_max} (at least 1000 seeds)",
      train_max >= 1000, f"got {train_max}")
check(f"EVAL_HOLDOUT_SEEDS has 10 seeds",
      len(EVAL_HOLDOUT_SEEDS) == 10, f"got {len(EVAL_HOLDOUT_SEEDS)}")
check(f"EVAL_FINAL_SEEDS has 500 seeds",
      len(EVAL_FINAL_SEEDS) == 500, f"got {len(EVAL_FINAL_SEEDS)}")

# No duplicates within each set
check("EVAL_HOLDOUT_SEEDS has no duplicates",
      len(holdout_set) == len(EVAL_HOLDOUT_SEEDS))
check("EVAL_FINAL_SEEDS has no duplicates",
      len(final_set) == len(EVAL_FINAL_SEEDS))

# Disjoint
overlap_th = train_set & holdout_set
check("training ∩ holdout = empty",
      len(overlap_th) == 0, f"overlap: {sorted(overlap_th)[:5]}")

overlap_tf = train_set & final_set
check("training ∩ final_eval = empty",
      len(overlap_tf) == 0, f"overlap: {sorted(overlap_tf)[:5]}")

overlap_hf = holdout_set & final_set
check("holdout ∩ final_eval = empty",
      len(overlap_hf) == 0, f"overlap: {sorted(overlap_hf)[:5]}")

# Print ranges for reference
print(f"\n  Ranges:")
print(f"    Training:   [0, {train_max}]: {len(train_set):,} seeds")
print(f"    Holdout:    [{min(holdout_set)}, {max(holdout_set)}]: {len(holdout_set)} seeds")
print(f"    Final eval: [{min(final_set)}, {max(final_set)}]: {len(final_set)} seeds")

total = PASS + FAIL
print(f"\n{'='*55}")
print(f"  {PASS}/{total} checks passed")
if FAIL == 0:
    print("  Seed isolation OK")
else:
    print(f"  {FAIL} FAILURE(S)")
print(f"{'='*55}")
sys.exit(0 if FAIL == 0 else 1)

import os
import glob
import argparse
import numpy as np


def find_eval_npzs(exp_dir: str):
    # Support both "seed_0" and "seed 0" just in case
    patterns = [
        os.path.join(exp_dir, "seed_*", "eval_logs", "evaluations.npz"),
        os.path.join(exp_dir, "seed *", "eval_logs", "evaluations.npz"),
    ]
    paths = []
    for pat in patterns:
        paths.extend(glob.glob(pat))
    return sorted(set(paths))


def load_npz(path: str):
    d = np.load(path)
    # SB3 EvalCallback saves:
    # timesteps: (K,)
    # results: (K, n_eval_episodes)
    t = d["timesteps"]
    results = d["results"]
    return t, results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_name", required=True, help="e.g., base, sde, rnd, sde_rnd")
    ap.add_argument("--root", default="runs/ablations", help="path to ablations root (default: runs/ablations)")
    ap.add_argument("--threshold", type=float, default=300.0, help="success threshold on episodic return")
    ap.add_argument("--summary", choices=["best", "final"], default="best",
                    help="use best or final evaluation point for summary stats")
    args = ap.parse_args()

    cwd = os.getcwd()
    root = os.path.abspath(args.root)
    exp_dir = os.path.join(root, args.exp_name)

    print("[INFO] cwd:", cwd)
    print("[INFO] root:", root)
    print("[INFO] exp_dir:", exp_dir)

    if not os.path.isdir(exp_dir):
        print(f"[ERROR] Experiment directory does not exist: {exp_dir}")
        print("        Double-check --exp_name and --root.")
        return

    npz_paths = find_eval_npzs(exp_dir)
    print(f"[INFO] found {len(npz_paths)} evaluations.npz files")
    for p in npz_paths:
        print("  -", p)

    if not npz_paths:
        print("[ERROR] No evaluations.npz found. This means EvalCallback did not write eval logs.")
        print("        Check that your run contains: seed_X/eval_logs/evaluations.npz")
        return

    per_seed = []
    for p in npz_paths:
        t, results = load_npz(p)
        eval_mean = results.mean(axis=1)  # (K,)
        success = (results >= args.threshold).mean(axis=1)  # (K,)

        if args.summary == "final":
            mean_val = float(eval_mean[-1])
            succ_val = float(success[-1])
            step = int(t[-1])
        else:
            idx = int(np.argmax(eval_mean))
            mean_val = float(eval_mean[idx])
            succ_val = float(success[idx])
            step = int(t[idx])

        seed_dir = os.path.basename(os.path.dirname(os.path.dirname(p)))  # seed_0 or seed 0
        per_seed.append((seed_dir, mean_val, succ_val, step))

    print("\nPer-seed summary:")
    for seed_dir, mean_val, succ_val, step in per_seed:
        print(f"  {seed_dir:>6}  {args.summary:>4} @ {step:>8}  "
              f"eval_mean={mean_val:>7.1f}  success={100*succ_val:>5.1f}%")

    means = np.array([x[1] for x in per_seed], dtype=float)
    succs = np.array([x[2] for x in per_seed], dtype=float)

    print("\nAcross seeds:")
    print(f"  eval_mean: {means.mean():.1f} ± {means.std():.1f}")
    print(f"  success : {100*succs.mean():.1f}% ± {100*succs.std():.1f}%")
    print("\n[DONE]")


if __name__ == "__main__":
    main()

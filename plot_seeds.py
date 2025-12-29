import os
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt


def find_seed_eval_files(exp_dir: str):
    """
    Supports both seed folder formats:
      - seed_0
      - seed 0
    """
    patterns = [
        os.path.join(exp_dir, "seed_*", "eval_logs", "evaluations.npz"),
        os.path.join(exp_dir, "seed *", "eval_logs", "evaluations.npz"),
    ]
    paths = []
    for pat in patterns:
        paths.extend(glob.glob(pat))
    # Deduplicate + stable order
    return sorted(set(paths))


def load_eval_curve(npz_path: str):
    """
    SB3 EvalCallback file contains:
      timesteps: (K,)
      results: (K, n_eval_episodes)
    """
    d = np.load(npz_path)
    t = d["timesteps"].astype(np.int64)
    results = d["results"].astype(np.float64)
    eval_mean = results.mean(axis=1)  # (K,)
    return t, eval_mean, results


def align_curves(curves):
    """
    curves: list of (t, y) pairs for different seeds.
    Align by intersection of timesteps; if not identical, interpolate onto common grid.
    """
    # Start with the first curve's timesteps
    t_common = curves[0][0]
    for (t, _) in curves[1:]:
        t_common = np.intersect1d(t_common, t)

    if len(t_common) == 0:
        raise RuntimeError("No overlapping eval timesteps across seeds. Check eval_freq consistency.")

    ys = []
    for (t, y) in curves:
        # Interpolate y onto t_common (safe even if identical)
        y_common = np.interp(t_common, t, y)
        ys.append(y_common)

    ys = np.stack(ys, axis=0)  # (n_seeds, K)
    return t_common, ys


def summarize_seed(results: np.ndarray, success_threshold: float):
    """
    results: (K, n_eval_episodes)
    returns:
      best_mean, final_mean, best_success, final_success
    """
    eval_mean = results.mean(axis=1)  # (K,)
    success = (results >= success_threshold).mean(axis=1)  # (K,)

    return (
        float(eval_mean.max()),
        float(eval_mean[-1]),
        float(success.max()),
        float(success[-1]),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs/ablations", help="Root directory containing experiment folders")
    ap.add_argument("--exps", default=None,
                    help="Comma-separated list of exp names to plot (e.g., base,sde,rnd,sde_rnd). "
                         "If omitted, auto-detect all experiments under root.")
    ap.add_argument("--success_threshold", type=float, default=300.0,
                    help="Return threshold for 'success rate' (default: 300)")
    ap.add_argument("--out_dir", default="runs/ablations/_plots", help="Where to save figures")
    ap.add_argument("--use", choices=["best", "final"], default="best",
                    help="Summary metric uses BEST or FINAL checkpoint (default: best)")
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Determine experiments to plot
    if args.exps is None:
        # Auto-detect folders in root (skip _plots)
        exp_names = [
            name for name in sorted(os.listdir(root))
            if os.path.isdir(os.path.join(root, name)) and not name.startswith("_")
        ]
    else:
        exp_names = [x.strip() for x in args.exps.split(",") if x.strip()]

    if not exp_names:
        raise RuntimeError(f"No experiments found. root={root}")

    # Collect per-experiment aligned curves and summary stats
    exp_to_curve = {}  # exp -> (t_common, mean, std)
    summary_rows = []  # list of dicts

    for exp in exp_names:
        exp_dir = os.path.join(root, exp)
        if not os.path.isdir(exp_dir):
            print(f"[WARN] Skipping missing exp dir: {exp_dir}")
            continue

        npz_paths = find_seed_eval_files(exp_dir)
        if not npz_paths:
            print(f"[WARN] Skipping {exp}: no evaluations.npz found under {exp_dir}")
            continue

        # Load each seed
        curves = []
        seed_summaries = []
        for p in npz_paths:
            t, eval_mean, results = load_eval_curve(p)
            curves.append((t, eval_mean))

            best_mean, final_mean, best_succ, final_succ = summarize_seed(results, args.success_threshold)
            seed_dir = os.path.basename(os.path.dirname(os.path.dirname(p)))  # seed_0 or seed 0

            seed_summaries.append({
                "seed_dir": seed_dir,
                "best_mean": best_mean,
                "final_mean": final_mean,
                "best_succ": best_succ,
                "final_succ": final_succ,
            })

        # Align seed curves and aggregate
        t_common, ys = align_curves(curves)
        mean = ys.mean(axis=0)
        std = ys.std(axis=0)
        exp_to_curve[exp] = (t_common, mean, std)

        # Aggregate summary (best or final)
        if args.use == "final":
            means = np.array([s["final_mean"] for s in seed_summaries], dtype=float)
            succs = np.array([s["final_succ"] for s in seed_summaries], dtype=float)
        else:
            means = np.array([s["best_mean"] for s in seed_summaries], dtype=float)
            succs = np.array([s["best_succ"] for s in seed_summaries], dtype=float)

        summary_rows.append({
            "exp": exp,
            "n_seeds": len(seed_summaries),
            "mean_reward_mean": float(means.mean()),
            "mean_reward_std": float(means.std()),
            "success_mean": float(succs.mean()),
            "success_std": float(succs.std()),
        })

        print(f"[OK] {exp}: seeds={len(seed_summaries)}  ({args.use}) "
              f"reward={means.mean():.1f}±{means.std():.1f}  "
              f"success={100*succs.mean():.1f}%±{100*succs.std():.1f}%")

    if not exp_to_curve:
        raise RuntimeError("No experiments had usable eval logs. Check your eval_logs/evaluations.npz files.")

    # -----------------------
    # Plot 1: Learning curves
    # -----------------------
    plt.figure()
    for exp, (t, mean, std) in exp_to_curve.items():
        plt.plot(t, mean, label=f"{exp} (mean)")
        plt.fill_between(t, mean - std, mean + std, alpha=0.2)
    plt.xlabel("Timesteps")
    plt.ylabel("Eval mean reward")
    plt.title(f"Learning curves (mean ± std across seeds) [{args.use}]")
    plt.legend()
    plt.tight_layout()
    out1 = os.path.join(out_dir, "learning_curves.png")
    plt.savefig(out1, dpi=200)
    print("Saved:", out1)

    # -----------------------------------------
    # Plot 2a: Bar chart of reward summary
    # -----------------------------------------
    summary_rows = sorted(summary_rows, key=lambda r: r["exp"])
    labels = [r["exp"] for r in summary_rows]
    reward_means = [r["mean_reward_mean"] for r in summary_rows]
    reward_stds = [r["mean_reward_std"] for r in summary_rows]

    plt.figure()
    x = np.arange(len(labels))
    plt.bar(x, reward_means, yerr=reward_stds, capsize=4)
    plt.xticks(x, labels, rotation=30, ha="right")
    plt.ylabel(f"{args.use.capitalize()} eval mean reward")
    plt.title(f"{args.use.capitalize()} eval reward (mean ± std across seeds)")
    plt.tight_layout()
    out2 = os.path.join(out_dir, "bar_best_mean.png" if args.use == "best" else "bar_final_mean.png")
    plt.savefig(out2, dpi=200)
    print("Saved:", out2)

    # -----------------------------------------
    # Plot 2b: Bar chart of success rate summary
    # -----------------------------------------
    succ_means = [100.0 * r["success_mean"] for r in summary_rows]
    succ_stds = [100.0 * r["success_std"] for r in summary_rows]

    plt.figure()
    plt.bar(x, succ_means, yerr=succ_stds, capsize=4)
    plt.xticks(x, labels, rotation=30, ha="right")
    plt.ylabel(f"Success rate (%)  [return ≥ {args.success_threshold:g}]")
    plt.title(f"{args.use.capitalize()} success rate (mean ± std across seeds)")
    plt.tight_layout()
    out3 = os.path.join(out_dir, "bar_success_rate.png")
    plt.savefig(out3, dpi=200)
    print("Saved:", out3)

    print("\n[DONE] Plots saved to:", out_dir)


if __name__ == "__main__":
    main()

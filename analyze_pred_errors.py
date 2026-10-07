"""
Plot the prediction errors logged by pred_error_logger.py.

Makes a 6-panel summary (mean/box plots of position and speed error per
neighbour, lane errors, how often each neighbour exists) and one histogram
of the position error per neighbour.


    python analyze_pred_errors.py --csv pred_errors.csv --out_dir . --label H5
"""
import argparse, os, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

ROLES  = ["front", "front_left", "front_right", "back_left", "back_right"]
COLORS = {
    "front":       "#E8593C",
    "front_left":  "#3A7FD5",
    "front_right": "#1D9E75",
    "back_left":   "#F5A623",
    "back_right":  "#9B59B6",
}


def load(csv_path):
    df = pd.read_csv(csv_path)
    present = df[df["vehicle_present"] == 1].copy()
    print(f"  Total rows:      {len(df)}")
    print(f"  Vehicle present: {len(present)}  ({100*len(present)/max(len(df),1):.1f}%)")
    print(f"  Scenarios:       {df['scenario'].nunique()}")
    print(f"  H:               {df['H'].iloc[0] if len(df) else 'n/a'}")
    return df, present


def summary_table(present):
    print(f"\n{'='*80}")
    print(f"  {'Role':<14} {'N':>6} {'Mean err_s':>12} {'Std err_s':>12} "
          f"{'Mean err_v':>12} {'Std err_v':>12} {'LC rate':>9}")
    print(f"{'='*80}")
    rows = []
    for role in ROLES:
        sub = present[present["role"] == role]
        if sub.empty: continue
        n    = len(sub)
        ms   = sub["error_s"].mean();   ss = sub["error_s"].std()
        mv   = sub["error_v"].mean();   sv = sub["error_v"].std()
        lc   = (sub["error_lane"].abs() > 0).mean()
        print(f"  {role:<14} {n:>6} {ms:>12.3f} {ss:>12.3f} "
              f"{mv:>12.3f} {sv:>12.3f} {lc:>9.1%}")
        rows.append(dict(role=role, n=n,
                         mean_err_s=round(ms,3), std_err_s=round(ss,3),
                         mean_err_v=round(mv,3), std_err_v=round(sv,3),
                         lane_change_rate=round(lc,4)))
    ms=present["error_s"].mean(); ss=present["error_s"].std()
    mv=present["error_v"].mean(); lc=(present["error_lane"].abs()>0).mean()
    print(f"{'='*80}")
    print(f"  {'ALL':<14} {len(present):>6} {ms:>12.3f} {ss:>12.3f} "
          f"{mv:>12.3f} {present['error_v'].std():>12.3f} {lc:>9.1%}")
    print(f"{'='*80}\n")
    return pd.DataFrame(rows)


def make_plots(df, present, out_path, label=""):
    roles_present = [r for r in ROLES if r in present["role"].values]

    # summary figure
    fig = plt.figure(figsize=(24, 14))
    fig.patch.set_facecolor("#F8F8F8")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.34)

    def style(ax, title, xlabel=None, ylabel=None):
        ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
        if xlabel: ax.set_xlabel(xlabel, fontsize=10)
        if ylabel: ax.set_ylabel(ylabel, fontsize=10)
        ax.spines[["top","right"]].set_visible(False)
        ax.tick_params(labelsize=9)

    colors = [COLORS[r] for r in roles_present]

    # mean position error per role
    ax = fig.add_subplot(gs[0,0])
    means_s = [present[present["role"]==r]["error_s"].mean() for r in roles_present]
    stds_s  = [present[present["role"]==r]["error_s"].std()  for r in roles_present]
    bars = ax.bar(range(len(roles_present)), means_s, color=colors,
                  edgecolor="white", yerr=stds_s, capsize=4)
    ax.axhline(0, color="red", linestyle="--", linewidth=1, alpha=0.6)
    ax.set_xticks(range(len(roles_present)))
    ax.set_xticklabels(roles_present, rotation=20, ha="right", fontsize=9)
    for bar,v in zip(bars,means_s):
        ax.text(bar.get_x()+bar.get_width()/2,
                bar.get_height()+(max(stds_s)*0.05 if v>=0 else -max(stds_s)*0.15),
                f"{v:.2f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    style(ax, "Mean position error by role (signed)", ylabel="error_s (m)")

    # mean speed error per role
    ax = fig.add_subplot(gs[0,1])
    means_v = [present[present["role"]==r]["error_v"].mean() for r in roles_present]
    stds_v  = [present[present["role"]==r]["error_v"].std()  for r in roles_present]
    bars = ax.bar(range(len(roles_present)), means_v, color=colors,
                  edgecolor="white", yerr=stds_v, capsize=4)
    ax.axhline(0, color="red", linestyle="--", linewidth=1, alpha=0.6)
    ax.set_xticks(range(len(roles_present)))
    ax.set_xticklabels(roles_present, rotation=20, ha="right", fontsize=9)
    for bar,v in zip(bars,means_v):
        ax.text(bar.get_x()+bar.get_width()/2,
                bar.get_height()+(max(stds_v)*0.05 if v>=0 else -max(stds_v)*0.15),
                f"{v:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    style(ax, "Mean velocity error by role (signed)", ylabel="error_v (m/s)")

    # lane errors
    ax = fig.add_subplot(gs[0,2])
    all_le = present["error_lane"].dropna()
    vals, counts = np.unique(all_le.astype(int), return_counts=True)
    ax.bar(vals, counts/counts.sum()*100, color="#3A7FD5", edgecolor="white", width=0.6)
    ax.set_xticks(vals)
    style(ax, "Lane prediction error distribution",
          xlabel="Predicted − Actual lane", ylabel="% of predictions")

    # position error spread
    ax = fig.add_subplot(gs[1,0])
    data = [present[present["role"]==r]["error_s"].clip(-30,30).values for r in roles_present]
    bp = ax.boxplot(data, patch_artist=True,
                    medianprops=dict(color="black", linewidth=2))
    for patch,color in zip(bp["boxes"],colors):
        patch.set_facecolor(color); patch.set_alpha(0.7)
    ax.axhline(0, color="red", linestyle="--", linewidth=1, alpha=0.6)
    ax.set_xticks(range(1,len(roles_present)+1))
    ax.set_xticklabels(roles_present, rotation=20, ha="right", fontsize=9)
    style(ax, "Signed position error (pred − actual)", ylabel="error_s (m)")

    # speed error spread
    ax = fig.add_subplot(gs[1,1])
    data_v = [present[present["role"]==r]["error_v"].clip(-5,5).values for r in roles_present]
    bp2 = ax.boxplot(data_v, patch_artist=True,
                     medianprops=dict(color="black", linewidth=2))
    for patch,color in zip(bp2["boxes"],colors):
        patch.set_facecolor(color); patch.set_alpha(0.7)
    ax.axhline(0, color="red", linestyle="--", linewidth=1, alpha=0.6)
    ax.set_xticks(range(1,len(roles_present)+1))
    ax.set_xticklabels(roles_present, rotation=20, ha="right", fontsize=9)
    style(ax, "Signed velocity error (pred − actual)", ylabel="error_v (m/s)")

    # how often each role is occupied
    ax = fig.add_subplot(gs[1,2])
    presence = [df[df["role"]==r]["vehicle_present"].mean()*100 for r in roles_present]
    bars = ax.bar(range(len(roles_present)), presence, color=colors, edgecolor="white")
    for bar,v in zip(bars,presence):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                f"{v:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks(range(len(roles_present)))
    ax.set_xticklabels(roles_present, rotation=20, ha="right", fontsize=9)
    ax.set_ylim(0, 115)
    style(ax, "Vehicle present at t_actual (%)", ylabel="%")

    H_steps = int(df["H"].iloc[0]) if len(df) else "?"
    H_secs  = H_steps*0.2 if isinstance(H_steps,int) else "?"
    fig.suptitle(f"Prediction Error Analysis {label}  "
                 f"(H={H_steps} steps × 0.2s = {H_secs:.1f}s)",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Summary plot -> {out_path}")

    # one histogram per role
    fig2, axes = plt.subplots(1, len(roles_present), figsize=(5*len(roles_present), 5))
    fig2.patch.set_facecolor("#F8F8F8")
    if len(roles_present) == 1: axes = [axes]

    for ax, role in zip(axes, roles_present):
        sub = present[present["role"]==role]["error_s"].clip(-20, 20)
        n   = len(sub)
        mu  = sub.mean(); sigma = sub.std()

        # Freedman-Diaconis bin width
        q75, q25 = np.percentile(sub, [75,25])
        iqr = q75 - q25
        bin_width = 2*iqr/(n**(1/3)) if iqr > 0 else 0.1
        bins = max(30, min(100, int((sub.max()-sub.min())/bin_width)))

        ax.hist(sub, bins=bins, color=COLORS[role], alpha=0.75,
                edgecolor="white", linewidth=0.3, density=False)
        ax.axvline(0,  color="red",   linestyle="--", linewidth=1.5, label="zero")
        ax.axvline(mu, color="black", linestyle="-",  linewidth=1.5,
                   label=f"μ={mu:.2f}")
        ax.set_title(f"{role}\nμ={mu:.2f}m  σ={sigma:.2f}m  n={n:,}",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel("error_s = pred − actual (m)", fontsize=10)
        ax.set_ylabel("Count", fontsize=10)
        ax.spines[["top","right"]].set_visible(False)
        ax.legend(fontsize=9)

    fig2.suptitle(f"Position Error Histograms per Role {label}  "
                  f"(H={H_steps}×0.2s={H_secs:.1f}s ahead)",
                  fontsize=13, fontweight="bold")
    plt.tight_layout()
    hist_path = out_path.replace(".png", "_histograms.png")
    plt.savefig(hist_path, dpi=150, bbox_inches="tight", facecolor=fig2.get_facecolor())
    plt.close()
    print(f"  Histogram plot -> {hist_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",     default="pred_errors.csv")
    parser.add_argument("--out_dir", default=".")
    parser.add_argument("--label",   default="")
    args = parser.parse_args()

    print(f"\nLoading {args.csv} ...")
    df, present = load(args.csv)

    if present.empty:
        print("No rows with vehicle_present=1 found."); return

    print("\nSummary by role:")
    summary_df = summary_table(present)
    suffix = f"_{args.label}" if args.label else ""
    summary_df.to_csv(os.path.join(args.out_dir, f"pred_error_summary{suffix}.csv"), index=False)
    print(f"  Summary CSV saved.")

    print("Generating plots ...")
    make_plots(df, present,
               os.path.join(args.out_dir, f"pred_error_plots{suffix}.png"),
               label=args.label)

    H_steps = int(df["H"].iloc[0])
    ms = present["error_s"].mean(); ss = present["error_s"].std()
    mv = present["error_v"].mean()
    lc = (present["error_lane"].abs()>0).mean()*100
    print(f"\nKey numbers:")
    print(f"  H = {H_steps} steps = {H_steps*0.2:.1f}s horizon")
    print(f"  Mean pos error:  {ms:.3f} m ± {ss:.3f}  (neg = pred ahead of actual)")
    print(f"  Mean vel error:  {mv:.3f} m/s  (neg = pred faster than actual)")
    print(f"  Wrong lane:      {lc:.1f}%")


if __name__ == "__main__":
    main()

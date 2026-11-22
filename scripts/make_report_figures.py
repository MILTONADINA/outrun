"""Render the REPORT.md figures from reports/metrics/*.json (no numbers are typed in)."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

M = Path("reports/metrics")
OUT = Path("reports/figures")


def J(n):
    return json.load((M / f"{n}.json").open())


def est(h, split, model, metric):
    m = J(h)["splits"][split][model]["metrics"][metric]
    return m["estimate"], m["low"], m["high"]


def fig_models():
    rows = [  # label, hazard, split, DL, best baseline, metric
        ("Storm CSI", "storm", "test", "storm.unet_vil_ir", "pysteps_extrapolation", "csi_mean_74_133"),
        ("Tornado AUC", "tornado", "test", "tornado.cnn", "tornado.cnn_raw", "auc_tor_vs_warned_null"),
        ("Flood AUPRC", "flood", "test", "flood.unet_raw", "logistic_regression", "auprc"),
        ("Wildfire NDWS PR-AUC", "wildfire", "test", "wildfire.unet", "pixel_logistic", "prauc"),
        ("Wildfire WSTS PR-AUC", "wildfire", "test_wsts2021", "wildfire.unet", "pixel_logistic", "prauc"),
        ("Ice zone AUPRC", "ice", "test", "ice.milcnn", "logistic", "zone_auprc"),
    ]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for i, (_lab, h, s, dl, bl, met) in enumerate(rows):
        for j, (name, c) in enumerate([(dl, "#1f77b4"), (bl, "#999999")]):
            e, lo, hi = est(h, s, name, met)
            ax.barh(i + (0.2 if j else -0.2), e, 0.38, color=c, xerr=[[e - lo], [hi - e]], capsize=2)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel("held-out test metric (95% bootstrap CI)")
    ax.legend(
        handles=[plt.Rectangle((0, 0), 1, 1, color="#1f77b4"), plt.Rectangle((0, 0), 1, 1, color="#999999")],
        labels=["deep model", "best baseline"],
        loc="lower right",
    )
    ax.set_title("Deep model vs best baseline (higher is better)")
    fig.tight_layout()
    fig.savefig(OUT / "models_vs_baselines.png", dpi=150)
    plt.close(fig)


def fig_routing():
    r = J("routing_may20-2019")["planners"]
    names = ["a", "a_prime", "b", "c", "d"]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.bar(names, [r[n]["mean_exposure_risk_min"] for n in names], color="#4c72b0")
    ax.set_ylim(5.0, 6.1)
    ax.set_ylabel("mean observed exposure (risk-min)")
    ax.set_title("may20-2019, 500 origins (axis starts at 5.0)")
    fig.tight_layout()
    fig.savefig(OUT / "routing_exposure.png", dpi=150)
    plt.close(fig)


def fig_gap():
    sc = ["elreno-2019", "flood-okc-2025", "ice-okc-2025", "may20-2019"]
    fig, ax = plt.subplots(figsize=(6, 3.4))
    for i, s in enumerate(sc):
        g = J(f"routing_{s}")["gap_closed"]
        e, lo, hi = g["estimate"], g["low"], g["high"]
        ax.errorbar(
            100 * e, i, xerr=[[100 * (e - lo)], [100 * (hi - e)]], fmt="o", capsize=3, color="#c44e52"
        )
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(range(len(sc)), sc)
    ax.set_xlabel("gap closed between (a) and oracle (d), % (95% CI)")
    ax.set_title("Routing gap closed (smoke: NaN, no risk)")
    fig.tight_layout()
    fig.savefig(OUT / "gap_closed.png", dpi=150)
    plt.close(fig)


def fig_evac():
    m = J("group_may20-2019")["evacuation"]["methods"]
    names = ["shortest", "safest", "msa", "sa", "sa_lambda0"]

    def mean(n, k):
        return sum(x[k] for x in m[n]) / len(m[n])

    fig, axs = plt.subplots(1, 2, figsize=(8, 3.4))
    for ax, k, t in [
        (axs[0], "vehicle_hours", "vehicle-hours"),
        (axs[1], "exposure_veh_h", "exposure (veh-h)"),
    ]:
        ax.bar(names, [mean(n, k) for n in names], color="#55a868")
        ax.set_title(t)
        ax.tick_params(axis="x", rotation=30)
    fig.suptitle("may20 metro evacuation, mean of 5 seeds")
    fig.tight_layout()
    fig.savefig(OUT / "evacuation.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    for f in (fig_models, fig_routing, fig_gap, fig_evac):
        f()

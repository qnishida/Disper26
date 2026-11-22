#!/usr/bin/env python3
"""Regenerate three Disper26 result figures from the bundled data.

No Disper26 extension or network connection is required. Units in stored
curves are km/s; errors plotted here are 100*abs(prediction-exact)/exact.
Edit COLORS, rcParams, and the figure functions for presentation changes.
"""
import argparse
import hashlib
import json
import os
import platform
import tempfile
from pathlib import Path

# Writable, disposable caches also work in restricted build environments.
_cache = tempfile.TemporaryDirectory(prefix="disper26-paper-plot-")
os.environ.setdefault("MPLCONFIGDIR", _cache.name + "/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", _cache.name + "/fontconfig")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "results"
FIG = ROOT / "Fig"
PLAN_L_PREFIX = "plan_l_v2"
COLORS = {1: "#D55E00", 2: "#0072B2", "exact": "#595959"}
CURVES = [("rayleigh", 0), ("rayleigh", 1), ("love", 0), ("love", 1)]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                     "axes.titlesize": 10, "axes.labelsize": 9,
                     "legend.fontsize": 8, "xtick.labelsize": 8,
                     "ytick.labelsize": 8, "pdf.fonttype": 42,
                     "ps.fonttype": 42, "axes.spines.top": False,
                     "axes.spines.right": False})


def save(fig, name):
    name = name.replace("plan_l_v2", PLAN_L_PREFIX)
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / (name + ".pdf"), bbox_inches="tight",
                metadata={"CreationDate": None, "ModDate": None,
                          "Title": name, "Creator": "scripts/plot_results.py"})
    fig.savefig(FIG / (name + ".png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def errors(a, wave, mode, order):
    key = wave + str(mode)
    exact = a[key + "_exact"]
    return 100 * np.abs(a[key + "_order%d" % order] - exact) / exact


def title(ax, i, wave, mode):
    ax.set_title("(%s) %s, %s" % ("abcd"[i], wave.capitalize(),
                 "fundamental" if mode == 0 else "first overtone"), loc="left")
    ax.grid(which="major", alpha=.22, linewidth=.6)


def amplitude_plot(a):
    # Additional points reuse opened holdout directions; original data stay intact.
    with np.load(DATA / "plan_l_v2_small_amplitude.npz") as extra:
        merged = dict(a)
        for key in a.files:
            if key != "frequency_hz":
                wave = key.split("_")[0].rstrip("01")
                mask = extra[wave + "_amplitude"] < .005
                merged[key] = np.concatenate([extra[key][mask], a[key]])
        a = merged
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4), sharex=True, sharey=True)
    for i, (ax, (wave, mode)) in enumerate(zip(axes.flat, CURVES)):
        amp = a[wave + "_amplitude"]
        levels = np.unique(amp)
        for order in (1, 2):
            e = errors(a, wave, mode, order)
            q = np.array([np.quantile(e[np.isclose(amp, v, rtol=0, atol=1e-14)],
                                      [.5, .9, .99]) for v in levels])
            assert np.all(q > 0)
            ax.plot(100 * levels, q[:, 1], "o-" if order == 2 else "s--",
                    color=COLORS[order], ms=3.4, lw=1.5,
                    label="Order %d: p90" % order)
            if order == 2:
                ax.fill_between(100 * levels, q[:, 0], q[:, 2],
                                color=COLORS[2], alpha=.14, label="Order 2: p50–p99")
        # Slope guides are offset from the p90 curves, anchored at 0.2%.
        for order, power in ((1, 2), (2, 3)):
            e = errors(a, wave, mode, order)
            anchor = np.quantile(e[np.isclose(amp, .002)], .9) * 4
            xx = np.array([.12, .8])
            yy = anchor * (xx / .2)**power
            ax.plot(xx, yy, ":", color=".4", lw=1)
            ax.text(xx[-1]*1.05, yy[-1], r"$h^%d$" % power,
                    color=".3", fontsize=8, va="center")
        ax.axvspan(8, 11, color=".92", zorder=-2)
        ax.axvline(8, color=".5", ls=":", lw=1)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(.09, 11)
        ax.set_xticks([.1, .2, .5, 1, 2, 4, 8, 10])
        ax.set_xticklabels(["0.1", "0.2", "0.5", "1", "2", "4", "8", "10"])
        title(ax, i, wave, mode)
        if i in (0, 2):
            ax.set_ylabel("Absolute phase-velocity error (%)")
        if i >= 2:
            ax.set_xlabel("Maximum shear-velocity anomaly (%)")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(.5, 1.01))
    fig.text(.5, .005, "64 models per amplitude; 10 frequencies. 0.1–0.2%: added diagnostic; >8%: stress test.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=[0, .035, 1, .94])
    save(fig, "plan_l_v2_error_amplitude")


def frequency_plot(a):
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4), sharex=True, sharey=True)
    for i, (ax, (wave, mode)) in enumerate(zip(axes.flat, CURVES)):
        at8 = np.isclose(a[wave + "_amplitude"], .08, rtol=0, atol=1e-14)
        for order in (1, 2):
            e = errors(a, wave, mode, order)[at8]
            ax.plot(a["frequency_hz"], np.quantile(e, .99, axis=0),
                    "o-" if order == 2 else "s--", color=COLORS[order],
                    ms=3.4, lw=1.5, label="Order %d: p99" % order)
            if order == 2:
                ax.plot(a["frequency_hz"], e.max(axis=0), ":", color=COLORS[2],
                        lw=1.2, label="Order 2: observed maximum")
        ax.axhline(.1, color=".4", lw=1, ls="-.", label="0.1% error scale")
        ax.set_yscale("log")
        title(ax, i, wave, mode)
        if i in (0, 2):
            ax.set_ylabel("Absolute phase-velocity error (%)")
        if i >= 2:
            ax.set_xlabel("Frequency (Hz)")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(.5, 1.02))
    fig.tight_layout(rect=[0, 0, 1, .90])
    save(fig, "plan_l_v2_error_frequency")


def runtime_plot():
    report = json.loads((DATA / "plan_l_v2_benchmark_report.json").read_text())
    rows = report["curve_results"]
    assert [(r["wave"], r["root_order"]) for r in rows] == CURVES
    assert report["surrogate_exact_solver_call_count"] == 0
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5))
    x, width = np.arange(4), .34
    for offset, field, label, color in [(-.5, "exact_scalar_seconds", "Exact", COLORS["exact"]),
                                       (.5, "surrogate_scalar_seconds", "Surrogate", COLORS[2])]:
        values = 1000 * np.array([r[field] for r in rows])
        axes[0].bar(x + offset * width, values, width, label=label, color=color, hatch='///' if offset<0 else '...', edgecolor='black', linewidth=.5)
    for offset, field, label, color in [(-.5, "scalar_speedup", "Scalar", COLORS[2]),
                                       (.5, "batch_speedup", "Batch of 8", COLORS[1])]:
        axes[1].bar(x + offset * width, [r[field] for r in rows], width,
                    label=label, color=color, hatch='///' if offset<0 else '...', edgecolor='black', linewidth=.5)
    for i, ax in enumerate(axes):
        ax.set_xticks(x)
        ax.set_xticklabels(["Rayleigh\n0", "Rayleigh\n1", "Love\n0", "Love\n1"])
        ax.set_yscale("log")
        ax.legend(frameon=False, loc="upper right")
        ax.grid(axis="y", alpha=.22, zorder=-1)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Time per 10-frequency curve (ms)")
    axes[0].set_ylim(.01, 4000)
    axes[0].set_title("(a) Scalar evaluation", loc="left")
    axes[1].set_ylabel("Exact / surrogate time")
    axes[1].set_ylim(500, 40000)
    if PLAN_L_PREFIX != "plan_l_v2":
        scalar_ms = [1000 * row[field] for row in rows
                     for field in ("exact_scalar_seconds", "surrogate_scalar_seconds")]
        speedups = [row[field] for row in rows
                    for field in ("scalar_speedup", "batch_speedup")]
        axes[0].set_ylim(min(scalar_ms) / 2, max(scalar_ms) * 3)
        axes[1].set_ylim(min(speedups) / 2, max(speedups) * 3)
    axes[1].set_title("(b) Evaluation speedup", loc="left")
    fig.tight_layout()
    save(fig, "plan_l_v2_runtime")


def main():
    global DATA, PLAN_L_PREFIX
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-l-data", type=Path, help="Directory produced by recompute_plan_l_eft.py all")
    args = parser.parse_args()
    if args.plan_l_data:
        DATA = args.plan_l_data
        PLAN_L_PREFIX = "plan_l_eft_p0_p3"
        manifest = json.loads((DATA / "manifest.json").read_text())
        if (manifest["p_by_wave"] != {"rayleigh": 0, "love": 3}
                or manifest["source"]["reference_id"] != "plan_l_v2"):
            raise ValueError("Expected the manuscript wave-specific EFT experiment")
    else:
        manifest = json.loads((DATA / "provenance.json").read_text())
    exports = manifest["exports"]
    required = {"plan_l_v2_holdout_curves.npz", "plan_l_v2_small_amplitude.npz", "plan_l_v2_benchmark_report.json"}
    if not required.issubset(exports):
        raise ValueError("Missing plotting inputs in the data manifest")
    for filename, expected in exports.items():
        if hashlib.sha256((DATA / filename).read_bytes()).hexdigest() != expected:
            raise ValueError("Manifest checksum mismatch: " + filename)
    with np.load(DATA / "plan_l_v2_holdout_curves.npz", allow_pickle=False) as a:
        amplitude_plot(a)
        frequency_plot(a)
    runtime_plot()
    print("Generated three PDF/PNG figure pairs.")
    print("Python", platform.python_version(), "NumPy", np.__version__,
          "Matplotlib", matplotlib.__version__)


if __name__ == "__main__":
    main()

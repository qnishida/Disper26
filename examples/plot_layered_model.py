"""Plot dispersion and Rayleigh eigenfunctions for the basic dispersion example."""
import argparse
from pathlib import Path

from matplotlib.figure import Figure

import disper26 as d26
from examples import basic_dispersion


def plot_dispersion(curves):
    fig = Figure(figsize=(6.2, 3.2), constrained_layout=True)
    ax = fig.subplots()
    for wave, color, marker in (("Rayleigh", "#236789", "o"), ("Love", "#ad501e", "s")):
        curve = curves[wave]
        ax.plot(curve.periods, curve.phase_velocity, marker=marker, color=color, label=wave)
    ax.set(xlabel="Period (s)", ylabel="Phase velocity (km/s)", xticks=curves["Rayleigh"].periods)
    ax.grid(axis="y", color="0.9")
    ax.legend(frameon=False)
    return fig


def plot_eigenfunctions(solution):
    profile = d26.compute_eigenfunctions(solution, num_sub_layers=40)
    ur = profile.components["ur"]
    uz = profile.components["uz"]
    fig = Figure(figsize=(5.0, 3.8), constrained_layout=True)
    ax = fig.subplots()
    ax.axvline(0, color="0.75", linewidth=1)
    ax.plot((ur / uz[0]).real, profile.depth, color="#236789", label=r"Horizontal $u_r$")
    ax.plot((uz / uz[0]).real, profile.depth, "--", color="#ad501e", label=r"Vertical $u_z$")
    ax.set(xlabel=r"Reduced displacement / $u_z(0)$", ylabel="Depth (km)",
           ylim=(profile.depth[-1], 0))
    ax.grid(axis="y", color="0.9")
    ax.legend(frameon=False)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path, help="Directory for the two PNG figures")
    output_dir = parser.parse_args().output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    curves = basic_dispersion.run()
    plot_dispersion(curves).savefig(output_dir / "dispersion.png", dpi=150)
    solution = next(s for s in curves["Rayleigh"].solutions if s.period == 5.0)
    plot_eigenfunctions(solution).savefig(output_dir / "rayleigh_eigenfunctions.png", dpi=150)


if __name__ == "__main__":
    main()

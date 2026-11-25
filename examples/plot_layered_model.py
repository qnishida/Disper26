"""Plot dispersion and Rayleigh eigenfunctions for the basic dispersion example."""
import argparse
from pathlib import Path

from matplotlib.figure import Figure

import disper26 as d26
from examples import basic_dispersion


def plot_dispersion(isotropic, vti):
    fig = Figure(figsize=(5.0, 2.8), constrained_layout=True)
    ax = fig.subplots()
    for curve, style, color, label in (
        (isotropic["Rayleigh"], "o-", "#236789", "Rayleigh (both models)"),
        (isotropic["Love"], "s-", "#ad501e", "Love (isotropic)"),
        (vti["Love"], "s--", "#ad501e", "Love (VTI)"),
    ):
        ax.plot(curve.periods, curve.phase_velocity, style, color=color, label=label,
                linewidth=1.2, markersize=2.5, markevery=2)
    ax.set(xlabel="Period (s)", ylabel="Phase velocity (km/s)", xticks=[2, 5, 10, 15, 20])
    ax.grid(axis="y", color="0.9")
    ax.legend(frameon=False, fontsize=9)
    return fig


def plot_eigenfunctions(solution):
    profile = d26.compute_eigenfunctions(solution, num_sub_layers=40)
    ur = profile.components["ur"]
    uz = profile.components["uz"]
    fig = Figure(figsize=(5.0, 3.8), constrained_layout=True)
    ax = fig.subplots()
    ax.axvline(0, color="0.75", linewidth=1)
    ax.axhline(solution.model.thicknesses[0], color="0.45", linewidth=0.8, linestyle=":")
    ax.plot((ur / uz[0]).real, profile.depth, color="#236789", label=r"Horizontal $u_r$")
    ax.plot((uz / uz[0]).real, profile.depth, "--", color="#ad501e", label=r"Vertical $u_z$")
    ax.set(title=f"Fundamental Rayleigh mode, {solution.period:g} s",
           xlabel=r"Reduced displacement / $u_z(0)$", ylabel="Depth (km)",
           ylim=(profile.depth[-1], 0))
    ax.grid(axis="y", color="0.9")
    ax.legend(frameon=False)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path, help="Directory for the two PNG figures")
    output_dir = parser.parse_args().output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    isotropic = basic_dispersion.run()
    vti = basic_dispersion.run(vti=True)
    plot_dispersion(isotropic, vti).savefig(output_dir / "dispersion.png", dpi=150)

    # Repeat the half-space material to sample its eigenfunctions from 10 to 30 km.
    model = d26.create_model([0., 10., 30.], kind="layer_cake", vs=[3., 4.5, 4.5],
                             vp=[5.2, 8., 8.], rho=[2.5, 3.3, 3.3])
    solution = d26.compute_dispersion(model, period=8., wave="Rayleigh", mode=0)
    plot_eigenfunctions(solution).savefig(output_dir / "rayleigh_eigenfunctions.png", dpi=150)


if __name__ == "__main__":
    main()

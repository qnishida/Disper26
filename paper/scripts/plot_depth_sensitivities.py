"""Plot signed Rayleigh ellipticity and Rayleigh/Love group-velocity sensitivities."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from generate_kernel_figures import DATA, PAPER, sha, velocity_basis
from generate_ellipticity_figure import physical


def load(directory, stem, model_hash):
    data = directory / (stem + ".npz")
    report = json.loads((directory / (stem + "_report.json")).read_text())
    if report["data_sha256"] != sha(data) or report["model_sha256"] != model_hash:
        raise ValueError("Model or dataset checksum mismatch: " + stem)
    return np.load(data, allow_pickle=False), report


def plot(kernel_dir, ellipticity_dir):
    model_path = DATA / "prem_like_kernel_model.json"
    model_hash = sha(model_path)
    z = np.asarray(json.loads(model_path.read_text())["depth_km"])
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9,
                         "legend.fontsize": 7, "legend.handlelength": 3.5, "pdf.fonttype": 42})
    fig, grid = plt.subplots(2, 3, figsize=(180 / 25.4, 186 / 25.4), sharey=True)
    colors = ['0.45', '#1765ad', '#39a9b8', '#bc243c', '#e58c21', '#42954b']
    styles = ['-', '--', ':', '-.', (0, (6, 2, 1, 2, 1, 2)), (0, (3, 1, 1, 1))]
    labels = [r'$\rho$', r'$\alpha_H$', r'$\alpha_V$', r'$\beta_H$', r'$\beta_V$', r'$\eta$']
    node_data, node_report = load(ellipticity_dir, 'prem_ellipticity_signed_nodes', model_hash)
    if not node_report['passed']:
        raise ValueError("Ellipticity finite-difference check failed")
    np.testing.assert_array_equal(node_data['depth_km'], z)
    rayleigh_keys = ('Rayleigh_05', 'Rayleigh_10')
    gradients = [node_data[key + '_normalized_gradient'] for key in rayleigh_keys]
    limit = 1.05 * np.ceil(max(np.max(abs(x)) for x in gradients) * 10) / 10
    for ax, gradient in zip(grid[:, 0], gradients):
        for values, color, style in zip(gradient.T, colors, styles):
            ax.plot(values, z, color=color, linestyle=style, lw=1.2,
                    marker='o', ms=3, markerfacecolor='white', markeredgewidth=.7)
        ax.set_xlim(-limit, limit)
        ax.set_xlabel(r'$(p_i/\varepsilon)\,\partial\varepsilon/\partial p_i$')
        ax.locator_params(axis='x', nbins=5)
    node_data.close()

    a,group_report=load(kernel_dir, 'prem_group_kernels_updated', model_hash)
    keys = ('Rayleigh_05', 'Love_05', 'Rayleigh_10', 'Love_10')
    for ax, key in zip(grid[:, 1:].flat, keys):
        p = a[key + '_profile_l8']
        mat = a[key + '_materials_l8'][p[:, 0].astype(int)]
        U = group_report['cases'][key]['U']
        continuous=velocity_basis(p[:,8:14],mat)*physical(mat)/U
        for i,(color,label) in enumerate(zip(colors,labels)):
            xx=[];dd=[]
            for j in range(len(z)-1):
                indices=np.flatnonzero((p[:,1]>z[j])&(p[:,1]<z[j+1]))
                if not len(indices):
                    continue
                indices=indices[np.unique(np.linspace(0,len(indices)-1,min(1000,len(indices))).astype(int))]
                # Extend the nearest interior samples to the exact interface depth.
                # Repeated depths produce horizontal jump connectors, not smoothing.
                xx.extend([continuous[indices[0],i],*continuous[indices,i],continuous[indices[-1],i]])
                dd.extend([z[j],*p[indices,1],z[j+1]])
            ax.plot(xx,dd,color=color,linestyle=styles[i],lw=1.2,label=label)
        ax.set_xlabel(r'$(p/U)K_{U,p}$ (km$^{-1}$)')
        ax.locator_params(axis='x', nbins=5)
    a.close()
    column_titles = ('Rayleigh ellipticity', 'Rayleigh group velocity', 'Love group velocity')
    for row, (frequency, key) in enumerate(zip((.05, .10), rayleigh_keys)):
        for col, (ax, title) in enumerate(zip(grid[row], column_titles)):
            letter = 'abcdef'[row * len(column_titles) + col]
            detail = f'{frequency:.2f} Hz'
            if col == 0:
                epsilon = node_report['cases'][key]['epsilon']
                detail += fr', $\varepsilon={epsilon:.3f}$'
                ax.set_ylabel('Flattened depth (km)')
            ax.set_title(f'({letter}) {title}\n{detail}', loc='left')
            ax.axvline(0, color='0.7', lw=.5, zorder=0)
            ax.grid(alpha=.15)
            ax.set_ylim(100, 0)
    handles = [Line2D([], [], color=c, linestyle=s) for c, s in zip(colors, styles)]
    fig.legend(handles, labels, loc='upper center', ncol=6, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, .95), h_pad=.8)
    output = PAPER / "Fig"
    output.mkdir(exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(output / ("prem_depth_sensitivities." + suffix), dpi=320, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel-data", type=Path, default=DATA)
    parser.add_argument("--ellipticity-data", type=Path, default=PAPER / "results/manuscript/ellipticity")
    args = parser.parse_args()
    plot(args.kernel_data, args.ellipticity_data)

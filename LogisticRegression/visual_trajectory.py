""" 
Visualize the trajectory of different communicators in the 2D PCA space, along with the loss landscape.

We visualize the global model trajectories through PCA. Since all the mechanisms start from the same original model $\theta^0$, we subtract $\theta^0$ from the collected global models to center the trajectories before computing the PCA projection matrix. The trajectories are then projected onto the 2-dimensional subspace spanned by the first two principal components.

How to run: first you need to run unlearn.py with --save_trajectory True to save the global model checkpoints for each communicator. Then you can run this script to visualize the trajectories.

Example command:
```
python visual_trajectory.py --legend_only   # to generate a legend-only figure
python visual_trajectory.py --alpha 1e8     # to visualize the trajectory for alpha=1e8, where alpha is the parameter for Dirichlet distribution in the data partitioning. The default is 1e8, which corresponds to the i.i.d. setting.
```

We visualize the trajectory under i.i.d. setting since the data heterogeneity can introduce additional noise to the trajectory of global models, making it harder to visualize the differences between communicators. The i.i.d. setting provides a clearer view of the trajectory differences.
"""


import os
import argparse

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from sklearn.decomposition import PCA

import sys 
from init import get_args 
from optimizer import *
from loader import *


plt.style.use('ggplot')

# plt.rc('text', usetex=True)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
sns.set_style(style="white") 
sns.set_context("paper", font_scale=1.2) 
mpl.rcParams.update({
    # 'font.family': 'sans-serif',
    # 'font.sans-serif': ['DejaVu Sans'], 
    'font.size': 14, 
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'legend.fontsize': 16,
    'xtick.labelsize': 10,
    'ytick.labelsize': 30
})
cmap = sns.color_palette("crest_r", as_cmap=True)
line_styles = ['-', '--', '-.', ':']
markers = ['o', '^', 'v', 'D', '<', '>', 's', '*']


def get_checkpoint_ef(efs, root_dir="saves/ckpts/", alpha="0.0"):
    """ return a dict of ef: [list of checkpoint paths in order] """
    checkpoints = {}
    for ef in efs:
        ef_dir = os.path.join(root_dir, ef + "_seed0_alpha" + alpha)
        if not os.path.exists(ef_dir):
            continue
        epoch_files = [f for f in os.listdir(ef_dir) if f.endswith('.npy')]
        epoch_nums = [int(f[:-4]) for f in epoch_files]
        epoch_nums.sort()
        checkpoints[ef] = [np.load(os.path.join(ef_dir, f"{epoch}.npy")) for epoch in epoch_nums]
    return checkpoints



def plot_traj(efs, ckpts, X, y, epochs, num_samples, desc, alpha): 
    o_point = ckpts['Uncompressed'][0]
    reference_traj = [ckpts[ef][i + 1] - o_point for i in range(epochs) for ef in efs[1:]] # len(ckpts[efs[0]]) - 1

    pca = PCA(n_components=2)
    pca.fit(reference_traj)

    ckpt_lists = [[ckpt - o_point for ckpt in ckpts[ef]] for ef in efs]
    ckpt_transform = [pca.transform(ckpt) for ckpt in ckpt_lists]

    x_min, x_max = np.min([ckpt[:epochs, 0].min() for ckpt in ckpt_transform]), np.max([ckpt[:epochs, 0].max() for ckpt in ckpt_transform])
    y_min, y_max = np.min([ckpt[:epochs, 1].min() for ckpt in ckpt_transform]), np.max([ckpt[:epochs, 1].max() for ckpt in ckpt_transform])
    dist = max(x_max - x_min, y_max - y_min) * 0.1
    x_min, x_max = round(x_min - dist * 2, 2), round(x_max + dist, 2)
    y_min, y_max = round(y_min - dist * 2, 2), round(y_max + dist, 2)
    print(f"x range: ({x_min:.2f}, {x_max:.2f}), y range: ({y_min:.2f}, {y_max:.2f})")

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, num_samples), np.linspace(y_min, y_max, num_samples))
    grid_points_2d = np.c_[xx.ravel(), yy.ravel()]

    loss_grid = []
    for point in grid_points_2d: 
        w_point = pca.inverse_transform(point) + o_point
        loss_lambda = {"f": 1.0, "R": 0.95, "U": 0.05}[desc]
        loss_value = logistic_loss(w_point, X, y, lam=1e-3) * loss_lambda
        loss_grid.append(loss_value)
    loss_grid = np.array(loss_grid).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(8, 6))

    contourf = ax.contourf(xx, yy, loss_grid, levels=30, cmap=cmap)
    contour = ax.contour(xx, yy, loss_grid, levels=30, colors="gray", linewidths=0.6)
    cb = fig.colorbar(contourf, ax=ax, orientation='vertical')
    cb.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
    cb.ax.tick_params(labelsize=18, labelcolor='black')
    for label in cb.ax.get_yticklabels():
        label.set_fontfamily('sans-serif')

    # ({} Communication Rounds)
    # if desc == "f":
    #     cb.set_label(r'Loss $f$', fontsize=21)
    # elif desc == "R":
    #     cb.set_label(r'Remaining Loss $R$', fontsize=21)
    # else: 
    #     cb.set_label(r'Unlearning Loss $U$', fontsize=21)

    colors = sns.color_palette("husl", n_colors=len(efs))

    for i, traj in enumerate(ckpt_transform):
        color = colors[i]
        label = efs[i]
        linestyle = line_styles[i % len(line_styles)]
        marker = markers[i % len(markers)]

        ax.plot(traj[:epochs, 0], traj[:epochs, 1], label=label, color=color, linestyle=linestyle, linewidth=3, marker=marker, markersize=12, markevery=epochs//5, markeredgewidth=1.5, markeredgecolor='white')

        if i == len(efs) - 1:
            ax.scatter(traj[0, 0], traj[0, 1], 
                    color=color, edgecolors='white', s=400, linewidths=2, zorder=10, marker='P', label='Start')
            
            ax.scatter(traj[epochs, 0], traj[epochs, 1], 
                    color=color, edgecolors='white', s=400, linewidths=2, zorder=10, marker='X', label='End')
        else:
            ax.scatter(traj[0, 0], traj[0, 1], 
                    color=color, edgecolors='white', s=400, linewidths=2, zorder=10, marker='P')
            
            ax.scatter(traj[epochs, 0], traj[epochs, 1], 
                    color=color, edgecolors='white', s=400, linewidths=2, zorder=10, marker='X')

    if desc == "f":
        ax.set_title(r"$U$ ({} Communication Rounds)".format(epochs), fontsize=25, y=-0.1)
    elif desc == "R":
        ax.set_title(r"$R$ ({} Communication Rounds)".format(epochs), fontsize=25, y=-0.1)
    else:
        ax.set_title(r"$F$ ({} Communication Rounds)".format(epochs), fontsize=25, y=-0.1)

    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    if epochs < 100: ax.invert_xaxis()  # ← mirror the x-axis for better visualization

    plt.tight_layout()
    # legend loc='lower right'
    # plt.legend(ncols=2, loc='lower right')

    save_name = f'img/strongconvex_trajectory_{desc}{alpha}_{epochs}_titled.pdf'
    fig.savefig(save_name)
    print(f"Saved trajectory plot to {save_name}")
    plt.show()


def plot_legend_only(efs, save_name='img/strongconvex_legend.pdf'):
    """
    generate a legend-only figure 
    """
    import matplotlib.lines as mlines

    colors = sns.color_palette("husl", n_colors=len(efs))

    fig, ax = plt.subplots(figsize=(len(efs) * 2.2 + 2.5, 1.0))
    ax.set_axis_off()

    handles = []
    for i, ef in enumerate(efs):
        color = colors[i]
        ls = line_styles[i % len(line_styles)]
        mk = markers[i % len(markers)]
        line = ax.plot([], [], label=ef, color=color, linestyle=ls,
                       linewidth=3, marker=mk, markersize=16,
                       markeredgewidth=1.5, markeredgecolor='white')[0]
        handles.append(line)

    last_color = colors[-1]
    start_handle = ax.scatter([], [], marker='P', color=last_color,
                               edgecolors='white', s=500, linewidths=2,
                               label='Start')
    end_handle = ax.scatter([], [], marker='X', color=last_color,
                             edgecolors='white', s=500, linewidths=2,
                             label='End')
    handles.extend([start_handle, end_handle])

    legend = ax.legend(handles=handles, loc='center', ncol=len(efs) + 2,
                       frameon=True, facecolor='lightgray', edgecolor='gray',
                       framealpha=0.5, fontsize=20,
                       handlelength=2.5, handletextpad=0.8,
                       columnspacing=1.5, labelcolor='black')

    fig.tight_layout()
    fig.savefig(save_name, bbox_inches='tight', pad_inches=0.02)
    print(f"Saved legend-only figure to {save_name}")
    plt.show()


if __name__ == "__main__":
    cmd_parser = argparse.ArgumentParser()
    cmd_parser.add_argument('--alpha', type=float, default=1e8, help='alpha value')
    cmd_parser.add_argument('--legend_only', action='store_true', help='Only generate the horizontal legend figure')
    cmd_parser.add_argument('--efs', type=str, nargs='+',
                            default=['Compressed', 'EF14', 'EF21', 'EControl', 'EFFACE', 'Uncompressed'],
                            help='Communicators to include in the legend')
    cmd_args = cmd_parser.parse_args()

    efs = tuple(cmd_args.efs)

    if cmd_args.legend_only:
        plot_legend_only(efs, save_name='img/strongconvex_legend.pdf')
        sys.exit(0)

    # Normal mode: load config and run trajectory
    args = get_args("config/config.yml")
    if cmd_args.alpha is not None:
        args.alpha = cmd_args.alpha

    # load data 
    _, _, _, client_indices, client_unlearn_indices, train_dataset = load_dataloader(args)

    train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)
    X, y = dataloader_to_numpy(train_loader)
    indices = [idx for indices in client_indices for idx in indices]
    unlearn_indices = [idx for indices in client_unlearn_indices for idx in indices]

    y[unlearn_indices] = -y[unlearn_indices]

    remain_indices = np.setdiff1d(indices, unlearn_indices)
    X_unlearn, y_unlearn = X[unlearn_indices], y[unlearn_indices]
    X_remain, y_remain = X[remain_indices], y[remain_indices]

    efs = ("Compressed", "EF14", "EF21", "EControl", "EFFACE", "Uncompressed") 

    ckpts = get_checkpoint_ef(efs, root_dir="saves/ckpts/", alpha=str(args.alpha))

    for epochs in [20, 200]:
            plot_traj(efs, ckpts, X, y, 
                      epochs=epochs, num_samples=100, desc=f"f", alpha=f"{args.alpha}" if args.alpha != 1e8 else "")
            plot_traj(efs, ckpts, X_remain, y_remain, 
                     epochs=epochs, num_samples=100, desc=f"R", alpha=f"{args.alpha}" if args.alpha != 1e8 else "")
            plot_traj(efs, ckpts, X_unlearn, y_unlearn, 
                      epochs=epochs, num_samples=100, desc=f"U", alpha=f"{args.alpha}" if args.alpha != 1e8 else "")

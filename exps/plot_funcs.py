import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # New import

from core.plotter import Plotter


# ======================== Plot train and val accuracy ========================
def retrieve_train_acc(run):
    df = pd.DataFrame(list(run.scan_history()))
    df["epoch"] += 1
    valid_df = df[~df["train/top1"].isna()]
    df_plot = pd.DataFrame({
        name: df.sort_values(by="_step").iloc[-1] for name, df in valid_df.groupby("epoch")
    }).T
    return df_plot


def retrieve_val_acc(run):
    df = run.history(keys=["val/top1"], x_axis="epoch")
    df["epoch"] += 1
    return df


def plot_acc(runs, name_mappings, save_dir, file_name):
    train_acc = {run.config["name"]: retrieve_train_acc(run) for run in runs}
    val_acc = {run.config["name"]: retrieve_val_acc(run) for run in runs}
    plotter = Plotter((1, 2), (16, 6), save_dir=save_dir)
    for k, v in name_mappings.items():
        plotter.plot(0, 0, 'epoch', 'train/top1', data=train_acc[k], label=v)
        plotter.plot(0, 1, 'epoch', 'val/top1', data=val_acc[k], label=v)
    plotter.set_title(0, 0, "Training  Accuracy")
    plotter.set_labels(0, 0, x_label="Epoch", y_label="Accuracy")
    plotter.legend(0, 0)

    plotter.set_title(0, 1, "Validation  Accuracy")
    plotter.set_labels(0, 1, x_label="Epoch", y_label="Accuracy")
    plotter.legend(0, 1)

    plotter.save(file_name)
    return plotter


# ======================== Plot Mean and Var ========================

def scan_history_for_mean_var(run, mean_key, var_key):
    df = pd.DataFrame(list(run.scan_history([mean_key, var_key, "_step"])))
    y_mean = pd.DataFrame(list(df[mean_key]))
    y_var = pd.DataFrame(list(df[var_key]))
    y_min = y_mean - y_var
    y_max = y_mean + y_var
    x = df["_step"]
    return x, y_mean, y_min, y_max


def plot_preact(run, mean_key, var_key, selected_layers, save_dir, save_name, absolute=False):
    x, y_mean, y_min, y_max = scan_history_for_mean_var(run, mean_key, var_key)
    cmap_name = mpl.colormaps["viridis"]
    color_gradient = np.linspace(1, 0, len(selected_layers))
    plotter = Plotter((1, 1), (16, 8), save_dir=save_dir)
    for color_idx, layer_idx in zip(color_gradient, selected_layers):
        plotter.plot(0, 0, x, y_mean.iloc[:, layer_idx], color=cmap_name(color_idx),
                     label=f"Layer {str(layer_idx + 1).zfill(2)}")
        plotter.fill_between(0, 0, x, y_min.iloc[:, layer_idx], y_max.iloc[:, layer_idx], alpha=0.25,
                             color=cmap_name(color_idx))
    plotter.legend(0, 0)
    plotter.set_labels(0, 0, "Steps", "Preactivation")
    plotter.save(save_name)
    return plotter


# ======================== 3D Plot ========================
# TODO Need fix
def plot3d_dist(df, mean_key, var_key):
    fig, axs = plt.subplots(1, 1, subplot_kw={'projection': '3d'})
    x = df["_step"]
    y_mean = pd.DataFrame(list(df[mean_key]))
    y_var = pd.DataFrame(list(df[var_key]))
    y_min = y_mean - y_var
    y_max = y_mean + y_var
    for j in range(y_mean.shape[1]):
        verts = [(x[j], y_max.iloc[j, i], j) for i in range(x.shape[0])] + [(x[j], y_min.iloc[j, i], j) for i in
                                                                            range(x.shape[0])]
        axs.add_collection3d(Poly3DCollection([verts], color='orange'))


def plot3d_distribution(distributions, legends):
    fig, axs = plt.subplots(1, 1, subplot_kw={'projection': '3d'}, figsize=(16, 9))
    axs.set_box_aspect(aspect=(1.6, 1, 0.8))


if __name__ == "__main__":
    api = wandb.Api(timeout=120)

    # import pandas as pd
    # import wandb
    # from matplotlib import pyplot as plt
    # from core.plotter import Plotter
    #
    # api = wandb.Api(timeout=120)
    # run = api.run("orangebai/comp_act/1h409oop")
    # df = run.history()
    print(1)
    # df = pd.DataFrame(list(run.scan_history(["weights_mean", "weights_var", "pre_act/mean", "pre_act/var", "step"])))

import os.path

import matplotlib as mpl
import seaborn as sns
from matplotlib import pyplot as plt


class Plotter:
    def __init__(self, shape, size, save_dir="", rcParams=None):
        rcParams = {
            'axes.titlesize': 22,
            'legend.fontsize': 16,
            'axes.labelsize': 16,
            'xtick.labelsize': 16,
            'ytick.labelsize': 16,
            'figure.titlesize': 22,
        }
        # See: https://matplotlib.org/stable/tutorials/introductory/customizing.html

        self.shape = shape
        self.save_dir = save_dir

        plt.style.use('seaborn')
        sns.set_theme(style="darkgrid")
        mpl.rcParams.update(rcParams if rcParams else {})
        # mpl.rcParams.update({"xtick.labelsize": 30})

        fig_height = shape[1] * size[0]
        fig_width = shape[0] * size[1]
        self.fig, self.ax = plt.subplots(*shape, figsize=(fig_height, fig_width))

    def set_labels(self, i, j, x_label, y_label, fontsize=16):
        ax = self._get_axis(i, j)
        ax.set_xlabel(x_label, fontsize=fontsize)
        ax.set_ylabel(y_label, fontsize=fontsize)

    def _get_axis(self, i, j):
        temp = self.ax
        if self.shape[0] != 1:
            temp = temp[i]
        if self.shape[1] != 1:
            temp = temp[j]
        return temp

    def plot(self, i, j, *args, **kwargs):
        ax = self._get_axis(i, j)
        ax.plot(*args, **kwargs)

    def legend(self, i, j, fontsize=16, loc="best"):
        ax = self._get_axis(i, j)
        ax.legend(fontsize=fontsize, loc=loc)

    def set_title(self, i, j, title):
        ax = self._get_axis(i, j)
        ax.set_title(title)

    def save(self, name):
        self.fig.savefig(os.path.join(self.save_dir, name + '.png'), bbox_inches='tight')

    def fill_between(self, i, j, x, y_min, y_max, *args, **kwargs):
        ax = self._get_axis(i, j)
        ax.fill_between(x, y_min, y_max, *args, **kwargs)

# if __name__ == "__main__":
#     plotter = Plotter((2, 1), (18, 8))
#     plotter.ax[0].set_title("aaa")
#     plotter.fig.show()
#
#     print(1)
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib as mpl
# from cycler import cycler
#
# plt.style.use('seaborn')
# mpl.rcParams['ytick.labelsize'] = 20
# mpl.rcParams['lines.linestyle'] = '--'
# fig, ax = plt.subplots()
# data = np.random.randn(50)
# ax.plot(data)
#
# plt.show()

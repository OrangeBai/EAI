from matplotlib import pyplot as plt


class Plotter:
    def __init__(self, shape, size):
        self.shape = shape
        fig_height = shape[0] * size[1]
        fig_width = shape[1] * size[0]
        self.fig, self.ax = plt.subplots(*shape, figsize=(fig_height, fig_width))
        print(1)

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

    def legend(self, i, j, *args, **kwargs):
        ax = self._get_axis(i, j)
        ax.legend()


if __name__ == "__main__":
    plotter = Plotter((2, 1), (18, 8))

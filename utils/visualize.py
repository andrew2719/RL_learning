import matplotlib.pyplot as plt


def show_grid(grid):

    plt.imshow(grid, cmap="viridis")
    plt.title("Grid World")
    plt.colorbar()
    plt.show()

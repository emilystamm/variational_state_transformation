import matplotlib.pyplot as plt

def plot_cost_fn(x, y, title=None, file='plot.png', subtitle = None):
    try: 
        x_name = 'Iterations' 
        y_name = 'Cost'

        fig, ax = plt.subplots()
        ax.margins(0.05)
        for name in y.keys():
            ax.plot(x, y[name], marker='o', linestyle='-', label=name)
        ax.legend()
        plt.xlabel(x_name)
        plt.ylabel(y_name)
        if subtitle: plt.title(subtitle)
        if title: plt.suptitle(title)
        plt.savefig(file)
        plt.show()
    except:
        print("Error,could not plot {}".format(title))
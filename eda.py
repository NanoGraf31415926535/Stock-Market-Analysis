import matplotlib.pyplot as plt

def plot_series(data, x, y, title, xlabel, ylabel, kind='line'):
    plt.figure(figsize=(14, 7))
    if kind == 'line':
        plt.plot(data[x], data[y], label=y)
    elif kind == 'bar':
        plt.bar(data[x], data[y], color='orange', label=y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()  # Ensure labels exist before calling legend()
    plt.xticks(rotation=45)
    plt.show()

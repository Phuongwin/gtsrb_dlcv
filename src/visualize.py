from collections import Counter
import matplotlib.pyplot as plt

def view_data_distribution(dataset, name: str):
    distribution = dict(Counter(dataset.targets))
    
    keys = list(distribution.keys())
    values = list(distribution.values())

    plt.barh(keys, values)

    for index, value in enumerate(values):
        plt.text(value, index, str(value))

    plt.savefig(f'visualizations/{name}.png')

def confusion_matrix():
    raise NotImplementedError
import seaborn as sns
import torch

from collections import Counter
import matplotlib.pyplot as plt

'''
Collection of visualization functions
'''

def view_data_distribution(dataset, name: str):
    distribution = dict(Counter(dataset.targets))
    
    keys = list(distribution.keys())
    values = list(distribution.values())

    fig, ax = plt.subplots()
    fig.set_size_inches(18.5, 10.5)
    ax.set_title("GTSRB Data Imbalances")
    ax.set_xlabel('Example count (int)')
    ax.set_ylabel('Class (int)')

    plt.barh(keys, values)

    for index, value in enumerate(values):
        plt.text(value, index, str(value))

    plt.savefig(f'visualizations/{name}.png')

def plot_train_validation(count, train_plot, valid_plot, experiment, type):
    plt.figure(figsize=(10,10))
    plt.plot(count, train_plot, color='blue', label='Train')
    plt.plot(count, valid_plot, color='magenta', label='Validation')
    plt.xlabel('Epochs')
    plt.ylabel(f'{type}')
    plt.legend()
    plt.title(f'{experiment} CNN Training/Validation {type}')
    plt.savefig(f'visualizations/{experiment.lower()}_{type.lower()}.png', dpi=300)

def save_confusion_matrix(cf, experiment):
    normalized_cf = cf / torch.sum(cf)

    fig, ax = plt.subplots()
    fig.set_size_inches(30, 20)

    ax = sns.heatmap(normalized_cf, annot=True, fmt='.1%', cmap="Blues")
    
    ax.set_title(f'{experiment} GTSRB Confusion Matrix')
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values')

    plt.savefig(f'visualizations/{experiment.lower()}_confusion_matrix_GTSRB.png', dpi = 300)
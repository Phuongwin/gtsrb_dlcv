import seaborn as sns
import torch

from collections import Counter
import matplotlib.pyplot as plt

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

def plot_train_validation(count, train_plot, valid_plot, name, save_name):
    plt.figure(figsize=(10,10))
    plt.plot(count, train_plot)
    plt.plot(count, valid_plot)
    plt.xlabel('Epochs')
    plt.ylabel(f'{name}')
    # plt.title(f'Imbalanced Handled CNN Training/Validation {name}')

    plt.savefig(f'visualizations/{save_name}.png', dpi=300)

def save_confusion_matrix(cf):
    
    normalized_cf = cf / torch.sum(cf)

    cf_heatmap = sns.heatmap(normalized_cf, annot=True, fmt='.1%', cmap="Blues")
    
    cf_heatmap.set_title("GTSRB Confusion Matrix with labels (as original integers)")
    cf_heatmap.set_xlabel('\nPredicted Values')
    cf_heatmap.set_ylabel('Actual Values')

    sns.set(rc={'figure.figsize': (35, 25)})

    plt.savefig('visualizations/confusion_matrix_GTSRB.png', dpi = 300)
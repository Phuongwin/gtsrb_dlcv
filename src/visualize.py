import seaborn as sns
import torch

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

def save_confusion_matrix(cf):
    
    normalized_cf = cf / torch.sum(cf)

    cf_heatmap = sns.heatmap(normalized_cf, annot=True, fmt='.1%', cmap="Blues")
    
    cf_heatmap.set_title("GTSRB Confusion Matrix with labels (as original integers)")
    cf_heatmap.set_xlabel('\nPredicted Values')
    cf_heatmap.set_ylabel('Actual Values')

    sns.set(rc={'figure.figsize': (35, 25)})

    plt.savefig('visualizations/confusion_matrix_GTSRB.png', dpi = 300)
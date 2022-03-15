# GTSRB Image Classification
Purpose: Perform Image classification on GTSRB dataset using a Convolutional Neural Network. Configure two separate experiments, first a baseline CNN preprocessing and training procedure and second an enhanced CNN preprocessing and training procedure that mitigates class imbalances through the usage of data augmentation and class imbalance mitigation.

This repository satisfies the requirements of the midterm project for the course 525.733 Deep Learning for Computer Vision at Johns Hopkins University. 

# Building
This repository uses a pipenv as a virtual environment to run the source code. To create the environment, perform the following:
```
cd gtsrb_dlcv
pipenv install
```
A Pipfile.lock file should appear which signifies the creation of the environment. To activate the environment, run the following:
```
pipenv shell
```

All dependencies should be install with Python 3.9.

# Usage
Configuring experiments have been simplified through the usage of a config.yaml file. Opening the file introduces multiple configuration parameters and hyperparameters to determine how to run the training and inferening script. Below each parameter is defined with default values associated.

**Configurations**
```
training (Boolean: True)
    - Determines whether to perform training script.
inference (Boolean: True)
    - Determines whether to perform inferencing - script automatically attempts to load default weights provided
imbalances (Boolean: True)
    - Determines whether to perform data augmentation and Weighted Random Sampler. Both are conjoined within this parameter.
dataset_path (String: './data/Final_Training/Images')
    - Points towards the root directory of the training data
save_path (String: './saved_models/weights_custom.pth')
    - Defines the location to save weights - include name of file.
weight_path (String: './saved_models/weights_custom.pth')
    - Points towards the location of the weights to be loaded.
```

**Hyperparameters**
```
train_allocation (Float: 0.8)
    - Percentage of dataset to be allocated towards the training set.
epoch (Integer: 2)
    - Defines number of passes of the entire training set.
batch_size (Integer: 32)
    - Define number of training examples utilized in one iteration.
learning_rate (Float: 0.001)
    - Define step size at each iteration while moving towards a minimum of loss function.
momentum (Float: 0.9)
    - Define momentum factor that contributes towards a gekoubg accelerating gradient vectors.
```

Once configurations and hyperparameters are desired values, run the script from the root directory of the repository as such:
```
python ./src/main.py
```

# Directory Structure
```
GTSRB_DLCV
|─── data
|       Final_Training
|           00000
|           00001
|           ...
|       signnames.csv
|─── notebooks
|       gtsrb_baseline.ipynb
|       gtsrb_enhanced.ipynb
|─── saved_models
|       basic_weights_colab_gpu.pth
|       basic_weights_local_cpu.pth
|       enhanced_weights_colab_gpu.pth
|       enhanced_weights_local_cpu.pth
|─── src
|       models
|           basic_cnn.py
|       main.py
|       model.py
|       preprocessing.py
|       transform.py
|       visualize.py
|─── visualizations
|       basic_accuracy.png
|       basic_confusion_matrix_GTSRB.png
|       basic_loss.png
|       data_distribution.png
|       enhanced_accuracy.png
|       enhanced_confusion_matrix_GTSRB.png
|       enhanced_loss.png
|─── Pipfile
|─── Pipfile.lock
|─── README.md
```

**Note:** Weights provide have been trained via Google Colab on the GPU or locally on the CPU. The author of this repository lacks a local GPU to use, hence depended on Google Colab for much of the experiment. All visualizations were produced using this repository using only 2 epochs.
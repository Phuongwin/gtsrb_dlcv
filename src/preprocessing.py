from collections import Counter



def handle_imbalances(dataset):
    #https://cs230.stanford.edu/projects_fall_2020/reports/55824835.pdf 
    # Potential Approach is to use WeightedRandomSample
    #https://towardsdatascience.com/getting-started-with-albumentation-winning-deep-learning-image-augmentation-technique-in-pytorch-47aaba0ee3f8
    print (dict(Counter(dataset.targets)))


    
    return dataset
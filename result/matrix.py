import os
import pandas as pd
import numpy as np
from my_utils import performance_util
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html
    model_names = ['mlp']#['mlp', 'rf', 'mlp_cnn', 'svm']
    status = 'fe'
    for model_name in model_names:
        # performance_util.get_sum_confusion_matrix(model_name, status)
        report = performance_util.get_average_classification_report(model_name, status)
        print(model_name)
        print(report)
        print('------------')
    print('done')


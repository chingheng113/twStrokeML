import os
import pandas as pd
import numpy as np
from my_utils import performance_util
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html

    model_names = ['mlp', 'rf', 'mlp_cnn', 'svm']
    # all, fs
    status = 'fs'
    # ischemic, hemorrhagic
    sub_class = 'hemorrhagic'
    hold_out_round = 0
    for model_name in model_names:
        precisions, recalls, fscores = performance_util.get_all_performance_scores(model_name, sub_class, status)
        print(model_name)
        print(round(np.mean(precisions), 3))
        print(round(np.std(precisions), 3))
        print(round(np.mean(recalls), 3))
        print(round(np.std(recalls), 3))
        print(round(np.mean(fscores), 3))
        print(round(np.std(fscores), 3))
        # report = performance_util.get_average_test_classification_report(model_name, sub_class, status)
        # print(report)
print('done')


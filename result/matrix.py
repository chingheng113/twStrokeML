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
    sub_class = 'ischemic'
    hold_out_round = 0
    for model_name in model_names:
        # performance_util.get_sum_confusion_matrix(model_name, status)
        report = performance_util.get_average_classification_report(hold_out_round, model_name, sub_class, status)
        report_hold = performance_util.get_holdout_classification_report(hold_out_round, model_name, sub_class, status)
        print(model_name)
        print(report)
        print('----')
        print(report_hold)
        print('############')
    print('done')


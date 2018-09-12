import os
import pandas as pd
import numpy as np
from my_utils import performance_util
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html
    model_names = ['mlp'] #['mlp', 'rf', 'mlp_cnn', 'svm']
    status = 'fs'
    # all, ischemic, hemorrhagic
    sub_class = 'hemorrhagic'

    for model_name in model_names:
        mean_fpr, mean_tpr, mean_auc, std_auc = performance_util.average_roc_auc(model_name, status, sub_class)
        plt.plot(mean_fpr, mean_tpr,
                 label=model_name+' (AUC = %0.3f $\pm$ %0.3f)' % (mean_auc, std_auc),
                 lw=1, alpha=.8)
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='black', label='Luck', alpha=.8)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('10-fold ROC curve of selected/extracted features dataset')
    plt.legend(loc="lower right")
    plt.show()

    #
    for model_name in model_names:
        fpr, tpr, auc = performance_util.calculate_holdout_roc_auc(model_name, status, sub_class)
        plt.plot(fpr, tpr,
                 label=model_name+' (AUC = %0.3f )' % auc,
                 lw=1, alpha=.8)
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='black', label='Luck', alpha=.8)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Testing ROC curve of selected/extracted features dataset')
    plt.legend(loc="lower right")
    plt.show()
    print('done')



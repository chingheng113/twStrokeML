import os
import pandas as pd
import numpy as np
from my_utils import performance_util
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html
    model_names = ['mlp', 'rf', 'mlp_cnn', 'svm']
    status = 'fs'
    for model_name in model_names:
        mean_fpr, mean_tpr, mean_auc, std_auc = performance_util.average_roc_auc(model_name, status)
        plt.plot(mean_fpr, mean_tpr,
                 label=model_name+' ROC (AUC = %0.3f $\pm$ %0.3f)' % (mean_auc, std_auc),
                 lw=1, alpha=.8)
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Luck', alpha=.8)
    plt.legend(loc="lower right")
    plt.show()

    # print(normal_mean_auc, normal_std_auc)
    # print(fs_mean_auc, fs_std_auc)
    # print(fe_mean_auc, fe_std_auc)
    print('done')



import os
import pandas as pd
import numpy as np
from my_utils import performance_util
import matplotlib.pyplot as plt


if __name__ == '__main__':
    model_names = ['mlp', 'rf', 'mlp_cnn', 'svm']
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(20,20))
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.3)

    plt.subplot(331)
    for model_name in model_names:
        fpr, tpr, auc = performance_util.calculate_holdout_roc_auc_all(model_name, 'normal')
        plt.plot(fpr, tpr,
                 label=model_name+' (AUC = %0.3f )' % auc,
                 lw=1, alpha=.8)
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='black', alpha=.8)
    plt.title('All type stroke - All features', fontsize=14)
    plt.legend(loc="lower right", prop={'size': 12})

    plt.subplot(332)
    for model_name in model_names:
        fpr, tpr, auc = performance_util.calculate_holdout_roc_auc_all(model_name, 'fs')
        plt.plot(fpr, tpr,
                 label=model_name+' (AUC = %0.3f )' % auc,
                 lw=1, alpha=.8)
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='black', alpha=.8)
    plt.title('All type stroke - Feature selection', fontsize=14)
    plt.legend(loc="lower right", prop={'size': 12})

    plt.subplot(333)
    for model_name in model_names:
        fpr, tpr, auc = performance_util.calculate_holdout_roc_auc_all(model_name, 'fe')
        plt.plot(fpr, tpr,
                 label=model_name+' (AUC = %0.3f )' % auc,
                 lw=1, alpha=.8)
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='black', alpha=.8)
    plt.title('All type stroke - Feature extension', fontsize=14)
    plt.legend(loc="lower right", prop={'size': 12})

    plt.subplot(334)
    for model_name in model_names:
        fpr, tpr, auc = performance_util.calculate_holdout_roc_auc(model_name, 'normal', 'ischemic')
        plt.plot(fpr, tpr,
                 label=model_name+' (AUC = %0.3f )' % auc,
                 lw=1, alpha=.8)
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='black', alpha=.8)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.title('Ischemic stroke - All Features', fontsize=14)
    plt.legend(loc="lower right", prop={'size': 12})


    plt.subplot(335)
    for model_name in model_names:
        fpr, tpr, auc = performance_util.calculate_holdout_roc_auc(model_name, 'fs', 'ischemic')
        plt.plot(fpr, tpr,
                 label=model_name+' (AUC = %0.3f )' % auc,
                 lw=1, alpha=.8)
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='black', alpha=.8)
    plt.title('Ischemic stroke - Feature selection', fontsize=14)
    plt.legend(loc="lower right", prop={'size': 12})

    plt.subplot(336)
    for model_name in model_names:
        fpr, tpr, auc = performance_util.calculate_holdout_roc_auc(model_name, 'fe', 'ischemic')
        plt.plot(fpr, tpr,
                 label=model_name+' (AUC = %0.3f )' % auc,
                 lw=1, alpha=.8)
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='black', alpha=.8)
    plt.title('Ischemic stroke - Feature extension', fontsize=14)
    plt.legend(loc="lower right", prop={'size': 12})

    plt.subplot(337)
    for model_name in model_names:
        fpr, tpr, auc = performance_util.calculate_holdout_roc_auc(model_name, 'normal', 'hemorrhagic')
        plt.plot(fpr, tpr,
                 label=model_name+' (AUC = %0.3f )' % auc,
                 lw=1, alpha=.8)
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='black', alpha=.8)
    plt.title('Hemorrhagic stroke - All Features', fontsize=14)
    plt.legend(loc="lower right", prop={'size': 12})

    plt.subplot(338)
    for model_name in model_names:
        fpr, tpr, auc = performance_util.calculate_holdout_roc_auc(model_name, 'fs', 'hemorrhagic')
        plt.plot(fpr, tpr,
                 label=model_name+' (AUC = %0.3f )' % auc,
                 lw=1, alpha=.8)
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='black', alpha=.8)
    plt.title('Hemorrhagic stroke - Feature selection', fontsize=14)
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.legend(loc="lower right", prop={'size': 12})

    plt.subplot(339)
    for model_name in model_names:
        fpr, tpr, auc = performance_util.calculate_holdout_roc_auc(model_name, 'fe', 'hemorrhagic')
        plt.plot(fpr, tpr,
                 label=model_name+' (AUC = %0.3f )' % auc,
                 lw=1, alpha=.8)
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='black', alpha=.8)
    plt.title('Hemorrhagic stroke - Feature extension', fontsize=14)
    plt.legend(loc="lower right", prop={'size': 12})

    fig.savefig("roc.png", dpi=300)
    plt.show()
    print('done')

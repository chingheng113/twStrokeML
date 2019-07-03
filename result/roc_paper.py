import os
import pandas as pd
import numpy as np
from my_utils import performance_util
import matplotlib.pyplot as plt


if __name__ == '__main__':
    model_names = ['mlp', 'rf', 'mlp_cnn', 'svm']
    present_names = ['ANN', 'RF', 'HANN', 'SVM']
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20,20))
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.3)

    # --- hold
    plt.subplot(231)
    for inx, model_name in enumerate(model_names):
        mean_fpr, mean_tpr, mean_auc, std_auc = performance_util.average_roc_auc(model_name, 'all', 'ischemic', 'hold')
        plt.plot(mean_fpr, mean_tpr,
                 label=present_names[inx]+' (AUC = %0.3f ± %0.3f)' % (mean_auc, std_auc),
                 lw=1, alpha=.8)
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='black', alpha=.8)
    plt.title('Ischemic stroke with all features', fontsize=14)
    plt.legend(loc="lower right", prop={'size': 12})

    plt.subplot(232)
    for inx, model_name in enumerate(model_names):
        mean_fpr, mean_tpr, mean_auc, std_auc = performance_util.average_roc_auc(model_name, 'all', 'hemorrhagic', 'hold')
        plt.plot(mean_fpr, mean_tpr,
                 label=present_names[inx]+' (AUC = %0.3f ± %0.3f)' % (mean_auc, std_auc),
                 lw=1, alpha=.8)
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='black', alpha=.8)
    plt.title('Hemorrhagic stroke with all features', fontsize=14)
    plt.legend(loc="lower right", prop={'size': 12})

    plt.subplot(233)
    for inx, model_name in enumerate(model_names):
        mean_fpr, mean_tpr, mean_auc, std_auc = performance_util.average_roc_auc(model_name, 'fs', 'ischemic', 'hold')
        plt.plot(mean_fpr, mean_tpr,
                 label=present_names[inx]+' (AUC = %0.3f ± %0.3f)' % (mean_auc, std_auc),
                 lw=1, alpha=.8)
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='black', alpha=.8)
    plt.title('Ischemic stroke with selected features', fontsize=14)
    plt.legend(loc="lower right", prop={'size': 12})

    plt.subplot(234)
    for inx, model_name in enumerate(model_names):
        mean_fpr, mean_tpr, mean_auc, std_auc = performance_util.average_roc_auc(model_name, 'fs', 'hemorrhagic', 'hold')
        plt.plot(mean_fpr, mean_tpr,
                 label=present_names[inx]+' (AUC = %0.3f ± %0.3f)' % (mean_auc, std_auc),
                 lw=1, alpha=.8)
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='black', alpha=.8)
    plt.title('Hemorrhagic stroke with selected features', fontsize=14)
    plt.legend(loc="lower right", prop={'size': 12})

    plt.subplot(235)
    for inx, model_name in enumerate(model_names):
        mean_fpr, mean_tpr, mean_auc, std_auc = performance_util.average_roc_auc(model_name, 'fs_nf', 'ischemic', 'hold')
        plt.plot(mean_fpr, mean_tpr,
                 label=present_names[inx]+' (AUC = %0.3f ± %0.3f)' % (mean_auc, std_auc),
                 lw=1, alpha=.8)
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='black', alpha=.8)
    plt.title('Ischemic stroke with selected features', fontsize=14)
    plt.legend(loc="lower right", prop={'size': 12})

    # --- testing
    # plt.subplot(221)
    # for inx, model_name in enumerate(model_names):
    #     mean_fpr, mean_tpr, mean_auc, std_auc = performance_util.average_roc_auc(model_name, 'all', 'ischemic', 'test')
    #     plt.plot(mean_fpr, mean_tpr,
    #              label=present_names[inx]+' (AUC = %0.3f ± %0.3f)' % (mean_auc, std_auc),
    #              lw=1, alpha=.8)
    # plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='black', alpha=.8)
    # plt.title('Ischemic stroke with all features', fontsize=14)
    # plt.legend(loc="lower right", prop={'size': 12})
    #
    # plt.subplot(222)
    # for inx, model_name in enumerate(model_names):
    #     mean_fpr, mean_tpr, mean_auc, std_auc = performance_util.average_roc_auc(model_name, 'all', 'hemorrhagic', 'test')
    #     plt.plot(mean_fpr, mean_tpr,
    #              label=present_names[inx]+' (AUC = %0.3f ± %0.3f)' % (mean_auc, std_auc),
    #              lw=1, alpha=.8)
    # plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='black', alpha=.8)
    # plt.title('Hemorrhagic stroke with all features', fontsize=14)
    # plt.legend(loc="lower right", prop={'size': 12})
    #
    # plt.subplot(223)
    # for inx, model_name in enumerate(model_names):
    #     mean_fpr, mean_tpr, mean_auc, std_auc = performance_util.average_roc_auc(model_name, 'fs', 'ischemic', 'test')
    #     plt.plot(mean_fpr, mean_tpr,
    #              label=present_names[inx]+' (AUC = %0.3f ± %0.3f)' % (mean_auc, std_auc),
    #              lw=1, alpha=.8)
    # plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='black', alpha=.8)
    # plt.title('Ischemic stroke with selected features', fontsize=14)
    # plt.legend(loc="lower right", prop={'size': 12})
    #
    # plt.subplot(224)
    # for inx, model_name in enumerate(model_names):
    #     mean_fpr, mean_tpr, mean_auc, std_auc = performance_util.average_roc_auc(model_name, 'fs', 'hemorrhagic', 'test')
    #     plt.plot(mean_fpr, mean_tpr,
    #              label=present_names[inx]+' (AUC = %0.3f ± %0.3f)' % (mean_auc, std_auc),
    #              lw=1, alpha=.8)
    # plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='black', alpha=.8)
    # plt.title('Hemorrhagic stroke with selected features', fontsize=14)
    # plt.legend(loc="lower right", prop={'size': 12})

    fig.savefig("roc.png", dpi=300)
    plt.show()
    print('done')



# plt.subplot(222)
# for inx, model_name in enumerate(model_names):
#     mean_fpr, mean_tpr, mean_auc, std_auc = performance_util.average_roc_auc(model_name, 'fs', 'ischemic', 'hold')
#     plt.plot(mean_fpr, mean_tpr,
#              label=present_names[inx]+' (AUC = %0.3f ± %0.3f)' % (mean_auc, std_auc),
#              lw=1, alpha=.8)
# plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='black', alpha=.8)
# plt.title('ischemic - hold - fs', fontsize=14)
# plt.legend(loc="lower right", prop={'size': 12})
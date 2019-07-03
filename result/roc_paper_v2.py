from my_utils import performance_util
import matplotlib.pyplot as plt


if __name__ == '__main__':
    mean_fpr, mean_tpr, mean_auc, std_auc = performance_util.average_roc_auc('rf', 'all_nf', 'ischemic', 'hold')
    plt.plot(mean_fpr, mean_tpr,
             label='clinical data (AUC = %0.3f ± %0.3f)' % (mean_auc, std_auc),
             lw=1, alpha=.8)
    mean_fpr, mean_tpr, mean_auc, std_auc = performance_util.average_roc_auc('rf', 'all', 'ischemic', 'hold')
    plt.plot(mean_fpr, mean_tpr,
             label='clinical data with follow-up (AUC = %0.3f ± %0.3f)' % (mean_auc, std_auc),
             lw=1, alpha=.8)
    mean_fpr, mean_tpr, mean_auc, std_auc = performance_util.average_roc_auc('rf', 'fs', 'ischemic', 'hold')
    plt.plot(mean_fpr, mean_tpr,
             label='Feature selection (AUC = %0.3f ± %0.3f)' % (mean_auc, std_auc),
             lw=1, alpha=.8)
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='black', alpha=.8)
    plt.title('Random Forest on ischemic stroke', fontsize=14)
    plt.legend(loc="lower right", prop={'size': 12})
    plt.show()

    # ==
    mean_fpr, mean_tpr, mean_auc, std_auc = performance_util.average_roc_auc('rf', 'all_nf', 'hemorrhagic', 'hold')
    plt.plot(mean_fpr, mean_tpr,
             label='clinical data (AUC = %0.3f ± %0.3f)' % (mean_auc, std_auc),
             lw=1, alpha=.8)
    mean_fpr, mean_tpr, mean_auc, std_auc = performance_util.average_roc_auc('rf', 'all', 'hemorrhagic', 'hold')
    plt.plot(mean_fpr, mean_tpr,
             label='clinical data with follow-up (AUC = %0.3f ± %0.3f)' % (mean_auc, std_auc),
             lw=1, alpha=.8)
    mean_fpr, mean_tpr, mean_auc, std_auc = performance_util.average_roc_auc('rf', 'fs', 'hemorrhagic', 'hold')
    plt.plot(mean_fpr, mean_tpr,
             label='Feature selection (AUC = %0.3f ± %0.3f)' % (mean_auc, std_auc),
             lw=1, alpha=.8)
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='black', alpha=.8)
    plt.title('Random Forest on hemorrhagic stroke', fontsize=14)
    plt.legend(loc="lower right", prop={'size': 12})
    plt.show()

    # ==
    mean_fpr, mean_tpr, mean_auc, std_auc = performance_util.average_roc_auc('svm', 'all_nf', 'ischemic', 'hold')
    plt.plot(mean_fpr, mean_tpr,
             label='clinical data (AUC = %0.3f ± %0.3f)' % (mean_auc, std_auc),
             lw=1, alpha=.8)
    mean_fpr, mean_tpr, mean_auc, std_auc = performance_util.average_roc_auc('svm', 'all', 'ischemic', 'hold')
    plt.plot(mean_fpr, mean_tpr,
             label='clinical data with follow-up (AUC = %0.3f ± %0.3f)' % (mean_auc, std_auc),
             lw=1, alpha=.8)
    mean_fpr, mean_tpr, mean_auc, std_auc = performance_util.average_roc_auc('svm', 'fs', 'ischemic', 'hold')
    plt.plot(mean_fpr, mean_tpr,
             label='Feature selection (AUC = %0.3f ± %0.3f)' % (mean_auc, std_auc),
             lw=1, alpha=.8)
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='black', alpha=.8)
    plt.title('SVM on ischemic stroke', fontsize=14)
    plt.legend(loc="lower right", prop={'size': 12})
    plt.show()

    # ==
    mean_fpr, mean_tpr, mean_auc, std_auc = performance_util.average_roc_auc('svm', 'all_nf', 'hemorrhagic', 'hold')
    plt.plot(mean_fpr, mean_tpr,
             label='clinical data (AUC = %0.3f ± %0.3f)' % (mean_auc, std_auc),
             lw=1, alpha=.8)
    mean_fpr, mean_tpr, mean_auc, std_auc = performance_util.average_roc_auc('svm', 'all', 'hemorrhagic', 'hold')
    plt.plot(mean_fpr, mean_tpr,
             label='clinical data with follow-up (AUC = %0.3f ± %0.3f)' % (mean_auc, std_auc),
             lw=1, alpha=.8)
    mean_fpr, mean_tpr, mean_auc, std_auc = performance_util.average_roc_auc('svm', 'fs', 'hemorrhagic', 'hold')
    plt.plot(mean_fpr, mean_tpr,
             label='Feature selection (AUC = %0.3f ± %0.3f)' % (mean_auc, std_auc),
             lw=1, alpha=.8)
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='black', alpha=.8)
    plt.title('SVM on hemorrhagic stroke', fontsize=14)
    plt.legend(loc="lower right", prop={'size': 12})
    plt.show()

    # ==
    mean_fpr, mean_tpr, mean_auc, std_auc = performance_util.average_roc_auc('mlp', 'all_nf', 'ischemic', 'hold')
    plt.plot(mean_fpr, mean_tpr,
             label='clinical data (AUC = %0.3f ± %0.3f)' % (mean_auc, std_auc),
             lw=1, alpha=.8)
    mean_fpr, mean_tpr, mean_auc, std_auc = performance_util.average_roc_auc('mlp', 'all', 'ischemic', 'hold')
    plt.plot(mean_fpr, mean_tpr,
             label='clinical data with follow-up (AUC = %0.3f ± %0.3f)' % (mean_auc, std_auc),
             lw=1, alpha=.8)
    mean_fpr, mean_tpr, mean_auc, std_auc = performance_util.average_roc_auc('mlp', 'fs', 'ischemic', 'hold')
    plt.plot(mean_fpr, mean_tpr,
             label='Feature selection (AUC = %0.3f ± %0.3f)' % (mean_auc, std_auc),
             lw=1, alpha=.8)
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='black', alpha=.8)
    plt.title('ANN on ischemic stroke', fontsize=14)
    plt.legend(loc="lower right", prop={'size': 12})
    plt.show()

    # ==
    mean_fpr, mean_tpr, mean_auc, std_auc = performance_util.average_roc_auc('mlp', 'all_nf', 'hemorrhagic', 'hold')
    plt.plot(mean_fpr, mean_tpr,
             label='clinical data (AUC = %0.3f ± %0.3f)' % (mean_auc, std_auc),
             lw=1, alpha=.8)
    mean_fpr, mean_tpr, mean_auc, std_auc = performance_util.average_roc_auc('mlp', 'all', 'hemorrhagic', 'hold')
    plt.plot(mean_fpr, mean_tpr,
             label='clinical data with follow-up (AUC = %0.3f ± %0.3f)' % (mean_auc, std_auc),
             lw=1, alpha=.8)
    mean_fpr, mean_tpr, mean_auc, std_auc = performance_util.average_roc_auc('mlp', 'fs', 'hemorrhagic', 'hold')
    plt.plot(mean_fpr, mean_tpr,
             label='Feature selection (AUC = %0.3f ± %0.3f)' % (mean_auc, std_auc),
             lw=1, alpha=.8)
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='black', alpha=.8)
    plt.title('ANN on hemorrhagic stroke', fontsize=14)
    plt.legend(loc="lower right", prop={'size': 12})
    plt.show()

    # ==
    mean_fpr, mean_tpr, mean_auc, std_auc = performance_util.average_roc_auc('mlp_cnn', 'all_nf', 'ischemic', 'hold')
    plt.plot(mean_fpr, mean_tpr,
             label='clinical data (AUC = %0.3f ± %0.3f)' % (mean_auc, std_auc),
             lw=1, alpha=.8)
    mean_fpr, mean_tpr, mean_auc, std_auc = performance_util.average_roc_auc('mlp_cnn', 'all', 'ischemic', 'hold')
    plt.plot(mean_fpr, mean_tpr,
             label='clinical data with follow-up (AUC = %0.3f ± %0.3f)' % (mean_auc, std_auc),
             lw=1, alpha=.8)
    mean_fpr, mean_tpr, mean_auc, std_auc = performance_util.average_roc_auc('mlp_cnn', 'fs', 'ischemic', 'hold')
    plt.plot(mean_fpr, mean_tpr,
             label='Feature selection (AUC = %0.3f ± %0.3f)' % (mean_auc, std_auc),
             lw=1, alpha=.8)
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='black', alpha=.8)
    plt.title('HANN on ischemic stroke', fontsize=14)
    plt.legend(loc="lower right", prop={'size': 12})
    plt.show()

    # ==
    mean_fpr, mean_tpr, mean_auc, std_auc = performance_util.average_roc_auc('mlp_cnn', 'all_nf', 'hemorrhagic', 'hold')
    plt.plot(mean_fpr, mean_tpr,
             label='clinical data (AUC = %0.3f ± %0.3f)' % (mean_auc, std_auc),
             lw=1, alpha=.8)
    mean_fpr, mean_tpr, mean_auc, std_auc = performance_util.average_roc_auc('mlp_cnn', 'all', 'hemorrhagic', 'hold')
    plt.plot(mean_fpr, mean_tpr,
             label='clinical data with follow-up (AUC = %0.3f ± %0.3f)' % (mean_auc, std_auc),
             lw=1, alpha=.8)
    mean_fpr, mean_tpr, mean_auc, std_auc = performance_util.average_roc_auc('mlp_cnn', 'fs', 'hemorrhagic', 'hold')
    plt.plot(mean_fpr, mean_tpr,
             label='Feature selection (AUC = %0.3f ± %0.3f)' % (mean_auc, std_auc),
             lw=1, alpha=.8)
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='black', alpha=.8)
    plt.title('HANN on hemorrhagic stroke', fontsize=14)
    plt.legend(loc="lower right", prop={'size': 12})
    plt.show()
from my_utils import performance_util
import matplotlib.pyplot as plt


if __name__ == '__main__':
    hold_or_test = 'train'

    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(20,20))
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.3)
    plt.subplot(421)
    mean_fpr, mean_tpr, mean_auc, std_auc = performance_util.average_roc_auc('rf', 'all_nf', 'ischemic', hold_or_test)
    plt.plot(mean_fpr, mean_tpr,
             label='Clinical feature set (AUC = %0.3f ± %0.3f)' % (mean_auc, std_auc),
             lw=1, alpha=.8)
    mean_fpr, mean_tpr, mean_auc, std_auc = performance_util.average_roc_auc('rf', 'all', 'ischemic', hold_or_test)
    plt.plot(mean_fpr, mean_tpr,
             label='Whole feature set (AUC = %0.3f ± %0.3f)' % (mean_auc, std_auc),
             lw=1, alpha=.8)
    mean_fpr, mean_tpr, mean_auc, std_auc = performance_util.average_roc_auc('rf', 'fs', 'ischemic', hold_or_test)
    plt.plot(mean_fpr, mean_tpr,
             label='Selected feature set (AUC = %0.3f ± %0.3f)' % (mean_auc, std_auc),
             lw=1, alpha=.8)
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='black', alpha=.8)
    plt.title('Random Forest on Ischemic Stroke', fontsize=14)
    plt.legend(loc="lower right", prop={'size': 12})
    # plt.show()

    # ==
    plt.subplot(422)
    mean_fpr, mean_tpr, mean_auc, std_auc = performance_util.average_roc_auc('rf', 'all_nf', 'hemorrhagic', hold_or_test)
    plt.plot(mean_fpr, mean_tpr,
             label='Clinical feature set (AUC = %0.3f ± %0.3f)' % (mean_auc, std_auc),
             lw=1, alpha=.8)
    mean_fpr, mean_tpr, mean_auc, std_auc = performance_util.average_roc_auc('rf', 'all', 'hemorrhagic', hold_or_test)
    plt.plot(mean_fpr, mean_tpr,
             label='Whole feature set (AUC = %0.3f ± %0.3f)' % (mean_auc, std_auc),
             lw=1, alpha=.8)
    mean_fpr, mean_tpr, mean_auc, std_auc = performance_util.average_roc_auc('rf', 'fs', 'hemorrhagic', hold_or_test)
    plt.plot(mean_fpr, mean_tpr,
             label='Selected feature set (AUC = %0.3f ± %0.3f)' % (mean_auc, std_auc),
             lw=1, alpha=.8)
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='black', alpha=.8)
    plt.title('Random Forest on Hemorrhagic Stroke', fontsize=14)
    plt.legend(loc="lower right", prop={'size': 12})
    # plt.show()

    # ==
    plt.subplot(423)
    mean_fpr, mean_tpr, mean_auc, std_auc = performance_util.average_roc_auc('svm', 'all_nf', 'ischemic', hold_or_test)
    plt.plot(mean_fpr, mean_tpr,
             label='Clinical feature set (AUC = %0.3f ± %0.3f)' % (mean_auc, std_auc),
             lw=1, alpha=.8)
    mean_fpr, mean_tpr, mean_auc, std_auc = performance_util.average_roc_auc('svm', 'all', 'ischemic', hold_or_test)
    plt.plot(mean_fpr, mean_tpr,
             label='Whole feature set (AUC = %0.3f ± %0.3f)' % (mean_auc, std_auc),
             lw=1, alpha=.8)
    mean_fpr, mean_tpr, mean_auc, std_auc = performance_util.average_roc_auc('svm', 'fs', 'ischemic', hold_or_test)
    plt.plot(mean_fpr, mean_tpr,
             label='Selected feature set (AUC = %0.3f ± %0.3f)' % (mean_auc, std_auc),
             lw=1, alpha=.8)
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='black', alpha=.8)
    plt.title('SVM on Ischemic Stroke', fontsize=14)
    plt.legend(loc="lower right", prop={'size': 12})
    # plt.show()

    # ==
    plt.subplot(424)
    mean_fpr, mean_tpr, mean_auc, std_auc = performance_util.average_roc_auc('svm', 'all_nf', 'hemorrhagic', hold_or_test)
    plt.plot(mean_fpr, mean_tpr,
             label='Clinical feature set (AUC = %0.3f ± %0.3f)' % (mean_auc, std_auc),
             lw=1, alpha=.8)
    mean_fpr, mean_tpr, mean_auc, std_auc = performance_util.average_roc_auc('svm', 'all', 'hemorrhagic', hold_or_test)
    plt.plot(mean_fpr, mean_tpr,
             label='Whole feature set (AUC = %0.3f ± %0.3f)' % (mean_auc, std_auc),
             lw=1, alpha=.8)
    mean_fpr, mean_tpr, mean_auc, std_auc = performance_util.average_roc_auc('svm', 'fs', 'hemorrhagic', hold_or_test)
    plt.plot(mean_fpr, mean_tpr,
             label='Selected feature set (AUC = %0.3f ± %0.3f)' % (mean_auc, std_auc),
             lw=1, alpha=.8)
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='black', alpha=.8)
    plt.title('SVM on Hemorrhagic Stroke', fontsize=14)
    plt.legend(loc="lower right", prop={'size': 12})
    # plt.show()

    # ==
    plt.subplot(425)
    mean_fpr, mean_tpr, mean_auc, std_auc = performance_util.average_roc_auc('mlp', 'all_nf', 'ischemic', hold_or_test)
    plt.plot(mean_fpr, mean_tpr,
             label='Clinical feature set (AUC = %0.3f ± %0.3f)' % (mean_auc, std_auc),
             lw=1, alpha=.8)
    mean_fpr, mean_tpr, mean_auc, std_auc = performance_util.average_roc_auc('mlp', 'all', 'ischemic', hold_or_test)
    plt.plot(mean_fpr, mean_tpr,
             label='Whole feature set (AUC = %0.3f ± %0.3f)' % (mean_auc, std_auc),
             lw=1, alpha=.8)
    mean_fpr, mean_tpr, mean_auc, std_auc = performance_util.average_roc_auc('mlp', 'fs', 'ischemic', hold_or_test)
    plt.plot(mean_fpr, mean_tpr,
             label='Selected feature set (AUC = %0.3f ± %0.3f)' % (mean_auc, std_auc),
             lw=1, alpha=.8)
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='black', alpha=.8)
    plt.title('ANN on Ischemic Stroke', fontsize=14)
    plt.legend(loc="lower right", prop={'size': 12})
    # plt.show()

    # ==
    plt.subplot(426)
    mean_fpr, mean_tpr, mean_auc, std_auc = performance_util.average_roc_auc('mlp', 'all_nf', 'hemorrhagic', hold_or_test)
    plt.plot(mean_fpr, mean_tpr,
             label='Clinical feature set (AUC = %0.3f ± %0.3f)' % (mean_auc, std_auc),
             lw=1, alpha=.8)
    mean_fpr, mean_tpr, mean_auc, std_auc = performance_util.average_roc_auc('mlp', 'all', 'hemorrhagic', hold_or_test)
    plt.plot(mean_fpr, mean_tpr,
             label='Whole feature set (AUC = %0.3f ± %0.3f)' % (mean_auc, std_auc),
             lw=1, alpha=.8)
    mean_fpr, mean_tpr, mean_auc, std_auc = performance_util.average_roc_auc('mlp', 'fs', 'hemorrhagic', hold_or_test)
    plt.plot(mean_fpr, mean_tpr,
             label='Selected feature set (AUC = %0.3f ± %0.3f)' % (mean_auc, std_auc),
             lw=1, alpha=.8)
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='black', alpha=.8)
    plt.title('ANN on Hemorrhagic Stroke', fontsize=14)
    plt.legend(loc="lower right", prop={'size': 12})
    # plt.show()

    # ==
    plt.subplot(427)
    mean_fpr, mean_tpr, mean_auc, std_auc = performance_util.average_roc_auc('mlp_cnn', 'all_nf', 'ischemic', hold_or_test)
    plt.plot(mean_fpr, mean_tpr,
             label='Clinical feature set (AUC = %0.3f ± %0.3f)' % (mean_auc, std_auc),
             lw=1, alpha=.8)
    mean_fpr, mean_tpr, mean_auc, std_auc = performance_util.average_roc_auc('mlp_cnn', 'all', 'ischemic', hold_or_test)
    plt.plot(mean_fpr, mean_tpr,
             label='Whole feature set (AUC = %0.3f ± %0.3f)' % (mean_auc, std_auc),
             lw=1, alpha=.8)
    mean_fpr, mean_tpr, mean_auc, std_auc = performance_util.average_roc_auc('mlp_cnn', 'fs', 'ischemic', hold_or_test)
    plt.plot(mean_fpr, mean_tpr,
             label='Selected feature set (AUC = %0.3f ± %0.3f)' % (mean_auc, std_auc),
             lw=1, alpha=.8)
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='black', alpha=.8)
    plt.title('HANN on Ischemic Stroke', fontsize=14)
    plt.legend(loc="lower right", prop={'size': 12})
    # plt.show()

    # ==
    plt.subplot(428)
    mean_fpr, mean_tpr, mean_auc, std_auc = performance_util.average_roc_auc('mlp_cnn', 'all_nf', 'hemorrhagic', hold_or_test)
    plt.plot(mean_fpr, mean_tpr,
             label='Clinical feature set (AUC = %0.3f ± %0.3f)' % (mean_auc, std_auc),
             lw=1, alpha=.8)
    mean_fpr, mean_tpr, mean_auc, std_auc = performance_util.average_roc_auc('mlp_cnn', 'all', 'hemorrhagic', hold_or_test)
    plt.plot(mean_fpr, mean_tpr,
             label='Whole feature set (AUC = %0.3f ± %0.3f)' % (mean_auc, std_auc),
             lw=1, alpha=.8)
    mean_fpr, mean_tpr, mean_auc, std_auc = performance_util.average_roc_auc('mlp_cnn', 'fs', 'hemorrhagic', hold_or_test)
    plt.plot(mean_fpr, mean_tpr,
             label='Selected feature set (AUC = %0.3f ± %0.3f)' % (mean_auc, std_auc),
             lw=1, alpha=.8)
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='black', alpha=.8)
    plt.title('HANN on Hemorrhagic Stroke', fontsize=14)
    plt.legend(loc="lower right", prop={'size': 12})

    fig.savefig("roc.png", dpi=600)
    plt.show()
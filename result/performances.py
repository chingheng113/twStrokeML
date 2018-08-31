import os
import pandas as pd
from my_utils import performance_util
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html
    filepath = 'mlp' + os.sep
    file_name = 'mlp_cnn_2c_fe_predict_result_test_0.csv'
    df = pd.read_csv(filepath+file_name, encoding='utf8')
    label = df['label']
    probas_ = df[['0', '1']].values
    predict = performance_util.labelize(probas_)
    print(confusion_matrix(label, predict))
    print(classification_report(label, predict))

    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(label, probas_[:, 1])
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (0, roc_auc))
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Luck', alpha=.8)

    plt.show()
    print('done')



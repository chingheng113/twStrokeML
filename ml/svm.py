from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
import numpy as np
from my_utils import data_util, plot_fig, performance_util

if __name__ == '__main__':
    # wholeset_Jim_nomissing_validated.csv
    seed = 7
    n_fold = 10
    np.random.seed(seed)
    kfold = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=seed)
    id_data, x_data, y_data = data_util.get_poor_god('wholeset_Jim_nomissing_validated.csv')
    classifier = SVC(kernel='linear')
    for index, (train, test) in enumerate(kfold.split(x_data, y_data)):
        classifier.fit(data_util.scale(x_data.iloc[train]), y_data.iloc[train])
        y_pred = classifier.predict(data_util.scale(x_data.iloc[test]))
        print(confusion_matrix(y_data.iloc[test], y_pred))
        print(classification_report(y_data.iloc[test], y_pred))
        print("SVM-輸出訓練集的準確率為：", classifier.score(data_util.scale(x_data.iloc[train]), y_data.iloc[train]))
        print("SVM-輸出測試集的準確率為：", classifier.score(data_util.scale(x_data.iloc[test]), y_data.iloc[test]))
    # id_data, x_data, y_data = data_util.get_individual('wholeset_Jim_nomissing_validated.csv')
    # classifier = SVC(kernel='linear', decision_function_shape='ovo')
    # for index, (train, test) in enumerate(kfold.split(x_data, y_data)):
    #     classifier.fit(data_util.scale(x_data.iloc[train]), y_data.iloc[train])
    #     y_pred = classifier.predict(data_util.scale(x_data.iloc[test]))
    #     print(confusion_matrix(y_data.iloc[test], y_pred))
    #     print(classification_report(y_data.iloc[test], y_pred))
    #     print("SVM-輸出訓練集的準確率為：", classifier.score(data_util.scale(x_data.iloc[train]), y_data.iloc[train]))
    #     print("SVM-輸出測試集的準確率為：", classifier.score(data_util.scale(x_data.iloc[test]), y_data.iloc[test]))


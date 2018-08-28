from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
import numpy as np
from my_utils import data_util, plot_fig, performance_util
from sklearn.tree import export_graphviz

if __name__ == '__main__':
    # wholeset_Jim_nomissing_validated.csv
    seed = 7
    n_fold = 2
    np.random.seed(seed)
    kfold = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=seed)
    id_data, x_data, y_data = data_util.get_poor_god('wholeset_Jim_nomissing_validated.csv')
    rf = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=7)
    for index, (train, test) in enumerate(kfold.split(x_data, y_data)):
        rf.fit(data_util.scale(x_data.iloc[train]), y_data.iloc[train])
        y_pred = rf.predict(data_util.scale(x_data.iloc[test]))
        print(confusion_matrix(y_data.iloc[test], y_pred))
        print(classification_report(y_data.iloc[test], y_pred))
        print("RF-輸出訓練集的準確率為：", rf.score(data_util.scale(x_data.iloc[train]), y_data.iloc[train]))
        print("RF-輸出測試集的準確率為：", rf.score(data_util.scale(x_data.iloc[test]), y_data.iloc[test]))
    performance_util.save_model(rf, 'rf_2c')




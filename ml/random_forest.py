from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
import numpy as np
from my_utils import data_util, performance_util
from feature_engineering import tsne_extraction as tsne
import os


if __name__ == '__main__':
    seed = 7
    np.random.seed(seed)
    # ******************
    # none = 0, feature selection = 1, feature extraction = 2
    experiment = 2
    n_fold = 10
    save_path = '..' + os.sep + 'result' + os.sep + 'rf' + os.sep
    # ******************
    if experiment == 0:
        id_data, x_data, y_data = data_util.get_poor_god('wholeset_Jim_nomissing_validated.csv')
        model_name = 'rf_2c_normal'
    elif experiment == 1:
        id_data, x_data, y_data = data_util.get_poor_god('wholeset_Jim_nomissing_validated_fs.csv')
        model_name = 'rf_2c_fs'
    else:
        id_data, x_data, y_data = data_util.get_poor_god('wholeset_Jim_nomissing_validated_fs.csv')
        model_name = 'rf_2c_fe'

    kfold = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=seed)
    rf = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=seed)
    for index, (train, test) in enumerate(kfold.split(x_data, y_data)):
        # Training
        x_train = data_util.scale(x_data.iloc[train])
        if experiment == 2:
            x_train = tsne.tsne_features_add(x_train, seed)
        rf.fit(data_util.scale(x_train), y_data.iloc[train])

        # Testing
        x_test = data_util.scale(x_data.iloc[test])
        if experiment == 2:
            x_test = tsne.tsne_features_add(x_test, seed)

        # Evaluation
        predict_result_train = id_data.iloc[train]
        train_probas = rf.predict_proba(x_train)
        predict_result_train['label'] = y_data.iloc[train]
        predict_result_train['0'] = train_probas[:, 0]
        predict_result_train['1'] = train_probas[:, 1]
        predict_result_train.to_csv(save_path + model_name + '_predict_result_train_'+str(index)+'.csv',
                                    sep=',', encoding='utf-8')

        predict_result_test = id_data.iloc[test]
        test_probas = rf.predict_proba(x_test)
        predict_result_test['label'] = y_data.iloc[test]
        predict_result_test['0'] = test_probas[:, 0]
        predict_result_test['1'] = test_probas[:, 1]
        predict_result_test.to_csv(save_path + model_name + '_predict_result_test_'+str(index)+'.csv',
                                   sep=',', encoding='utf-8')

        performance_util.save_model(rf, model_name+'_'+str(index))




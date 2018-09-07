from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score
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
    save_path = '..' + os.sep + 'result' + os.sep + 'svm' + os.sep
    # ******************
    if experiment == 0:
        id_data_all, x_data_all, y_data_all = data_util.get_poor_god('wholeset_Jim_nomissing_validated.csv')
        model_name = 'svm_2c_normal'
    elif experiment == 1:
        id_data_all, x_data_all, y_data_all = data_util.get_poor_god('wholeset_Jim_nomissing_validated_fs.csv')
        model_name = 'svm_2c_fs'
    else:
        id_data_all, x_data_all, y_data_all = data_util.get_poor_god('wholeset_Jim_nomissing_validated_fs.csv')
        model_name = 'svm_2c_fe'

    # --
    id_data, id_data_hold, x_data, x_hold, y_data, y_hold = train_test_split(id_data_all, x_data_all, y_data_all, test_size=0.3, random_state=seed)
    # --

    test_acc_array = []
    kfold = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=seed)
    classifier = SVC(kernel='linear', probability=True, random_state=seed, verbose=True)
    for index, (train, test) in enumerate(kfold.split(x_data, y_data)):
        # Training
        x_train = data_util.scale(x_data.iloc[train])
        if experiment == 2:
            x_train = tsne.tsne_features_add(x_train, seed)
        classifier.fit(x_train, y_data.iloc[train])

        # Testing
        x_test = data_util.scale(x_data.iloc[test])
        if experiment == 2:
            x_test = tsne.tsne_features_add(x_test, seed)

        # Evaluation
        predict_result_train = id_data.iloc[train]
        train_probas = classifier.predict_proba(x_train)
        predict_result_train['label'] = y_data.iloc[train]
        predict_result_train['0'] = train_probas[:, 0]
        predict_result_train['1'] = train_probas[:, 1]
        predict_result_train.to_csv(save_path + model_name + '_predict_result_train_'+str(index)+'.csv',
                                    sep=',', encoding='utf-8')

        predict_result_test = id_data.iloc[test]
        test_probas = classifier.predict_proba(x_test)
        predict_result_test['label'] = y_data.iloc[test]
        predict_result_test['0'] = test_probas[:, 0]
        predict_result_test['1'] = test_probas[:, 1]
        predict_result_test.to_csv(save_path + model_name + '_predict_result_test_'+str(index)+'.csv',
                                   sep=',', encoding='utf-8')
        test_acc = accuracy_score(y_data.iloc[test], classifier.predict(x_test))
        test_acc_array.append(test_acc)
        performance_util.save_model(classifier, model_name+'_'+str(index))
    print('10-CV Done')
    # --
    best_model_inx = test_acc_array.index(max(test_acc_array))
    hold_model = performance_util.load_ml_model(model_name, best_model_inx)
    x_hold = data_util.scale(x_hold)
    if experiment == 2:
        x_hold = tsne.tsne_features_add(x_hold, seed)
    predict_result_hold = id_data_hold
    holdout_probas = hold_model.predict_proba(x_hold)
    predict_result_hold['label'] = y_hold
    predict_result_hold['0'] = holdout_probas[:, 0]
    predict_result_hold['1'] = holdout_probas[:, 1]
    predict_result_hold.to_csv(save_path + model_name + '_predict_result_hold.csv',
                               sep=',', encoding='utf-8')
    print('hold-out Done')

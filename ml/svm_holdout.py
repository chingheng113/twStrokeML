import os
import sys
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import numpy as np
sys.path.append("..")
from my_utils import data_util, performance_util


def do_svm(hold_out_round, sub_class, experiment):
    np.random.seed(hold_out_round)
    if sub_class == 'ischemic':
        id_train_all, x_train_all, y_train_all = data_util.get_poor_god_downsample(
            'training_is_' + str(hold_out_round) + '.csv', sub_class=sub_class)
        id_hold, x_hold, y_hold = data_util.get_poor_god(
            'hold_is_' + str(hold_out_round) + '.csv', sub_class=sub_class)
    else:
        id_train_all, x_train_all, y_train_all = data_util.get_poor_god_downsample(
            'training_he_' + str(hold_out_round) + '.csv', sub_class=sub_class)
        id_hold, x_hold, y_hold = data_util.get_poor_god(
            'hold_he_' + str(hold_out_round) + '.csv', sub_class=sub_class)
    #
    if experiment == 0:
        save_path = '..' + os.sep + 'result' + os.sep + 'svm' + os.sep + 'all' + os.sep
        model_name = 'svm_'+sub_class+'_h_'+str(hold_out_round)
    else:
        x_train_all = data_util.feature_selection(x_train_all, sub_class)
        x_hold = data_util.feature_selection(x_hold, sub_class)
        save_path = '..' + os.sep + 'result' + os.sep + 'svm' + os.sep + 'fs' + os.sep
        model_name = 'svm_fs_'+sub_class+'_h_'+str(hold_out_round)
    #
    test_acc_array = []
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=hold_out_round)
    classifier = SVC(kernel='linear', probability=True, random_state=hold_out_round, verbose=True)
    for index, (train, test) in enumerate(kfold.split(x_train_all, y_train_all)):
        # Training
        x_train = data_util.scale(x_train_all.iloc[train])
        y_train = y_train_all.iloc[train]
        # Testing
        x_test = data_util.scale(x_train_all.iloc[test])
        y_test = y_train_all.iloc[test]
        # train on 90% training
        classifier.fit(x_train, y_train)
        predict_result_train = id_train_all.iloc[train]
        train_probas = classifier.predict_proba(x_train)
        predict_result_train['label'] = y_train
        predict_result_train['0'] = train_probas[:, 0]
        predict_result_train['1'] = train_probas[:, 1]
        predict_result_train.to_csv(save_path + model_name + '_train_cv'+str(index)+'.csv',
                                    sep=',', encoding='utf-8')
        # Evaluation on 10% training
        predict_result_test = id_train_all.iloc[test]
        test_probas = classifier.predict_proba(x_test)
        predict_result_test['label'] = y_test
        predict_result_test['0'] = test_probas[:, 0]
        predict_result_test['1'] = test_probas[:, 1]
        predict_result_test.to_csv(save_path + model_name + '_test_cv'+str(index)+'.csv',
                                   sep=',', encoding='utf-8')
        test_acc = accuracy_score(y_test, classifier.predict(x_test))
        test_acc_array.append(test_acc)
        performance_util.save_model(classifier, model_name+'_'+str(index))
    print('10-CV Done')
    # --
    best_model_inx = test_acc_array.index(max(test_acc_array))
    hold_model = performance_util.load_ml_model(model_name, best_model_inx)
    x_hold = data_util.scale(x_hold)
    predict_result_hold = id_hold
    holdout_probas = hold_model.predict_proba(x_hold)
    predict_result_hold['label'] = y_hold
    predict_result_hold['0'] = holdout_probas[:, 0]
    predict_result_hold['1'] = holdout_probas[:, 1]
    predict_result_hold.to_csv(save_path + model_name + '_hold.csv',
                               sep=',', encoding='utf-8')
    print('hold-out Done')


if __name__ == '__main__':
    hold_out_round = 0
    # ischemic, hemorrhagic
    sub_class = 'ischemic'
    # none = 0, feature selection = 1
    experiment = 1
    #
    # do_svm(hold_out_round, sub_class, experiment)
    do_svm(int(sys.argv[1]), sys.argv[2], int(sys.argv[3]))

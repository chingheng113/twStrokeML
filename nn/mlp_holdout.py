from sklearn.model_selection import StratifiedKFold
from keras.layers import Dense, Activation, Dropout
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras import optimizers
import numpy as np
import os
import sys
sys.path.append("..")
from my_utils import data_util, performance_util


def mlp_binary(x, y, para, indx):
    nb_features = x.shape[1]
    nb_classes = y.shape[1]
    hidden_num = int(round((nb_features+nb_classes)*2/3, 0))
    # model
    # early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=1000, verbose=1, mode='auto')
    filepath = '..'+os.sep+'saved_model'+os.sep+para['model_name']+'_'+str(indx)
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    model = Sequential(name=para['model_name'])
    model.add(Dense(hidden_num, input_dim=nb_features, use_bias=True))
    model.add(Activation('relu'))
    model.add(Dropout(para['drop_rate']))
    model.add(Dense(units=nb_classes))
    model.add(Activation('softmax'))
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.sgd(lr=5e-3),
                  metrics=['accuracy'])
    print(model.summary())
    history = model.fit(x, y,
                        batch_size=para['size_of_batch'],
                        epochs=para['nb_epoch'],
                        validation_split=0.33,
                        shuffle=True,
                        callbacks=[checkpoint])
    return history, model


def do_mlp(hold_out_round, sub_class, experiment):
    np.random.seed(hold_out_round)
    if sub_class == 'ischemic':
        id_train_all, x_train_all, y_train_all = data_util.get_poor_god(
            'training_is_' + str(hold_out_round) + '.csv', sub_class=sub_class)
        id_hold, x_hold, y_hold = data_util.get_poor_god(
            'hold_is_' + str(hold_out_round) + '.csv', sub_class=sub_class)
    else:
        id_train_all, x_train_all, y_train_all = data_util.get_poor_god(
            'training_he_' + str(hold_out_round) + '.csv', sub_class=sub_class)
        id_hold, x_hold, y_hold = data_util.get_poor_god(
            'hold_he_' + str(hold_out_round) + '.csv', sub_class=sub_class)
    #
    if experiment == 0:
        save_path = '..' + os.sep + 'result' + os.sep + 'mlp' + os.sep + 'all' + os.sep
        parameter = {'model_name': 'mlp_'+sub_class+'_h_'+str(hold_out_round),
                     'size_of_batch': 56,
                     'nb_epoch': 150,
                     'drop_rate': 0.4}
    else:
        x_train_all = data_util.feature_selection(x_train_all, sub_class)
        x_hold = data_util.feature_selection(x_hold, sub_class)
        save_path = '..' + os.sep + 'result' + os.sep + 'mlp' + os.sep + 'fs' + os.sep
        parameter = {'model_name': 'mlp_fs_'+sub_class+'_h_'+str(hold_out_round),
                     'size_of_batch': 56,
                     'nb_epoch': 150,
                     'drop_rate': 0.4}


    test_acc_array = []
    test_loss_array = []
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=hold_out_round)
    for index, (train, test) in enumerate(kfold.split(x_train_all, y_train_all)):
        # training
        x_train = data_util.scale(x_train_all.iloc[train])
        y_train = y_train_all.iloc[train]
        # Testing
        x_test = data_util.scale(x_train_all.iloc[test])
        y_test = y_train_all.iloc[test]
        # train on 90% training
        history, model = mlp_binary(x_train, to_categorical(y_train), parameter, index)
        performance_util.save_train_validation(save_path+parameter['model_name'], history, 'acc', str(index))
        predict_result_train = id_train_all.iloc[train]
        train_probas = model.predict(x_train)
        predict_result_train['label'] = y_train
        predict_result_train['0'] = train_probas[:, 0]
        predict_result_train['1'] = train_probas[:, 1]
        predict_result_train.to_csv(save_path + parameter['model_name'] + '_train_cv'+str(index)+'.csv',
                                    sep=',', encoding='utf-8')
        # Evaluation on 10% training
        predict_result_test = id_train_all.iloc[test]
        test_probas = model.predict(x_test)
        predict_result_test['label'] = y_test
        predict_result_test['0'] = test_probas[:, 0]
        predict_result_test['1'] = test_probas[:, 1]
        predict_result_test.to_csv(save_path + parameter['model_name'] + '_test_cv'+str(index)+'.csv',
                                   sep=',', encoding='utf-8')

        loss, acc = model.evaluate(x_test, to_categorical(y_test), verbose=0)
        test_acc_array.append(acc)
        test_loss_array.append(loss)
        # plot_fig.plot_acc_loss(history, 'acc')

    performance_util.save_test(save_path+parameter['model_name'], test_acc_array, test_loss_array)
    print('10-CV Done')
    # --
    best_model_inx = test_acc_array.index(max(test_acc_array))
    hold_model = performance_util.load_nn_model(parameter['model_name'], best_model_inx)
    x_hold = data_util.scale(x_hold)
    predict_result_hold = id_hold
    holdout_probas = hold_model.predict(x_hold)
    predict_result_hold['label'] = y_hold
    predict_result_hold['0'] = holdout_probas[:, 0]
    predict_result_hold['1'] = holdout_probas[:, 1]
    predict_result_hold.to_csv(save_path + parameter['model_name'] + '_hold.csv',
                               sep=',', encoding='utf-8')
    print('hold-out Done')


if __name__ == '__main__':
    hold_out_round = 0
    # ischemic, hemorrhagic
    sub_class = 'ischemic'
    # none = 0, feature selection = 1
    experiment = 1
    #
    # do_mlp(hold_out_round, sub_class, experiment)
    do_mlp(int(sys.argv[1]), sys.argv[2], int(sys.argv[3]))
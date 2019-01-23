from my_utils import data_util, performance_util, plot_fig
from sklearn.model_selection import StratifiedKFold, train_test_split
from keras.layers import Dense, Activation, Dropout, BatchNormalization
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Sequential
from keras import optimizers
from feature_engineering import tsne_extraction as tsne
import numpy as np
import os


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


if __name__ == '__main__':
    seed = 7
    np.random.seed(seed)
    # ******************
    # none = 0, feature selection = 1, feature extraction = 2
    experiment = 1
    n_fold = 10
    # all, ischemic, hemorrhagic . 56,24
    sub_class = 'ischemic'
    save_path = '..' + os.sep + 'result' + os.sep + 'mlp' + os.sep
    # ******************
    if experiment == 0:
        id_data_all, x_data_all, y_data_all = data_util.get_poor_god('wholeset_Jim_nomissing_validated.csv', sub_class)
        parameter = {'model_name': 'mlp_2c_normal_'+sub_class,
                     'size_of_batch': 56,
                     'nb_epoch': 150,
                     'drop_rate': 0.4}
    elif experiment == 1:
        id_data_all, x_data_all, y_data_all = data_util.get_poor_god('wholeset_Jim_nomissing_validated.csv', sub_class)
        x_data_all = data_util.feature_selection(x_data_all, sub_class)
        parameter = {'model_name': 'mlp_2c_fs_'+sub_class,
                     'size_of_batch': 56,
                     'nb_epoch': 150,
                     'drop_rate': 0.4}
    else:
        id_data_all, x_data_all, y_data_all = data_util.get_poor_god('wholeset_Jim_nomissing_validated.csv', sub_class)
        x_data_all = data_util.feature_selection(x_data_all, sub_class)
        parameter = {'model_name': 'mlp_2c_fe_'+sub_class,
                     'size_of_batch': 56,
                     'nb_epoch': 150,
                     'drop_rate': 0.4}

    # --
    id_data, id_data_hold, x_data, x_hold, y_data, y_hold = train_test_split(id_data_all, x_data_all, y_data_all, test_size=0.3, random_state=seed)
    # --

    test_acc_array = []
    test_loss_array = []
    kfold = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=seed)
    for index, (train, test) in enumerate(kfold.split(x_data, y_data)):
        # training
        x_train = data_util.scale(x_data.iloc[train])
        if experiment == 2:
            x_train = tsne.tsne_features_add(x_train, seed)
        history, model = mlp_binary(x_train, to_categorical(y_data.iloc[train]), parameter, index)
        performance_util.save_train_validation(save_path+parameter['model_name'], history, 'acc', str(index))

        # Testing
        x_test = data_util.scale(x_data.iloc[test])
        if experiment == 2:
            x_test = tsne.tsne_features_add(x_test, seed)

        # Evaluation
        predict_result_train = id_data.iloc[train]
        train_probas = model.predict(x_train)
        predict_result_train['label'] = y_data.iloc[train]
        predict_result_train['0'] = train_probas[:, 0]
        predict_result_train['1'] = train_probas[:, 1]
        predict_result_train.to_csv(save_path + parameter['model_name'] + '_predict_result_train_'+str(index)+'.csv',
                                    sep=',', encoding='utf-8')

        predict_result_test = id_data.iloc[test]
        test_probas = model.predict(x_test)
        predict_result_test['label'] = y_data.iloc[test]
        predict_result_test['0'] = test_probas[:, 0]
        predict_result_test['1'] = test_probas[:, 1]
        predict_result_test.to_csv(save_path + parameter['model_name'] + '_predict_result_test_'+str(index)+'.csv',
                                   sep=',', encoding='utf-8')

        loss, acc = model.evaluate(x_test, to_categorical(y_data.iloc[test]), verbose=0)
        test_acc_array.append(acc)
        test_loss_array.append(loss)
        # plot_fig.plot_acc_loss(history, 'acc')

    performance_util.save_test(save_path+parameter['model_name'], test_acc_array, test_loss_array)
    print('10-CV Done')
    # --
    best_model_inx = test_acc_array.index(max(test_acc_array))
    hold_model = performance_util.load_nn_model(parameter['model_name'], best_model_inx)
    x_hold = data_util.scale(x_hold)
    if experiment == 2:
        x_hold = tsne.tsne_features_add(x_hold, seed)
    predict_result_hold = id_data_hold
    holdout_probas = hold_model.predict(x_hold)
    predict_result_hold['label'] = y_hold
    predict_result_hold['0'] = holdout_probas[:, 0]
    predict_result_hold['1'] = holdout_probas[:, 1]
    predict_result_hold.to_csv(save_path + parameter['model_name'] + '_predict_result_hold.csv',
                               sep=',', encoding='utf-8')
    print('hold-out Done')
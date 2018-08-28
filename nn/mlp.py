from my_utils import data_util, plot_fig, performance_util
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from keras.layers import Dense, Activation, Dropout, BatchNormalization
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras import optimizers
import numpy as np


def mlp_multi(x, y, para):
    nb_features = x.shape[1]
    nb_classes = y.shape[1]
    hidden_num = int(round((nb_features+nb_classes)*2/3, 0))
    # model
    early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=1000, verbose=1, mode='auto')
    callbacks_list = [early_stop]
    model = Sequential(name=para['model_name'])
    model.add(Dense(hidden_num, input_dim=nb_features))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(para['drop_rate']))
    model.add(Dense(units=nb_classes))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.sgd(),
                  metrics=['categorical_accuracy'])
    history = model.fit(x, y,
                        batch_size=para['size_of_batch'],
                        epochs=para['nb_epoch'],
                        validation_split=0.33,
                        shuffle=True,
                        callbacks=callbacks_list)
    return history, model


def mlp_binary(x, y, para):
    nb_features = x.shape[1]
    nb_classes = y.shape[1]
    hidden_num = int(round((nb_features+nb_classes)*2/3, 0))
    # model
    early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=1000, verbose=1, mode='auto')
    callbacks_list = [early_stop]
    model = Sequential(name=para['model_name'])
    model.add(Dense(hidden_num, input_dim=nb_features, use_bias=True))
    model.add(Activation('relu'))
    model.add(Dropout(para['drop_rate']))
    model.add(Dense(units=nb_classes))
    model.add(Activation('softmax'))
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.sgd(lr=5e-3, momentum=0.5),
                  metrics=['accuracy'])
    print(model.summary())
    history = model.fit(x, y,
                        batch_size=para['size_of_batch'],
                        epochs=para['nb_epoch'],
                        validation_split=0.33,
                        shuffle=True,
                        callbacks=callbacks_list)
    return history, model


if __name__ == '__main__':
    # wholeset_Jim_nomissing_validated.csv
    seed = 7
    n_fold = 2
    np.random.seed(seed)
    parameter = {'model_name': 'mlp',
            'size_of_batch': 128,
            'nb_epoch': 150,
            'drop_rate': 0.3}
    kfold = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=seed)
    history_array = []
    test_acc_array = []
    test_loss_array = []
    predict_array = []
    # ====== Multi-classes
    # id_data, x_data, y_data = data_util.get_individual('wholeset_Jim_nomissing_validated.csv')
    # prfm_para = {'fn':'mlp_indiv', 'matric':'categorical_accuracy'}
    # for index, (train, test) in enumerate(kfold.split(x_data, y_data)):
    #     history, model = mlp_multi(data_util.scale(x_data.iloc[train]),
    #                                to_categorical(y_data.iloc[train]),
    #                                parameter)
    #     history_array.append(history)
    #     loss, acc = model.evaluate(data_util.scale(x_data.iloc[test]),
    #                                to_categorical(y_data.iloc[test]),
    #                                verbose=0)
    #     test_acc_array.append(acc)
    #     test_loss_array.append(loss)
    # ====== Binary
    id_data, x_data, y_data = data_util.get_poor_god('wholeset_Jim_nomissing_validated.csv')
    prfm_para = {'fn':'mlp_2c', 'matric':'acc'}
    for index, (train, test) in enumerate(kfold.split(x_data, y_data)):
        history, model = mlp_binary(data_util.scale(x_data.iloc[train]),
                                    to_categorical(y_data.iloc[train]),
                                    parameter)
        history_array.append(history)

        loss, acc = model.evaluate(data_util.scale(x_data.iloc[test]),
                                   to_categorical(y_data.iloc[test]),
                                   verbose=0)
        test_acc_array.append(acc)
        test_loss_array.append(loss)

        predict_result = id_data.iloc[test]
        predict_result['true'] = y_data.iloc[test]
        y_pred = performance_util.labelize(model.predict(data_util.scale(x_data.iloc[test])))
        predict_result['predict'] = y_pred
        predict_array.append(predict_result)
        print(confusion_matrix(y_data.iloc[test], y_pred))
        print(classification_report(y_data.iloc[test], y_pred))

    print('===> Test:', np.mean(test_acc_array))
    performance_util.save_performance_all(prfm_para['fn'], history_array, test_acc_array,
                                          test_loss_array, predict_array, prfm_para['matric'])
    plot_fig.plot_acc_loss_all(history_array, prfm_para['matric'])
    print('Done')
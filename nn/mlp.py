from my_utils import data_util, plot_fig, performance_util
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from keras.layers import Dense, Activation, Dropout, BatchNormalization
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.optimizers import sgd
import numpy as np


def mlp_multi(x, y, para):
    nb_features = x.shape[1]
    nb_classes = y.shape[1]
    # model
    early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=100, verbose=1, mode='auto')
    callbacks_list = [early_stop]
    model = Sequential(name=para['model_name'])
    model.add(Dense(50, input_dim=nb_features))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(para['drop_rate']))
    model.add(Dense(units=nb_classes))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd(),
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
    # model
    early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=100, verbose=1, mode='auto')
    callbacks_list = [early_stop]
    model = Sequential(name=para['model_name'])
    model.add(Dense(50, input_dim=nb_features))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(para['drop_rate']))
    model.add(Dense(units=nb_classes))
    model.add(Activation('softmax'))
    model.compile(loss='binary_crossentropy',
                  optimizer=sgd(),
                  metrics=['accuracy'])
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
    n_fold = 10
    np.random.seed(seed)
    parameter = {'model_name': 'mlp',
            'size_of_batch': 500,
            'nb_epoch': 500,
            'drop_rate': 0.5}
    kfold = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=seed)
    history_array = []
    test_acc_array = []
    test_loss_array = []
    # ====== Multi-classes
    # x_data, y_data = data_util.get_individual('wholeset_Jim_nomissing_validated.csv')
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
    x_data, y_data = data_util.get_poor_god('wholeset_Jim_nomissing.csv')
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


    # plot_fig.plot_acc_loss_all(history_array, 'categorical_accuracy')
    performance_util.save_performance_all('mlp_2C', history_array, test_acc_array, test_loss_array, 'acc')
    print('===> Test:', np.mean(test_acc_array))
    print('Done')
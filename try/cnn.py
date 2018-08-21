from my_utils import data_util, plot_fig, performance_util
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from keras.layers import Dense, Activation, Dropout, BatchNormalization, Conv1D, MaxPool1D, Flatten
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras import optimizers
import numpy as np


def cnn_binary(x, y, para):
    x = np.expand_dims(x, 2)
    nb_features = x.shape[1]
    nb_classes = y.shape[1]
    early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=2000, verbose=1, mode='auto')
    callbacks_list = [early_stop]

    model = Sequential()
    model.add(Conv1D(filters=4, kernel_size=2, strides=2, input_shape=(nb_features, 1)))
    model.add(Activation('relu'))
    # model.add(MaxPool1D())
    # model.add(Dropout(para['drop_rate']))
    # layer_Final
    model.add(Flatten())
    model.add(Dense(units=nb_classes))
    model.add(Activation('softmax'))
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.sgd(lr=1e-3, momentum=0.5),
                  metrics=['accuracy'])
    history = model.fit(x, y,
                        batch_size=para['size_of_batch'],
                        epochs=para['nb_epoch'],
                        shuffle=True,
                        validation_split=0.33,
                        callbacks=callbacks_list)
    return history, model


if __name__ == '__main__':
    # wholeset_Jim_nomissing_validated.csv
    seed = 7
    n_fold = 2
    np.random.seed(seed)
    parameter = {'model_name': 'mlp',
            'size_of_batch': 32,
            'nb_epoch': 500,
            'drop_rate': 0.}
    kfold = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=seed)
    history_array = []
    test_acc_array = []
    test_loss_array = []
    predict_array = []
    # ====== Binary
    id_data, x_data, y_data = data_util.get_poor_god('wholeset_Jim_nomissing_validated.csv')
    for index, (train, test) in enumerate(kfold.split(x_data, y_data)):
        x_train_cnn, x_train_mlp = data_util.split_cnn_mlp_input(x_data.iloc[train])

        history, model = cnn_binary(data_util.scale(x_train_cnn),
                                    to_categorical(y_data.iloc[train]),
                                    parameter)
        history_array.append(history)

        x_test_cnn, x_test_mlp = data_util.split_cnn_mlp_input(x_data.iloc[test])
        x_test_cnn = np.expand_dims(data_util.scale(x_test_cnn), 2)
        loss, acc = model.evaluate(x_test_cnn,
                                   to_categorical(y_data.iloc[test]),
                                   verbose=0)
        test_acc_array.append(acc)
        test_loss_array.append(loss)

        y_pred = model.predict(x_test_cnn)
        # predict_array.append(y_pred, y_data.iloc[test])
        print(confusion_matrix(y_data.iloc[test], performance_util.labelize(y_pred)))
        print(classification_report(y_data.iloc[test], performance_util.labelize(y_pred)))

    print('===> Test:', np.mean(test_acc_array))
    # performance_util.save_performance_all('mlp_2C', history_array, test_acc_array, test_loss_array, 'acc')
    plot_fig.plot_acc_loss_all(history_array, 'acc')
    print('Done')
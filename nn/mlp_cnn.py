from my_utils import data_util, plot_fig, performance_util
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from keras.layers import Input, Conv1D, MaxPool1D, Flatten, concatenate, Dense, Activation, Dropout, BatchNormalization, maximum
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.models import Model
from keras import optimizers
import numpy as np


# def mlp_cnn_multi(x_cnn, x_mlp, y, para):
#     cnn_nb_features = x_cnn.shape[1]
#     mlp_nb_features = x_mlp.shape[1]
#     nb_classes = y.shape[1]
#     x_cnn = np.expand_dims(x_cnn, 2)
#     # model
#     early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=500, verbose=1, mode='auto')
#     callbacks_list = [early_stop]
#     # cnn
#     cnn_input = Input(shape=(cnn_nb_features, 1))
#     conv1 = Conv1D(filters=8, kernel_size=2, strides=2, activation='relu')(cnn_input)
#     conv2 = Conv1D(filters=16, kernel_size=2, strides=1, activation='relu')(conv1)
#     flate = Flatten()(conv2)
#     # mlp
#     mlp_input = Input(shape=(mlp_nb_features,))
#     hidden1 = Dense(50, activation='relu')(mlp_input)
#     bn1 = BatchNormalization()(hidden1)
#     hidden2 = Dense(25, activation='relu')(bn1)
#     bn2 = BatchNormalization()(hidden2)
#     act = Activation('relu')(bn2)
#     # merge
#     merge = concatenate([flate, act])
#     dro1 = Dropout(0.3)(merge)
#     hidden_merge1 = Dense(150, activation='relu')(dro1)
#     dro2 = Dropout(0.3)(hidden_merge1)
#     hidden_merge2 = Dense(100, activation='relu')(dro2)
#     dro3 = Dropout(0.3)(hidden_merge2)
#     hidden_merge3 = Dense(50, activation='relu')(dro3)
#     output = Dense(nb_classes, activation='softmax')(hidden_merge3)
#     model = Model(inputs=[cnn_input, mlp_input], outputs=output)
#
#     model.compile(loss='categorical_crossentropy',
#                   optimizer=optimizers.sgd(lr=1e-2),
#                   metrics=['categorical_accuracy'])
#     history = model.fit([x_cnn, x_mlp], y,
#                         batch_size=para['size_of_batch'],
#                         epochs=para['nb_epoch'],
#                         shuffle=True,
#                         validation_split=0.33,
#                         callbacks=callbacks_list)
#     return history, model
def mlp_cnn_binary(x_cnn, x_mlp, y, para):
    cnn_nb_features = x_cnn.shape[1]
    mlp_nb_features = x_mlp.shape[1]
    nb_classes = y.shape[1]
    x_cnn = np.expand_dims(x_cnn, 2)
    # model
    early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=1000, verbose=1, mode='auto')
    callbacks_list = [early_stop]
    # cnn
    cnn_input = Input(shape=(cnn_nb_features, 1))
    conv1 = Conv1D(filters=8, kernel_size=2, strides=2, activation='relu')(cnn_input)
    flate = Flatten()(conv1)
    # mlp
    mlp_input = Input(shape=(mlp_nb_features,))
    hidden1 = Dense(128, activation='relu')(mlp_input)
    # merge
    merge = concatenate([flate, hidden1])
    dro1 = Dropout(0.3)(merge)
    hidden_merge1 = Dense(128, activation='relu')(dro1)
    output = Dense(nb_classes, activation='softmax')(hidden_merge1)

    model = Model(inputs=[cnn_input, mlp_input], outputs=output)
    # print(model.summary())
    plot_fig.plot_model(model, para['model_name'])
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.sgd(lr=1e-3, momentum=0.5),
                  metrics=['accuracy'])
    history = model.fit([x_cnn, x_mlp], y,
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
    parameter = {'model_name': 'mlp_cnn',
                 'size_of_batch': 32,
                 'nb_epoch': 500}
    kfold = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=seed)
    history_array = []
    test_acc_array = []
    test_loss_array = []
    # ====== Multi-classes
    # x_data, y_data = data_util.get_individual('wholeset_Jim_nomissing_validated.csv')
    # for index, (train, test) in enumerate(kfold.split(x_data, y_data)):
    #     x_train_cnn, x_train_mlp = data_util.split_cnn_mlp_input(x_data.iloc[train])
    #     history, model = mlp_cnn_multi(data_util.scale(x_train_cnn),
    #                                    data_util.scale(x_train_mlp),
    #                                    to_categorical(y_data.iloc[train]),
    #                                    parameter)
    #     history_array.append(history)
    #
    #     x_test_cnn, x_test_mlp = data_util.split_cnn_mlp_input(x_data.iloc[test])
    #     x_test_cnn = np.expand_dims(data_util.scale(x_test_cnn), 2)
    #     x_test_mlp = data_util.scale(x_test_mlp)
    #     loss, acc = model.evaluate([x_test_cnn, x_test_mlp],
    #                                to_categorical(y_data.iloc[test]),
    #                                verbose=0)
    #     test_acc_array.append(acc)
    #     test_loss_array.append(loss)
    # ====== Binary
    x_data, y_data = data_util.get_poor_god('wholeset_Jim_nomissing_validated.csv')
    for index, (train, test) in enumerate(kfold.split(x_data, y_data)):
        x_train_cnn, x_train_mlp = data_util.split_cnn_mlp_input(x_data.iloc[train])
        history, model = mlp_cnn_binary(data_util.scale(x_train_cnn),
                                       data_util.scale(x_train_mlp),
                                       to_categorical(y_data.iloc[train]),
                                       parameter)
        history_array.append(history)

        x_test_cnn, x_test_mlp = data_util.split_cnn_mlp_input(x_data.iloc[test])
        x_test_cnn = np.expand_dims(data_util.scale(x_test_cnn), 2)
        x_test_mlp = data_util.scale(x_test_mlp)
        loss, acc = model.evaluate([x_test_cnn, x_test_mlp],
                                   to_categorical(y_data.iloc[test]),
                                   verbose=0)
        test_acc_array.append(acc)
        test_loss_array.append(loss)

    print('===> Test:', np.mean(test_acc_array))
    plot_fig.plot_acc_loss_all(history_array, 'acc')
    performance_util.save_performance_all('mlp_cnn_2C', history_array, test_acc_array, test_loss_array,
                                          'acc')
    print('Done')
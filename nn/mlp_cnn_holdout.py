from my_utils import data_util, plot_fig, performance_util
from sklearn.model_selection import StratifiedKFold
from keras.layers import Input, Conv1D, Flatten, Dense, Activation, Dropout
from keras import layers
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.models import Model
from keras import optimizers
import numpy as np
import os


def mlp_cnn_binary(x_cnn, x_mlp, y, para, indx):
    cnn_nb_features = x_cnn.shape[1]
    mlp_nb_features = x_mlp.shape[1]
    nb_classes = y.shape[1]
    x_cnn = np.expand_dims(x_cnn, 2)
    mlp_hidden_num = int(round((mlp_nb_features+nb_classes)*2/3, 0))
    # model
    # early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=100, verbose=1, mode='auto')
    filepath = '..'+os.sep+'saved_model'+os.sep+para['model_name']+'_'+str(indx)
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    # cnn
    cnn_input = Input(shape=(cnn_nb_features, 1))
    conv1 = Conv1D(filters=10, kernel_size=2, strides=2, activation='relu')(cnn_input)
    flate = Flatten()(conv1)
    h1 = Dense(100, activation='relu')(flate)
    cnn_s = Dense(nb_classes)(h1)
    # mlp
    mlp_input = Input(shape=(mlp_nb_features,))
    hidden1 = Dense(mlp_hidden_num, activation='relu')(mlp_input)
    dr = Dropout(para['drop_rate'])(hidden1)
    mlp_s = Dense(nb_classes)(dr)
    # merge
    merge = layers.add([cnn_s, mlp_s])
    output = Activation('softmax')(merge)
    model = Model(inputs=[cnn_input, mlp_input], outputs=output)
    # print(model.summary())
    plot_fig.plot_model(model, para['model_name'])
    # 5e-3
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.sgd(lr=5e-3),
                  metrics=['accuracy'])
    history = model.fit([x_cnn, x_mlp], y,
                        batch_size=para['size_of_batch'],
                        epochs=para['nb_epoch'],
                        shuffle=True,
                        validation_split=0.33,
                        callbacks=[checkpoint],
                        verbose=0)
    return history, model


def do_mlp_cnn(hold_out_round, sub_class, experiment):
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
        save_path = '..' + os.sep + 'result' + os.sep + 'mlp_cnn' + os.sep + 'all' + os.sep
        parameter = {'model_name': 'mlp_cnn_'+sub_class+'_h_'+str(hold_out_round),
                     'size_of_batch': 56,
                     'nb_epoch': 150,
                     'drop_rate': 0.5}
    else:
        save_path = '..' + os.sep + 'result' + os.sep + 'mlp_cnn' + os.sep + 'fs' + os.sep
        selected_features = data_util.get_selected_feature_name(sub_class)
        parameter = {'model_name': 'mlp_cnn_fs_'+sub_class+'_h_'+str(hold_out_round),
                     'size_of_batch': 56,
                     'nb_epoch': 150,
                     'drop_rate': 0.5}

    test_acc_array = []
    test_loss_array = []
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=hold_out_round)
    for index, (train, test) in enumerate(kfold.split(x_train_all, y_train_all)):
        # training
        x_train_cnn, x_train_mlp = data_util.split_cnn_mlp_input(x_train_all.iloc[train])
        if experiment == 1:
            x_train_cnn, x_train_mlp = data_util.selected_cnn_mlp_input(x_train_cnn, x_train_mlp, selected_features)
        x_train_cnn = data_util.scale(x_train_cnn)
        x_train_mlp = data_util.scale(x_train_mlp)
        y_train = y_train_all.iloc[train]

        # Testing
        x_test_cnn, x_test_mlp = data_util.split_cnn_mlp_input(x_train_all.iloc[test])
        if experiment == 1:
            x_test_cnn, x_test_mlp = data_util.selected_cnn_mlp_input(x_test_cnn, x_test_mlp, selected_features)
        x_test_cnn = np.expand_dims(data_util.scale(x_test_cnn), 2)
        x_test_mlp = data_util.scale(x_test_mlp)
        y_test = y_train_all.iloc[test]

        # train on 90% training
        history, model = mlp_cnn_binary(x_train_cnn,x_train_mlp, to_categorical(y_train), parameter, index)
        performance_util.save_train_validation(save_path+parameter['model_name'], history, 'acc', str(index))
        predict_result_train = id_train_all.iloc[train]
        x_train_cnn = np.expand_dims(x_train_cnn, 2)
        train_probas = model.predict([x_train_cnn, x_train_mlp])
        predict_result_train['label'] = y_train
        predict_result_train['0'] = train_probas[:, 0]
        predict_result_train['1'] = train_probas[:, 1]
        predict_result_train.to_csv(save_path + parameter['model_name'] + '_train_cv'+str(index)+'.csv',
                                    sep=',', encoding='utf-8')
        # Evaluation on 10% training
        predict_result_test = id_train_all.iloc[test]
        test_probas = model.predict([x_test_cnn, x_test_mlp])
        predict_result_test['label'] = y_test
        predict_result_test['0'] = test_probas[:, 0]
        predict_result_test['1'] = test_probas[:, 1]
        predict_result_test.to_csv(save_path + parameter['model_name'] + '_test_cv'+str(index)+'.csv',
                                   sep=',', encoding='utf-8')

        loss, acc = model.evaluate([x_test_cnn, x_test_mlp], to_categorical(y_test), verbose=0)
        test_acc_array.append(acc)
        test_loss_array.append(loss)
        # plot_fig.plot_acc_loss(history, 'acc')
    performance_util.save_test(save_path+parameter['model_name'], test_acc_array, test_loss_array)
    print('10-CV Done')
    # --
    best_model_inx = test_acc_array.index(max(test_acc_array))
    hold_model = performance_util.load_nn_model(parameter['model_name'], best_model_inx)
    x_hold_cnn, x_hold_mlp = data_util.split_cnn_mlp_input(x_hold)
    if experiment == 1:
        x_hold_cnn, x_hold_mlp = data_util.selected_cnn_mlp_input(x_hold_cnn, x_hold_mlp, selected_features)

    x_hold_cnn = np.expand_dims(data_util.scale(x_hold_cnn), 2)
    x_hold_mlp = data_util.scale(x_hold_mlp)
    predict_result_hold = id_hold
    holdout_probas = hold_model.predict([x_hold_cnn, x_hold_mlp])
    predict_result_hold['label'] = y_hold
    predict_result_hold['0'] = holdout_probas[:, 0]
    predict_result_hold['1'] = holdout_probas[:, 1]
    predict_result_hold.to_csv(save_path + parameter['model_name']  + '_hold.csv',
                               sep=',', encoding='utf-8')
    print('hold-out Done')


if __name__ == '__main__':
    hold_out_round = 0
    # ischemic, hemorrhagic
    sub_class = 'ischemic'
    # none = 0, feature selection = 1
    experiment = 1
    #
    do_mlp_cnn(hold_out_round, sub_class, experiment)
from my_utils import data_util, plot_fig, performance_util
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from keras.layers import Input, Conv1D, MaxPool1D, Flatten, concatenate, Dense, Activation, Dropout, BatchNormalization
from keras.layers.core import ActivityRegularization
from keras import layers, regularizers
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping
from feature_engineering import tsne_extraction as tsne
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
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    # cnn
    cnn_input = Input(shape=(cnn_nb_features, 1))
    conv1 = Conv1D(filters=10, kernel_size=2, strides=2, activation='relu')(cnn_input)
    flate = Flatten()(conv1)
    h1 = Dense(50, activation='relu')(flate)
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
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.sgd(lr=5e-3, momentum=0.5),
                  metrics=['accuracy'])
    history = model.fit([x_cnn, x_mlp], y,
                        batch_size=para['size_of_batch'],
                        epochs=para['nb_epoch'],
                        shuffle=True,
                        validation_split=0.33,
                        callbacks=[checkpoint],
                        verbose=0)
    return history, model


if __name__ == '__main__':
    seed = 7
    np.random.seed(seed)
    # ******************
    # none = 0, feature selection = 1, feature extraction = 2
    experiment = 2
    n_fold = 2
    save_path = '..' + os.sep + 'result' + os.sep + 'mlp_cnn' + os.sep
    # ******************
    id_data, x_data, y_data = data_util.get_poor_god('wholeset_Jim_nomissing_validated.csv')
    if experiment == 0:
        parameter = {'model_name': 'mlp_cnn_2c_normal',
                     'size_of_batch': 128,
                     'nb_epoch': 150,
                     'drop_rate': 0.2}
    elif experiment == 1:
        selected_features = data_util.get_selected_feature_name('wholeset_Jim_nomissing_validated_fs.csv')
        parameter = {'model_name': 'mlp_cnn_2c_fs',
                     'size_of_batch': 128,
                     'nb_epoch': 150,
                     'drop_rate': 0.2}
    else:
        selected_features = data_util.get_selected_feature_name('wholeset_Jim_nomissing_validated_fs.csv')
        parameter = {'model_name': 'mlp_cnn_2c_fe',
                     'size_of_batch': 128,
                     'nb_epoch': 150,
                     'drop_rate': 0.2}

    test_acc_array = []
    test_loss_array = []
    kfold = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=seed)


    for index, (train, test) in enumerate(kfold.split(x_data, y_data)):
        # training
        x_train_cnn, x_train_mlp = data_util.split_cnn_mlp_input(x_data.iloc[train])
        if experiment == 1:
            x_train_cnn, x_train_mlp = data_util.selected_cnn_mlp_input(x_train_cnn, x_train_mlp, selected_features)
        elif experiment == 2:
            x_train_cnn, x_train_mlp = data_util.selected_cnn_mlp_input(x_train_cnn, x_train_mlp, selected_features)
            x_train_mlp = tsne.tsne_features_add(x_train_mlp, seed)

        x_train_cnn = data_util.scale(x_train_cnn)
        x_train_mlp = data_util.scale(x_train_mlp)
        history, model = mlp_cnn_binary(x_train_cnn,x_train_mlp, to_categorical(y_data.iloc[train]), parameter, index)
        performance_util.save_train_validation(save_path+parameter['model_name'], history, 'acc', str(index))

        # Testing
        x_test_cnn, x_test_mlp = data_util.split_cnn_mlp_input(x_data.iloc[test])
        if experiment == 1:
            x_test_cnn, x_test_mlp = data_util.selected_cnn_mlp_input(x_test_cnn, x_test_mlp, selected_features)
        elif experiment == 2:
            x_test_cnn, x_test_mlp = data_util.selected_cnn_mlp_input(x_test_cnn, x_test_mlp, selected_features)
            x_test_mlp = tsne.tsne_features_add(x_test_mlp, seed)

        x_test_cnn = data_util.scale(x_test_cnn)
        x_test_cnn = np.expand_dims(data_util.scale(x_test_cnn), 2)
        x_test_mlp = data_util.scale(x_test_mlp)

        # Evaluation
        predict_result_train = id_data.iloc[train]
        x_train_cnn = np.expand_dims(data_util.scale(x_train_cnn), 2)
        train_probas = model.predict([x_train_cnn, x_train_mlp])
        predict_result_train['label'] = y_data.iloc[train]
        predict_result_train['0'] = train_probas[:, 0]
        predict_result_train['1'] = train_probas[:, 1]
        predict_result_train.to_csv(save_path + parameter['model_name'] + '_predict_result_train_'+str(index)+'.csv',
                                    sep=',', encoding='utf-8')

        predict_result_test = id_data.iloc[test]
        test_probas = model.predict([x_test_cnn, x_test_mlp])
        predict_result_test['label'] = y_data.iloc[test]
        predict_result_test['0'] = test_probas[:, 0]
        predict_result_test['1'] = test_probas[:, 1]
        predict_result_test.to_csv(save_path + parameter['model_name'] + '_predict_result_test_'+str(index)+'.csv',
                                   sep=',', encoding='utf-8')



        loss, acc = model.evaluate([x_test_cnn, x_test_mlp], to_categorical(y_data.iloc[test]), verbose=0)
        test_acc_array.append(acc)
        test_loss_array.append(loss)

    performance_util.save_test(save_path+parameter['model_name'], test_acc_array, test_loss_array)
    print('Done')
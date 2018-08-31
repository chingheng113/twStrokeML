from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import pickle
import numpy as np
import pandas as pd
import os


def save_train_validation(path, hist, metrics, index):
    train_acc = []
    train_loss = []
    val_loss = []
    val_acc = []
    train_acc.append(hist.history[metrics])
    train_loss.append(hist.history['val_'+metrics])
    val_acc.append(np.asarray(hist.history['loss']))
    val_loss.append(np.asarray(hist.history['val_loss']))
    np.savetxt(path+'_train_acc_'+index+'.csv', train_acc, delimiter=',')
    np.savetxt(path+'_train_loss_'+index+'.csv', train_loss, delimiter=',')
    np.savetxt(path+'_val_acc_'+index+'.csv', val_loss, delimiter=',')
    np.savetxt(path+'_val_loss_'+index+'.csv', val_acc, delimiter=',')


def save_test(path, test_acc_array, test_loss_array):
    np.savetxt(path + '_test_acc.csv', test_acc_array, delimiter=',')
    np.savetxt(path + '_test_loss.csv', test_loss_array, delimiter=',')


def save_prediction(fn, prediction_array):
    for inx, prediction in enumerate(prediction_array):
        if inx == 0:
            df = prediction
        else:
            df = pd.concat([df, prediction])
        np.savetxt('..' + os.sep + 'result' + os.sep + fn + '_predictions.csv', df, delimiter=',', fmt='%s')


def labelize(y_arr):
    y_label = []
    for y in y_arr:
        y_label = np.append(y_label, np.argmax(y))
    return y_label


def save_model(model, name):
    with open('..'+os.sep+'saved_model'+os.sep+name+'.pickle', 'wb') as f:
        pickle.dump(model, f)


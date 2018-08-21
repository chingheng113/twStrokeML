import numpy as np
import pandas as pd
import os


def save_performance_all(fn, history_array, test_acc_array, test_loss_array, prediction_array, metrics):
    save_train_validation(fn, history_array, metrics)
    save_test(fn, test_acc_array, test_loss_array)
    save_prediction(fn, prediction_array)

def save_train_validation(fn, history_array, metrics):
    train_acc = []
    train_loss = []
    val_loss = []
    val_acc = []
    for hist in history_array:
        train_acc.append(hist.history[metrics])
        train_loss.append(hist.history['val_'+metrics])
        val_acc.append(np.asarray(hist.history['loss']))
        val_loss.append(np.asarray(hist.history['val_loss']))
    np.savetxt('..'+os.sep+'result'+os.sep+fn+'_train_acc.csv', train_acc, delimiter=',')
    np.savetxt('..'+os.sep+'result'+os.sep+fn+'_train_loss.csv', train_loss, delimiter=',')
    np.savetxt('..'+os.sep+'result'+os.sep+fn+'_val_acc.csv', val_loss, delimiter=',')
    np.savetxt('..'+os.sep+'result'+os.sep+fn+'_val_loss.csv', val_acc, delimiter=',')


def save_test(fn, test_acc_array, test_loss_array):
    np.savetxt('..' + os.sep + 'result' + os.sep + fn + '_test_acc.csv', test_acc_array, delimiter=',')
    np.savetxt('..' + os.sep + 'result' + os.sep + fn + '_test_loss.csv', test_loss_array, delimiter=',')


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
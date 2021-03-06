from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_fscore_support
from keras.models import load_model
import pickle
from sklearn.externals import joblib
from scipy import interp
from my_utils import data_util
import numpy as np
import pandas as pd
import os


def save_train_validation(path, hist, metrics, index):
    train_acc = []
    train_loss = []
    val_loss = []
    val_acc = []
    train_acc.append(hist.history[metrics])
    val_acc.append(hist.history['val_'+metrics])
    train_loss.append(np.asarray(hist.history['loss']))
    val_loss.append(np.asarray(hist.history['val_loss']))
    np.savetxt(path+'_train_acc_'+index+'.csv', train_acc, delimiter=',')
    np.savetxt(path+'_train_loss_'+index+'.csv', train_loss, delimiter=',')
    np.savetxt(path+'_val_acc_'+index+'.csv', val_acc, delimiter=',')
    np.savetxt(path+'_val_loss_'+index+'.csv', val_loss, delimiter=',')


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


def load_ml_model(model_name, best_model_inx):
    model_path = data_util.get_model_path(model_name+'_'+str(best_model_inx))+'.pickle'
    return joblib.load(model_path)


def load_nn_model(model_name, best_model_inx):
    model_path = data_util.get_model_path(model_name+'_'+str(best_model_inx))
    return load_model(model_path)


def calculate_holdout_roc_auc(hold_out_round, model_name, status, sub_class):
    filepath = model_name + os.sep + status + os.sep
    if status == 'fs':
        file_name = model_name + '_' + status + '_' + sub_class + '_h_' + str(hold_out_round) + '_hold.csv'
    else:
        file_name = model_name + '_' + sub_class + '_h_' + str(hold_out_round) + '_hold.csv'
    df = pd.read_csv(filepath+file_name, encoding='utf8')
    label = df['label']
    probas_ = df[['0', '1']].values
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(label, probas_[:, 1])
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc


def calculate_holdout_roc_auc_all(hold_out_round, model_name, status):
    filepath = '..'+os.sep+'result_all'+os.sep+model_name + os.sep + status + os.sep
    file_name = model_name+'_2c_'+status+'_predict_result_hold.csv'
    df = pd.read_csv(filepath+file_name, encoding='utf8')
    label = df['label']
    probas_ = df[['0', '1']].values
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(label, probas_[:, 1])
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc


def calculate_roc_auc(hold_out_round, model_name, status, cv, sub_class, inx):
    filepath = model_name + os.sep + status + os.sep
    if status == 'fs':
        if cv == 'test':
            file_name = model_name + '_' + status + '_' + sub_class + '_h_' + str(hold_out_round) + '_test_cv' + str(inx) + '.csv'
        else:
            file_name = model_name + '_' + status + '_' + sub_class + '_h_' + str(hold_out_round) + '_hold.csv'
    elif status == 'all':
        if cv == 'test':
            file_name = model_name + '_' + sub_class + '_h_' + str(hold_out_round) + '_test_cv' + str(inx) + '.csv'
        else:
            file_name = model_name + '_' + sub_class + '_h_' + str(hold_out_round) + '_hold.csv'
    elif status == 'all_nf':
        if cv == 'test':
            file_name = model_name + '_nf_' + sub_class + '_h_' + str(hold_out_round) + '_test_cv' + str(inx) + '.csv'
        else:
            file_name = model_name + '_nf_' + sub_class + '_h_' + str(hold_out_round) + '_hold.csv'
    else:
        if cv == 'test':
            file_name = model_name + '_' + status + '_' + sub_class + '_h_' + str(hold_out_round) + '_test_cv' + str(inx) + '.csv'
        else:
            file_name = model_name + '_' + status + '_' + sub_class + '_h_' + str(hold_out_round) + '_hold.csv'
    df = pd.read_csv(filepath+file_name, encoding='utf8')
    label = df['label']
    probas_ = df[['0', '1']].values
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(label, probas_[:, 1])
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc


def average_roc_auc(model_name, status, sub_class, cv):
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 300)
    for inx in range(0, 10, 1):
        fpr, tpr, roc_auc = calculate_roc_auc(inx, model_name, status, cv, sub_class, inx)
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        aucs.append(roc_auc)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    return mean_fpr, mean_tpr, mean_auc, std_auc


def get_confusion_matrix(model_name, status, sub_class, inx):
    filepath = model_name + os.sep + status + os.sep
    file_name = model_name+'_2c_'+status+'_'+sub_class+'_predict_result_test_'+str(inx)+'.csv'
    df = pd.read_csv(filepath+file_name, encoding='utf8')
    label = df['label']
    probas_ = df[['0', '1']].values
    predict = labelize(probas_)
    return confusion_matrix(label, predict)


def get_sum_confusion_matrix(model_name, status, sub_class):
    sum_array = np.zeros((2, 2))
    for inx in range(0,10,1):
        confusion_matrix = get_confusion_matrix(model_name, status, sub_class, inx)
        sum_array = sum_array + confusion_matrix
    return sum_array


def get_classification_report(model_name, status, sub_class, inx):
    filepath = model_name + os.sep + status + os.sep
    file_name = model_name+'_2c_'+status+'_'+sub_class+'_predict_result_test_'+str(inx)+'.csv'
    df = pd.read_csv(filepath+file_name, encoding='utf8')
    label = df['label']
    probas_ = df[['0', '1']].values
    predict = labelize(probas_)
    return classification_report(label, predict)


def get_average_test_classification_report(model_name, sub_class, status):
    for inx in range(0,10,1):
        filepath = model_name + os.sep + status + os.sep
        if status == 'fs':
            file_name = model_name + '_' + status + '_' + sub_class + '_h_' + str(inx) + '_hold.csv'
        else:
            file_name = model_name + '_' + sub_class + '_h_' + str(inx) + '_hold.csv'
        df = pd.read_csv(filepath+file_name, encoding='utf8')
        label = list(df['label'].values)
        probas_ = df[['0', '1']].values
        predict = list(labelize(probas_))
        if inx == 0:
            labels = label
            predicts = predict
        else:
            labels.extend(label)
            predicts.extend(predict)
    return classification_report(labels, predicts, digits=3)


def get_all_performance_scores(model_name, sub_class, status):
    precisions = []
    recalls = []
    fscores = []
    for inx in range(0,10,1):
        filepath = model_name + os.sep + status + os.sep
        if status == 'fs':
            file_name = model_name + '_' + status + '_' + sub_class + '_h_' + str(inx) + '_hold.csv'
        elif status == 'all':
            file_name = model_name + '_' + sub_class + '_h_' + str(inx) + '_hold.csv'
        elif status == 'all_nf':
            file_name = model_name + '_nf_' + sub_class + '_h_' + str(inx) + '_hold.csv'
        else:
            file_name = model_name + '_fs_nf_' + sub_class + '_h_' + str(inx) + '_hold.csv'

        df = pd.read_csv(filepath+file_name, encoding='utf8')
        label = list(df['label'].values)
        probas_ = df[['0', '1']].values
        predict = list(labelize(probas_))
        precision, recall, fscore, support = precision_recall_fscore_support(label, predict, average='macro')
        precisions.extend([precision])
        recalls.extend([recall])
        fscores.extend([fscore])
    return precisions, recalls, fscores

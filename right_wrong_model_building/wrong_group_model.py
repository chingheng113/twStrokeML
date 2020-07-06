import pandas as pd
import os
from sklearn.model_selection import KFold, train_test_split
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.metrics import roc_curve, auc
import imblearn
import xgboost as xgb # conda install py-xgboost
import lime
import lime.lime_tabular
import warnings
from lime import submodular_pick
import matplotlib.pyplot as plt

data = pd.read_csv(os.path.join('..', 'result', 'all_right_wrong_i.csv'))
data.drop(['change_1m', 'change_3m'], axis=1, inplace=True)
data = data.assign(MRS_TX_3=pd.cut(data['MRS_TX_3'], [-1, 2, 7], labels=[0, 1]))

wrong_data = data[data.ctype == 0]
wrong_data.drop(['ctype'], axis=1, inplace=True)
id_wrong_data = wrong_data[['ICASE_ID', 'IDCASE_ID']]
y_wrong_data = wrong_data[['MRS_TX_3']]
x_wrong_data = wrong_data.drop(['ICASE_ID', 'IDCASE_ID', 'MRS_TX_3'], axis=1)


aucs = []
for i in range(1):
    X_train, X_test, y_train, y_test = train_test_split(x_wrong_data, y_wrong_data, test_size=0.3, random_state=i, stratify=y_wrong_data)
# for train_index, test_index in KFold(n_splits=10).split(x_wrong_data):
#     X_train, X_test = x_wrong_data.iloc[train_index], x_wrong_data.iloc[test_index]
#     y_train, y_test = y_wrong_data.iloc[train_index], y_wrong_data.iloc[test_index]
    id_test = id_wrong_data.loc[X_test.index]
    # scaling
    scaler = preprocessing.MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    # over-sampling
    print('before', y_train.groupby(['MRS_TX_3']).size())
    sm = imblearn.over_sampling.SMOTE(random_state=42)
    X_train, y_train = sm.fit_resample(X_train, y_train)
    print('after', y_train.groupby(['MRS_TX_3']).size())
    # define the model
    # model = ExtraTreesClassifier(n_estimators=250,  random_state=0)
    model = xgb.XGBClassifier(objective='binary:logistic', learning_rate=0.5)
    # model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train.values.ravel())
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    result = pd.concat([id_test, y_test], axis=1)
    result['predict_0'] = y_pred_proba[:,0]
    result['predict_1'] = y_pred_proba[:, 1]
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    aucs.append(roc_auc_score(y_test, y_pred_proba[:, 1]))

    explainer = lime.lime_tabular.LimeTabularExplainer(X_train,
                                                       mode='classification', training_labels=y_train,
                                                       feature_names=x_wrong_data.columns.values,
                                                       class_names = ['0', '1'],
                                                       random_state=369)
    # one instance
    # idx = 12
    # exp = explainer.explain_instance(X_test[idx], model.predict_proba, num_features=5)
    # print('id: %d' % idx)
    # # xgboost(like sklearn) expects X as 2D data(n_samples, n_features).If you want to predict only one sample
    # # you can reshape your feature vector to a 2D array
    # print('Probability (1) =', model.predict_proba(np.array(X_test[idx]).reshape((1, -1)))[0, 1])
    # print('True class: %s' % y_test.values[idx])
    # print('Explanation for class right')
    # print('\n'.join(map(str, exp.as_list())))

    # Submodular Pick
    sp_obj = submodular_pick.SubmodularPick(explainer, X_train, model.predict_proba,
                                            sample_size=50,
                                            num_features=5,
                                            num_exps_desired=1)
    for exp in sp_obj.sp_explanations:
        # https://stackoverflow.com/questions/60914598/keyerror-1-in-using-sp-lime-with-lightgbm
        exp.as_pyplot_figure(label=exp.available_labels()[0])
        plt.show()
        print(exp.as_list(label=exp.available_labels()[0]))

    W = pd.DataFrame([dict(this.as_list(label=this.available_labels()[0])) for this in sp_obj.explanations])
    # W.to_csv('see2.csv', index=False)
print(np.mean(aucs), np.std(aucs))
print('done')
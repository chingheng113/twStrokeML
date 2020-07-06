import pandas as pd
import numpy as np
import os
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import imblearn
import xgboost as xgb # conda install py-xgboost
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
from lime import submodular_pick
from sklearn.ensemble import ExtraTreesClassifier


data = pd.read_csv(os.path.join('..', 'result', 'all_right_wrong_i.csv'))
data = data.drop_duplicates(['ICASE_ID', 'IDCASE_ID'])

y_data = data[['ctype']]
print(y_data.shape)
data2 = pd.read_csv(os.path.join('..', 'data_source', 'TSR_2018_3m_noMissing_validated.csv'))
X_data = pd.merge(data[['ICASE_ID', 'IDCASE_ID']], data2, how='inner', on=['ICASE_ID', 'IDCASE_ID'])
X_data = X_data.drop_duplicates(['ICASE_ID', 'IDCASE_ID'])
X_data = X_data.drop(['ICASE_ID', 'IDCASE_ID', 'MRS_TX_1', 'MRS_TX_3'], axis=1)
print(X_data.shape)


for i in range(1):
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3, random_state=i, stratify=y_data)
    # id_test = id_data.loc[X_test.index]
    # scaling
    scaler = preprocessing.MinMaxScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), index=X_train.index, columns=X_train.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X_test.columns)
    # if no scaling, do dataframe to ndarray for lime function
    # X_train = X_train.to_numpy()
    # X_test = X_test.to_numpy()

    # over-sampling
    print('before', y_train.groupby(['ctype']).size())
    sm = imblearn.under_sampling.RandomUnderSampler(random_state=888)
    # sm = imblearn.combine.SMOTETomek(random_state=369)
    X_train, y_train = sm.fit_resample(X_train, y_train)
    print('after', y_train.groupby(['ctype']).size())

    # feature selection
    feature_names = X_train.columns.values
    forest = ExtraTreesClassifier()
    forest.fit(X_train, y_train)
    importances = forest.feature_importances_
    cutoff = np.std(importances[importances > 0.]) + np.min(importances[importances > 0.])
    indices = np.argsort(importances)[::-1]
    # Print the feature ranking
    selected_feature = []
    print("Feature ranking:")
    for i in range(X_train.shape[1]):
        if importances[indices[i]] > cutoff:
            print("%d. feature %d (%f) %s" % (i + 1, indices[i], importances[indices[i]], feature_names[indices[i]]))
            selected_feature.append(feature_names[indices[i]])

    # To avoid feature mismatching error
    # you must convert dataframe to numpy before xgboost training while using lime
    X_train = X_train[selected_feature]
    X_train = X_train.to_numpy()
    X_test = X_test[selected_feature]
    X_test = X_test.to_numpy()
    print(X_train.shape)
    print(X_test.shape)
    # model training
    # model = ExtraTreesClassifier(n_estimators=250, random_state=0)
    model = xgb.XGBClassifier(objective='binary:logistic', learning_rate=0.5)
    # model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    print(roc_auc_score(y_test, y_pred_proba[:, 1]))
    # print(model.intercept_)
    # print(model.coef_)

    explainer = lime.lime_tabular.LimeTabularExplainer(X_train,
                                                       mode='classification', training_labels=y_train,
                                                       feature_names=selected_feature, random_state=369)

    # idx = 12 #9
    # exp = explainer.explain_instance(X_test[idx], model.predict_proba, num_features=5)
    # print('id: %d' % idx)
    # # xgboost(like sklearn) expects X as 2D data(n_samples, n_features).If you want to predict only one sample
    # # you can reshape your feature vector to a 2D array
    # print('Probability (right case) =', model.predict_proba(np.array(X_test[idx]).reshape((1, -1)))[0, 1])
    # print('True class: %s' % y_test.values[idx])
    # print('Explanation for class right')
    # print('\n'.join(map(str, exp.as_list())))
    # fig = exp.as_pyplot_figure()
    # plt.show()

    # Submodular Pick
    sp_obj = submodular_pick.SubmodularPick(explainer, X_train, model.predict_proba,
                                            sample_size=50,
                                            num_features=10,
                                            num_exps_desired=6)
    for exp in sp_obj.sp_explanations:
        # https://stackoverflow.com/questions/60914598/keyerror-1-in-using-sp-lime-with-lightgbm
        exp.as_pyplot_figure(label=exp.available_labels()[0])
        plt.show()
        print(exp.as_list(label=exp.available_labels()[0]))
    print('done')
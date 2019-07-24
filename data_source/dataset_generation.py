from my_utils import data_util
import pandas as pd
from data_source import selected_variables as sv
from sklearn.model_selection import StratifiedKFold, train_test_split


def get_ischemic(df):
    return df[(df['ICD_ID_1.0'] == 1) | (df['ICD_ID_2.0'] == 1)]


def get_hemorrhagic(df):
    return df[(df['ICD_ID_3.0'] == 1) | (df['ICD_ID_4.0'] == 1)]


def make_dummy(df, category_features):
    for fe in category_features:
        dummies = pd.get_dummies(df[fe], prefix=fe)
        for i, dummy in enumerate(dummies):
            df.insert(loc=df.columns.get_loc(fe)+i+1, column=dummy, value=dummies[dummy].values)
    df.drop(category_features, axis=1, inplace=True)
    return df


if __name__ == '__main__':
    seed = range(1, 11, 1)
    print(seed)
    df_all = data_util.load_all('TSR_2018_3m_noMissing_validated.csv')
    dummy_cols = sv.mcase_ca+sv.dcase_info_ca+sv.dcase_icd_ca+['OFFDT_ID']+sv.drfur_ca
    df_all = make_dummy(df_all, dummy_cols)
    df_is = get_ischemic(df_all)
    df_he = get_hemorrhagic(df_all)
    for i in range(0, 10):
        df_train_is, df_hold_is = train_test_split(df_is, test_size=0.3, random_state=seed[i], stratify=df_is['MRS_TX_3'])
        data_util.save_dataframe_to_csv(df_train_is, 'training_is_'+str(i))
        data_util.save_dataframe_to_csv(df_hold_is, 'hold_is_' + str(i))

        df_train_he, df_hold_he = train_test_split(df_he, test_size=0.3, random_state=seed[i], stratify=df_he['MRS_TX_3'])
        data_util.save_dataframe_to_csv(df_train_he, 'training_he_'+str(i))
        data_util.save_dataframe_to_csv(df_hold_he, 'hold_he_' + str(i))
    print('Done')

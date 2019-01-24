from my_utils import data_util
from sklearn.model_selection import StratifiedKFold, train_test_split


def get_ischemic(df):
    return df[(df['ICD_ID_1.0'] == 1) | (df['ICD_ID_2.0'] == 1)]


def get_hemorrhagic(df):
    return df[(df['ICD_ID_3.0'] == 1) | (df['ICD_ID_4.0'] == 1)]


if __name__ == '__main__':
    seed = range(1,11,1)
    print(seed)
    df_all = data_util.load_all('TSR_2018_3m_noMissing_validated.csv')
    df_is= get_ischemic(df_all)
    df_he = get_hemorrhagic(df_all)
    for i in range(0, 10):
        df_train_is, df_hold_is = train_test_split(df_is, test_size=0.3, random_state=seed[i], stratify=df_is['MRS_3'])
        data_util.save_dataframe_to_csv(df_train_is, 'training_is_'+str(i))
        data_util.save_dataframe_to_csv(df_hold_is, 'hold_is_' + str(i))

        df_train_he, df_hold_he = train_test_split(df_he, test_size=0.3, random_state=seed[i], stratify=df_he['MRS_3'])
        data_util.save_dataframe_to_csv(df_train_he, 'training_he_'+str(i))
        data_util.save_dataframe_to_csv(df_hold_he, 'hold_he_' + str(i))
    print('Done')

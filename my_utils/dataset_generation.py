from my_utils import data_util
from sklearn.model_selection import StratifiedKFold, train_test_split


if __name__ == '__main__':
    seed = range(1,11,1)
    print(seed)
    df_all = data_util.load_all('TSR_2018_3m_noMissing_validated.csv')
    for i in range(0, 10):
        df_train, df_hold = train_test_split(df_all, test_size=0.3, random_state=seed[0])
        data_util.save_dataframe_to_csv(df_train, 'training_'+str(i))
        data_util.save_dataframe_to_csv(df_hold, 'hold_' + str(i))
    print(df_train.shape)
    print(df_hold.shape)
    print(df_all.shape)
    print('Done')
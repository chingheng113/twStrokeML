from sklearn import preprocessing as sp
from sklearn.utils import resample
from sklearn.feature_selection import VarianceThreshold
import pandas as pd
import numpy as np
import os


def get_model_path(file_name):
    dirname = os.path.dirname(__file__)
    filepath = os.path.join(dirname, '..' + os.sep + 'saved_model' + os.sep)
    return os.path.join(filepath + file_name)


def get_file_path(file_name):
    dirname = os.path.dirname(__file__)
    filepath = os.path.join(dirname, '..' + os.sep + 'data_source' + os.sep)
    return os.path.join(filepath + file_name)


def load_all(fn):
    read_file_path = get_file_path(fn)
    df = pd.read_csv(read_file_path, encoding='utf8')
    # df = df.ix[:50]
    return df.sample(frac=1)


def get_x_y_data(df):
    id_data = df[['ICASE_ID', 'IDCASE_ID']]
    y_data = df[['MRS_TX_3']]
    x_data = df.drop(['ICASE_ID', 'IDCASE_ID', 'MRS_TX_3'], axis=1)
    return id_data, x_data, y_data


def get_individual(fn):
    df = load_all(fn)
    id_data, x_data, y_data = get_x_y_data(df)
    return id_data, x_data, y_data


def get_ischemic(fn):
    df = load_all(fn)
    return df[(df['ICD_ID_1.0'] == 1) | (df['ICD_ID_2.0'] == 1)]


def get_hemorrhagic(fn):
    df = load_all(fn)
    return df[(df['ICD_ID_3.0'] == 1) | (df['ICD_ID_4.0'] == 1)]


def get_binarized_label(df):
    # Good <= 2, Poor >= 3
    start = min(df['MRS_TX_3'].values) - 1.0
    end = max(df['MRS_TX_3'].values) + 1.0
    y_data = pd.cut(df['MRS_TX_3'], [start, 3, end], labels=[0, 1], right=False)
    return y_data


def get_poor_god(fn, sub_class='all'):
    df = pd.DataFrame()
    if sub_class == 'ischemic':
        df = get_ischemic(fn)
    elif sub_class == 'hemorrhagic':
        df = get_hemorrhagic(fn)
    else:
        df = load_all(fn)
    id_data, x_data, y_data = get_x_y_data(df)
    # Good <= 2, Poor >= 3
    start = min(df['MRS_TX_3'].values) - 1.0
    end = max(df['MRS_TX_3'].values) + 1.0
    y_data = pd.cut(df['MRS_TX_3'], [start, 3, end], labels=[0, 1], right=False)
    return id_data, x_data, y_data


def get_poor_god_downsample(fn, sub_class='all'):
    df = pd.DataFrame()
    if sub_class == 'ischemic':
        df = get_ischemic(fn)
    elif sub_class == 'hemorrhagic':
        df = get_hemorrhagic(fn)
    else:
        df = load_all(fn)
    id_data, x_data, y_data = get_x_y_data(df)
    # Good <= 2, Poor >= 3
    start = min(df['MRS_TX_3'].values) - 1.0
    end = max(df['MRS_TX_3'].values) + 1.0
    y_data = pd.cut(df['MRS_TX_3'], [start, 3, end], labels=[0, 1], right=False)
    if y_data[y_data == 0].size > y_data[y_data == 1].size:
        resample_size = y_data[y_data == 1].size
        df_majority = y_data[y_data == 0]
        df_minority = y_data[y_data == 1]
    else:
        resample_size = y_data[y_data == 0].size
        df_majority = y_data[y_data == 1]
        df_minority = y_data[y_data == 0]
    df_majority_downsampled = resample(df_majority,
                                   replace=False,    # sample without replacement
                                   n_samples=resample_size,     # to match minority class
                                   random_state=7) # reproducible results
    resample_inx = pd.concat([df_majority_downsampled, df_minority], axis=0).sample(frac=1)
    resampled_id_data = id_data.loc[resample_inx.index]
    resampled_x_data = x_data.loc[resample_inx.index]
    resampled_y_data = y_data.loc[resample_inx.index]
    return resampled_id_data, resampled_x_data, resampled_y_data


def variance_threshold_selector(data, threshold=0.5):
    selector = VarianceThreshold(threshold)
    selector.fit(data)
    return data[data.columns[selector.get_support(indices=True)]]


def get_selected_feature_name(sub_type):
    dirname = os.path.dirname(__file__)
    filepath = os.path.join(dirname, '..' + os.sep + 'feature_engineering' + os.sep)
    df = pd.read_csv(filepath+os.sep+'selected_features_'+sub_type+'.csv', encoding='utf8')
    feature_names = df['f_name']
    return feature_names.values


def split_cnn_mlp_input(x_data):
    # cnn_col = ["discharged_mrs", "MRS_1",
    #            "NIHS_1a_in", "NIHS_1a_out", "NIHS_1b_in", "NIHS_1b_out", "NIHS_1c_in", "NIHS_1c_out",
    #            "NIHS_2_in", "NIHS_2_out", "NIHS_3_in", "NIHS_3_out", "NIHS_4_in", "NIHS_4_out",
    #            "NIHS_5aL_in", "NIHS_5aL_out", "NIHS_5bR_in", "NIHS_5bR_out",
    #            "NIHS_6aL_in", "NIHS_6aL_out", "NIHS_6bR_in", "NIHS_6bR_out", "NIHS_7_in", "NIHS_7_out",
    #            "NIHS_8_in", "NIHS_8_out", "NIHS_9_in", "NIHS_9_out", "NIHS_10_in", "NIHS_10_out",
    #            "NIHS_11_in", "NIHS_11_out",
    #            'OMAS_FL', 'AMAS_FL', 'OMAG_FL', 'AMAG_FL', 'OMTI_FL', 'AMTI_FL', 'OMCL_FL', 'AMCL_FL', 'OMWA_FL',
    #            'OMPL_FL', 'AMPL_FL', 'OMANH_FL', 'AMWA_FL', 'AMANH_FL', 'OMAND_FL', 'AMAND_FL', 'OMLI_FL', 'AMLI_FL']
    cnn_col = ["discharged_mrs", "MRS_TX_1",
           "NIHS_1a_in", "NIHS_1a_out", "NIHS_1b_in", "NIHS_1b_out", "NIHS_1c_in", "NIHS_1c_out",
           "NIHS_2_in", "NIHS_2_out", "NIHS_3_in", "NIHS_3_out", "NIHS_4_in", "NIHS_4_out",
           "NIHS_5aL_in", "NIHS_5aL_out", "NIHS_5bR_in", "NIHS_5bR_out",
           "NIHS_6aL_in", "NIHS_6aL_out", "NIHS_6bR_in", "NIHS_6bR_out", "NIHS_7_in", "NIHS_7_out",
           "NIHS_8_in", "NIHS_8_out", "NIHS_9_in", "NIHS_9_out", "NIHS_10_in", "NIHS_10_out",
           "NIHS_11_in", "NIHS_11_out"]

    x_cnn = x_data[cnn_col]
    x_mlp = x_data.drop(cnn_col, axis=1)
    return x_cnn, x_mlp


def selected_cnn_mlp_input(x_cnn, x_mlp, selected_features):
    cnn_features = x_cnn.columns.values
    diff_feature = np.setdiff1d(selected_features, cnn_features)
    x_mlp = x_mlp[diff_feature]
    return x_cnn, x_mlp


def feature_selection(df, sub_type):
    selected_features = get_selected_feature_name(sub_type).ravel()
    return df[selected_features]


def scale(x_data):
    # x_data = np.round(sp.MinMaxScaler(feature_range=(0, 1)).fit_transform(x_data), 3)
    x_data = np.round(sp.StandardScaler().fit_transform(x_data), 3)
    return x_data


def save_dataframe_to_csv(df, file_name):
    dirname = os.path.dirname(__file__)
    filepath = os.path.join(dirname, '..'+os.sep+'data_source'+os.sep)
    df.to_csv(filepath + file_name + '.csv', sep=',', encoding='utf-8', index=False)


def save_np_array_to_csv(array, file_name):
    dirname = os.path.dirname(__file__)
    filepath = os.path.join(dirname, '..'+os.sep+'data_source'+os.sep)
    np.savetxt(filepath + file_name + '.csv', array, delimiter=',', fmt='%s')


if __name__ == '__main__':
    resampled_id_data, resampled_x_data, resampled_y_data = get_poor_god('wholeset_Jim_nomissing_validated.csv', 'all')
    df_all = pd.concat([resampled_id_data, resampled_x_data, resampled_y_data], axis=1)
    save_dataframe_to_csv(df_all, 'str_all_good_poor')
    print(df_all.shape)

from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import os

cnn_col = ["MRS_1", "discharged_mrs",
           "NIHS_1a_in", "NIHS_1a_out", "NIHS_1b_in", "NIHS_1b_out", "NIHS_1c_in", "NIHS_1c_out",
           "NIHS_2_in", "NIHS_2_out", "NIHS_3_in", "NIHS_3_out", "NIHS_4_in", "NIHS_4_out", "NIHS_5aL_in", "NIHS_5aL_out",
           "NIHS_5bR_in", "NIHS_5bR_out", "NIHS_6aL_in", "NIHS_6aL_out", "NIHS_6bR_in", "NIHS_6bR_out",
           "NIHS_7_in", "NIHS_7_out", "NIHS_8_in", "NIHS_8_out", "NIHS_9_in", "NIHS_9_out", "NIHS_10_in", "NIHS_10_out",
           "NIHS_11_in", "NIHS_11_out"]


def get_file_path(file_name):
    dirname = os.path.dirname(__file__)
    filepath = os.path.join(dirname, '..' + os.sep + 'data_source' + os.sep)
    return os.path.join(filepath + file_name)


def load_all(fn):
    read_file_path = get_file_path(fn)
    df = pd.read_csv(read_file_path, encoding='utf8')
    return df.sample(frac=1)


def get_x_y_data(df):
    y_data = df[['MRS_3']]
    x_data = df.drop(['ICASE_ID', 'IDCASE_ID', 'MRS_3'], axis=1)
    return x_data, y_data


def get_individual(fn):
    df = load_all(fn)
    x_data, y_data = get_x_y_data(df)
    return x_data, y_data


def get_poor_god(fn):
    df = load_all(fn)
    x_data, y_data = get_x_y_data(df)
    # Good < 3, Poor >= 3
    start = min(y_data.values)[0] - 1.0
    end = max(y_data.values)[0] + 1.0
    y_data = pd.cut(y_data['MRS_3'], [start, 3, end], labels=[0, 1], right=False)
    return x_data, y_data


def split_cnn_mlp_input(x_data):
    cnn_col = ["discharged_mrs", "MRS_1",
               "NIHS_1a_in", "NIHS_1a_out", "NIHS_1b_in", "NIHS_1b_out", "NIHS_1c_in", "NIHS_1c_out",
               "NIHS_2_in", "NIHS_2_out", "NIHS_3_in", "NIHS_3_out", "NIHS_4_in", "NIHS_4_out",
               "NIHS_5aL_in", "NIHS_5aL_out", "NIHS_5bR_in", "NIHS_5bR_out",
               "NIHS_6aL_in", "NIHS_6aL_out", "NIHS_6bR_in", "NIHS_6bR_out", "NIHS_7_in", "NIHS_7_out",
               "NIHS_8_in", "NIHS_8_out", "NIHS_9_in", "NIHS_9_out", "NIHS_10_in", "NIHS_10_out",
               "NIHS_11_in", "NIHS_11_out"]
    x_cnn = x_data[cnn_col]
    x_mlp = x_data.drop(cnn_col, axis=1)
    return x_cnn, x_mlp

def scale(x_data):
    min_max_scaler = MinMaxScaler()
    x_data = min_max_scaler.fit_transform(x_data)
    return x_data
import pandas as pd
import gc
import numpy as np

debug = True

dtypes = {
    'ip': 'uint32',
    'app': 'uint16',
    'device': 'uint16',
    'os': 'uint16',
    'channel': 'uint16',
    'is_attributed': 'uint8',
    'click_id': 'uint32',
}

train_path = '../data/train_sample.csv'
test_path = '../data/test.csv'
result_path = '../data/result.csv'

skip_rows = 109903890
load_rows = 40000000
if debug is True:
    val_size = 2500
else:
    val_size = 2500000

def load_data(train_path,test_path, skip_rows, load_rows, dtypes):

    if debug is True:
        print("加载训练数据.....")
        train_data = pd.read_csv(train_path,
                                 dtype=dtypes,
                                 usecols=['ip', 'app', 'device', 'os', 'channel', 'click_time', 'is_attributed'],
                                 parse_dates=['click_time'])

        print("加载测试数据.....")
        test_data = pd.read_csv(test_path,
                                dtype=dtypes,
                                usecols=['click_id', 'ip', 'app', 'device', 'os', 'channel', 'click_time'],
                                parse_dates=['click_time'],
                                nrows=10000)
    else:
        print("加载训练数据.....")
        train_data = pd.read_csv(train_path,
                                 dtype=dtypes,
                                 usecols=['ip', 'app', 'device', 'os', 'channel', 'click_time', 'is_attributed'],
                                 parse_dates=['click_time'],
                                 skiprows=range(1, skip_rows),
                                 nrows=load_rows)
        print("加载测试数据.....")
        test_data = pd.read_csv(test_path,
                                 dtype=dtypes,
                                 usecols=['click_id','ip','app','device','os','channel','click_time'],
                                 parse_dates=['click_time'])


    print("数据加载完成.....")
    return train_data,test_data


def feature_deal(train_data,test_data):

    len_train = len(train_data)
    print("训练集大小%d"%(len_train))

    print("合并测试数据和训练数据.....")
    train_data = train_data.append(test_data)
    del test_data
    gc.collect()

    # 特征工程 stage = 1
    print("特征工程 stage = 1 ......")
    train_data['day'] = pd.to_datetime(train_data['click_time']).dt.day.astype('uint8')
    train_data['hour'] = pd.to_datetime(train_data['click_time']).dt.hour.astype('uint8')
    print(train_data.columns)

    print("nunique: ip_channel.....")
    tmp_data = train_data[['ip', 'channel']].groupby(['ip'])['channel'].nunique().reset_index().rename(columns={'channel': 'X0'})
    train_data = train_data.merge(tmp_data, on=['ip'], how='left')
    del tmp_data
    gc.collect()
    print(train_data.columns)


    print("nunique: ip_device_os_app.....")
    tmp_data = train_data[['ip', 'device','os','app']].groupby(['ip', 'device', 'os'])['app'].nunique().reset_index().rename(columns={'app':'X1'})
    train_data = train_data.merge(tmp_data, on=['ip', 'device', 'os'], how='left')
    del tmp_data
    gc.collect()
    print(train_data.columns)

    print("nunique: ip_day_os_hour.....")
    tmp_data = train_data[['ip', 'day', 'hour']].groupby(['ip', 'day'])['hour'].nunique().reset_index().rename(columns={'hour':'X2'})
    train_data = train_data.merge(tmp_data,on=['ip', 'day'],how='left')
    del tmp_data
    gc.collect()
    print(train_data.columns)

    print("nunique: ip_app.....")
    tmp_data = train_data[['ip', 'app']].groupby(['ip'])['app'].nunique().reset_index().rename(columns={'app':'X3'})
    train_data = train_data.merge(tmp_data,on=['ip'],how='left')
    del tmp_data
    gc.collect()
    print(train_data.columns)

    print("nunique: ip_device.....")
    tmp_data = train_data[['ip', 'device']].groupby(['ip'])['device'].nunique().reset_index().rename(columns={'device':'X4'})
    train_data = train_data.merge(tmp_data, on=['ip'], how='left')
    del tmp_data
    gc.collect()

    print("nunique: ip_app_os.....")
    tmp_data = train_data[['ip', 'app', 'os']].groupby(['ip', 'app'])['os'].nunique().reset_index().rename(columns={'os':'X5'})
    train_data = train_data.merge(tmp_data, on=['ip', 'app'], how='left')
    del tmp_data
    gc.collect()
    print(train_data.columns)

    print("nunique: app_channel.....")
    tmp_data = train_data[['app', 'channel']].groupby(['app'])['channel'].nunique().reset_index().rename(columns={'channel':'X6'})
    train_data = train_data.merge(tmp_data, on=['app'], how='left')
    del tmp_data
    gc.collect()
    print(train_data.columns)

    # 特征工程 stage = 2
    print("特征工程 stage = 2 ......")
    tmp_data = train_data[['ip', 'device', 'os', 'app']].groupby(['ip', 'device', 'os'])['app'].cumcount()
    train_data['X7'] = tmp_data.values
    del tmp_data
    gc.collect()

    tmp_data = train_data[['ip', 'os']].groupby(['ip'])['os'].cumcount()
    train_data['X8'] = tmp_data.values
    del tmp_data
    gc.collect()

    # 特征工程 stage = 3
    print("特征工程 stage = 3 ......")
    D = 2 ** 26
    train_data['category'] = (train_data['ip'].astype(str) + "_" + train_data['app'].astype(str) + "_" + train_data['device'].astype(str) + "_" + train_data['os'].astype(str)).apply(hash) % D
    click_buffer = np.full(D, 3000000000, dtype=np.uint32)
    train_data['epochtime'] = train_data['click_time'].astype(np.int64) // 10 ** 9
    next_clicks = []
    for category, t in zip(reversed(train_data['category'].values), reversed(train_data['epochtime'].values)):
        next_clicks.append(click_buffer[category] - t)
        click_buffer[category] = t
    del (click_buffer)
    next_click = list(reversed(next_clicks))
    train_data['nextClick'] = next_click
    train_data['nextClick_shift'] = pd.DataFrame(next_click).shift(+1).values

    del next_click
    gc.collect()

    # 特征工程 stage = 4
    print("特征工程 stage = 4 ......")
    print(train_data.columns)
    tmp_data = train_data[['ip', 'day', 'hour', 'channel']].groupby(by=['ip', 'day', 'hour'])['channel'].count().reset_index().rename(index=str, columns={'channel': 'ip_tcount'})
    train_data = train_data.merge(tmp_data, on=['ip', 'day', 'hour'], how='left')
    del tmp_data
    gc.collect()

    print('grouping by ip-app combination...')
    tmp_data = train_data[['ip', 'app', 'channel']].groupby(by=['ip', 'app'])['channel'].count().reset_index().rename(index=str, columns={'channel': 'ip_app_count'})
    train_data = train_data.merge(tmp_data, on=['ip', 'app'], how='left')
    del tmp_data
    gc.collect()

    print('grouping by ip-app-os combination...')
    tmp_data = train_data[['ip', 'app', 'os', 'channel']].groupby(by=['ip', 'app', 'os'])['channel'].count().reset_index().rename(index=str, columns={'channel': 'ip_app_os_count'})
    train_data = train_data.merge(tmp_data, on=['ip', 'app', 'os'], how='left')
    del tmp_data
    gc.collect()

    # Adding features with var and mean hour (inspired from nuhsikander's script)
    print('grouping by : ip_day_chl_var_hour')
    tmp_data = train_data[['ip', 'day', 'hour', 'channel']].groupby(by=['ip', 'day', 'channel'])['hour'].var().reset_index().rename(index=str, columns={'hour': 'ip_tchan_count'})
    train_data = train_data.merge(tmp_data, on=['ip', 'day', 'channel'], how='left')
    del tmp_data
    gc.collect()

    print('grouping by : ip_app_os_var_hour')
    tmp_data = train_data[['ip', 'app', 'os', 'hour']].groupby(by=['ip', 'app', 'os'])['hour'].var().reset_index().rename(index=str, columns={'hour': 'ip_app_os_var'})
    train_data = train_data.merge(tmp_data, on=['ip', 'app', 'os'], how='left')
    del tmp_data
    gc.collect()

    print('grouping by : ip_app_channel_var_day')
    tmp_data = train_data[['ip', 'app', 'channel', 'day']].groupby(by=['ip', 'app', 'channel'])['day'].var().reset_index().rename(index=str, columns={'day': 'ip_app_channel_var_day'})
    train_data = train_data.merge(tmp_data, on=['ip', 'app', 'channel'], how='left')
    del tmp_data
    gc.collect()

    print('grouping by : ip_app_chl_mean_hour')
    tmp_data = train_data[['ip', 'app', 'channel', 'hour']].groupby(by=['ip', 'app', 'channel'])['hour'].mean().reset_index().rename(index=str, columns={'hour': 'ip_app_channel_mean_hour'})
    train_data = train_data.merge(tmp_data, on=['ip', 'app', 'channel'], how='left')
    del tmp_data
    gc.collect()


    train_data['ip_tcount'] = train_data['ip_tcount'].astype('uint16')
    train_data['ip_app_count'] = train_data['ip_app_count'].astype('uint16')
    train_data['ip_app_os_count'] = train_data['ip_app_os_count'].astype('uint16')


    # predictors = list(train_data.columns.values)
    # ip_index = predictors.index('ip')
    # click_id_index = predictors.index('click_id')
    # click_time_index = predictors.index('click_time')
    # category_index = predictors.index('category')
    # epochtime_index = predictors.index('epochtime')
    # is_attributed_index = predictors.index('is_attributed')
    # del predictors[ip_index]
    # del predictors[click_id_index]
    # del predictors[click_time_index]
    # del predictors[category_index]
    # del predictors[epochtime_index]
    # del predictors[is_attributed_index]


    print("特征工程:测试数据和训练数据拆分.....")
    test_data = train_data[len_train:]
    val_data = train_data[(len_train - val_size):len_train]
    train_data = train_data[:(len_train - val_size)]


    print("train size: ", len(train_data))
    print("valid size: ", len(val_data))
    print("test size : ", len(test_data))

    print("特征工程完成......")
    return train_data,val_data,test_data


def apply():
    train_data, test_data = load_data(train_path,test_path, skip_rows, load_rows, dtypes)
    train_data, val_data, test_data  = feature_deal(train_data,test_data)
    return train_data,val_data,test_data
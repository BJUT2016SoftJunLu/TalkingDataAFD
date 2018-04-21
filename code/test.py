import pandas as pd
import gc
import numpy as np
import time
import model_lightgbm as lgb
import matplotlib.pyplot as plt
#
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
    val_size=2500
else:
    val_size = 2500000



def lgb_modelfit_nocv(params, dtrain, dvalid, predictors, target='target', objective='binary', metrics='auc',feval=None, early_stopping_rounds=20, num_boost_round=3000, verbose_eval=10, categorical_features=None):
    lgb_params = {
        'boosting_type': 'gbdt',
        'objective': objective,
        'metric':metrics,
        'learning_rate': 0.2,
        #'is_unbalance': 'true',  #because training data is unbalance (replaced with scale_pos_weight)
        'num_leaves': 31,  # we should let it be smaller than 2^(max_depth)
        'max_depth': -1,  # -1 means no limit
        'min_child_samples': 20,  # Minimum number of data need in a child(min_data_in_leaf)
        'max_bin': 255,  # Number of bucketed bin for feature values
        'subsample': 0.6,  # Subsample ratio of the training instance.
        'subsample_freq': 0,  # frequence of subsample, <=0 means no enable
        'colsample_bytree': 0.3,  # Subsample ratio of columns when constructing each tree.
        'min_child_weight': 5,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
        'subsample_for_bin': 200000,  # Number of samples for constructing bin
        'min_split_gain': 0,  # lambda_l1, lambda_l2 and min_gain_to_split to regularization
        'reg_alpha': 0,  # L1 regularization term on weights
        'reg_lambda': 0,  # L2 regularization term on weights
        'nthread': 4,
        'verbose': 0,
        'metric':metrics
    }

    lgb_params.update(params)

    print("preparing validation datasets")

    xgtrain = lgb.Dataset(dtrain[predictors].values, label=dtrain[target].values,
                          feature_name=predictors,
                          categorical_feature=categorical_features
                          )
    xgvalid = lgb.Dataset(dvalid[predictors].values, label=dvalid[target].values,
                          feature_name=predictors,
                          categorical_feature=categorical_features
                          )

    evals_results = {}

    bst1 = lgb.train(lgb_params,
                     xgtrain,
                     valid_sets=[xgtrain, xgvalid],
                     valid_names=['train','valid'],
                     evals_result=evals_results,
                     num_boost_round=num_boost_round,
                     early_stopping_rounds=early_stopping_rounds,
                     verbose_eval=10,
                     feval=feval)

    return bst1,bst1.best_iteration



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


def model_train(train_data,val_data):

    predictors = []
    predictors.extend(['app', 'device', 'os', 'channel', 'hour', 'day',
                       'ip_tcount', 'ip_tchan_count', 'ip_app_count',
                       'ip_app_os_count', 'ip_app_os_var',
                       'ip_app_channel_var_day', 'ip_app_channel_mean_hour','nextClick','nextClick_shift'])

    for i in range(0,9):
        predictors.append("X%d"%(i))

    print('predictors is ', predictors)

    target = 'is_attributed'
    categorical = ['app', 'device', 'os', 'channel', 'hour', 'day']

    params = {
        'learning_rate': 0.20,
        # 'is_unbalance': 'true', # replaced with scale_pos_weight argument
        'num_leaves': 7,  # 2^max_depth - 1
        'max_depth': 3,  # -1 means no limit
        'min_child_samples': 100,  # Minimum number of data need in a child(min_data_in_leaf)
        'max_bin': 100,  # Number of bucketed bin for feature values
        'subsample': 0.7,  # Subsample ratio of the training instance.
        'subsample_freq': 1,  # frequence of subsample, <=0 means no enable
        'colsample_bytree': 0.9,  # Subsample ratio of columns when constructing each tree.
        'min_child_weight': 0,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
        'scale_pos_weight': 200  # because training data is extremely unbalanced
    }

    print("开始训练...")
    start_time = time.time()

    (bst, best_iteration) = lgb_modelfit_nocv(params,
                                              train_data,
                                              val_data,
                                              predictors,
                                              target,
                                              objective='binary',
                                              metrics='auc',
                                              early_stopping_rounds=30,
                                              verbose_eval=True,
                                              num_boost_round=1000,
                                              categorical_features=categorical)

    print('训练完成.... [{}]'.format(time.time() - start_time))
    del train_data
    del val_data
    gc.collect()

    print('显示特征的重要程度.......')
    ax = lgb.plot_importance(bst, max_num_features=100)
    plt.show()

    return predictors,bst,best_iteration


def model_predict(result_path,test_data,predictors,bst,best_iteration):

    result = pd.DataFrame()
    result['click_id'] = test_data['click_id'].astype('int')

    print("开始预测...")
    result['is_attributed'] = bst.predict(test_data[predictors], num_iteration=best_iteration)
    result.to_csv(result_path, index=False)
    print("任务完成...")
    return

def main():
   train_data,test_data = load_data(train_path, test_path, skip_rows, load_rows,dtypes)
   train_data, val_data, test_data = feature_deal(train_data, test_data)
   predictors, bst, best_iteration  = model_train(train_data,val_data)
   model_predict(result_path,test_data,predictors,bst,best_iteration)
   return

if __name__ == "__main__":
    main()



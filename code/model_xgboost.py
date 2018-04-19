import pandas as pd
import xgboost as xgb
import  gc
import time

predictors = []
predictors.extend(['app', 'device', 'os', 'channel', 'hour', 'day',
                   'ip_tcount', 'ip_tchan_count', 'ip_app_count',
                   'ip_app_os_count', 'ip_app_os_var',
                   'ip_app_channel_var_day', 'ip_app_channel_mean_hour', 'nextClick', 'nextClick_shift'])

for i in range(0, 9):
    predictors.append("X%d" % (i))

def model_train(train_data):

    start_time = time.time()


    # 设置模型参数
    params = {'eta': 0.3,
              'tree_method': "hist",
              'grow_policy': "lossguide",
              'max_leaves': 1400,
              'max_depth': 0,
              'subsample': 0.9,
              'colsample_bytree': 0.7,
              'colsample_bylevel': 0.7,
              'min_child_weight': 0,
              'alpha': 4,
              'objective': 'binary:logistic',
              'scale_pos_weight': 9,
              'eval_metric': 'auc',
              'nthread': 8,
              'random_state': 99,
              'silent': True}

    train_data_y = train_data['is_attributed']
    column_list = list(train_data.columns.values)
    del column_list[column_list.index('is_attributed')]
    train_data_x = train_data[column_list]

    print('[{}] 开始 XGBoost 训练'.format(time.time() - start_time))


    dtrain = xgb.DMatrix(train_data_x[predictors], train_data_y)
    # 释放训练数据内存空间
    del train_data_x, train_data_y
    gc.collect()
    watchlist = [(dtrain, 'train')]
    # 生成模型
    model = xgb.train(params, dtrain, 30, watchlist, maximize=True, verbose_eval=1)

    print('[{}] 结束 XGBoost 训练'.format(time.time() - start_time))

    return model

def model_predict(test_data,result_path,model):

    result = pd.DataFrame()
    result['click_id'] = test_data['click_id'].astype('int')

    print("XGBoost 开始预测.....")
    dtest = xgb.DMatrix(test_data[predictors])
    del test_data
    gc.collect()
    result['is_attributed'] = model.predict(dtest, ntree_limit=model.best_ntree_limit)
    result.to_csv(result_path, float_format='%.8f', index=False)

    return


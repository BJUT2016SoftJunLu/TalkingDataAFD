import pandas as pd
import gc
import time
import lightgbm as lgb
import matplotlib.pyplot as plt
#
predictors = []
predictors.extend(['app', 'device', 'os', 'channel', 'hour', 'day',
                   'ip_tcount', 'ip_tchan_count', 'ip_app_count',
                   'ip_app_os_count', 'ip_app_os_var',
                   'ip_app_channel_var_day', 'ip_app_channel_mean_hour', 'nextClick', 'nextClick_shift'])

for i in range(0, 9):
    predictors.append("X%d" % (i))

target = 'is_attributed'
categorical = ['app', 'device', 'os', 'channel', 'hour', 'day']

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
    print(lgb_params)

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


def model_train(train_data,val_data):

    params = {
        'learning_rate': 0.10,
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

    # print('显示特征的重要程度.......')
    # ax = lgb.plot_importance(bst, max_num_features=100)
    # plt.show()

    return bst,best_iteration


def model_predict(result_path,test_data,bst,best_iteration):

    result = pd.DataFrame()
    result['click_id'] = test_data['click_id'].astype('int')

    print("开始预测...")
    result['is_attributed'] = bst.predict(test_data[predictors], num_iteration=best_iteration)
    result.to_csv(result_path, index=False)
    print("任务完成...")
    return

def main():
   # train_data,test_data = load_data(train_path, test_path, skip_rows, load_rows,dtypes)
   # train_data, val_data, test_data = feature_deal(train_data, test_data)
   # predictors, bst, best_iteration  = model_train(train_data,val_data)
   # model_predict(result_path,test_data,predictors,bst,best_iteration)
   return

if __name__ == "__main__":
    main()



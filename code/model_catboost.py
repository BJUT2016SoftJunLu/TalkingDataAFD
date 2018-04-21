#
# import catboost as cb
#
# def model_train(train_data,val_data):
#
#     predictors = []
#     predictors.extend(['app', 'device', 'os', 'channel', 'hour', 'day',
#                        'ip_tcount', 'ip_tchan_count', 'ip_app_count',
#                        'ip_app_os_count', 'ip_app_os_var',
#                        'ip_app_channel_var_day', 'ip_app_channel_mean_hour', 'nextClick', 'nextClick_shift'])
#
#     for i in range(0, 9):
#         predictors.append("X%d" % (i))
#
#     target = 'is_attributed'
#
#     y_train = train_data[target]
#     x_trian = train_data[predictors]
#
#     y_val = val_data[target]
#     x_val = val_data[predictors]
#
#
#     train_pool = cb.Pool(x_trian, y_train)
#     valid_pool = cb.Pool(x_val, y_val)
#
#
#     RANDOM_STATE = 131
#     model_catboost = cb.CatBoostClassifier(iterations=2000,
#                                 learning_rate=0.05,
#                                 l2_leaf_reg=15,
#                                 depth=6,
#                                 leaf_estimation_iterations=3,
#                                 border_count=64,
#                                 loss_function='MultiClass',
#                                 custom_metric=['Accuracy'],
#                                 eval_metric='Accuracy',
#                                 random_seed=RANDOM_STATE,
#                                 classes_count=41
#                                 ).fit(train_pool, eval_set=valid_pool, verbose=False, plot=True)
#
#     return model_catboost
#
#
# def predict(model,test_data):
#     test_pool = cb.Pool(test_data)
#     result = model.predict(test_pool)
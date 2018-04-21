import featrue_one
import model_xgboost
import model_lightgbm

result_path_lightgbm = '../data/result_lightgbm_2.csv'
result_path_xgboost = '../data/result_xgboost.csv'

test_data_path = '../data/fe_test.csv'
train_data_path = '../data/fe_train.csv'
val_data_path = '../data/fe_val.csv'

def main():
    train_data, val_data, test_data = featrue_one.apply()

    train_data.to_csv(train_data_path)
    val_data.to_csv(val_data_path)
    test_data.to_csv(test_data_path)


    # xgboost_model = model_xgboost.model_train(train_data)
    # model_xgboost.model_predict(test_data, result_path_xgboost, xgboost_model)

    bst,best_iteration = model_lightgbm.model_train(train_data,val_data)
    model_lightgbm.model_predict(result_path_lightgbm,test_data,bst,best_iteration)

    return

if __name__ == "__main__":
    main()
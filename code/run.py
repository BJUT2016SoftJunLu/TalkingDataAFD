import featrue_one
import model_xgboost
import model_lightgbm

result_path_lightgbm = '../data/result_lightgbm.csv'
result_path_xgboost = '../data/result_xgboost.csv'

def main():
    train_data, val_data, test_data = featrue_one.apply()

    xgboost_model = model_xgboost.model_train(train_data)
    model_xgboost.model_predict(test_data, result_path_xgboost, xgboost_model)

    # bst,best_iteration = model_lightgbm.model_train(train_data,val_data)
    # model_lightgbm.model_predict(result_path_lightgbm,test_data,bst,best_iteration)

    return

if __name__ == "__main__":
    main()
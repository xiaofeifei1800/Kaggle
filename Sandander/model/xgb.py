
import cPickle
import datetime
import numpy as np
import pandas as pd
import xgboost as xgb



target_cols = ['ind_ahor_fin_ult1','ind_aval_fin_ult1','ind_cco_fin_ult1','ind_cder_fin_ult1','ind_cno_fin_ult1','ind_ctju_fin_ult1','ind_ctma_fin_ult1','ind_ctop_fin_ult1','ind_ctpp_fin_ult1','ind_deco_fin_ult1','ind_deme_fin_ult1','ind_dela_fin_ult1','ind_ecue_fin_ult1','ind_fond_fin_ult1','ind_hip_fin_ult1','ind_plan_fin_ult1','ind_pres_fin_ult1','ind_reca_fin_ult1','ind_tjcr_fin_ult1','ind_valo_fin_ult1','ind_viv_fin_ult1','ind_nomina_ult1','ind_nom_pens_ult1','ind_recibo_ult1']
target_cols = target_cols[2:]

def runXGB(train_X, train_y, seed_val=845895):
	param = {}
	param['objective'] = 'multi:softprob'
	param['eta'] = 0.09
	param['max_depth'] = 6
	param['silent'] = 1
	param['num_class'] = 22
	param['eval_metric'] = "mlogloss"
	param['min_child_weight'] = 12
	param['subsample'] = 0.85
	param['colsample_bytree'] = 0.9
	param['seed'] = seed_val
	num_rounds = 70

	plst = list(param.items())
	xgtrain = xgb.DMatrix(train_X, label=train_y)
	model = xgb.train(plst, xgtrain, num_rounds)
	return model

if __name__ == "__main__":

    start_time = datetime.datetime.now()
    data_path = "/.../Santander/data"
    with open(data_path+ "x_train.pkl", "rb") as f:
        train_X = cPickle.load(f)

    with open(data_path+ "y_train.pkl", "rb") as f:
        train_y = cPickle.load(f)

    train_X = np.array(train_X)
    train_y = np.array(train_y)

    print(train_X.shape, train_y.shape)

    print("Building model..")
    model = runXGB(train_X, train_y, seed_val=0)
    del train_X, train_y

    with open(data_path+ "x_test.pkl", "rb") as f:
        test_X = cPickle.load(f)

    test_X = np.array(test_X)
    print(test_X.shape)
    xgtest = xgb.DMatrix(test_X)

    print("Predicting..")
    preds = model.predict(xgtest)
    del test_X, xgtest

    print("Getting the top products..")
    test_id = np.array(pd.read_csv(data_path + "test.csv", usecols=['ncodpers'])['ncodpers'])
    new_products = []

    with open(data_path +"cust_dict.pkl", "rb") as f:
        cust_dict = cPickle.load(f)

    for i, idx in enumerate(test_id):
        new_products.append([max(x1 - x2,0) for (x1, x2) in zip(preds[i,:], cust_dict[idx])])
    target_cols = np.array(target_cols)
    preds = np.argsort(np.array(new_products), axis=1)
    preds = np.fliplr(preds)[:,:7]
    final_preds = [" ".join(list(target_cols[pred])) for pred in preds]
    out_df = pd.DataFrame({'ncodpers':test_id, 'added_products':final_preds})
    out_df.to_csv('sub_xgb_new.csv', index=False)

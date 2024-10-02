# Grid search example using 5 fold validation
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, train_test_split
import time
import pandas as pd 

df = pd.read_stata("ecart_data/nlp_clindiag_reason_data.dta")

import configparser
cfgfile = "ecart_data/config.cfg"
cfg = configparser.ConfigParser()
cfg.read(cfgfile)
train_csv = pd.read_csv("FINAL/val.csv",sep="|")
print(len(train_csv))

cfg_inputs = cfg.get('DATAPOINTS','inputs').split(", ")
cfg_labels = cfg.get('DATAPOINTS','labels').split(", ")

# Sample dataset
# X, y = your_data_here (X is the feature set and y is the target)

folder_path = "FINAL_remove_false_alarm/"
dev_df = pd.read_csv(folder_path+"all.csv", sep="|")

def read_data_from_csv(fn):
    train_csv = pd.read_csv(fn, sep="|")
    print(len(train_csv))

    cfg_inputs = cfg.get('DATAPOINTS','inputs').split(", ")
    cfg_labels = cfg.get('DATAPOINTS','labels').split(", ")

    X, Y = [], [] 
    for index, row in train_csv.iterrows():
        print(" =========== ")
        inputs = row[cfg_inputs]
        labels = row[cfg_labels]
        print(inputs)
        X.append(inputs)
        Y.append(labels.tolist()[22])
    return X, Y 

def extract_data_from_pd(df):
    cfg_inputs = cfg.get('DATAPOINTS','inputs').split(", ")
    cfg_labels = cfg.get('DATAPOINTS','labels').split(", ")

    X, Y = [], [] 
    for index, row in df.iterrows():
        #print(" =========== ")
        inputs = row[cfg_inputs]
        labels = row[cfg_labels]
        #print(inputs)
        X.append(inputs)
        Y.append(labels.tolist()[22])
        
    return X, Y 
   
train_X, train_Y = read_data_from_csv(folder_path+"all.csv")

## Remove the cat variables for now untill getting all the cat variables 
selected_columns = ['age','sbp_impute', 'dbp_impute',
       'ppi_impute', 'o2sat_impute', 'temp_c_impute', 'avpu_impute',
       'albumin_impute', 'alk_phos_impute', 'anion_gap_impute',
       'bili_total_impute', 'bun_impute', 'bun_cr_ratio_impute',
       'calcium_impute', 'chloride', 'co2_impute', 'creatinine_impute',
       'gluc_ser_impute', 'hb_impute', 'platelet_count_impute',
       'potassium_impute', 'sgot_impute', 'sodium_impute',
       'total_protein_impute', 'wbc_impute']

def grid_search_5_fold(input_X, input_Y,cv_num=5):
    # with GPU Enhancement 
    start_time = time.time()
    # Define the parameter grid
    param_grid = {
        'n_estimators': [50, 100, 250, 500],
        'max_depth': [2, 5, 10, 15, 20],
        'learning_rate': [0.005, 0.01, 0.05, 0.1],
        'min_child_weight': [1, 2, 3]
    }

    # Instantiate an XGBoost classifier object
    xgb_clf = xgb.XGBClassifier()

    # GridSearchCV
    grid_search = GridSearchCV(estimator=xgb_clf, param_grid=param_grid, cv=cv_num, verbose=1)

    # Fit the grid search to the data
    grid_search.fit(input_X, input_Y)

    end_time = time.time()
    time_taken = (end_time - start_time) / 60
    # Best parameters and best score
    print("Best parameters:", grid_search.best_params_)
    print("Best score:", grid_search.best_score_)
    print(f"Running time: {time_taken:.4f} mins")
    
    return grid_search 

def grid_search_5_fold_GPU(input_X, input_Y, cv_num=5):
    # with GPU Enhancement 
    start_time = time.time()
    # Define the parameter grid
    param_grid = {
        'n_estimators': [50, 100, 250, 500],
        'max_depth': [2, 5, 10, 15, 20],
        'learning_rate': [0.005, 0.01, 0.05, 0.1],
        'min_child_weight': [1, 2, 3]
    }

    # Instantiate an XGBoost classifier object
    xgb_clf = xgb.XGBClassifier(tree_method='gpu_hist',predictor='gpu_predictor',gpu_id=2)

    # GridSearchCV
    grid_search = GridSearchCV(estimator=xgb_clf, param_grid=param_grid, cv=cv_num, verbose=1)

    # Fit the grid search to the data
    grid_search.fit(selected_X, new_Y)

    end_time = time.time()
    time_taken = (end_time - start_time) / 60
    # Best parameters and best score
    print("Best parameters:", grid_search.best_params_)
    print("Best score:", grid_search.best_score_)
    print(f"Running time With GPU: {time_taken:.4f} mins")
    
    return grid_search 



from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score 

def main_kfold_xgboost(data,k_num=5):
    """Input: data: pd.dataframe for all patients
    Output: AUROC on 5-fold on all patients 
    """
    # Step 1: get all IDs, perform KFold split 
    kf = KFold(n_splits=5)
    all_enc_ids = data['encounter_id']  
    print(f"TOTAL ENCOUNTER NUM: {len(all_enc_ids)}")
    
    selected_columns = ['age','sbp_impute', 'dbp_impute',
       'ppi_impute', 'o2sat_impute', 'temp_c_impute', 'avpu_impute',
       'albumin_impute', 'alk_phos_impute', 'anion_gap_impute',
       'bili_total_impute', 'bun_impute', 'bun_cr_ratio_impute',
       'calcium_impute', 'chloride', 'co2_impute', 'creatinine_impute',
       'gluc_ser_impute', 'hb_impute', 'platelet_count_impute',
       'potassium_impute', 'sgot_impute', 'sodium_impute',
       'total_protein_impute', 'wbc_impute']
    
    test_enc_ids = {}
    _ = 0 # batch index for k-fold split 
    batch_auroc = {}
    batch_yproba = {}
    # Step 2: 
    for train_index, test_index in kf.split(all_enc_ids):
        # a. Perform KFold Split,
        print(f"===== STARTING AT THE {_}TH BATCH OF K-FOLD ======")
        train_df = dev_df.iloc[train_index]
        print(f"training set size: {train_df.shape}")
        test_df = dev_df.iloc[test_index]
        print(f"test set size: {test_df.shape}")
        test_enc_ids[_] = test_df["encounter_id"].tolist()
        test_X, test_Y = extract_data_from_pd(test_df)
        train_X, train_Y = extract_data_from_pd(train_df)
        
        # b. for each split, perform grid search 
        new_X = pd.DataFrame(train_X)
        selected_X = new_X[selected_columns]
        new_Y = replace_nan_in_y(train_Y)
        grid_search = grid_search_5_fold_GPU(selected_X, new_Y,cv_num=5)  
        
        # c. pick the best hyperparam and train the clf
        best_params = grid_search.best_params_ 
        #best_xgb_clf = xgb.XGBClassifier(**best_params, device="cuda")
        best_params["tree_method"] = "gpu_hist"
        best_xgb_clf_gpu = xgb.XGBClassifier(**best_params,predictor="gpu_predictor", gpu_id=1)
        start_time = time.time()
        best_xgb_clf_gpu.fit(selected_X, new_Y) 
        end_time = time.time()
        
        time_taken_training = (end_time - start_time) / 60  
        
        print(f"Time Taken Training: {time_taken_training:.4f} mins")
        
        # d. evaluate on test set         
        test_X = pd.DataFrame(test_X)
        
        test_Y = replace_nan_in_y(test_Y)
        test_selected_X = test_X[selected_columns]
        print(f"XGB Test Input Shape {test_selected_X.shape}")
        
        y_pred_proba = best_xgb_clf_gpu.predict_proba(test_selected_X)[:, 1]
        auroc = roc_auc_score(test_Y, y_pred_proba)
        batch_yproba[_] = y_pred_proba 
        batch_auroc[_] = auroc 
        
        print(f"Batch {_} AUROC: {auroc:.4f}")
        _ +=1 
        
    return test_enc_ids, batch_auroc, batch_yproba 


test_enc_ids, batch_auroc, batch_yproba  = main_kfold_xgboost(dev_df)


## Draw the AUROC
import matplotlib.pyplot as plt
from sklearn import metrics
fpr, tpr, _ = metrics.roc_curve(all_y_golds, all_y_probs)
auc = metrics.roc_auc_score(all_y_golds, all_y_probs)
plt.plot(fpr, tpr, label="UW, auc="+str(auc))
plt.legend(loc=4)
plt.show()

## With perfect calibration 
from sklearn.calibration import calibration_curve 
true_prob, pred_prob = calibration_curve(all_y_golds, all_y_probs, n_bins=10)
#auc = metrics.roc_auc_score(all_y_golds, all_y_probs)
#plt.figure(figuresize=(8,8))
plt.plot(pred_prob, true_prob, marker='o', linewidth=1, label="Model")
plt.plot([0,1],[0,1], linestyle='--', label="Perfectly calibrated")
plt.xlabel('Pred Prob')
plt.ylabel('True Prob in each bin')
plt.title('Calibration curve') 
#plt.plot(fpr, tpr, label="UW, auc="+str(auc))
plt.legend(loc=4)
plt.show()

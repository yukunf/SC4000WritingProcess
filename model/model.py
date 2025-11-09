import lightgbm as lgb
import joblib
import pandas as pd
import os
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

N_SPLITS = 5
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

if os.path.exists("../data/"):
    data_path = "../data/"
else:
    data_path = "data/"
df_train = pd.read_csv(f"{data_path}train_preprocessed.csv")
df_test = pd.read_csv(f"{data_path}test_preprocessed.csv")

df_train = df_train.drop('id', axis=1)
test_ids = df_test['id']  # Save IDs for submission
df_test = df_test.drop('id', axis=1)

X = df_train.iloc[:, :-1]  
X_test = df_test
y = df_train.iloc[:, -1].values
preds = np.zeros(len(df_train))
test_preds = np.zeros(len(df_test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_train, y_train = X.iloc[train_idx], y[train_idx]
    X_val, y_val = X.iloc[val_idx], y[val_idx]
    # params from https://www.kaggle.com/code/alexryzhkov/lgbm-and-nn-on-sentences#LightGBM-train-and-predict
    params = {
        "objective": "regression_l1",
        "metric": "rmse",
        "n_estimators": 12000,
        "verbosity": -1,
        "random_state": 42,
        "reg_alpha": 0.007678095440286993,
        "reg_lambda": 0.34230534302168353,
        "colsample_bytree": 0.627061253588415,
        "subsample": 0.854942238828458,
        "learning_rate": 0.038697981947473245,
        "num_leaves": 22,
        "max_depth": 37,
        "min_child_samples": 18,
        "n_jobs": 4,
    }
    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(200, verbose=False)],
    )
    models = []
    models.append(model)
    val_preds = model.predict(X_val)
    preds[val_idx] = val_preds
    test_preds += model.predict(X_test) / N_SPLITS

joblib.dump(models, f"lgbm_.pkl")
rmse = np.sqrt(mean_squared_error(y, preds))
print(f"{rmse}")
submission = pd.DataFrame({"id": test_ids, "score": test_preds})
submission.to_csv(f"{data_path}submission.csv", index=False)

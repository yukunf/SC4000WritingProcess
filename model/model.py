import lightgbm as lgb
import joblib
import pandas as pd
import os
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.svm import SVC
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

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

X = df_train.iloc[:, :-3]
X_essay = df_train.iloc[:, -2]
X_operations = df_train.iloc[:, -3]
# Test data has no score column, so last 2 columns are operation and text
X_test = df_test.iloc[:, :-2]
X_essay_test = df_test.iloc[:, -1]
X_operations_test = df_test.iloc[:, -2]
y = df_train.iloc[:, -1].values
preds = np.zeros(len(df_train))
test_preds = np.zeros(len(df_test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_train, y_train = X.iloc[train_idx], y[train_idx]
    X_essay_train, X_operations_train = X_essay.iloc[train_idx], X_operations.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y[val_idx]

    # tfidf training
    X_essay_val, X_operations_val = X_essay.iloc[val_idx], X_operations.iloc[val_idx]
    tfidf_essay = TfidfVectorizer(max_features=30000)
    X_essay_train = tfidf_essay.fit_transform(X_essay_train)
    X_essay_val = tfidf_essay.transform(X_essay_val)
    X_essay_test_tfidf = tfidf_essay.transform(X_essay_test)
    tfidf_operations = TfidfVectorizer(max_features=30000)
    X_operations_train = tfidf_operations.fit_transform(X_operations_train)
    X_operations_val = tfidf_operations.transform(X_operations_val)
    X_operations_test_tfidf = tfidf_operations.transform(X_operations_test)
    svd_essay = TruncatedSVD(n_components=64, random_state=42,)
    X_essay_train_svd = svd_essay.fit_transform(X_essay_train)
    X_essay_val_svd = svd_essay.transform(X_essay_val)
    X_essay_test_svd = svd_essay.transform(X_essay_test_tfidf)
    svd_operations = TruncatedSVD(n_components=32, random_state=42)
    X_operations_train_svd = svd_operations.fit_transform(X_operations_train)
    X_operations_val_svd = svd_operations.transform(X_operations_val)
    X_operations_test_svd = svd_operations.transform(X_operations_test_tfidf)

    X_train = np.hstack([
        X_train.values,
        X_essay_train_svd,
        X_operations_train_svd,
    ])
    X_val = np.hstack([
        X_val.values,
        X_essay_val_svd,
        X_operations_val_svd,
    ])
    X_test_fold = np.hstack([
        X_test.values,
        X_essay_test_svd,
        X_operations_test_svd,
    ])
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
        callbacks=[lgb.early_stopping(1500, verbose=False)],
    )
    models = []
    models.append(model)
    val_preds = model.predict(X_val)
    preds[val_idx] = val_preds
    test_preds += model.predict(X_test_fold) / N_SPLITS

joblib.dump(models, f"lgbm_.pkl")
rmse = np.sqrt(mean_squared_error(y, preds))
print(f"{rmse}")
submission = pd.DataFrame({"id": test_ids, "score": test_preds})
submission.to_csv(f"{data_path}submission.csv", index=False)

from typing import Dict, List, Tuple, Optional
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import time
import numpy as np
import pandas as pd
import joblib
import pickle
import catboost as cb
import xgboost as xgb
import lightgbm as lgb
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"


class EnhancedEnsembleModel:
    def __init__(
        self,
        n_splits: int = 5,
        random_state: int = 42,
    ):
        self.n_splits = n_splits
        self.random_state = random_state
        self.kf = KFold(n_splits=n_splits, shuffle=True,
                        random_state=random_state)

        self.lgb_models: List[lgb.LGBMRegressor] = []
        self.xgb_models: List[xgb.XGBRegressor] = []
        self.catboost_models: List[cb.CatBoostRegressor] = []
        self.stacking_model: Optional[Ridge] = None

        self.lgb_params = {
            "objective": "regression_l1",
            "metric": "rmse",
            "n_estimators": 12000,
            "verbosity": -1,
            "random_state": random_state,
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

        self.xgb_params = {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "n_estimators": 5000,
            "random_state": random_state,
            "learning_rate": 0.03,
            "max_depth": 8,
            "min_child_weight": 3,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "n_jobs": 1,
            "tree_method": "hist",
        }

        self.catboost_params = {
            "iterations": 5000,
            "learning_rate": 0.03,
            "depth": 8,
            "l2_leaf_reg": 3,
            "random_state": random_state,
            "verbose": False,
            "thread_count": 4,
            "loss_function": "RMSE",
        }

    def train_single_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        model_type: str,
    ) -> Tuple:
        if model_type == "lgb":
            model = lgb.LGBMRegressor(**self.lgb_params)
            X_train_np = X_train if isinstance(
                X_train, np.ndarray) else X_train.values
            X_val_np = X_val if isinstance(X_val, np.ndarray) else X_val.values
            model.fit(
                X_train_np,
                y_train,
                eval_set=[(X_val_np, y_val)],
                callbacks=[lgb.early_stopping(200, verbose=False)],
            )

        elif model_type == "xgb":
            model = xgb.XGBRegressor(
                **self.xgb_params, early_stopping_rounds=200)
            model.fit(X_train, y_train, eval_set=[
                      (X_val, y_val)], verbose=False)

        elif model_type == "catboost":
            model = cb.CatBoostRegressor(**self.catboost_params)
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=200,
                verbose=False,
            )

        X_val_pred = X_val if isinstance(X_val, np.ndarray) else X_val.values
        val_preds = model.predict(X_val_pred)
        val_score = np.sqrt(mean_squared_error(y_val, val_preds))

        return model, val_preds, val_score

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> Dict[str, float]:
        X_np = X.values if isinstance(X, pd.DataFrame) else X

        oof_lgb = np.zeros(len(X))
        oof_xgb = np.zeros(len(X))
        oof_catboost = np.zeros(len(X))

        lgb_scores = []
        xgb_scores = []
        catboost_scores = []

        print("=" * 50)
        print("Training Enhanced Ensemble Model")
        print("=" * 50)

        for fold, (train_idx, val_idx) in enumerate(self.kf.split(X_np, y)):
            print(f"\nFold {fold + 1}/{self.n_splits}")

            X_train, X_val = X_np[train_idx], X_np[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            start_time = time.time()
            lgb_model, lgb_val_preds, lgb_score = self.train_single_model(
                X_train, y_train, X_val, y_val, "lgb"
            )
            self.lgb_models.append(lgb_model)
            oof_lgb[val_idx] = lgb_val_preds
            lgb_scores.append(lgb_score)

            print(
                f"  LightGBM - Val RMSE: {lgb_score:.5f} ({time.time() - start_time:.1f}s)"
            )

            start_time = time.time()
            xgb_model, xgb_val_preds, xgb_score = self.train_single_model(
                X_train, y_train, X_val, y_val, "xgb"
            )
            self.xgb_models.append(xgb_model)
            oof_xgb[val_idx] = xgb_val_preds
            xgb_scores.append(xgb_score)

            print(
                f"  XGBoost   - Val RMSE: {xgb_score:.5f} ({time.time() - start_time:.1f}s)"
            )

            start_time = time.time()
            cb_model, cb_val_preds, cb_score = self.train_single_model(
                X_train, y_train, X_val, y_val, "catboost"
            )
            self.catboost_models.append(cb_model)
            oof_catboost[val_idx] = cb_val_preds
            catboost_scores.append(cb_score)

            print(
                f"  CatBoost  - Val RMSE: {cb_score:.5f} ({time.time() - start_time:.1f}s)"
            )

        lgb_oof_score = np.sqrt(mean_squared_error(y, oof_lgb))
        xgb_oof_score = np.sqrt(mean_squared_error(y, oof_xgb))
        catboost_oof_score = np.sqrt(mean_squared_error(y, oof_catboost))

        print("\n" + "=" * 50)
        print("Out-of-Fold Results:")
        print(f"  LightGBM:  {lgb_oof_score:.5f}")
        print(f"  XGBoost:   {xgb_oof_score:.5f}")
        print(f"  CatBoost:  {catboost_oof_score:.5f}")

        stacking_features = np.column_stack([oof_lgb, oof_xgb, oof_catboost])
        self.stacking_model = Ridge(alpha=1.0, random_state=self.random_state)
        self.stacking_model.fit(stacking_features, y)

        ensemble_preds = self.stacking_model.predict(stacking_features)
        ensemble_score = np.sqrt(mean_squared_error(y, ensemble_preds))

        print(f"  Stacked:   {ensemble_score:.5f}")
        print(
            f"\nStacking weights: LGB={self.stacking_model.coef_[0]:.3f}, "
            f"XGB={self.stacking_model.coef_[1]:.3f}, "
            f"CB={self.stacking_model.coef_[2]:.3f}"
        )
        print("=" * 50)

        return {
            "lgb_oof": lgb_oof_score,
            "xgb_oof": xgb_oof_score,
            "catboost_oof": catboost_oof_score,
            "ensemble_oof": ensemble_score,
        }

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_np = X.values if isinstance(X, pd.DataFrame) else X

        lgb_preds = np.zeros(len(X))
        xgb_preds = np.zeros(len(X))
        catboost_preds = np.zeros(len(X))

        for lgb_model in self.lgb_models:
            lgb_preds += lgb_model.predict(X_np) / self.n_splits

        for xgb_model in self.xgb_models:
            xgb_preds += xgb_model.predict(X_np) / self.n_splits

        for cb_model in self.catboost_models:
            catboost_preds += cb_model.predict(X_np) / self.n_splits

        stacking_features = np.column_stack(
            [lgb_preds, xgb_preds, catboost_preds])
        final_preds = self.stacking_model.predict(stacking_features)

        return final_preds

    def save(self, path: str, use_pickle: bool = True):
        os.makedirs(path, exist_ok=True)

        if use_pickle:
            with open(os.path.join(path, "lgb_models.pkl"), "wb") as f:
                pickle.dump(self.lgb_models, f)
            with open(os.path.join(path, "xgb_models.pkl"), "wb") as f:
                pickle.dump(self.xgb_models, f)
            with open(os.path.join(path, "catboost_models.pkl"), "wb") as f:
                pickle.dump(self.catboost_models, f)
            with open(os.path.join(path, "stacking_model.pkl"), "wb") as f:
                pickle.dump(self.stacking_model, f)
            config = {
                "n_splits": self.n_splits,
                "random_state": self.random_state,
            }
            with open(os.path.join(path, "config.pkl"), "wb") as f:
                pickle.dump(config, f)
        else:
            joblib.dump(self.lgb_models, os.path.join(path, "lgb_models.pkl"))
            joblib.dump(self.xgb_models, os.path.join(path, "xgb_models.pkl"))
            joblib.dump(self.catboost_models, os.path.join(
                path, "catboost_models.pkl"))
            joblib.dump(self.stacking_model, os.path.join(
                path, "stacking_model.pkl"))
            config = {
                "n_splits": self.n_splits,
                "random_state": self.random_state,
            }
            joblib.dump(config, os.path.join(path, "config.pkl"))

        print(
            f"Model saved to {path} using {'pickle' if use_pickle else 'joblib'}")

    def save_lgbm_only(self, filepath: str):
        with open(filepath, "wb") as f:
            pickle.dump(self.lgb_models, f)

        print(f"LightGBM models saved to {filepath}")

    def load(self, path: str, use_pickle: bool = True):
        if use_pickle:
            with open(os.path.join(path, "lgb_models.pkl"), "rb") as f:
                self.lgb_models = pickle.load(f)
            with open(os.path.join(path, "xgb_models.pkl"), "rb") as f:
                self.xgb_models = pickle.load(f)
            with open(os.path.join(path, "catboost_models.pkl"), "rb") as f:
                self.catboost_models = pickle.load(f)
            with open(os.path.join(path, "stacking_model.pkl"), "rb") as f:
                self.stacking_model = pickle.load(f)
            with open(os.path.join(path, "config.pkl"), "rb") as f:
                config = pickle.load(f)
        else:
            self.lgb_models = joblib.load(os.path.join(path, "lgb_models.pkl"))
            self.xgb_models = joblib.load(os.path.join(path, "xgb_models.pkl"))
            self.catboost_models = joblib.load(
                os.path.join(path, "catboost_models.pkl"))
            self.stacking_model = joblib.load(
                os.path.join(path, "stacking_model.pkl"))
            config = joblib.load(os.path.join(path, "config.pkl"))

        self.n_splits = config["n_splits"]
        self.random_state = config["random_state"]

        print(
            f"Model loaded from {path} using {'pickle' if use_pickle else 'joblib'}")

    def load_lgbm_only(self, filepath: str):
        with open(filepath, "rb") as f:
            self.lgb_models = pickle.load(f)

        print(f"LightGBM models loaded from {filepath}")


def train_enhanced_ensemble(
    data_path: str = "../data/",
    save_path: str = "../data/",
    n_splits: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    if not os.path.exists(data_path):
        data_path = "data/"

    train_file = f"{data_path}train_preprocessed.csv"
    test_file = f"{data_path}test_preprocessed.csv"

    df_train = pd.read_csv(train_file)
    df_test = pd.read_csv(test_file)

    df_train = df_train.drop("id", axis=1)
    test_ids = df_test["id"]
    df_test = df_test.drop("id", axis=1)

    X = df_train.iloc[:, :-1]
    y = df_train.iloc[:, -1].values
    X_test = df_test

    print(f"Training data shape: {X.shape}")
    print(f"Test data shape: {X_test.shape}")

    model = EnhancedEnsembleModel(
        n_splits=n_splits,
        random_state=random_state,
    )

    model.fit(X, y)

    test_preds = model.predict(X_test)

    model_save_path = f"{save_path}model_enhanced_ensemble"
    model.save(model_save_path, use_pickle=True)

    lgbm_save_path = f"{save_path}lgbm.pkl"
    model.save_lgbm_only(lgbm_save_path)

    submission = pd.DataFrame({"id": test_ids, "score": test_preds})
    submission_path = f"{save_path}submission_enhanced_ensemble.csv"
    submission.to_csv(submission_path, index=False)

    print(f"\nSubmission saved to {submission_path}")
    print("Submission preview:")
    print(submission.head())
    print("\nScore statistics:")
    print(submission["score"].describe())

    return submission


if __name__ == "__main__":
    submission = train_enhanced_ensemble(
        data_path="../data/",
        save_path="../data/",
        n_splits=5,
        random_state=42,
    )

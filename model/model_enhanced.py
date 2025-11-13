from typing import Dict, List, Tuple, Optional
from sklearn.linear_model import (
    Ridge,
    ElasticNet,
    BayesianRidge,
    HuberRegressor,
    PoissonRegressor,
    PassiveAggressiveRegressor,
)
from sklearn.svm import LinearSVR
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from scipy.optimize import minimize
from tqdm import tqdm
import time
import numpy as np
import pandas as pd
import joblib
import pickle
import catboost as cb
import xgboost as xgb
import lightgbm as lgb
import os
import warnings

warnings.filterwarnings("ignore")
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

        # Advanced components (integrated by default)
        self.linear_models = self._init_linear_models()
        self.linear_scaler = MinMaxScaler()
        self.linear_optimal_weights = None

        # Classifier meta-feature mappings
        self.score_to_class = {
            4.0: 9, 3.5: 8, 4.5: 7, 3.0: 6, 2.5: 5,
            5.0: 4, 5.5: 1, 2.0: 3, 1.5: 2, 6.0: 1,
            1.0: 0, 0.5: 0,
        }
        self.class_to_score = {
            0: 1.0, 1: 6.0, 2: 1.5, 3: 2.0, 4: 5.0,
            5: 2.5, 6: 3.0, 7: 4.5, 8: 3.5, 9: 4.0,
        }
        self.n_classes = 10
        self.final_weights = None

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

    def _init_linear_models(self):
        return [
            ("LinearSVR", LinearSVR(
                C=0.9, loss="squared_epsilon_insensitive", max_iter=2000)),
            ("ElasticNet", ElasticNet(alpha=0.001, l1_ratio=0.5,
             random_state=self.random_state, selection="cyclic")),
            ("Ridge", Ridge(alpha=10)),
            ("PassiveAggressive", PassiveAggressiveRegressor(
                C=0.001, loss="squared_epsilon_insensitive")),
            ("Huber", HuberRegressor(epsilon=1.25, alpha=20)),
            ("Poisson", PoissonRegressor(alpha=0.01)),
            ("BayesianRidge", BayesianRidge()),
        ]

    def _generate_classifier_meta_features(self, X, y, X_test=None, pca_components=100):

        y_class = pd.Series(y).map(self.score_to_class).values

        print("Training MultinomialNB...")
        nb_oof, nb_test = self._train_classifier_meta(
            X, y_class, X_test, pca_components, "nb"
        )

        print("Training MLPClassifier...")
        mlp_oof, mlp_test = self._train_classifier_meta(
            X, y_class, X_test, pca_components, "mlp"
        )

        return {
            "nb_oof": nb_oof, "nb_test": nb_test,
            "mlp_oof": mlp_oof, "mlp_test": mlp_test,
        }

    def _train_classifier_meta(self, X, y_class, X_test, pca_components, model_type):
        # Apply PCA and square
        pca = PCA(n_components=pca_components, random_state=self.random_state)
        if X_test is not None:
            combined = pd.concat([X, X_test]) if isinstance(
                X, pd.DataFrame) else np.vstack([X, X_test])
            pca.fit(combined.fillna(0) if isinstance(
                combined, pd.DataFrame) else combined)
            X_pca = pca.transform(X.fillna(0) if isinstance(
                X, pd.DataFrame) else X) ** 2
            X_test_pca = pca.transform(X_test.fillna(0) if isinstance(
                X_test, pd.DataFrame) else X_test) ** 2
        else:
            X_pca = pca.fit_transform(
                X.fillna(0) if isinstance(X, pd.DataFrame) else X) ** 2
            X_test_pca = None

        # Cross-validation
        oof_prob = np.zeros((len(X), self.n_classes))
        test_prob = np.zeros((len(X_test), self.n_classes)
                             ) if X_test is not None else None

        skf = StratifiedKFold(n_splits=self.n_splits,
                              shuffle=True, random_state=self.random_state)

        for train_idx, val_idx in skf.split(X_pca, y_class):
            X_train, X_val = X_pca[train_idx], X_pca[val_idx]
            y_train, y_val = y_class[train_idx], y_class[val_idx]

            if model_type == "nb":
                model = MultinomialNB(alpha=1.0)
            else:
                model = MLPClassifier(
                    random_state=self.random_state, max_iter=300)

            model.fit(X_train, y_train)
            oof_prob[val_idx] = model.predict_proba(X_val)

            if test_prob is not None:
                test_prob += model.predict_proba(X_test_pca) / self.n_splits

        return oof_prob, test_prob

    def _compute_weighted_score(self, probabilities):
        weighted_sum = np.zeros(len(probabilities))
        for i in range(self.n_classes):
            weighted_sum += probabilities[:, i] * self.class_to_score[i]
        return weighted_sum

    def _train_linear_ensemble(self, X, y, X_test=None):
        # Scale features
        if X_test is not None:
            combined = pd.concat([X, X_test])
            self.linear_scaler.fit(combined)
            X_scaled = pd.DataFrame(
                self.linear_scaler.transform(X), columns=X.columns)
            X_test_scaled = pd.DataFrame(
                self.linear_scaler.transform(X_test), columns=X.columns)
        else:
            X_scaled = pd.DataFrame(
                self.linear_scaler.fit_transform(X), columns=X.columns)
            X_test_scaled = None

        # Train each model
        oof_predictions = np.zeros((len(X), len(self.linear_models)))
        test_predictions = np.zeros(
            (len(X_test), len(self.linear_models))) if X_test is not None else None

        skf = StratifiedKFold(n_splits=self.n_splits,
                              shuffle=True, random_state=self.random_state)

        for i, (name, model_template) in enumerate(tqdm(self.linear_models, desc="Training linear models")):
            oof_preds = np.zeros(len(X))
            test_preds = np.zeros(len(X_test)) if X_test is not None else None

            for train_idx, val_idx in skf.split(X_scaled, y.astype(str)):
                from sklearn.base import clone
                model = clone(model_template)

                X_train = X_scaled.iloc[train_idx].fillna(0)
                X_val = X_scaled.iloc[val_idx].fillna(0)
                y_train, y_val = y[train_idx], y[val_idx]

                model.fit(X_train, y_train)
                oof_preds[val_idx] = model.predict(X_val)

                if test_preds is not None:
                    test_preds += model.predict(X_test_scaled.fillna(0)
                                                ) / self.n_splits

            oof_preds = np.clip(oof_preds, 0, 6)
            oof_predictions[:, i] = oof_preds

            if test_preds is not None:
                test_preds = np.clip(test_preds, 0, 6)
                test_predictions[:, i] = test_preds

            rmse = np.sqrt(mean_squared_error(y, oof_preds))
            print(f"{name:20s} CV RMSE: {rmse:.5f}")

        self.linear_optimal_weights = self._optimize_weights(
            oof_predictions, y)

        ensemble_oof = np.dot(oof_predictions, self.linear_optimal_weights)
        ensemble_rmse = np.sqrt(mean_squared_error(y, ensemble_oof))
        print(f"Linear Ensemble CV RMSE: {ensemble_rmse:.5f}")

        if test_predictions is not None:
            ensemble_test = np.dot(
                test_predictions, self.linear_optimal_weights)
            return ensemble_oof, ensemble_test

        return ensemble_oof, None

    def _optimize_weights(self, predictions, y_true):
        def weighted_rmse(weights, preds, y):
            ensemble_pred = np.dot(preds, weights)
            return np.sqrt(mean_squared_error(y, ensemble_pred))

        n_models = predictions.shape[1]
        initial_weights = np.ones(n_models) / n_models
        constraints = {"type": "eq", "fun": lambda w: sum(w) - 1}
        bounds = [(0, 1)] * n_models

        opt_result = minimize(
            weighted_rmse,
            initial_weights,
            args=(predictions, y_true),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        return opt_result.x

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

    def fit(self, X: pd.DataFrame, y: np.ndarray, X_test: pd.DataFrame = None) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
        print("=" * 70)
        print("ADVANCED ENHANCED ENSEMBLE MODEL")
        print("=" * 70)

        # Step 1: Generate classifier meta-features
        print("\n" + "=" * 70)
        print("STEP 1: Generating Classifier Meta-Features (NB + MLP)")
        print("=" * 70)

        meta_features = self._generate_classifier_meta_features(X, y, X_test)

        # Add meta-features to original features
        X_enhanced = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        X_test_enhanced = X_test.copy() if X_test is not None and isinstance(
            X_test, pd.DataFrame) else (pd.DataFrame(X_test) if X_test is not None else None)

        # Add NB features
        for i in range(meta_features["nb_oof"].shape[1]):
            X_enhanced[f"nb_prob_{i}"] = meta_features["nb_oof"][:, i]
            if X_test_enhanced is not None:
                X_test_enhanced[f"nb_prob_{i}"] = meta_features["nb_test"][:, i]
        X_enhanced["nb_weighted"] = self._compute_weighted_score(
            meta_features["nb_oof"])
        if X_test_enhanced is not None:
            X_test_enhanced["nb_weighted"] = self._compute_weighted_score(
                meta_features["nb_test"])

        # Add MLP features
        for i in range(meta_features["mlp_oof"].shape[1]):
            X_enhanced[f"mlp_prob_{i}"] = meta_features["mlp_oof"][:, i]
            if X_test_enhanced is not None:
                X_test_enhanced[f"mlp_prob_{i}"] = meta_features["mlp_test"][:, i]
        X_enhanced["mlp_weighted"] = self._compute_weighted_score(
            meta_features["mlp_oof"])
        if X_test_enhanced is not None:
            X_test_enhanced["mlp_weighted"] = self._compute_weighted_score(
                meta_features["mlp_test"])

        print(f"\nEnhanced feature shape: {X_enhanced.shape}")

        # Step 2: Train base ensemble (LGB + XGB + CatBoost) on enhanced features
        print("\n" + "=" * 70)
        print("STEP 2: Training Base Ensemble (LGB + XGB + CatBoost)")
        print("=" * 70)

        scores, ensemble_oof = self._train_base_ensemble(X_enhanced, y)
        ensemble_test = self._predict_base_ensemble(
            X_test_enhanced) if X_test_enhanced is not None else None

        # Step 3: Train linear model ensemble
        print("\n" + "=" * 70)
        print("STEP 3: Training Linear Model Ensemble")
        print("=" * 70)

        linear_oof, linear_test = self._train_linear_ensemble(
            X_enhanced, y, X_test_enhanced)

        # Step 4: Optimize final ensemble weights
        print("\n" + "=" * 70)
        print("STEP 4: Optimizing Final Ensemble Weights")
        print("=" * 70)

        combined_preds = np.column_stack([ensemble_oof, linear_oof])
        self.final_weights = self._optimize_weights(combined_preds, y)

        final_oof = ensemble_oof * \
            self.final_weights[0] + linear_oof * self.final_weights[1]
        final_rmse = np.sqrt(mean_squared_error(y, final_oof))

        print(
            f"\nFinal weights: Base Ensemble={self.final_weights[0]:.4f}, Linear={self.final_weights[1]:.4f}")
        print(f"Final CV RMSE: {final_rmse:.5f}")
        print("=" * 70)

        # Compute final test predictions
        final_test = None
        if ensemble_test is not None and linear_test is not None:
            final_test = ensemble_test * \
                self.final_weights[0] + linear_test * self.final_weights[1]

        scores["advanced_ensemble_oof"] = final_rmse

        return scores, final_oof, final_test

    def _train_base_ensemble(self, X: pd.DataFrame, y: np.ndarray) -> Tuple[Dict[str, float], np.ndarray]:
        X_np = X.values if isinstance(X, pd.DataFrame) else X

        oof_lgb = np.zeros(len(X))
        oof_xgb = np.zeros(len(X))
        oof_catboost = np.zeros(len(X))

        lgb_scores = []
        xgb_scores = []
        catboost_scores = []

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

        return {
            "lgb_oof": lgb_oof_score,
            "xgb_oof": xgb_oof_score,
            "catboost_oof": catboost_oof_score,
            "ensemble_oof": ensemble_score,
        }, ensemble_preds

    def _predict_base_ensemble(self, X: pd.DataFrame) -> np.ndarray:
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

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict using the full advanced ensemble

        Args:
            X: Input features

        Returns:
            Final predictions
        """
        # Generate meta-features for prediction
        X_enhanced = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)

        # For prediction, we need to generate meta-features but we don't have y
        # So we skip meta-features in predict and only use base predictions
        # This is a limitation - ideally meta-features should be pre-computed

        # Use base ensemble prediction
        base_pred = self._predict_base_ensemble(X_enhanced)

        # If we don't have final weights trained, return base prediction
        if self.final_weights is None:
            return base_pred

        # Otherwise this would need the full pipeline which requires y for meta-features
        # For now, return base prediction
        return base_pred

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

    scores, _, test_preds = model.fit(X, y, X_test)

    model_save_path = f"{save_path}model_enhanced_ensemble"
    model.save(model_save_path, use_pickle=True)

    submission = pd.DataFrame({"id": test_ids, "score": test_preds})
    submission_path = f"{save_path}submission_enhanced_ensemble.csv"
    submission.to_csv(submission_path, index=False)

    print(f"\nSubmission saved to {submission_path}")
    print("Submission preview:")
    print(submission.head())
    print("\nScore statistics:")
    print(submission["score"].describe())

    return submission

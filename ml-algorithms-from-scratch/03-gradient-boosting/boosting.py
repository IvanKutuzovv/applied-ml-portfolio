from __future__ import annotations

from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import roc_auc_score

from typing import Optional, Tuple, List
from tqdm.auto import tqdm

from sklearn.base import BaseEstimator, ClassifierMixin


class Boosting(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        base_model_class=DecisionTreeRegressor,
        base_model_params: Optional[dict] = None,
        n_estimators: int = 20,
        learning_rate: float = 0.05,
        subsample: float = 1.0,
        bagging_temperature: float = 1.0,
        bootstrap_type: str = 'Bernoulli',
        rsm: float = 1.0,
        goss: bool = False,
        goss_k: float = 0.2,
        dart: bool = False,           
        dropout_rate: float = 0.05,     
        cat_features: Optional[List[int]] = None,
        early_stopping_rounds: int | None = None,
        eval_metric: str | None = None,
        random_state: int | None = None,
        verbose: bool = False,
    ):
        self.base_model_class = base_model_class
        self.base_model_params = base_model_params if base_model_params is not None else {}
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.bagging_temperature = bagging_temperature
        self.bootstrap_type = bootstrap_type
        self.rsm = rsm
        self.goss = goss
        self.goss_k = goss_k
        self.dart = dart
        self.dropout_rate = dropout_rate
        self.cat_features = cat_features
        self.early_stopping_rounds = early_stopping_rounds
        self.eval_metric = eval_metric if eval_metric else 'val_roc_auc'
        self.random_state = random_state
        self.verbose = verbose
        
        self.models = []
        self.gammas = []
        self.feature_indices_list = []
        self.history = defaultdict(list)
        self.classes_ = np.array([0, 1])
        self.n_features = 0
        self.feature_importances_ = None
        self.cat_mappings_ = {} 
        self.global_target_mean_ = 0.0
        self.initial_score_ = 0.0  # Для инициализации (log-odds)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -15, 15)))

    def _loss_fn(self, y, z):
        # y здесь ожидается {-1, 1}
        return -np.log(self._sigmoid(y * z) + 1e-9).mean()

    def _loss_derivative(self, y, z):
        # dl/dz = -y * (1 - sigma(yz))
        return -y * (1 - self._sigmoid(y * z))

    @property
    def _estimator_type(self):
        return "classifier"

    def _get_params_safe(self):
        return self.base_model_params.copy()

    def _get_sample_weights(self, n_samples: int, rng: np.random.Generator) -> np.ndarray:
        if self.bootstrap_type == 'Bernoulli':
            n_samples_subset = int(n_samples * self.subsample)
            sample_indices = rng.choice(n_samples, size=n_samples_subset, replace=False)
            mask = np.zeros(n_samples, dtype=bool)
            mask[sample_indices] = True
            return mask
        elif self.bootstrap_type == 'Bayesian':
            if self.bagging_temperature > 0:
                return rng.gamma(1.0, self.bagging_temperature, size=n_samples)
            return np.ones(n_samples)
        return np.ones(n_samples, dtype=bool)

    def _calculate_feature_importances(self) -> np.ndarray:
        if not self.models or self.n_features == 0:
            return np.array([])
        weighted_importances = np.zeros(self.n_features)
        for model, gamma, feat_idx in zip(self.models, self.gammas, self.feature_indices_list):
            if hasattr(model, 'feature_importances_'):
                # Нормализуем gamma для корректного взвешивания
                weighted_importances[feat_idx] += (self.learning_rate * abs(gamma)) * model.feature_importances_
        total_sum = weighted_importances.sum()
        return weighted_importances / total_sum if total_sum > 0 else weighted_importances

    def _cat_fit(self, X: np.ndarray, y: np.ndarray):
        if not self.cat_features: return
        y_binary = np.where(y == -1, 0, 1)
        self.global_target_mean_ = np.mean(y_binary)
        self.cat_mappings_ = {}
        
        alpha = 10.0 
        
        for feat_idx in self.cat_features:
            values = X[:, feat_idx]
            unique_vals, counts = np.unique(values, return_counts=True)
            self.cat_mappings_[feat_idx] = {}
            for val, count in zip(unique_vals, counts):
                mean_target = np.mean(y_binary[values == val])
                smoothed_val = (mean_target * count + self.global_target_mean_ * alpha) / (count + alpha)
                self.cat_mappings_[feat_idx][val] = smoothed_val

    def _cat_transform(self, X: np.ndarray) -> np.ndarray:
        if not self.cat_features: return X
        X_transformed = X.copy().astype(float)
        for feat_idx, mapping in self.cat_mappings_.items():
            col_vals = X[:, feat_idx]
            encoded_col = np.full(col_vals.shape, self.global_target_mean_)
            for cat_val, encoded_val in mapping.items():
                encoded_col[col_vals == cat_val] = encoded_val
            X_transformed[:, feat_idx] = encoded_col
        return X_transformed

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, eval_set: Tuple[np.ndarray, np.ndarray] | None = None, use_best_model: bool = False):
        self.models, self.gammas, self.feature_indices_list = [], [], []
        self.history = defaultdict(list)
        self.n_features = X_train.shape[1]
        rng = np.random.default_rng(self.random_state)
        
        train_targets = np.where(y_train == 1, 1, -1)
              
        self._cat_fit(X_train, train_targets)
        X_train_proc = self._cat_transform(X_train)
        
        pos_ratio = np.mean(y_train == 1)
        pos_ratio = np.clip(pos_ratio, 1e-6, 1 - 1e-6)
        self.initial_score_ = np.log(pos_ratio / (1 - pos_ratio))
        
        train_preds = np.full(X_train.shape[0], self.initial_score_)

        X_val_proc, val_targets, val_preds = None, None, None
        if eval_set is not None:
            X_val, y_val = eval_set
            X_val_proc = self._cat_transform(X_val)
            val_targets = np.where(y_val == 1, 1, -1)
            val_preds = np.full(X_val.shape[0], self.initial_score_)

        best_score = -np.inf if 'roc_auc' in self.eval_metric else np.inf
        best_iteration = 0
        no_improvement = 0

        for iteration in range(self.n_estimators):
            # DART
            dropped_indices = []
            if self.dart and len(self.models) > 0:
                mask = rng.uniform(0, 1, size=len(self.models)) <= self.dropout_rate
                dropped_indices = np.where(mask)[0]
                if len(dropped_indices) == 0: 
                    dropped_indices = np.array([rng.integers(0, len(self.models))])
                
                current_train_preds = np.full(X_train.shape[0], self.initial_score_)
                keep_indices = [idx for idx in range(len(self.models)) if idx not in dropped_indices]
                for idx in keep_indices:
                    current_train_preds += self.learning_rate * self.gammas[idx] * \
                                           self.models[idx].predict(X_train_proc[:, self.feature_indices_list[idx]])
            else:
                current_train_preds = train_preds

            residuals = -self._loss_derivative(train_targets, current_train_preds)

            # RSM
            n_feat_sub = max(1, int(self.n_features * self.rsm))
            feat_idx = rng.choice(self.n_features, size=n_feat_sub, replace=False)
            
            # GOSS 
            if self.goss:
                abs_res = np.abs(residuals)
                top_n = int(X_train.shape[0] * self.goss_k)
                sorted_idx = np.argsort(abs_res)[::-1]
                top_idx = sorted_idx[:top_n]
                rest_idx = sorted_idx[top_n:]
                
                sampled_rest_n = int(len(rest_idx) * self.subsample)
                if sampled_rest_n > 0:
                    sampled_rest_idx = rng.choice(rest_idx, size=sampled_rest_n, replace=False)
                    final_idx = np.concatenate([top_idx, sampled_rest_idx])
                    w_multiplier = 1.0 / self.subsample 
                else:
                    final_idx = top_idx
                    w_multiplier = 1.0

                sample_weight = np.ones(len(final_idx))
                sample_weight[len(top_idx):] = w_multiplier
                
                X_batch = X_train_proc[final_idx][:, feat_idx]
                res_batch = residuals[final_idx]
            else:
                sw = self._get_sample_weights(X_train.shape[0], rng)
                if self.bootstrap_type == 'Bernoulli':
                    X_batch, res_batch, sample_weight = X_train_proc[sw][:, feat_idx], residuals[sw], None
                else:
                    X_batch, res_batch, sample_weight = X_train_proc[:, feat_idx], residuals, sw

            model = self.base_model_class(**self._get_params_safe())
            model.fit(X_batch, res_batch, sample_weight=sample_weight)
            
            new_tree_preds = model.predict(X_train_proc[:, feat_idx])
            
            gamma = self.find_optimal_gamma(train_targets, current_train_preds, new_tree_preds, residuals)

            # Коррекция Gamma 
            if self.dart and len(dropped_indices) > 0:
                k = len(dropped_indices)
                gamma *= (1.0 / (k + 1))
                for idx in dropped_indices:
                    self.gammas[idx] *= (k / (k + 1))
            
            self.models.append(model)
            self.gammas.append(gamma)
            self.feature_indices_list.append(feat_idx)

            if self.dart:
                train_preds = np.full(X_train.shape[0], self.initial_score_)
                for m, g, f_idx in zip(self.models, self.gammas, self.feature_indices_list):
                    train_preds += self.learning_rate * g * m.predict(X_train_proc[:, f_idx])
            else:
                train_preds += self.learning_rate * gamma * new_tree_preds

            train_auc = roc_auc_score(y_train, self._sigmoid(train_preds))
            self.history["train_roc_auc"].append(train_auc)
            self.history["train_loss"].append(self._loss_fn(train_targets, train_preds))

            if eval_set is not None:
                new_val_tree_preds = model.predict(X_val_proc[:, feat_idx])
                if self.dart:
                     val_preds = np.full(X_val.shape[0], self.initial_score_)
                     for m, g, f_idx in zip(self.models, self.gammas, self.feature_indices_list):
                        val_preds += self.learning_rate * g * m.predict(X_val_proc[:, f_idx])
                else:
                    val_preds += self.learning_rate * gamma * new_val_tree_preds
                
                val_auc = roc_auc_score(y_val, self._sigmoid(val_preds))
                self.history["val_roc_auc"].append(val_auc)
                self.history["val_loss"].append(self._loss_fn(val_targets, val_preds))

                curr_score = self.history[self.eval_metric][-1]
                is_max = 'roc_auc' in self.eval_metric
                if (is_max and curr_score > best_score) or (not is_max and curr_score < best_score):
                    best_score, best_iteration, no_improvement = curr_score, len(self.models), 0
                else:
                    no_improvement += 1
                
                if self.early_stopping_rounds and no_improvement >= self.early_stopping_rounds:
                    if self.verbose: print(f"Stop at {iteration}. Best: {best_iteration}, Score: {best_score}")
                    break

        if use_best_model and eval_set is not None:
            self.models = self.models[:best_iteration]
            self.gammas = self.gammas[:best_iteration]
            self.feature_indices_list = self.feature_indices_list[:best_iteration]

        self.feature_importances_ = self._calculate_feature_importances()
        return self

    def find_optimal_gamma(self, y, old_preds, new_preds, residuals): # оптимизация времени нахождения шага
        
        current_probas = self._sigmoid(old_preds)
        nom = (residuals * new_preds).sum()
        
        denom = (new_preds**2 * current_probas * (1 - current_probas)).sum()
        
        if denom == 0:
            return 0
            
        return nom / (denom + 1e-9)

    def predict_proba(self, X: np.ndarray):
        X_proc = self._cat_transform(X)
        preds = np.full(X.shape[0], self.initial_score_)
        for m, g, f_idx in zip(self.models, self.gammas, self.feature_indices_list):
            preds += self.learning_rate * g * m.predict(X_proc[:, f_idx])
        p = self._sigmoid(preds)
        return np.column_stack([1 - p, p])

    def predict(self, X: np.ndarray):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def score(self, X: np.ndarray, y: np.ndarray):
        return roc_auc_score(y, self.predict_proba(X)[:, 1])

    def plot_history(self, keys: Optional[List[str]] = None):
        if keys is None: keys = list(self.history.keys())
        plt.figure(figsize=(10, 5))
        for key in keys:
            if key in self.history:
                plt.plot(self.history[key], label=key)
        plt.legend(); plt.grid(True); plt.show()
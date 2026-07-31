import numpy as np
from descents import BaseDescent
from dataclasses import dataclass
from enum import auto, Enum
from scipy.sparse.linalg import svds
from typing import Dict, Type, Optional


class LossFunction(Enum):
    MSE = auto()
    MAE = auto()
    LogCosh = auto()
    Huber = auto()

class LinearRegression:
    def __init__(
        self,
        optimizer: Optional[BaseDescent | str] = None,
        l2_coef: float = 0.0,
        tolerance: float = 1e-6,
        max_iter: int = 1000,
        loss_function: LossFunction = LossFunction.MSE
    ):
        self.optimizer = optimizer
        if isinstance(optimizer, BaseDescent):
            self.optimizer.set_model(self)
        self.l2_coef = l2_coef
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.loss_function = loss_function
        self.w = None
        self.X_train = None
        self.y_train = None
        self.loss_history = []


    def fit_analytical(self, X: np.ndarray, y: np.ndarray):
        self.X_train = X
        self.y_train = y

        n_features = X.shape[1]
        I = np.eye(n_features)
        XtX = X.T @ X + self.l2_coef * I
        Xty = X.T @ y
        self.w = np.linalg.solve(XtX, Xty)
        return self

    def fit_svd(self, X: np.ndarray, y: np.ndarray, n_components: Optional[int] = None):
        self.X_train = X
        self.y_train = y
        U, S, Vt = svds(X, k=4)
        S, U, Vt = S[::-1], U[:, ::-1], Vt[::-1, :]
        S_reg = S / (S**2 + self.l2_coef)
        self.w = Vt.T @ np.diag(S_reg) @ U.T @ y
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.w

    def compute_gradients(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        if self.loss_function is LossFunction.MSE:
            N = X.shape[0]
            y_pred = X @ self.w
            grad = (2 / N) * X.T @ (y_pred - y)
            return grad
        return None

    def compute_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        if self.loss_function is LossFunction.MSE:
            N = X.shape[0]
            y_pred = X @ self.w
            loss = np.mean((y_pred - y) ** 2)
            return loss
        return 0.0


    def fit(self, X: np.ndarray, y: np.ndarray):
        self.X_train, self.y_train = X, y

        if self.optimizer is None:
            return self.fit_analytical(X, y)

        if isinstance(self.optimizer, str) and self.optimizer.lower() == "svd":
            return self.fit_svd(X, y)
        
        if self.w is None:
            self.w = np.zeros(X.shape[1])

        if isinstance(self.optimizer, BaseDescent):
            current_loss = self.compute_loss(X, y)
            self.loss_history.append(current_loss)
            
            for _ in range(self.max_iter):
                delta_w = self.optimizer.step()
                new_loss = self.compute_loss(X, y)
                self.loss_history.append(new_loss)
            return self
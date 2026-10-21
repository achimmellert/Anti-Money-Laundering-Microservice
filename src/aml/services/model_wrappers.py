"""
PyTorch Model. Not used in production code yet.
"""

import numpy as np

try:
    import pandas as pd
except ImportError:
    pd = None
from optuna.integration import PyTorchLightningPruningCallback

try:
    from pytorch_tabular import TabularModel
    from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig
    from pytorch_tabular.models import CategoryEmbeddingModelConfig
except ImportError:
    TabularModel = None  # type: ignore
    DataConfig = OptimizerConfig = TrainerConfig = None  # type: ignore
    CategoryEmbeddingModelConfig = None  # type: ignore
from sklearn.base import BaseEstimator, ClassifierMixin


class SklearnPyTorchWrapper(BaseEstimator, ClassifierMixin):
    """Sklearn-compatible wrapper"""

    def __init__(
        self,
        continuous_cols,
        categorical_cols=None,
        layers="64-32",
        dropout=0.2,
        learning_rate=1e-3,
        epochs=15,
        batch_size=512,
        trial=None,
    ):
        self.continuous_cols = continuous_cols
        self.categorical_cols = categorical_cols if categorical_cols is not None else []
        self.layers = layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.trial = trial
        self.tabular_model = None
        self.target_col = "_target"

    def _prepare_df(self, X, y=None):
        all_cols = self.continuous_cols + self.categorical_cols
        if hasattr(X, "to_pandas"):
            df = X.to_pandas()
        elif isinstance(X, np.ndarray):
            df = pd.DataFrame(X, columns=all_cols)
        else:
            df = X.copy()

        if y is not None:
            if hasattr(y, "to_numpy"):
                df[self.target_col] = y.to_numpy()
            elif hasattr(y, "values"):
                df[self.target_col] = y.values
            else:
                df[self.target_col] = y
        return df

    def fit(self, X, y, X_val=None, y_val=None):
        train_df = self._prepare_df(X, y)
        val_df = self._prepare_df(X_val, y_val) if X_val is not None else None

        callbacks = []
        if self.trial and val_df is not None:
            callbacks.append(PyTorchLightningPruningCallback(self.trial, monitor="valid_loss"))

        data_config = DataConfig(
            target=[self.target_col],
            continuous_cols=self.continuous_cols,
            categorical_cols=self.categorical_cols,
            normalize_continuous_features=True,
            continuous_feature_transform="yeo-johnson",
        )

        model_config = CategoryEmbeddingModelConfig(
            task="classification",
            layers=self.layers,
            activation="ReLU",
            dropout=self.dropout,
            learning_rate=self.learning_rate,
        )

        trainer_config = TrainerConfig(
            batch_size=self.batch_size,
            max_epochs=self.epochs,
            callbacks=callbacks,
            accelerator="gpu",
            devices=1,
            progress_bar="none",
            early_stopping="valid_loss" if val_df is not None else None,
            early_stopping_patience=5 if val_df is not None else 0,
        )

        self.tabular_model = TabularModel(
            data_config=data_config,
            model_config=model_config,
            optimizer_config=OptimizerConfig(),
            trainer_config=trainer_config,
            verbose=False,
        )

        self.tabular_model.fit(train=train_df, validation=val_df)
        return self

    def predict_proba(self, X):
        test_df = self._prepare_df(X)
        pred_df = self.tabular_model.predict(test_df)
        prob_cols = sorted([col for col in pred_df.columns if "probability" in col])
        return pred_df[prob_cols].values

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

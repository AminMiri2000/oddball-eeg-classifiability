"""Classifiers: shrinkage LDA, logistic regression and EEGNet."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ============================================================
# Type aliases
# ============================================================

ClassicalModelName = Literal["lda", "logreg"]
AnyModelName = Literal["lda", "logreg", "eegnet"]


# ============================================================
# Classical baselines
# ============================================================


def build_lda() -> Pipeline:
    """
    Build the LDA baseline used throughout the thesis pipeline.

    Notes
    -----
    - StandardScaler is used before LDA for consistency with the benchmark pipeline
    - shrinkage='auto' with solver='lsqr' is a strong and stable EEG baseline
    """
    return Pipeline(
        [
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            (
                "clf",
                LinearDiscriminantAnalysis(
                    solver="lsqr",
                    shrinkage="auto",
                ),
            ),
        ]
    )


def build_logreg(max_iter: int = 2000) -> Pipeline:
    """
    Build the Logistic Regression baseline used throughout the thesis pipeline.

    Notes
    -----
    - StandardScaler is applied before the classifier
    - class_weight='balanced' helps with class imbalance
    """
    return Pipeline(
        [
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            (
                "clf",
                LogisticRegression(
                    solver="lbfgs",
                    max_iter=int(max_iter),
                    class_weight="balanced",
                ),
            ),
        ]
    )


# ============================================================
# EEGNet
# ============================================================


class EEGNet(nn.Module):
    """
    Canonical pooled EEGNet-style model for stimulus classification.

    Expected input
    --------------
    x : torch.Tensor
        Shape (N, 1, C, T)

    Architecture
    ------------
    Block 1:
        temporal conv -> batch norm -> depthwise spatial conv
        -> batch norm -> ELU -> avg pool -> dropout

    Block 2:
        depthwise temporal conv -> pointwise conv
        -> batch norm -> ELU -> avg pool -> dropout

    Classifier:
        flatten -> linear

    Design goals
    ------------
    - Works with cropped windows such as 0.0 to 0.6 s
    - Uses safe dynamic pooling so short control windows do not crash
    - Matches the model family that performed best in your benchmark
    """

    def __init__(
        self,
        n_ch: int,
        n_times: int,
        n_classes: int = 2,
        f1: int = 8,
        d: int = 2,
        f2: int = 16,
        k1: int = 32,
        k2: int = 16,
        dropout: float = 0.25,
    ):
        super().__init__()

        if n_ch <= 0:
            raise ValueError(f"n_ch must be positive, got {n_ch}")
        if n_times <= 0:
            raise ValueError(f"n_times must be positive, got {n_times}")
        if n_classes <= 1:
            raise ValueError(f"n_classes must be >= 2, got {n_classes}")
        if f1 <= 0 or d <= 0 or f2 <= 0:
            raise ValueError(f"Invalid channels: f1={f1}, d={d}, f2={f2} must all be positive")
        if k1 <= 0 or k2 <= 0:
            raise ValueError(f"Kernel sizes must be positive, got k1={k1}, k2={k2}")

        # ------------------------------------------------------------
        # Safe dynamic pooling
        # ------------------------------------------------------------
        # Main benchmark window (0.0–0.6 s) should still behave as intended,
        # while shorter windows (e.g. pre-stim only) should not crash.
        pool1_t = 4
        t_after_pool1 = max(1, n_times // pool1_t)
        pool2_t = min(8, t_after_pool1)

        self.n_ch = n_ch
        self.n_times = n_times
        self.n_classes = n_classes

        self.f1 = f1
        self.d = d
        self.f2 = f2
        self.k1 = k1
        self.k2 = k2
        self.dropout_p = dropout

        self.pool1_t = pool1_t
        self.pool2_t = pool2_t

        # ------------------------------------------------------------
        # Block 1: temporal conv + depthwise spatial conv
        # ------------------------------------------------------------
        self.block1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=f1,
                kernel_size=(1, k1),
                padding=(0, k1 // 2),
                bias=False,
            ),
            nn.BatchNorm2d(f1),
            nn.Conv2d(
                in_channels=f1,
                out_channels=f1 * d,
                kernel_size=(n_ch, 1),
                groups=f1,
                bias=False,
            ),
            nn.BatchNorm2d(f1 * d),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, pool1_t)),
            nn.Dropout(dropout),
        )

        # ------------------------------------------------------------
        # Block 2: separable temporal conv
        # ------------------------------------------------------------
        self.block2 = nn.Sequential(
            nn.Conv2d(
                in_channels=f1 * d,
                out_channels=f1 * d,
                kernel_size=(1, k2),
                padding=(0, k2 // 2),
                groups=f1 * d,
                bias=False,
            ),
            nn.Conv2d(
                in_channels=f1 * d,
                out_channels=f2,
                kernel_size=(1, 1),
                bias=False,
            ),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, pool2_t)),
            nn.Dropout(dropout),
        )

        # ------------------------------------------------------------
        # Infer classifier size dynamically
        # ------------------------------------------------------------
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_ch, n_times)
            z = self.block1(dummy)
            z = self.block2(z)
            feat_dim = z.numel()

        self.feat_dim = int(feat_dim)
        self.classifier = nn.Linear(self.feat_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape (N, 1, C, T)

        Returns
        -------
        logits : torch.Tensor
            Shape (N, n_classes)
        """
        if x.ndim != 4:
            raise ValueError(f"Expected input shape (N, 1, C, T), got {tuple(x.shape)}")
        if x.shape[1] != 1:
            raise ValueError(f"Expected input channel dimension = 1, got {x.shape[1]}")
        if x.shape[2] != self.n_ch:
            raise ValueError(f"Expected channel dimension {self.n_ch}, got {x.shape[2]}")

        x = self.block1(x)
        x = self.block2(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x

    def summary_dict(self) -> dict:
        """
        Return a lightweight model summary for logging/debugging.
        """
        return {
            "model_name": "EEGNet",
            "n_ch": self.n_ch,
            "n_times": self.n_times,
            "n_classes": self.n_classes,
            "f1": self.f1,
            "d": self.d,
            "f2": self.f2,
            "k1": self.k1,
            "k2": self.k2,
            "dropout": self.dropout_p,
            "pool1_t": self.pool1_t,
            "pool2_t": self.pool2_t,
            "feat_dim": self.feat_dim,
        }


# ============================================================
# Model factory helpers
# ============================================================


def get_classical_model(name: ClassicalModelName):
    """
    Return a classical baseline model by name.
    """
    name = str(name).lower()

    if name == "lda":
        return build_lda()
    if name == "logreg":
        return build_logreg()

    raise ValueError(f"Unknown classical model '{name}'. Valid options: 'lda', 'logreg'")


def get_model(
    name: AnyModelName,
    *,
    n_ch: int | None = None,
    n_times: int | None = None,
    n_classes: int = 2,
    eegnet_kwargs: dict | None = None,
):
    """
    Generic model factory.

    Parameters
    ----------
    name : {"lda", "logreg", "eegnet"}
        Model name

    n_ch : int, optional
        Required for EEGNet

    n_times : int, optional
        Required for EEGNet

    n_classes : int
        Number of classes for EEGNet

    eegnet_kwargs : dict, optional
        Additional kwargs passed to EEGNet

    Returns
    -------
    model
        sklearn Pipeline for classical models
        torch.nn.Module for EEGNet
    """
    name = str(name).lower()

    if name in {"lda", "logreg"}:
        return get_classical_model(name)

    if name == "eegnet":
        if n_ch is None or n_times is None:
            raise ValueError("EEGNet requires both n_ch and n_times")

        eegnet_kwargs = {} if eegnet_kwargs is None else dict(eegnet_kwargs)

        # Avoid passing n_classes twice
        if "n_classes" in eegnet_kwargs:
            n_classes = eegnet_kwargs.pop("n_classes")

        return EEGNet(
            n_ch=n_ch,
            n_times=n_times,
            n_classes=n_classes,
            **eegnet_kwargs,
        )

    raise ValueError(f"Unknown model '{name}'. Valid options: 'lda', 'logreg', 'eegnet'")


# ============================================================
# Utility helpers
# ============================================================


def count_trainable_parameters(model: nn.Module) -> int:
    """
    Count trainable parameters in a PyTorch model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_eegnet_summary(model: EEGNet) -> None:
    """
    Print a compact human-readable EEGNet summary.
    """
    info = model.summary_dict()
    print("=" * 70)
    print("EEGNet summary")
    for key, value in info.items():
        print(f"{key:>12s} : {value}")
    print(f"{'n_params':>12s} : {count_trainable_parameters(model)}")

"""Model evaluation on precomputed folds.

* :func:`evaluate_classical_benchmark` – LDA / logistic regression per subject.
* :func:`evaluate_eegnet_benchmark` – EEGNet with a group-aware inner validation
  split, class-weighted loss and early stopping on validation AUC."""

from __future__ import annotations

from typing import Dict, Iterable, Literal, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.base import ClassifierMixin

from oddball_eeg.features import make_classical_features
from oddball_eeg.loader import LoadedDataset
from oddball_eeg.metrics import classification_metric_dict
from oddball_eeg.models import get_model
from oddball_eeg.preprocessing import (
    SplitArrays,
    build_dataloaders,
    build_datasets_from_split,
    make_class_weight_tensor,
)
from oddball_eeg.reproducibility import seed_everything
from oddball_eeg.splits import make_group_train_val_split

ClassicalModelName = Literal["lda", "logreg"]


def evaluate_classical_model_on_folds(
    X: np.ndarray,
    y: np.ndarray,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
    model_name: ClassicalModelName,
    subject: str | None = None,
    seed: int | None = None,
    feature_mode: str = "raw_flat",
    times: np.ndarray | None = None,
    sfreq: float | None = None,
    bin_ms: int = 50,
) -> pd.DataFrame:
    """
    Evaluate a classical baseline on precomputed grouped folds.

    Parameters
    ----------
    X : np.ndarray
        Shape (N, C, T)

    y : np.ndarray
        Shape (N,)

    folds : sequence of (train_idx, test_idx)

    model_name : {"lda", "logreg"}

    subject : str, optional
        Included in the output table for convenience

    seed : int, optional
        Included in the output table for convenience

    Returns
    -------
    pd.DataFrame
        One row per fold
    """
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y).astype(int)

    clf: ClassifierMixin = get_model(model_name)

    rows = []

    for fold_id, (tr_idx, te_idx) in enumerate(folds, start=1):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        F_tr = make_classical_features(
            X_tr,
            feature_mode=feature_mode,
            times=times,
            sfreq=sfreq,
            bin_ms=bin_ms,
        )
        F_te = make_classical_features(
            X_te,
            feature_mode=feature_mode,
            times=times,
            sfreq=sfreq,
            bin_ms=bin_ms,
        )

        clf.fit(F_tr, y_tr)

        if hasattr(clf, "predict_proba"):
            scores = clf.predict_proba(F_te)[:, 1]
        else:
            # Fallback if a future model does not provide predict_proba
            scores = clf.decision_function(F_te)

        y_pred = clf.predict(F_te)

        row = {
            "subject": subject,
            "seed": seed,
            "fold": int(fold_id),
            "model": str(model_name),
            "n_train": int(len(tr_idx)),
            "n_test": int(len(te_idx)),
            "train_pos_rate": float(y_tr.mean()),
            "test_pos_rate": float(y_te.mean()),
            "feature_mode": feature_mode,
            "bin_ms": int(bin_ms) if feature_mode == "time_binned" else np.nan,
        }
        row.update(classification_metric_dict(y_te, scores, y_pred))
        rows.append(row)

    return pd.DataFrame(rows)


def evaluate_classical_benchmark(
    data: LoadedDataset,
    folds_by_sub: Dict[str, list[tuple[np.ndarray, np.ndarray]]],
    model_names: Sequence[ClassicalModelName] = ("lda", "logreg"),
    seed: int | None = None,
    verbose: bool = True,
    feature_mode: str = "raw_flat",
    bin_ms: int = 50,
) -> pd.DataFrame:
    """
    Run classical baselines across all subjects in a LoadedDataset.
    """
    all_rows = []

    for sub in data.subjects:
        X = data.X_by_sub[sub]
        y = data.y_by_sub[sub]
        folds = folds_by_sub[sub]

        for model_name in model_names:
            if verbose:
                print(f"[CLASSICAL] subject={sub} model={model_name}")

            df_sub = evaluate_classical_model_on_folds(
                X=X,
                y=y,
                folds=folds,
                model_name=model_name,
                subject=sub,
                seed=seed,
                feature_mode=feature_mode,
                times=data.times,
                sfreq=data.sfreq,
                bin_ms=bin_ms,
            )
            all_rows.append(df_sub)

    if not all_rows:
        raise RuntimeError("No classical benchmark results were produced.")

    return pd.concat(all_rows, ignore_index=True)


def evaluate_eegnet_loader(
    model: nn.Module,
    loader: Iterable,
    device: torch.device | str,
) -> dict:
    """
    Evaluate EEGNet on a dataloader.

    The dataset is expected to return x with shape (C, T), so we insert the
    singleton image-like channel dimension before forwarding:
        (N, C, T) -> (N, 1, C, T)
    """
    model.eval()

    all_y = []
    all_prob = []
    all_loss = []

    ce_plain = nn.CrossEntropyLoss()

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.unsqueeze(1).to(device)  # (N, 1, C, T)
            yb = yb.to(device)

            logits = model(xb)
            loss = ce_plain(logits, yb)

            prob1 = torch.softmax(logits, dim=1)[:, 1]

            all_y.append(yb.detach().cpu().numpy())
            all_prob.append(prob1.detach().cpu().numpy())
            all_loss.append(float(loss.item()))

    y_true = np.concatenate(all_y).astype(int)
    prob = np.concatenate(all_prob)

    # For binary classification with 2 logits, argmax is equivalent to prob > 0.5
    y_pred = (prob >= 0.5).astype(int)

    out = {
        "loss": float(np.mean(all_loss)) if len(all_loss) > 0 else float("nan"),
        "y_true": y_true,
        "prob": prob,
        "y_pred": y_pred,
    }
    out.update(classification_metric_dict(y_true, prob, y_pred))
    return out


def train_eegnet_one_fold(
    X_train_fold: np.ndarray,
    y_train_fold: np.ndarray,
    groups_train_fold: np.ndarray,
    X_test_fold: np.ndarray,
    y_test_fold: np.ndarray,
    *,
    device: torch.device | str = "cpu",
    normalize: bool = True,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    epochs: int = 60,
    patience: int = 10,
    val_size: float = 0.15,
    val_n_tries: int = 200,
    use_class_weight: bool = True,
    grad_clip_norm: float = 1.0,
    eegnet_kwargs: dict | None = None,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """
    Train and evaluate EEGNet on one grouped outer fold.
    """
    seed_everything(seed)

    tr_sub_idx, val_sub_idx = make_group_train_val_split(
        y_train=y_train_fold,
        groups_train=groups_train_fold,
        val_size=val_size,
        seed=seed,
        n_tries=val_n_tries,
    )

    split = SplitArrays(
        X_train=X_train_fold[tr_sub_idx],
        y_train=y_train_fold[tr_sub_idx],
        X_val=X_train_fold[val_sub_idx],
        y_val=y_train_fold[val_sub_idx],
        X_test=X_test_fold,
        y_test=y_test_fold,
    )

    ds_train, ds_val, ds_test, _norm_stats = build_datasets_from_split(
        split,
        normalize=normalize,
    )

    train_loader, val_loader, test_loader = build_dataloaders(
        ds_train=ds_train,
        ds_val=ds_val,
        ds_test=ds_test,
        batch_size=batch_size,
        device=device,
    )

    eegnet_kwargs = {} if eegnet_kwargs is None else dict(eegnet_kwargs)

    model: nn.Module = get_model(
        "eegnet",
        n_ch=X_train_fold.shape[1],
        n_times=X_train_fold.shape[2],
        eegnet_kwargs=eegnet_kwargs,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    if use_class_weight:
        class_weight = make_class_weight_tensor(split.y_train, device=device)
        ce = nn.CrossEntropyLoss(weight=class_weight)
    else:
        ce = nn.CrossEntropyLoss()

    history = []
    best_val_auc = -np.inf
    best_epoch = None
    best_state = None
    patience_left = patience

    for ep in range(1, epochs + 1):
        model.train()
        batch_losses = []

        for xb, yb in train_loader:
            xb = xb.unsqueeze(1).to(device)  # (N, 1, C, T)
            yb = yb.to(device)

            logits = model(xb)
            loss = ce(logits, yb)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            if grad_clip_norm is not None and grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

            optimizer.step()
            batch_losses.append(float(loss.item()))

        train_loss = float(np.mean(batch_losses)) if len(batch_losses) > 0 else float("nan")
        val_metrics = evaluate_eegnet_loader(model, val_loader, device=device)

        row = {
            "epoch": int(ep),
            "train_loss": train_loss,
            "val_loss": float(val_metrics["loss"]),
            "val_auc": float(val_metrics["auc"]),
            "val_pr_auc": float(val_metrics["pr_auc"]),
            "val_bal_acc": float(val_metrics["bal_acc"]),
        }
        history.append(row)

        if verbose and (ep == 1 or ep % 5 == 0 or ep == epochs):
            print(
                f"epoch={ep:03d}  "
                f"train_loss={train_loss:.4f}  "
                f"val_loss={val_metrics['loss']:.4f}  "
                f"val_auc={val_metrics['auc']:.4f}"
            )

        if np.isfinite(val_metrics["auc"]) and val_metrics["auc"] > best_val_auc + 1e-4:
            best_val_auc = float(val_metrics["auc"])
            best_epoch = int(ep)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                if verbose:
                    print(f"Early stopping at epoch {ep}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = evaluate_eegnet_loader(model, test_loader, device=device)

    return {
        "model": model,
        "history_df": pd.DataFrame(history),
        "best_epoch": best_epoch,
        "best_val_auc": float(best_val_auc),
        "test_metrics": test_metrics,
    }


def evaluate_eegnet_subject(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    subject: str,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    device: torch.device | str = "cpu",
    normalize: bool = True,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    epochs: int = 60,
    patience: int = 10,
    val_size: float = 0.15,
    val_n_tries: int = 200,
    use_class_weight: bool = True,
    grad_clip_norm: float = 1.0,
    eegnet_kwargs: dict | None = None,
    seed: int = 42,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """
    Evaluate EEGNet across all folds for one subject.
    """
    fold_rows = []
    histories = {}

    for fold_id, (tr_idx, te_idx) in enumerate(folds, start=1):
        if verbose:
            print("=" * 70)
            print(f"Subject {subject} | Fold {fold_id}/{len(folds)}")

        result = train_eegnet_one_fold(
            X_train_fold=X[tr_idx],
            y_train_fold=y[tr_idx],
            groups_train_fold=groups[tr_idx],
            X_test_fold=X[te_idx],
            y_test_fold=y[te_idx],
            device=device,
            normalize=normalize,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            epochs=epochs,
            patience=patience,
            val_size=val_size,
            val_n_tries=val_n_tries,
            use_class_weight=use_class_weight,
            grad_clip_norm=grad_clip_norm,
            eegnet_kwargs=eegnet_kwargs,
            seed=seed + fold_id,
            verbose=verbose,
        )

        test_metrics = result["test_metrics"]

        row = {
            "subject": subject,
            "seed": seed,
            "fold": int(fold_id),
            "model": "eegnet",
            "best_epoch": result["best_epoch"],
            "best_val_auc": result["best_val_auc"],
            "n_train": int(len(tr_idx)),
            "n_test": int(len(te_idx)),
            "train_pos_rate": float(y[tr_idx].mean()),
            "test_pos_rate": float(y[te_idx].mean()),
            "auc": float(test_metrics["auc"]),
            "pr_auc": float(test_metrics["pr_auc"]),
            "acc": float(test_metrics["acc"]),
            "bal_acc": float(test_metrics["bal_acc"]),
            "f1": float(test_metrics["f1"]),
            "tn": int(test_metrics["tn"]),
            "fp": int(test_metrics["fp"]),
            "fn": int(test_metrics["fn"]),
            "tp": int(test_metrics["tp"]),
        }
        fold_rows.append(row)
        histories[fold_id] = result["history_df"]

    return pd.DataFrame(fold_rows), histories


def evaluate_eegnet_benchmark(
    data: LoadedDataset,
    folds_by_sub: Dict[str, list[tuple[np.ndarray, np.ndarray]]],
    *,
    device: torch.device | str = "cpu",
    normalize: bool = True,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    epochs: int = 60,
    patience: int = 10,
    val_size: float = 0.15,
    val_n_tries: int = 200,
    use_class_weight: bool = True,
    grad_clip_norm: float = 1.0,
    eegnet_kwargs: dict | None = None,
    seed: int = 42,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """
    Run EEGNet across all subjects in a LoadedDataset.
    """
    all_results = []
    all_histories = {}

    for sub in data.subjects:
        X = data.X_by_sub[sub]
        y = data.y_by_sub[sub]
        groups = data.groups_by_sub[sub]
        folds = folds_by_sub[sub]

        df_sub, histories_sub = evaluate_eegnet_subject(
            X=X,
            y=y,
            groups=groups,
            subject=sub,
            folds=folds,
            device=device,
            normalize=normalize,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            epochs=epochs,
            patience=patience,
            val_size=val_size,
            val_n_tries=val_n_tries,
            use_class_weight=use_class_weight,
            grad_clip_norm=grad_clip_norm,
            eegnet_kwargs=eegnet_kwargs,
            seed=seed,
            verbose=verbose,
        )

        all_results.append(df_sub)
        all_histories[sub] = histories_sub

    if not all_results:
        raise RuntimeError("No EEGNet benchmark results were produced.")

    return pd.concat(all_results, ignore_index=True), all_histories

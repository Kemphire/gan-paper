"""Leakage-safe SMOTE feature selection with GAN augmentation comparisons."""

from __future__ import annotations

import random
import time
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import torch
from imblearn.over_sampling import ADASYN, SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from pycaret.classification import ClassificationExperiment
from sklearn.base import clone
from sklearn.feature_selection import RFECV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

from smote_gan_final import f1_g, f1_sg, hellingerDistance


TARGET_COLUMN = "Class"
MODEL_IDS = ("lr", "svm", "ridge", "dt", "rf", "ada", "gbc", "et", "xgboost")
N_RUNS = 30
N_FOLDS = 5
MIN_FEATURES = 10
CORRELATION_THRESHOLD = 0.8
GAN_LEARNING_RATE = 0.0002
GAN_EPOCHS = 150
GAN_BATCH_SIZE = 128


def _as_frame(value, columns=None):
    if isinstance(value, pd.DataFrame):
        return value.reset_index(drop=True)
    if columns is None:
        columns = pd.RangeIndex(np.asarray(value).shape[1])
    return pd.DataFrame(value, columns=columns).reset_index(drop=True)


def _as_series(value):
    return pd.Series(value).reset_index(drop=True).astype(int)


def _load_binary_dataset(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path).dropna(axis=0).copy()
    if TARGET_COLUMN not in data:
        raise ValueError(f"{path.name} must contain a {TARGET_COLUMN!r} target column.")
    if data[TARGET_COLUMN].nunique() != 2:
        raise ValueError(f"{path.name} must have exactly two target classes.")
    # Existing GAN helpers operate on labels 0 and 1.
    data[TARGET_COLUMN] = pd.factorize(data[TARGET_COLUMN], sort=True)[0]
    return data


def _pycaret_split(data: pd.DataFrame, seed: int):
    """Fit PyCaret preprocessing only on the 80% training partition."""
    experiment = ClassificationExperiment(
        target=TARGET_COLUMN,
        train_size=0.8,
        normalize=True,
        session_id=seed,
        fold=N_FOLDS,
        verbose=False,
    )
    experiment.fit(data)
    X_train = _as_frame(experiment.get_config("X_train_transformed"))
    X_test = _as_frame(
        experiment.preprocess_pipeline.transform(experiment.X_test), X_train.columns
    )
    y_train = _as_series(experiment.get_config("y_train"))
    y_test = _as_series(experiment.get_config("y_test"))
    if not X_train.columns.equals(X_test.columns):
        raise ValueError(
            "PyCaret produced non-aligned transformed train/test features."
        )
    return X_train, X_test, y_train, y_test, experiment


def _create_estimator(experiment, model_id: str):
    """Obtain PyCaret's estimator while keeping fitting under our CV pipelines."""
    result = experiment.create_model(model_id, verbose=False)
    estimator = result.pipeline.steps[-1][1]
    # RFECV requires importances.  The PyCaret SVM is normally RBF, which has
    # none; a linear SVM keeps the requested model family and permits RFECV.
    if model_id == "svm" and "kernel" in estimator.get_params(deep=False):
        estimator.set_params(kernel="linear")
    return clone(estimator)


def _importance_getter(fitted_pipeline):
    classifier = fitted_pipeline.named_steps["classifier"]
    if hasattr(classifier, "coef_"):
        coefficients = np.asarray(classifier.coef_)
        return (
            np.abs(coefficients)
            if coefficients.ndim == 1
            else np.mean(np.abs(coefficients), axis=0)
        )
    if hasattr(classifier, "feature_importances_"):
        return classifier.feature_importances_
    raise ValueError(
        f"{type(classifier).__name__} does not expose RFECV feature importances."
    )


def select_model_and_features(experiment, X_train: pd.DataFrame, y_train: pd.Series):
    """Run RFECV with SMOTE inside each fold, then choose by CV accuracy."""
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    min_features = min(MIN_FEATURES, X_train.shape[1])
    records = []
    for model_id in MODEL_IDS:
        try:
            estimator = _create_estimator(experiment, model_id)
            pipeline = ImbPipeline(
                [("smote", SMOTE(random_state=42)), ("classifier", estimator)]
            )
            selector = RFECV(
                estimator=pipeline,
                step=1,
                cv=cv,
                scoring="accuracy",
                min_features_to_select=min_features,
                importance_getter=_importance_getter,
            )
            selector.fit(X_train, y_train)
            selected = X_train.columns[selector.support_].tolist()
            scores = cross_val_score(
                ImbPipeline(
                    [
                        ("smote", SMOTE(random_state=42)),
                        ("classifier", clone(estimator)),
                    ]
                ),
                X_train[selected],
                y_train,
                cv=cv,
                scoring="accuracy",
            )
            records.append(
                {
                    "Model": model_id,
                    "CV_Accuracy_Mean": float(scores.mean()),
                    "CV_Accuracy_Std": float(scores.std()),
                    "Number_of_Features": len(selected),
                    "Selected_Features": selected,
                    "Estimator": estimator,
                    "Status": "ok",
                }
            )
        except Exception as exc:
            records.append({"Model": model_id, "Status": "failed", "Error": str(exc)})
    successful = [record for record in records if record["Status"] == "ok"]
    if not successful:
        raise RuntimeError("Feature selection failed for every requested model.")
    return max(successful, key=lambda record: record["CV_Accuracy_Mean"]), records


def _metrics(name, estimator, X_train, y_train, X_test, y_test):
    fitted = clone(estimator).fit(X_train, y_train)
    prediction = fitted.predict(X_test)
    return {
        "Method": name,
        "Accuracy": accuracy_score(y_test, prediction),
        "F1_Weighted": f1_score(
            y_test, prediction, average="weighted", zero_division=0
        ),
        "Precision_Weighted": precision_score(
            y_test, prediction, average="weighted", zero_division=0
        ),
        "Recall_Weighted": recall_score(
            y_test, prediction, average="weighted", zero_division=0
        ),
    }


def _minority_label(y_train: pd.Series) -> int:
    return int(min(Counter(y_train).items(), key=lambda item: item[1])[0])


def compare_augmentations(estimator, X_train, y_train, X_test, y_test):
    """Score all methods using one selected estimator and one feature subset."""
    start = time.perf_counter()
    X_smote, y_smote = SMOTE(random_state=42).fit_resample(X_train, y_train)
    smote_ms = (time.perf_counter() - start) * 1000
    X_smote, y_smote = _as_frame(X_smote, X_train.columns), _as_series(y_smote)
    minority, majority = _minority_label(y_train), 0
    majority = int(1 - minority)
    real_minority = X_train.loc[y_train == minority].to_numpy()
    synthetic_smote = X_smote.iloc[len(X_train) :].to_numpy()
    methods = [
        _metrics("Raw", estimator, X_train, y_train, X_test, y_test),
        _metrics("SMOTE", estimator, X_smote, y_smote, X_test, y_test),
    ]
    diagnostics = {
        "Original_Class_Distribution": str(dict(Counter(y_train))),
        "SMOTE_Class_Distribution": str(dict(Counter(y_smote))),
        "Synthetic_Samples": len(synthetic_smote),
        "SMOTE_Time_ms": smote_ms,
        "SMOTE_Hellinger": hellingerDistance(real_minority, synthetic_smote),
    }
    device = torch.device("cpu")
    try:
        smote_tensor = torch.from_numpy(synthetic_smote).float().to(device)
        start = time.perf_counter()
        generator_sg = f1_sg(
            X_train,
            y_train,
            X_smote,
            y_smote,
            real_minority,
            np.ones(len(real_minority)),
            smote_tensor,
            device,
            GAN_LEARNING_RATE,
            GAN_EPOCHS,
            GAN_BATCH_SIZE,
            minority,
            majority,
        )
        generated_sg = generator_sg(smote_tensor).detach().cpu().numpy()
        diagnostics["SMOTified_GAN_Time_ms"] = (time.perf_counter() - start) * 1000
        diagnostics["SMOTified_GAN_Hellinger"] = hellingerDistance(
            real_minority, generated_sg
        )
        X_sg = pd.DataFrame(
            np.vstack((X_train.to_numpy(), generated_sg)), columns=X_train.columns
        )
        methods.append(
            _metrics("SMOTified_GAN", estimator, X_sg, y_smote, X_test, y_test)
        )

        start = time.perf_counter()
        generator_g = f1_g(
            X_train,
            y_train,
            X_smote,
            y_smote,
            real_minority,
            np.ones(len(real_minority)),
            device,
            GAN_LEARNING_RATE,
            GAN_EPOCHS,
            GAN_BATCH_SIZE,
            minority,
            majority,
        )
        noise = torch.randn((len(synthetic_smote), X_train.shape[1]), device=device)
        generated_g = generator_g(noise).detach().cpu().numpy()
        diagnostics["GAN_Time_ms"] = (time.perf_counter() - start) * 1000
        diagnostics["GAN_Hellinger"] = hellingerDistance(real_minority, generated_g)
        X_g = pd.DataFrame(
            np.vstack((X_train.to_numpy(), generated_g)), columns=X_train.columns
        )
        methods.append(_metrics("GAN", estimator, X_g, y_smote, X_test, y_test))
    except Exception as exc:
        diagnostics["GAN_Error"] = str(exc)
    try:
        start = time.perf_counter()
        X_ada, y_ada = ADASYN(sampling_strategy=0.95, random_state=42).fit_resample(
            X_train, y_train
        )
        diagnostics["ADASYN_Time_ms"] = (time.perf_counter() - start) * 1000
        X_ada, y_ada = _as_frame(X_ada, X_train.columns), _as_series(y_ada)
        methods.append(_metrics("ADASYN", estimator, X_ada, y_ada, X_test, y_test))
        synthetic_ada = X_ada.iloc[len(X_train) :].to_numpy()
        ada_tensor = torch.from_numpy(synthetic_ada).float().to(device)
        start = time.perf_counter()
        generator_ag = f1_sg(
            X_train,
            y_train,
            X_ada,
            y_ada,
            real_minority,
            np.ones(len(real_minority)),
            ada_tensor,
            device,
            GAN_LEARNING_RATE,
            GAN_EPOCHS,
            GAN_BATCH_SIZE,
            minority,
            majority,
        )
        generated_ag = generator_ag(ada_tensor).detach().cpu().numpy()
        diagnostics["ADASYN_GAN_Time_ms"] = (time.perf_counter() - start) * 1000
        diagnostics["ADASYN_Hellinger"] = hellingerDistance(
            real_minority, synthetic_ada
        )
        diagnostics["ADASYN_GAN_Hellinger"] = hellingerDistance(
            real_minority, generated_ag
        )
        X_ag = pd.DataFrame(
            np.vstack((X_train.to_numpy(), generated_ag)), columns=X_train.columns
        )
        methods.append(_metrics("ADASYN_GAN", estimator, X_ag, y_ada, X_test, y_test))
    except Exception as exc:
        diagnostics["ADASYN_Error"] = str(exc)
    return methods, diagnostics


def write_inference_artifacts(X_train, y_train, selected, output_dir: Path, stem: str):
    rows = []
    for feature in selected:
        try:
            fitted = sm.Logit(y_train, sm.add_constant(X_train[[feature]])).fit(disp=0)
            rows.append(
                {
                    "Feature": feature,
                    "Single_Feature_Coeff": fitted.params[feature],
                    "Single_Feature_pval": fitted.pvalues[feature],
                }
            )
        except Exception as exc:
            rows.append({"Feature": feature, "Inference_Error": str(exc)})
    inference = pd.DataFrame(rows)
    try:
        fitted = sm.Logit(y_train, sm.add_constant(X_train[selected])).fit(disp=0)
        all_features = pd.DataFrame(
            {
                "Feature": fitted.params.index,
                "All_Features_Coeff": fitted.params.values,
                "All_Features_pval": fitted.pvalues.values,
            }
        )
        inference = inference.merge(
            all_features[all_features.Feature != "const"], on="Feature", how="left"
        )
    except Exception as exc:
        inference["All_Features_Inference_Error"] = str(exc)
    inference.to_csv(output_dir / f"{stem}_inference.csv", index=False)
    correlation = X_train[selected].corr(method="pearson")
    correlation.to_csv(output_dir / f"{stem}_correlation.csv")
    pairs = (
        correlation.where(np.triu(np.ones(correlation.shape), k=1).astype(bool))
        .stack()
        .reset_index()
    )
    pairs.columns = ["Feature1", "Feature2", "Correlation"]
    pairs.loc[pairs.Correlation.abs() > CORRELATION_THRESHOLD].to_csv(
        output_dir / f"{stem}_strong_correlations.csv", index=False
    )
    plt.figure(figsize=(max(8, len(selected)), max(6, len(selected) * 0.75)))
    sns.heatmap(
        correlation,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5,
    )
    plt.title("Selected-feature Pearson correlation (original training data)")
    plt.tight_layout()
    plt.savefig(output_dir / f"{stem}_correlation_heatmap.png", dpi=180)
    plt.close()


def run_dataset(path: Path, run_index: int, output_dir: Path):
    seed = 42 + run_index
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    X_train, X_test, y_train, y_test, experiment = _pycaret_split(
        _load_binary_dataset(path), seed
    )
    best, selection_records = select_model_and_features(experiment, X_train, y_train)
    selected = best["Selected_Features"]
    methods, diagnostics = compare_augmentations(
        best["Estimator"], X_train[selected], y_train, X_test[selected], y_test
    )
    stem = f"{path.stem}_run_{run_index + 1:02d}"
    write_inference_artifacts(X_train, y_train, selected, output_dir, stem)
    selection_rows = []
    for record in selection_records:
        row = {
            key: value
            for key, value in record.items()
            if key not in {"Estimator", "Selected_Features"}
        }
        row.update(
            {
                "Dataset": path.stem,
                "Run": run_index + 1,
                "Seed": seed,
                "Selected_Features": "|".join(record.get("Selected_Features", [])),
            }
        )
        selection_rows.append(row)
    comparison_rows = [
        {
            **method,
            **diagnostics,
            "Dataset": path.stem,
            "Run": run_index + 1,
            "Seed": seed,
            "Selected_Model": best["Model"],
            "Selected_Features": "|".join(selected),
        }
        for method in methods
    ]
    print(
        f"{path.stem} run {run_index + 1}/{N_RUNS}: {best['Model']} ({best['CV_Accuracy_Mean']:.4f} CV), {len(selected)} features"
    )
    return selection_rows, comparison_rows


def main():
    root = Path(__file__).resolve().parent
    output_dir = root / "NewResults" / "final_2"
    output_dir.mkdir(parents=True, exist_ok=True)
    selection_rows, comparison_rows, failures = [], [], []
    for dataset_path in sorted((root / "Datasets").glob("*.csv")):
        for run_index in range(N_RUNS):
            try:
                selected, compared = run_dataset(dataset_path, run_index, output_dir)
                selection_rows.extend(selected)
                comparison_rows.extend(compared)
            except Exception as exc:
                failures.append(
                    {
                        "Dataset": dataset_path.stem,
                        "Run": run_index + 1,
                        "Error": str(exc),
                    }
                )
                print(f"{dataset_path.stem} run {run_index + 1} failed: {exc}")
    pd.DataFrame(selection_rows).to_csv(
        output_dir / "model_selection_runs.csv", index=False
    )
    comparisons = pd.DataFrame(comparison_rows)
    comparisons.to_csv(output_dir / "augmentation_comparison_runs.csv", index=False)
    if not comparisons.empty:
        metrics = ["Accuracy", "F1_Weighted", "Precision_Weighted", "Recall_Weighted"]
        comparisons.groupby(["Dataset", "Method"])[metrics].agg(["mean", "std"]).to_csv(
            output_dir / "augmentation_comparison_summary.csv"
        )
    pd.DataFrame(failures).to_csv(output_dir / "failures.csv", index=False)


if __name__ == "__main__":
    main()

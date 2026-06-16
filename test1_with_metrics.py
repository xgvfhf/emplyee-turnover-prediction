# hr_attrition_full_report_with_plots.py
# -*- coding: utf-8 -*-
"""
Generates:
1) Class imbalance bar chart
2) Pearson correlation heatmap (numeric features)
3) Logistic Regression top-10 coefficients (feature importance)
4) SHAP summary plot for SVM and MLP (Kernel SHAP; sampled for speed)
5) ROC curve for best SVM (by test recall)
6) Confusion matrix for best SVM (with tuned threshold)
7) t-SNE plots:
   - colored by TRUE Attrition (train)
   - colored by PREDICTED risk (train, best SVM proba)
Also runs:
- Data preparation (duplicates, ID-like, constants, high missing, outlier clipping)
- RepeatedStratifiedKFold (5x3) GridSearch on multiple models
- Threshold tuning (maximize Recall with Precision constraint)
- Saves results CSV
"""

import warnings
warnings.filterwarnings("ignore")

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split, GridSearchCV
from sklearn.metrics import (
    recall_score, precision_score, f1_score, roc_auc_score,
    confusion_matrix, make_scorer, roc_curve, auc
)
from sklearn.base import clone

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

from sklearn.manifold import TSNE

# SHAP (install if missing: pip install shap)
import shap


# =========================
# CONFIG
# =========================
RANDOM_STATE = 42
CSV_PATH = r"WA_Fn-UseC_-HR-Employee-Attrition.csv"

# Decision constraint: maximize Recall, keep Precision >= MIN_PRECISION if possible
MIN_PRECISION = 0.40

# Threshold grid for selection
THRESHOLDS = np.linspace(0.05, 0.95, 91)

# Data prep settings
DROP_MISSING_COL_THRESHOLD = 0.40
DROP_HIGH_CARDINALITY = True
HIGH_CARDINALITY_RATIO = 0.50
HIGH_CARDINALITY_ABS = 200

# Outliers
OUTLIERS_MODE = "clip"   # "clip" or "drop"
OUTLIERS_IQR_K = 1.5

MANUAL_DROP_COLS = [
    # "EmployeeNumber", "EmployeeCount", "Over18", "StandardHours"
]

# Plot outputs
FIG_DPI = 300

# SHAP sampling (to keep Kernel SHAP feasible)
SHAP_BACKGROUND_N = 120
SHAP_EXPLAIN_N = 250

# t-SNE sampling (optional; set None to use full train)
TSNE_N = None  # e.g. 800 for faster


# =========================
# Utilities
# =========================
def safe_params_to_json(params: dict) -> str:
    """JSON-safe representation of best_params_ (handles SMOTE objects)."""
    out = {}
    for k, v in params.items():
        if hasattr(v, "get_params") and hasattr(v, "__class__"):
            cls = v.__class__.__name__
            try:
                vp = v.get_params()
                keep_keys = ["random_state", "sampling_strategy", "k_neighbors", "n_jobs"]
                keep = {kk: vp[kk] for kk in keep_keys if kk in vp}
                out[k] = f"{cls}({keep})" if keep else cls
            except Exception:
                out[k] = cls
        else:
            if isinstance(v, (np.integer,)):
                v = int(v)
            elif isinstance(v, (np.floating,)):
                v = float(v)
            out[k] = v
    return json.dumps(out, ensure_ascii=False)


def map_target(y_raw: pd.Series) -> pd.Series:
    y = y_raw.astype(str).str.strip().str.lower().map({"yes": 1, "no": 0})
    if y.isna().any():
        raise ValueError(f"Cannot map target values. Unique: {sorted(y_raw.unique().tolist())}")
    return y.astype(int)


def detect_id_like_columns(df: pd.DataFrame, target_col: str) -> list:
    drop = set()

    for c in ["EmployeeCount", "EmployeeNumber", "Over18", "StandardHours"]:
        if c in df.columns and c != target_col:
            drop.add(c)

    for c in df.columns:
        if c == target_col:
            continue
        cl = c.lower()
        if cl in {"id", "employeeid"} or cl.endswith("_id") or cl.startswith("id_"):
            drop.add(c)
        if "employee" in cl and ("number" in cl or cl.endswith("no")):
            drop.add(c)

    for c in MANUAL_DROP_COLS:
        if c in df.columns and c != target_col:
            drop.add(c)

    return sorted(drop)


def drop_constant_columns(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    const_cols = [c for c in df.columns if c != target_col and df[c].nunique(dropna=False) <= 1]
    if const_cols:
        df = df.drop(columns=const_cols, errors="ignore")
    return df


def drop_high_missing_columns(df: pd.DataFrame, target_col: str, threshold: float) -> pd.DataFrame:
    miss_ratio = df.isna().mean()
    drop_cols = [c for c, r in miss_ratio.items() if c != target_col and r > threshold]
    if drop_cols:
        df = df.drop(columns=drop_cols, errors="ignore")
    return df


def drop_high_cardinality_categoricals(
    df: pd.DataFrame, target_col: str, ratio_thr: float, abs_thr: int
) -> pd.DataFrame:
    if not DROP_HIGH_CARDINALITY:
        return df
    n = len(df)
    drop_cols = []
    for c in df.columns:
        if c == target_col:
            continue
        if df[c].dtype == object:
            nun = df[c].nunique(dropna=True)
            if (n > 0 and nun / n > ratio_thr) or (nun > abs_thr):
                drop_cols.append(c)
    if drop_cols:
        df = df.drop(columns=drop_cols, errors="ignore")
    return df


def clip_outliers_iqr(df: pd.DataFrame, numeric_cols: list, k: float = 1.5) -> pd.DataFrame:
    df = df.copy()
    for c in numeric_cols:
        x = df[c].astype(float)
        q1 = x.quantile(0.25)
        q3 = x.quantile(0.75)
        iqr = q3 - q1
        lo = q1 - k * iqr
        hi = q3 + k * iqr
        df[c] = x.clip(lower=lo, upper=hi)
    return df


def drop_outlier_rows_iqr(df: pd.DataFrame, numeric_cols: list, k: float = 1.5) -> pd.DataFrame:
    if not numeric_cols:
        return df
    mask = pd.Series(True, index=df.index)
    for c in numeric_cols:
        x = df[c].astype(float)
        q1 = x.quantile(0.25)
        q3 = x.quantile(0.75)
        iqr = q3 - q1
        lo = q1 - k * iqr
        hi = q3 + k * iqr
        mask &= x.between(lo, hi) | x.isna()
    return df.loc[mask].copy()


def data_preparation(csv_path: str) -> tuple[pd.DataFrame, pd.Series, dict]:
    df = pd.read_csv(csv_path)
    report = {"rows_start": int(len(df)), "cols_start": int(df.shape[1])}

    # target
    target_col = None
    for cand in ["Attrition", "attrition", "TARGET", "target", "y"]:
        if cand in df.columns:
            target_col = cand
            break
    if target_col is None:
        raise ValueError("Target column not found. Expected 'Attrition'.")

    # duplicates
    before = len(df)
    df = df.drop_duplicates()
    report["duplicates_removed"] = int(before - len(df))

    # id-like drop
    id_cols = detect_id_like_columns(df, target_col)
    report["id_like_dropped"] = id_cols
    if id_cols:
        df = df.drop(columns=id_cols, errors="ignore")

    # constant drop
    before_cols = df.shape[1]
    df = drop_constant_columns(df, target_col)
    report["constant_cols_removed"] = int(before_cols - df.shape[1])

    # high missing drop
    before_cols = df.shape[1]
    df = drop_high_missing_columns(df, target_col, DROP_MISSING_COL_THRESHOLD)
    report["high_missing_cols_removed"] = int(before_cols - df.shape[1])

    # high-card categoricals drop
    before_cols = df.shape[1]
    df = drop_high_cardinality_categoricals(df, target_col, HIGH_CARDINALITY_RATIO, HIGH_CARDINALITY_ABS)
    report["high_cardinality_cols_removed"] = int(before_cols - df.shape[1])

    y = map_target(df[target_col])
    X = df.drop(columns=[target_col])

    # outliers
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    report["outliers_mode"] = OUTLIERS_MODE
    report["outliers_k"] = OUTLIERS_IQR_K

    if OUTLIERS_MODE == "clip":
        X = clip_outliers_iqr(X, num_cols, k=OUTLIERS_IQR_K)
        report["outlier_rows_removed"] = 0
    elif OUTLIERS_MODE == "drop":
        tmp = pd.concat([X, y.rename("__y__")], axis=1)
        before = len(tmp)
        tmp = drop_outlier_rows_iqr(tmp, num_cols, k=OUTLIERS_IQR_K)
        report["outlier_rows_removed"] = int(before - len(tmp))
        y = tmp["__y__"].astype(int)
        X = tmp.drop(columns=["__y__"])
    else:
        raise ValueError("OUTLIERS_MODE must be 'clip' or 'drop'.")

    report["rows_final"] = int(len(X))
    report["cols_final"] = int(X.shape[1])
    return X, y, report


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    num_pipe = ImbPipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_pipe = ImbPipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3
    )


def build_models_and_grids(random_state: int = 42):
    return {
        "LogReg": (
            LogisticRegression(max_iter=5000, solver="liblinear", random_state=random_state),
            {"clf__C": [0.1, 1, 3, 10],
             "clf__class_weight": [None, "balanced"]},
        ),
        "RandomForest": (
            RandomForestClassifier(random_state=random_state, n_jobs=-1),
            {"clf__n_estimators": [300, 600],
             "clf__max_depth": [None, 10, 20],
             "clf__min_samples_split": [2, 5],
             "clf__class_weight": [None, "balanced"]},
        ),
        "DecisionTree": (
            DecisionTreeClassifier(random_state=random_state),
            {"clf__max_depth": [None, 6, 12, 20],
             "clf__min_samples_split": [2, 5],
             "clf__class_weight": [None, "balanced"]},
        ),
        "SVM_RBF": (
            SVC(kernel="rbf", probability=True, random_state=random_state),
            {"clf__C": [0.5, 1, 3, 10],
             "clf__gamma": ["scale", "auto"],
             "clf__class_weight": [None, "balanced"]},
        ),
        "GradientBoosting": (
            GradientBoostingClassifier(random_state=random_state),
            {"clf__n_estimators": [200, 400],
             "clf__learning_rate": [0.05, 0.1],
             "clf__max_depth": [2, 3]},
        ),
        "KNN": (
            KNeighborsClassifier(),
            {"clf__n_neighbors": [5, 11, 15],
             "clf__weights": ["uniform", "distance"]},
        ),
        "MLP": (
            MLPClassifier(random_state=random_state, max_iter=1000),
            {"clf__hidden_layer_sizes": [(64,), (128,), (64, 32)],
             "clf__alpha": [1e-4, 1e-3, 1e-2],
             "clf__learning_rate_init": [1e-3, 5e-4]},
        ),
    }


def predict_proba_pos(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        s = model.decision_function(X)
        s = (s - s.min()) / (s.max() - s.min() + 1e-9)
        return s
    return model.predict(X).astype(float)


def oof_probabilities(estimator, X, y, cv):
    oof = np.zeros(len(y), dtype=float)
    for tr_idx, va_idx in cv.split(X, y):
        est = clone(estimator)
        est.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        oof[va_idx] = predict_proba_pos(est, X.iloc[va_idx])
    return oof


def pick_threshold_with_precision_constraint(oof_proba, y_true, thresholds, min_precision: float):
    best = None
    best_pair = (-1.0, -1.0, -1.0)  # (recall, f1, precision)

    for t in thresholds:
        y_hat = (oof_proba >= t).astype(int)
        p = precision_score(y_true, y_hat, zero_division=0)
        r = recall_score(y_true, y_hat, zero_division=0)
        f = f1_score(y_true, y_hat, zero_division=0)

        if p >= min_precision:
            pair = (r, f, p)
            if pair > best_pair:
                best_pair = pair
                best = (float(t), r, p, f)

    if best is not None:
        return {"threshold": best[0], "oof_recall": float(best[1]), "oof_precision": float(best[2]),
                "oof_f1": float(best[3]), "constraint_met": True}

    # fallback: best F1 then recall
    best_f = (-1.0, -1.0)
    best_t = 0.5
    best_stats = (0.0, 0.0, 0.0)
    for t in thresholds:
        y_hat = (oof_proba >= t).astype(int)
        p = precision_score(y_true, y_hat, zero_division=0)
        r = recall_score(y_true, y_hat, zero_division=0)
        f = f1_score(y_true, y_hat, zero_division=0)
        if (f, r) > best_f:
            best_f = (f, r)
            best_t = float(t)
            best_stats = (r, p, f)

    return {"threshold": best_t, "oof_recall": float(best_stats[0]), "oof_precision": float(best_stats[1]),
            "oof_f1": float(best_stats[2]), "constraint_met": False}


def to_dense(X):
    return X.toarray() if hasattr(X, "toarray") else np.asarray(X)


# =========================
# PLOTS
# =========================
def plot_class_distribution(y: pd.Series):
    counts = y.value_counts().sort_index()
    plt.figure(figsize=(6, 4))
    plt.bar(["No", "Yes"], [counts.get(0, 0), counts.get(1, 0)])
    plt.title("Class Distribution (Attrition)")
    plt.xlabel("Attrition")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("class_distribution.png", dpi=FIG_DPI)
    plt.show()


def plot_correlation_heatmap(X: pd.DataFrame):
    num = X.select_dtypes(include=[np.number]).copy()
    if num.shape[1] == 0:
        print("No numeric columns available for Pearson correlation heatmap.")
        return

    corr = num.corr(method="pearson").values
    labels = num.columns.tolist()

    plt.figure(figsize=(12, 10))
    im = plt.imshow(corr, aspect="auto")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title("Pearson Correlation Heatmap (Numerical Features)")
    plt.xticks(range(len(labels)), labels, rotation=90, fontsize=8)
    plt.yticks(range(len(labels)), labels, fontsize=8)
    plt.tight_layout()
    plt.savefig("correlation_heatmap_pearson.png", dpi=FIG_DPI)
    plt.show()


def plot_logreg_top10_coefficients(fitted_logreg_pipe: ImbPipeline, preprocessor_fitted: ColumnTransformer):
    clf = fitted_logreg_pipe.named_steps["clf"]
    if not hasattr(clf, "coef_"):
        print("LogReg classifier has no coef_.")
        return

    # feature names from fitted preprocessor
    feat_names = preprocessor_fitted.get_feature_names_out()
    coefs = clf.coef_.ravel()

    df = pd.DataFrame({"feature": feat_names, "coef": coefs})
    df["abs"] = df["coef"].abs()
    top = df.sort_values("abs", ascending=False).head(10).sort_values("coef")

    plt.figure(figsize=(9, 6))
    plt.barh(top["feature"], top["coef"])
    plt.title("Logistic Regression: Top 10 Features (Coefficients)")
    plt.xlabel("Coefficient (log-odds; >0 increases attrition risk)")
    plt.tight_layout()
    plt.savefig("logreg_top10_coefficients.png", dpi=FIG_DPI)
    plt.show()


def plot_roc_curve(model_pipe, X_test, y_test, title="ROC Curve (SVM)"):
    proba = predict_proba_pos(model_pipe, X_test)
    fpr, tpr, _ = roc_curve(y_test, proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("roc_curve_svm.png", dpi=FIG_DPI)
    plt.show()


def plot_confusion_matrix(cm, title="Confusion Matrix (SVM)", filename="confusion_matrix_svm.png"):
    # cm = [[TN, FP], [FN, TP]]
    plt.figure(figsize=(5.2, 4.6))
    plt.imshow(cm, aspect="auto")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks([0, 1], ["No", "Yes"])
    plt.yticks([0, 1], ["No", "Yes"])

    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i][j]), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(filename, dpi=FIG_DPI)
    plt.show()


def plot_tsne_true_and_pred(preprocessor_fitted, svm_pipe, X_train, y_train):
    # optional sampling for speed
    if TSNE_N is not None and TSNE_N < len(X_train):
        idx = X_train.sample(TSNE_N, random_state=RANDOM_STATE).index
        Xp = X_train.loc[idx]
        yp = y_train.loc[idx]
    else:
        Xp, yp = X_train, y_train

    X_enc = to_dense(preprocessor_fitted.transform(Xp))

    tsne = TSNE(n_components=2, perplexity=30, init="pca", learning_rate="auto", random_state=RANDOM_STATE)
    X_2d = tsne.fit_transform(X_enc)

    # TRUE labels
    plt.figure(figsize=(7, 6))
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=yp.values, alpha=0.7)
    plt.title("t-SNE Projection (True Attrition)")
    plt.xlabel("t-SNE-1")
    plt.ylabel("t-SNE-2")
    plt.colorbar(label="Attrition (1=Yes)")
    plt.tight_layout()
    plt.savefig("tsne_true_attrition.png", dpi=FIG_DPI)
    plt.show()

    # PREDICTED risk (probabilities)
    proba = predict_proba_pos(svm_pipe, Xp)
    plt.figure(figsize=(7, 6))
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=proba, alpha=0.7)
    plt.title("t-SNE Projection (Predicted Attrition Risk)")
    plt.xlabel("t-SNE-1")
    plt.ylabel("t-SNE-2")
    plt.colorbar(label="Predicted risk P(Attrition=Yes)")
    plt.tight_layout()
    plt.savefig("tsne_predicted_risk.png", dpi=FIG_DPI)
    plt.show()


# =========================
# SHAP (Kernel) for SVM / MLP
# =========================
def fit_classifier_for_shap(
    best_pipe: ImbPipeline,
    preprocessor_fitted: ColumnTransformer,
    X_train: pd.DataFrame,
    y_train: pd.Series
):
    """
    To make SHAP feasible and stable:
    - Extract classifier hyperparams from best_pipe
    - Preprocess X_train using fitted preprocessor
    - If best_pipe uses SMOTE (sampler != passthrough), apply SMOTE on encoded data
    - Fit a standalone classifier on encoded (and optionally resampled) data
    Returns: (fitted_clf, X_enc, y_enc) where X_enc is dense numpy.
    """
    sampler = best_pipe.named_steps.get("sampler", "passthrough")
    clf = best_pipe.named_steps["clf"]

    X_enc = to_dense(preprocessor_fitted.transform(X_train))
    y_enc = y_train.values.astype(int)

    if sampler != "passthrough":
        # apply same sampler on encoded data
        if isinstance(sampler, SMOTE):
            sm = clone(sampler)
            X_enc, y_enc = sm.fit_resample(X_enc, y_enc)

    clf2 = clone(clf)
    clf2.fit(X_enc, y_enc)
    return clf2, X_enc, y_enc


def shap_summary_kernel(clf_fitted, X_enc, feature_names, model_name: str):
    """
    Kernel SHAP (sampled):
    - background: SHAP_BACKGROUND_N
    - explain: SHAP_EXPLAIN_N
    """
    rng = np.random.default_rng(RANDOM_STATE)
    n = X_enc.shape[0]

    bg_n = min(SHAP_BACKGROUND_N, n)
    ex_n = min(SHAP_EXPLAIN_N, n)

    bg_idx = rng.choice(n, size=bg_n, replace=False)
    ex_idx = rng.choice(n, size=ex_n, replace=False)

    X_bg = X_enc[bg_idx]
    X_ex = X_enc[ex_idx]

    def f_proba_pos(X):
        return clf_fitted.predict_proba(X)[:, 1]

    explainer = shap.KernelExplainer(f_proba_pos, X_bg)
    shap_values = explainer.shap_values(X_ex, nsamples="auto")

    # summary plot
    shap.summary_plot(
        shap_values,
        features=X_ex,
        feature_names=feature_names,
        show=False,
        max_display=10
    )
    plt.title(f"Top 10 Attrition Factors ({model_name}, SHAP)")
    plt.tight_layout()
    plt.savefig(f"shap_summary_{model_name.lower()}.png", dpi=FIG_DPI)
    plt.show()


# =========================
# MAIN
# =========================
def main():
    # 1) Data preparation
    X, y, prep_report = data_preparation(CSV_PATH)
    print("\n=== DATA PREPARATION REPORT ===")
    print(json.dumps(prep_report, ensure_ascii=False, indent=2))

    # 1a) Class imbalance plot (on full cleaned dataset)
    plot_class_distribution(y)

    # 2) Holdout split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    # 2a) Correlation heatmap on TRAIN (EDA)
    plot_correlation_heatmap(X_train)

    # 3) Preprocessing
    preprocessor = build_preprocessor(X_train)

    # For plots/explainability we need a fitted preprocessor on TRAIN only
    preprocessor_fitted = clone(preprocessor).fit(X_train)
    feature_names = preprocessor_fitted.get_feature_names_out()

    # 4) CV
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=RANDOM_STATE)

    # 5) Balancing options: none vs SMOTE
    samplers = ["passthrough", SMOTE(random_state=RANDOM_STATE)]

    scoring = {
        "recall": make_scorer(recall_score),
        "precision": make_scorer(precision_score),
        "f1": make_scorer(f1_score),
        "roc_auc": "roc_auc",
    }

    models = build_models_and_grids(RANDOM_STATE)
    rows = []

    best_svm = {
        "pipe": None,
        "thr": None,
        "score": (-1.0, -1.0, -1.0),  # (test_recall, test_f1, test_precision)
        "cm": None
    }

    best_logreg_pipe = None
    best_mlp_pipe = None

    for model_name, (model, grid) in models.items():
        print(f"\n=== Model: {model_name} ===")

        pipe = ImbPipeline(steps=[
            ("prep", preprocessor),
            ("sampler", "passthrough"),
            ("clf", model),
        ])

        param_grid = {"sampler": samplers, **grid}

        gs = GridSearchCV(
            estimator=pipe,
            param_grid=param_grid,
            scoring=scoring,
            refit="recall",
            cv=cv,
            n_jobs=-1,
            verbose=0,
        )
        gs.fit(X_train, y_train)

        best_idx = gs.best_index_
        cv_recall = float(gs.cv_results_["mean_test_recall"][best_idx])
        cv_prec = float(gs.cv_results_["mean_test_precision"][best_idx])
        cv_f1 = float(gs.cv_results_["mean_test_f1"][best_idx])
        cv_auc = float(gs.cv_results_["mean_test_roc_auc"][best_idx])

        # threshold tuning on OOF
        oof = oof_probabilities(gs.best_estimator_, X_train, y_train, cv=cv)
        thr_info = pick_threshold_with_precision_constraint(
            oof, y_train.values, thresholds=THRESHOLDS, min_precision=MIN_PRECISION
        )

        # fit best estimator on full train
        best_est = clone(gs.best_estimator_)
        best_est.fit(X_train, y_train)

        # evaluate on holdout
        proba_test = predict_proba_pos(best_est, X_test)
        pred_test = (proba_test >= thr_info["threshold"]).astype(int)

        test_recall = float(recall_score(y_test, pred_test, zero_division=0))
        test_prec = float(precision_score(y_test, pred_test, zero_division=0))
        test_f1 = float(f1_score(y_test, pred_test, zero_division=0))
        try:
            test_auc = float(roc_auc_score(y_test, proba_test))
        except Exception:
            test_auc = float("nan")

        cm = confusion_matrix(y_test, pred_test).tolist()

        rows.append({
            "model": model_name,
            "best_params": safe_params_to_json(gs.best_params_),
            "cv_recall": cv_recall,
            "cv_precision": cv_prec,
            "cv_f1": cv_f1,
            "cv_roc_auc": cv_auc,
            "threshold": float(thr_info["threshold"]),
            "oof_recall@thr": float(thr_info["oof_recall"]),
            "oof_precision@thr": float(thr_info["oof_precision"]),
            "oof_f1@thr": float(thr_info["oof_f1"]),
            "precision_constraint_met": bool(thr_info["constraint_met"]),
            "test_recall": test_recall,
            "test_precision": test_prec,
            "test_f1": test_f1,
            "test_roc_auc": test_auc,
            "confusion_matrix": json.dumps(cm),
        })

        met = "MET" if thr_info["constraint_met"] else "NOT_MET"
        print(
            f"CV(best-by-recall): R={cv_recall:.3f}, P={cv_prec:.3f}, F1={cv_f1:.3f}, AUC={cv_auc:.3f} | "
            f"thr={thr_info['threshold']:.2f} ({met}, minP={MIN_PRECISION}) | "
            f"TEST: R={test_recall:.3f}, P={test_prec:.3f}, F1={test_f1:.3f}, AUC={test_auc:.3f}"
        )

        # store best SVM / best LogReg / best MLP
        if model_name == "SVM_RBF":
            score_tuple = (test_recall, test_f1, test_prec)
            if score_tuple > best_svm["score"]:
                best_svm["score"] = score_tuple
                best_svm["pipe"] = best_est
                best_svm["thr"] = float(thr_info["threshold"])
                best_svm["cm"] = cm

        if model_name == "LogReg":
            best_logreg_pipe = best_est

        if model_name == "MLP":
            best_mlp_pipe = best_est

    # Results CSV
    res = pd.DataFrame(rows).sort_values(["test_recall", "test_f1", "test_precision"], ascending=False)
    out_csv = "results_with_data_prep_repeatedcv_auc_fullplots.csv"
    res.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"\nSaved -> {out_csv}")

    print("\nTOP-10:")
    print(res.head(10)[[
        "model", "test_recall", "test_precision", "test_f1", "test_roc_auc",
        "cv_recall", "cv_precision", "cv_f1", "cv_roc_auc",
        "threshold", "precision_constraint_met"
    ]])

    # =========================
    # PLOTS based on fitted models
    # =========================

    # 1) Logistic Regression Top-10 (coefficients)
    if best_logreg_pipe is not None:
        plot_logreg_top10_coefficients(best_logreg_pipe, preprocessor_fitted)
    else:
        print("LogReg model not available for coefficient plot.")

    # 2) ROC curve for SVM (best by test recall/f1/precision)
    if best_svm["pipe"] is not None:
        plot_roc_curve(best_svm["pipe"], X_test, y_test, title="ROC Curve (SVM_RBF)")
    else:
        print("SVM model not available for ROC curve.")

    # 3) Confusion matrix for SVM (with tuned threshold)
    if best_svm["cm"] is not None:
        plot_confusion_matrix(best_svm["cm"], title="Confusion Matrix (SVM_RBF, tuned threshold)")
    else:
        print("SVM confusion matrix not available.")

    # 4) t-SNE true vs predicted risk (best SVM)
    if best_svm["pipe"] is not None:
        plot_tsne_true_and_pred(preprocessor_fitted, best_svm["pipe"], X_train, y_train)
    else:
        print("Skipping t-SNE: no SVM model.")

    # 5) SHAP for SVM and MLP (Kernel SHAP; sampled)
    # NOTE: Kernel SHAP is expensive; we fit a standalone classifier on encoded train (with SMOTE if chosen).
    if best_svm["pipe"] is not None:
        print("\n=== SHAP (Kernel) for SVM_RBF (sampled) ===")
        svm_clf, X_enc_svm, y_enc_svm = fit_classifier_for_shap(
            best_svm["pipe"], preprocessor_fitted, X_train, y_train
        )
        shap_summary_kernel(svm_clf, X_enc_svm, feature_names, model_name="SVM_RBF")
    else:
        print("Skipping SHAP for SVM: no fitted SVM pipe.")

    if best_mlp_pipe is not None:
        print("\n=== SHAP (Kernel) for MLP (sampled) ===")
        mlp_clf, X_enc_mlp, y_enc_mlp = fit_classifier_for_shap(
            best_mlp_pipe, preprocessor_fitted, X_train, y_train
        )
        shap_summary_kernel(mlp_clf, X_enc_mlp, feature_names, model_name="MLP")
    else:
        print("Skipping SHAP for MLP: no fitted MLP pipe.")

    print("\nDone. Figures saved as PNG files in the current folder.")


if __name__ == "__main__":
    main()
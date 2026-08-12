import os
import time
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier


# =====================================================================
# Label mapping
# =====================================================================
CLASS_NAMES_ALL = ["0: No-Cough", "1: Dry", "2: Wet", "3: Unknown"]
CLASS_NAMES_COUGH = ["1: Dry", "2: Wet", "3: Unknown"]


# =====================================================================
# Paths and configuration
# =====================================================================
PATH_FEATURES_DIR = r"C:\Users\Usuario\Desktop\UNI MARINA\master\TFM\TFM_resp_dis\features_extracted_multiclass_4c"
OUTPUT_DIR = os.path.join(PATH_FEATURES_DIR, "stage2_drywet_plots")

CONFIDENCE_GRID = np.linspace(0.50, 0.90, 9)
MARGIN_GRID = np.linspace(0.05, 0.35, 7)
GENERATE_STAGE2_PCA_PLOT = True

XGB_PARAMS_STAGE1 = {
    "n_estimators": 180,
    "learning_rate": 0.05,
    "max_depth": 5,
    "random_state": 42,
    "n_jobs": -1,
}

XGB_PARAMS_STAGE2 = {
    "n_estimators": 200,
    "learning_rate": 0.03,
    "max_depth": 4,
    "subsample": 0.85,
    "colsample_bytree": 0.9,
    "random_state": 42,
    "n_jobs": -1,
}

SVM_PCA_PARAMS = {
    "pca_variance": 0.95,
    "C": 1.5,
}

STAGE2_CANDIDATES = [
    {
        "name": "xgb_drywet_reject",
        "kind": "xgb_drywet",
    },
    {
        "name": "svm_rbf_pca_drywet_reject",
        "kind": "svm_rbf_pca_drywet",
    },
    {
        "name": "svm_poly_pca_drywet_reject",
        "kind": "svm_poly_pca_drywet",
    },
]


# =====================================================================
# Helpers
# =====================================================================
@dataclass
class Stage2Result:
    name: str
    kind: str
    confidence_threshold: float
    margin_threshold: float
    score: float
    model: object


def load_features() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    x_train_raw = np.load(os.path.join(PATH_FEATURES_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(PATH_FEATURES_DIR, "y_train.npy"))
    folds_train = np.load(os.path.join(PATH_FEATURES_DIR, "folds_train.npy"))

    x_val_raw = np.load(os.path.join(PATH_FEATURES_DIR, "X_val.npy"))
    y_val = np.load(os.path.join(PATH_FEATURES_DIR, "y_val.npy"))

    x_test_raw = np.load(os.path.join(PATH_FEATURES_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(PATH_FEATURES_DIR, "y_test.npy"))

    train_mean = np.mean(x_train_raw, axis=0, keepdims=True)
    train_std = np.std(x_train_raw, axis=0, keepdims=True)

    x_train = (x_train_raw - train_mean) / (train_std + 1e-8)
    x_val = (x_val_raw - train_mean) / (train_std + 1e-8)
    x_test = (x_test_raw - train_mean) / (train_std + 1e-8)

    return x_train, y_train, folds_train, x_val, y_val, x_test, y_test


def build_stage1_model(x_train: np.ndarray, y_train: np.ndarray) -> XGBClassifier:
    y_train_bin = (y_train > 0).astype(int)
    scale_pos_weight = (y_train_bin == 0).sum() / max((y_train_bin == 1).sum(), 1)

    model_stage1 = XGBClassifier(
        **XGB_PARAMS_STAGE1,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="logloss",
    )
    model_stage1.fit(x_train, y_train_bin)
    return model_stage1


def evaluate_stage1_cv(x_train: np.ndarray, y_train: np.ndarray, folds_train: np.ndarray) -> None:
    y_train_bin = (y_train > 0).astype(int)
    print("\n" + "=" * 80)
    print(" STAGE 1: 5-FOLD CV FOR COUGH / NO-COUGH DETECTION")
    print("=" * 80)

    fold_scores = []
    for fold_idx in range(5):
        val_mask = folds_train == fold_idx
        tr_mask = ~val_mask

        x_tr = x_train[tr_mask]
        y_tr = y_train_bin[tr_mask]
        x_va = x_train[val_mask]
        y_va = y_train_bin[val_mask]

        scale_pos_weight = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
        model = XGBClassifier(
            **XGB_PARAMS_STAGE1,
            scale_pos_weight=scale_pos_weight,
            objective="binary:logistic",
            eval_metric="logloss",
        )
        model.fit(x_tr, y_tr)
        preds = model.predict(x_va)

        fold_f1 = f1_score(y_va, preds, average="binary", zero_division=0)
        fold_scores.append(fold_f1)
        print(f"Fold {fold_idx + 1}: binary F1 = {fold_f1:.4f}")

    print(f"Mean 5-fold binary F1: {np.mean(fold_scores):.4f}")


def prepare_stage2_data(
    x_train: np.ndarray,
    y_train: np.ndarray,
    folds_train: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mask_cough = y_train > 0
    x_train_cough = x_train[mask_cough]
    y_train_cough = y_train[mask_cough] - 1  # Dry=0, Wet=1, Unknown=2
    folds_train_cough = folds_train[mask_cough]

    mask_drywet = y_train_cough < 2
    x_train_drywet = x_train_cough[mask_drywet]
    y_train_drywet = y_train_cough[mask_drywet]  # Dry=0, Wet=1

    return x_train_cough, y_train_cough, folds_train_cough, x_train_drywet, y_train_drywet, mask_cough


def plot_stage2_pca_distribution(x_stage2: np.ndarray, y_stage2: np.ndarray) -> None:
    pca = PCA(n_components=2, random_state=42)
    x_2d = pca.fit_transform(x_stage2)

    plt.figure(figsize=(10, 7))
    palette = {
        0: ("tab:blue", "Dry"),
        1: ("tab:orange", "Wet"),
        2: ("tab:green", "Unknown"),
    }

    for class_id, (color, label) in palette.items():
        mask = y_stage2 == class_id
        if np.any(mask):
            plt.scatter(
                x_2d[mask, 0],
                x_2d[mask, 1],
                s=18,
                alpha=0.65,
                c=color,
                label=f"{label} (n={mask.sum()})",
                edgecolors="none",
            )

    plt.title(
        f"Stage 2 PCA projection of cough samples\n"
        f"Explained variance: {pca.explained_variance_ratio_[0]:.3f} + {pca.explained_variance_ratio_[1]:.3f}"
    )
    plt.xlabel("PCA component 1")
    plt.ylabel("PCA component 2")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()

    output_path = os.path.join(OUTPUT_DIR, "stage2_pca_distribution.png")
    plt.savefig(output_path, dpi=160)
    plt.close()
    print(f"Saved PCA distribution plot to: {output_path}")


def fit_stage2_model(kind: str, x_train: np.ndarray, y_train: np.ndarray) -> object:
    if kind == "xgb_drywet":
        y_train_bin = y_train.astype(int)
        scale_pos_weight = (y_train_bin == 0).sum() / max((y_train_bin == 1).sum(), 1)
        model = XGBClassifier(
            **XGB_PARAMS_STAGE2,
            scale_pos_weight=scale_pos_weight,
            objective="binary:logistic",
            eval_metric="logloss",
        )
        model.fit(x_train, y_train_bin)
        return model

    if kind in {"svm_rbf_pca_drywet", "svm_poly_pca_drywet"}:
        kernel = "rbf" if kind == "svm_rbf_pca_drywet" else "poly"
        model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=SVM_PCA_PARAMS["pca_variance"], random_state=42)),
                (
                    "svc",
                    SVC(
                        C=SVM_PCA_PARAMS["C"],
                        kernel=kernel,
                        degree=3,
                        gamma="scale",
                        class_weight="balanced",
                        probability=True,
                        random_state=42,
                    ),
                ),
            ]
        )
        model.fit(x_train, y_train.astype(int))
        return model

    raise ValueError(f"Unsupported stage2 kind: {kind}")


def predict_stage2_binary_reject(
    kind: str,
    model: object,
    x_input: np.ndarray,
    confidence_threshold: float = 0.55,
    margin_threshold: float = 0.15,
) -> np.ndarray:
    proba = model.predict_proba(x_input)
    if proba.ndim == 1:
        proba = proba.reshape(-1, 1)

    if proba.shape[1] == 1:
        proba = np.hstack([1.0 - proba, proba])

    p_dry = proba[:, 0]
    p_wet = proba[:, 1]
    pred_class = np.argmax(proba, axis=1)
    max_prob = np.max(proba, axis=1)
    margin = np.abs(p_dry - p_wet)

    final_preds = np.full(len(x_input), 2, dtype=int)
    mask_decided = (max_prob >= confidence_threshold) & (margin >= margin_threshold)
    final_preds[mask_decided & (pred_class == 0)] = 0
    final_preds[mask_decided & (pred_class == 1)] = 1
    return final_preds


def evaluate_stage2_candidate(
    candidate: dict,
    x_train_cough: np.ndarray,
    y_train_cough: np.ndarray,
    folds_train_cough: np.ndarray,
) -> Stage2Result:
    print("\n" + "=" * 80)
    print(f" STAGE 2 CANDIDATE: {candidate['name']}")
    print("=" * 80)

    best_conf = None
    best_margin = None
    best_score = -1.0

    for conf_thr in CONFIDENCE_GRID:
        for margin_thr in MARGIN_GRID:
            fold_scores = []
            for fold_idx in range(5):
                val_mask = folds_train_cough == fold_idx
                tr_mask = ~val_mask

                x_tr_all = x_train_cough[tr_mask]
                y_tr_all = y_train_cough[tr_mask]
                x_va = x_train_cough[val_mask]
                y_va = y_train_cough[val_mask]

                drywet_mask = y_tr_all < 2
                x_tr = x_tr_all[drywet_mask]
                y_tr = y_tr_all[drywet_mask]

                model = fit_stage2_model(candidate["kind"], x_tr, y_tr)
                preds = predict_stage2_binary_reject(
                    candidate["kind"],
                    model,
                    x_va,
                    confidence_threshold=float(conf_thr),
                    margin_threshold=float(margin_thr),
                )

                fold_f1 = f1_score(y_va, preds, average="macro", zero_division=0)
                fold_scores.append(fold_f1)

            mean_score = float(np.mean(fold_scores))
            print(f"conf={conf_thr:.2f}, margin={margin_thr:.2f} -> mean CV macro F1: {mean_score:.4f}")

            if mean_score > best_score:
                best_score = mean_score
                best_conf = float(conf_thr)
                best_margin = float(margin_thr)

    drywet_mask = y_train_cough < 2
    x_final_train = x_train_cough[drywet_mask]
    y_final_train = y_train_cough[drywet_mask]
    final_model = fit_stage2_model(candidate["kind"], x_final_train, y_final_train)

    print(f"Best confidence threshold for {candidate['name']}: {best_conf:.2f}")
    print(f"Best margin threshold for {candidate['name']}: {best_margin:.2f}")
    print(f"Best mean CV macro F1: {best_score:.4f}")

    return Stage2Result(
        name=candidate["name"],
        kind=candidate["kind"],
        confidence_threshold=best_conf,
        margin_threshold=best_margin,
        score=best_score,
        model=final_model,
    )


def predict_hierarchical(x_input: np.ndarray, model_stage1: XGBClassifier, stage2_result: Stage2Result) -> np.ndarray:
    preds_stage1 = model_stage1.predict(x_input)
    final_preds = np.zeros(len(x_input), dtype=int)

    mask_cough = preds_stage1 == 1
    if np.sum(mask_cough) > 0:
        x_cough = x_input[mask_cough]
        preds_stage2 = predict_stage2_binary_reject(
            stage2_result.kind,
            stage2_result.model,
            x_cough,
            confidence_threshold=stage2_result.confidence_threshold,
            margin_threshold=stage2_result.margin_threshold,
        )
        final_preds[mask_cough] = preds_stage2 + 1

    return final_preds


def evaluate_global_pipeline(
    model_stage1: XGBClassifier,
    stage2_result: Stage2Result,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> None:
    print("\n" + "=" * 80)
    print(" GLOBAL EVALUATION OF THE 2-STAGE PIPELINE")
    print("=" * 80)

    for split_name, x_split, y_split in [
        ("TRAIN      ", x_train, y_train),
        ("VALIDATION ", x_val, y_val),
        ("TEST (BLIND)", x_test, y_test),
    ]:
        y_pred = predict_hierarchical(x_split, model_stage1, stage2_result)
        acc = accuracy_score(y_split, y_pred)
        macro_f1 = f1_score(y_split, y_pred, average="macro", zero_division=0)
        print(f"{split_name} -> Accuracy: {acc:.4f} | Macro F1-Score: {macro_f1:.4f}")

    y_pred_test = predict_hierarchical(x_test, model_stage1, stage2_result)
    print("\nClassification report on TEST (4 classes):")
    print(classification_report(y_test, y_pred_test, target_names=CLASS_NAMES_ALL, digits=4, zero_division=0))

    print("\nConfusion matrix on TEST:")
    cm = confusion_matrix(y_test, y_pred_test)
    print(pd.DataFrame(cm, index=CLASS_NAMES_ALL, columns=CLASS_NAMES_ALL))

    mask_test_cough = y_test > 0
    x_test_cough = x_test[mask_test_cough]
    y_test_cough = y_test[mask_test_cough] - 1
    preds_stage2_test = predict_stage2_binary_reject(
        stage2_result.kind,
        stage2_result.model,
        x_test_cough,
        confidence_threshold=stage2_result.confidence_threshold,
        margin_threshold=stage2_result.margin_threshold,
    )

    print("\nStage 2 only, on TEST cough samples:")
    print(
        classification_report(
            y_test_cough,
            preds_stage2_test,
            target_names=["0: Dry", "1: Wet", "2: Unknown"],
            digits=4,
            zero_division=0,
        )
    )


def main() -> None:
    print("=" * 80)
    print(" TWO-STAGE MULTICLASSIFICATION TEST")
    print(" Stage 1: cough / no-cough")
    print(" Stage 2: cough type with reject option")
    print("=" * 80)

    x_train, y_train, folds_train, x_val, y_val, x_test, y_test = load_features()

    print(f"Loaded train shape: {x_train.shape}, val shape: {x_val.shape}, test shape: {x_test.shape}")
    print(f"Stage 1 labels: {np.unique((y_train > 0).astype(int), return_counts=True)}")

    evaluate_stage1_cv(x_train, y_train, folds_train)

    print("\n" + "=" * 80)
    print(" TRAINING STAGE 1 FINAL MODEL")
    print("=" * 80)
    model_stage1 = build_stage1_model(x_train, y_train)

    (
        x_train_cough,
        y_train_cough,
        folds_train_cough,
        x_train_drywet,
        y_train_drywet,
        _,
    ) = prepare_stage2_data(x_train, y_train, folds_train)

    print(f"Train cough samples: {x_train_cough.shape[0]}")
    print(f"Train dry/wet samples (for binary candidates): {x_train_drywet.shape[0]}")

    if GENERATE_STAGE2_PCA_PLOT:
        print("\n" + "=" * 80)
        print(" PCA DIAGNOSTIC PLOT FOR STAGE 2")
        print("=" * 80)
        plot_stage2_pca_distribution(x_train_cough, y_train_cough)

    candidate_results: list[Stage2Result] = []
    for candidate in STAGE2_CANDIDATES:
        result = evaluate_stage2_candidate(
            candidate,
            x_train_cough,
            y_train_cough,
            folds_train_cough,
        )
        candidate_results.append(result)

    best_stage2_result = max(candidate_results, key=lambda item: item.score)

    print("\n" + "=" * 80)
    print(" BEST STAGE 2 CONFIGURATION")
    print("=" * 80)
    print(f"name: {best_stage2_result.name}")
    print(f"kind: {best_stage2_result.kind}")
    print(f"confidence threshold: {best_stage2_result.confidence_threshold:.2f}")
    print(f"margin threshold: {best_stage2_result.margin_threshold:.2f}")
    print(f"mean CV macro F1: {best_stage2_result.score:.4f}")

    evaluate_global_pipeline(
        model_stage1,
        best_stage2_result,
        x_train,
        y_train,
        x_val,
        y_val,
        x_test,
        y_test,
    )


if __name__ == "__main__":
    main()
import os
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier


CLASS_NAMES_GLOBAL = ["0: No-Cough", "1: Dry", "2: Wet"]
CLASS_NAMES_STAGE2 = ["0: Dry", "1: Wet"]

PATH_STAGE1_DIR = r"C:\Users\Usuario\Desktop\UNI MARINA\master\TFM\TFM_resp_dis\features_extracted_multiclass_4c"
PATH_STAGE2_DIR = r"C:\Users\Usuario\Desktop\UNI MARINA\master\TFM\TFM_resp_dis\features_extracted_multiclass_4c_cochleogram_experiment_random"

COCHLEOGRAM_BLOCKS = 8

STAGE1_GRID = [
    {"n_estimators": 200, "max_depth": None, "min_samples_leaf": 2, "class_weight": "balanced"},
    {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 2, "class_weight": "balanced_subsample"},
    {"n_estimators": 200, "max_depth": 15, "min_samples_leaf": 2, "class_weight": "balanced"},
    {"n_estimators": 300, "max_depth": 15, "min_samples_leaf": 2, "class_weight": "balanced_subsample"},
]

STAGE2_SVM_GRID = [
    {"kernel": "rbf", "C": 0.5, "gamma": "scale"},
    {"kernel": "rbf", "C": 1.0, "gamma": "scale"},
    {"kernel": "rbf", "C": 2.0, "gamma": "scale"},
    {"kernel": "rbf", "C": 5.0, "gamma": "scale"},
    {"kernel": "poly", "C": 0.5, "gamma": "scale", "degree": 2},
    {"kernel": "poly", "C": 1.0, "gamma": "scale", "degree": 2},
    {"kernel": "poly", "C": 2.0, "gamma": "scale", "degree": 3},
]

STAGE2_XGB_GRID = [
    {"max_depth": 3, "learning_rate": 0.01, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 1},
    {"max_depth": 3, "learning_rate": 0.03, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 1},
    {"max_depth": 3, "learning_rate": 0.05, "subsample": 0.85, "colsample_bytree": 0.9, "min_child_weight": 1},
    {"max_depth": 4, "learning_rate": 0.03, "subsample": 0.85, "colsample_bytree": 0.9, "min_child_weight": 1},
    {"max_depth": 4, "learning_rate": 0.05, "subsample": 0.85, "colsample_bytree": 0.9, "min_child_weight": 1},
    {"max_depth": 5, "learning_rate": 0.03, "subsample": 0.9, "colsample_bytree": 0.9, "min_child_weight": 3},
    {"max_depth": 5, "learning_rate": 0.05, "subsample": 0.9, "colsample_bytree": 0.9, "min_child_weight": 3},
    {"max_depth": 6, "learning_rate": 0.03, "subsample": 0.9, "colsample_bytree": 0.8, "min_child_weight": 3},
]

SMOTE_GRID = [
    {"name": "minority_k3", "sampling_strategy": "minority", "k_neighbors": 3},
    {"name": "minority_k5", "sampling_strategy": "minority", "k_neighbors": 5},
    {"name": "half_k3", "sampling_strategy": 0.5, "k_neighbors": 3},
    {"name": "half_k5", "sampling_strategy": 0.5, "k_neighbors": 5},
    {"name": "full_k3", "sampling_strategy": 1.0, "k_neighbors": 3},
    {"name": "full_k5", "sampling_strategy": 1.0, "k_neighbors": 5},
]


@dataclass
class Stage1Result:
    params: dict
    score: float
    model: object


@dataclass
class Stage2Result:
    name: str
    kind: str
    params: dict
    score: float
    model: object


def load_stage1_features():
    x_train_raw = np.load(os.path.join(PATH_STAGE1_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(PATH_STAGE1_DIR, "y_train.npy"))
    folds_train = np.load(os.path.join(PATH_STAGE1_DIR, "folds_train.npy"))

    x_val_raw = np.load(os.path.join(PATH_STAGE1_DIR, "X_val.npy"))
    y_val = np.load(os.path.join(PATH_STAGE1_DIR, "y_val.npy"))

    x_test_raw = np.load(os.path.join(PATH_STAGE1_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(PATH_STAGE1_DIR, "y_test.npy"))

    train_mean = np.mean(x_train_raw, axis=0, keepdims=True)
    train_std = np.std(x_train_raw, axis=0, keepdims=True)

    x_train = (x_train_raw - train_mean) / (train_std + 1e-8)
    x_val = (x_val_raw - train_mean) / (train_std + 1e-8)
    x_test = (x_test_raw - train_mean) / (train_std + 1e-8)

    return x_train, y_train, folds_train, x_val, y_val, x_test, y_test


def load_stage2_features():
    x_train = np.load(os.path.join(PATH_STAGE2_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(PATH_STAGE2_DIR, "y_train.npy"))
    folds_train = np.load(os.path.join(PATH_STAGE2_DIR, "folds_train.npy"))

    x_val = np.load(os.path.join(PATH_STAGE2_DIR, "X_val.npy"))
    y_val = np.load(os.path.join(PATH_STAGE2_DIR, "y_val.npy"))

    x_test = np.load(os.path.join(PATH_STAGE2_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(PATH_STAGE2_DIR, "y_test.npy"))

    return x_train, y_train, folds_train, x_val, y_val, x_test, y_test


def cochleogram_flat_to_block_features(cochleogram_flat: np.ndarray, n_blocks: int = COCHLEOGRAM_BLOCKS) -> np.ndarray:
    side = int(np.sqrt(cochleogram_flat.size))
    if side * side != cochleogram_flat.size:
        raise ValueError(f"Expected a square cochleogram vector, got length {cochleogram_flat.size}.")
    if side % n_blocks != 0:
        raise ValueError(f"Cochleogram side {side} is not divisible by n_blocks={n_blocks}.")

    cochleogram = cochleogram_flat.reshape(side, side)
    block_size = side // n_blocks
    block_features = []

    for row_block in range(n_blocks):
        row_start = row_block * block_size
        row_end = row_start + block_size
        for col_block in range(n_blocks):
            col_start = col_block * block_size
            col_end = col_start + block_size
            block = cochleogram[row_start:row_end, col_start:col_end]
            block_features.extend([
                float(np.mean(block)),
                float(np.std(block)),
            ])

    return np.asarray(block_features, dtype=np.float32)


def transform_stage2_block_features(x_stage2: np.ndarray) -> np.ndarray:
    return np.vstack([cochleogram_flat_to_block_features(sample) for sample in x_stage2])


def apply_smote_if_possible(
    x_train: np.ndarray,
    y_train: np.ndarray,
    sampling_strategy: str | float = "minority",
    k_neighbors: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    class_counts = np.bincount(y_train)
    if len(class_counts) < 2 or np.min(class_counts) < 2:
        return x_train, y_train

    minority_count = int(np.min(class_counts))
    if k_neighbors is None:
        k_neighbors = min(5, max(1, minority_count - 1))
    if k_neighbors < 1:
        return x_train, y_train

    smote = SMOTE(random_state=42, k_neighbors=k_neighbors, sampling_strategy=sampling_strategy)
    return smote.fit_resample(x_train, y_train)


def fit_stage1_candidate(params, x_train, y_train_bin):
    model = RandomForestClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        min_samples_leaf=params["min_samples_leaf"],
        min_samples_split=params["min_samples_leaf"] * 2,
        class_weight=params["class_weight"],
        random_state=42,
        n_jobs=-1,
    )
    model.fit(x_train, y_train_bin)
    return model


def evaluate_stage1_candidate(params, x_train, y_train, folds_train):
    y_train_bin = (y_train > 0).astype(int)
    fold_scores = []

    for fold_idx in range(5):
        val_mask = folds_train == fold_idx
        tr_mask = ~val_mask

        x_tr = x_train[tr_mask]
        y_tr = y_train_bin[tr_mask]
        x_va = x_train[val_mask]
        y_va = y_train_bin[val_mask]

        model = fit_stage1_candidate(params, x_tr, y_tr)
        preds = model.predict(x_va)
        fold_scores.append(f1_score(y_va, preds, average="binary", zero_division=0))

    return float(np.mean(fold_scores))


def train_best_stage1(x_train, y_train, folds_train):
    best_params = None
    best_score = -1.0

    print("\n" + "=" * 80)
    print(" STAGE 1: RF BINARY CV FOR COUGH / NO-COUGH")
    print("=" * 80)

    for idx, params in enumerate(STAGE1_GRID, start=1):
        score = evaluate_stage1_candidate(params, x_train, y_train, folds_train)
        print(f"[{idx}/{len(STAGE1_GRID)}] params={params} -> mean CV binary F1: {score:.4f}")
        if score > best_score:
            best_score = score
            best_params = params

    final_model = fit_stage1_candidate(best_params, x_train, (y_train > 0).astype(int))

    print("\n" + "=" * 80)
    print(" BEST STAGE 1 CONFIGURATION")
    print("=" * 80)
    print(f"params: {best_params}")
    print(f"mean CV binary F1: {best_score:.4f}")

    return Stage1Result(params=best_params, score=best_score, model=final_model)


def prepare_stage2_data(x_train_stage2, y_train_stage2, folds_train_stage2):
    mask_cough = np.isin(y_train_stage2, [1, 2])
    x_train_cough = x_train_stage2[mask_cough]
    y_train_cough = (y_train_stage2[mask_cough] - 1).astype(int)
    folds_train_cough = folds_train_stage2[mask_cough]

    return x_train_cough, y_train_cough, folds_train_cough


STAGE2_SVM_C_GRID = [0.5, 1.0, 2.0]


def fit_stage2_candidate(kind, params, x_train, y_train, smote_cfg=None):
    final_smote_cfg = smote_cfg or {"name": "minority_k5", "sampling_strategy": "minority", "k_neighbors": 5}
    x_train_res, y_train_res = apply_smote_if_possible(
        x_train,
        y_train,
        sampling_strategy=final_smote_cfg["sampling_strategy"],
        k_neighbors=final_smote_cfg["k_neighbors"],
    )

    if kind == "svm_rbf":
        model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    SVC(
                        kernel="rbf",
                        C=params["C"],
                        gamma=params["gamma"],
                        class_weight="balanced",
                        probability=True,
                        random_state=42,
                    ),
                ),
            ]
        )
        model.fit(x_train_res, y_train_res)
        return model

    if kind == "svm_poly":
        model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    SVC(
                        kernel="poly",
                        C=params["C"],
                        gamma=params["gamma"],
                        degree=params["degree"],
                        class_weight="balanced",
                        probability=True,
                        random_state=42,
                    ),
                ),
            ]
        )
        model.fit(x_train_res, y_train_res)
        return model

    if kind == "xgb":
        positive_count = max(int(np.sum(y_train == 1)), 1)
        negative_count = max(int(np.sum(y_train == 0)), 1)
        scale_pos_weight = negative_count / positive_count

        model = XGBClassifier(
            n_estimators=200,
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            subsample=params["subsample"],
            colsample_bytree=params["colsample_bytree"],
            min_child_weight=params["min_child_weight"],
            scale_pos_weight=scale_pos_weight,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )
        model.fit(x_train_res, y_train_res)
        return model

    raise ValueError(f"Unsupported stage2 kind: {kind}")


def evaluate_stage2_candidate(kind, params, x_train_cough, y_train_cough, folds_train_cough, smote_cfg=None):
    fold_scores = []

    for fold_idx in range(5):
        val_mask = folds_train_cough == fold_idx
        tr_mask = ~val_mask

        x_tr = x_train_cough[tr_mask]
        y_tr = y_train_cough[tr_mask]
        x_va = x_train_cough[val_mask]
        y_va = y_train_cough[val_mask]

        model = fit_stage2_candidate(kind, params, x_tr, y_tr, smote_cfg=smote_cfg)
        preds = model.predict(x_va)
        fold_scores.append(f1_score(y_va, preds, average="macro", zero_division=0))

    mean_score = float(np.mean(fold_scores))
    smote_label = smote_cfg["name"] if smote_cfg else "default"
    print(f"kind={kind} | smote={smote_label} | params={params} -> mean CV macro F1: {mean_score:.4f}")
    return mean_score


def train_best_stage2(x_train_stage2, y_train_stage2, folds_train_stage2):
    x_train_cough, y_train_cough, folds_train_cough = prepare_stage2_data(
        x_train_stage2,
        y_train_stage2,
        folds_train_stage2,
    )

    print("\n" + "=" * 80)
    print(" STAGE 2: DRY VS WET WITH BLOCK COCHLEOGRAM FEATURES")
    print("=" * 80)

    candidates = []
    for smote_cfg in SMOTE_GRID:
        for params in STAGE2_SVM_GRID:
            kind = "svm_rbf" if params["kernel"] == "rbf" else "svm_poly"
            score = evaluate_stage2_candidate(
                kind,
                params,
                x_train_cough,
                y_train_cough,
                folds_train_cough,
                smote_cfg=smote_cfg,
            )
            candidates.append(
                Stage2Result(
                    name=f"{kind}_{params}_smote_{smote_cfg['name']}",
                    kind=kind,
                    params={**params, "smote": smote_cfg},
                    score=score,
                    model=None,
                )
            )

        for params in STAGE2_XGB_GRID:
            score = evaluate_stage2_candidate(
                "xgb",
                params,
                x_train_cough,
                y_train_cough,
                folds_train_cough,
                smote_cfg=smote_cfg,
            )
            candidates.append(
                Stage2Result(
                    name=f"xgb_depth{params['max_depth']}_lr{params['learning_rate']}_smote_{smote_cfg['name']}",
                    kind="xgb",
                    params={**params, "smote": smote_cfg},
                    score=score,
                    model=None,
                )
            )

    best_stage2 = max(candidates, key=lambda item: item.score)

    final_model = fit_stage2_candidate(
        best_stage2.kind,
        {k: v for k, v in best_stage2.params.items() if k != "smote"},
        x_train_cough,
        y_train_cough,
        smote_cfg=best_stage2.params.get("smote"),
    )
    best_stage2.model = final_model

    print("\n" + "=" * 80)
    print(" BEST STAGE 2 CONFIGURATION")
    print("=" * 80)
    print(f"name: {best_stage2.name}")
    print(f"kind: {best_stage2.kind}")
    print(f"params: {best_stage2.params}")
    print(f"mean CV macro F1: {best_stage2.score:.4f}")

    return best_stage2


def predict_hierarchical(x_stage1_input, x_stage2_input, stage1_model, stage2_result):
    preds_stage1 = stage1_model.predict(x_stage1_input)
    final_preds = np.zeros(len(x_stage1_input), dtype=int)

    mask_cough = preds_stage1 == 1
    if np.any(mask_cough):
        x_cough = x_stage2_input[mask_cough]
        preds_stage2 = stage2_result.model.predict(x_cough)
        final_preds[mask_cough] = preds_stage2 + 1

    return final_preds


def evaluate_global_pipeline(
    stage1_model,
    stage2_result,
    x_train_stage1,
    x_train_stage2,
    y_train,
    x_val_stage1,
    x_val_stage2,
    y_val,
    x_test_stage1,
    x_test_stage2,
    y_test,
):
    print("\n" + "=" * 80)
    print(" GLOBAL EVALUATION OF THE 2-STAGE PIPELINE")
    print("=" * 80)

    for split_name, x_split_stage1, x_split_stage2, y_split in [
        ("TRAIN      ", x_train_stage1, x_train_stage2, y_train),
        ("VALIDATION ", x_val_stage1, x_val_stage2, y_val),
        ("TEST (BLIND)", x_test_stage1, x_test_stage2, y_test),
    ]:
        mask_eval = y_split != 3
        x_split_stage1_eval = x_split_stage1[mask_eval]
        x_split_stage2_eval = x_split_stage2[mask_eval]
        y_split_eval = y_split[mask_eval]

        y_pred = predict_hierarchical(x_split_stage1_eval, x_split_stage2_eval, stage1_model, stage2_result)
        acc = accuracy_score(y_split_eval, y_pred)
        prec_macro = precision_score(y_split_eval, y_pred, average="macro", zero_division=0)
        rec_macro = recall_score(y_split_eval, y_pred, average="macro", zero_division=0)
        f1_macro = f1_score(y_split_eval, y_pred, average="macro", zero_division=0)
        print(
            f"{split_name} -> Accuracy: {acc:.4f} | Precision: {prec_macro:.4f} | "
            f"Recall: {rec_macro:.4f} | Macro F1: {f1_macro:.4f} | "
            f"(Unknown excluded: {np.sum(y_split == 3)})"
        )

    mask_test_eval = y_test != 3
    y_pred_test = predict_hierarchical(
        x_test_stage1[mask_test_eval],
        x_test_stage2[mask_test_eval],
        stage1_model,
        stage2_result,
    )
    y_test_eval = y_test[mask_test_eval]

    print("\nClassification report on TEST (3 classes, Unknown excluded):")
    print(classification_report(y_test_eval, y_pred_test, target_names=CLASS_NAMES_GLOBAL, digits=4, zero_division=0))

    print("\nConfusion matrix on TEST:")
    cm = confusion_matrix(y_test_eval, y_pred_test)
    print(pd.DataFrame(cm, index=CLASS_NAMES_GLOBAL, columns=CLASS_NAMES_GLOBAL))

    mask_test_cough = np.isin(y_test, [1, 2])
    x_test_cough = x_test_stage2[mask_test_cough]
    y_test_cough = y_test[mask_test_cough] - 1

    preds_stage2_test = stage2_result.model.predict(x_test_cough)

    print("\nStage 2 only, on TEST cough samples:")
    print(
        classification_report(
            y_test_cough,
            preds_stage2_test,
            target_names=CLASS_NAMES_STAGE2,
            digits=4,
            zero_division=0,
        )
    )


def main():
    print("=" * 80)
    print(" TWO-STAGE TRAINING: MFCC STAGE 1 + BLOCK-COCHLEOGRAM STAGE 2")
    print("=" * 80)

    x_train_s1, y_train_s1, folds_train_s1, x_val_s1, y_val_s1, x_test_s1, y_test_s1 = load_stage1_features()
    x_train_s2, y_train_s2, folds_train_s2, x_val_s2, y_val_s2, x_test_s2, y_test_s2 = load_stage2_features()

    if not np.array_equal(y_train_s1, y_train_s2):
        raise ValueError("Stage 1 and stage 2 train labels are not aligned.")
    if not np.array_equal(folds_train_s1, folds_train_s2):
        raise ValueError("Stage 1 and stage 2 folds are not aligned.")
    if not np.array_equal(y_val_s1, y_val_s2) or not np.array_equal(y_test_s1, y_test_s2):
        raise ValueError("Stage 1 and stage 2 split labels are not aligned.")

    print(f"Loaded stage 1 train shape: {x_train_s1.shape}, val shape: {x_val_s1.shape}, test shape: {x_test_s1.shape}")
    print(f"Loaded stage 2 train shape: {x_train_s2.shape}, val shape: {x_val_s2.shape}, test shape: {x_test_s2.shape}")
    print(f"Stage 1 labels: {np.unique((y_train_s1 > 0).astype(int), return_counts=True)}")

    x_train_s2_blocks = transform_stage2_block_features(x_train_s2)
    x_val_s2_blocks = transform_stage2_block_features(x_val_s2)
    x_test_s2_blocks = transform_stage2_block_features(x_test_s2)

    print(f"Stage 2 block features shape: {x_train_s2_blocks.shape}")

    stage1_start = time.time()
    stage1_result = train_best_stage1(x_train_s1, y_train_s1, folds_train_s1)
    print(f"Stage 1 training time: {time.time() - stage1_start:.2f}s")

    stage2_start = time.time()
    stage2_result = train_best_stage2(x_train_s2_blocks, y_train_s2, folds_train_s2)
    print(f"Stage 2 training time: {time.time() - stage2_start:.2f}s")

    evaluate_global_pipeline(
        stage1_result.model,
        stage2_result,
        x_train_s1,
        x_train_s2_blocks,
        y_train_s1,
        x_val_s1,
        x_val_s2_blocks,
        y_val_s1,
        x_test_s1,
        x_test_s2_blocks,
        y_test_s1,
    )


if __name__ == "__main__":
    main()
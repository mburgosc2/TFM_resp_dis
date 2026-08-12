import os
import time
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    classification_report, confusion_matrix
)

# Mapeo de nombres
CLASS_NAMES_ALL = ['0: No-Cough', '1: Dry', '2: Wet', '3: Unknown']
CLASS_NAMES_COUGH = ['1: Dry', '2: Wet', '3: Unknown']

# =====================================================================
# 📂 FASE 1: CARGA DE CARACTERÍSTICAS
# =====================================================================
PATH_FEATURES_DIR = r"C:\Users\Usuario\Desktop\UNI MARINA\master\TFM\TFM_resp_dis\features_extracted_multiclass_4c"

print("=" * 80)
print(" Loading Pre-extracted Features - 2x XGBoost + SMOTE in 5-FOLD CV...")
print("=" * 80)

X_train_raw = np.load(os.path.join(PATH_FEATURES_DIR, "X_train.npy"))
y_train = np.load(os.path.join(PATH_FEATURES_DIR, "y_train.npy"))
folds_train = np.load(os.path.join(PATH_FEATURES_DIR, "folds_train.npy"))

X_val_raw = np.load(os.path.join(PATH_FEATURES_DIR, "X_val.npy"))
y_val = np.load(os.path.join(PATH_FEATURES_DIR, "y_val.npy"))

X_test_raw = np.load(os.path.join(PATH_FEATURES_DIR, "X_test.npy"))
y_test = np.load(os.path.join(PATH_FEATURES_DIR, "y_test.npy"))

# Normalización Z-score
train_mean = np.mean(X_train_raw, axis=0, keepdims=True)
train_std = np.std(X_train_raw, axis=0, keepdims=True)

X_train = (X_train_raw - train_mean) / (train_std + 1e-8)
X_val = (X_val_raw - train_mean) / (train_std + 1e-8)
X_test = (X_test_raw - train_mean) / (train_std + 1e-8)

# =====================================================================
# 🟢 ETAPA 1: DETECTOR BINARIO (NO-COUGH vs COUGH)
# =====================================================================
print("\n" + "=" * 80)
print(" 🟢 ETAPA 1: ENTRENANDO DETECTOR BINARIO (XGBoost)")
print("=" * 80)

y_train_bin = (y_train > 0).astype(int)
scale_pos_weight = (y_train_bin == 0).sum() / (y_train_bin == 1).sum()

model_stage1 = XGBClassifier(
    n_estimators=150,
    learning_rate=0.05,
    max_depth=5,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1
)

start_s1 = time.time()
model_stage1.fit(X_train, y_train_bin)
print(f"✓ Etapa 1 entrenada en {time.time() - start_s1:.2f}s")

# =====================================================================
# 🔵 ETAPA 2: GRID SEARCH 5-FOLDS CON SMOTE DENTRO DEL LOOP DE CV
# =====================================================================
print("\n" + "=" * 80)
print(" 🔵 ETAPA 2: GRID SEARCH 5-FOLDS CON SMOTE SOBRE TOSES (Dry, Wet, Unknown)")
print(" Target Optimization Metric: Macro F1-Score")
print("=" * 80)

# Filtrar solo toses
mask_train_cough = (y_train > 0)
X_train_cough = X_train[mask_train_cough]
y_train_cough = y_train[mask_train_cough] - 1  # Map [1,2,3] -> [0,1,2]
folds_train_cough = folds_train[mask_train_cough]

param_grid = {
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.03, 0.05],
    'subsample': [0.7, 0.85],
    'colsample_bytree': [0.6, 0.8]
}

best_score = -1.0
best_params = {}
combo_counter = 0

total_combos = (len(param_grid['max_depth']) * 
                len(param_grid['learning_rate']) * 
                len(param_grid['subsample']) * 
                len(param_grid['colsample_bytree']))

start_grid_time = time.time()

for md in param_grid['max_depth']:
    for lr in param_grid['learning_rate']:
        for ss in param_grid['subsample']:
            for cs in param_grid['colsample_bytree']:
                combo_counter += 1
                fold_f1_scores = []

                for fold_idx in range(5):
                    cv_val_mask = (folds_train_cough == fold_idx)
                    cv_train_mask = ~cv_val_mask
                    
                    X_cv_tr_raw, y_cv_tr_raw = X_train_cough[cv_train_mask], y_train_cough[cv_train_mask]
                    X_cv_va, y_cv_va = X_train_cough[cv_val_mask], y_train_cough[cv_val_mask]
                    
                    # ⚡ APLICAR SMOTE ÚNICAMENTE EN EL TRAIN DE ESTE FOLD (Evita Data Leakage)
                    smote = SMOTE(random_state=42, k_neighbors=5)
                    X_cv_tr, y_cv_tr = smote.fit_resample(X_cv_tr_raw, y_cv_tr_raw)
                    
                    xgb_fold = XGBClassifier(
                        objective='multi:softprob',
                        num_class=3,
                        n_estimators=150,
                        max_depth=md,
                        learning_rate=lr,
                        subsample=ss,
                        colsample_bytree=cs,
                        random_state=42,
                        n_jobs=-1
                    )
                    
                    xgb_fold.fit(X_cv_tr, y_cv_tr)
                    preds_fold = xgb_fold.predict(X_cv_va)
                    
                    f1_macro = f1_score(y_cv_va, preds_fold, average='macro', zero_division=0)
                    fold_f1_scores.append(f1_macro)

                mean_cv_f1 = np.mean(fold_f1_scores)
                
                if mean_cv_f1 > best_score:
                    best_score = mean_cv_f1
                    best_params = {'max_depth': md, 'learning_rate': lr, 'subsample': ss, 'colsample_bytree': cs}
                
                print(f"[{combo_counter}/{total_combos}] depth={md}, lr={lr}, sub={ss}, col={cs} => Mean CV Macro F1: {mean_cv_f1:.4f}")

print("\n" + "=" * 80)
print(f" 🏆 MEJORES HIPERPARÁMETROS ETAPA 2 (Macro F1 CV: {best_score:.4f}):")
for k, v in best_params.items():
    print(f"    - {k}: {v}")
print(f"    - Búsqueda completada en {time.time() - start_grid_time:.2f}s")
print("=" * 80)

# ⚡ APLICAR SMOTE EN TODO EL CONJUNTO TRAIN DE TOSES PARA ENTRENAR EL MODELO FINAL DE ETAPA 2
smote_final = SMOTE(random_state=42, k_neighbors=5)
X_train_cough_resampled, y_train_cough_resampled = smote_final.fit_resample(X_train_cough, y_train_cough)

model_stage2 = XGBClassifier(
    objective='multi:softprob',
    num_class=3,
    n_estimators=200,
    max_depth=best_params['max_depth'],
    learning_rate=best_params['learning_rate'],
    subsample=best_params['subsample'],
    colsample_bytree=best_params['colsample_bytree'],
    random_state=42,
    n_jobs=-1
)
model_stage2.fit(X_train_cough_resampled, y_train_cough_resampled)

# =====================================================================
# ⚙️ INFERENCIA HÍBRIDA EN CASCADA Y EVALUACIÓN FINAL
# =====================================================================
def predict_hierarchical(X_input):
    preds_s1 = model_stage1.predict(X_input)
    final_preds = np.zeros(len(X_input), dtype=int)
    
    mask_cough = (preds_s1 == 1)
    if np.sum(mask_cough) > 0:
        preds_s2 = model_stage2.predict(X_input[mask_cough])
        final_preds[mask_cough] = preds_s2 + 1
            
    return final_preds

print("\n" + "=" * 80)
print(" 📊 EVALUACIÓN GLOBAL DEL PIPELINE EN CASCADA (4 CLASES)")
print("=" * 80)

for split_name, X_set, y_true in [
    ("TRAIN      ", X_train, y_train),
    ("VALIDATION ", X_val, y_val),
    ("TEST (BLIND)", X_test, y_test)
]:
    y_pred = predict_hierarchical(X_set)
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    print(f"📊 Conjunto {split_name} -> Accuracy: {acc:.4f} | Macro F1-Score: {f1_macro:.4f}")

# Reporte detallado en TEST
y_pred_test = predict_hierarchical(X_test)
print("\n📋 Classification Report Global (4 Clases):")
print(classification_report(y_test, y_pred_test, target_names=CLASS_NAMES_ALL, digits=4, zero_division=0))

print("\n🧩 Matriz de Confusión Global:")
cm = confusion_matrix(y_test, y_pred_test)
print(pd.DataFrame(cm, index=CLASS_NAMES_ALL, columns=CLASS_NAMES_ALL))

# Desglose Etapa 2 independiente
print("\n" + "=" * 80)
print(" 🔵 EVALUACIÓN INDEPENDIENTE ETAPA 2 EN TEST (Solo Toses Reales)")
print("=" * 80)

mask_test_cough = (y_test > 0)
X_test_cough = X_test[mask_test_cough]
y_test_cough = y_test[mask_test_cough] - 1

preds_s2_test = model_stage2.predict(X_test_cough)
print(classification_report(y_test_cough, preds_s2_test, target_names=CLASS_NAMES_COUGH, digits=4, zero_division=0))
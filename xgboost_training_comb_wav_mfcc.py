import os
import time
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, classification_report, confusion_matrix
)

CLASS_NAMES = ['0: No-Cough', '1: Dry', '2: Wet', '3: Unknown']

# =====================================================================
# 📂 FASE 1: CARGA Y CONCATENACIÓN DE CARACTERÍSTICAS (MFCC + WAVELETS)
# =====================================================================
PATH_MFCC_DIR = r"C:\Users\Usuario\Desktop\UNI MARINA\master\TFM\TFM_resp_dis\features_extracted_multiclass_4c"
PATH_WAVELET_DIR = r"C:\Users\Usuario\Desktop\UNI MARINA\master\TFM\TFM_resp_dis\features_extracted_wavelets_4c"

print("=" * 80)
print(" 🔗 CONCATENANDO CARACTERÍSTICAS: MFCCs (117) + WAVELETS (38) = 155 FEATURES")
print("=" * 80)

# Cargar MFCCs
X_tr_mfcc = np.load(os.path.join(PATH_MFCC_DIR, "X_train.npy"))
X_va_mfcc = np.load(os.path.join(PATH_MFCC_DIR, "X_val.npy"))
X_te_mfcc = np.load(os.path.join(PATH_MFCC_DIR, "X_test.npy"))

y_train = np.load(os.path.join(PATH_MFCC_DIR, "y_train.npy"))
folds_train = np.load(os.path.join(PATH_MFCC_DIR, "folds_train.npy"))
y_val = np.load(os.path.join(PATH_MFCC_DIR, "y_val.npy"))
y_test = np.load(os.path.join(PATH_MFCC_DIR, "y_test.npy"))

# Cargar Wavelets
X_tr_wav = np.load(os.path.join(PATH_WAVELET_DIR, "X_train.npy"))
X_va_wav = np.load(os.path.join(PATH_WAVELET_DIR, "X_val.npy"))
X_te_wav = np.load(os.path.join(PATH_WAVELET_DIR, "X_test.npy"))

# Concatenar a lo largo del eje de características (axis=1)
X_train_raw = np.hstack([X_tr_mfcc, X_tr_wav])
X_val_raw = np.hstack([X_va_mfcc, X_va_wav])
X_test_raw = np.hstack([X_te_mfcc, X_te_wav])

print(f"✓ Matriz X_train fusionada: {X_train_raw.shape}")
print(f"✓ Matriz X_val fusionada:   {X_val_raw.shape}")
print(f"✓ Matriz X_test fusionada:  {X_test_raw.shape}")

# =====================================================================
# ⚖️ FASE 2: NORMALIZACIÓN ESTRICTA Z-SCORE SOBRE EL VECTOR FUSIONADO
# =====================================================================
train_mean = np.mean(X_train_raw, axis=0, keepdims=True)
train_std = np.std(X_train_raw, axis=0, keepdims=True)

X_train = (X_train_raw - train_mean) / (train_std + 1e-8)
X_val = (X_val_raw - train_mean) / (train_std + 1e-8)
X_test = (X_test_raw - train_mean) / (train_std + 1e-8)

# =====================================================================
# 🔍 FASE 3: GRID SEARCH DE HIPERPARÁMETROS (155 FEATURES, 5-FOLDS)
# =====================================================================
print("\n" + "=" * 80)
print(" 🔄 INICIANDO GRID SEARCH XGBOOST SOBRE FUSIÓN (5-FOLDS ESTRATIFICADOS)")
print(" Target Optimization Metric: Macro F1-Score")
print("=" * 80)

param_grid = {
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.03, 0.05],
    'subsample': [0.7, 0.8],
    'colsample_bytree': [0.6, 0.8],  # Submuestreo de características clave al tener 155
    'min_child_weight': [1, 3]
}

best_score = -1.0
best_params = {}
combo_counter = 0

total_combos = (len(param_grid['max_depth']) * 
                len(param_grid['learning_rate']) * 
                len(param_grid['subsample']) * 
                len(param_grid['colsample_bytree']) * 
                len(param_grid['min_child_weight']))

start_grid_time = time.time()

for depth in param_grid['max_depth']:
    for lr in param_grid['learning_rate']:
        for sub in param_grid['subsample']:
            for col in param_grid['colsample_bytree']:
                for mcw in param_grid['min_child_weight']:
                    combo_counter += 1
                    fold_f1_scores = []

                    for fold_idx in range(5):
                        cv_val_mask = (folds_train == fold_idx)
                        cv_train_mask = ~cv_val_mask
                        
                        X_cv_train, y_cv_train = X_train[cv_train_mask], y_train[cv_train_mask]
                        X_cv_val, y_cv_val = X_train[cv_val_mask], y_train[cv_val_mask]
                        
                        sample_weights_cv = compute_sample_weight('balanced', y_cv_train)
                        
                        xgb_fold = XGBClassifier(
                            n_estimators=150,
                            max_depth=depth,
                            learning_rate=lr,
                            subsample=sub,
                            colsample_bytree=col,
                            min_child_weight=mcw,
                            objective='multi:softprob',
                            num_class=4,
                            random_state=42,
                            n_jobs=-1
                        )
                        
                        xgb_fold.fit(X_cv_train, y_cv_train, sample_weight=sample_weights_cv)
                        preds_fold = xgb_fold.predict(X_cv_val)
                        f1_macro = f1_score(y_cv_val, preds_fold, average='macro', zero_division=0)
                        fold_f1_scores.append(f1_macro)
                    
                    mean_cv_f1 = np.mean(fold_f1_scores)
                    print(f"[{combo_counter}/{total_combos}] Depth: {depth} | LR: {lr:.2f} | Sub: {sub} | Col: {col} | MCW: {mcw} => Mean CV Macro F1: {mean_cv_f1:.4f}")
                    
                    if mean_cv_f1 > best_score:
                        best_score = mean_cv_f1
                        best_params = {
                            'max_depth': depth,
                            'learning_rate': lr,
                            'subsample': sub,
                            'colsample_bytree': col,
                            'min_child_weight': mcw
                        }

print("\n" + "=" * 80)
print(f" 🏆 MEJORES HIPERPARÁMETROS FUSIÓN (Macro F1 CV: {best_score:.4f}):")
for k, v in best_params.items():
    print(f"    - {k}: {v}")
print(f"    - Tiempo total de búsqueda: {time.time() - start_grid_time:.2f}s")
print("=" * 80)

# =====================================================================
# 🎯 FASE 4: ENTRENAMIENTO Y EVALUACIÓN FINAL EN TEST
# =====================================================================
print("\n ENTRENANDO MODELO XGBOOST FUSIONADO ÓPTIMO SOBRE TODO TRAIN...")

sample_weights_train = compute_sample_weight('balanced', y_train)

xgb_best = XGBClassifier(
    n_estimators=200,
    max_depth=best_params['max_depth'],
    learning_rate=best_params['learning_rate'],
    subsample=best_params['subsample'],
    colsample_bytree=best_params['colsample_bytree'],
    min_child_weight=best_params['min_child_weight'],
    objective='multi:softprob',
    num_class=4,
    random_state=42,
    n_jobs=-1
)

start_time = time.time()
xgb_best.fit(X_train, y_train, sample_weight=sample_weights_train)
print(f"✓ Modelo óptimo entrenado en {time.time() - start_time:.2f}s")

# Extracción de métricas
for split_name, X_set, y_true in [
    ("TRAIN      ", X_train, y_train),
    ("VALIDATION ", X_val, y_val),
    ("TEST (BLIND)", X_test, y_test)
]:
    y_pred = xgb_best.predict(X_set)
    y_proba = xgb_best.predict_proba(X_set)
    
    acc = accuracy_score(y_true, y_pred)
    prec_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    auc_ovr = roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro')
    
    print(f"\n📊 Conjunto {split_name}:")
    print(f"   • Accuracy Global  : {acc:.4f}")
    print(f"   • Macro Precision  : {prec_macro:.4f}")
    print(f"   • Macro Recall     : {rec_macro:.4f}")
    print(f"   • Macro F1-Score   : {f1_macro:.4f}")
    print(f"   • Weighted F1-Score: {f1_weighted:.4f}")
    print(f"   • ROC-AUC (OvR)    : {auc_ovr:.4f}")

# =====================================================================
# 🔍 FASE 5: REPORTE DETALLADO Y MATRIZ DE CONFUSIÓN EN TEST
# =====================================================================
print("\n" + "=" * 80)
print(" REPORTE DETALLADO POR CLASE EN TEST SET (XGBOOST + FUSIÓN MFCC/WAVELET)")
print("=" * 80)

y_pred_test = xgb_best.predict(X_test)

print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred_test, target_names=CLASS_NAMES, digits=4, zero_division=0))

print("\n🧩 Matriz de Confusión (Filas: Reales, Columnas: Predichos):")
cm = confusion_matrix(y_test, y_pred_test)
cm_df = pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES)
print(cm_df)

print("\n" + "=" * 80)
print(" Pipeline Fusión XGBoost terminado.")
print("=" * 80)
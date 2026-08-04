import os
import time
import numpy as np
import pandas as pd
import sklearn
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, classification_report, confusion_matrix
)

# Mapeo de nombres para reporte claro
CLASS_NAMES = ['0: No-Cough', '1: Dry', '2: Wet', '3: Unknown']

# =====================================================================
# 📂 FASE 1: CARGA DE CARACTERÍSTICAS PRE-EXTRAÍDAS (MULTICLASE 4C)
# =====================================================================
PATH_FEATURES_DIR = r"C:\Users\Usuario\Desktop\UNI MARINA\master\TFM\TFM_resp_dis\features_extracted_multiclass_4c"


print("=" * 80)
print(" Loading Pre-extracted Features (.npy files) - SVM MULTICLASS...")
print("=" * 80)

X_train_raw = np.load(os.path.join(PATH_FEATURES_DIR, "X_train.npy"))
y_train = np.load(os.path.join(PATH_FEATURES_DIR, "y_train.npy"))
folds_train = np.load(os.path.join(PATH_FEATURES_DIR, "folds_train.npy"))

X_val_raw = np.load(os.path.join(PATH_FEATURES_DIR, "X_val.npy"))
y_val = np.load(os.path.join(PATH_FEATURES_DIR, "y_val.npy"))

X_test_raw = np.load(os.path.join(PATH_FEATURES_DIR, "X_test.npy"))
y_test = np.load(os.path.join(PATH_FEATURES_DIR, "y_test.npy"))

# =====================================================================
# ⚖️ FASE 2: NORMALIZACIÓN ESTRICTA Z-SCORE (ESENCIAL PARA SVM)
# =====================================================================
# Nota: SVM es EXTREMADAMENTE sensible a la escala de las características
train_mean = np.mean(X_train_raw, axis=0, keepdims=True)
train_std = np.std(X_train_raw, axis=0, keepdims=True)

X_train = (X_train_raw - train_mean) / (train_std + 1e-8)
X_val = (X_val_raw - train_mean) / (train_std + 1e-8)
X_test = (X_test_raw - train_mean) / (train_std + 1e-8)

print(f"✓ X_train escalado: {X_train.shape}  | Mean ≈ {X_train.mean():.4f} | Std ≈ {X_train.std():.4f}")

# =====================================================================
# 🔍 FASE 3: GRID SEARCH DE HIPERPARÁMETROS PARA SVM (5-FOLDS)
# =====================================================================
print("\n" + "=" * 80)
print(" 🔄 INICIANDO GRID SEARCH SVM (5-FOLDS ESTRATIFICADOS)")
print(" Target Optimization Metric: Macro F1-Score")
print("=" * 80)

param_grid = {
    'C': [0.5, 1.0, 2.0, 5.0],
    'gamma': ['scale', 'auto', 0.01, 0.001],
    'class_weight': ['balanced']
}

best_score = -1.0
best_params = {}
combo_counter = 0

total_combos = len(param_grid['C']) * len(param_grid['gamma'])

start_grid_time = time.time()

for c_val in param_grid['C']:
    for g_val in param_grid['gamma']:
        combo_counter += 1
        fold_f1_scores = []

        for fold_idx in range(5):
            cv_val_mask = (folds_train == fold_idx)
            cv_train_mask = ~cv_val_mask
            
            X_cv_train, y_cv_train = X_train[cv_train_mask], y_train[cv_train_mask]
            X_cv_val, y_cv_val = X_train[cv_val_mask], y_train[cv_val_mask]
            
            svm_fold = SVC(
                kernel='rbf',
                C=c_val,
                gamma=g_val,
                class_weight='balanced',
                probability=True,
                random_state=42
            )
            
            svm_fold.fit(X_cv_train, y_cv_train)
            preds_fold = svm_fold.predict(X_cv_val)
            f1_macro = f1_score(y_cv_val, preds_fold, average='macro', zero_division=0)
            fold_f1_scores.append(f1_macro)
        
        mean_cv_f1 = np.mean(fold_f1_scores)
        print(f"[{combo_counter}/{total_combos}] C: {c_val:<4} | Gamma: {str(g_val):<6} => Mean CV Macro F1: {mean_cv_f1:.4f}")
        
        if mean_cv_f1 > best_score:
            best_score = mean_cv_f1
            best_params = {'C': c_val, 'gamma': g_val}

print("\n" + "=" * 80)
print(f" 🏆 MEJORES HIPERPARÁMETROS SVM (Macro F1 CV: {best_score:.4f}):")
for k, v in best_params.items():
    print(f"    - {k}: {v}")
print(f"    - Tiempo total de búsqueda: {time.time() - start_grid_time:.2f}s")
print("=" * 80)

# =====================================================================
# 🎯 FASE 4: ENTRENAMIENTO MODELO SVM ÓPTIMO Y EVALUACIÓN FINAL
# =====================================================================
print("\n ENTRENANDO MODELO SVM ÓPTIMO SOBRE TODO TRAIN...")

svm_best = SVC(
    kernel='rbf',
    C=best_params['C'],
    gamma=best_params['gamma'],
    class_weight='balanced',
    probability=True,
    random_state=42
)

start_time = time.time()
svm_best.fit(X_train, y_train)
print(f"✓ Modelo óptimo SVM entrenado en {time.time() - start_time:.2f}s")

# Extracción de métricas
for split_name, X_set, y_true in [
    ("TRAIN      ", X_train, y_train),
    ("VALIDATION ", X_val, y_val),
    ("TEST (BLIND)", X_test, y_test)
]:
    y_pred = svm_best.predict(X_set)
    y_proba = svm_best.predict_proba(X_set)
    
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
print(" REPORTE DETALLADO POR CLASE EN TEST SET (SVM OPTIMIZADO)")
print("=" * 80)

y_pred_test = svm_best.predict(X_test)

print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred_test, target_names=CLASS_NAMES, digits=4, zero_division=0))

print("\n🧩 Matriz de Confusión (Filas: Reales, Columnas: Predichos):")
cm = confusion_matrix(y_test, y_pred_test)
cm_df = pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES)
print(cm_df)

print("\n" + "=" * 80)
print(" Pipeline SVM Grid Search terminado.")
print("=" * 80)
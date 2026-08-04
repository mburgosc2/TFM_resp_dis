import os
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
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
print(" Loading Pre-extracted Features (.npy files) - MULTICLASS 4 CLASSES...")
print("=" * 80)

X_train_raw = np.load(os.path.join(PATH_FEATURES_DIR, "X_train.npy"))
y_train = np.load(os.path.join(PATH_FEATURES_DIR, "y_train.npy"))
folds_train = np.load(os.path.join(PATH_FEATURES_DIR, "folds_train.npy"))

X_val_raw = np.load(os.path.join(PATH_FEATURES_DIR, "X_val.npy"))
y_val = np.load(os.path.join(PATH_FEATURES_DIR, "y_val.npy"))

X_test_raw = np.load(os.path.join(PATH_FEATURES_DIR, "X_test.npy"))
y_test = np.load(os.path.join(PATH_FEATURES_DIR, "y_test.npy"))

# =====================================================================
# ⚖️ FASE 2: NORMALIZACIÓN ESTRICTA Z-SCORE (ANTI-LEAKAGE)
# =====================================================================
train_mean = np.mean(X_train_raw, axis=0, keepdims=True)
train_std = np.std(X_train_raw, axis=0, keepdims=True)

X_train = (X_train_raw - train_mean) / (train_std + 1e-8)
X_val = (X_val_raw - train_mean) / (train_std + 1e-8)
X_test = (X_test_raw - train_mean) / (train_std + 1e-8)

# =====================================================================
# 🔍 FASE 3: GRID SEARCH DE HIPERPARÁMETROS + CLASS WEIGHTS (5-FOLDS CV)
# =====================================================================
print("\n" + "=" * 80)
print(" 🔄 INICIANDO GRID SEARCH CON VALIDACIÓN CRUZADA INTERNA (5-FOLDS)")
print(" Target Optimization Metric: Macro F1-Score")
print("=" * 80)

# 1. Definir diferentes estrategias de penalización de pesos {0: No-Cough, 1: Dry, 2: Wet, 3: Unknown}
param_grid_weights = [
    'balanced',                                   # Estándar inversamente proporcional
    'balanced_subsample',                         # Estándar por bootstrap
    {0: 0.5, 1: 1.0, 2: 3.0, 3: 2.0},             # Moderada penalización a minoritarias
    {0: 0.3, 1: 1.0, 2: 5.0, 3: 4.0},             # Fuerte penalización a minoritarias
    {0: 0.2, 1: 1.0, 2: 8.0, 3: 6.0}              # Agresiva penalización a Wet y Unknown
]

# 2. Definir parámetros de regularización del árbol
param_grid_depth = [6, 8, 10, 12]
param_grid_leaf = [2, 5, 8]

best_score = -1.0
best_params = {}
grid_results = []

combo_counter = 0
total_combos = len(param_grid_weights) * len(param_grid_depth) * len(param_grid_leaf)

for weights in param_grid_weights:
    for depth in param_grid_depth:
        for leaf in param_grid_leaf:
            combo_counter += 1
            fold_f1_scores = []

            for fold_idx in range(5):
                cv_val_mask = (folds_train == fold_idx)
                cv_train_mask = ~cv_val_mask
                
                X_cv_train, y_cv_train = X_train[cv_train_mask], y_train[cv_train_mask]
                X_cv_val, y_cv_val = X_train[cv_val_mask], y_train[cv_val_mask]
                
                rf_fold = RandomForestClassifier(
                    n_estimators=200,             # Suficiente para Grid Search eficiente
                    max_depth=depth,
                    min_samples_leaf=leaf,
                    min_samples_split=leaf * 2,
                    class_weight=weights,
                    random_state=42,
                    n_jobs=-1
                )
                
                rf_fold.fit(X_cv_train, y_cv_train)
                preds_fold = rf_fold.predict(X_cv_val)
                f1_macro = f1_score(y_cv_val, preds_fold, average='macro', zero_division=0)
                fold_f1_scores.append(f1_macro)
            
            mean_cv_f1 = np.mean(fold_f1_scores)
            weights_str = "balanced" if isinstance(weights, str) else f"custom_{weights[2]}x"
            
            grid_results.append({
                'weights': weights_str,
                'max_depth': depth,
                'min_samples_leaf': leaf,
                'mean_cv_f1_macro': mean_cv_f1
            })
            
            print(f"[{combo_counter}/{total_combos}] Weights: {weights_str:<12} | Depth: {depth:<2} | MinLeaf: {leaf:<2} => Mean CV Macro F1: {mean_cv_f1:.4f}")
            
            if mean_cv_f1 > best_score:
                best_score = mean_cv_f1
                best_params = {
                    'class_weight': weights,
                    'max_depth': depth,
                    'min_samples_leaf': leaf,
                    'min_samples_split': leaf * 2
                }

print("\n" + "=" * 80)
print(f" 🏆 MEJORES HIPERPARÁMETROS ENCONTRADOS (Macro F1 CV: {best_score:.4f}):")
print(f"    - class_weight:     {best_params['class_weight']}")
print(f"    - max_depth:        {best_params['max_depth']}")
print(f"    - min_samples_leaf: {best_params['min_samples_leaf']}")
print("=" * 80)

# =====================================================================
# 🎯 FASE 4: ENTRENAMIENTO MODELO ÓPTIMO Y EVALUACIÓN FINAL
# =====================================================================
print("\n ENTRENANDO MODELO FINAL OPTIMIZADO SOBRE TODO TRAIN...")

rf_best = RandomForestClassifier(
    n_estimators=300,
    max_depth=best_params['max_depth'],
    min_samples_leaf=best_params['min_samples_leaf'],
    min_samples_split=best_params['min_samples_split'],
    class_weight=best_params['class_weight'],
    random_state=42,
    n_jobs=-1
)

start_time = time.time()
rf_best.fit(X_train, y_train)
print(f"✓ Modelo óptimo entrenado en {time.time() - start_time:.2f}s")

# Extracción de métricas
for split_name, X_set, y_true in [
    ("TRAIN      ", X_train, y_train),
    ("VALIDATION ", X_val, y_val),
    ("TEST (BLIND)", X_test, y_test)
]:
    y_pred = rf_best.predict(X_set)
    y_proba = rf_best.predict_proba(X_set)
    
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
print(" REPORTE DETALLADO POR CLASE EN TEST SET (MODELO OPTIMIZADO)")
print("=" * 80)

y_pred_test = rf_best.predict(X_test)

print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred_test, target_names=CLASS_NAMES, digits=4, zero_division=0))

print("\n🧩 Matriz de Confusión (Filas: Reales, Columnas: Predichos):")
cm = confusion_matrix(y_test, y_pred_test)
cm_df = pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES)
print(cm_df)
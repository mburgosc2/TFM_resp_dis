import os
import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# =====================================================================
# 📂 FASE 1: CARGA DE CARACTERÍSTICAS PRE-EXTRAÍDAS (117 DESCRIPTORES)
# =====================================================================
PATH_FEATURES_DIR = r"C:\Users\Usuario\Desktop\UNI MARINA\master\TFM\TFM_resp_dis\features_extracted_experiment_random"

print("=" * 80)
# Las cargas se realizan sobre las matrices de NumPy salvadas por el extractor de audio
print(" Loading Pre-extracted Features (.npy files)...")
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
print("\n" + "=" * 80)
print(" PARÁMETROS DE NORMALIZACIÓN (Calculados únicamente de TRAIN)")
print("=" * 80)

train_mean = np.mean(X_train_raw, axis=0, keepdims=True)
train_std = np.std(X_train_raw, axis=0, keepdims=True)

# Normalización Z-Score evitando divisiones por cero (épsilon = 1e-8)
X_train = (X_train_raw - train_mean) / (train_std + 1e-8)
X_val = (X_val_raw - train_mean) / (train_std + 1e-8)
X_test = (X_test_raw - train_mean) / (train_std + 1e-8)

print(f"✓ X_train escalado: {X_train.shape}  | Mean ≈ {X_train.mean():.4f} | Std ≈ {X_train.std():.4f}")
print(f"✓ X_val escalado:   {X_val.shape}")
print(f"✓ X_test escalado:  {X_test.shape}")

# Guardar parámetros históricos para inferencia en producción (Opcional)
normalization_params = {'mean': train_mean, 'std': train_std}
np.save(os.path.join(PATH_FEATURES_DIR, 'normalization_params.npy'), normalization_params, allow_pickle=True)

# =====================================================================
# 🌲 FASE 3: VALIDACIÓN CRUZADA INTERNA (5-FOLDS)
# =====================================================================
print("\n" + "=" * 80)
print(" 🔄 EJECUTANDO VALIDACIÓN CRUZADA (5-FOLDS ESTRATIFICADOS)")
print("=" * 80)

fold_f1_scores = []

for fold_idx in range(5):
    # Crear máscaras booleanas basadas en la columna generada en la fase de partición
    cv_val_mask = (folds_train == fold_idx)
    cv_train_mask = ~cv_val_mask
    
    X_cv_train, y_cv_train = X_train[cv_train_mask], y_train[cv_train_mask]
    X_cv_val, y_cv_val = X_train[cv_val_mask], y_train[cv_val_mask]
    
    # Instanciar el bosque con hiperparámetros estables contra sobreajuste
    rf_fold = RandomForestClassifier(
        n_estimators=300,        # Aumentado a 300 para estabilizar la varianza espectral
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced', # Crucial debido al desbalance de clases del dataset maestro
        random_state=42,
        n_jobs=1
    )
    
    rf_fold.fit(X_cv_train, y_cv_train)
    preds_fold = rf_fold.predict(X_cv_val)
    f1 = f1_score(y_cv_val, preds_fold, zero_division=0)
    fold_f1_scores.append(f1)
    print(f"   • Fold {fold_idx} - F1-Score: {f1:.4f}")

print(f"\n📈 RENDIMIENTO MEDIO DEL MODELO EN VALIDACIÓN CRUZADA: {np.mean(fold_f1_scores):.4f}")

# =====================================================================
# 🎯 FASE 4: ENTRENAMIENTO FINAL Y EVALUACIÓN DEL TEST BLINDADO
# =====================================================================
print("\n" + "=" * 80)
print(" ENTRENAMIENTO GENERAL Y EVALUACIÓN FINAL EN TEST SET")
print("=" * 80)

rf_final = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

start_time = time.time()
rf_final.fit(X_train, y_train)
print(f"✓ Modelo final entrenado sobre el dataset completo en {time.time() - start_time:.2f}s")

# Extracción de métricas
for split_name, X_set, y_true in [
    ("TRAIN      ", X_train, y_train),
    ("VALIDATION ", X_val, y_val),
    ("TEST (BLIND)", X_test, y_test)
]:
    y_pred = rf_final.predict(X_set)
    y_proba = rf_final.predict_proba(X_set)[:, 1]
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_proba)
    
    print(f"\n📊 Conjunto {split_name}:")
    print(f"   • Accuracy:  {acc:.4f}")
    print(f"   • Precision: {prec:.4f}")
    print(f"   • Recall:    {rec:.4f} (Sensibilidad)")
    print(f"   • F1-Score:  {f1:.4f}")
    print(f"   • ROC-AUC:   {auc:.4f}")

print("\n" + "=" * 80)
print(" Pipeline terminado. Datos listos para la discusión de resultados.")
print("=" * 80)
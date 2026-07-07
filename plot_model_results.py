import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

# =====================================================================
# 📂 FASE 1: CARGA Y PREPARACIÓN DE DATOS
# =====================================================================
PATH_FEATURES_DIR = r"C:\Users\Usuario\Desktop\UNI MARINA\master\TFM\TFM_resp_dis\features_extracted_experiment_random"
PATH_OUTPUT_GRAPHS = r"C:\Users\Usuario\Desktop\UNI MARINA\master\TFM\TFM_resp_dis\graphs_results_experiment_random"
os.makedirs(PATH_OUTPUT_GRAPHS, exist_ok=True)

X_train_raw = np.load(os.path.join(PATH_FEATURES_DIR, "X_train.npy"))
y_train = np.load(os.path.join(PATH_FEATURES_DIR, "y_train.npy"))
X_test_raw = np.load(os.path.join(PATH_FEATURES_DIR, "X_test.npy"))
y_test = np.load(os.path.join(PATH_FEATURES_DIR, "y_test.npy"))

# Normalización Z-Score idéntica (usando parámetros de TRAIN)
train_mean = np.mean(X_train_raw, axis=0, keepdims=True)
train_std = np.std(X_train_raw, axis=0, keepdims=True)
X_train = (X_train_raw - train_mean) / (train_std + 1e-8)
X_test = (X_test_raw - train_mean) / (train_std + 1e-8)

# Re-entrenar modelo base final
rf_final = RandomForestClassifier(
    n_estimators=300, max_depth=15, min_samples_split=5,
    min_samples_leaf=2, class_weight='balanced', random_state=42, n_jobs=1
)
rf_final.fit(X_train, y_train)

# Predicciones para el conjunto ciego de TEST
y_test_pred = rf_final.predict(X_test)
y_test_proba = rf_final.predict_proba(X_test)[:, 1]

# =====================================================================
# 📊 GRÁFICA 1: MATRIZ DE CONFUSIÓN (TEST SET)
# =====================================================================
cm = confusion_matrix(y_test, y_test_pred)

# Calcular porcentajes por fila (valores reales)
cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

# Crear las etiquetas combinadas (Conteo absoluto y Porcentaje)
labels = [f"{v}\n({p:.1f}%)" for v, p in zip(cm.flatten(), cm_percent.flatten())]
labels = np.asarray(labels).reshape(2, 2)

plt.figure(figsize=(7, 5.5))
sns.heatmap(
    cm, annot=labels, fmt="", cmap="Blues", cbar=False,
    xticklabels=["Control (No-Tos)", "Tos"],
    yticklabels=["Control (No-Tos)", "Tos"],
    annot_kws={"fontsize": 12, "weight": "bold"}
)

plt.title("Matriz de Confusión - Conjunto de Test Blindado", fontsize=13, fontweight="bold", pad=15)
plt.xlabel("Clase Predicha por el Modelo", fontsize=11, labelpad=10)
plt.ylabel("Clase Real (Ground Truth)", fontsize=11, labelpad=10)
plt.tight_layout()

path_save_cm = os.path.join(PATH_OUTPUT_GRAPHS, "confusion_matrix_test.png")
plt.savefig(path_save_cm, dpi=300, bbox_inches='tight')
plt.show()
print(f"💾 Matriz de confusión guardada en: {path_save_cm}")

# =====================================================================
# 📈 GRÁFICA 2: CURVA ROC (TEST SET)
# =====================================================================
fpr, tpr, _ = roc_curve(y_test, y_test_proba)
roc_auc = roc_auc_score(y_test, y_test_proba)

plt.figure(figsize=(7.5, 6))
plt.plot(fpr, tpr, color="darkorange", lw=2.5, label=f"Curva ROC (AUC = {roc_auc:.4f})")
plt.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--", label="Clasificador Aleatorio (AUC = 0.5000)")

plt.xlim([-0.01, 1.0])
plt.ylim([0.0, 1.02])
plt.grid(True, linestyle=":", alpha=0.6)

plt.title("Curva ROC - Rendimiento de Clasificación de Tos", fontsize=13, fontweight="bold", pad=15)
plt.xlabel("Tasa de Falsos Positivos (1 - Especificidad)", fontsize=11)
plt.ylabel("Tasa de Verdaderos Positivos (Sensibilidad / Recall)", fontsize=11)
plt.legend(loc="lower right", fontsize=11)
plt.tight_layout()

path_save_roc = os.path.join(PATH_OUTPUT_GRAPHS, "roc_curve_test.png")
plt.savefig(path_save_roc, dpi=300, bbox_inches='tight')
plt.show()
print(f"💾 Curva ROC guardada en: {path_save_roc}")
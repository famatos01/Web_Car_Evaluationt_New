import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score, 
    confusion_matrix, 
    classification_report, 
    adjusted_rand_score, 
    silhouette_score
)
from imblearn.over_sampling import SMOTE
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Cargar dataset
car_evaluation = fetch_ucirepo(id=19)
X = car_evaluation.data.features
y = car_evaluation.data.targets

# Verificar distribución de clases
print("\nDistribución original de clases:")
print(y.value_counts())

# 2. Preprocesamiento
# Codificación de características ordinales
encoder = OrdinalEncoder(
    categories=[
        ['low', 'med', 'high', 'vhigh'],  # buying
        ['low', 'med', 'high', 'vhigh'],  # maint
        ['2', '3', '4', '5more'],         # doors
        ['2', '4', 'more'],               # persons
        ['small', 'med', 'big'],          # lug_boot
        ['low', 'med', 'high']            # safety
    ]
)
X_encoded = encoder.fit_transform(X)

# Codificación de etiquetas
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y.values.ravel())

# 3. División estratificada de datos
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, 
    y_encoded, 
    test_size=0.3, 
    stratify=y_encoded,  # Mantener distribución de clases
    random_state=42
)

# 4. Balanceo de clases con SMOTE
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

print("\nDistribución después de SMOTE:")
print(pd.Series(y_train).value_counts())

# 5. Entrenamiento de modelos
models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        class_weight='balanced',
        random_state=42
    ),
    "Árbol de Decisión": DecisionTreeClassifier(
        max_depth=8,
        class_weight='balanced',
        random_state=42
    ),
    "Red Neuronal": MLPClassifier(
        hidden_layer_sizes=(100, 50),
        activation='relu',
        early_stopping=True,
        max_iter=1000,
        random_state=42
    )
}

for name, model in models.items():
    print(f"\n=== Entrenando {name} ===")
    
    # Entrenamiento
    model.fit(X_train, y_train)
    
    # Predicción
    y_pred = model.predict(X_test)
    
    # Evaluación
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nPrecisión: {accuracy:.2f}")
    
    # Reporte detallado por clase
    report = classification_report(
        y_test, 
        y_pred, 
        target_names=label_encoder.classes_, 
        output_dict=True
    )
    print(pd.DataFrame(report).transpose().round(2))
    
    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_
    )
    plt.title(f'Matriz de Confusión - {name}')
    plt.xlabel('Predicho')
    plt.ylabel('Real')
    plt.savefig(f'static/confusion_{name.lower().replace(" ", "_")}.png')
    plt.close()
    
    # Guardar modelo
    joblib.dump(model, f'model/{name.lower().replace(" ", "_")}_model.pkl')

# 6. Clustering de validación
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(X_encoded)

# Métricas de clustering
ari = adjusted_rand_score(y_encoded, clusters)
silhouette = silhouette_score(X_encoded, clusters)
print(f"\n=== Validación con Clustering ===")
print(f"Adjusted Rand Index: {ari:.2f}")
print(f"Silhouette Score: {silhouette:.2f}")

# 7. Guardar encoders
joblib.dump(encoder, 'model/encoder.pkl')
joblib.dump(label_encoder, 'model/label_encoder.pkl')
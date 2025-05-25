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
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# Paso 1: Cargar el dataset
car_evaluation = fetch_ucirepo(id=19)
X = car_evaluation.data.features  # Características (6 variables categóricas)
y = car_evaluation.data.targets   # Etiquetas (clases: unacc, acc, good, vgood)

# Paso 2: Codificar variables categóricas
# Definir órdenes para codificación ordinal
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
X_encoded = encoder.fit_transform(X)  # Transformar características a números

# Codificar etiquetas (target)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y.values.ravel())  # Convertir a array 1D

# Paso 3: Dividir datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, 
    y_encoded, 
    test_size=0.3, 
    random_state=42
)

# Paso 4: Entrenar y evaluar modelos supervisados
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Árbol de Decisión": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Red Neuronal": MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000, random_state=42)
}

for name, model in models.items():
    # Entrenar
    model.fit(X_train, y_train)
    
    # Predecir
    y_pred = model.predict(X_test)
    
    # Evaluar
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n--- {name} ---")
    print(f"Precisión: {accuracy:.2f}")
    print("Reporte de Clasificación:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    
    # Guardar modelo
    joblib.dump(model, f'model/{name.lower().replace(" ", "_")}_model.pkl')
    
    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=label_encoder.classes_, 
                yticklabels=label_encoder.classes_)
    plt.title(f'Matriz de Confusión - {name}')
    plt.xlabel('Predicho')
    plt.ylabel('Real')
    plt.savefig(f'static/confusion_{name.lower().replace(" ", "_")}.png')
    plt.close()

# Paso 5: Validación con Clustering (K-Means)
kmeans = KMeans(n_clusters=4, random_state=42)  # 4 clases en el target
clusters = kmeans.fit_predict(X_encoded)

# Métricas de calidad de clustering
ari = adjusted_rand_score(y_encoded, clusters)  # Comparación con etiquetas reales
silhouette = silhouette_score(X_encoded, clusters)  # Cohesión de clusters
print(f"\n--- Clustering (K-Means) ---")
print(f"Adjusted Rand Index: {ari:.2f} (1 = perfecto, 0 = aleatorio)")
print(f"Silhouette Score: {silhouette:.2f} (-1 a 1, mayor es mejor)")

# Guardar encoders para uso en la web
joblib.dump(encoder, 'model/encoder.pkl')
joblib.dump(label_encoder, 'model/label_encoder.pkl')
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
import numpy as np

def clean_dataset(df, target_column=None):
    """
    Realiza limpieza completa de un dataset:
    1. Maneja valores faltantes
    2. Elimina duplicados
    3. Valida y corrige categorías
    4. Normaliza nombres de columnas
    """
    cleaned_df = df.copy()
    
    # 1. Normalizar nombres de columnas
    cleaned_df.columns = [col.strip().lower().replace(' ', '_') for col in cleaned_df.columns]
    
    # 2. Manejo de valores faltantes
    print("\nValores faltantes antes de la limpieza:")
    print(cleaned_df.isnull().sum())
    
    # Para columnas categóricas: rellenar con la moda
    categorical_cols = cleaned_df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        mode_val = cleaned_df[col].mode()[0]
        cleaned_df[col] = cleaned_df[col].fillna(mode_val)
        print(f"Columna '{col}': {cleaned_df[col].isnull().sum()} valores faltantes después de imputar con moda '{mode_val}'")
    
    # 3. Eliminar duplicados
    initial_rows = len(cleaned_df)
    cleaned_df = cleaned_df.drop_duplicates()
    final_rows = len(cleaned_df)
    duplicates_removed = initial_rows - final_rows
    print(f"\nSe eliminaron {duplicates_removed} filas duplicadas")
    
    # 4. Validación de categorías (solo para variables categóricas)
    expected_categories = {
        'buying': ['low', 'med', 'high', 'vhigh'],
        'maint': ['low', 'med', 'high', 'vhigh'],
        'doors': ['2', '3', '4', '5more'],
        'persons': ['2', '4', 'more'],
        'lug_boot': ['small', 'med', 'big'],
        'safety': ['low', 'med', 'high']
    }
    
    print("\nValidación de categorías:")
    for col, expected in expected_categories.items():
        if col in cleaned_df.columns:
            unique_vals = cleaned_df[col].unique()
            invalid_vals = [val for val in unique_vals if val not in expected]
            
            if invalid_vals:
                # Corregir valores inválidos con la moda
                mode_val = cleaned_df[col].mode()[0]
                cleaned_df.loc[cleaned_df[col].isin(invalid_vals), col] = mode_val
                print(f"Columna '{col}': Se corrigieron {len(invalid_vals)} valores inválidos a '{mode_val}'")
            else:
                print(f"Columna '{col}': Todas las categorías son válidas")
    
    # 5. Separar características y objetivo si está especificado
    if target_column:
        X = cleaned_df.drop(columns=[target_column])
        y = cleaned_df[[target_column]]
        return X, y
    else:
        return cleaned_df

# 1. Cargar y limpiar dataset
car_evaluation = fetch_ucirepo(id=19)
X = car_evaluation.data.features
y = car_evaluation.data.targets

# Convertir a DataFrame para la limpieza
df = pd.concat([X, y], axis=1)

# CORRECCIÓN: Recibir ambos valores de retorno (X, y)
X_cleaned, y_cleaned = clean_dataset(df, target_column='class')

# Verificar distribución de clases (usar y_cleaned)
print("\nDistribución original de clases:")
print(y_cleaned['class'].value_counts())

# 2. Preprocesamiento
# (X_cleaned y y_cleaned ya están disponibles)

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
X_encoded = encoder.fit_transform(X_cleaned)

# Codificación de etiquetas
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_cleaned.values.ravel())

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

# 5. Entrenamiento de modelos y comparación
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

# Diccionario para almacenar métricas de cada modelo
model_metrics = {}

for name, model in models.items():
    print(f"\n=== Entrenando {name} ===")
    
    # Entrenamiento
    model.fit(X_train, y_train)
    
    # Predicción
    y_pred = model.predict(X_test)
    
    # Evaluación
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test, 
        y_pred, 
        target_names=label_encoder.classes_, 
        output_dict=True
    )
    
    # Almacenar métricas para comparación
    model_metrics[name] = {
        'accuracy': accuracy,
        'f1_macro': report['macro avg']['f1-score'],
        'f1_weighted': report['weighted avg']['f1-score']
    }
    
    # Imprimir resultados individuales
    print(f"\nPrecisión: {accuracy:.4f}")
    print(f"F1-score (Macro): {report['macro avg']['f1-score']:.4f}")
    print(f"F1-score (Weighted): {report['weighted avg']['f1-score']:.4f}")
    print(pd.DataFrame(report).transpose().round(4))
    
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

# 6. Comparación de modelos y selección del mejor
print("\n" + "="*60)
print("COMPARACIÓN FINAL DE MODELOS")
print("="*60)

# Crear DataFrame con las métricas
results_df = pd.DataFrame.from_dict(model_metrics, orient='index')
results_df = results_df.sort_values(by='accuracy', ascending=False)
print("\nResumen de métricas por modelo:")
print(results_df)

# Identificar el mejor modelo en base a accuracy
best_model_name = results_df.index[0]
best_accuracy = results_df.iloc[0]['accuracy']
best_f1_macro = results_df.iloc[0]['f1_macro']

print("\n" + "="*60)
print(f"MEJOR MODELO: {best_model_name}")
print(f"Precisión: {best_accuracy:.4f}")
print(f"F1-score (Macro): {best_f1_macro:.4f}")
print("="*60)

# Gráfico comparativo de precisión
plt.figure(figsize=(10, 6))
results_df['accuracy'].plot(kind='bar', color='skyblue')
plt.title('Comparación de Precisión entre Modelos')
plt.ylabel('Precisión')
plt.ylim(0.8, 1.0)
plt.xticks(rotation=15)
for i, v in enumerate(results_df['accuracy']):
    plt.text(i, v + 0.005, f'{v:.4f}', ha='center')
plt.savefig('static/model_comparison.png')
plt.close()

# 7. Clustering de validación
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(X_encoded)

# Métricas de clustering
ari = adjusted_rand_score(y_encoded, clusters)
silhouette = silhouette_score(X_encoded, clusters)
print(f"\n=== Validación con Clustering ===")
print(f"Adjusted Rand Index: {ari:.2f}")
print(f"Silhouette Score: {silhouette:.2f}")

# 8. Guardar encoders
joblib.dump(encoder, 'model/encoder.pkl')
joblib.dump(label_encoder, 'model/label_encoder.pkl')
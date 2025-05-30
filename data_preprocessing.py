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
import os

def clean_dataset(df, target_column=None):
    """
    Limpieza adaptativa basada en inspección real de datos:
    1. Análisis inicial de estructura
    2. Manejo flexible de valores faltantes
    3. Validación de categorías según datos observados
    """
    # Paso 1: Inspección inicial
    print("\n" + "="*60)
    print("INSPECCIÓN INICIAL DEL DATASET")
    print("="*60)
    print("Columnas originales:", df.columns.tolist())
    print("\nTipos de datos:\n", df.dtypes)
    print("\nValores faltantes iniciales:\n", df.isnull().sum())
    print("\nMuestra inicial de datos (2 registros):")
    print(df.head(2))
    
    cleaned_df = df.copy()
    
    # Normalizar nombres de columnas
    cleaned_df.columns = [col.strip().lower().replace(' ', '_') for col in cleaned_df.columns]
    print("\n" + "-"*60)
    print("Columnas normalizadas:", cleaned_df.columns.tolist())
    
    # Paso 2: Manejo adaptativo de valores faltantes
    print("\n" + "="*60)
    print("MANEJO DE VALORES FALTANTES")
    print("="*60)
    for col in cleaned_df.columns:
        if cleaned_df[col].isnull().sum() > 0:
            if cleaned_df[col].dtype == 'object':
                mode_val = cleaned_df[col].mode()[0]
                cleaned_df[col] = cleaned_df[col].fillna(mode_val)
                print(f"[CATEGÓRICA] '{col}': Imputados {cleaned_df[col].isnull().sum()} faltantes con moda '{mode_val}'")
            else:
                median_val = cleaned_df[col].median()
                cleaned_df[col] = cleaned_df[col].fillna(median_val)
                print(f"[NUMÉRICA] '{col}': Imputados {cleaned_df[col].isnull().sum()} faltantes con mediana {median_val:.2f}")
        else:
            print(f"[OK] '{col}': Sin valores faltantes")
    
    # Paso 3: Eliminación de duplicados
    initial_rows = len(cleaned_df)
    cleaned_df = cleaned_df.drop_duplicates()
    final_rows = len(cleaned_df)
    duplicates_removed = initial_rows - final_rows
    print("\n" + "="*60)
    print("ELIMINACIÓN DE DUPLICADOS")
    print("="*60)
    print(f"Registros iniciales: {initial_rows}")
    print(f"Registros finales: {final_rows}")
    print(f"Duplicados eliminados: {duplicates_removed}")
    
    # Paso 4: Validación adaptativa de categorías
    print("\n" + "="*60)
    print("VALIDACIÓN DE CATEGORÍAS")
    print("="*60)
    categorical_cols = cleaned_df.select_dtypes(include=['object', 'category']).columns
    
    for col in categorical_cols:
        # Limpieza básica de categorías
        cleaned_df[col] = cleaned_df[col].astype(str).str.strip().str.lower()
        
        # Determinar categorías únicas
        unique_vals = cleaned_df[col].unique().tolist()
        print(f"\nColumna '{col}':")
        print(f"- Valores únicos: {unique_vals}")
        
        # Identificar valores inusuales
        value_counts = cleaned_df[col].value_counts(normalize=True) * 100
        unusual_threshold = 1.0  # Valores con <1% de frecuencia
        unusual_values = value_counts[value_counts < unusual_threshold].index.tolist()
        
        if unusual_values:
            # Corregir con la categoría más común
            mode_val = cleaned_df[col].mode()[0]
            print(f"  - Valores inusuales detectados: {unusual_values}")
            print(f"  - Reemplazando {len(unusual_values)} valores con '{mode_val}'")
            cleaned_df[col] = cleaned_df[col].replace(unusual_values, mode_val)
        else:
            print("  - Sin valores inusuales detectados")
    
    # Paso 5: Análisis de distribución de clases (si existe target)
    if target_column and target_column in cleaned_df.columns:
        print("\n" + "="*60)
        print("DISTRIBUCIÓN DE CLASES")
        print("="*60)
        class_dist = cleaned_df[target_column].value_counts(normalize=True) * 100
        print(class_dist)
        
        # Identificar clases minoritarias
        minority_classes = class_dist[class_dist < 5].index.tolist()
        if minority_classes:
            print(f"Alert: Clases minoritarias detectadas: {minority_classes}")
    
    # Separar características y objetivo si está especificado
    if target_column and target_column in cleaned_df.columns:
        X = cleaned_df.drop(columns=[target_column])
        y = cleaned_df[[target_column]]
        return X, y
    else:
        return cleaned_df

# Crear directorios necesarios
os.makedirs('model', exist_ok=True)
os.makedirs('static', exist_ok=True)

# ========= EJECUCIÓN PRINCIPAL =========
if __name__ == "__main__":
    # 1. Cargar dataset SIN modificar
    print("\n" + "="*60)
    print("CARGANDO DATASET ORIGINAL")
    print("="*60)
    car_evaluation = fetch_ucirepo(id=19)
    
    # Crear DataFrame sin modificaciones
    raw_df = pd.concat([car_evaluation.data.features, car_evaluation.data.targets], axis=1)
    
    # 2. Limpieza ADAPTATIVA basada en datos reales
    X_cleaned, y_cleaned = clean_dataset(raw_df, target_column='class')
    
    # Verificar distribución de clases después de limpieza
    print("\n" + "="*60)
    print("DISTRIBUCIÓN FINAL DE CLASES")
    print("="*60)
    print(y_cleaned['class'].value_counts())
    
    # 3. Preprocesamiento con codificación basada en datos reales
    # Identificar categorías únicas para cada columna
    print("\n" + "="*60)
    print("PREPARANDO CODIFICACIÓN")
    print("="*60)
    
    # Obtener categorías únicas ordenadas para cada característica
    buying_cats = sorted(X_cleaned['buying'].unique())
    maint_cats = sorted(X_cleaned['maint'].unique())
    doors_cats = sorted(X_cleaned['doors'].unique())
    persons_cats = sorted(X_cleaned['persons'].unique())
    lug_boot_cats = sorted(X_cleaned['lug_boot'].unique())
    safety_cats = sorted(X_cleaned['safety'].unique())
    
    print("Categorías para 'buying':", buying_cats)
    print("Categorías para 'maint':", maint_cats)
    print("Categorías para 'doors':", doors_cats)
    print("Categorías para 'persons':", persons_cats)
    print("Categorías para 'lug_boot':", lug_boot_cats)
    print("Categorías para 'safety':", safety_cats)
    
    encoder = OrdinalEncoder(
        categories=[
            buying_cats, 
            maint_cats, 
            doors_cats, 
            persons_cats, 
            lug_boot_cats, 
            safety_cats
        ]
    )
    X_encoded = encoder.fit_transform(X_cleaned)
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_cleaned.values.ravel())
    
    # 4. División estratificada de datos
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, 
        y_encoded, 
        test_size=0.3, 
        stratify=y_encoded,
        random_state=42
    )
    
    # 5. Balanceo de clases con SMOTE (solo en datos de entrenamiento)
    print("\n" + "="*60)
    print("BALANCEO DE CLASES CON SMOTE")
    print("="*60)
    print("Distribución antes de SMOTE:")
    print(pd.Series(y_train).value_counts())
    
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    
    print("\nDistribución después de SMOTE:")
    print(pd.Series(y_train).value_counts())
    
    # 6. Entrenamiento de modelos y comparación
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
    
    # Diccionario para almacenar métricas
    model_metrics = {}
    
    print("\n" + "="*60)
    print("ENTRENAMIENTO DE MODELOS")
    print("="*60)
    
    for name, model in models.items():
        print(f"\n=== Entrenando {name} ===")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(
            y_test, 
            y_pred, 
            target_names=label_encoder.classes_, 
            output_dict=True
        )
        
        model_metrics[name] = {
            'accuracy': accuracy,
            'f1_macro': report['macro avg']['f1-score'],
            'f1_weighted': report['weighted avg']['f1-score']
        }
        
        print(f"\nPrecisión: {accuracy:.4f}")
        print(f"F1-score (Macro): {report['macro avg']['f1-score']:.4f}")
        print(f"F1-score (Weighted): {report['weighted avg']['f1-score']:.4f}")
        
        # Matriz de confusión
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 8))
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
        plt.tight_layout()
        plt.savefig(f'static/confusion_{name.lower().replace(" ", "_")}.png')
        plt.close()
        print(f"Matriz de confusión guardada en static/confusion_{name.lower().replace(' ', '_')}.png")
        
        # Guardar modelo
        model_path = f'model/{name.lower().replace(" ", "_")}_model.pkl'
        joblib.dump(model, model_path)
        print(f"Modelo guardado en {model_path}")
    
    # 7. Comparación de modelos
    print("\n" + "="*60)
    print("COMPARACIÓN FINAL DE MODELOS")
    print("="*60)
    
    results_df = pd.DataFrame.from_dict(model_metrics, orient='index')
    results_df = results_df.sort_values(by='accuracy', ascending=False)
    print("\nResumen de métricas por modelo:")
    print(results_df)
    
    best_model_name = results_df.index[0]
    best_model_metrics = results_df.loc[best_model_name]
    
    print("\n" + "="*60)
    print(f"MEJOR MODELO: {best_model_name}")
    print(f"Precisión: {best_model_metrics['accuracy']:.4f}")
    print(f"F1-score (Macro): {best_model_metrics['f1_macro']:.4f}")
    print(f"F1-score (Weighted): {best_model_metrics['f1_weighted']:.4f}")
    print("="*60)
    
    # Gráfico comparativo
    plt.figure(figsize=(12, 7))
    results_df['accuracy'].plot(kind='bar', color='skyblue')
    plt.title('Comparación de Precisión entre Modelos', fontsize=14)
    plt.ylabel('Precisión', fontsize=12)
    plt.ylim(0.8, 1.0)
    plt.xticks(rotation=15, fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    for i, v in enumerate(results_df['accuracy']):
        plt.text(i, v + 0.005, f'{v:.4f}', ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('static/model_comparison.png')
    plt.close()
    print("Gráfico comparativo guardado en static/model_comparison.png")
    
    # 8. Clustering de validación
    print("\n" + "="*60)
    print("VALIDACIÓN CON CLUSTERING")
    print("="*60)
    
    kmeans = KMeans(n_clusters=4, random_state=42)
    clusters = kmeans.fit_predict(X_encoded)
    
    # Métricas de clustering
    ari = adjusted_rand_score(y_encoded, clusters)
    silhouette = silhouette_score(X_encoded, clusters)
    
    print(f"Adjusted Rand Index: {ari:.4f}")
    print(f"Silhouette Score: {silhouette:.4f}")
    
    # Gráfico de clustering
    plt.figure(figsize=(10, 7))
    plt.scatter(
        X_encoded[:, 0], 
        X_encoded[:, 1], 
        c=clusters, 
        cmap='viridis',
        alpha=0.6
    )
    plt.title('Visualización de Clusters', fontsize=14)
    plt.xlabel('Feature 1', fontsize=12)
    plt.ylabel('Feature 2', fontsize=12)
    plt.colorbar(label='Cluster')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig('static/clusters.png')
    plt.close()
    print("Visualización de clusters guardada en static/clusters.png")
    
    # 9. Guardar encoders
    joblib.dump(encoder, 'model/encoder.pkl')
    joblib.dump(label_encoder, 'model/label_encoder.pkl')
    print("\nEncoders guardados en model/encoder.pkl y model/label_encoder.pkl")
    
    print("\n" + "="*60)
    print("PROCESO COMPLETADO EXITOSAMENTE")
    print("="*60)
from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Cargar modelos y encoders
encoder = joblib.load('model/encoder.pkl')
label_encoder = joblib.load('model/label_encoder.pkl')
models = {
    "Random Forest": joblib.load('model/random_forest_model.pkl'),
    "Árbol de Decisión": joblib.load('model/árbol_de_decisión_model.pkl'),
    "Red Neuronal": joblib.load('model/red_neuronal_model.pkl')
}

@app.route('/')
def home():
    """Renderizar la página principal con el formulario."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Procesar el formulario y mostrar la predicción."""
    # Obtener datos del formulario
    features = [
        request.form['buying'],
        request.form['maint'],
        request.form['doors'],
        request.form['persons'],
        request.form['lug_boot'],
        request.form['safety']
    ]
    
    # Paso 1: Codificar características
    encoded_features = encoder.transform([features])  # Transformar a array 2D
    
    # Paso 2: Obtener modelo seleccionado
    selected_model = request.form['model']
    model = models[selected_model]
    
    # Paso 3: Predecir
    prediction = model.predict(encoded_features)
    predicted_class = label_encoder.inverse_transform(prediction)[0]  # Convertir número a etiqueta
    
    # Obtener imagen de la matriz de confusión
    confusion_image = f'confusion_{selected_model.lower().replace(" ", "_")}.png'
    
    return render_template('result.html', 
                         prediction=predicted_class,
                         model_name=selected_model,
                         confusion_image=confusion_image)

if __name__ == '__main__':
    app.run(debug=True)  # Ejecutar en modo desarrollo
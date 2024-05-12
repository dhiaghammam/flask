from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Charger le modèle de régression logistique
logistic_model = joblib.load('logistic_model.sav')

# Routes
@app.route('/')
def index():
    return render_template('index2.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Récupérer les données du formulaire HTML
    data = request.form.to_dict()
    
    # Créer un DataFrame Pandas avec les données
    df = pd.DataFrame(data, index=[0])
    
    # Effectuer la prédiction avec le modèle de régression logistique
    logistic_prediction = logistic_model.predict(df)
    
    # Préparer la réponse pour l'envoi
    response = {
        'logistic_prediction': logistic_prediction[0]
    }
    
    # Envoyer la prédiction au format JSON
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask, request, render_template, jsonify
import pandas as pd
import psycopg2 as ps
import numpy as np
from sklearn.cluster import KMeans
import joblib
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Charger le modèle KMeans depuis le fichier .sav
kmeans_model = joblib.load('kmeans_model.sav')

# Connecter à la base de données
conn2 = ps.connect(dbname="SPN_DW", 
                   user="postgres", 
                   password="dhia", 
                   host="localhost", 
                   port="5432")
cur = conn2.cursor()

# Requête SQL pour obtenir les données
query = """
SELECT 
    co2."Type", 
    co2."CO2_Emission", 
    co2."Plate_Number"
FROM 
    public."CO2" AS co2
JOIN 
    public."FactTransactionsServices" AS fact ON co2."ID_CO2Emission"= fact."ID_CO2Emission"
JOIN
    public."CarJdid" AS cars ON cars."idCar" = fact."idCar";"""

cur.execute(query)
rows = cur.fetchall()

columns = [desc[0] for desc in cur.description]
df = pd.DataFrame(rows, columns=columns)

# Supprimer les lignes où Plate_Number est 'None'
df_filtered = df.dropna(subset=['Plate_Number'])

# Sélectionner les caractéristiques pertinentes (CO2_Emission)
X = df_filtered['CO2_Emission'].values.reshape(-1, 1)

# Ajouter les étiquettes de cluster au DataFrame
df_filtered['Cluster'] = kmeans_model.predict(X)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommendations', methods=['GET'])
def recommendations():
    emission_limit = float(request.args.get('emission_limit'))
    if emission_limit is None:
        return jsonify({'error': 'Emission limit is required'}), 400
    
    # Filtrer les données en fonction de la limite d'émission
    filtered_data = df_filtered[df_filtered['CO2_Emission'] <= emission_limit]
    
    # Grouper par numéro de plaque et recommander pour chaque groupe
    recommendations = filtered_data.groupby("Plate_Number").agg({"Type": "first", "CO2_Emission": "min"}).reset_index()
    
    return jsonify(recommendations.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True)
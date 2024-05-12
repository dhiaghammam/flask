from flask import Flask, request, render_template, jsonify
import pandas as pd
import psycopg2 as ps
import joblib
import numpy as np
from flask_cors import CORS  # Import CORS module

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "http://localhost:4200"}})  # Allow requests from Angular app
# Establish database connection
conn2 = ps.connect(dbname="SPN_DW", user="postgres", password="dhia", host="localhost", port="5432")
cur = conn2.cursor()

# Fetch data from the database
query = """
SELECT 
    client."Client_name", 
    client."Type_of_request", 
    client."Arrival_date", 
    client."Departure_date", 
    cars."Type",
    fact."Total_amount" AS "Total_amount"
FROM 
    public."DimClients" AS client
JOIN 
    public."FactTransactionsServices" AS fact ON client."Id_Client" = fact."idClient" 
JOIN 
    public."CarJdid" AS cars ON fact."idCar" = cars."idCar";
"""
cur.execute(query)
rows = cur.fetchall()

columns = [desc[0] for desc in cur.description]
df = pd.DataFrame(rows, columns=columns)

# Data preprocessing
# Remove currency units
df['Total_amount'] = df['Total_amount'].str.replace(' CHF', '')
# Replace incorrect decimal separators
df['Total_amount'] = df['Total_amount'].str.replace(',', '.')
df['Total_amount'] = df['Total_amount'].replace('25O', '250')
# Convert Total_amount to numeric
df['Total_amount'] = pd.to_numeric(df['Total_amount'], errors='coerce')

# Impute missing values with median
median_total_amount = df['Total_amount'].median()
df['Total_amount'] = df['Total_amount'].fillna(median_total_amount)

# Convert Type_of_request to lowercase
df['Type_of_request'] = df['Type_of_request'].str.lower()

# Drop rows with missing Type_of_request values
df = df.dropna(subset=['Type_of_request'])

# Pivot table for counting requests per client
pivot_table = df.pivot_table(index='Client_name', columns='Type_of_request', aggfunc='size', fill_value=0)
pivot_table['Total_requests'] = pivot_table.sum(axis=1)

# Calculate total amount per client
total_amount_per_client = df.groupby('Client_name')['Total_amount'].sum().reset_index()

# Merge DataFrames
merged_df = pd.merge(pivot_table[['Total_requests']], total_amount_per_client, on='Client_name')

# Determine threshold for loyal customers
def determine_threshold(df):
    total_requests_quartiles = np.percentile(df['Total_requests'], [25, 50, 75])
    total_amount_quartiles = np.percentile(df['Total_amount'], [25, 50, 75])
    request_threshold = total_requests_quartiles[2]
    amount_threshold = total_amount_quartiles[2]
    return request_threshold, amount_threshold

request_threshold, amount_threshold = determine_threshold(merged_df)
merged_df['Loyal_customer'] = ((merged_df['Total_requests'] >= request_threshold) & (merged_df['Total_amount'] >= amount_threshold)).astype(int)

# Load the model
svm_model = joblib.load(r'C:\Users\dhia\Desktop\FlashApp\Clientsfid.sav')

@app.route('/')
def index():
    return render_template('index1.html')

# Flask route
@app.route('/predict', methods=['POST'])
def predict():
    # Receive data from form
    Total_requests = int(request.form['Total_requests'])
    Total_amount = float(request.form['Total_amount'])

    # Make prediction
    prediction = svm_model.predict([[Total_requests, Total_amount]])

    # Convert prediction to text
    prediction_text = "Loyal Customer" if prediction[0] == 1 else "Not Loyal Customer"

    # Return prediction as JSON response
    return jsonify({'prediction_text': prediction_text})


if __name__ == '__main__':
    app.run(debug=True)

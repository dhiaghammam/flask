from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import psycopg2 as ps
import random
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the SARIMA model
model = joblib.load(r'C:\Users\dhia\Desktop\FlashApp\temp.sav')

# Load data into DataFrame
def load_data():
    conn = ps.connect(dbname="SPNPI", 
                      user="postgres", 
                      password="dhia", 
                      host="localhost", 
                      port="5432")
    query = """
        SELECT 
            "Date_of_request", 
            "Total_Charges" 
        FROM 
            public."chrges";
    """
    df = pd.read_sql(query, conn)
    conn.close()
    # Convert 'Date_of_request' to datetime
    df['Date_of_request'] = pd.to_datetime(df['Date_of_request'], dayfirst=True)
    # Set 'Date_of_request' as index
    df.set_index('Date_of_request', inplace=True)
    return df

@app.route('/')
def index():
    return render_template('temp.html')

@app.route('/get_charges', methods=['POST'])
def get_charges():
    data = request.get_json()
    date = pd.to_datetime(data['date'])
    df = load_data()
    try:
        charges = df.loc[date, 'Total_Charges']
        charges = charges.item() if isinstance(charges, pd.Series) else charges
    except (KeyError, ValueError):
        # If the specified date is not found or there's an error converting charges to a scalar, return a random number between 10 and 50
        charges = random.randint(10, 50)
    return jsonify({'date': date.strftime('%Y-%m-%d'), 'charges': charges})

if __name__ == '__main__':
    app.run(debug=True)

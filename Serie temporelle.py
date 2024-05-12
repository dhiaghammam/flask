#!/usr/bin/env python
# coding: utf-8

# In[1]:


#SERIE TEMPORELLE


# In[17]:


import pandas as pd
import psycopg2 as ps
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# Establish a connection to the database
conn = ps.connect(dbname="SPNPI", 
                  user="postgres", 
                  password="dhia", 
                  host="localhost", 
                  port="5432")

# SQL query to fetch data
query = """
    SELECT 
        "Date_of_request", 
        "Total_Charges" 
    FROM 
        public."chrges";
"""

# Load data into DataFrame
df = pd.read_sql(query, conn)

# Close the database connection
conn.close()

# Convert 'Date_of_request' to datetime
df['Date_of_request'] = pd.to_datetime(df['Date_of_request'], dayfirst=True)

# Check data types and values
print(df.info())
print(df)

# Aggregate charges by date
df = df.groupby('Date_of_request')['Total_Charges'].sum().reset_index()

# Set 'Date_of_request' as index
df.set_index('Date_of_request', inplace=True)

# Fit ARIMA model
model = ARIMA(df, order=(5, 1, 1))  # Adjust the order if needed
model_fit = model.fit()

# Display model summary
print(model_fit.summary())

# Make predictions for the next 365 days
forecast = model_fit.forecast(steps=365)

# Plot the forecast
forecast_dates = pd.date_range(start=df.index[-1], periods=366, freq='D')[1:]  # Generate future dates
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Total_Charges'], label='Historical Data')
plt.plot(forecast_dates, forecast, label='Forecast', color='red')
plt.title('Forecast of Total Charges')
plt.xlabel('Date')
plt.ylabel('Total Charges')
plt.legend()
plt.grid(True)
plt.show()


# In[18]:


import pandas as pd
import psycopg2 as ps
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# Establish a connection to the database
conn = ps.connect(dbname="SPNPI", 
                  user="postgres", 
                  password="dhia", 
                  host="localhost", 
                  port="5432")

# SQL query to fetch data
query = """
    SELECT 
        "Date_of_request", 
        "Total_Charges" 
    FROM 
        public."chrges";
"""

# Load data into DataFrame
df = pd.read_sql(query, conn)

# Close the database connection
conn.close()

# Convert 'Date_of_request' to datetime
df['Date_of_request'] = pd.to_datetime(df['Date_of_request'], dayfirst=True)

# Check data types and values
print(df.info())
print(df)

# Aggregate charges by date
df = df.groupby('Date_of_request')['Total_Charges'].sum().reset_index()

# Set 'Date_of_request' as index
df.set_index('Date_of_request', inplace=True)

# Fit ARIMA model
model = ARIMA(df, order=(2, 1, 2))  # Trying different ARIMA parameters
model_fit = model.fit()

# Display model summary
print(model_fit.summary())

# Make predictions for the next 365 days
forecast = model_fit.forecast(steps=365)

# Plot the forecast
forecast_dates = pd.date_range(start=df.index[-1], periods=366, freq='D')[1:]  # Generate future dates
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Total_Charges'], label='Historical Data')
plt.plot(forecast_dates, forecast, label='Forecast', color='red')
plt.title('Forecast of Total Charges')
plt.xlabel('Date')
plt.ylabel('Total Charges')
plt.legend()
plt.grid(True)
plt.show()


# In[24]:


import pandas as pd
import psycopg2 as ps
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
import joblib
# Establish a connection to the database
conn = ps.connect(dbname="SPNPI", 
                  user="postgres", 
                  password="dhia", 
                  host="localhost", 
                  port="5432")

# SQL query to fetch data
query = """
    SELECT 
        "Date_of_request", 
        "Total_Charges" 
    FROM 
        public."chrges";
"""

# Load data into DataFrame
df = pd.read_sql(query, conn)

# Close the database connection
conn.close()

# Convert 'Date_of_request' to datetime
df['Date_of_request'] = pd.to_datetime(df['Date_of_request'], dayfirst=True)

# Check data types and values
print(df.info())
print(df)

# Aggregate charges by date
df = df.groupby('Date_of_request')['Total_Charges'].sum().reset_index()

# Set 'Date_of_request' as index
df.set_index('Date_of_request', inplace=True)

# Fit SARIMA model
model = SARIMAX(df, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
model_fit = model.fit(disp=False)

# Make predictions for the next 365 days
forecast = model_fit.forecast(steps=365)

# Get the last date in the historical data
last_date = df.index[-1]

# Generate future dates for the forecast
forecast_dates = pd.date_range(start=last_date, periods=365, freq='D')

# Plot the forecast
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Total_Charges'], label='Historical Data')
plt.plot(forecast_dates, forecast, label='Forecast', color='red')
plt.title('Forecast of Total Charges')
plt.xlabel('Date')
plt.ylabel('Total Charges')
plt.legend()
plt.grid(True)
plt.show()
joblib.dump(model, r'C:\Users\dhia\Desktop\FlashApp\temp.sav')


# In[2]:


import pandas as pd  # - with datafame
import pandas.io.sql as sqlio  # - with sql query
import psycopg2 as ps # - with postgresql database (to connect!)
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf




# In[3]:


conn2 = ps.connect(dbname="SPNPI", 
                   user = "postgres", 
                   password = "dhia", 
                   host = "localhost", 
                   port = "5432") 
# lets connect


# In[4]:


cur = conn2.cursor()


# In[5]:


query = """
SELECT 
    charges."Date_of_request", 
    charges."Total_Charges" 
    
FROM 
    public."chrges" AS charges;

"""

# Rollback the current transaction
conn2.rollback()

# Re-execute your query
cur.execute(query)
rows = cur.fetchall()


# In[6]:


columns = [desc[0] for desc in cur.description]
df = pd.DataFrame(rows, columns=columns)


# In[6]:


import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# Convertir 'Date_of_request' en datetime
df['Date_of_request'] = pd.to_datetime(df['Date_of_request'], dayfirst=True)

# Vérifions les types de données et les valeurs
print(df.info())
print(df)

# Agrégation des charges par date
df = df.groupby('Date_of_request')['Total_Charges'].sum().reset_index()

# Sélection des données pour la modélisation
df.set_index('Date_of_request', inplace=True)

# Ajustement du modèle ARIMA
model = ARIMA(df, order=(1, 0, 1))
model_fit = model.fit()

# Affichage du résumé du modèle
print(model_fit.summary())

# Prédictions pour les 5 prochains jours
preds = model_fit.get_forecast(steps=5)

# Affichage des prédictions
forecast_dates = pd.date_range(start=df.index[-1], periods=6, freq='D')[1:]  # Generate future dates
plt.figure(figsize=(10, 5))
plt.plot(df.index, df['Total_Charges'], label='Historical Data')
plt.plot(forecast_dates, preds.predicted_mean, label='Forecast', color='red')
plt.title('Forecast of Total Charges')
plt.xlabel('Date')
plt.ylabel('Total Charges')
plt.legend()
plt.grid(True)
plt.show()


# In[21]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import STL

# Load data and preprocess if necessary
# Assuming df contains the time series data

# Convert 'Date_of_request' to datetime
df['Date_of_request'] = pd.to_datetime(df['Date_of_request'], dayfirst=True)

# Aggregate charges by date
df = df.groupby('Date_of_request')['Total_Charges'].sum().reset_index()

# Set 'Date_of_request' as index
df.set_index('Date_of_request', inplace=True)

# Exponential Smoothing
# Choose appropriate smoothing parameters (alpha, beta, gamma)
alpha = 0.3
beta = 0.1
gamma = 0.05

# Fit the model
model_es = ExponentialSmoothing(df, trend='add', seasonal='add', seasonal_periods=7)
model_es_fit = model_es.fit(smoothing_level=alpha, smoothing_slope=beta, smoothing_seasonal=gamma)

# Predict 90 days ahead
forecast_es = model_es_fit.forecast(steps=90)

# Plot Exponential Smoothing Forecast
plt.figure(figsize=(10, 5))
plt.plot(df.index, df['Total_Charges'], label='Historical Data')
plt.plot(forecast_es.index, forecast_es, label='Exponential Smoothing Forecast', color='red')
plt.title('Exponential Smoothing Forecast of Total Charges')
plt.xlabel('Date')
plt.ylabel('Total Charges')
plt.legend()
plt.grid(True)
plt.show()

# Seasonal Decomposition (STL)
# Extract the univariate time series
endog = df['Total_Charges']

# Specify seasonal period manually (for example, weekly seasonality)
seasonal_period = 7

stl = STL(endog, seasonal=seasonal_period)
result = stl.fit()

# Forecast 90 days ahead
forecast_stl = result.forecast(steps=90)

# Plot Seasonal Decomposition Forecast
plt.figure(figsize=(10, 5))
plt.plot(df.index, df['Total_Charges'], label='Historical Data')
plt.plot(forecast_stl.index, forecast_stl, label='STL Forecast', color='green')
plt.title('STL Forecast of Total Charges')
plt.xlabel('Date')
plt.ylabel('Total Charges')
plt.legend()
plt.grid(True)
plt.show()


# In[25]:


# Prédictions pour l'année suivante (365 jours)
next_year_preds = model_fit.get_forecast(steps=365)

# Affichage des prédictions pour l'année suivante
next_year_dates = pd.date_range(start=df.index[-1], periods=366, freq='D')[1:]  # Générer des dates pour l'année suivante
plt.figure(figsize=(10, 5))
plt.plot(df.index, df['Total_Charges'], label='Historical Data')
plt.plot(next_year_dates, next_year_preds.predicted_mean, label='Next Year Forecast', color='blue')  # Ajout des prédictions pour l'année suivante
plt.title('Forecast of Total Charges for Next Year')
plt.xlabel('Date')
plt.ylabel('Total Charges')
plt.legend()
plt.grid(True)
plt.show()


# In[17]:


import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Supposons que 'df' est votre DataFrame après l'agrégation des données.
# Convertir 'Date_of_request' en datetime et 'Total_Charges' en float
# Convertir 'Date_of_request' en datetime
df['Date_of_request'] = pd.to_datetime(df['Date_of_request'], dayfirst=True)

# Vérifions les types de données et les valeurs
print(df.info())
print(df)

# Agrégation des charges par date et conversion en DataFrame
df = df.groupby('Date_of_request')['Total_Charges'].sum().reset_index()

# Ajustement du modèle ARIMA (pas besoin de convertir l'index en période)
model = ARIMA(df['Total_Charges'], order=(1, 0, 1))



model_fit = model.fit()

# Affichage du résumé du modèle
print(model_fit.summary())

# Prédictions pour l'année 2024
forecast_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=365, freq='D')

# Effectuer les prédictions
preds = model_fit.get_forecast(steps=365)
forecast_mean = preds.predicted_mean

# Affichage des prévisions pour l'année 2024
plt.figure(figsize=(10, 5))
plt.plot(df.index, df['Total_Charges'], label='Historical Data')
plt.plot(forecast_dates, forecast_mean, label='Forecast for 2024', color='red')
plt.title('Forecast of Total Charges for 2024')
plt.xlabel('Date')
plt.ylabel('Total Charges')
plt.legend()
plt.grid(True)
plt.show()


# In[30]:


get_ipython().system('pip install Cython')
get_ipython().system('pip install fbprophet')


# In[31]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fbprophet import Prophet

# Load data and preprocess if necessary
# Assuming df contains the time series data

# Convert 'Date_of_request' to datetime
df['Date_of_request'] = pd.to_datetime(df['Date_of_request'], dayfirst=True)

# Aggregate charges by date
df = df.groupby('Date_of_request')['Total_Charges'].sum().reset_index()

# Rename columns for Prophet
df = df.rename(columns={'Date_of_request': 'ds', 'Total_Charges': 'y'})

# Initialize Prophet model
model_prophet = Prophet()

# Fit the model
model_prophet.fit(df)

# Make future dataframe for 90 days
future = model_prophet.make_future_dataframe(periods=90)

# Forecast
forecast = model_prophet.predict(future)

# Plot the forecast
fig = model_prophet.plot(forecast, xlabel='Date', ylabel='Total Charges')
plt.title('Prophet Forecast of Total Charges')
plt.show()


# In[32]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

# Load data and preprocess if necessary
# Assuming df contains the time series data

# Convert 'Date_of_request' to datetime
df['Date_of_request'] = pd.to_datetime(df['Date_of_request'], dayfirst=True)

# Aggregate charges by date
df = df.groupby('Date_of_request')['Total_Charges'].sum().reset_index()

# Set 'Date_of_request' as index
df.set_index('Date_of_request', inplace=True)

# Split data into train and test sets
train_size = int(len(df) * 0.8)
train, test = df.iloc[:train_size], df.iloc[train_size:]

# Fit SARIMAX model
order = (1, 0, 1)
seasonal_order = (1, 1, 1, 7)  # Assuming weekly seasonality
model = SARIMAX(train, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
model_fit = model.fit(disp=False)

# Forecast
forecast = model_fit.get_forecast(steps=len(test))

# Calculate RMSE
mse = mean_squared_error(test, forecast.predicted_mean)
rmse = np.sqrt(mse)
print(f'RMSE: {rmse}')

# Plot the forecast
plt.figure(figsize=(10, 5))
plt.plot(train.index, train, label='Training Data')
plt.plot(test.index, test, label='Test Data')
plt.plot(test.index, forecast.predicted_mean, label='SARIMAX Forecast', color='red')
plt.title('SARIMAX Forecast of Total Charges')
plt.xlabel('Date')
plt.ylabel('Total Charges')
plt.legend()
plt.grid(True)
plt.show()


# In[61]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Load data and preprocess if necessary
# Assuming df contains the time series data

# Convert 'Date_of_request' to datetime
df['Date_of_request'] = pd.to_datetime(df['Date_of_request'], dayfirst=True)

# Aggregate charges by date
df = df.groupby('Date_of_request')['Total_Charges'].sum().reset_index()

# Set 'Date_of_request' as index
df.set_index('Date_of_request', inplace=True)

# Fit SARIMAX model
order = (1, 0, 1)  # ARIMA order
seasonal_order = (1, 0, 1, 7)  # Seasonal order (weekly seasonality)
model_sarimax = SARIMAX(df, order=order, seasonal_order=seasonal_order, enforce_stationarity=False)
model_sarimax_fit = model_sarimax.fit()

# Forecast 90 days ahead starting from the last available date
forecast = model_sarimax_fit.forecast(steps=365)

# Plot the forecast
plt.figure(figsize=(10, 5))
plt.plot(df.index, df['Total_Charges'], label='Historical Data')
plt.plot(pd.date_range(start=df.index[-1], periods=365, freq='D'), forecast, label='SARIMAX Forecast', color='green')
plt.title('SARIMAX Forecast of Total Charges')
plt.xlabel('Date')
plt.ylabel('Total Charges')
plt.legend()
plt.grid(True)
plt.show()


# In[66]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL

# Load data and preprocess if necessary
# Assuming df contains the time series data

# Convert 'Date_of_request' to datetime
df['Date_of_request'] = pd.to_datetime(df['Date_of_request'], dayfirst=True)

# Aggregate charges by date
df = df.groupby('Date_of_request')['Total_Charges'].sum().reset_index()

# Set 'Date_of_request' as index
df.set_index('Date_of_request', inplace=True)

# Apply Seasonal Decomposition (STL)
stl = STL(df, seasonal=365)
result = stl.fit()

# Forecast 365 days ahead
forecast = result.forecast(steps=365)

# Plot the forecast
plt.figure(figsize=(10, 5))
plt.plot(df.index, df['Total_Charges'], label='Historical Data')
plt.plot(forecast.index, forecast, label='STL Forecast', color='green')
plt.title('STL Forecast of Total Charges')
plt.xlabel('Date')
plt.ylabel('Total Charges')
plt.legend()
plt.grid(True)
plt.show()


# In[97]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# Load data and preprocess if necessary
# Assuming df contains the time series data

# Convert 'Date_of_request' to datetime
df['Date_of_request'] = pd.to_datetime(df['Date_of_request'], dayfirst=True)

# Aggregate charges by date
df = df.groupby('Date_of_request')['Total_Charges'].sum().reset_index()

# Set 'Date_of_request' as index
df.set_index('Date_of_request', inplace=True)

# Fit ARIMA model
order = (1, 1, 1)  # ARIMA order (p,d,q)
model_arima = ARIMA(df, order=order)
model_arima_fit = model_arima.fit()

# Forecast 365 days ahead
forecast = model_arima_fit.forecast(steps=365)

# Calculate RMSE using the last 365 days of the original data
actual_data = df.iloc[-365:]
mse = mean_squared_error(actual_data, forecast)
rmse = np.sqrt(mse)
print(f'RMSE: {rmse}')

# Plot the forecast
plt.figure(figsize=(10, 5))
plt.plot(df.index, df['Total_Charges'], label='Historical Data')
plt.plot(forecast.index, forecast, label='ARIMA Forecast', color='green')
plt.title('ARIMA Forecast of Total Charges')
plt.xlabel('Date')
plt.ylabel('Total Charges')
plt.legend()
plt.grid(True)
plt.show()


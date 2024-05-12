#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Segmenter les voitures selon l'emission du co2


# In[1]:





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


# In[3]:


conn2 = ps.connect(dbname="SPN_DW", 
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
    co2."Type", 
    co2."CO2_Emission", 
    co2."Plate_Number"
FROM 
    public."CO2" AS co2
JOIN 
    public."FactTransactionsServices" AS fact ON co2."ID_CO2Emission"= fact."ID_CO2Emission"
JOIN
    public."CarJdid" AS cars ON cars."idCar" = fact."idCar";"""

# Rollback the current transaction
conn2.rollback()

# Re-execute your query
cur.execute(query)
rows = cur.fetchall()


# In[6]:


columns = [desc[0] for desc in cur.description]
df = pd.DataFrame(rows, columns=columns)


# In[7]:


print(df.head())


# In[8]:


# Supprimer les lignes où Plate_Number est 'None'
df_filtered = df.dropna(subset=['Plate_Number'])

# Afficher le DataFrame filtré
print(df_filtered)


# In[9]:


plate_numbers_unique = df_filtered['Plate_Number'].unique()
print(plate_numbers_unique)


# In[10]:


# Calculer le total des émissions de CO2 pour chaque Plate_Number
total_emissions_per_plate_number = df_filtered.groupby('Plate_Number')['CO2_Emission'].sum()

# Afficher les totaux des émissions de CO2 pour chaque Plate_Number
print(total_emissions_per_plate_number)


# In[11]:


from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Sélectionner les caractéristiques pertinentes (CO2_Emission)
X = df_filtered['CO2_Emission'].values.reshape(-1, 1)

# Choix du nombre optimal de clusters en utilisant la méthode de la silhouette
silhouette_scores = []
for i in range(2, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X)
    silhouette_scores.append(silhouette_score(X, kmeans.labels_))

# Tracer la courbe du score de silhouette pour choisir le nombre de clusters
plt.plot(range(2, 11), silhouette_scores, marker='o')
plt.xlabel('Nombre de clusters')
plt.ylabel('Score de silhouette')
plt.title('Méthode de la silhouette pour choisir le nombre de clusters')
plt.show()

# Entraîner le modèle KMeans avec le nombre optimal de clusters
optimal_num_clusters =3   # Choix arbitraire pour l'exemple
kmeans_model = KMeans(n_clusters=optimal_num_clusters, random_state=42)
kmeans_model.fit(X)

# Obtenir les étiquettes de cluster assignées à chaque échantillon
cluster_labels = kmeans_model.labels_

# Ajouter les étiquettes de cluster au DataFrame
df_filtered['Cluster'] = cluster_labels

# Analyser les clusters
cluster_centers = kmeans_model.cluster_centers_
for i, center in enumerate(cluster_centers):
    print(f"Cluster {i+1} - Centre: {center}")

# Visualisation des clusters en fonction des émissions de CO2 et des Plate_Number
plt.scatter(df_filtered.index, df_filtered['CO2_Emission'], c=df_filtered['Cluster'], cmap='viridis')
plt.xlabel('Plate_Number')
plt.ylabel('CO2_Emission')
plt.title('Clustering des émissions de CO2 en fonction de Plate_Number')
plt.colorbar(label='Cluster')
plt.show()


# In[19]:


# Trier le DataFrame en fonction des Plate_Number
df_sorted = df_filtered.sort_index()

# Visualisation des clusters en fonction des émissions de CO2 et des Plate_Number
plt.scatter(df_sorted['Plate_Number'], df_sorted['CO2_Emission'], c=df_sorted['Cluster'], cmap='viridis')
plt.xlabel('Plate_Number')
plt.ylabel('CO2_Emission')
plt.title('Clustering des émissions de CO2 en fonction de Plate_Number')
plt.colorbar(label='Cluster')
plt.xticks(rotation=90)  # Faire pivoter les étiquettes de l'axe des x pour une meilleure lisibilité
plt.show()


# In[12]:


# Supposons que df_filtered est le DataFrame contenant les données des voitures avec les clusters attribués

def recommander_voiture(cluster_id, nb_recommandations=5):
    # Sélectionner les voitures appartenant au cluster donné
    cluster_cars = df_filtered[df_filtered['Cluster'] == cluster_id]
    
    # Exclure la voiture actuelle de l'utilisateur (le cas échéant)
    # Vous pouvez demander à l'utilisateur de saisir la voiture dont il souhaite obtenir des recommandations
    # Par exemple : current_car = input("Entrez le numéro de la voiture actuelle : ")
    # Ensuite, vous pouvez exclure cette voiture des recommandations
    
    # Recommender les voitures similaires dans le même cluster
    recommandations = cluster_cars.sample(n=nb_recommandations, replace=False)
    
    return recommandations

# Exemple d'utilisation
cluster_id_utilisateur = 0  # Suppose que l'utilisateur est dans le cluster 0
recommandations = recommander_voiture(cluster_id_utilisateur)
print("Voitures recommandées pour l'utilisateur :")
print(recommandations)


# In[13]:


from sklearn.cluster import AgglomerativeClustering
import seaborn as sns

# Sélectionner les caractéristiques pertinentes (CO2_Emission)
X = df_filtered['CO2_Emission'].values.reshape(-1, 1)

# Choix du nombre de clusters pour le CAH (vous pouvez ajuster ce nombre selon vos besoins)
num_clusters = 3

# Créer un modèle de CAH
cah_model = AgglomerativeClustering(n_clusters=num_clusters)

# Adapter le modèle aux données et prédire les clusters
cah_labels = cah_model.fit_predict(X)

# Ajouter les étiquettes de cluster au DataFrame
df_filtered['CAH_Cluster'] = cah_labels

# Analyser les clusters
cluster_counts = df_filtered['CAH_Cluster'].value_counts()
print("Nombre d'échantillons dans chaque cluster (CAH) :")
print(cluster_counts)

# Visualisation des clusters en fonction des émissions de CO2
sns.scatterplot(data=df_filtered, x=df_filtered.index, y='CO2_Emission', hue='CAH_Cluster', palette='viridis')
plt.xlabel('Plate_Number')
plt.ylabel('CO2_Emission')
plt.title('Clustering des émissions de CO2 avec CAH')
plt.show()



# In[14]:


from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt

# Calculer la matrice de distance euclidienne entre les échantillons
distance_matrix = pdist(df_filtered['CO2_Emission'].values.reshape(-1, 1))

# Calculer la matrice de liaison (linkage matrix) avec la méthode 'ward'
Z = linkage(distance_matrix, method='ward')

# Tracer le dendrogramme
plt.figure(figsize=(10, 6))
dendrogram(Z)
plt.title('Dendrogramme pour le clustering hiérarchique agglomératif (CAH)')
plt.xlabel('Indice de l\'échantillon')
plt.ylabel('Distance euclidienne')
plt.show()


# In[15]:


from scipy.cluster.hierarchy import cophenet

# Calcul du score CAH
cah_score, cophenet_dist = cophenet(Z, distance_matrix)

print("Score de cohérence agglomératif (CAH) :", cah_score)


# In[16]:


import seaborn as sns

# Utiliser la colonne 'Type' pour la visualisation des clusters
sns.scatterplot(data=df_filtered, x='CO2_Emission', y='Type', hue='Cluster')


# In[17]:


from sklearn.metrics import silhouette_score

# Calculer le score de silhouette pour les clusters
silhouette_avg = silhouette_score(X, kmeans.labels_)
print("Score de silhouette moyen:", silhouette_avg)


# def recommander_vehicules(df, type_vehicule, limite_emission):
#     return df[(df["Type"] == type_vehicule) & (df["CO2_Emission"] <= limite_emission)]
# 
# # Fonction pour grouper par numéro de plaque et recommander pour chaque groupe
# def recommander_par_plaque(df, type_vehicule, limite_emission):
#     grouped = df.groupby("Plate_Number")
#     recommandations_par_plaque = pd.DataFrame(columns=df.columns)
#     for plaque, group in grouped:
#         recommandations = recommander_vehicules(group, type_vehicule, limite_emission)
#         if not recommandations.empty:
#             recommandations_par_plaque = pd.concat([recommandations_par_plaque, recommandations.head(1)])  # Ajoutez seulement la première recommandation
#     return recommandations_par_plaque
# 
# # Exemple d'utilisation
# type_vehicule = "Mercedes V Class"
# limite_emission = 0.5  # Limite d'émission de CO2 en tonnes
# recommandations = recommander_par_plaque(df, type_vehicule, limite_emission)
# print("Recommandations pour {} avec une limite d'émission de {} tonnes de CO2 :".format(type_vehicule, limite_emission))
# print(recommandations)

# In[29]:


def recommander_vehicules(df, type_vehicule, limite_emission):
    return df[(df["Type"] == type_vehicule) & (df["CO2_Emission"] <= limite_emission)]

# Fonction pour grouper par numéro de plaque et recommander pour chaque groupe
def recommander_par_plaque(df, type_vehicule, limite_emission):
    grouped = df.groupby("Plate_Number")
    recommandations_par_plaque = pd.DataFrame(columns=df.columns)
    for plaque, group in grouped:
        recommandations = recommander_vehicules(group, type_vehicule, limite_emission)
        if not recommandations.empty:
            recommandations_par_plaque = pd.concat([recommandations_par_plaque, recommandations.head(1)])  # Ajoutez seulement la première recommandation
    return recommandations_par_plaque

# Exemple d'utilisation
type_vehicule = "Mercedes GLE"
limite_emission = 14.0  # Limite d'émission de CO2 en tonnes
recommandations = recommander_par_plaque(df, type_vehicule, limite_emission)
print("Recommandations pour {} avec une limite d'émission de {} tonnes de CO2 :".format(type_vehicule, limite_emission))
print(recommandations)


# In[34]:


# Fonction de recommandation basée sur les émissions de CO2
def recommander_vehicules_par_CO2(df, limite_emission):
    return df[df["CO2_Emission"] <= limite_emission].groupby("Plate_Number").agg({"Type": "first", "CO2_Emission": "min"})

# Exemple d'utilisation
limite_emission = 200  # Limite d'émission de CO2 en tonnes
recommandations = recommander_vehicules_par_CO2(df, limite_emission)
print("Recommandations pour une limite d'émission de {} tonnes de CO2 :".format(limite_emission))
print(recommandations)


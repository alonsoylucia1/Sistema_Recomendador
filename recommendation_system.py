# recommendation_system.py

# importar librerias necesarias
import numpy as np
import pandas as pd
from   sklearn.model_selection import train_test_split
from   sklearn.metrics import mean_squared_error                
from   sklearn.neighbors import NearestNeighbors
# cargar el dataset
def load_data():
    # Cargar el dataset de ratings
    ratings = pd.read_csv('ratings.csv')
    return ratings  
# preprocesar los datos
def preprocess_data(ratings):
    # Convertir el dataset a una matriz de usuarios y películas
    user_movie_matrix = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)
    return user_movie_matrix
# entrenar el modelo
def train_model(user_movie_matrix):
    # Entrenar el modelo de vecinos más cercanos
    model = NearestNeighbors(metric='cosine', algorithm='brute')
    model.fit(user_movie_matrix.values)
    return model
# hacer recomendaciones
def make_recommendations(model, user_movie_matrix, user_id, n_recommendations=5):
    # Encontrar el índice del usuario
    user_index = user_movie_matrix.index.get_loc(user_id)
    
    # Encontrar los vecinos más cercanos
    distances, indices = model.kneighbors(user_movie_matrix.iloc[user_index, :].values.reshape(1, -1), n_neighbors=n_recommendations+1)
    
    # Obtener las recomendaciones
    recommendations = []
    for i in range(1, len(distances.flatten())):
        movie_index = indices.flatten()[i]
        movie_id = user_movie_matrix.columns[movie_index]
        recommendations.append(movie_id)
    
    return recommendations





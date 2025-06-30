# recommendation_system.py

# importar librerias necesarias
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import NearestNeighbors

# cargar el dataset
def load_data():
    # Cargar el dataset de ratings con separador '|', sin encabezados
    ratings = pd.read_csv('dataset/ratings.csv', sep='|', header=None, names=['userId', 'movieTitle', 'rating'])
    return ratings

# preprocesar los datos
def preprocess_data(ratings):
    # Convertir el dataset a una matriz de usuarios y películas (por título)
    user_movie_matrix = ratings.pivot(index='userId', columns='movieTitle', values='rating').fillna(0)
    return user_movie_matrix

# entrenar el modelo
def train_model(user_movie_matrix):
    # Entrenar el modelo de vecinos más cercanos
    model = NearestNeighbors(metric='cosine', algorithm='brute')
    model.fit(user_movie_matrix.values)
    return model

# hacer recomendaciones
def make_recommendations(model, user_movie_matrix, user_id, n_recommendations=5):
    # Verificar si el usuario existe
    if user_id not in user_movie_matrix.index:
        raise ValueError(f"El usuario {user_id} no existe en el dataset.")
    user_index = user_movie_matrix.index.get_loc(user_id)
    distances, indices = model.kneighbors(user_movie_matrix.iloc[user_index, :].values.reshape(1, -1), n_neighbors=n_recommendations+1)
    # Obtener películas que el usuario no ha visto
    user_ratings = user_movie_matrix.iloc[user_index]
    unseen_movies = user_ratings[user_ratings == 0].index
    # Recomendar películas vistas por usuarios similares que el usuario no ha visto
    recommendations = []
    for neighbor_idx in indices.flatten()[1:]:
        neighbor_ratings = user_movie_matrix.iloc[neighbor_idx]
        for movie in unseen_movies:
            if neighbor_ratings[movie] > 0 and movie not in recommendations:
                recommendations.append(movie)
            if len(recommendations) >= n_recommendations:
                break
        if len(recommendations) >= n_recommendations:
            break
    return recommendations

if __name__ == "__main__":
    ratings = load_data()
    user_movie_matrix = preprocess_data(ratings)
    model = train_model(user_movie_matrix)
    # Cambia el user_id por uno existente en tu dataset
    user_id = ratings['userId'].iloc[0]
    recomendaciones = make_recommendations(model, user_movie_matrix, user_id=user_id)
    print(f"Recomendaciones para el usuario {user_id}:")
    for rec in recomendaciones:
        print(rec)





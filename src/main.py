# -*- coding: utf-8 -*-
"""
Projeto: Previsão do Uso de Bicicletas Alugadas
Autor: Alencar Porto
Data: 04/01/2025

Este projeto prevê o número de bicicletas alugadas diariamente usando uma rede neural em TensorFlow.
"""

# Importação das bibliotecas necessárias
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from math import sqrt
import os

# Caminho do dataset
DATA_PATH = os.path.join("..", "data", "bike-sharing-daily.csv")

def load_and_clean_data(file_path):
    """Carrega e limpa os dados."""
    bike = pd.read_csv(file_path)
    bike = bike.drop(labels=['instant', 'casual', 'registered'], axis=1)
    bike['dteday'] = pd.to_datetime(bike['dteday'])
    bike.index = pd.DatetimeIndex(bike['dteday'])
    bike = bike.drop(labels=['dteday'], axis=1)
    return bike

def visualize_data(bike):
    """Visualiza os dados e suas correlações."""
    bike['cnt'].asfreq('W').plot(linewidth=3, title="Bike usage per week")
    plt.show()
    sns.heatmap(bike.corr(), annot=True)
    plt.title('Correlation Heatmap')
    plt.show()

def preprocess_data(bike):
    """Prepara os dados para o modelo."""
    X_cat = bike[['season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday', 'weathersit']]
    X_numerical = bike[['temp', 'hum', 'windspeed']]
    y = bike[['cnt']]

    # One-hot encoding para variáveis categóricas
    onehotencoder = OneHotEncoder()
    X_cat = onehotencoder.fit_transform(X_cat).toarray()

    # Combine categorias e numéricos
    X = np.hstack([X_cat, X_numerical.values])

    # Escalar a variável dependente
    scaler = MinMaxScaler()
    y_scaled = scaler.fit_transform(y)

    # Dividir os dados
    X_train, X_test, y_train, y_test = train_test_split(X, y_scaled, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, scaler

def build_and_train_model(X_train, y_train):
    """Constrói e treina o modelo."""
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(units=100, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(units=100, activation='relu'),
        tf.keras.layers.Dense(units=1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    history = model.fit(X_train, y_train, epochs=25, batch_size=50, validation_split=0.2)
    return model, history

def evaluate_model(model, history, X_test, y_test, scaler):
    """Avalia o modelo."""
    # Plotar a perda
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss Progress During Training')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Prever e avaliar
    y_pred = model.predict(X_test)
    y_pred_orig = scaler.inverse_transform(y_pred)
    y_test_orig = scaler.inverse_transform(y_test)

    mae = mean_absolute_error(y_test_orig, y_pred_orig)
    mse = mean_squared_error(y_test_orig, y_pred_orig)
    rmse = sqrt(mse)
    r2 = r2_score(y_test_orig, y_pred_orig)
    print(f"MAE: {mae:.2f}, MSE: {mse:.2f}, RMSE: {rmse:.2f}, R2: {r2:.2f}")

    # Gráfico de comparação
    plt.scatter(y_test_orig, y_pred_orig, color='red')
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title('True Values vs Predictions')
    plt.show()

def main():
    """Função principal."""
    bike = load_and_clean_data(DATA_PATH)
    visualize_data(bike)
    X_train, X_test, y_train, y_test, scaler = preprocess_data(bike)
    model, history = build_and_train_model(X_train, y_train)
    evaluate_model(model, history, X_test, y_test, scaler)

if __name__ == "__main__":
    main()

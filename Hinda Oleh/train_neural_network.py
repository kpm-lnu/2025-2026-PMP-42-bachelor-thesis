"""
Проста нейронна мережа для передбачення переміщень
на основі даних, згенерованих МСЕ
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def load_data(json_file="training_data.json"):
    """Завантажує дані з JSON файлу"""
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    X = []  # вхідні параметри: [p, E, r, z]
    y = []  # вихідні значення: u_r
    
    for sample in data:
        p = sample["parameters"]["pressure"]
        E = sample["parameters"]["E"]
        
        for node in sample["result"]["nodes"]:
            r = node["r"]
            z = node["z"]
            u_r = node["u_r"]
            
            X.append([p, E, r, z])
            y.append(u_r)
    
    return np.array(X), np.array(y)


def build_model(input_dim=4):
    """Створює нейронну мережу"""
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    return model


def train_and_evaluate(X, y, epochs=100, batch_size=32, test_size=0.2):
    """Навчає нейронну мережу та оцінює результати"""
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
    
    model = build_model(input_dim=X.shape[1])
    
    print("\n" + "=" * 60)
    print("АРХІТЕКТУРА НЕЙРОННОЇ МЕРЕЖІ")
    print("=" * 60)
    model.summary()
    
    print("\n" + "=" * 60)
    print("НАВЧАННЯ НЕЙРОННОЇ МЕРЕЖІ")
    print("=" * 60)
    
    history = model.fit(
        X_train_scaled, y_train_scaled,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        verbose=1
    )
    
    y_pred_scaled = model.predict(X_test_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled).flatten()
    
    mae = np.mean(np.abs(y_test - y_pred))
    mse = np.mean((y_test - y_pred)**2)
    r2 = 1 - np.sum((y_test - y_pred)**2) / np.sum((y_test - np.mean(y_test))**2)
    
    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТИ ОЦІНКИ")
    print("=" * 60)
    print(f"MAE (середня абсолютна похибка): {mae:.4f}")
    print(f"MSE (середньоквадратична похибка): {mse:.4f}")
    print(f"R² (коефіцієнт детермінації): {r2:.4f}")
    
    plot_results(history, y_test, y_pred)
    
    return model, scaler_X, scaler_y, history


def plot_results(history, y_test, y_pred):
    """Візуалізує результати навчання"""
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].plot(history.history['loss'], label='Training')
    axes[0].plot(history.history['val_loss'], label='Validation')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss (MSE)')
    axes[0].set_title('Loss dynamics during training')
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].scatter(y_test, y_pred, alpha=0.5)
    axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[1].set_xlabel('True u_r')
    axes[1].set_ylabel('Predicted u_r')
    axes[1].set_title('Predictions vs Reality')
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('loss_history.png', dpi=300, bbox_inches='tight')
    plt.show()


def predict_example(model, scaler_X, scaler_y):
    """Приклад використання навченої моделі"""
    
    print("\n" + "=" * 60)
    print("ПРИКЛАД ПЕРЕДБАЧЕННЯ")
    print("=" * 60)
    
    test_input = np.array([[50.0, 1.0, 1.0, 0.5]])
    test_input_scaled = scaler_X.transform(test_input)
    prediction_scaled = model.predict(test_input_scaled)
    prediction = scaler_y.inverse_transform(prediction_scaled)
    
    print(f"Вхідні параметри: p=50, E=1.0, r=1.0, z=0.5")
    print(f"Передбачене переміщення u_r = {prediction[0][0]:.4f}")


def main():
    print("=" * 60)
    print("НЕЙРОННА МЕРЕЖА ДЛЯ ПЕРЕДБАЧЕННЯ ПЕРЕМІЩЕНЬ")
    print("=" * 60)
    
    print("\nЗавантаження даних...")
    X, y = load_data("training_data.json")
    print(f"Завантажено {len(X)} зразків")
    print(f"Вхідні параметри: p, E, r, z")
    print(f"Вихідні параметри: u_r")
    
    model, scaler_X, scaler_y, history = train_and_evaluate(
        X, y, 
        epochs=100,
        batch_size=16,
        test_size=0.2
    )
    
    predict_example(model, scaler_X, scaler_y)
    model.save("neural_network_model.h5")
    print("\nМодель збережено у файл: neural_network_model.h5")


if __name__ == "__main__":
    main()
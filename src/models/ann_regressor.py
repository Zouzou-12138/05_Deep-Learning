"""
ANN Regressor/Classifier Module
Implements a Multi-Layer Perceptron (MLP) for structured data prediction.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler


class MobilePriceANN:
    def __init__(self, input_dim, num_classes=None, is_regression=False):
        """
        Initialize the ANN model.

        Args:
            input_dim (int): Number of input features.
            num_classes (int): Number of output classes (for classification).
            is_regression (bool): If True, configure for regression task.
        """
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.is_regression = is_regression
        self.model = self._build_model()
        self.scaler = StandardScaler()

    def _build_model(self):
        """Constructs the neural network architecture."""
        model = Sequential([
            # Input Layer + First Hidden Layer
            Dense(64, activation='relu', input_shape=(self.input_dim,)),
            BatchNormalization(),
            Dropout(0.3),

            # Second Hidden Layer
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),

            # Third Hidden Layer
            Dense(16, activation='relu'),
            Dropout(0.2),
        ])

        # Output Layer configuration based on task type
        if self.is_regression:
            model.add(Dense(1, activation='linear'))  # Regression output
            loss_func = 'mse'
            metrics = ['mae']
        else:
            # Classification output
            activation_func = 'softmax' if self.num_classes > 2 else 'sigmoid'
            units = self.num_classes if self.num_classes > 2 else 1
            model.add(Dense(units, activation=activation_func))

            loss_func = 'categorical_crossentropy' if self.num_classes > 2 else 'binary_crossentropy'
            metrics = ['accuracy']

        model.compile(optimizer=Adam(learning_rate=0.001), loss=loss_func, metrics=metrics)
        return model

    def preprocess_data(self, X_train, X_test=None):
        """Standardize input features."""
        X_train_scaled = self.scaler.fit_transform(X_train)
        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
            return X_train_scaled, X_test_scaled
        return X_train_scaled

    def train(self, X, y, epochs=50, batch_size=32, validation_split=0.2):
        """Train the model."""
        # Scale data before training
        X_scaled = self.preprocess_data(X)

        history = self.model.fit(
            X_scaled, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1
        )
        return history

    def predict(self, X):
        """Make predictions on new data."""
        X_scaled = self.preprocess_data(X)
        return self.model.predict(X_scaled)
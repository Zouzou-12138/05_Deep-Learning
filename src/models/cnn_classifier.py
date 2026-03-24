"""
CNN Classifier Module
Implements a Convolutional Neural Network for image classification tasks.
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class ImageClassifierCNN:
    def __init__(self, input_shape=(64, 64, 3), num_classes=10):
        """
        Initialize the CNN model.

        Args:
            input_shape (tuple): Height, Width, Channels of input images.
            num_classes (int): Number of image categories.
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self._build_model()

    def _build_model(self):
        """Constructs the CNN architecture."""
        model = Sequential([
            # Convolutional Block 1
            Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),

            # Convolutional Block 2
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),

            # Fully Connected Layers
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),

            # Output Layer
            Dense(self.num_classes, activation='softmax')
        ])

        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    def get_data_generators(self, train_dir, test_dir, batch_size=32):
        """
        Configure data generators with augmentation for training.

        Args:
            train_dir (str): Path to training data folder.
            test_dir (str): Path to testing data folder.
            batch_size (int): Batch size for training.
        """
        # Data Augmentation for training set
        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )

        # Only rescaling for test set
        test_datagen = ImageDataGenerator(rescale=1. / 255)

        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(self.input_shape[0], self.input_shape[1]),
            batch_size=batch_size,
            class_mode='categorical'
        )

        test_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size=(self.input_shape[0], self.input_shape[1]),
            batch_size=batch_size,
            class_mode='categorical'
        )

        return train_generator, test_generator

    def train(self, train_generator, test_generator, epochs=15):
        """Train the model using generators."""
        history = self.model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // train_generator.batch_size,
            epochs=epochs,
            validation_data=test_generator,
            validation_steps=test_generator.samples // test_generator.batch_size
        )
        return history
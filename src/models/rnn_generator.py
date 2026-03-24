"""
RNN/LSTM Text Generator Module
Implements a sequence model for text generation (e.g., lyrics, poetry).
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint


class LyricGeneratorRNN:
    def __init__(self, vocab_size, max_sequence_len, embedding_dim=128):
        """
        Initialize the RNN model.

        Args:
            vocab_size (int): Size of the vocabulary (number of unique words/tokens).
            max_sequence_len (int): Maximum length of input sequences.
            embedding_dim (int): Dimensionality of the embedding vector.
        """
        self.vocab_size = vocab_size
        self.max_sequence_len = max_sequence_len
        self.embedding_dim = embedding_dim
        self.model = self._build_model()

    def _build_model(self):
        """Constructs the LSTM architecture."""
        model = Sequential([
            # Embedding Layer: Converts integer tokens to dense vectors
            Embedding(input_dim=self.vocab_size, output_dim=self.embedding_dim, input_length=self.max_sequence_len),

            # LSTM Layer 1
            LSTM(128, return_sequences=True),
            Dropout(0.2),

            # LSTM Layer 2
            LSTM(128),
            Dropout(0.2),

            # Dense Output Layer
            Dense(128, activation='relu'),
            Dense(self.vocab_size, activation='softmax')
        ])

        model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )
        return model

    def train(self, X, y, epochs=50, batch_size=64, save_path='experiments/trained_models/lyric_gen.h5'):
        """
        Train the model on sequence data.

        Args:
            X (np.array): Input sequences (padding applied).
            y (np.array): One-hot encoded next words.
            epochs (int): Number of training epochs.
            batch_size (int): Batch size.
            save_path (str): Path to save the best model weights.
        """
        checkpoint = ModelCheckpoint(
            save_path,
            monitor='loss',
            verbose=1,
            save_best_only=True,
            mode='min'
        )

        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[checkpoint],
            verbose=1
        )
        return history

    def generate_text(self, seed_text, tokenizer, next_words=50, temperature=1.0):
        """
        Generate text autoregressively.

        Args:
            seed_text (str): The starting phrase.
            tokenizer: Keras Tokenizer object fitted on training data.
            next_words (int): Number of words to generate.
            temperature (float): Sampling temperature (higher = more random).
        """
        output_text = seed_text

        for _ in range(next_words):
            # Tokenize and pad the current sequence
            token_list = tokenizer.texts_to_sequences([output_text])[0]
            token_list = tf.keras.preprocessing.sequence.pad_sequences(
                [token_list], maxlen=self.max_sequence_len, padding='pre'
            )

            # Predict probabilities
            predicted_probs = self.model.predict(token_list, verbose=0)[0]

            # Apply temperature scaling
            predicted_probs = np.log(predicted_probs) / temperature
            exp_probs = np.exp(predicted_probs)
            renormalized_probs = exp_probs / np.sum(exp_probs)

            # Sample from the distribution
            predicted_index = np.random.choice(len(renormalized_probs), p=renormalized_probs)

            # Convert index back to word
            output_word = ""
            for word, index in tokenizer.word_index.items():
                if index == predicted_index:
                    output_word = word
                    break

            output_text += " " + output_word

        return output_text


"""
model_ssl_with_foundation.py - SSL-ECG model with foundation embeddings as input

This model uses 512-dim foundation embeddings (instead of raw 2560-sample ECG)
as input to the SSL-ECG transformation recognition architecture.

Architecture:
    Raw ECG (2560) → Foundation Model (frozen) → 512-dim embeddings
                                                       ↓
                                          SSL-ECG Encoder (trainable)
                                                       ↓
                                          Multi-task transformation prediction (7 tasks)
                                                       ↓
                                          Learned features (256-dim) → Downstream tasks

The foundation model embeddings replace the raw ECG input, providing richer
pre-trained representations for the SSL learning process.

Author: Claude (2025-11-14)
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


def build_ssl_encoder_for_embeddings(
    input_dim=512,
    hidden_dims=[256, 256, 128],
    output_dim=256,
    dropout_rate=0.5,
    l2_reg=0.0001,
    name='ssl_encoder'
):
    """
    Build SSL-ECG encoder that takes foundation embeddings as input.

    This encoder learns to predict transformations applied to ECG signals
    by working with 512-dim foundation embeddings instead of raw signals.

    Args:
        input_dim: Dimension of input embeddings (default: 512 from foundation model)
        hidden_dims: List of hidden layer dimensions
        output_dim: Dimension of learned feature representation
        dropout_rate: Dropout rate for regularization
        l2_reg: L2 regularization coefficient
        name: Model name

    Returns:
        Keras Model with:
            - Input: (batch_size, 512) foundation embeddings
            - Outputs:
                - features: (batch_size, output_dim) learned representations
                - task_0 to task_6: 7 transformation prediction heads
    """

    # Input: foundation embeddings
    input_embeddings = keras.Input(shape=(input_dim,), name='foundation_embeddings')

    # Shared encoder layers
    x = input_embeddings

    for i, hidden_dim in enumerate(hidden_dims):
        x = layers.Dense(
            hidden_dim,
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(l2_reg),
            name=f'{name}_dense_{i}'
        )(x)
        x = layers.BatchNormalization(name=f'{name}_bn_{i}')(x)
        x = layers.Dropout(dropout_rate, name=f'{name}_dropout_{i}')(x)

    # Feature representation
    features = layers.Dense(
        output_dim,
        activation='relu',
        kernel_regularizer=keras.regularizers.l2(l2_reg),
        name='features'
    )(x)
    features = layers.BatchNormalization(name='features_bn')(features)

    # Multi-task transformation prediction heads
    # Task 0: Original vs transformed
    task_0 = layers.Dense(128, activation='relu', name='task_0_dense_1')(features)
    task_0 = layers.Dropout(dropout_rate, name='task_0_dropout')(task_0)
    task_0 = layers.Dense(1, activation='sigmoid', name='task_0')(task_0)

    # Task 1: Noise detection
    task_1 = layers.Dense(128, activation='relu', name='task_1_dense_1')(features)
    task_1 = layers.Dropout(dropout_rate, name='task_1_dropout')(task_1)
    task_1 = layers.Dense(1, activation='sigmoid', name='task_1')(task_1)

    # Task 2: Scale detection
    task_2 = layers.Dense(128, activation='relu', name='task_2_dense_1')(features)
    task_2 = layers.Dropout(dropout_rate, name='task_2_dropout')(task_2)
    task_2 = layers.Dense(1, activation='sigmoid', name='task_2')(task_2)

    # Task 3: Negation detection
    task_3 = layers.Dense(128, activation='relu', name='task_3_dense_1')(features)
    task_3 = layers.Dropout(dropout_rate, name='task_3_dropout')(task_3)
    task_3 = layers.Dense(1, activation='sigmoid', name='task_3')(task_3)

    # Task 4: Flip detection
    task_4 = layers.Dense(128, activation='relu', name='task_4_dense_1')(features)
    task_4 = layers.Dropout(dropout_rate, name='task_4_dropout')(task_4)
    task_4 = layers.Dense(1, activation='sigmoid', name='task_4')(task_4)

    # Task 5: Permutation detection
    task_5 = layers.Dense(128, activation='relu', name='task_5_dense_1')(features)
    task_5 = layers.Dropout(dropout_rate, name='task_5_dropout')(task_5)
    task_5 = layers.Dense(1, activation='sigmoid', name='task_5')(task_5)

    # Task 6: Time warping detection
    task_6 = layers.Dense(128, activation='relu', name='task_6_dense_1')(features)
    task_6 = layers.Dropout(dropout_rate, name='task_6_dropout')(task_6)
    task_6 = layers.Dense(1, activation='sigmoid', name='task_6')(task_6)

    # Create model
    model = keras.Model(
        inputs=input_embeddings,
        outputs={
            'features': features,
            'task_0': task_0,
            'task_1': task_1,
            'task_2': task_2,
            'task_3': task_3,
            'task_4': task_4,
            'task_5': task_5,
            'task_6': task_6
        },
        name=name
    )

    return model


def compile_ssl_model(
    model,
    learning_rate=0.001,
    loss_weights=[0.195, 0.195, 0.195, 0.0125, 0.0125, 0.195, 0.195]
):
    """
    Compile SSL model with multi-task loss.

    Args:
        model: Keras model
        learning_rate: Initial learning rate
        loss_weights: Weights for each transformation task

    Returns:
        Compiled model
    """

    # Loss for each task
    losses = {
        'task_0': 'binary_crossentropy',
        'task_1': 'binary_crossentropy',
        'task_2': 'binary_crossentropy',
        'task_3': 'binary_crossentropy',
        'task_4': 'binary_crossentropy',
        'task_5': 'binary_crossentropy',
        'task_6': 'binary_crossentropy',
    }

    # Loss weights
    loss_weights_dict = {
        'task_0': loss_weights[0],
        'task_1': loss_weights[1],
        'task_2': loss_weights[2],
        'task_3': loss_weights[3],
        'task_4': loss_weights[4],
        'task_5': loss_weights[5],
        'task_6': loss_weights[6],
    }

    # Metrics for each task
    metrics = {
        'task_0': ['accuracy'],
        'task_1': ['accuracy'],
        'task_2': ['accuracy'],
        'task_3': ['accuracy'],
        'task_4': ['accuracy'],
        'task_5': ['accuracy'],
        'task_6': ['accuracy'],
    }

    # Optimizer
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    # Compile
    model.compile(
        optimizer=optimizer,
        loss=losses,
        loss_weights=loss_weights_dict,
        metrics=metrics
    )

    return model


def extract_ssl_features(model, embeddings, batch_size=256):
    """
    Extract learned features from SSL model.

    Args:
        model: Trained SSL model
        embeddings: Foundation embeddings, shape (n_samples, 512)
        batch_size: Batch size for prediction

    Returns:
        features: Learned features, shape (n_samples, 256)
    """
    outputs = model.predict(embeddings, batch_size=batch_size, verbose=0)

    if isinstance(outputs, dict):
        features = outputs['features']
    else:
        # If multiple outputs returned as list/tuple
        features = outputs[0]

    return features


# Example usage and testing
if __name__ == '__main__':
    print("Testing SSL-ECG model with foundation embeddings\n")

    # Create model
    print("Creating SSL encoder...")
    model = build_ssl_encoder_for_embeddings(
        input_dim=512,
        hidden_dims=[256, 256, 128],
        output_dim=256,
        dropout_rate=0.5,
        l2_reg=0.0001
    )

    # Compile
    print("Compiling model...")
    model = compile_ssl_model(model, learning_rate=0.001)

    # Print summary
    print("\nModel Summary:")
    model.summary()

    # Test with dummy data
    print("\nTesting with dummy data...")
    dummy_embeddings = np.random.randn(100, 512).astype('float32')
    dummy_labels = {
        'task_0': np.random.randint(0, 2, (100, 1)).astype('float32'),
        'task_1': np.random.randint(0, 2, (100, 1)).astype('float32'),
        'task_2': np.random.randint(0, 2, (100, 1)).astype('float32'),
        'task_3': np.random.randint(0, 2, (100, 1)).astype('float32'),
        'task_4': np.random.randint(0, 2, (100, 1)).astype('float32'),
        'task_5': np.random.randint(0, 2, (100, 1)).astype('float32'),
        'task_6': np.random.randint(0, 2, (100, 1)).astype('float32'),
    }

    # Forward pass
    outputs = model.predict(dummy_embeddings[:10], verbose=0)
    print(f"✓ Forward pass works")
    print(f"  Features shape: {outputs['features'].shape}")
    print(f"  Task_0 shape: {outputs['task_0'].shape}")

    # Extract features
    features = extract_ssl_features(model, dummy_embeddings)
    print(f"✓ Feature extraction works: {features.shape}")

    # Train for one epoch
    print("\nTraining for 1 epoch...")
    history = model.fit(
        dummy_embeddings,
        dummy_labels,
        batch_size=32,
        epochs=1,
        verbose=1
    )

    print("\n✓ All tests passed!")

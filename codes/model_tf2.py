"""
model.py - TensorFlow 2.x version

Self-supervised model for ECG stress detection using modern TF2/Keras API.

This is a MIGRATED version from TensorFlow 1.14 to TensorFlow 2.15+
All functionality remains identical, but uses modern APIs:
- tf.keras.layers instead of tf.layers
- Keras Model API instead of session-based execution
- Eager execution by default
- No placeholders or sessions needed

Compatible with Python 3.8-3.11 and TensorFlow 2.x

Author: Migrated by Claude (2025-11-07)
Original: Sarkar et al. (2020)
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, regularizers
import os
import numpy as np
from sklearn import metrics as sklearn_metrics

window_size = 2560
transform_task = [0, 1, 2, 3, 4, 5, 6]


class SSLECGModel(keras.Model):
    """
    Self-supervised learning model for ECG using multi-task transformation recognition.

    The model learns representations by predicting which transformation was applied
    to the input signal (noise, scaling, negation, flipping, permutation, time warping).

    Architecture:
    - 3 convolutional blocks with increasing filters (32 -> 64 -> 128)
    - Global average pooling to extract features
    - 7 parallel task heads (one per transformation)

    Args:
        drop_rate (float): Dropout rate for regularization
        hidden_nodes (int): Number of hidden units in task heads
        l2_reg (float): L2 regularization coefficient
    """

    def __init__(self, drop_rate=0.6, hidden_nodes=128, l2_reg=0.0001, name='ssl_ecg_model', **kwargs):
        super(SSLECGModel, self).__init__(name=name, **kwargs)

        self.drop_rate = drop_rate
        self.hidden_nodes = hidden_nodes
        self.l2_reg = l2_reg

        # Conv Block 1
        self.conv1_1 = layers.Conv1D(32, 32, strides=1, padding='same',
                                      kernel_regularizer=regularizers.l2(l2_reg), name='conv_layer_1')
        self.conv1_2 = layers.Conv1D(32, 32, strides=1, padding='same',
                                      kernel_regularizer=regularizers.l2(l2_reg), name='conv_layer_2')
        self.leaky1_1 = layers.LeakyReLU(name='leaky_1_1')
        self.leaky1_2 = layers.LeakyReLU(name='leaky_1_2')

        self.gap1 = layers.GlobalAveragePooling1D(name='GAP1')
        self.mp1 = layers.MaxPooling1D(pool_size=8, strides=2, padding='valid', name='mp1')

        # Conv Block 2
        self.conv2_1 = layers.Conv1D(64, 16, strides=1, padding='same',
                                      kernel_regularizer=regularizers.l2(l2_reg), name='conv_layer_3')
        self.conv2_2 = layers.Conv1D(64, 16, strides=1, padding='same',
                                      kernel_regularizer=regularizers.l2(l2_reg), name='conv_layer_4')
        self.leaky2_1 = layers.LeakyReLU(name='leaky_2_1')
        self.leaky2_2 = layers.LeakyReLU(name='leaky_2_2')

        self.gap2 = layers.GlobalAveragePooling1D(name='GAP2')
        self.mp2 = layers.MaxPooling1D(pool_size=8, strides=2, padding='valid', name='mp2')

        # Conv Block 3
        self.conv3_1 = layers.Conv1D(128, 8, strides=1, padding='same',
                                      kernel_regularizer=regularizers.l2(l2_reg), name='conv_layer_5')
        self.conv3_2 = layers.Conv1D(128, 8, strides=1, padding='same',
                                      kernel_regularizer=regularizers.l2(l2_reg), name='conv_layer_6')
        self.leaky3_1 = layers.LeakyReLU(name='leaky_3_1')
        self.leaky3_2 = layers.LeakyReLU(name='leaky_3_2')

        self.gap3 = layers.GlobalAveragePooling1D(name='GAP3')
        self.gap_final = layers.GlobalAveragePooling1D(name='GAP_final')

        # Task heads (7 parallel heads for transformation recognition)
        self.task_heads = []
        for task_id in range(7):
            task_head = keras.Sequential([
                layers.Dense(hidden_nodes, kernel_regularizer=regularizers.l2(l2_reg),
                           name=f'task_{task_id}_dense_1'),
                layers.LeakyReLU(name=f'task_{task_id}_leaky_1'),
                layers.Dropout(drop_rate, name=f'task_{task_id}_dropout_1'),
                layers.Dense(hidden_nodes, kernel_regularizer=regularizers.l2(l2_reg),
                           name=f'task_{task_id}_dense_2'),
                layers.LeakyReLU(name=f'task_{task_id}_leaky_2'),
                layers.Dropout(drop_rate, name=f'task_{task_id}_dropout_2'),
                layers.Dense(1, name=f'task_{task_id}_output')
            ], name=f'task_{task_id}_head')
            self.task_heads.append(task_head)

    def call(self, inputs, training=False):
        """
        Forward pass through the network.

        Args:
            inputs: Input tensor of shape (batch_size, window_size, 1)
            training: Boolean flag for training mode

        Returns:
            Dictionary containing:
            - 'features': Extracted feature vector (for downstream tasks)
            - 'task_outputs': List of 7 task predictions
            - 'conv1', 'conv2', 'conv3': Intermediate feature maps
        """
        # Conv Block 1
        x = self.conv1_1(inputs)
        x = self.leaky1_1(x)
        x = self.conv1_2(x)
        x = self.leaky1_2(x)

        conv1_features = self.gap1(x)

        # Pool and continue
        x = self.mp1(x)

        # Conv Block 2
        x = self.conv2_1(x)
        x = self.leaky2_1(x)
        x = self.conv2_2(x)
        x = self.leaky2_2(x)

        conv2_features = self.gap2(x)

        # Pool and continue
        x = self.mp2(x)

        # Conv Block 3
        x = self.conv3_1(x)
        x = self.leaky3_1(x)
        x = self.conv3_2(x)
        x = self.leaky3_2(x)

        conv3_features = self.gap3(x)

        # Final GAP - this is the feature vector for downstream tasks
        features = self.gap_final(x)

        # Task heads
        task_outputs = []
        for task_head in self.task_heads:
            task_output = task_head(features, training=training)
            task_outputs.append(task_output)

        return {
            'features': features,
            'task_outputs': task_outputs,
            'conv1': conv1_features,
            'conv2': conv2_features,
            'conv3': conv3_features
        }

    def get_config(self):
        """Get model configuration for saving/loading."""
        config = super().get_config()
        config.update({
            'drop_rate': self.drop_rate,
            'hidden_nodes': self.hidden_nodes,
            'l2_reg': self.l2_reg
        })
        return config


def build_ssl_model(input_shape=(2560, 1), drop_rate=0.6, hidden_nodes=128, l2_reg=0.0001):
    """
    Build and return the self-supervised ECG model.

    Args:
        input_shape: Shape of input ECG windows (default: (2560, 1))
        drop_rate: Dropout rate for regularization
        hidden_nodes: Number of hidden units in task heads
        l2_reg: L2 regularization coefficient

    Returns:
        Compiled Keras model
    """
    model = SSLECGModel(
        drop_rate=drop_rate,
        hidden_nodes=hidden_nodes,
        l2_reg=l2_reg
    )

    # Build the model by calling it once
    dummy_input = tf.zeros((1,) + input_shape)
    _ = model(dummy_input, training=False)

    return model


def create_downstream_classifier(input_dim, output_dim, hidden_nodes=512, dropout=0.0, l2_reg=0.0):
    """
    Create a downstream classifier for stress/outcome prediction.

    This is the model used after extracting features from the self-supervised model.

    Args:
        input_dim: Dimension of input features
        output_dim: Number of output classes (2 for binary, >2 for multiclass)
        hidden_nodes: Number of hidden units
        dropout: Dropout rate
        l2_reg: L2 regularization coefficient

    Returns:
        Compiled Keras model for classification
    """
    model = keras.Sequential([
        layers.Dense(hidden_nodes, activation='relu',
                    kernel_regularizer=regularizers.l2(l2_reg),
                    input_shape=(input_dim,)),
        layers.Dropout(dropout),
        layers.Dense(hidden_nodes, activation='relu',
                    kernel_regularizer=regularizers.l2(l2_reg)),
        layers.Dropout(dropout),
        layers.Dense(output_dim, activation='sigmoid' if output_dim == 2 else 'softmax')
    ])

    loss = 'binary_crossentropy' if output_dim == 2 else 'categorical_crossentropy'
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=loss,
        metrics=['accuracy']
    )

    return model


def create_downstream_regressor(input_dim, hidden_nodes=512, dropout=0.0, l2_reg=0.0):
    """
    Create a downstream regressor for continuous outcome prediction.

    Args:
        input_dim: Dimension of input features
        hidden_nodes: Number of hidden units
        dropout: Dropout rate
        l2_reg: L2 regularization coefficient

    Returns:
        Compiled Keras model for regression
    """
    model = keras.Sequential([
        layers.Dense(hidden_nodes, activation='relu',
                    kernel_regularizer=regularizers.l2(l2_reg),
                    input_shape=(input_dim,)),
        layers.Dropout(dropout),
        layers.Dense(hidden_nodes, activation='relu',
                    kernel_regularizer=regularizers.l2(l2_reg)),
        layers.Dropout(dropout),
        layers.Dense(hidden_nodes, activation='relu',
                    kernel_regularizer=regularizers.l2(l2_reg)),
        layers.Dropout(dropout),
        layers.Dense(hidden_nodes, activation='relu',
                    kernel_regularizer=regularizers.l2(l2_reg)),
        layers.Dropout(dropout),
        layers.Dense(1, activation=None)
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mae',
        metrics=['mae', 'mse']
    )

    return model


def train_downstream_classifier(x_train, y_train, x_test, y_test,
                                identifier, kfold, output_dir,
                                epochs=100, batch_size=128, verbose=2):
    """
    Train a downstream classification model.

    Args:
        x_train: Training features
        y_train: Training labels (one-hot encoded)
        x_test: Test features
        y_test: Test labels (one-hot encoded)
        identifier: Name of the task (e.g., 'stress')
        kfold: Current fold number
        output_dir: Directory to save results
        epochs: Number of training epochs
        batch_size: Batch size for training
        verbose: Verbosity level

    Returns:
        Trained model and test predictions
    """
    input_dim = x_train.shape[1]
    output_dim = y_train.shape[1]

    model = create_downstream_classifier(input_dim, output_dim)

    # Callbacks
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=15, restore_best_weights=True
    )

    # Train
    history = model.fit(
        x_train, y_train,
        validation_data=(x_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=verbose
    )

    # Predict
    y_train_pred = model.predict(x_train, batch_size=batch_size, verbose=0)
    y_test_pred = model.predict(x_test, batch_size=batch_size, verbose=0)

    # Save predictions
    pred_dir = os.path.join(output_dir, 'prediction')
    os.makedirs(pred_dir, exist_ok=True)
    np.save(os.path.join(pred_dir, f'te_pred_{identifier}_fold_{kfold}.npy'), y_test_pred)
    np.save(os.path.join(pred_dir, f'te_real_{identifier}_fold_{kfold}.npy'), y_test)

    # Calculate metrics
    y_train_true = np.argmax(y_train, axis=1)
    y_test_true = np.argmax(y_test, axis=1)
    y_train_pred_class = np.argmax(y_train_pred, axis=1)
    y_test_pred_class = np.argmax(y_test_pred, axis=1)

    # Save metrics
    save_classification_metrics(
        y_train_true, y_train_pred_class,
        os.path.join(output_dir, 'ER_result', f'tr_{identifier}.csv'),
        kfold
    )
    save_classification_metrics(
        y_test_true, y_test_pred_class,
        os.path.join(output_dir, 'ER_result', f'te_{identifier}.csv'),
        kfold
    )

    return model, y_test_pred


def train_downstream_regressor(x_train, y_train, x_test, y_test,
                               identifier, kfold, output_dir,
                               epochs=100, batch_size=128, verbose=2):
    """
    Train a downstream regression model.

    Args:
        x_train: Training features
        y_train: Training targets
        x_test: Test features
        y_test: Test targets
        identifier: Name of the task (e.g., 'pdq', 'pss')
        kfold: Current fold number
        output_dir: Directory to save results
        epochs: Number of training epochs
        batch_size: Batch size for training
        verbose: Verbosity level

    Returns:
        Trained model and test predictions
    """
    input_dim = x_train.shape[1]

    model = create_downstream_regressor(input_dim)

    # Callbacks
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=15, restore_best_weights=True
    )

    # Train
    history = model.fit(
        x_train, y_train,
        validation_data=(x_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=verbose
    )

    # Predict
    y_train_pred = model.predict(x_train, batch_size=batch_size, verbose=0)
    y_test_pred = model.predict(x_test, batch_size=batch_size, verbose=0)

    # Save predictions
    pred_dir = os.path.join(output_dir, 'prediction')
    os.makedirs(pred_dir, exist_ok=True)
    np.save(os.path.join(pred_dir, f'te_pred_{identifier}_fold_{kfold}.npy'), y_test_pred)
    np.save(os.path.join(pred_dir, f'te_real_{identifier}_fold_{kfold}.npy'), y_test)

    # Calculate metrics
    save_regression_metrics(
        y_train, y_train_pred.flatten(),
        os.path.join(output_dir, 'ER_result', f'tr_{identifier}.csv'),
        kfold
    )
    save_regression_metrics(
        y_test, y_test_pred.flatten(),
        os.path.join(output_dir, 'ER_result', f'te_{identifier}.csv'),
        kfold
    )

    return model, y_test_pred


def save_classification_metrics(y_true, y_pred, filepath, kfold):
    """Save classification metrics to CSV."""
    import csv

    TN, FP, FN, TP = sklearn_metrics.confusion_matrix(y_true, y_pred).ravel()

    metrics_dict = {
        'fold': kfold,
        'Accuracy': np.round((TP + TN) / (TP + TN + FP + FN), 4),
        'F1': np.round(2*TP / (2*TP + FP + FN), 4),
        'Sensitivity': np.round(TP / (TP + FN), 4),
        'Specificity': np.round(TN / (FP + TN), 4),
        'PPV': np.round(TP / (TP + FP), 4),
        'NPV': np.round(TN / (TN + FN), 4),
        'rocauc': np.round(sklearn_metrics.roc_auc_score(y_true, y_pred), 4)
    }

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    file_exists = os.path.isfile(filepath)

    with open(filepath, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=metrics_dict.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(metrics_dict)


def save_regression_metrics(y_true, y_pred, filepath, kfold):
    """Save regression metrics to CSV."""
    import csv

    metrics_dict = {
        'fold': kfold,
        'mse': np.round(sklearn_metrics.mean_squared_error(y_true, y_pred), 4),
        'mae': np.round(sklearn_metrics.mean_absolute_error(y_true, y_pred), 4),
        'rmse': np.round(sklearn_metrics.mean_squared_error(y_true, y_pred, squared=False), 4),
        'r2_score': np.round(sklearn_metrics.r2_score(y_true, y_pred), 4)
    }

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    file_exists = os.path.isfile(filepath)

    with open(filepath, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=metrics_dict.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(metrics_dict)

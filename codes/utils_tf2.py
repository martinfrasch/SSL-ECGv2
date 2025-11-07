"""
utils.py - TensorFlow 2.x version

Utility functions for SSL-ECG including data batching, feature extraction,
and result processing.

This is a MIGRATED version from TensorFlow 1.14 to TensorFlow 2.15+
- Uses eager execution (no sessions)
- Modern TF2 APIs
- Compatible with Python 3.8-3.11

Author: Migrated by Claude (2025-11-07)
Original: Sarkar et al. (2020)
"""

import os
import tensorflow as tf
import numpy as np
import csv
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import signal_transformation_task as stt
from mlxtend.evaluate import confusion_matrix
import time
import cv2
from scipy import signal as scipy_signal
import matplotlib.pyplot as plt
import pickle

window_size = 2560
transform_task = [0, 1, 2, 3, 4, 5, 6]


# ============================================================================
# Label and Loss Functions
# ============================================================================

def get_label_matrix(labels, num_tasks=7):
    """
    Convert labels to matrix format for multi-task learning.

    Args:
        labels: Array of shape (batch_size, num_tasks) with one-hot encoded labels
        num_tasks: Number of transformation tasks

    Returns:
        List of label tensors, one per task
    """
    label_list = []
    for i in range(num_tasks):
        task_labels = labels[:, i:i+1]
        label_list.append(task_labels)
    return label_list


def calculate_weighted_loss(loss_coefficients, task_losses):
    """
    Calculate weighted combination of task losses.

    Args:
        loss_coefficients: List of weights for each task
        task_losses: List of loss values for each task

    Returns:
        Weighted sum of losses
    """
    total_loss = 0.0
    for coeff, loss in zip(loss_coefficients, task_losses):
        total_loss += coeff * loss
    return total_loss


# ============================================================================
# Data Batch Creation
# ============================================================================

def make_batch(signal_batch, noise_amount, scaling_factor, permutation_pieces,
               time_warping_pieces, time_warping_stretch_factor,
               time_warping_squeeze_factor):
    """
    Apply transformations to a batch of signals for self-supervised learning.

    For each signal in the batch, creates 7 versions:
    1. Original
    2. Noised
    3. Scaled
    4. Negated
    5. Flipped
    6. Permuted
    7. Time warped

    Args:
        signal_batch: Batch of ECG signals
        noise_amount: SNR for noise addition
        scaling_factor: Factor for scaling transformation
        permutation_pieces: Number of pieces for permutation
        time_warping_pieces: Number of pieces for time warping
        time_warping_stretch_factor: Stretch factor for time warping
        time_warping_squeeze_factor: Squeeze factor for time warping

    Yields:
        (transformed_batch, labels) tuple where:
        - transformed_batch: Shape (7, signal_length) array of transformations
        - labels: Shape (7, 7) one-hot encoded labels
    """
    for i in range(len(signal_batch)):
        signal = signal_batch[i]
        signal = np.trim_zeros(signal, 'b')
        sampling_freq = len(signal) // 10

        # Apply transformations
        noised_signal = stt.add_noise_with_SNR(signal, noise_amount=noise_amount)
        scaled_signal = stt.scaled(signal, factor=scaling_factor)
        negated_signal = stt.negate(signal)
        flipped_signal = stt.hor_filp(signal)
        permuted_signal = stt.permute(signal, pieces=permutation_pieces)
        time_warped_signal = stt.time_warp(
            signal, sampling_freq,
            pieces=time_warping_pieces,
            stretch_factor=time_warping_stretch_factor,
            squeeze_factor=time_warping_squeeze_factor
        )

        # Make time_warped_signal same size as original
        tw_start_index = np.random.randint(0, len(time_warped_signal) - len(signal) + 1)
        tw_stop_index = tw_start_index + len(signal)
        time_warped_signal = time_warped_signal[tw_start_index:tw_stop_index]

        # Reshape all signals
        signal = signal.reshape(len(signal), 1)
        noised_signal = noised_signal.reshape(len(noised_signal), 1)
        scaled_signal = scaled_signal.reshape(len(scaled_signal), 1)
        negated_signal = negated_signal.reshape(len(negated_signal), 1)
        flipped_signal = flipped_signal.reshape(len(flipped_signal), 1)
        permuted_signal = permuted_signal.reshape(len(permuted_signal), 1)
        time_warped_signal = time_warped_signal.reshape(len(time_warped_signal), 1)

        # Create batch and labels
        batch = [signal, noised_signal, scaled_signal, negated_signal,
                flipped_signal, permuted_signal, time_warped_signal]
        labels = tf.keras.utils.to_categorical(transform_task, num_classes=len(transform_task))

        # Pad sequences to same length
        batch = tf.keras.preprocessing.sequence.pad_sequences(
            batch, dtype='float32', padding='post'
        )

        yield batch, labels


def make_total_batch(data, length, batchsize, noise_amount, scaling_factor,
                    permutation_pieces, time_warping_pieces,
                    time_warping_stretch_factor, time_warping_squeeze_factor):
    """
    Create batches of transformed signals for training.

    Args:
        data: ECG signal data
        length: Total number of samples
        batchsize: Number of samples per batch
        (other args): Transformation parameters

    Yields:
        (total_batch, total_labels, counter, steps) tuple
    """
    steps = length // batchsize + 1

    for counter in range(steps):
        signal_batch = data[np.mod(np.arange(counter*batchsize, (counter+1)*batchsize), length)]

        gen_op = make_batch(
            signal_batch, noise_amount, scaling_factor,
            permutation_pieces, time_warping_pieces,
            time_warping_stretch_factor, time_warping_squeeze_factor
        )

        total_batch = np.array([])
        total_labels = np.array([])

        for batch, labels in gen_op:
            total_batch = np.vstack((total_batch, batch)) if total_batch.size else batch
            total_labels = np.vstack((total_labels, labels)) if total_labels.size else labels

        yield total_batch, total_labels, counter, steps


def unison_shuffled_copies(a, b):
    """Shuffle two arrays in unison."""
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


# ============================================================================
# Feature Extraction and Normalization
# ============================================================================

def extract_features_with_model(model, data, batch_size=128):
    """
    Extract features from ECG signals using the trained model.

    This is the TF2 version - much simpler than TF1!
    Just use model.predict() with eager execution.

    Args:
        model: Trained SSLECGModel instance
        data: ECG signals of shape (n_samples, signal_length)
        batch_size: Batch size for processing

    Returns:
        Extracted features of shape (n_samples, feature_dim)
    """
    # Ensure data has correct shape (n_samples, signal_length, 1)
    if len(data.shape) == 2:
        data = data.reshape(data.shape[0], data.shape[1], 1)

    # Extract features using the model
    # Model returns a dictionary, we want the 'features' key
    all_features = []

    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        outputs = model(batch, training=False)
        features = outputs['features'].numpy()
        all_features.append(features)

    return np.vstack(all_features)


def normalize_features(x_train, x_test, scaler_path=None):
    """
    Normalize features using training set statistics (StandardScaler).

    This ensures features have zero mean and unit variance, which improves
    model training and prevents features with larger scales from dominating.

    Args:
        x_train (numpy.ndarray): Training features (N_train, feature_dim)
        x_test (numpy.ndarray): Test features (N_test, feature_dim)
        scaler_path (str, optional): Path to save the fitted scaler

    Returns:
        tuple: (x_train_norm, x_test_norm, scaler)
            - x_train_norm: Normalized training features
            - x_test_norm: Normalized test features
            - scaler: Fitted StandardScaler object
    """
    scaler = StandardScaler()
    x_train_norm = scaler.fit_transform(x_train)
    x_test_norm = scaler.transform(x_test)

    if scaler_path is not None:
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"✓ Feature scaler saved to: {scaler_path}")

    print(f"✓ Features normalized - Train: {x_train_norm.shape}, Test: {x_test_norm.shape}")

    return x_train_norm, x_test_norm, scaler


# ============================================================================
# Results and Metrics
# ============================================================================

def get_results_ssl(y_true, y_pred):
    """
    Calculate accuracy and F1 scores for self-supervised tasks.

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        (accuracy, f1_score) arrays for each task
    """
    accuracy = np.full((1, 7), np.nan)
    f1_score = np.full((1, 7), np.nan)

    if y_true.shape == y_pred.shape:
        for i in range(len(transform_task)):
            accuracy[:, i] = np.round(metrics.accuracy_score(y_true[:, i], y_pred[:, i]), 2)
            f1_score[:, i] = np.round(metrics.f1_score(y_true[:, i], y_pred[:, i], labels=[0, 1]), 2)
    else:
        print("Error in self-supervised result calculation: shape mismatch")

    return accuracy, f1_score


def write_result(accuracy, f1_score, epoch_number, result_dict):
    """Store results in dictionary."""
    result = [accuracy, f1_score]
    result_dict.update({epoch_number: result})
    return result_dict


def write_result_csv(kfold, epoch_number, result_store, f1_score):
    """Write F1 scores to CSV file."""
    f1_score = f1_score[0]
    os.makedirs(os.path.dirname(result_store), exist_ok=True)

    with open(result_store, 'a', newline='') as csvfile:
        fieldnames = ['fold', 'epoch', 'org', 'noised', 'scaled', 'neg', 'flip', 'perm', 'time_warp']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write header if file is empty
        if os.stat(result_store).st_size == 0:
            writer.writeheader()

        writer.writerow({
            'fold': kfold,
            'epoch': epoch_number,
            'org': f1_score[0],
            'noised': f1_score[1],
            'scaled': f1_score[2],
            'neg': f1_score[3],
            'flip': f1_score[4],
            'perm': f1_score[5],
            'time_warp': f1_score[6]
        })


def fetch_all_loss(all_losses, loss_task):
    """Accumulate losses for each task."""
    for i in range(len(transform_task)):
        loss_task[i] = np.add(loss_task[i], all_losses[i])
    return loss_task


def fetch_pred_labels(y_preds, pred_task):
    """Accumulate predicted labels."""
    y_preds = np.squeeze(np.asarray(y_preds, dtype=np.int32)).T
    if np.all(pred_task == -1):
        pred_task = y_preds
    else:
        pred_task = np.vstack((pred_task, y_preds))
    return pred_task


def fetch_true_labels(labels, true_task):
    """Accumulate true labels."""
    if np.all(true_task == -1):
        true_task = labels
    else:
        true_task = np.vstack((true_task, labels))
    return true_task


# ============================================================================
# Utility Functions
# ============================================================================

def current_time():
    """Get current system time as string."""
    return time.strftime("%Y_%m_%d_%H_%M", time.gmtime())


def makedirs(path):
    """Create directory if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path)


def make_window(signal, fs, overlap, window_size_sec):
    """
    Segment signal into overlapping windows.

    Args:
        signal: Input signal
        fs: Sampling frequency
        overlap: Overlap percentage (0-100)
        window_size_sec: Window size in seconds

    Returns:
        Segmented signal array
    """
    window_size = fs * window_size_sec
    overlap = int(window_size * (overlap / 100))
    start = 0
    segmented = np.zeros((1, window_size), dtype=int)

    while start + window_size <= len(signal):
        segment = signal[start:start+window_size]
        segment = segment.reshape(1, len(segment))
        segmented = np.append(segmented, segment, axis=0)
        start = start + window_size - overlap

    return segmented[1:]


def normalize(x, x_mean, x_std):
    """Z-score normalization."""
    x_scaled = (x - x_mean) / x_std
    return x_scaled


def load_data(path):
    """Load data from .npy file."""
    dataset = np.load(path, allow_pickle=True)
    return dataset


def save_list(mylist, filename):
    """Save list to CSV file."""
    for i in range(len(mylist)):
        temp = mylist[i]
        with open(filename, 'a', newline='') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            wr.writerow(temp)


def downsample(signal_orig, sampling_freq_old, sampling_freq_new):
    """Downsample signal to new frequency."""
    new_len = int(signal_orig.shape[0] / sampling_freq_old * sampling_freq_new)
    signal_downsampled = cv2.resize(signal_orig, (1, new_len), interpolation=cv2.INTER_LINEAR)
    return signal_downsampled


def filter_ecg(ecg, ecg_sampling_freq):
    """Apply bandpass filter to ECG signal."""
    b, a = scipy_signal.iirdesign(
        wp=0.8, ws=0.4, gpass=1, gstop=60,
        ftype='cheby2', fs=ecg_sampling_freq
    )
    filtered_ecg = scipy_signal.filtfilt(b, a, ecg, axis=0)
    return filtered_ecg


def plot(title, signal):
    """Plot signal with title."""
    plt.plot(signal)
    plt.title(title)
    plt.show()

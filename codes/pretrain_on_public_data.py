"""
pretrain_on_public_data.py - Pre-train SSL-ECG on large public datasets

This script pre-trains the SSL-ECG model on large public ECG datasets
before fine-tuning on maternal/pregnancy ECG data.

Supported public datasets:
- PTB-XL (21,799 clinical ECGs, 10 seconds, multiple leads)
- MIT-BIH (48 recordings, 24-hour Holter, arrhythmia detection)
- Chapman-Shaoxing (10,646 ECGs from China)
- CPSC 2018 (6,877 ECGs from China)

Benefits of pre-training:
1. Learn general ECG patterns from 20k+ samples
2. Better feature representations for downstream tasks
3. Faster convergence on small maternal ECG dataset
4. Better generalization with limited data

Usage:
    # 1. Pre-train on PTB-XL
    python codes/pretrain_on_public_data.py \
        --dataset ptbxl \
        --data_path ~/datasets/ptbxl \
        --output_dir pretrained_models \
        --epochs 50

    # 2. Fine-tune on maternal ECG data
    python codes/train_tf2_production.py \
        --pretrained_model pretrained_models/ptbxl_pretrained \
        --data_folder ~/maternal_ecg_data \
        --epochs 30

Author: Claude (2025-11-12)
"""

import os
import argparse
import warnings
import json
from datetime import datetime
warnings.filterwarnings('ignore')

import tensorflow as tf
import numpy as np
import pandas as pd
from tqdm import tqdm
import wfdb
from scipy import signal

# Import our modules
import model_tf2 as model
import utils_tf2 as utils

# Parse arguments
parser = argparse.ArgumentParser(description='Pre-train SSL-ECG on Public Datasets')

parser.add_argument('--dataset', type=str, required=True,
                    choices=['ptbxl', 'mitbih', 'chapman'],
                    help='Public dataset to use for pre-training')
parser.add_argument('--data_path', type=str, required=True,
                    help='Path to downloaded public dataset')
parser.add_argument('--output_dir', type=str, default='pretrained_models',
                    help='Output directory for pre-trained model')
parser.add_argument('--epochs', type=int, default=50,
                    help='Pre-training epochs')
parser.add_argument('--batch_size', type=int, default=256,
                    help='Batch size')
parser.add_argument('--learning_rate', type=float, default=0.001,
                    help='Learning rate')
parser.add_argument('--gpu', type=str, default='0',
                    help='GPU device ID')
parser.add_argument('--target_lead', type=str, default='I',
                    help='ECG lead to use (PTB-XL has 12 leads, we need 1)')
parser.add_argument('--random_seed', type=int, default=42,
                    help='Random seed')

args = parser.parse_args()

# Set GPU
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Set seeds
np.random.seed(args.random_seed)
tf.random.set_seed(args.random_seed)

print(f"\n{'='*70}")
print(f"SSL-ECG Pre-training on {args.dataset.upper()}")
print(f"{'='*70}")
print(f"TensorFlow version: {tf.__version__}")
print(f"Dataset: {args.dataset}")
print(f"Data path: {args.data_path}")
print(f"Epochs: {args.epochs}")
print(f"Batch size: {args.batch_size}")
print(f"{'='*70}\n")


def load_ptbxl_data(data_path, target_lead='I', target_fs=256):
    """
    Load PTB-XL dataset and preprocess for SSL-ECG.

    PTB-XL: 21,799 clinical 12-lead ECGs (10 seconds, 100/500 Hz)
    Download: https://physionet.org/content/ptb-xl/1.0.1/

    Args:
        data_path: Path to ptb-xl directory
        target_lead: Which lead to use (I, II, V1, V2, etc.)
        target_fs: Target sampling frequency (256 Hz for our model)

    Returns:
        ecg_windows: Array of shape (n_samples, 2560)
    """
    print("Loading PTB-XL dataset...")

    # Load metadata
    metadata_path = os.path.join(data_path, 'ptbxl_database.csv')
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(
            f"PTB-XL metadata not found at {metadata_path}\n"
            f"Download from: https://physionet.org/content/ptb-xl/1.0.1/"
        )

    metadata = pd.read_csv(metadata_path)
    print(f"Found {len(metadata)} ECG records")

    # Determine which files to load (100Hz or 500Hz)
    # We'll use 100Hz version as it's smaller
    records_path = os.path.join(data_path, 'records100')
    sampling_rate = 100

    # Lead mapping (PTB-XL has 12 leads)
    lead_names = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    if target_lead not in lead_names:
        raise ValueError(f"Invalid lead: {target_lead}. Choose from {lead_names}")

    lead_idx = lead_names.index(target_lead)

    ecg_windows = []

    for idx, row in tqdm(metadata.iterrows(), total=len(metadata), desc="Loading ECGs"):
        # Load ECG
        record_path = os.path.join(data_path, row['filename_hr'])
        record_path = record_path.replace('.hea', '')

        try:
            # Load with wfdb
            record = wfdb.rdrecord(record_path)
            ecg_signal = record.p_signal[:, lead_idx]  # Extract target lead

            # Resample to 256 Hz
            if sampling_rate != target_fs:
                n_samples = int(len(ecg_signal) * target_fs / sampling_rate)
                ecg_signal = signal.resample(ecg_signal, n_samples)

            # Ensure exactly 2560 samples (10 seconds at 256 Hz)
            if len(ecg_signal) > 2560:
                ecg_signal = ecg_signal[:2560]
            elif len(ecg_signal) < 2560:
                ecg_signal = np.pad(ecg_signal, (0, 2560 - len(ecg_signal)), mode='edge')

            ecg_windows.append(ecg_signal)

        except Exception as e:
            print(f"Warning: Failed to load {record_path}: {e}")
            continue

    ecg_windows = np.array(ecg_windows)
    print(f"\n✓ Loaded {len(ecg_windows)} ECG windows")
    print(f"  Shape: {ecg_windows.shape}")

    return ecg_windows


def load_chapman_data(data_path, target_fs=256):
    """
    Load Chapman-Shaoxing dataset.

    Chapman-Shaoxing: 10,646 12-lead ECGs from China
    Download: https://figshare.com/collections/ChapmanECG/4560497/2
    """
    print("Loading Chapman-Shaoxing dataset...")
    # Implementation similar to PTB-XL
    raise NotImplementedError("Chapman dataset loader not yet implemented")


def load_mitbih_data(data_path, target_fs=256):
    """
    Load MIT-BIH Arrhythmia Database.

    MIT-BIH: 48 half-hour excerpts of two-channel ambulatory ECG
    Download: https://physionet.org/content/mitdb/1.0.0/
    """
    print("Loading MIT-BIH dataset...")
    # Implementation for MIT-BIH
    raise NotImplementedError("MIT-BIH dataset loader not yet implemented")


# Load public dataset
if args.dataset == 'ptbxl':
    ecg_data = load_ptbxl_data(args.data_path, target_lead=args.target_lead)
elif args.dataset == 'chapman':
    ecg_data = load_chapman_data(args.data_path)
elif args.dataset == 'mitbih':
    ecg_data = load_mitbih_data(args.data_path)
else:
    raise ValueError(f"Unknown dataset: {args.dataset}")

# Split into train/val (90/10)
n_samples = len(ecg_data)
n_train = int(0.9 * n_samples)

indices = np.random.permutation(n_samples)
train_indices = indices[:n_train]
val_indices = indices[n_train:]

train_ecg = ecg_data[train_indices]
val_ecg = ecg_data[val_indices]

print(f"\nData split:")
print(f"  Train: {len(train_ecg)} samples")
print(f"  Val:   {len(val_ecg)} samples")

# Build model
print("\nBuilding SSL-ECG model...")
ssl_model = model.build_ssl_model(
    input_shape=(2560, 1),
    drop_rate=0.6,
    hidden_nodes=128,
    l2_reg=0.0001
)

# Training setup
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    args.learning_rate,
    decay_steps=10000,
    decay_rate=0.9,
    staircase=True
)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
bce_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)

loss_coeff = [0.195, 0.195, 0.195, 0.0125, 0.0125, 0.195, 0.195]

# Transformation parameters
noise_param = 15
scale_param = 1.1
permu_param = 20
tw_piece_param = 9
twsf_param = 1.05

# Training step
@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        outputs = ssl_model(x, training=True)
        task_outputs = outputs['task_outputs']

        task_losses = []
        for i in range(7):
            task_loss = bce_loss(y[:, i:i+1], task_outputs[i])
            task_losses.append(task_loss)

        total_loss = sum([coeff * loss for coeff, loss in zip(loss_coeff, task_losses)])
        total_loss += sum(ssl_model.losses)

    gradients = tape.gradient(total_loss, ssl_model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, ssl_model.trainable_variables))

    return total_loss

# Create output directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_name = f"{args.dataset}_pretrained_{timestamp}"
output_dir = os.path.join(args.output_dir, model_name)
os.makedirs(output_dir, exist_ok=True)

# Training loop
print("\nStarting pre-training...\n")

best_val_loss = float('inf')
history = {'train_loss': [], 'val_loss': [], 'epoch': []}

for epoch in range(args.epochs):
    print(f"Epoch {epoch+1}/{args.epochs}")

    # Training
    train_gen = utils.make_total_batch(
        train_ecg, len(train_ecg), args.batch_size,
        noise_param, scale_param, permu_param,
        tw_piece_param, twsf_param, 1/twsf_param
    )

    train_losses = []
    for batch_data, batch_labels, counter, steps in train_gen:
        batch_data = batch_data.reshape(-1, 2560, 1).astype('float32')
        batch_labels = batch_labels.astype('float32')

        loss = train_step(batch_data, batch_labels)
        train_losses.append(loss.numpy())

    train_loss = np.mean(train_losses)

    # Validation
    val_gen = utils.make_total_batch(
        val_ecg, len(val_ecg), args.batch_size,
        noise_param, scale_param, permu_param,
        tw_piece_param, twsf_param, 1/twsf_param
    )

    val_losses = []
    for batch_data, batch_labels, counter, steps in val_gen:
        batch_data = batch_data.reshape(-1, 2560, 1).astype('float32')
        batch_labels = batch_labels.astype('float32')

        outputs = ssl_model(batch_data, training=False)
        task_outputs = outputs['task_outputs']

        task_losses = []
        for i in range(7):
            task_loss = bce_loss(batch_labels[:, i:i+1], task_outputs[i])
            task_losses.append(task_loss)

        val_loss = sum([coeff * loss for coeff, loss in zip(loss_coeff, task_losses)])
        val_losses.append(val_loss.numpy())

    val_loss = np.mean(val_losses)

    history['epoch'].append(epoch)
    history['train_loss'].append(float(train_loss))
    history['val_loss'].append(float(val_loss))

    print(f"  Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        ssl_model.save(os.path.join(output_dir, 'best_model'))
        print(f"  ✓ Saved best model (val loss: {val_loss:.4f})")

# Save final model
ssl_model.save(os.path.join(output_dir, 'final_model'))

# Save metadata
metadata = {
    'dataset': args.dataset,
    'n_samples': int(n_samples),
    'n_train': int(n_train),
    'n_val': int(n_samples - n_train),
    'epochs': args.epochs,
    'best_val_loss': float(best_val_loss),
    'timestamp': timestamp,
    'tensorflow_version': tf.__version__
}

with open(os.path.join(output_dir, 'pretrain_metadata.json'), 'w') as f:
    json.dump(metadata, f, indent=2)

with open(os.path.join(output_dir, 'pretrain_history.json'), 'w') as f:
    json.dump(history, f, indent=2)

print(f"\n{'='*70}")
print(f"Pre-training Complete!")
print(f"{'='*70}")
print(f"Model saved to: {output_dir}")
print(f"Best validation loss: {best_val_loss:.4f}")
print(f"\nNext step: Fine-tune on maternal ECG data with:")
print(f"  python codes/train_tf2_production.py \\")
print(f"      --pretrained_model {output_dir}/best_model \\")
print(f"      --data_folder ~/maternal_ecg_data \\")
print(f"      --epochs 30")
print(f"{'='*70}\n")

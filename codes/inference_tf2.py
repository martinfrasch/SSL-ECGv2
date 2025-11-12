"""
inference_tf2.py - Inference pipeline for SSL-ECG trained models

This script loads a trained SSL-ECG model and runs inference on new ECG data.

Usage:
    # Basic inference - extract features only
    python inference_tf2.py \
        --model_dir trained_models/ssl_ecg_fold0_20251112_143022 \
        --input_data new_ecgs.npy \
        --output_file features.npy

    # Full inference - extract features and run downstream task
    python inference_tf2.py \
        --model_dir trained_models/ssl_ecg_fold0_20251112_143022 \
        --input_data new_ecgs.npy \
        --output_file predictions.csv \
        --task stress \
        --downstream_model trained_models/stress_classifier.h5

Input data format:
    - NumPy array (.npy file)
    - Shape: (n_samples, 2560) or (n_samples, 2560, 1)
    - Each sample is a 10-second ECG window at 256 Hz
    - Values should be in the same units as training data

Output:
    - If --task not specified: Features (n_samples, 256) saved as .npy
    - If --task specified: Predictions saved as .csv

Author: Claude (2025-11-12)
"""

import os
import argparse
import json
import pickle
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Parse arguments
parser = argparse.ArgumentParser(description='SSL-ECG Inference Pipeline')

# Required arguments
parser.add_argument('--model_dir', type=str, required=True,
                    help='Path to trained model directory')
parser.add_argument('--input_data', type=str, required=True,
                    help='Path to input ECG data (.npy file)')
parser.add_argument('--output_file', type=str, required=True,
                    help='Path to output file (.npy for features, .csv for predictions)')

# Optional arguments
parser.add_argument('--task', type=str, default=None,
                    choices=['stress', 'pss', 'pdq', 'fsi', 'cortisol'],
                    help='Downstream task (if not specified, only features are extracted)')
parser.add_argument('--downstream_model', type=str, default=None,
                    help='Path to downstream task model (.h5 or SavedModel directory)')
parser.add_argument('--batch_size', type=int, default=256,
                    help='Batch size for inference')
parser.add_argument('--normalize', type=lambda x: str(x).lower() == 'true', default=True,
                    help='Normalize features using saved scaler')
parser.add_argument('--model_format', type=str, default='savedmodel',
                    choices=['savedmodel', 'h5'],
                    help='Model format to load (savedmodel or h5)')
parser.add_argument('--verbose', type=int, default=1,
                    help='Verbosity level (0=quiet, 1=normal, 2=debug)')
parser.add_argument('--gpu', type=str, default='0',
                    help='GPU device ID')

args = parser.parse_args()

# Set GPU
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
gpus = tf.config.list_physical_devices('GPU')
if len(gpus) > 0:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if args.verbose >= 1:
        print(f"Using GPU: {gpus[0]}")
else:
    if args.verbose >= 1:
        print("No GPU detected, using CPU")

if args.verbose >= 1:
    print(f"\n{'='*70}")
    print(f"SSL-ECG Inference Pipeline")
    print(f"{'='*70}")
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Model directory: {args.model_dir}")
    print(f"Input data: {args.input_data}")
    print(f"Output file: {args.output_file}")
    if args.task:
        print(f"Task: {args.task}")
        if args.downstream_model:
            print(f"Downstream model: {args.downstream_model}")
    print(f"{'='*70}\n")

# Validate paths
if not os.path.exists(args.model_dir):
    raise FileNotFoundError(f"Model directory not found: {args.model_dir}")

if not os.path.exists(args.input_data):
    raise FileNotFoundError(f"Input data not found: {args.input_data}")

# Load model metadata
metadata_path = os.path.join(args.model_dir, 'deployment_metadata.json')
if os.path.exists(metadata_path):
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    if args.verbose >= 2:
        print("Model metadata:")
        print(json.dumps(metadata, indent=2))
else:
    if args.verbose >= 1:
        print("Warning: deployment_metadata.json not found, using defaults")
    metadata = {}

# Load configuration
config_path = os.path.join(args.model_dir, 'config.json')
if os.path.exists(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    if args.verbose >= 2:
        print("\nTraining configuration:")
        print(json.dumps(config, indent=2))
else:
    config = {}

# Load model
if args.verbose >= 1:
    print("Loading model...")

if args.model_format == 'savedmodel':
    model_path = os.path.join(args.model_dir, 'exported_model', 'saved_model')
    if not os.path.exists(model_path):
        # Try H5 as fallback
        if args.verbose >= 1:
            print(f"SavedModel not found at {model_path}, trying H5 format...")
        model_path = os.path.join(args.model_dir, 'exported_model', 'model.h5')
        args.model_format = 'h5'
else:
    model_path = os.path.join(args.model_dir, 'exported_model', 'model.h5')

if not os.path.exists(model_path):
    raise FileNotFoundError(
        f"Model not found at {model_path}\n"
        f"Expected either:\n"
        f"  - {os.path.join(args.model_dir, 'exported_model', 'saved_model')}\n"
        f"  - {os.path.join(args.model_dir, 'exported_model', 'model.h5')}"
    )

try:
    model = tf.keras.models.load_model(model_path)
    if args.verbose >= 1:
        print(f"✓ Model loaded successfully from: {model_path}")
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")

# Load feature scaler if normalization is enabled
scaler = None
if args.normalize:
    scaler_path = os.path.join(args.model_dir, 'feature_scaler.pkl')
    if os.path.exists(scaler_path):
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        if args.verbose >= 1:
            print(f"✓ Feature scaler loaded from: {scaler_path}")
    else:
        if args.verbose >= 1:
            print(f"Warning: Feature scaler not found at {scaler_path}")
            print("  Features will NOT be normalized (may affect performance)")
        args.normalize = False

# Load input data
if args.verbose >= 1:
    print(f"\nLoading input data from: {args.input_data}")

input_data = np.load(args.input_data, allow_pickle=True)

if args.verbose >= 1:
    print(f"Input data shape: {input_data.shape}")

# Validate and reshape input data
if len(input_data.shape) == 2:
    # Shape: (n_samples, 2560) -> (n_samples, 2560, 1)
    if input_data.shape[1] != 2560:
        # Check if it has metadata columns (subject_id, labels, etc.)
        if input_data.shape[1] == 2567:
            # Extract ECG signal only (skip first 7 columns)
            if args.verbose >= 1:
                print("  Detected metadata columns, extracting ECG signal (columns 7-2567)")
            input_data = input_data[:, 6:]  # Note: 6: in 0-indexed
        elif input_data.shape[1] > 2560:
            raise ValueError(
                f"Invalid input shape: {input_data.shape}\n"
                f"Expected (n_samples, 2560) or (n_samples, 2567) with metadata,\n"
                f"but got {input_data.shape[1]} columns"
            )
        else:
            raise ValueError(
                f"Invalid ECG window size: {input_data.shape[1]}\n"
                f"Expected 2560 samples (10 seconds at 256 Hz)"
            )

    input_data = input_data.reshape(-1, 2560, 1)
elif len(input_data.shape) == 3:
    # Shape: (n_samples, 2560, 1) - already correct
    if input_data.shape[1] != 2560 or input_data.shape[2] != 1:
        raise ValueError(
            f"Invalid input shape: {input_data.shape}\n"
            f"Expected (n_samples, 2560, 1)"
        )
else:
    raise ValueError(
        f"Invalid input shape: {input_data.shape}\n"
        f"Expected (n_samples, 2560) or (n_samples, 2560, 1)"
    )

n_samples = input_data.shape[0]
if args.verbose >= 1:
    print(f"Preprocessed data shape: {input_data.shape}")
    print(f"Number of samples: {n_samples}")

# Extract features
if args.verbose >= 1:
    print(f"\nExtracting features (batch_size={args.batch_size})...")

features_list = []
n_batches = int(np.ceil(n_samples / args.batch_size))

if args.verbose >= 1:
    pbar = tqdm(total=n_batches, desc="Processing batches")

for i in range(0, n_samples, args.batch_size):
    batch_end = min(i + args.batch_size, n_samples)
    batch_data = input_data[i:batch_end]

    # Run inference
    outputs = model.predict(batch_data, verbose=0)

    # Extract features
    if isinstance(outputs, dict):
        batch_features = outputs['features']
    else:
        # If model returns tuple/list, features should be first element
        batch_features = outputs[0] if isinstance(outputs, (list, tuple)) else outputs

    features_list.append(batch_features)

    if args.verbose >= 1:
        pbar.update(1)

if args.verbose >= 1:
    pbar.close()

features = np.vstack(features_list)

if args.verbose >= 1:
    print(f"✓ Features extracted: {features.shape}")

# Normalize features if scaler is available
if args.normalize and scaler is not None:
    if args.verbose >= 1:
        print("Normalizing features...")
    features = scaler.transform(features)
    if args.verbose >= 1:
        print(f"✓ Features normalized")

# Save or process features
if args.task is None:
    # Just save features
    if args.verbose >= 1:
        print(f"\nSaving features to: {args.output_file}")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)

    np.save(args.output_file, features)

    if args.verbose >= 1:
        print(f"✓ Features saved successfully!")
        print(f"\nOutput shape: {features.shape}")
        print(f"Output file: {args.output_file}")

else:
    # Run downstream task
    if args.downstream_model is None:
        raise ValueError(
            f"--downstream_model is required when --task is specified\n"
            f"Please provide path to trained {args.task} model"
        )

    if not os.path.exists(args.downstream_model):
        raise FileNotFoundError(f"Downstream model not found: {args.downstream_model}")

    if args.verbose >= 1:
        print(f"\nLoading downstream model for {args.task}...")
        print(f"Model path: {args.downstream_model}")

    try:
        downstream_model = tf.keras.models.load_model(args.downstream_model)
        if args.verbose >= 1:
            print(f"✓ Downstream model loaded successfully")
    except Exception as e:
        raise RuntimeError(f"Failed to load downstream model: {e}")

    if args.verbose >= 1:
        print(f"\nRunning inference for {args.task}...")

    predictions = downstream_model.predict(features, batch_size=args.batch_size, verbose=0)

    if args.verbose >= 1:
        print(f"✓ Predictions generated: {predictions.shape}")

    # Create output DataFrame
    if args.task == 'stress':
        # Binary classification - convert to class labels
        pred_classes = np.argmax(predictions, axis=1)
        pred_probs = predictions[:, 1]  # Probability of class 1 (stressed)

        df = pd.DataFrame({
            'sample_id': np.arange(n_samples),
            'predicted_class': pred_classes,
            'stress_probability': pred_probs,
            'low_stress_probability': predictions[:, 0],
            'high_stress_probability': predictions[:, 1]
        })
    else:
        # Regression tasks (pss, pdq, fsi, cortisol)
        df = pd.DataFrame({
            'sample_id': np.arange(n_samples),
            f'predicted_{args.task}': predictions.flatten()
        })

    # Save predictions
    if args.verbose >= 1:
        print(f"\nSaving predictions to: {args.output_file}")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)

    df.to_csv(args.output_file, index=False)

    if args.verbose >= 1:
        print(f"✓ Predictions saved successfully!")
        print(f"\nOutput:")
        print(df.head())
        print(f"\nFull results: {args.output_file}")

if args.verbose >= 1:
    print(f"\n{'='*70}")
    print(f"Inference Complete!")
    print(f"{'='*70}\n")

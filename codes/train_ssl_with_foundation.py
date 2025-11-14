#!/usr/bin/env python3
"""
train_ssl_with_foundation.py - Train SSL-ECG with foundation model embeddings

This script implements the CORRECT SSL-ECG architecture with foundation model:

    Raw ECG (2560) → Apply transformations (7 tasks)
                           ↓
                  Foundation Model (frozen) → 512-dim embeddings
                           ↓
                  SSL-ECG Encoder (trainable) → Multi-task learning
                           ↓
                  Learned features (256-dim) → Downstream tasks

This provides the best of both worlds:
- Foundation model's powerful pre-trained representations (512-dim)
- SSL-ECG's transformation-based representation learning (256-dim)

Training time: ~2 hours for SSL + ~30 min for downstream (vs 30-40 hours from scratch)
Expected performance: Stress AUC 0.70-0.74 (best performance)

Usage:
    # Single fold
    python codes/train_ssl_with_foundation.py \
        --data_folder ~/maternal_ecg_data \
        --kfold 0

    # All 5 folds
    for fold in {0..4}; do
        python codes/train_ssl_with_foundation.py \
            --data_folder ~/maternal_ecg_data \
            --kfold $fold
    done

Author: Claude (2025-11-14)
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import pickle
import json
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

# ML libraries
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score
)

# Add codes directory to path
sys.path.append(os.path.dirname(__file__))

# Import local modules
import datasets
import foundation_microservice_client as fmc
import ssl_data_generator_with_foundation as ssl_gen
import model_ssl_with_foundation as ssl_model


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train SSL-ECG with foundation model embeddings'
    )

    # Data arguments
    parser.add_argument('--data_folder', type=str, required=True,
                        help='Path to maternal ECG data folder')
    parser.add_argument('--kfold', type=int, default=0,
                        help='Which fold to train (0-4)')
    parser.add_argument('--total_fold', type=int, default=5,
                        help='Total number of folds')

    # Microservice arguments
    parser.add_argument('--api_url', type=str, default=None,
                        help='Microservice URL (default: production Cloud Run)')
    parser.add_argument('--use_auth', type=bool, default=True,
                        help='Use GCP authentication')

    # SSL training arguments
    parser.add_argument('--ssl_epochs', type=int, default=30,
                        help='Number of SSL pre-training epochs')
    parser.add_argument('--ssl_batch_size', type=int, default=128,
                        help='Batch size for SSL training')
    parser.add_argument('--ssl_lr', type=float, default=0.001,
                        help='Learning rate for SSL training')
    parser.add_argument('--dropout_rate', type=float, default=0.5,
                        help='Dropout rate')
    parser.add_argument('--l2_reg', type=float, default=0.0001,
                        help='L2 regularization coefficient')

    # Feature extraction arguments
    parser.add_argument('--extract_batch_size', type=int, default=100,
                        help='Batch size for foundation embedding extraction')

    # Output arguments
    parser.add_argument('--output_dir', type=str, default='trained_models_ssl_foundation',
                        help='Output directory for trained models')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Verbosity level')

    return parser.parse_args()


def train_ssl_model(
    train_ecg,
    generator,
    args,
    output_dir
):
    """
    Train SSL-ECG model on foundation embeddings.

    Args:
        train_ecg: Training ECG windows, shape (n_samples, 2560)
        generator: SSL data generator
        args: Command line arguments
        output_dir: Output directory

    Returns:
        model: Trained SSL model
    """
    if args.verbose:
        print("\n" + "="*70)
        print("Step 1: SSL Pre-training")
        print("="*70)
        print(f"Training on {len(train_ecg)} ECG windows")
        print(f"This will generate {len(train_ecg) * 7} training samples")
        print()

    # Generate SSL training dataset
    print("Generating SSL training dataset...")
    print("(This extracts foundation embeddings for all transformations)")
    train_embeddings, train_labels = generator.generate_ssl_dataset(
        ecg_data=train_ecg,
        extract_batch_size=args.extract_batch_size,
        shuffle=True
    )

    print(f"✓ SSL dataset generated:")
    print(f"  Embeddings: {train_embeddings.shape}")
    print(f"  Total samples: {len(train_embeddings)}")
    print()

    # Build SSL model
    print("Building SSL-ECG encoder model...")
    model = ssl_model.build_ssl_encoder_for_embeddings(
        input_dim=512,
        hidden_dims=[256, 256, 128],
        output_dim=256,
        dropout_rate=args.dropout_rate,
        l2_reg=args.l2_reg
    )

    # Compile model
    model = ssl_model.compile_ssl_model(
        model,
        learning_rate=args.ssl_lr
    )

    if args.verbose:
        print("\nModel summary:")
        model.summary()
        print()

    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='loss',
            factor=0.5,
            patience=3,
            verbose=1,
            min_lr=1e-6
        ),
        keras.callbacks.ModelCheckpoint(
            os.path.join(output_dir, 'ssl_model_best.h5'),
            monitor='loss',
            save_best_only=True,
            verbose=1
        )
    ]

    # Train SSL model
    print(f"Training SSL model for {args.ssl_epochs} epochs...")
    print("(Learning to predict transformations from foundation embeddings)\n")

    history = model.fit(
        train_embeddings,
        train_labels,
        batch_size=args.ssl_batch_size,
        epochs=args.ssl_epochs,
        callbacks=callbacks,
        verbose=1,
        validation_split=0.1
    )

    # Save training history
    history_path = os.path.join(output_dir, 'ssl_training_history.json')
    with open(history_path, 'w') as f:
        # Convert history to JSON-serializable format
        history_dict = {
            key: [float(val) for val in values]
            for key, values in history.history.items()
        }
        json.dump(history_dict, f, indent=2)

    print(f"\n✓ SSL training complete!")
    print(f"✓ Model saved to: {output_dir}/ssl_model_best.h5")
    print(f"✓ History saved to: {history_path}")

    return model


def extract_learned_features(
    model,
    ecg_data,
    foundation_client,
    args
):
    """
    Extract learned features from trained SSL model.

    Args:
        model: Trained SSL model
        ecg_data: ECG windows, shape (n_samples, 2560)
        foundation_client: Foundation microservice client
        args: Command line arguments

    Returns:
        features: Learned features, shape (n_samples, 256)
    """
    if args.verbose:
        print("\n" + "="*70)
        print("Step 2: Feature Extraction")
        print("="*70)
        print(f"Extracting learned features from {len(ecg_data)} ECG windows")
        print()

    # Extract foundation embeddings (no transformations, just original)
    print("Extracting foundation embeddings...")
    foundation_embeddings = foundation_client.extract_features(
        ecg_data=ecg_data,
        sampling_rate=256.0,
        verbose=args.verbose
    )

    # Extract learned features from SSL model
    print("Extracting SSL-learned features...")
    learned_features = ssl_model.extract_ssl_features(
        model=model,
        embeddings=foundation_embeddings,
        batch_size=256
    )

    if args.verbose:
        print(f"✓ Feature extraction complete:")
        print(f"  Foundation embeddings: {foundation_embeddings.shape}")
        print(f"  SSL-learned features: {learned_features.shape}")
        print()

    return learned_features


def train_downstream_models(
    train_features,
    test_features,
    train_labels,
    test_labels,
    verbose=1
):
    """
    Train downstream models for maternal stress prediction.

    Args:
        train_features: Training features, shape (n_samples, 256)
        test_features: Test features
        train_labels: Dictionary of training labels
        test_labels: Dictionary of test labels
        verbose: Verbosity level

    Returns:
        results: Dictionary of trained models and metrics
    """
    results = {}

    if verbose:
        print("\n" + "="*70)
        print("Step 3: Downstream Model Training")
        print("="*70)

    # Normalize features
    scaler = StandardScaler()
    train_features_norm = scaler.fit_transform(train_features)
    test_features_norm = scaler.transform(test_features)

    results['scaler'] = scaler

    # 1. Stress Classification
    if 'stress' in train_labels and train_labels['stress'] is not None:
        if verbose:
            print("\n1. Training Stress Classifier...")

        stress_clf = LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight='balanced'
        )
        stress_clf.fit(train_features_norm, train_labels['stress'])

        # Predictions
        stress_pred = stress_clf.predict(test_features_norm)
        stress_proba = stress_clf.predict_proba(test_features_norm)[:, 1]

        # Metrics
        stress_metrics = {
            'accuracy': accuracy_score(test_labels['stress'], stress_pred),
            'precision': precision_score(test_labels['stress'], stress_pred, zero_division=0),
            'recall': recall_score(test_labels['stress'], stress_pred, zero_division=0),
            'f1': f1_score(test_labels['stress'], stress_pred, zero_division=0),
            'auc': roc_auc_score(test_labels['stress'], stress_proba)
        }

        results['stress'] = {
            'model': stress_clf,
            'metrics': stress_metrics,
            'predictions': stress_pred,
            'probabilities': stress_proba
        }

        if verbose:
            print(f"  Accuracy: {stress_metrics['accuracy']:.4f}")
            print(f"  AUC: {stress_metrics['auc']:.4f}")
            print(f"  F1: {stress_metrics['f1']:.4f}")

    # 2. PSS Score Regression
    if 'PSS' in train_labels and train_labels['PSS'] is not None:
        if verbose:
            print("\n2. Training PSS Score Regressor...")

        pss_reg = Ridge(alpha=1.0, random_state=42)
        pss_reg.fit(train_features_norm, train_labels['PSS'])

        pss_pred = pss_reg.predict(test_features_norm)
        pss_metrics = {
            'mse': mean_squared_error(test_labels['PSS'], pss_pred),
            'rmse': np.sqrt(mean_squared_error(test_labels['PSS'], pss_pred)),
            'mae': mean_absolute_error(test_labels['PSS'], pss_pred),
            'r2': r2_score(test_labels['PSS'], pss_pred)
        }

        results['PSS'] = {
            'model': pss_reg,
            'metrics': pss_metrics,
            'predictions': pss_pred
        }

        if verbose:
            print(f"  R²: {pss_metrics['r2']:.4f}")
            print(f"  RMSE: {pss_metrics['rmse']:.4f}")

    # 3-5. Other regressors (PDQ, FSI, cortisol)
    for task_name, task_label in [('PDQ', 'PDQ'), ('FSI', 'FSI'), ('cortisol', 'cortisol')]:
        if task_label in train_labels and train_labels[task_label] is not None:
            if verbose:
                print(f"\nTraining {task_name} Regressor...")

            reg = Ridge(alpha=1.0, random_state=42)
            reg.fit(train_features_norm, train_labels[task_label])

            pred = reg.predict(test_features_norm)
            metrics = {
                'mse': mean_squared_error(test_labels[task_label], pred),
                'rmse': np.sqrt(mean_squared_error(test_labels[task_label], pred)),
                'mae': mean_absolute_error(test_labels[task_label], pred),
                'r2': r2_score(test_labels[task_label], pred)
            }

            results[task_name] = {
                'model': reg,
                'metrics': metrics,
                'predictions': pred
            }

            if verbose:
                print(f"  R²: {metrics['r2']:.4f}")
                print(f"  RMSE: {metrics['rmse']:.4f}")

    return results


def save_results(results, ssl_model_obj, output_dir, fold_id, verbose=1):
    """Save trained models and results."""
    os.makedirs(output_dir, exist_ok=True)

    if verbose:
        print("\n" + "="*70)
        print(f"Saving Results to: {output_dir}")
        print("="*70)

    # Save SSL model
    ssl_model_path = os.path.join(output_dir, 'ssl_model_final.h5')
    ssl_model_obj.save(ssl_model_path)
    if verbose:
        print(f"✓ Saved SSL model: {ssl_model_path}")

    # Save scaler
    scaler_path = os.path.join(output_dir, 'feature_scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(results['scaler'], f)
    if verbose:
        print(f"✓ Saved feature scaler: {scaler_path}")

    # Save downstream models
    metrics_summary = {}

    for task_name in ['stress', 'PSS', 'PDQ', 'FSI', 'cortisol']:
        if task_name in results:
            model_path = os.path.join(output_dir, f'{task_name}_model.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(results[task_name]['model'], f)

            pred_path = os.path.join(output_dir, f'{task_name}_predictions.npy')
            np.save(pred_path, results[task_name]['predictions'])

            metrics_summary[task_name] = results[task_name]['metrics']

            if verbose:
                print(f"✓ Saved {task_name} model: {model_path}")

    # Save metrics summary
    metrics_path = os.path.join(output_dir, 'results.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics_summary, f, indent=2)

    if verbose:
        print(f"✓ Saved metrics: {metrics_path}")
        print()


def main():
    """Main training function."""
    args = parse_args()

    print("="*70)
    print("SSL-ECG Training with Foundation Model Embeddings")
    print("="*70)
    print(f"Fold: {args.kfold}/{args.total_fold - 1}")
    print(f"Data folder: {args.data_folder}")
    print()

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(
        args.output_dir,
        f"ssl_foundation_fold{args.kfold}_{timestamp}"
    )
    os.makedirs(output_dir, exist_ok=True)

    # Initialize foundation microservice client
    print("Initializing ECG Foundation Microservice Client...")
    foundation_client = fmc.load_microservice_client(
        api_url=args.api_url,
        use_auth=args.use_auth
    )
    print()

    # Initialize SSL data generator
    print("Initializing SSL Data Generator...")
    generator = ssl_gen.SSLDataGeneratorWithFoundation(
        foundation_client=foundation_client,
        verbose=args.verbose
    )
    print()

    # Load data
    print("Loading maternal ECG data...")
    (train_ECG, train_stress, train_PSS, train_PDQ, train_FSI, train_cortisol,
     test_ECG, test_stress, test_PSS, test_PDQ, test_FSI, test_cortisol,
     train_subjects, test_subjects) = datasets.train_test_split_felicity_kfold(
        args.data_folder,
        kfold=args.kfold,
        total_fold=args.total_fold
    )

    print(f"✓ Loaded data:")
    print(f"  Train: {len(train_ECG)} windows from {len(set(train_subjects))} subjects")
    print(f"  Test: {len(test_ECG)} windows from {len(set(test_subjects))} subjects")

    # Verify no subject overlap
    overlap = set(train_subjects) & set(test_subjects)
    if overlap:
        raise ValueError(f"Subject overlap detected: {overlap}")
    print(f"✓ No subject overlap (critical for valid results)")

    # Train SSL model
    ssl_model_obj = train_ssl_model(
        train_ecg=train_ECG,
        generator=generator,
        args=args,
        output_dir=output_dir
    )

    # Extract learned features
    train_features = extract_learned_features(
        model=ssl_model_obj,
        ecg_data=train_ECG,
        foundation_client=foundation_client,
        args=args
    )

    test_features = extract_learned_features(
        model=ssl_model_obj,
        ecg_data=test_ECG,
        foundation_client=foundation_client,
        args=args
    )

    # Prepare labels
    train_labels = {
        'stress': train_stress,
        'PSS': train_PSS,
        'PDQ': train_PDQ,
        'FSI': train_FSI,
        'cortisol': train_cortisol
    }

    test_labels = {
        'stress': test_stress,
        'PSS': test_PSS,
        'PDQ': test_PDQ,
        'FSI': test_FSI,
        'cortisol': test_cortisol
    }

    # Train downstream models
    results = train_downstream_models(
        train_features,
        test_features,
        train_labels,
        test_labels,
        verbose=args.verbose
    )

    # Save results
    save_results(results, ssl_model_obj, output_dir, args.kfold, verbose=args.verbose)

    # Print summary
    print("="*70)
    print("Training Complete!")
    print("="*70)
    print(f"Results saved to: {output_dir}")
    print("\nPerformance Summary:")

    if 'stress' in results:
        print(f"  Stress AUC: {results['stress']['metrics']['auc']:.4f}")
    if 'PSS' in results:
        print(f"  PSS R²: {results['PSS']['metrics']['r2']:.4f}")
    if 'PDQ' in results:
        print(f"  PDQ R²: {results['PDQ']['metrics']['r2']:.4f}")
    if 'FSI' in results:
        print(f"  FSI R²: {results['FSI']['metrics']['r2']:.4f}")
    if 'cortisol' in results:
        print(f"  Cortisol R²: {results['cortisol']['metrics']['r2']:.4f}")

    print("\nArchitecture:")
    print("  Raw ECG → Transformations → Foundation (frozen) → SSL-ECG (trained)")
    print("  → Learned features → Downstream models")
    print()


if __name__ == '__main__':
    main()

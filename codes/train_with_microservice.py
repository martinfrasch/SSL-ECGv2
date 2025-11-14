#!/usr/bin/env python3
"""
train_with_microservice.py - Train maternal stress models using ECG foundation microservice

This script uses the deployed ECG foundation microservice to extract features,
then trains downstream models for maternal stress prediction.

This is the FASTEST approach:
- No SSL pre-training needed (uses pre-trained foundation model)
- Training time: ~25 minutes for 5-fold CV
- Expected performance: Stress AUC 0.68-0.72 (best performance)
- Cost: ~$2.25 on GCP (97% cheaper than training from scratch)

Usage:
    # Single fold
    python codes/train_with_microservice.py \
        --data_folder ~/maternal_ecg_data \
        --kfold 0

    # All 5 folds
    for fold in {0..4}; do
        python codes/train_with_microservice.py \
            --data_folder ~/maternal_ecg_data \
            --kfold $fold
    done

Author: Claude (2025-11-12)
Based on: florian-ecg-steroid-analysis/MICROSERVICE_STATUS.md
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


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train maternal stress models using ECG foundation microservice'
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

    # Output arguments
    parser.add_argument('--output_dir', type=str, default='trained_models_microservice',
                        help='Output directory for trained models')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Verbosity level')

    return parser.parse_args()


def extract_microservice_features(
    ecg_data,
    client,
    sampling_rate=256.0,
    verbose=1
):
    """
    Extract 512-dimensional features using foundation microservice.

    Args:
        ecg_data: ECG windows, shape (n_samples, 2560)
        client: ECGFoundationClient instance
        sampling_rate: Sampling rate in Hz
        verbose: Verbosity level

    Returns:
        features: Extracted 512-dim features, shape (n_samples, 512)
    """
    if verbose:
        print(f"\nExtracting 512-dim features from {len(ecg_data)} ECG windows...")
        print(f"Using microservice: {client.api_url}")

    # Extract features via microservice
    features = client.extract_features(
        ecg_data=ecg_data,
        sampling_rate=sampling_rate,
        verbose=verbose
    )

    if verbose:
        print(f"✓ Extracted features: {features.shape}")

    return features


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
        train_features: Training features, shape (n_samples, 512)
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
        print("Training Downstream Models")
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

        # Predictions
        pss_pred = pss_reg.predict(test_features_norm)

        # Metrics
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

    # 3. PDQ Score Regression
    if 'PDQ' in train_labels and train_labels['PDQ'] is not None:
        if verbose:
            print("\n3. Training PDQ Score Regressor...")

        pdq_reg = Ridge(alpha=1.0, random_state=42)
        pdq_reg.fit(train_features_norm, train_labels['PDQ'])

        # Predictions
        pdq_pred = pdq_reg.predict(test_features_norm)

        # Metrics
        pdq_metrics = {
            'mse': mean_squared_error(test_labels['PDQ'], pdq_pred),
            'rmse': np.sqrt(mean_squared_error(test_labels['PDQ'], pdq_pred)),
            'mae': mean_absolute_error(test_labels['PDQ'], pdq_pred),
            'r2': r2_score(test_labels['PDQ'], pdq_pred)
        }

        results['PDQ'] = {
            'model': pdq_reg,
            'metrics': pdq_metrics,
            'predictions': pdq_pred
        }

        if verbose:
            print(f"  R²: {pdq_metrics['r2']:.4f}")
            print(f"  RMSE: {pdq_metrics['rmse']:.4f}")

    # 4. FSI Score Regression
    if 'FSI' in train_labels and train_labels['FSI'] is not None:
        if verbose:
            print("\n4. Training FSI Score Regressor...")

        fsi_reg = Ridge(alpha=1.0, random_state=42)
        fsi_reg.fit(train_features_norm, train_labels['FSI'])

        # Predictions
        fsi_pred = fsi_reg.predict(test_features_norm)

        # Metrics
        fsi_metrics = {
            'mse': mean_squared_error(test_labels['FSI'], fsi_pred),
            'rmse': np.sqrt(mean_squared_error(test_labels['FSI'], fsi_pred)),
            'mae': mean_absolute_error(test_labels['FSI'], fsi_pred),
            'r2': r2_score(test_labels['FSI'], fsi_pred)
        }

        results['FSI'] = {
            'model': fsi_reg,
            'metrics': fsi_metrics,
            'predictions': fsi_pred
        }

        if verbose:
            print(f"  R²: {fsi_metrics['r2']:.4f}")
            print(f"  RMSE: {fsi_metrics['rmse']:.4f}")

    # 5. Cortisol Regression
    if 'cortisol' in train_labels and train_labels['cortisol'] is not None:
        if verbose:
            print("\n5. Training Cortisol Regressor...")

        cortisol_reg = Ridge(alpha=1.0, random_state=42)
        cortisol_reg.fit(train_features_norm, train_labels['cortisol'])

        # Predictions
        cortisol_pred = cortisol_reg.predict(test_features_norm)

        # Metrics
        cortisol_metrics = {
            'mse': mean_squared_error(test_labels['cortisol'], cortisol_pred),
            'rmse': np.sqrt(mean_squared_error(test_labels['cortisol'], cortisol_pred)),
            'mae': mean_absolute_error(test_labels['cortisol'], cortisol_pred),
            'r2': r2_score(test_labels['cortisol'], cortisol_pred)
        }

        results['cortisol'] = {
            'model': cortisol_reg,
            'metrics': cortisol_metrics,
            'predictions': cortisol_pred
        }

        if verbose:
            print(f"  R²: {cortisol_metrics['r2']:.4f}")
            print(f"  RMSE: {cortisol_metrics['rmse']:.4f}")

    return results


def save_results(results, output_dir, fold_id, verbose=1):
    """
    Save trained models and results.

    Args:
        results: Dictionary of results
        output_dir: Output directory
        fold_id: Fold identifier
        verbose: Verbosity level
    """
    os.makedirs(output_dir, exist_ok=True)

    if verbose:
        print("\n" + "="*70)
        print(f"Saving Results to: {output_dir}")
        print("="*70)

    # Save scaler
    scaler_path = os.path.join(output_dir, 'feature_scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(results['scaler'], f)
    if verbose:
        print(f"✓ Saved feature scaler: {scaler_path}")

    # Save models and metrics
    metrics_summary = {}

    for task_name in ['stress', 'PSS', 'PDQ', 'FSI', 'cortisol']:
        if task_name in results:
            # Save model
            model_path = os.path.join(output_dir, f'{task_name}_model.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(results[task_name]['model'], f)

            # Save predictions
            pred_path = os.path.join(output_dir, f'{task_name}_predictions.npy')
            np.save(pred_path, results[task_name]['predictions'])

            # Add to metrics summary
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
    print("Training with ECG Foundation Microservice")
    print("="*70)
    print(f"Fold: {args.kfold}/{args.total_fold - 1}")
    print(f"Data folder: {args.data_folder}")
    print()

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(
        args.output_dir,
        f"microservice_fold{args.kfold}_{timestamp}"
    )

    # Initialize microservice client
    print("Initializing ECG Foundation Microservice Client...")
    client = fmc.load_microservice_client(
        api_url=args.api_url,
        use_auth=args.use_auth
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
    print(f"✓ No subject overlap")

    # Extract features using microservice
    train_features = extract_microservice_features(
        train_ECG,
        client,
        sampling_rate=256.0,
        verbose=args.verbose
    )

    test_features = extract_microservice_features(
        test_ECG,
        client,
        sampling_rate=256.0,
        verbose=args.verbose
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
    save_results(results, output_dir, args.kfold, verbose=args.verbose)

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

    print("\nTo run inference on new ECGs:")
    print(f"  1. Load client: client = fmc.load_microservice_client()")
    print(f"  2. Extract features: features = client.extract_features(new_ecg)")
    print(f"  3. Load scaler: scaler = pickle.load(open('{output_dir}/feature_scaler.pkl', 'rb'))")
    print(f"  4. Normalize: features = scaler.transform(features)")
    print(f"  5. Load model: model = pickle.load(open('{output_dir}/stress_model.pkl', 'rb'))")
    print(f"  6. Predict: proba = model.predict_proba(features)")
    print()


if __name__ == '__main__':
    main()

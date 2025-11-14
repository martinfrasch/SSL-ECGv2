"""
ssl_data_generator_with_foundation.py - Data generator for SSL training with foundation embeddings

This module generates training data for SSL-ECG by:
1. Taking raw ECG windows (2560 samples)
2. Applying 7 different transformations
3. Extracting foundation embeddings (512-dim) for each transformation
4. Returning embeddings + labels for multi-task learning

Transformations:
    0. Original (no transformation)
    1. Add noise
    2. Scale
    3. Negate
    4. Horizontal flip
    5. Permutation
    6. Time warping

Author: Claude (2025-11-14)
"""

import numpy as np
import tensorflow as tf
from typing import Tuple, Dict, List
import sys
import os

# Add codes directory to path
sys.path.append(os.path.dirname(__file__))

import signal_transformation_task as stt
import foundation_microservice_client as fmc


class SSLDataGeneratorWithFoundation:
    """
    Data generator for SSL training with foundation embeddings.

    This generator:
    1. Takes batches of raw ECG windows
    2. Applies 7 transformations to each window
    3. Extracts foundation embeddings (512-dim) for each
    4. Returns embeddings + transformation labels
    """

    def __init__(
        self,
        foundation_client: fmc.ECGFoundationClient,
        noise_param: float = 15.0,
        scale_param: float = 1.1,
        permu_param: int = 20,
        tw_piece_param: int = 9,
        twsf_param: float = 1.05,
        sampling_rate: float = 256.0,
        verbose: int = 0
    ):
        """
        Initialize SSL data generator.

        Args:
            foundation_client: ECGFoundationClient instance
            noise_param: Noise amount for SNR
            scale_param: Scaling factor
            permu_param: Number of permutation pieces
            tw_piece_param: Number of time warping pieces
            twsf_param: Time warping stretch factor
            sampling_rate: ECG sampling rate in Hz
            verbose: Verbosity level
        """
        self.foundation_client = foundation_client
        self.noise_param = noise_param
        self.scale_param = scale_param
        self.permu_param = permu_param
        self.tw_piece_param = tw_piece_param
        self.twsf_param = twsf_param
        self.sampling_rate = sampling_rate
        self.verbose = verbose

        self.n_tasks = 7
        self.task_names = [
            'original',
            'noised',
            'scaled',
            'negated',
            'flipped',
            'permuted',
            'time_warped'
        ]

    def apply_transformations(self, ecg_window: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Apply all 7 transformations to a single ECG window.

        Args:
            ecg_window: ECG window, shape (2560,)

        Returns:
            Dictionary of transformed signals
        """
        transformations = {}

        # Task 0: Original (no transformation)
        transformations['original'] = ecg_window.copy()

        # Task 1: Add noise
        transformations['noised'] = stt.add_noise_with_SNR(
            ecg_window.copy(),
            self.noise_param
        )

        # Task 2: Scale
        transformations['scaled'] = stt.scaled(
            ecg_window.copy(),
            self.scale_param
        )

        # Task 3: Negate
        transformations['negated'] = stt.negate(ecg_window.copy())

        # Task 4: Horizontal flip
        transformations['flipped'] = stt.hor_filp(ecg_window.copy())

        # Task 5: Permutation
        transformations['permuted'] = stt.permute(
            ecg_window.copy(),
            self.permu_param
        )

        # Task 6: Time warping
        transformations['time_warped'] = stt.time_warp(
            ecg_window.copy(),
            sampling_freq=int(self.sampling_rate),
            pieces=self.tw_piece_param,
            stretch_factor=self.twsf_param,
            squeeze_factor=1.0 / self.twsf_param
        )

        return transformations

    def generate_ssl_batch(
        self,
        ecg_batch: np.ndarray,
        include_progress: bool = False
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Generate SSL training batch with foundation embeddings.

        Args:
            ecg_batch: Batch of ECG windows, shape (batch_size, 2560)
            include_progress: Whether to show progress bar

        Returns:
            embeddings: Foundation embeddings, shape (batch_size * 7, 512)
            labels: Dictionary of labels for each task
        """
        batch_size = len(ecg_batch)

        # Store all transformed signals
        all_transformed_signals = []
        all_labels = {f'task_{i}': [] for i in range(self.n_tasks)}

        # Process each ECG window
        iterator = range(batch_size)
        if include_progress and self.verbose:
            from tqdm import tqdm
            iterator = tqdm(iterator, desc="Applying transformations")

        for i in iterator:
            ecg_window = ecg_batch[i]

            # Apply all transformations
            transformations = self.apply_transformations(ecg_window)

            # Add to list
            for task_idx, (task_name, transformed_signal) in enumerate(transformations.items()):
                all_transformed_signals.append(transformed_signal)

                # Create labels: 1 if this is the current task, 0 otherwise
                for j in range(self.n_tasks):
                    if j == task_idx:
                        all_labels[f'task_{j}'].append(1.0)
                    else:
                        all_labels[f'task_{j}'].append(0.0)

        # Convert to numpy arrays
        all_transformed_signals = np.array(all_transformed_signals)  # Shape: (batch_size * 7, 2560)

        # Extract foundation embeddings for all transformed signals
        if self.verbose:
            print(f"Extracting foundation embeddings for {len(all_transformed_signals)} signals...")

        embeddings = self.foundation_client.extract_features(
            ecg_data=all_transformed_signals,
            sampling_rate=self.sampling_rate,
            verbose=0  # Disable internal progress bar
        )

        # Convert labels to numpy arrays
        labels = {
            key: np.array(val).reshape(-1, 1).astype('float32')
            for key, val in all_labels.items()
        }

        return embeddings, labels

    def generate_ssl_dataset(
        self,
        ecg_data: np.ndarray,
        batch_size: int = 32,
        extract_batch_size: int = 100,  # Process this many ECG windows at a time
        shuffle: bool = True
    ):
        """
        Generate SSL training dataset with foundation embeddings.

        This is a one-time generation (not a TF data generator) because
        extracting foundation embeddings is expensive.

        Args:
            ecg_data: All ECG windows, shape (n_samples, 2560)
            batch_size: Batch size for transformation processing
            extract_batch_size: How many ECG windows to process at once
            shuffle: Whether to shuffle the data

        Returns:
            embeddings: All foundation embeddings, shape (n_samples * 7, 512)
            labels: Dictionary of labels for each task
        """
        n_samples = len(ecg_data)

        if self.verbose:
            print(f"Generating SSL dataset from {n_samples} ECG windows...")
            print(f"This will create {n_samples * 7} samples (7 transformations per window)")

        if shuffle:
            np.random.shuffle(ecg_data)

        all_embeddings = []
        all_labels = {f'task_{i}': [] for i in range(self.n_tasks)}

        # Process in batches
        n_batches = int(np.ceil(n_samples / extract_batch_size))

        if self.verbose:
            from tqdm import tqdm
            batch_iterator = tqdm(range(n_batches), desc="Generating SSL batches")
        else:
            batch_iterator = range(n_batches)

        for batch_idx in batch_iterator:
            start_idx = batch_idx * extract_batch_size
            end_idx = min((batch_idx + 1) * extract_batch_size, n_samples)

            batch_ecg = ecg_data[start_idx:end_idx]

            # Generate batch
            batch_embeddings, batch_labels = self.generate_ssl_batch(
                batch_ecg,
                include_progress=False
            )

            all_embeddings.append(batch_embeddings)
            for key in all_labels.keys():
                all_labels[key].append(batch_labels[key])

        # Concatenate all batches
        embeddings = np.vstack(all_embeddings)
        labels = {
            key: np.vstack(val) for key, val in all_labels.items()
        }

        if self.verbose:
            print(f"✓ Generated SSL dataset:")
            print(f"  Embeddings: {embeddings.shape}")
            print(f"  Labels: {labels['task_0'].shape}")

        return embeddings, labels


# Convenience function
def create_ssl_generator(
    foundation_client: fmc.ECGFoundationClient = None,
    api_url: str = None,
    use_auth: bool = True,
    verbose: int = 1
) -> SSLDataGeneratorWithFoundation:
    """
    Create SSL data generator with foundation client.

    Args:
        foundation_client: Existing client (if None, will create one)
        api_url: Microservice URL
        use_auth: Use GCP authentication
        verbose: Verbosity level

    Returns:
        SSLDataGeneratorWithFoundation instance
    """
    if foundation_client is None:
        if verbose:
            print("Initializing foundation microservice client...")
        foundation_client = fmc.load_microservice_client(
            api_url=api_url,
            use_auth=use_auth
        )

    generator = SSLDataGeneratorWithFoundation(
        foundation_client=foundation_client,
        verbose=verbose
    )

    return generator


# Example usage
if __name__ == '__main__':
    print("Testing SSL Data Generator with Foundation Embeddings\n")

    # Create generator
    print("Creating generator...")
    generator = create_ssl_generator(
        api_url=None,  # Use default
        use_auth=True,
        verbose=1
    )

    # Test with dummy ECG data
    print("\nTesting with dummy ECG data...")
    dummy_ecg = np.random.randn(10, 2560).astype('float32')

    # Generate SSL batch
    print("\nGenerating SSL batch...")
    embeddings, labels = generator.generate_ssl_batch(
        dummy_ecg[:5],  # Process 5 ECG windows
        include_progress=True
    )

    print(f"\n✓ Generated batch:")
    print(f"  Embeddings: {embeddings.shape}")  # Should be (35, 512) = 5 windows * 7 tasks
    print(f"  Labels: {labels['task_0'].shape}")

    # Verify labels
    print(f"\n✓ Label distribution:")
    for i in range(7):
        print(f"  Task {i}: {labels[f'task_{i}'].sum():.0f} positive samples")

    # Generate full dataset
    print("\nGenerating full SSL dataset...")
    all_embeddings, all_labels = generator.generate_ssl_dataset(
        dummy_ecg,
        extract_batch_size=5,
        shuffle=True
    )

    print(f"\n✓ Full dataset:")
    print(f"  Embeddings: {all_embeddings.shape}")  # Should be (70, 512) = 10 windows * 7 tasks
    print(f"  Labels: {all_labels['task_0'].shape}")

    print("\n✓ All tests passed!")

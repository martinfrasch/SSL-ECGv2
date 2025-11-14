"""
foundation_model_loader.py - Load and adapt ECG foundation models

This module provides utilities to load pre-trained ECG foundation models
from various sources (Hugging Face, TensorFlow Hub, PhysioNet) and adapt
them for use with the SSL-ECG architecture.

Supported foundation models:
- ECG-FM (Hugging Face) - 1.66M ECGs
- ResNet-ECG (TensorFlow Hub) - Clinical ECG foundation
- PhysioNet SSL (Custom) - Multi-task learning

Author: Claude (2025-11-12)
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
import requests
import zipfile
from tqdm import tqdm


class ECGFoundationModel:
    """
    Wrapper for ECG foundation models from various sources.

    Usage:
        # Load foundation model
        foundation = ECGFoundationModel.from_pretrained('ecg-fm')

        # Extract features
        features = foundation.extract_features(ecg_data)

        # Or get encoder for fine-tuning
        encoder = foundation.get_encoder()
    """

    def __init__(self, model, model_type='tfhub', input_shape=(2560, 1)):
        """
        Initialize foundation model wrapper.

        Args:
            model: Loaded model (TF model, PyTorch model, etc.)
            model_type: Type of model ('tfhub', 'huggingface', 'custom')
            input_shape: Expected input shape (for ECG: (2560, 1))
        """
        self.model = model
        self.model_type = model_type
        self.input_shape = input_shape

    @classmethod
    def from_pretrained(cls, model_name, cache_dir='./foundation_models'):
        """
        Load a pre-trained ECG foundation model.

        Args:
            model_name: Name of foundation model
                - 'ecg-fm': ECG Foundation Model (Hugging Face)
                - 'resnet-ecg': ResNet-based ECG model (TFHub)
                - 'physionet-ssl': PhysioNet self-supervised model
                - 'ptbxl-resnet': PTB-XL pre-trained ResNet
            cache_dir: Directory to cache downloaded models

        Returns:
            ECGFoundationModel instance
        """
        os.makedirs(cache_dir, exist_ok=True)

        if model_name == 'ecg-fm':
            return cls._load_ecg_fm(cache_dir)
        elif model_name == 'resnet-ecg':
            return cls._load_resnet_ecg(cache_dir)
        elif model_name == 'physionet-ssl':
            return cls._load_physionet_ssl(cache_dir)
        elif model_name == 'ptbxl-resnet':
            return cls._load_ptbxl_resnet(cache_dir)
        else:
            raise ValueError(f"Unknown model: {model_name}")

    @classmethod
    def _load_ecg_fm(cls, cache_dir):
        """
        Load ECG Foundation Model from Hugging Face.

        This is a placeholder - replace with actual model loading when available.
        """
        print("Loading ECG Foundation Model from Hugging Face...")

        model_path = os.path.join(cache_dir, 'ecg-fm')

        if not os.path.exists(model_path):
            print("Downloading ECG-FM model...")
            print("Note: This is a placeholder. Replace with actual download URL.")

            # Placeholder - in reality, would download from Hugging Face
            # from transformers import AutoModel
            # model = AutoModel.from_pretrained("cardio/ecg-fm")

            # For now, create a stub
            raise NotImplementedError(
                "ECG-FM download not yet implemented.\n"
                "Please manually download from Hugging Face and place in foundation_models/ecg-fm/\n"
                "Or use 'ptbxl-resnet' which can be trained with the existing pretrain_on_public_data.py"
            )

        # Load the model
        model = tf.keras.models.load_model(model_path)
        return cls(model, model_type='huggingface')

    @classmethod
    def _load_resnet_ecg(cls, cache_dir):
        """
        Load ResNet-based ECG model from TensorFlow Hub.
        """
        print("Loading ResNet-ECG from TensorFlow Hub...")

        # Placeholder URL - replace with actual TFHub model
        hub_url = "https://tfhub.dev/google/ecg-resnet/1"

        try:
            import tensorflow_hub as hub
            model = hub.KerasLayer(hub_url, trainable=True)
            return cls(model, model_type='tfhub')
        except Exception as e:
            raise NotImplementedError(
                f"Failed to load from TensorFlow Hub: {e}\n"
                "Use 'ptbxl-resnet' to train your own foundation model."
            )

    @classmethod
    def _load_physionet_ssl(cls, cache_dir):
        """
        Load PhysioNet SSL model (custom implementation).
        """
        print("Loading PhysioNet SSL model...")

        model_path = os.path.join(cache_dir, 'physionet-ssl')

        if not os.path.exists(model_path):
            print("\nPhysioNet SSL model not found.")
            print("You can create one by pre-training on public data:")
            print("  python codes/pretrain_on_public_data.py \\")
            print("      --dataset ptbxl \\")
            print("      --output_dir foundation_models/physionet-ssl")
            raise FileNotFoundError(f"Model not found: {model_path}")

        model = tf.keras.models.load_model(model_path)
        return cls(model, model_type='custom')

    @classmethod
    def _load_ptbxl_resnet(cls, cache_dir):
        """
        Load or train a ResNet model on PTB-XL.

        This uses the existing pretrain_on_public_data.py script.
        """
        print("Loading PTB-XL pre-trained ResNet...")

        # Check if already exists
        model_dirs = list(Path(cache_dir).glob('ptbxl_pretrained_*'))

        if not model_dirs:
            print("\nPTB-XL pre-trained model not found.")
            print("Please pre-train first:")
            print("  python codes/pretrain_on_public_data.py \\")
            print("      --dataset ptbxl \\")
            print("      --data_path ~/datasets/ptbxl \\")
            print("      --output_dir foundation_models \\")
            print("      --epochs 50")
            raise FileNotFoundError("PTB-XL pre-trained model not found")

        # Use most recent
        model_dir = sorted(model_dirs)[-1]
        model_path = os.path.join(model_dir, 'best_model')

        print(f"Loading from: {model_path}")
        model = tf.keras.models.load_model(model_path)
        return cls(model, model_type='custom')

    def extract_features(self, ecg_data, batch_size=256, verbose=1):
        """
        Extract features from ECG data using foundation model.

        Args:
            ecg_data: ECG signals, shape (n_samples, 2560) or (n_samples, 2560, 1)
            batch_size: Batch size for processing
            verbose: Verbosity level

        Returns:
            features: Extracted features, shape (n_samples, feature_dim)
        """
        # Ensure correct shape
        if len(ecg_data.shape) == 2:
            ecg_data = ecg_data.reshape(-1, 2560, 1)

        if verbose:
            print(f"Extracting features from {len(ecg_data)} ECG samples...")

        # Extract features in batches
        features_list = []

        if verbose:
            pbar = tqdm(range(0, len(ecg_data), batch_size), desc="Processing batches")
        else:
            pbar = range(0, len(ecg_data), batch_size)

        for i in pbar:
            batch = ecg_data[i:i+batch_size]

            # Forward pass
            if self.model_type == 'tfhub':
                batch_features = self.model(batch)
            else:
                # For custom models, extract features before task heads
                outputs = self.model(batch, training=False)
                if isinstance(outputs, dict):
                    batch_features = outputs['features']
                else:
                    batch_features = outputs

            features_list.append(batch_features.numpy())

        features = np.vstack(features_list)

        if verbose:
            print(f"✓ Extracted features: {features.shape}")

        return features

    def get_encoder(self):
        """
        Get the encoder part of the foundation model for fine-tuning.

        Returns:
            encoder: Keras model representing the feature extractor
        """
        if self.model_type == 'custom':
            # For our SSL models, create encoder up to GAP layer
            # Extract layers before task heads
            encoder_layers = []
            for layer in self.model.layers:
                if 'task_' in layer.name:
                    break
                encoder_layers.append(layer)

            # Create encoder model
            encoder = keras.Sequential(encoder_layers, name='foundation_encoder')
            return encoder
        else:
            # For other models, return as-is
            return self.model

    def save(self, save_path):
        """Save the foundation model."""
        os.makedirs(save_path, exist_ok=True)

        if self.model_type == 'tfhub':
            # TFHub models need special handling
            print(f"Saving TFHub model to: {save_path}")
            self.model.save(os.path.join(save_path, 'model'))
        else:
            self.model.save(save_path)

        # Save metadata
        import json
        metadata = {
            'model_type': self.model_type,
            'input_shape': self.input_shape
        }
        with open(os.path.join(save_path, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)


def download_ptbxl_if_needed(data_path='~/datasets/ptbxl'):
    """
    Check if PTB-XL is downloaded, provide instructions if not.

    Args:
        data_path: Path where PTB-XL should be located

    Returns:
        bool: True if dataset exists, False otherwise
    """
    data_path = os.path.expanduser(data_path)

    if os.path.exists(os.path.join(data_path, 'ptbxl_database.csv')):
        return True

    print("\n" + "="*70)
    print("PTB-XL Dataset Not Found")
    print("="*70)
    print("\nPTB-XL is required for pre-training the foundation model.")
    print("Follow these steps to download:\n")
    print(f"1. Create directory:")
    print(f"   mkdir -p {data_path}")
    print(f"\n2. Download PTB-XL (~5 GB):")
    print(f"   cd {data_path}")
    print(f"   wget https://physionet.org/static/published-projects/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1.zip")
    print(f"\n3. Unzip:")
    print(f"   unzip ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1.zip")
    print(f"   mv ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1/* .")
    print(f"\n4. Re-run this script")
    print("="*70 + "\n")

    return False


# Convenience function
def load_foundation_model(model_name='ptbxl-resnet', cache_dir='./foundation_models'):
    """
    Convenience function to load ECG foundation model.

    Args:
        model_name: Name of foundation model
        cache_dir: Cache directory

    Returns:
        ECGFoundationModel instance
    """
    return ECGFoundationModel.from_pretrained(model_name, cache_dir)


if __name__ == '__main__':
    # Test loading
    print("Testing ECG Foundation Model Loader\n")

    try:
        # Try loading PTB-XL pre-trained model
        foundation = load_foundation_model('ptbxl-resnet')
        print("✓ Successfully loaded foundation model")

        # Test feature extraction
        dummy_ecg = np.random.randn(10, 2560, 1).astype('float32')
        features = foundation.extract_features(dummy_ecg, verbose=1)
        print(f"✓ Feature extraction works: {features.shape}")

    except FileNotFoundError as e:
        print(f"Foundation model not found: {e}")
        print("\nTo create a foundation model, run:")
        print("  python codes/pretrain_on_public_data.py --dataset ptbxl --epochs 50")

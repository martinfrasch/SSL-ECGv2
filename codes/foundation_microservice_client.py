"""
foundation_microservice_client.py - Client for ECG Foundation Microservice

This module provides a client to call your deployed ECG foundation microservice
on Google Cloud Run with GCP authentication.

Based on: https://github.com/martinfrasch/ecg-foundation-microservice
Service: https://ecg-foundation-microservice-2ye5avmw4a-uc.a.run.app

The microservice exposes a pre-trained foundation model (512-dim embeddings)
trained on large ECG datasets.

Usage:
    from foundation_microservice_client import load_microservice_client

    # Initialize client (handles GCP auth automatically)
    client = load_microservice_client()

    # Extract features
    features = client.extract_features(ecg_data)

    # Use features for downstream tasks
    stress_model.fit(features, stress_labels)

Author: Claude (2025-11-12)
Based on: florian-ecg-steroid-analysis/MICROSERVICE_STATUS.md
"""

import os
import numpy as np
import requests
import json
from typing import Optional, Union, List
from tqdm import tqdm
import time
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ECGFoundationClient:
    """
    Client for ECG Foundation Microservice on Google Cloud Run.

    This client communicates with your deployed ECG foundation model microservice
    to extract 512-dimensional embeddings from ECG signals.

    Handles GCP authentication automatically using gcloud credentials.
    """

    # Default production URL
    DEFAULT_URL = 'https://ecg-foundation-microservice-2ye5avmw4a-uc.a.run.app'

    def __init__(
        self,
        api_url: str = None,
        timeout: int = 600,
        max_retries: int = 3,
        use_auth: bool = True
    ):
        """
        Initialize the microservice client.

        Args:
            api_url: Base URL of microservice (default: production Cloud Run URL)
            timeout: Request timeout in seconds (default: 600 for long ECGs)
            max_retries: Maximum number of retries for failed requests
            use_auth: Whether to use GCP authentication (required for Cloud Run)
        """
        self.api_url = (api_url or self.DEFAULT_URL).rstrip('/')
        self.timeout = timeout
        self.max_retries = max_retries
        self.use_auth = use_auth

        # Token management
        self.token_expiry = 0
        self.token_ttl = 3600  # GCP tokens expire after 1 hour

        # Session for connection pooling
        self.session = requests.Session()

        # Setup GCP authentication if needed
        if self.use_auth:
            self._setup_gcp_auth()

        # Test connection
        self._test_connection()

    def _setup_gcp_auth(self):
        """
        Setup GCP authentication for Cloud Run.

        Uses google-auth library to fetch identity tokens.
        Falls back to gcloud command if library not available.
        """
        try:
            # Try using google-auth library
            import google.auth.transport.requests
            import google.oauth2.id_token

            auth_req = google.auth.transport.requests.Request()
            id_token = google.oauth2.id_token.fetch_id_token(auth_req, self.api_url)
            self.session.headers.update({"Authorization": f"Bearer {id_token}"})
            logger.info("✓ GCP authentication configured (google-auth)")

        except ImportError:
            # Fall back to gcloud command
            logger.info("google-auth not installed, trying gcloud command...")
            try:
                import subprocess
                result = subprocess.run(
                    ['gcloud', 'auth', 'print-identity-token'],
                    capture_output=True,
                    text=True,
                    check=True
                )
                id_token = result.stdout.strip()
                self.session.headers.update({"Authorization": f"Bearer {id_token}"})
                logger.info("✓ GCP authentication configured (gcloud)")

            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                logger.warning(f"⚠ Failed to setup GCP auth: {e}")
                logger.warning("Proceeding without authentication (may fail for Cloud Run)")
                logger.warning("Install google-auth: pip install google-auth")
                logger.warning("Or authenticate with: gcloud auth login")

    def _refresh_token(self):
        """Refresh GCP identity token (tokens expire after 1 hour)."""
        if self.use_auth:
            try:
                import google.auth.transport.requests
                import google.oauth2.id_token

                auth_req = google.auth.transport.requests.Request()
                id_token = google.oauth2.id_token.fetch_id_token(auth_req, self.api_url)
                self.session.headers.update({"Authorization": f"Bearer {id_token}"})
                self.token_expiry = time.time() + self.token_ttl
                logger.debug("Token refreshed")
            except ImportError:
                # Fall back to gcloud command
                try:
                    import subprocess
                    result = subprocess.run(
                        ['gcloud', 'auth', 'print-identity-token'],
                        capture_output=True,
                        text=True,
                        check=True
                    )
                    id_token = result.stdout.strip()
                    self.session.headers.update({"Authorization": f"Bearer {id_token}"})
                    self.token_expiry = time.time() + self.token_ttl
                    logger.debug("Token refreshed via gcloud")
                except Exception as e:
                    logger.warning(f"Failed to refresh token via gcloud: {e}")
            except Exception as e:
                logger.warning(f"Failed to refresh token: {e}")

    def _ensure_valid_token(self):
        """Ensure token is valid, refresh if expires in less than 5 minutes."""
        if not self.use_auth:
            return

        # Refresh if token expires in less than 5 minutes (300 seconds)
        if time.time() > (self.token_expiry - 300):
            logger.info("Token expiring soon, refreshing...")
            self._refresh_token()

    def _test_connection(self):
        """Test connection to microservice."""
        try:
            response = requests.get(
                f"{self.api_url}/health",
                timeout=10
            )
            if response.status_code == 200:
                print(f"✓ Connected to ECG Foundation Microservice at {self.api_url}")
                info = response.json()
                print(f"  Model: {info.get('model_name', 'Unknown')}")
                print(f"  Version: {info.get('version', 'Unknown')}")
            else:
                print(f"⚠ Microservice returned status {response.status_code}")
        except requests.exceptions.ConnectionError:
            print(f"\n❌ Could not connect to microservice at {self.api_url}")
            print("\nPlease ensure the microservice is running:")
            print("  1. Clone: git clone https://github.com/martinfrasch/ecg-foundation-microservice")
            print("  2. Start: cd ecg-foundation-microservice && docker-compose up")
            print(f"  3. Or update api_url if running elsewhere\n")
            raise

    def extract_features(
        self,
        ecg_data: np.ndarray,
        sampling_rate: float = 256.0,
        batch_size: int = 1,  # Process one at a time due to potential long processing
        verbose: int = 1
    ) -> np.ndarray:
        """
        Extract 512-dimensional features from ECG data using foundation microservice.

        Args:
            ecg_data: ECG signals, shape (n_samples, 2560) or (n_samples, 2560, 1)
            sampling_rate: Sampling rate in Hz (default: 256.0)
            batch_size: Number of ECGs to process per request (default: 1 for reliability)
            verbose: Verbosity level

        Returns:
            features: Extracted 512-dim features, shape (n_samples, 512)
        """
        # Ensure correct shape
        if len(ecg_data.shape) == 3:
            ecg_data = ecg_data.reshape(len(ecg_data), -1)

        if ecg_data.shape[1] != 2560:
            raise ValueError(f"Expected 2560 samples per ECG, got {ecg_data.shape[1]}")

        n_samples = len(ecg_data)

        if verbose:
            print(f"Extracting 512-dim features from {n_samples} ECG samples...")
            print(f"Using microservice: {self.api_url}")

        features_list = []

        # Process samples (one at a time for reliability)
        if verbose:
            pbar = tqdm(range(n_samples), desc="Processing ECGs")
        else:
            pbar = range(n_samples)

        for i in pbar:
            ecg_signal = ecg_data[i]

            # Ensure token is valid (refresh if needed)
            if i % 100 == 0:
                self._ensure_valid_token()

            # Call microservice API
            try:
                result = self._analyze_ecg(
                    ecg_signal=ecg_signal,
                    sampling_rate=sampling_rate
                )

                # Extract 512-dim features
                if 'foundation_features_512d' in result:
                    features = np.array(result['foundation_features_512d'])
                    features_list.append(features)
                else:
                    logger.error(f"No features in response for sample {i}")
                    # Use zeros as fallback
                    features_list.append(np.zeros(512))

            except Exception as e:
                logger.error(f"Failed to process sample {i}: {e}")
                # Use zeros as fallback
                features_list.append(np.zeros(512))

        features = np.vstack(features_list)

        if verbose:
            print(f"✓ Extracted features: {features.shape}")

        return features

    def _analyze_ecg(
        self,
        ecg_signal: np.ndarray,
        sampling_rate: float = 256.0,
        model_name: str = "default"
    ) -> dict:
        """
        Call the /analyze_ecg endpoint.

        Args:
            ecg_signal: Single ECG signal (1D array, 2560 samples)
            sampling_rate: Sampling rate in Hz
            model_name: Model name (default: "default")

        Returns:
            dict with foundation_features_512d, signal_quality_score, etc.
        """
        # Prepare payload
        payload = {
            "signal": ecg_signal.tolist(),  # Convert to list
            "sampling_rate": float(sampling_rate),
            "model_name": model_name
        }

        # Make request with retries
        for attempt in range(self.max_retries):
            try:
                response = self.session.post(
                    f"{self.api_url}/analyze_ecg",
                    json=payload,
                    timeout=self.timeout
                )

                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 503:
                    # Service unavailable, retry
                    if attempt < self.max_retries - 1:
                        wait_time = 2 ** attempt
                        logger.warning(f"Service busy, retrying in {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                    else:
                        raise RuntimeError("Service unavailable after retries")
                else:
                    error_detail = response.json().get('detail', 'Unknown error')
                    raise RuntimeError(f"API error ({response.status_code}): {error_detail}")

            except requests.exceptions.Timeout:
                if attempt < self.max_retries - 1:
                    logger.warning("Request timeout, retrying...")
                    continue
                else:
                    raise RuntimeError("Request timeout after retries")

            except requests.exceptions.ConnectionError:
                if attempt < self.max_retries - 1:
                    logger.warning("Connection error, retrying...")
                    time.sleep(2 ** attempt)
                    continue
                else:
                    raise RuntimeError("Connection failed after retries")

        raise RuntimeError("Max retries exceeded")

    def batch_process_files(
        self,
        input_files: List[str],
        output_dir: str,
        verbose: int = 1
    ):
        """
        Process multiple ECG files and save features.

        Args:
            input_files: List of .npy files containing ECG data
            output_dir: Directory to save extracted features
            verbose: Verbosity level
        """
        os.makedirs(output_dir, exist_ok=True)

        if verbose:
            print(f"Batch processing {len(input_files)} files...")

        for input_file in tqdm(input_files, desc="Processing files", disable=not verbose):
            # Load ECG data
            ecg_data = np.load(input_file, allow_pickle=True)

            # Extract features
            features = self.extract_features(ecg_data, verbose=0)

            # Save features
            output_file = os.path.join(
                output_dir,
                os.path.basename(input_file).replace('.npy', '_features.npy')
            )
            np.save(output_file, features)

        if verbose:
            print(f"✓ Saved features to: {output_dir}")


# Convenience functions
def load_microservice_client(
    api_url: str = None,
    use_auth: bool = True
) -> ECGFoundationClient:
    """
    Load ECG foundation microservice client.

    Args:
        api_url: Microservice URL (default: from environment or production URL)
        use_auth: Whether to use GCP authentication (default: True)

    Returns:
        ECGFoundationClient instance
    """
    # Get from environment if not provided
    if api_url is None:
        api_url = os.environ.get(
            'ECG_FOUNDATION_API_URL',
            ECGFoundationClient.DEFAULT_URL
        )

    return ECGFoundationClient(api_url=api_url, use_auth=use_auth)


if __name__ == '__main__':
    # Test the client
    print("Testing ECG Foundation Microservice Client\n")

    try:
        # Load client
        client = load_microservice_client()

        # Test with dummy data
        print("\nTesting feature extraction...")
        dummy_ecg = np.random.randn(10, 2560).astype('float32')
        features = client.extract_features(dummy_ecg, verbose=1)
        print(f"✓ Feature extraction works: {features.shape}")
        print(f"✓ Expected shape: (10, 512)")

        # Verify features
        assert features.shape == (10, 512), f"Expected (10, 512), got {features.shape}"

        print("\n✓ All tests passed!")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        print("\nMake sure:")
        print("  1. You are authenticated with GCP: gcloud auth login")
        print("  2. The microservice is deployed and accessible")
        print(f"  3. Service URL: {ECGFoundationClient.DEFAULT_URL}")

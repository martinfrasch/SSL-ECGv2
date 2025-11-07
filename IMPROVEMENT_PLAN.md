# Improvement Plan for SSL-ECGv2

## Table of Contents
1. [Critical Fixes](#critical-fixes)
2. [High-Priority Improvements](#high-priority-improvements)
3. [Medium-Priority Improvements](#medium-priority-improvements)
4. [Code Quality Enhancements](#code-quality-enhancements)
5. [Implementation Roadmap](#implementation-roadmap)

---

## Critical Fixes

### 1. Fix Train/Test Split - Subject-Wise Cross-Validation

**Priority:** CRITICAL
**Estimated Effort:** 2-4 hours
**Files to Modify:** `codes/datasets.py`

#### Current Implementation (INCORRECT):
```python
def _get_train_test_index(data, kfold):
    np.random.seed(999999)
    person = np.unique(data[:, 0])

    # Current: Splits each person's data across train/test
    for k in person:
        index = np.where(data[:,0] == k)[0]
        np.random.shuffle(index)
        l = len(index)
        start = kfold*int(l//total_fold)
        end = start + int(l//total_fold)
        # ... adds some windows to test, rest to train
```

#### Proposed Implementation (CORRECT):
```python
def train_test_split_felicity_kfold_subject_wise(data_folder, kfold, total_fold, overlap_pct, type_m_or_f='mecg', random_seed=42):
    """
    Perform subject-wise k-fold cross-validation.

    Each fold has completely different subjects in train vs test.
    This ensures the model is evaluated on its ability to generalize to NEW subjects.

    Args:
        data_folder: Path to data directory
        kfold: Current fold index (0 to total_fold-1)
        total_fold: Total number of folds
        overlap_pct: Overlap percentage for windowing
        type_m_or_f: Type of ECG ('mecg' or 'aecg')
        random_seed: Random seed for reproducibility

    Returns:
        train_data: Training data array
        test_data: Test data array
    """
    felicitys_data = load_data(os.path.join(data_folder, 'felicitys_'+ type_m_or_f + '_' + str(overlap_pct)+'.npy'))

    # Get unique subjects
    np.random.seed(random_seed)
    subjects = np.unique(felicitys_data[:, 0])
    np.random.shuffle(subjects)

    # Split subjects into folds
    n_subjects = len(subjects)
    fold_size = n_subjects // total_fold

    # Determine test subjects for this fold
    test_start_idx = kfold * fold_size
    test_end_idx = test_start_idx + fold_size

    # Handle last fold (may have extra subjects)
    if kfold == total_fold - 1:
        test_end_idx = n_subjects

    test_subjects = subjects[test_start_idx:test_end_idx]
    train_subjects = np.setdiff1d(subjects, test_subjects)

    # Get all data indices for train and test subjects
    train_indices = np.where(np.isin(felicitys_data[:, 0], train_subjects))[0]
    test_indices = np.where(np.isin(felicitys_data[:, 0], test_subjects))[0]

    train_data = felicitys_data[train_indices]
    test_data = felicitys_data[test_indices]

    print(f"Fold {kfold}: Train subjects: {len(train_subjects)}, Test subjects: {len(test_subjects)}")
    print(f"Fold {kfold}: Train samples: {len(train_data)}, Test samples: {len(test_data)}")

    return train_data, test_data
```

#### Validation Code:
```python
def verify_no_subject_overlap(train_data, test_data):
    """Verify that train and test sets have no overlapping subjects."""
    train_subjects = set(np.unique(train_data[:, 0]))
    test_subjects = set(np.unique(test_data[:, 0]))
    overlap = train_subjects & test_subjects

    if overlap:
        raise ValueError(f"CRITICAL: Found overlapping subjects in train/test: {overlap}")
    else:
        print(f"✓ Validation passed: No subject overlap between train ({len(train_subjects)} subjects) and test ({len(test_subjects)} subjects)")

    return True
```

#### Integration Steps:
1. Add new function to `codes/datasets.py`
2. Update `codes/train.py` line 141 to use new function:
   ```python
   felicity_train_data, felicity_test_data = datasets.train_test_split_felicity_kfold_subject_wise(
       data_folder, kfold=k, total_fold=total_fold, overlap_pct=overlap_pct,
       type_m_or_f=data_tag, random_seed=args.random_seed
   )
   # Add validation
   datasets.verify_no_subject_overlap(felicity_train_data, felicity_test_data)
   ```
3. Run comparative experiments (old vs new split) to quantify the impact

---

## High-Priority Improvements

### 2. Add Feature Normalization for Downstream Tasks

**Priority:** HIGH
**Estimated Effort:** 1-2 hours
**Files to Modify:** `codes/train.py`, `codes/model.py`

#### Implementation:

**In `codes/utils.py`, add:**
```python
from sklearn.preprocessing import StandardScaler

def normalize_features(x_train, x_test):
    """
    Normalize features using training set statistics.

    Args:
        x_train: Training features (N_train, feature_dim)
        x_test: Test features (N_test, feature_dim)

    Returns:
        x_train_norm: Normalized training features
        x_test_norm: Normalized test features
        scaler: Fitted StandardScaler (for future use)
    """
    scaler = StandardScaler()
    x_train_norm = scaler.fit_transform(x_train)
    x_test_norm = scaler.transform(x_test)

    return x_train_norm, x_test_norm, scaler
```

**In `codes/train.py`, modify lines 274-284:**
```python
x_tr_feature = utils.extract_feature(x_original = train_ECG, featureset_size = featureset_size,
                                     batch_super = batchsize, input_tensor = input_tensor,
                                     isTrain = isTrain, drop_out = drop_out,
                                     extract_layer = main_branch, sess = sess)
x_te_feature = utils.extract_feature(x_original = test_ECG, featureset_size = featureset_size,
                                     batch_super = batchsize, input_tensor = input_tensor,
                                     isTrain = isTrain, drop_out = drop_out,
                                     extract_layer = main_branch, sess = sess)

# NEW: Normalize features
x_tr_feature, x_te_feature, feature_scaler = utils.normalize_features(x_tr_feature, x_te_feature)

# Save scaler for future use
scaler_path = os.path.join(feature_saved_path, 'feature_scaler.pkl')
import pickle
with open(scaler_path, 'wb') as f:
    pickle.dump(feature_scaler, f)

if epoch_counter==epoch-1:
    model.model_classification(x_tr_feature = x_tr_feature, y_tr = train_stress,
                              x_te_feature = x_te_feature, y_te = test_stress, ...)
    # ... rest of downstream tasks
```

---

### 3. Make Random Seeds Configurable

**Priority:** HIGH
**Estimated Effort:** 2-3 hours
**Files to Modify:** `codes/train.py`, `codes/datasets.py`

#### Implementation:

**In `codes/train.py`, add argument parsing:**
```python
import argparse

parser = argparse.ArgumentParser(description='SSL-ECG Training')
parser.add_argument('--random_seed', type=int, default=42,
                    help='Random seed for reproducibility')
parser.add_argument('--data_split_seed', type=int, default=None,
                    help='Separate seed for data splitting (if None, uses random_seed)')
parser.add_argument('--num_trials', type=int, default=1,
                    help='Number of trials with different random seeds')
args = parser.parse_args()

# Set seeds
if args.data_split_seed is None:
    args.data_split_seed = args.random_seed

np.random.seed(args.random_seed)
tf.set_random_seed(args.random_seed)
```

**In `codes/datasets.py`, update all functions:**
```python
def train_test_split_felicity_kfold_subject_wise(data_folder, kfold, total_fold,
                                                   overlap_pct, type_m_or_f='mecg',
                                                   random_seed=42):  # Add parameter
    np.random.seed(random_seed)  # Use parameter instead of hardcoded value
    # ... rest of function
```

**Add experiment runner script:**
```python
# codes/run_multiple_trials.py
import subprocess
import numpy as np
import pandas as pd

def run_multiple_trials(num_trials=5, base_seed=42):
    """Run multiple trials with different random seeds and aggregate results."""
    results = []

    for trial in range(num_trials):
        seed = base_seed + trial
        print(f"\n{'='*50}")
        print(f"Running Trial {trial+1}/{num_trials} with seed {seed}")
        print(f"{'='*50}\n")

        cmd = f"python train.py --random_seed {seed} --data_split_seed {base_seed}"
        subprocess.run(cmd, shell=True)

        # Collect results
        # ... (load results from saved files)
        results.append(trial_results)

    # Aggregate results
    results_df = pd.DataFrame(results)
    print("\n" + "="*50)
    print("AGGREGATE RESULTS ACROSS ALL TRIALS")
    print("="*50)
    print(results_df.describe())
    print(f"\nMean ± Std:")
    for col in results_df.columns:
        mean = results_df[col].mean()
        std = results_df[col].std()
        print(f"{col}: {mean:.4f} ± {std:.4f}")

    return results_df

if __name__ == "__main__":
    run_multiple_trials(num_trials=5, base_seed=42)
```

---

### 4. Fix Feature Extraction Efficiency

**Priority:** HIGH
**Estimated Effort:** 1 hour
**Files to Modify:** `codes/utils.py`

#### Current Implementation (INCORRECT):
```python
def extract_feature(x_original, featureset_size, batch_super, ...):
    feature_set = np.zeros((1, featureset_size), dtype = int)
    length = np.shape(x_original)[0]
    steps = length //batch_super +1
    for j in range(steps):
        signal_batch = x_original[np.mod(np.arange(j*batch_super,(j+1)*batch_super), length)]
        # This uses modulo, causing duplicates!
```

#### Proposed Implementation (CORRECT):
```python
def extract_feature(x_original, featureset_size, batch_super, input_tensor,
                    isTrain, drop_out, extract_layer, sess):
    """
    Extract features from ECG signals using the trained self-supervised model.

    Args:
        x_original: Original ECG signals (N_samples, signal_length)
        featureset_size: Dimension of the feature vector
        batch_super: Batch size for processing
        ... (other parameters)

    Returns:
        x_feature: Extracted features (N_samples, featureset_size)
    """
    length = np.shape(x_original)[0]
    feature_list = []

    # Process in batches without duplicates
    for start_idx in range(0, length, batch_super):
        end_idx = min(start_idx + batch_super, length)
        signal_batch = x_original[start_idx:end_idx]

        # Pad last batch if needed
        if signal_batch.shape[0] < batch_super:
            pad_size = batch_super - signal_batch.shape[0]
            signal_batch = np.vstack([
                signal_batch,
                np.zeros((pad_size, signal_batch.shape[1]))
            ])
            is_padded = True
        else:
            is_padded = False

        # Reshape for model input
        signal_batch = signal_batch.reshape(signal_batch.shape[0], signal_batch.shape[1], 1)

        # Extract features
        fetched = sess.run(extract_layer, {
            input_tensor: signal_batch,
            isTrain: False,
            drop_out: 0.0
        })

        # Remove padding from features if needed
        if is_padded:
            fetched = fetched[:-pad_size]

        feature_list.append(fetched)

    # Concatenate all features
    x_feature = np.vstack(feature_list)

    assert x_feature.shape[0] == length, \
        f"Feature extraction error: got {x_feature.shape[0]} features, expected {length}"

    return x_feature
```

---

## Medium-Priority Improvements

### 5. Add Comprehensive Logging and Experiment Tracking

**Priority:** MEDIUM
**Estimated Effort:** 3-4 hours
**Files to Create/Modify:** `codes/train.py`, `codes/logger.py`

#### Implementation:

**Create `codes/logger.py`:**
```python
import logging
import json
import os
from datetime import datetime

class ExperimentLogger:
    """Logger for tracking experiments and results."""

    def __init__(self, experiment_name, log_dir='logs'):
        self.experiment_name = experiment_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = os.path.join(log_dir, f"{experiment_name}_{self.timestamp}")
        os.makedirs(self.log_dir, exist_ok=True)

        # Setup logging
        self.logger = logging.getLogger(experiment_name)
        self.logger.setLevel(logging.INFO)

        # File handler
        fh = logging.FileHandler(os.path.join(self.log_dir, 'experiment.log'))
        fh.setLevel(logging.INFO)

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

        self.metrics = {}

    def log_params(self, params):
        """Log experiment parameters."""
        self.logger.info("Experiment Parameters:")
        for key, value in params.items():
            self.logger.info(f"  {key}: {value}")

        # Save to JSON
        with open(os.path.join(self.log_dir, 'params.json'), 'w') as f:
            json.dump(params, f, indent=2)

    def log_metric(self, name, value, fold=None, epoch=None):
        """Log a metric value."""
        key = f"{name}"
        if fold is not None:
            key += f"_fold{fold}"
        if epoch is not None:
            key += f"_epoch{epoch}"

        if key not in self.metrics:
            self.metrics[key] = []
        self.metrics[key].append(value)

        self.logger.info(f"{key}: {value}")

    def save_metrics(self):
        """Save all metrics to JSON."""
        with open(os.path.join(self.log_dir, 'metrics.json'), 'w') as f:
            json.dump(self.metrics, f, indent=2)
```

**In `codes/train.py`, integrate logger:**
```python
from logger import ExperimentLogger

# Initialize logger
logger = ExperimentLogger(f"SSL_ECG_{data_tag}_{current_time}")
logger.log_params({
    'data_tag': data_tag,
    'batchsize': batchsize,
    'epoch': epoch,
    'learning_rate': initial_learning_rate,
    'dropout': drop_rate,
    'window_size': window_size,
    'total_fold': total_fold,
    'random_seed': args.random_seed,
    # ... other params
})

# During training
for k in range(total_fold):
    for epoch_counter in range(epoch):
        # ... training code ...
        logger.log_metric('train_loss', tr_output_loss, fold=k, epoch=epoch_counter)
        logger.log_metric('train_accuracy', tr_epoch_accuracy, fold=k, epoch=epoch_counter)
        logger.log_metric('test_loss', te_output_loss, fold=k, epoch=epoch_counter)
        logger.log_metric('test_accuracy', te_epoch_accuracy, fold=k, epoch=epoch_counter)

logger.save_metrics()
```

---

### 6. Add Input Validation and Error Handling

**Priority:** MEDIUM
**Estimated Effort:** 2-3 hours
**Files to Modify:** All utility functions

#### Example Implementation:

**Add to `codes/utils.py`:**
```python
def validate_ecg_signal(signal, expected_length=None, signal_name="ECG"):
    """
    Validate ECG signal array.

    Args:
        signal: ECG signal array
        expected_length: Expected signal length (optional)
        signal_name: Name for error messages

    Raises:
        ValueError: If validation fails
    """
    if signal is None:
        raise ValueError(f"{signal_name} is None")

    if not isinstance(signal, np.ndarray):
        raise TypeError(f"{signal_name} must be numpy array, got {type(signal)}")

    if len(signal.shape) == 0 or signal.shape[0] == 0:
        raise ValueError(f"{signal_name} is empty")

    if np.any(np.isnan(signal)):
        raise ValueError(f"{signal_name} contains NaN values")

    if np.any(np.isinf(signal)):
        raise ValueError(f"{signal_name} contains infinite values")

    if expected_length is not None and signal.shape[0] != expected_length:
        raise ValueError(f"{signal_name} has length {signal.shape[0]}, expected {expected_length}")

    return True

def make_window(signal, fs, overlap, window_size_sec):
    """
    Perform cropped signals of window_size seconds for the whole signal.

    Args:
        signal: Input signal
        fs: Sampling frequency
        overlap: Overlap percentage (0-100)
        window_size_sec: Window size in seconds

    Returns:
        segmented: Segmented signal array

    Raises:
        ValueError: If input parameters are invalid
    """
    # Input validation
    validate_ecg_signal(signal, signal_name="input signal")

    if fs <= 0:
        raise ValueError(f"Sampling frequency must be positive, got {fs}")

    if not 0 <= overlap < 100:
        raise ValueError(f"Overlap must be in [0, 100), got {overlap}")

    if window_size_sec <= 0:
        raise ValueError(f"Window size must be positive, got {window_size_sec}")

    window_size = fs * window_size_sec

    if len(signal) < window_size:
        raise ValueError(f"Signal length ({len(signal)}) is shorter than window size ({window_size})")

    # ... rest of function
```

---

## Code Quality Enhancements

### 7. Add Comprehensive Documentation

**Priority:** MEDIUM
**Estimated Effort:** 4-6 hours

#### Implementation Guidelines:

**Module-level docstrings:**
```python
"""
datasets.py - Data loading and splitting utilities for SSL-ECG

This module provides functions for:
- Loading preprocessed ECG data from .npy files
- Splitting data into train/test sets using subject-wise k-fold cross-validation
- Ensuring no subject overlap between training and test sets

Functions:
    load_data: Load ECG data from .npy file
    train_test_split_felicity_kfold_subject_wise: Perform subject-wise k-fold CV
    verify_no_subject_overlap: Validate train/test split
"""
```

**Function docstrings (Google style):**
```python
def train_test_split_felicity_kfold_subject_wise(data_folder, kfold, total_fold,
                                                   overlap_pct, type_m_or_f='mecg',
                                                   random_seed=42):
    """
    Perform subject-wise k-fold cross-validation for ECG data.

    This function ensures that training and test sets contain data from completely
    different subjects (no subject overlap). This is critical for evaluating the
    model's ability to generalize to new, unseen patients.

    Args:
        data_folder (str): Path to directory containing preprocessed data files
        kfold (int): Current fold index (0 to total_fold-1)
        total_fold (int): Total number of folds for cross-validation
        overlap_pct (int): Overlap percentage used during windowing (for filename)
        type_m_or_f (str, optional): Type of ECG data ('mecg' or 'aecg').
            Defaults to 'mecg'.
        random_seed (int, optional): Random seed for reproducibility. Defaults to 42.

    Returns:
        tuple: A tuple containing:
            - train_data (numpy.ndarray): Training data array with shape
              (n_train_samples, n_features)
            - test_data (numpy.ndarray): Test data array with shape
              (n_test_samples, n_features)

    Raises:
        FileNotFoundError: If data file does not exist
        ValueError: If kfold >= total_fold or other invalid parameters

    Example:
        >>> train_data, test_data = train_test_split_felicity_kfold_subject_wise(
        ...     data_folder='./data',
        ...     kfold=0,
        ...     total_fold=5,
        ...     overlap_pct=0,
        ...     type_m_or_f='mecg',
        ...     random_seed=42
        ... )
        >>> print(f"Train subjects: {len(np.unique(train_data[:, 0]))}")
        Train subjects: 40

    Note:
        The first column (index 0) of the data array must contain subject IDs.
        Data format: [subject_id, stress_label, pss, pdq, fsi, cortisol, ...ECG_windows...]
    """
    # Function implementation
```

---

### 8. Add Unit Tests

**Priority:** MEDIUM
**Estimated Effort:** 6-8 hours

#### Create `tests/test_datasets.py`:
```python
import unittest
import numpy as np
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'codes'))

from datasets import train_test_split_felicity_kfold_subject_wise, verify_no_subject_overlap

class TestDataSplitting(unittest.TestCase):

    def setUp(self):
        """Create mock data for testing."""
        # Create mock data: 5 subjects, 10 samples each
        self.mock_data = []
        for subject_id in range(1, 6):
            for sample_idx in range(10):
                # Format: [subject_id, label, feature1, feature2, ...]
                sample = [subject_id, np.random.randint(0, 2)] + list(np.random.randn(10))
                self.mock_data.append(sample)

        self.mock_data = np.array(self.mock_data)

        # Save to temp file
        self.temp_dir = './temp_test_data'
        os.makedirs(self.temp_dir, exist_ok=True)
        self.temp_file = os.path.join(self.temp_dir, 'felicitys_mecg_0.npy')
        np.save(self.temp_file, self.mock_data)

    def tearDown(self):
        """Clean up temp files."""
        if os.path.exists(self.temp_file):
            os.remove(self.temp_file)
        if os.path.exists(self.temp_dir):
            os.rmdir(self.temp_dir)

    def test_no_subject_overlap(self):
        """Test that train and test sets have no overlapping subjects."""
        for fold in range(5):
            train_data, test_data = train_test_split_felicity_kfold_subject_wise(
                self.temp_dir, fold, total_fold=5, overlap_pct=0,
                type_m_or_f='mecg', random_seed=42
            )

            train_subjects = set(np.unique(train_data[:, 0]))
            test_subjects = set(np.unique(test_data[:, 0]))

            overlap = train_subjects & test_subjects
            self.assertEqual(len(overlap), 0,
                           f"Fold {fold}: Found overlapping subjects: {overlap}")

    def test_all_subjects_used(self):
        """Test that all subjects appear in exactly one fold as test."""
        all_test_subjects = set()

        for fold in range(5):
            train_data, test_data = train_test_split_felicity_kfold_subject_wise(
                self.temp_dir, fold, total_fold=5, overlap_pct=0,
                type_m_or_f='mecg', random_seed=42
            )

            test_subjects = set(np.unique(test_data[:, 0]))

            # Check no subject appears in multiple test folds
            overlap_with_previous = all_test_subjects & test_subjects
            self.assertEqual(len(overlap_with_previous), 0,
                           f"Subject(s) {overlap_with_previous} appear in multiple test folds")

            all_test_subjects.update(test_subjects)

        # Check all subjects were used
        original_subjects = set(np.unique(self.mock_data[:, 0]))
        self.assertEqual(all_test_subjects, original_subjects,
                        "Not all subjects were used in test folds")

    def test_data_integrity(self):
        """Test that train/test split preserves data integrity."""
        for fold in range(5):
            train_data, test_data = train_test_split_felicity_kfold_subject_wise(
                self.temp_dir, fold, total_fold=5, overlap_pct=0,
                type_m_or_f='mecg', random_seed=42
            )

            # Check no data loss
            total_samples = train_data.shape[0] + test_data.shape[0]
            self.assertEqual(total_samples, self.mock_data.shape[0],
                           f"Data loss detected in fold {fold}")

            # Check data shape preserved
            self.assertEqual(train_data.shape[1], self.mock_data.shape[1])
            self.assertEqual(test_data.shape[1], self.mock_data.shape[1])

    def test_reproducibility(self):
        """Test that same seed produces same split."""
        train1, test1 = train_test_split_felicity_kfold_subject_wise(
            self.temp_dir, 0, total_fold=5, overlap_pct=0,
            type_m_or_f='mecg', random_seed=42
        )

        train2, test2 = train_test_split_felicity_kfold_subject_wise(
            self.temp_dir, 0, total_fold=5, overlap_pct=0,
            type_m_or_f='mecg', random_seed=42
        )

        np.testing.assert_array_equal(train1, train2)
        np.testing.assert_array_equal(test1, test2)

if __name__ == '__main__':
    unittest.main()
```

---

## Implementation Roadmap

### Phase 1: Critical Fixes (Week 1)
**Goal:** Fix data leakage and re-evaluate model

- [ ] Day 1-2: Implement subject-wise cross-validation
- [ ] Day 3: Add validation checks
- [ ] Day 4-5: Re-run experiments with corrected split
- [ ] Day 6: Compare old vs new results
- [ ] Day 7: Document findings and update paper

**Deliverables:**
- Fixed `datasets.py` with subject-wise splitting
- Validation code to prevent future leakage
- Comparative analysis report
- Updated performance metrics

### Phase 2: High-Priority Improvements (Week 2)
**Goal:** Improve model reliability and reproducibility

- [ ] Day 1: Add feature normalization
- [ ] Day 2: Make random seeds configurable
- [ ] Day 3: Fix feature extraction efficiency
- [ ] Day 4-5: Run multi-seed experiments
- [ ] Day 6-7: Analyze results and create visualizations

**Deliverables:**
- Normalized feature pipeline
- Multi-seed experiment runner
- Statistical analysis across seeds (mean ± std)
- Updated codebase

### Phase 3: Medium-Priority Improvements (Week 3-4)
**Goal:** Enhance maintainability and usability

- [ ] Week 3, Day 1-2: Add comprehensive logging
- [ ] Week 3, Day 3-4: Add input validation
- [ ] Week 3, Day 5: Add error handling
- [ ] Week 4, Day 1-2: Write documentation
- [ ] Week 4, Day 3-5: Create unit tests

**Deliverables:**
- Experiment tracking system
- Validated and robust code
- Comprehensive documentation
- Test suite

### Phase 4: Code Quality (Week 5)
**Goal:** Clean up and refactor

- [ ] Day 1-2: Refactor duplicate code
- [ ] Day 3: Improve variable naming
- [ ] Day 4: Add type hints
- [ ] Day 5: Final review and cleanup

**Deliverables:**
- Clean, maintainable codebase
- Code review report

---

## Testing Checklist

After implementing fixes, verify:

### Critical Tests
- [ ] **No subject overlap:** Verify train/test have different subjects
- [ ] **All subjects used:** Each subject appears in exactly one test fold
- [ ] **Data integrity:** No data loss during splitting
- [ ] **Reproducibility:** Same seed produces same results
- [ ] **Performance drop:** New results should be lower (fixing leakage)

### Integration Tests
- [ ] **End-to-end training:** Full training pipeline completes without errors
- [ ] **Feature extraction:** Correct number of features extracted
- [ ] **Normalization:** Features properly normalized with correct statistics
- [ ] **Multi-seed:** Multiple seeds produce consistent distribution of results

### Validation Tests
- [ ] **Sanity check:** Within-subject split should still get high performance
- [ ] **Ablation study:** Compare within vs between subject splits
- [ ] **Statistical significance:** Report confidence intervals

---

## Expected Impact

### Performance Changes After Fixes

**Prediction:**
- **Current (buggy) results:** Likely showing 85-95% accuracy
- **Fixed results:** Expected 65-80% accuracy (20-30% drop is normal)

This drop is EXPECTED and CORRECT because:
1. Current results are inflated due to subject leakage
2. The model currently memorizes subject-specific patterns
3. Fixed evaluation tests true generalization to new subjects

### Reporting Guidelines

When reporting fixed results:
```
Previous results (incorrect): XX.X% ± Y.Y% (within-subject split)
Corrected results: AA.A% ± B.B% (between-subject split)

Note: The performance difference is due to fixing a critical data leakage issue
where the original implementation used within-subject cross-validation. The
corrected results better reflect the model's ability to generalize to new patients.
```

---

## Success Metrics

### Technical Metrics
- ✅ Zero subject overlap between train/test (verified)
- ✅ Consistent results across multiple random seeds (std < 5%)
- ✅ All unit tests passing
- ✅ Feature normalization implemented
- ✅ Comprehensive logging in place

### Scientific Metrics
- ✅ Proper between-subject cross-validation
- ✅ Results reported with confidence intervals
- ✅ Ablation study comparing split strategies
- ✅ Statistical significance tests performed

### Code Quality Metrics
- ✅ All functions documented with docstrings
- ✅ Input validation on critical functions
- ✅ Unit test coverage > 80%
- ✅ No hardcoded parameters
- ✅ Reproducible experiments

---

## Maintenance Plan

### Regular Tasks
1. **Weekly:** Review experiment logs
2. **Monthly:** Run full test suite
3. **Quarterly:** Update dependencies
4. **Annually:** Review and update documentation

### Version Control
- Use semantic versioning: MAJOR.MINOR.PATCH
- Tag releases after major fixes
- Maintain CHANGELOG.md

### Documentation Updates
- Keep README up to date with installation/usage
- Document all parameter changes
- Update citations and references

---

## Contact and Support

For questions or issues related to these improvements:
1. Check documentation in `/docs`
2. Review test examples in `/tests`
3. Open an issue on GitHub
4. Contact: [maintainer email]

---

## References

- [Machine Learning Best Practices](https://developers.google.com/machine-learning/guides/rules-of-ml)
- [Cross-Validation in ML](https://scikit-learn.org/stable/modules/cross_validation.html)
- [Data Leakage in ML](https://machinelearningmastery.com/data-leakage-machine-learning/)
- [Python Testing Best Practices](https://realpython.com/python-testing/)

---

**Document Version:** 1.0
**Last Updated:** 2025-11-07
**Status:** Ready for Implementation

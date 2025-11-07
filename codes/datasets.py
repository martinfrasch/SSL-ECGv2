"""
datasets.py - Data loading and splitting utilities for SSL-ECG

This module provides functions for:
- Loading preprocessed ECG data from .npy files
- Splitting data into train/test sets using subject-wise k-fold cross-validation
- Ensuring no subject overlap between training and test sets

Data format: [subject_id, stress_label, pss, pdq, fsi, cortisol, ...ECG_windows...]
"""

import os
import warnings
## get filename during run time and set pwd manually
dirname, filename = os.path.split(os.path.abspath(__file__))
print("running: {}".format(filename) )

from pathlib import Path
import numpy as np
from sklearn.model_selection import KFold

def load_data(path):
    """
    Load ECG data from .npy file.

    Args:
        path (str): Path to .npy file

    Returns:
        numpy.ndarray: Loaded dataset
    """
    dataset = np.load(path, allow_pickle=True)
    return dataset


def verify_no_subject_overlap(train_data, test_data, verbose=True):
    """
    Verify that train and test sets have no overlapping subjects.

    Args:
        train_data (numpy.ndarray): Training data (first column must be subject IDs)
        test_data (numpy.ndarray): Test data (first column must be subject IDs)
        verbose (bool): If True, print validation results

    Returns:
        bool: True if validation passes

    Raises:
        ValueError: If overlapping subjects are found
    """
    train_subjects = set(np.unique(train_data[:, 0]).astype(int))
    test_subjects = set(np.unique(test_data[:, 0]).astype(int))
    overlap = train_subjects & test_subjects

    if overlap:
        raise ValueError(
            f"CRITICAL DATA LEAKAGE DETECTED!\n"
            f"Found {len(overlap)} overlapping subjects in train/test: {sorted(overlap)}\n"
            f"Train subjects: {sorted(train_subjects)}\n"
            f"Test subjects: {sorted(test_subjects)}"
        )

    if verbose:
        print(f"✓ Validation passed: No subject overlap")
        print(f"  Train: {len(train_subjects)} subjects, {len(train_data)} samples")
        print(f"  Test:  {len(test_subjects)} subjects, {len(test_data)} samples")

    return True


def train_test_split_felicity_kfold(data_folder, kfold, total_fold, overlap_pct,
                                     type_m_or_f='mecg', random_seed=42,
                                     subject_wise=True, validate=True):
    """
    Perform k-fold cross-validation for ECG data with SUBJECT-WISE splitting.

    This function ensures that training and test sets contain data from completely
    different subjects (no subject overlap). This is critical for evaluating the
    model's ability to generalize to new, unseen patients.

    Args:
        data_folder (str): Path to directory containing preprocessed data files
        kfold (int): Current fold index (0 to total_fold-1)
        total_fold (int): Total number of folds for cross-validation
        overlap_pct (int): Overlap percentage used during windowing (for filename)
        type_m_or_f (str): Type of ECG data ('mecg' or 'aecg'). Default: 'mecg'
        random_seed (int): Random seed for reproducibility. Default: 42
        subject_wise (bool): If True, use subject-wise split (CORRECT).
                           If False, use old within-subject split (INCORRECT).
                           Default: True
        validate (bool): If True, validate no subject overlap. Default: True

    Returns:
        tuple: (train_data, test_data)
            - train_data (numpy.ndarray): Training data array
            - test_data (numpy.ndarray): Test data array

    Raises:
        FileNotFoundError: If data file does not exist
        ValueError: If validation fails (subject overlap detected)

    Example:
        >>> train_data, test_data = train_test_split_felicity_kfold(
        ...     data_folder='./data',
        ...     kfold=0,
        ...     total_fold=5,
        ...     overlap_pct=0,
        ...     type_m_or_f='mecg',
        ...     random_seed=42
        ... )
        >>> print(f"Train subjects: {len(np.unique(train_data[:, 0]))}")

    Note:
        Setting subject_wise=False will use the OLD INCORRECT method that causes
        data leakage. This option is only provided for comparison purposes.
    """

    # Load data
    data_path = os.path.join(data_folder, f'felicitys_{type_m_or_f}_{overlap_pct}.npy')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    felicitys_data = load_data(data_path)

    if subject_wise:
        # NEW CORRECT METHOD: Subject-wise cross-validation
        train_data, test_data = _subject_wise_split(
            felicitys_data, kfold, total_fold, random_seed
        )
    else:
        # OLD INCORRECT METHOD: Within-subject split (causes data leakage)
        warnings.warn(
            "WARNING: Using within-subject split (subject_wise=False) causes data leakage! "
            "This inflates performance metrics by 20-30%. Use subject_wise=True for correct evaluation.",
            UserWarning
        )
        train_index, test_index = _get_train_test_index_old(
            felicitys_data, kfold, total_fold, random_seed
        )
        train_data = felicitys_data[train_index]
        test_data = felicitys_data[test_index]

    # Validate no subject overlap
    if validate:
        verify_no_subject_overlap(train_data, test_data, verbose=True)

    return train_data, test_data


def _subject_wise_split(data, kfold, total_fold, random_seed):
    """
    CORRECT: Split data by subjects (between-subject cross-validation).

    Each fold has completely different subjects in train vs test.
    This ensures the model is evaluated on its ability to generalize to NEW subjects.

    Args:
        data (numpy.ndarray): Full dataset
        kfold (int): Current fold index
        total_fold (int): Total number of folds
        random_seed (int): Random seed

    Returns:
        tuple: (train_data, test_data)
    """
    # Set random seed for reproducibility
    np.random.seed(random_seed)

    # Get unique subjects and shuffle them
    subjects = np.unique(data[:, 0]).astype(int)
    np.random.shuffle(subjects)

    # Calculate fold boundaries
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
    train_indices = np.where(np.isin(data[:, 0], train_subjects))[0]
    test_indices = np.where(np.isin(data[:, 0], test_subjects))[0]

    train_data = data[train_indices]
    test_data = data[test_indices]

    print(f"Fold {kfold}/{total_fold-1}: Train subjects: {len(train_subjects)}, "
          f"Test subjects: {len(test_subjects)}")

    return train_data, test_data


def _get_train_test_index_old(data, kfold, total_fold, random_seed):
    """
    INCORRECT: Old method that splits WITHIN each subject (causes data leakage).

    DO NOT USE THIS METHOD for real experiments!
    This is kept only for comparison and debugging purposes.

    Args:
        data (numpy.ndarray): Full dataset
        kfold (int): Current fold index
        total_fold (int): Total number of folds
        random_seed (int): Random seed

    Returns:
        tuple: (train_index, test_index)
    """
    np.random.seed(random_seed)
    person = np.unique(data[:, 0])

    test_index = np.zeros((1,1))
    for k in person:
        index = np.where(data[:,0] == k)[0]
        np.random.shuffle(index)
        l = len(index)
        start = kfold*int(l//total_fold)
        end = start + int(l//total_fold)

        if np.all(test_index ==0):
            test_index = index[start:end]
        else:
            test_index = np.hstack((test_index, index[start:end]))

    train_index = np.setdiff1d(np.arange(len(data)), test_index)

    return train_index, test_index

    
def train_test_subjects(dataset, test_pct, random_seed=42):
    """
    Split subjects into train and test groups.

    Args:
        dataset (numpy.ndarray): Dataset with subject IDs in column 1
        test_pct (float): Percentage of subjects for testing (0.0-1.0)
        random_seed (int): Random seed for reproducibility. Default: 42

    Returns:
        tuple: (train_subs, test_subs) - Arrays of subject IDs
    """
    np.random.seed(random_seed)
    person = np.unique(dataset[:, 1])
    no_test_subs = int(np.round(len(person) * test_pct))
    test_subs = np.random.choice(person, size= no_test_subs, replace=False)
    train_subs = np.setdiff1d(person, test_subs)

    return train_subs, test_subs

def get_train_test_index(data, test_pct, random_seed=42):
    """
    Get train/test indices based on subject-wise split.

    Args:
        data (numpy.ndarray): Dataset with subject IDs in column 1
        test_pct (float): Percentage of subjects for testing (0.0-1.0)
        random_seed (int): Random seed for reproducibility. Default: 42

    Returns:
        tuple: (train_index, test_index) - Arrays of data indices
    """
    train_subs, test_subs = train_test_subjects(data, test_pct, random_seed)

    test_index = np.zeros((1,1))
    for k in test_subs:
        index = np.where(data[:,1] == k)
        if np.all(test_index ==0):
            test_index = index[0]
        else:
            test_index = np.hstack((test_index, index[0]))

    train_index = np.setdiff1d(np.arange(len(data)), test_index)

    return train_index, test_index

def train_test_index_felicity(data, test_pct, random_seed=42):
    """
    Get train/test indices for Felicity dataset based on subject-wise split.

    Args:
        data (numpy.ndarray): Dataset with subject IDs in column 0
        test_pct (float): Percentage of subjects for testing (0.0-1.0)
        random_seed (int): Random seed for reproducibility. Default: 42

    Returns:
        tuple: (train_index, test_index) - Arrays of data indices
    """
    def _train_test_subjects(dataset, test_pct, seed):
        np.random.seed(seed)
        person = np.unique(dataset[:, 0])
        no_test_subs = int(np.round(len(person) * test_pct))
        test_subs = np.random.choice(person, size= no_test_subs, replace=False)
        train_subs = np.setdiff1d(person, test_subs)

        return train_subs, test_subs

    train_subs, test_subs = _train_test_subjects(data, test_pct, random_seed)

    test_index = np.zeros((1,1))
    for k in test_subs:
        index = np.where(data[:,0] == k)
        if np.all(test_index ==0):
            test_index = index[0]
        else:
            test_index = np.hstack((test_index, index[0]))

    train_index = np.setdiff1d(np.arange(len(data)), test_index)

    return train_index, test_index
                                   

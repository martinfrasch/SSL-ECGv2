"""
test_datasets.py - Unit tests for data splitting functions

These tests verify that the subject-wise cross-validation is working correctly
and that there is no data leakage between train and test sets.

Run tests with:
    python -m pytest tests/test_datasets.py -v

Or:
    python tests/test_datasets.py
"""

import unittest
import numpy as np
import os
import sys

# Add parent directory to path to import codes
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'codes'))

from datasets import (
    train_test_split_felicity_kfold,
    verify_no_subject_overlap,
    _subject_wise_split
)


class TestDataSplitting(unittest.TestCase):
    """Test suite for data splitting functions."""

    def setUp(self):
        """Create mock data for testing."""
        # Create mock data: 5 subjects, 10 samples each
        self.mock_data = []
        for subject_id in range(1, 6):
            for sample_idx in range(10):
                # Format: [subject_id, label, pss, pdq, fsi, cortisol, ...features (10)]
                sample = [subject_id, np.random.randint(0, 2)] + \
                         [np.random.rand() for _ in range(4)] + \
                         list(np.random.randn(10))
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

    def test_no_subject_overlap_subject_wise(self):
        """Test that subject-wise split has no overlapping subjects."""
        print("\n" + "="*60)
        print("TEST: No subject overlap (subject-wise split)")
        print("="*60)

        for fold in range(5):
            train_data, test_data = train_test_split_felicity_kfold(
                self.temp_dir, fold, total_fold=5, overlap_pct=0,
                type_m_or_f='mecg', random_seed=42, subject_wise=True,
                validate=False  # We'll validate manually
            )

            train_subjects = set(np.unique(train_data[:, 0]).astype(int))
            test_subjects = set(np.unique(test_data[:, 0]).astype(int))

            overlap = train_subjects & test_subjects
            self.assertEqual(len(overlap), 0,
                           f"Fold {fold}: Found overlapping subjects: {overlap}")

            print(f"✓ Fold {fold}: No overlap - Train subjects: {sorted(train_subjects)}, "
                  f"Test subjects: {sorted(test_subjects)}")

    def test_within_subject_has_overlap(self):
        """Test that within-subject split DOES have overlap (the bug!)."""
        print("\n" + "="*60)
        print("TEST: Within-subject split HAS overlap (demonstrates the bug)")
        print("="*60)

        overlap_found = False
        for fold in range(5):
            train_data, test_data = train_test_split_felicity_kfold(
                self.temp_dir, fold, total_fold=5, overlap_pct=0,
                type_m_or_f='mecg', random_seed=42, subject_wise=False,
                validate=False  # Don't validate, we expect this to be wrong
            )

            train_subjects = set(np.unique(train_data[:, 0]).astype(int))
            test_subjects = set(np.unique(test_data[:, 0]).astype(int))

            overlap = train_subjects & test_subjects
            if len(overlap) > 0:
                overlap_found = True
                print(f"✓ Fold {fold}: Found overlap (as expected for buggy version): {sorted(overlap)}")

        self.assertTrue(overlap_found,
                       "Within-subject split should have overlap (this is the bug!)")

    def test_all_subjects_used(self):
        """Test that all subjects appear in exactly one fold as test."""
        print("\n" + "="*60)
        print("TEST: All subjects used exactly once")
        print("="*60)

        all_test_subjects = set()

        for fold in range(5):
            train_data, test_data = train_test_split_felicity_kfold(
                self.temp_dir, fold, total_fold=5, overlap_pct=0,
                type_m_or_f='mecg', random_seed=42, subject_wise=True,
                validate=False
            )

            test_subjects = set(np.unique(test_data[:, 0]).astype(int))

            # Check no subject appears in multiple test folds
            overlap_with_previous = all_test_subjects & test_subjects
            self.assertEqual(len(overlap_with_previous), 0,
                           f"Subject(s) {overlap_with_previous} appear in multiple test folds")

            all_test_subjects.update(test_subjects)
            print(f"✓ Fold {fold}: Test subjects: {sorted(test_subjects)}")

        # Check all subjects were used
        original_subjects = set(np.unique(self.mock_data[:, 0]).astype(int))
        self.assertEqual(all_test_subjects, original_subjects,
                        "Not all subjects were used in test folds")

        print(f"✓ All {len(original_subjects)} subjects used exactly once")

    def test_data_integrity(self):
        """Test that train/test split preserves data integrity."""
        print("\n" + "="*60)
        print("TEST: Data integrity preserved")
        print("="*60)

        for fold in range(5):
            train_data, test_data = train_test_split_felicity_kfold(
                self.temp_dir, fold, total_fold=5, overlap_pct=0,
                type_m_or_f='mecg', random_seed=42, subject_wise=True,
                validate=False
            )

            # Check no data loss
            total_samples = train_data.shape[0] + test_data.shape[0]
            self.assertEqual(total_samples, self.mock_data.shape[0],
                           f"Data loss detected in fold {fold}")

            # Check data shape preserved
            self.assertEqual(train_data.shape[1], self.mock_data.shape[1])
            self.assertEqual(test_data.shape[1], self.mock_data.shape[1])

            print(f"✓ Fold {fold}: Data integrity preserved - "
                  f"Train: {train_data.shape[0]}, Test: {test_data.shape[0]}, "
                  f"Total: {total_samples}")

    def test_reproducibility(self):
        """Test that same seed produces same split."""
        print("\n" + "="*60)
        print("TEST: Reproducibility with same seed")
        print("="*60)

        train1, test1 = train_test_split_felicity_kfold(
            self.temp_dir, 0, total_fold=5, overlap_pct=0,
            type_m_or_f='mecg', random_seed=42, subject_wise=True,
            validate=False
        )

        train2, test2 = train_test_split_felicity_kfold(
            self.temp_dir, 0, total_fold=5, overlap_pct=0,
            type_m_or_f='mecg', random_seed=42, subject_wise=True,
            validate=False
        )

        np.testing.assert_array_equal(train1, train2)
        np.testing.assert_array_equal(test1, test2)

        print("✓ Same seed produces identical splits")

    def test_different_seeds_different_splits(self):
        """Test that different seeds produce different splits."""
        print("\n" + "="*60)
        print("TEST: Different seeds produce different splits")
        print("="*60)

        train1, test1 = train_test_split_felicity_kfold(
            self.temp_dir, 0, total_fold=5, overlap_pct=0,
            type_m_or_f='mecg', random_seed=42, subject_wise=True,
            validate=False
        )

        train2, test2 = train_test_split_felicity_kfold(
            self.temp_dir, 0, total_fold=5, overlap_pct=0,
            type_m_or_f='mecg', random_seed=123, subject_wise=True,
            validate=False
        )

        # Subjects should be different (with high probability)
        subjects1 = set(np.unique(test1[:, 0]).astype(int))
        subjects2 = set(np.unique(test2[:, 0]).astype(int))

        # They might overlap by chance, but shouldn't be identical
        self.assertNotEqual(subjects1, subjects2,
                          "Different seeds should produce different test subjects")

        print(f"✓ Seed 42 test subjects: {sorted(subjects1)}")
        print(f"✓ Seed 123 test subjects: {sorted(subjects2)}")

    def test_validation_raises_error_on_overlap(self):
        """Test that validation function raises error when overlap detected."""
        print("\n" + "="*60)
        print("TEST: Validation detects overlap")
        print("="*60)

        # Create overlapping data
        train_data = self.mock_data[:30]  # Subjects 1, 2, 3
        test_data = self.mock_data[20:40]  # Subjects 2, 3, 4 (overlap!)

        with self.assertRaises(ValueError) as context:
            verify_no_subject_overlap(train_data, test_data)

        self.assertIn("DATA LEAKAGE DETECTED", str(context.exception))
        print("✓ Validation correctly detected overlap and raised error")

    def test_fold_sizes_balanced(self):
        """Test that fold sizes are approximately balanced."""
        print("\n" + "="*60)
        print("TEST: Fold sizes are balanced")
        print("="*60)

        test_subject_counts = []

        for fold in range(5):
            _, test_data = train_test_split_felicity_kfold(
                self.temp_dir, fold, total_fold=5, overlap_pct=0,
                type_m_or_f='mecg', random_seed=42, subject_wise=True,
                validate=False
            )

            n_test_subjects = len(np.unique(test_data[:, 0]))
            test_subject_counts.append(n_test_subjects)
            print(f"Fold {fold}: {n_test_subjects} test subjects")

        # All folds should have 1 subject (5 subjects / 5 folds)
        expected_size = len(np.unique(self.mock_data[:, 0])) // 5
        for count in test_subject_counts:
            self.assertTrue(count == expected_size or count == expected_size + 1,
                          f"Fold size {count} too far from expected {expected_size}")

        print(f"✓ Fold sizes balanced around {expected_size} subjects per fold")


def run_tests():
    """Run all tests with detailed output."""
    suite = unittest.TestLoader().loadTestsFromTestCase(TestDataSplitting)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("="*60)

    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)

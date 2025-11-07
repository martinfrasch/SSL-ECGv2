# Code Review: SSL-ECGv2 - Data Leakage and Correctness Analysis

**Date:** 2025-11-07
**Reviewer:** Claude
**Repository:** SSL-ECGv2 (Detection of Maternal and Fetal Stress from ECG with Self-supervised Representation Learning)

## Executive Summary

This review identified **one critical data leakage flaw** and several other significant issues that compromise the validity of the experimental results. The most severe issue is a **within-subject train/test split** that allows the model to learn subject-specific patterns, leading to inflated performance estimates and poor generalization to new subjects.

---

## Critical Issues

### 🔴 CRITICAL: Within-Subject Data Leakage in Train/Test Split

**Location:** `codes/datasets.py:14-42` (function `train_test_split_felicity_kfold`)

**Issue:**
The k-fold cross-validation splits each subject's data temporally, placing some time windows from the same subject in the training set and other windows in the test set. This is a fundamental experimental design flaw.

**Evidence:**
```python
def _get_train_test_index(data, kfold):
    np.random.seed(999999)
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
```

**Problem Analysis:**
1. The code iterates through each person/subject (`for k in person`)
2. For each person, it takes their data windows and splits them into folds
3. Some windows from person X go to training, others go to testing
4. **Result:** Train and test sets contain data from the SAME subjects

**Impact:**
- **Severe overestimation of model performance** - The model learns subject-specific ECG characteristics (heart rate patterns, morphology, etc.)
- **Poor generalization** - The model cannot generalize to truly new, unseen subjects
- **Invalid scientific conclusions** - Published results likely overestimate real-world performance
- **Violates fundamental ML principle** - Test data should represent the target distribution (new patients)

**Why This is Wrong:**
In medical applications, the goal is typically to predict stress/health outcomes for **new patients** not seen during training. The current split tests only whether the model can recognize patterns in different time segments from the same patients it was trained on.

**Correct Approach:**
Should use **between-subject** (subject-wise) cross-validation:
- Fold 1: Train on subjects 1-40, test on subjects 41-50
- Fold 2: Train on subjects 1-30 + 41-50, test on subjects 31-40
- etc.

---

## High-Priority Issues

### ⚠️ No Feature Normalization for Downstream Tasks

**Location:** `codes/train.py:274-284`, `codes/model.py:99-200`

**Issue:**
Extracted features from the self-supervised model are fed directly into downstream classifiers/regressors without normalization or standardization.

**Evidence:**
```python
# train.py:274-275
x_tr_feature = utils.extract_feature(x_original = train_ECG, ...)
x_te_feature = utils.extract_feature(x_original = test_ECG, ...)

# model.py:124-137 - Features used directly
model.add(keras.layers.Dense(hidden_nodes, input_dim=input_dimension, activation='relu', ...))
model.fit(x_tr_feature, y_tr, epochs=epoch_super, ...)
```

**Impact:**
- Suboptimal performance of downstream tasks
- Potential numerical instability
- Different feature scales can bias the model

**Recommended Fix:**
Add feature standardization using training set statistics:
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_tr_feature = scaler.fit_transform(x_tr_feature)
x_te_feature = scaler.transform(x_te_feature)
```

---

### ⚠️ Hardcoded Random Seeds Reduce Robustness

**Location:** `codes/datasets.py:17`, `codes/datasets.py:46`, `codes/datasets.py:73`

**Issue:**
Multiple hardcoded random seeds throughout the codebase:
- Line 17: `np.random.seed(999999)`
- Line 46: `np.random.seed(12223)`
- Line 73: `np.random.seed(12223)`

**Impact:**
- Results cannot be validated with different random seeds
- Potential for "cherry-picking" by running with different seeds
- Standard practice is to report mean ± std across multiple random seeds

**Recommended Fix:**
- Make random seeds configurable parameters
- Run experiments with multiple seeds (e.g., 5-10 different seeds)
- Report mean and standard deviation of metrics

---

### ⚠️ Inefficient Feature Extraction

**Location:** `codes/utils.py:272-284`

**Issue:**
Feature extraction uses modulo indexing which processes the same samples multiple times when `length` is not divisible by `batch_super`.

**Evidence:**
```python
def extract_feature(x_original, featureset_size, batch_super, input_tensor, isTrain, drop_out, extract_layer, sess):
    feature_set = np.zeros((1, featureset_size), dtype = int)
    length = np.shape(x_original)[0]
    steps = length //batch_super +1
    for j in range(steps):
        signal_batch = x_original[np.mod(np.arange(j*batch_super,(j+1)*batch_super), length)]
        # ... process batch
    x_feature = feature_set[1:length+1] ## resizing to the original signal
```

**Impact:**
- Duplicate processing of samples
- Incorrect feature ordering
- Potential for subtle bugs

**Recommended Fix:**
```python
for j in range(0, length, batch_super):
    end_idx = min(j + batch_super, length)
    signal_batch = x_original[j:end_idx]
    # Handle last batch if needed
```

---

## Medium-Priority Issues

### ⚙️ Deprecated TensorFlow 1.x API

**Location:** Throughout codebase

**Issue:**
Code uses TensorFlow 1.14.0 with deprecated APIs and session-based execution.

**Impact:**
- Cannot leverage modern TensorFlow features
- Difficult to maintain and extend
- Security vulnerabilities in old TF versions
- Poor performance compared to TF 2.x

**Recommended Fix:**
Migrate to TensorFlow 2.x with eager execution and Keras functional API.

---

### ⚙️ Batch Normalization in Test Mode Issue

**Location:** `codes/model.py:15-16`

**Issue:**
Batch normalization layers use training statistics during inference if not properly managed.

**Evidence:**
```python
if batch_norm:
    conv = tf.layers.batch_normalization(conv, training=isTraining, name = name, reuse=reuse)
```

**Impact:**
- May cause different behavior between training and testing
- The `training=isTraining` flag should properly handle this, but needs verification

**Verification Needed:**
Ensure UPDATE_OPS are properly handled (they are at `train.py:108`)

---

### ⚙️ Missing Input Validation

**Location:** Various functions throughout

**Issue:**
Functions don't validate inputs (e.g., array shapes, value ranges).

**Examples:**
- `make_window()` doesn't check if signal length >= window_size
- `make_batch()` doesn't validate transformation parameters
- No checks for NaN or infinite values

**Impact:**
- Silent failures or cryptic error messages
- Difficult debugging
- Potential for incorrect results

---

## Code Quality Issues

### 📝 Inconsistent Normalization Implementation

**Location:** `codes/felicity.py:116-123`, `codes/felicity.py:201-208`

**Issue:**
Two identical `_normalize()` functions defined as nested functions (code duplication).

**Recommendation:**
Move to a shared utility function:
```python
# In utils.py
def robust_normalize(x, lower_percentile=0.025, upper_percentile=0.975):
    """Z-score normalization using robust statistics (trimmed std)"""
    temp = np.sort(x)
    lower_idx = int(lower_percentile * temp.shape[0])
    upper_idx = int(upper_percentile * temp.shape[0])
    x_std = np.std(temp[lower_idx:upper_idx])
    x_mean = np.mean(temp)
    return (x - x_mean) / x_std
```

---

### 📝 Unclear Variable Names

**Location:** Various

**Issues:**
- `te_` vs `test_` prefix inconsistency
- `tr_` vs `train_` prefix inconsistency
- Single letter loop variables in complex contexts (`k`, `i`, `j`)

---

### 📝 Missing Documentation

**Issues:**
- No docstrings for most functions
- Complex algorithms (k-fold splitting) lack explanatory comments
- No high-level documentation of data format
- Transformation parameters lack justification

---

## Positive Aspects

✅ **Good:** Subject-level normalization (lines 151, 227 in `felicity.py`) normalizes each subject independently, avoiding information leakage across subjects.

✅ **Good:** Self-supervised model training uses only training data (lines 198-217 in `train.py`), with test data only used for evaluation.

✅ **Good:** Proper use of dropout during training and disabling during testing.

✅ **Good:** Data augmentation through signal transformations (noise, scaling, permutation, time warping).

---

## Recommendations Priority List

### Immediate (Critical)

1. **Fix the train/test split** - Implement subject-wise cross-validation
   - This invalidates current published results
   - Must be fixed before any claims of model performance

### High Priority

2. **Add feature normalization** for downstream tasks
3. **Make random seeds configurable** and run multiple trials
4. **Fix feature extraction** to avoid duplicate processing

### Medium Priority

5. **Migrate to TensorFlow 2.x** for maintainability
6. **Add comprehensive input validation**
7. **Add proper logging** and experiment tracking (e.g., MLflow, Weights & Biases)

### Low Priority (Code Quality)

8. **Refactor duplicate code**
9. **Add docstrings and documentation**
10. **Improve variable naming**
11. **Add unit tests**

---

## Suggested Test Scenarios

After fixes, validate with:

1. **Sanity check:** Train on subject A, test on subject A → should get very high performance
2. **Proper split:** Train on subjects A-D, test on subject E → realistic performance
3. **Different seeds:** Run with 10 different random seeds, report mean ± std
4. **Ablation study:** Compare within-subject vs between-subject splits to quantify the leakage effect

---

## Conclusion

The primary concern about data leakage is **confirmed and critical**. The within-subject train/test split is a fundamental flaw that invalidates the experimental results. All published performance metrics likely significantly overestimate the model's ability to generalize to new patients.

**Recommendation:** Do not use this code for production or clinical applications until the critical data leakage issue is fixed and the model is re-evaluated with proper between-subject cross-validation.

---

## Files Analyzed

- `codes/train.py` - Main training script
- `codes/datasets.py` - Data loading and splitting (CRITICAL ISSUE HERE)
- `codes/model.py` - Model architectures
- `codes/utils.py` - Utility functions
- `codes/felicity.py` - Data preprocessing and extraction
- `codes/preprocessing.py` - Signal preprocessing
- `codes/signal_transformation_task.py` - Data augmentation

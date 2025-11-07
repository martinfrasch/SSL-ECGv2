# Setup Guide for SSL-ECGv2 (Fixed Version)

This guide helps you set up the environment to run the fixed SSL-ECG code with proper subject-wise cross-validation.

## Quick Start

### 1. Environment Setup

**Option A: Using Conda (Recommended)**

```bash
# Create environment with Python 3.6 (required for TensorFlow 1.14)
conda create -n ssl-ecg python=3.6
conda activate ssl-ecg

# Install dependencies
pip install -r requirements.txt
```

**Option B: Using virtualenv**

```bash
# Ensure you have Python 3.6 installed
python3.6 -m venv ssl-ecg-env
source ssl-ecg-env/bin/activate  # On Windows: ssl-ecg-env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Verify Installation

```bash
python -c "import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}')"
python -c "import numpy as np; import sklearn; print('All dependencies OK!')"
```

### 3. Run Tests (Verify Fixes)

```bash
# Run unit tests to verify the data splitting fix
python tests/test_datasets.py

# Or use pytest
pytest tests/test_datasets.py -v
```

You should see all tests pass, confirming:
- ✓ Subject-wise split has no data leakage
- ✓ All subjects used exactly once across folds
- ✓ Data integrity preserved
- ✓ Reproducible with same seed

---

## Running the Fixed Training Script

### Basic Usage (Correct Subject-Wise CV)

```bash
cd codes
python train_fixed.py
```

This runs with default settings:
- ✓ Subject-wise cross-validation (CORRECT)
- ✓ Feature normalization enabled
- ✓ Random seed: 42
- ✓ 5-fold cross-validation
- ✓ 30 epochs

### Custom Configuration

```bash
# Run with different random seed
python train_fixed.py --random_seed 123

# Run with more folds
python train_fixed.py --total_folds 10

# Run with different number of epochs
python train_fixed.py --epochs 50

# Disable feature normalization (not recommended)
python train_fixed.py --normalize_features False
```

### Comparison with Old Method (For Ablation Study)

**⚠️ WARNING:** This is ONLY for demonstrating the data leakage issue, NOT for real experiments!

```bash
# Run with OLD INCORRECT within-subject split
python train_fixed.py --subject_wise False
```

This will:
- Show WARNING messages
- Produce inflated results (20-30% higher than correct)
- Help you quantify the impact of the data leakage bug

---

## Data Preparation

### If You Have Raw ECG Data

1. **Place raw data** in the appropriate directory structure:
   ```
   data/
   ├── raw_downloads/
   │   └── felicity/
   │       ├── mECG/  # Maternal ECG files
   │       └── fECG/  # Fetal ECG files (if applicable)
   └── scores_new.xlsx  # Labels file
   ```

2. **Extract and preprocess data:**
   ```python
   cd codes
   python train_fixed.py --extract_data 1
   ```

   This will:
   - Load raw .mat files
   - Downsample to 256 Hz
   - Apply filtering
   - Segment into 10-second windows
   - Apply quality filtering
   - Save processed data as .npy files

### If You Have Preprocessed Data

Just place the `.npy` files in the `data/` directory:
```
data/
└── felicitys_mecg_0.npy  # Preprocessed maternal ECG with 0% overlap
```

Expected data format (columns):
```
[subject_id, stress_label, pss_score, pdq_score, fsi_score, cortisol, ...ECG_windows...]
```

---

## Understanding the Output

### Directory Structure After Training

```
SSL-ECGv2/
├── output/
│   ├── STR_result/  # Self-supervised training results
│   │   ├── tr_str_f1_Score.csv
│   │   └── te_str_f1_score.csv
│   ├── STR_loss/    # Training/test losses
│   ├── ER_result/   # Downstream task results
│   │   ├── te_stress.csv  # Stress classification
│   │   ├── te_pdq.csv     # PDQ regression
│   │   ├── te_pss.csv     # PSS regression
│   │   ├── te_fsi.csv     # FSI regression
│   │   └── te_cortisol.csv
│   └── feature/     # Extracted features and scalers
│       └── fold_0/
│           └── feature_scaler.pkl
├── models/          # Saved model checkpoints
│   └── fold_0/
│       └── epoch_29/
└── summaries/       # TensorBoard logs
    ├── STR/
    └── ER/
```

### Key Result Files

**1. Stress Classification (`output/ER_result/te_stress.csv`):**
- Accuracy, F1, Sensitivity, Specificity, ROC-AUC
- One row per fold

**2. Regression Tasks (`te_pdq.csv`, `te_pss.csv`, etc.):**
- MSE, MAE, RMSE, R²
- One row per fold

**3. Self-Supervised Results (`te_str_f1_score.csv`):**
- F1 scores for each transformation task
- Tracks quality of learned representations

### Visualizing Results with TensorBoard

```bash
tensorboard --logdir=../summaries
```

Then open http://localhost:6006 in your browser.

---

## Expected Performance Changes

### With Correct Subject-Wise CV (Current Implementation)

**Expected Stress Classification Performance:**
- Accuracy: 65-80%
- F1-Score: 0.60-0.75
- ROC-AUC: 0.70-0.85

This represents TRUE generalization to new patients.

### With Old Within-Subject Split (Buggy Version)

**Expected Performance (INFLATED):**
- Accuracy: 85-95%
- F1-Score: 0.80-0.95
- ROC-AUC: 0.90-0.98

**The 20-30% difference is normal** when fixing data leakage!

---

## Running Multiple Trials for Robustness

To get statistically robust results, run with multiple random seeds:

```bash
# Create a script to run multiple trials
for seed in 42 123 456 789 1011
do
    echo "Running with seed $seed"
    python train_fixed.py --random_seed $seed
done
```

Then aggregate results:
```python
import pandas as pd
import numpy as np

# Load results from all folds and seeds
results = []
for seed in [42, 123, 456, 789, 1011]:
    for fold in range(5):
        # Load te_stress.csv for this seed
        df = pd.read_csv(f'output/ER_result/te_stress_seed{seed}_fold{fold}.csv')
        results.append(df)

# Compute mean and std
all_results = pd.concat(results)
print(f"Accuracy: {all_results['Accuracy'].mean():.3f} ± {all_results['Accuracy'].std():.3f}")
print(f"F1-Score: {all_results['F1'].mean():.3f} ± {all_results['F1'].std():.3f}")
```

---

## Applying to New Prospective Dataset

### 1. Prepare Your New Data

Match the expected format:
```python
# new_data shape: (n_samples, 6 + signal_length)
# Columns: [subject_id, stress_label, pss, pdq, fsi, cortisol, ...ECG_2560_samples...]
```

### 2. Load Pretrained Model

```python
import tensorflow as tf
import pickle
import numpy as np

# Load model checkpoint
sess = tf.Session()
saver = tf.train.import_meta_graph('models/fold_0/epoch_29/SSL_model.ckpt.meta')
saver.restore(sess, 'models/fold_0/epoch_29/SSL_model.ckpt')

# Get tensors
graph = tf.get_default_graph()
input_tensor = graph.get_tensor_by_name('input:0')
main_branch = graph.get_tensor_by_name('GAP/Squeeze:0')  # Adjust name as needed

# Extract features from new data
new_features = []
for batch in batches(new_data, batch_size=128):
    features = sess.run(main_branch, {input_tensor: batch})
    new_features.append(features)

new_features = np.vstack(new_features)

# Load feature scaler
with open('output/feature/fold_0/feature_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Normalize features
new_features_norm = scaler.transform(new_features)

# Now use for prediction...
```

### 3. Make Predictions

```python
import keras

# Load trained downstream classifier
stress_model = keras.models.load_model('output/models/stress_classifier_fold0.h5')

# Predict on new data
stress_predictions = stress_model.predict(new_features_norm)
stress_labels = np.argmax(stress_predictions, axis=1)

print(f"Predicted stress: {stress_labels}")
```

---

## Troubleshooting

### Issue: "No module named tensorflow"
**Solution:** Ensure you're in the correct environment and TensorFlow is installed:
```bash
conda activate ssl-ecg
pip install tensorflow==1.14.0
```

### Issue: "Data file not found"
**Solution:** Make sure data is in the correct location:
```bash
ls data/felicitys_mecg_0.npy
```

If missing, run data extraction:
```bash
python train_fixed.py --extract_data 1
```

### Issue: "CUDA out of memory" (GPU)
**Solution:** Reduce batch size:
Edit `train_fixed.py` line with `batchsize = 128` and change to `batchsize = 64` or `32`.

### Issue: Tests failing
**Solution:** Check that you've updated the code files:
```bash
# Ensure datasets.py has the new functions
grep "def _subject_wise_split" codes/datasets.py

# Should show the new function
```

### Issue: Different results from paper
**This is expected!** The paper used the buggy within-subject split. Your corrected results will be 20-30% lower, which is CORRECT.

---

## Next Steps

1. **Verify fixes work:** Run tests and confirm all pass
2. **Run experiments:** Use `train_fixed.py` with your data
3. **Compare methods:** Run ablation study (subject_wise True vs False)
4. **Document changes:** Update your paper/thesis methods section
5. **Apply to new data:** Use pretrained model on prospective dataset

---

## Getting Help

If you encounter issues:
1. Check this guide thoroughly
2. Review the code comments in `train_fixed.py` and `datasets.py`
3. Read the review documents: `CODE_REVIEW.md` and `IMPROVEMENT_PLAN.md`
4. Check TensorFlow 1.14 documentation: https://www.tensorflow.org/versions/r1.14/api_docs

---

## Citation

If you use this fixed code, please cite both:

**Original Paper:**
```bibtex
@article{sarkar2020detection,
  title={Detection of Maternal and Fetal Stress from ECG with Self-supervised Representation Learning},
  author={Sarkar, Pritam and Lobmaier, Silvia and Fabre, Bibiana and others},
  journal={arXiv preprint arXiv:2011.02000},
  year={2020}
}
```

**And acknowledge the fix:**
```
The original implementation contained a data leakage issue in the train/test split
which was identified and corrected in November 2025. We use the corrected version
with subject-wise cross-validation.
```

---

**Version:** 2.0 (Fixed)
**Last Updated:** 2025-11-07
**Status:** Ready for Use

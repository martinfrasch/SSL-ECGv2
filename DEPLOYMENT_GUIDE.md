# SSL-ECG Deployment Guide

Complete guide for training and deploying SSL-ECG models on Google Cloud Platform from your MacOS machine.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Start](#quick-start)
3. [Training on GCP](#training-on-gcp)
4. [Inference on GCP](#inference-on-gcp)
5. [Local Training & Inference](#local-training--inference)
6. [Cost Estimates](#cost-estimates)
7. [Troubleshooting](#troubleshooting)
8. [Advanced Usage](#advanced-usage)

---

## Prerequisites

### MacOS Setup

1. **Install Google Cloud SDK**
   ```bash
   # Download and install from: https://cloud.google.com/sdk/docs/install
   # Or use Homebrew:
   brew install --cask google-cloud-sdk
   ```

2. **Authenticate with GCP**
   ```bash
   gcloud auth login
   gcloud config set project YOUR_PROJECT_ID
   ```

3. **Enable Required APIs**
   ```bash
   gcloud services enable compute.googleapis.com
   ```

4. **Verify gcloud is working**
   ```bash
   gcloud compute zones list
   ```

### Data Preparation

Your ECG data must be in NumPy format (`.npy` file):

**Format for Training:**
```python
# Shape: (n_windows, 2567)
# Columns:
#   [0] = subject_id (integer, 1-N)
#   [1] = stress_label (0 or 1)
#   [2] = PSS score (float)
#   [3] = PDQ score (float)
#   [4] = FSI score (float)
#   [5] = cortisol level (float)
#   [6] = unused (0)
#   [7:2567] = ECG signal (2560 samples = 10 sec at 256 Hz)
```

**Format for Inference:**
```python
# Shape: (n_windows, 2560) or (n_windows, 2567)
# Just ECG signals (2560 samples each) or with metadata
```

**Example - Prepare Training Data:**
```python
import numpy as np

# Assuming you have:
# - subject_ids: array of subject IDs
# - stress_labels: array of 0/1 labels
# - pss_scores: array of PSS scores
# - ecg_windows: array of ECG signals (n_windows, 2560)

data = np.hstack([
    subject_ids.reshape(-1, 1),
    stress_labels.reshape(-1, 1),
    pss_scores.reshape(-1, 1),
    pdq_scores.reshape(-1, 1),
    fsi_scores.reshape(-1, 1),
    cortisol_levels.reshape(-1, 1),
    np.zeros((len(subject_ids), 1)),
    ecg_windows
])

np.save('my_ecg_data.npy', data)
print(f"Saved data shape: {data.shape}")
```

**Example - Prepare Inference Data:**
```python
import numpy as np

# Assuming you have ECG windows (n_windows, 2560)
ecg_windows = ...  # Your ECG data

np.save('new_ecgs.npy', ecg_windows)
print(f"Saved data shape: {ecg_windows.shape}")
```

---

## Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/martinfrasch/SSL-ECGv2.git
cd SSL-ECGv2
git checkout claude/tf2-migration-011CUtwG7jQk8UkVsFwcjUs3
```

### 2. Train Model on GCP
```bash
bash scripts/deploy_to_gcp.sh \
    --instance-name my-ssl-ecg-training \
    --zone us-central1-a \
    --gpu-type nvidia-tesla-v100 \
    --data-dir /path/to/your/ecg/data \
    --kfold 0 \
    --total-fold 5 \
    --epochs 30
```

This will:
- ✓ Create a GCP VM with V100 GPU
- ✓ Upload your data
- ✓ Train the model
- ✓ Download trained model to `./trained_models/`
- ✓ Stop the VM (to save costs)

**Expected time:** 6-8 hours
**Expected cost:** $13-17

### 3. Run Inference on New ECGs
```bash
bash scripts/inference_on_gcp.sh \
    --model-dir ./trained_models/ssl_ecg_fold0_* \
    --input-data /path/to/new_ecgs.npy \
    --output-file predictions.npy
```

This will:
- ✓ Create/reuse GCP VM
- ✓ Upload your model and new ECGs
- ✓ Extract features
- ✓ Download results to `./inference_results/predictions.npy`

**Expected time:** 5-10 minutes
**Expected cost:** $0.05-0.10

---

## Training on GCP

### Basic Training

Train a single fold with default settings:

```bash
bash scripts/deploy_to_gcp.sh \
    --data-dir ~/my_ecg_data \
    --kfold 0 \
    --total-fold 5
```

### Full 5-Fold Cross-Validation

Train all 5 folds sequentially:

```bash
# Create a script to run all folds
for fold in {0..4}; do
    bash scripts/deploy_to_gcp.sh \
        --instance-name ssl-ecg-fold$fold \
        --data-dir ~/my_ecg_data \
        --kfold $fold \
        --total-fold 5 \
        --output-dir ./trained_models
done
```

### Advanced Training Options

```bash
bash scripts/deploy_to_gcp.sh \
    --instance-name my-training-vm \
    --zone us-west1-b \
    --gpu-type nvidia-tesla-v100 \
    --data-dir ~/ecg_data \
    --kfold 0 \
    --total-fold 5 \
    --epochs 50 \
    --batch-size 256 \
    --keep-vm \
    --output-dir ./my_models
```

**Options:**
- `--instance-name`: VM name (default: `ssl-ecg-training`)
- `--zone`: GCP zone (default: `us-central1-a`)
  - `us-central1-a`: Iowa (cheapest)
  - `us-west1-b`: Oregon
  - `us-east1-b`: South Carolina
  - `europe-west4-a`: Netherlands
- `--gpu-type`: GPU type
  - `nvidia-tesla-v100`: Fastest (recommended) - $1.90/hr
  - `nvidia-tesla-t4`: Cheaper but slower - $0.50/hr
- `--data-dir`: Local path to your ECG data
- `--kfold`: Current fold (0-based)
- `--total-fold`: Total number of folds
- `--epochs`: Training epochs (default: 30)
- `--batch-size`: Batch size (default: 256)
- `--keep-vm`: Keep VM running after training (default: stops VM)
- `--output-dir`: Local output directory (default: `./trained_models`)

### Training Output

After training completes, you'll find:

```
trained_models/
└── ssl_ecg_fold0_20251112_143022/
    ├── exported_model/
    │   ├── saved_model/         ← Use this for inference
    │   ├── model.h5
    │   └── model_weights.h5
    ├── checkpoints/
    │   └── best_model.h5
    ├── logs/                    ← TensorBoard logs
    ├── results/
    │   ├── train_features.npy
    │   └── test_features.npy
    ├── config.json
    ├── deployment_metadata.json
    ├── feature_scaler.pkl       ← Important for inference!
    ├── model_architecture.json
    ├── subject_split.json
    ├── training_history.json
    └── README.md
```

### View Training Progress

```bash
# View TensorBoard logs locally
tensorboard --logdir=trained_models/ssl_ecg_fold0_*/logs

# Open browser to: http://localhost:6006
```

---

## Inference on GCP

### Basic Inference (Extract Features Only)

```bash
bash scripts/inference_on_gcp.sh \
    --model-dir trained_models/ssl_ecg_fold0_20251112_143022 \
    --input-data new_ecgs.npy \
    --output-file features.npy
```

**Output:** `inference_results/features.npy` (shape: `[n_samples, 256]`)

### Inference with Downstream Task

If you have a trained downstream model (e.g., stress classifier):

```bash
bash scripts/inference_on_gcp.sh \
    --model-dir trained_models/ssl_ecg_fold0_20251112_143022 \
    --input-data new_ecgs.npy \
    --output-file stress_predictions.csv \
    --task stress \
    --downstream-model trained_models/stress_classifier.h5
```

**Output:** `inference_results/stress_predictions.csv`
```csv
sample_id,predicted_class,stress_probability,low_stress_probability,high_stress_probability
0,1,0.87,0.13,0.87
1,0,0.23,0.77,0.23
...
```

### Inference Options

```bash
bash scripts/inference_on_gcp.sh \
    --instance-name my-inference-vm \
    --zone us-central1-a \
    --model-dir trained_models/ssl_ecg_fold0_20251112_143022 \
    --input-data new_ecgs.npy \
    --output-file results.npy \
    --batch-size 512 \
    --task stress \
    --downstream-model stress_model.h5 \
    --keep-vm
```

**Options:**
- `--model-dir`: Path to trained model directory
- `--input-data`: Path to new ECG data (.npy file)
- `--output-file`: Output filename
- `--task`: Downstream task (`stress`, `pss`, `pdq`, `fsi`, `cortisol`)
- `--downstream-model`: Path to downstream task model
- `--batch-size`: Batch size for inference (default: 256)
- `--keep-vm`: Keep VM running after inference

---

## Local Training & Inference

If you have a local GPU or want to test on CPU:

### Local Training

```bash
# 1. Setup environment
conda create -n ssl-ecg python=3.10
conda activate ssl-ecg
pip install -r requirements_tf2.txt

# 2. Run training
python codes/train_tf2_production.py \
    --data_folder ~/ecg_data \
    --kfold 0 \
    --total_fold 5 \
    --output_dir trained_models \
    --epochs 30 \
    --batch_size 128 \
    --save_model True \
    --verbose 2
```

### Local Inference

```bash
python codes/inference_tf2.py \
    --model_dir trained_models/ssl_ecg_fold0_20251112_143022 \
    --input_data new_ecgs.npy \
    --output_file features.npy \
    --batch_size 256 \
    --verbose 1
```

---

## Cost Estimates

### Training Costs (Single Fold)

| GPU Type | Time | Hourly Rate | Total Cost |
|----------|------|-------------|------------|
| V100 (recommended) | 6-8 hrs | $1.90/hr | $11-15 |
| T4 (budget) | 10-12 hrs | $0.50/hr | $5-6 |

**5-Fold CV Total:** $55-75 (V100) or $25-30 (T4)

### Inference Costs

| GPU Type | Time (1000 samples) | Hourly Rate | Total Cost |
|----------|---------------------|-------------|------------|
| T4 (recommended for inference) | 5 min | $0.60/hr | $0.05 |
| V100 (overkill) | 3 min | $2.10/hr | $0.11 |

**Tip:** Use T4 for inference, V100 only for training.

### Storage Costs

- Trained model: ~1 GB per fold → $0.02/month per fold
- Total for 5 folds: ~$0.10/month

**Recommendation:** Download models to your Mac and delete from GCP after training.

---

## Troubleshooting

### Issue 1: "Quota 'GPUS_ALL_REGIONS' exceeded"

**Solution:**
```bash
# Request quota increase in GCP Console
# Or use a different zone
bash scripts/deploy_to_gcp.sh --zone us-west1-b ...
```

### Issue 2: "Instance not found" during inference

**Solution:**
```bash
# Check if VM was deleted
gcloud compute instances list

# Create new VM if needed (script will auto-create)
bash scripts/inference_on_gcp.sh ...
```

### Issue 3: "Data file not found" on GCP

**Solution:**
```bash
# Ensure data directory exists locally
ls -la ~/my_ecg_data/

# Check if file was uploaded
gcloud compute ssh my-vm --zone=us-central1-a
ls -la ~/ecg_data/
```

### Issue 4: Training times out

**Solution:**
```bash
# SSH into VM and check status
gcloud compute ssh my-training-vm --zone=us-central1-a
cd SSL-ECGv2
ls -la trained_models/

# If training is still running, attach to see progress
# Training script saves checkpoints, so you won't lose progress
```

### Issue 5: "Failed to load model"

**Solution:**
```python
# Check model directory structure
import os
model_dir = 'trained_models/ssl_ecg_fold0_20251112_143022'
print(os.listdir(os.path.join(model_dir, 'exported_model')))

# Should see: ['saved_model', 'model.h5', 'model_weights.h5']

# If saved_model is missing, use H5 format:
python codes/inference_tf2.py --model_format h5 ...
```

### Issue 6: Out of Memory during training

**Solution:**
```bash
# Reduce batch size
bash scripts/deploy_to_gcp.sh --batch-size 64 ...

# Or use larger VM
bash scripts/deploy_to_gcp.sh --machine-type n1-highmem-8 ...
```

---

## Advanced Usage

### 1. Parallel Training (Multiple Folds Simultaneously)

Train multiple folds in parallel using different VMs:

```bash
# Terminal 1
bash scripts/deploy_to_gcp.sh --instance-name fold0 --kfold 0 --data-dir ~/data &

# Terminal 2
bash scripts/deploy_to_gcp.sh --instance-name fold1 --kfold 1 --data-dir ~/data &

# Terminal 3
bash scripts/deploy_to_gcp.sh --instance-name fold2 --kfold 2 --data-dir ~/data &

# Etc.
```

**Cost:** Same total, but completes 5x faster!

### 2. Hyperparameter Tuning

Create a sweep script:

```bash
#!/bin/bash
# hyperparam_sweep.sh

LEARNING_RATES=(0.001 0.0005 0.0001)
BATCH_SIZES=(128 256 512)

for lr in "${LEARNING_RATES[@]}"; do
    for bs in "${BATCH_SIZES[@]}"; do
        echo "Training with lr=$lr, batch_size=$bs"

        bash scripts/deploy_to_gcp.sh \
            --instance-name hparam-lr${lr}-bs${bs} \
            --data-dir ~/data \
            --learning-rate $lr \
            --batch-size $bs \
            --output-dir ./hparam_results/lr${lr}_bs${bs}
    done
done
```

**Note:** You'll need to modify `deploy_to_gcp.sh` to pass `--learning-rate` to the training script.

### 3. Transfer Learning on New Dataset

Use a trained model as starting point:

```python
# train_transfer_learning.py
import tensorflow as tf

# Load pretrained model
base_model = tf.keras.models.load_model('trained_models/ssl_ecg_fold0_*/exported_model/saved_model')

# Freeze base layers
for layer in base_model.layers[:-5]:  # Freeze all but last 5 layers
    layer.trainable = False

# Add your custom head
# ... (your downstream task)

# Fine-tune on new data
model.compile(...)
model.fit(new_data, ...)
```

### 4. Batch Inference on Large Datasets

For very large datasets (>100k samples):

```python
# batch_inference.py
import numpy as np
import tensorflow as tf
from tqdm import tqdm

model = tf.keras.models.load_model('trained_models/.../exported_model/saved_model')

# Process in chunks to avoid memory issues
input_data = np.load('huge_dataset.npy', mmap_mode='r')  # Memory-mapped
n_samples = input_data.shape[0]
chunk_size = 10000

features_list = []

for i in tqdm(range(0, n_samples, chunk_size)):
    chunk = input_data[i:i+chunk_size]
    chunk = chunk.reshape(-1, 2560, 1)

    outputs = model.predict(chunk, batch_size=256, verbose=0)
    features_list.append(outputs['features'])

    # Save intermediate results
    if (i // chunk_size) % 10 == 0:
        np.save(f'features_checkpoint_{i}.npy', np.vstack(features_list))

# Final save
features = np.vstack(features_list)
np.save('all_features.npy', features)
```

### 5. Model Ensembling

Combine predictions from multiple folds:

```python
# ensemble_inference.py
import numpy as np
import tensorflow as tf

model_dirs = [
    'trained_models/ssl_ecg_fold0_*',
    'trained_models/ssl_ecg_fold1_*',
    'trained_models/ssl_ecg_fold2_*',
    'trained_models/ssl_ecg_fold3_*',
    'trained_models/ssl_ecg_fold4_*',
]

models = [tf.keras.models.load_model(f'{d}/exported_model/saved_model')
          for d in model_dirs]

# Extract features from all models
ecg_data = np.load('test_data.npy').reshape(-1, 2560, 1)

features_all = []
for model in models:
    outputs = model.predict(ecg_data)
    features_all.append(outputs['features'])

# Average features (or concatenate)
features_ensemble = np.mean(features_all, axis=0)  # Shape: (n_samples, 256)

# Now use ensemble features for downstream task
```

---

## Best Practices

### 1. Data Organization

```
my_project/
├── raw_data/
│   ├── subject_001.csv
│   ├── subject_002.csv
│   └── ...
├── processed_data/
│   ├── training_data.npy
│   └── inference_data.npy
├── trained_models/
│   ├── ssl_ecg_fold0_*/
│   ├── ssl_ecg_fold1_*/
│   └── ...
├── inference_results/
│   └── predictions_*.csv
└── scripts/
    └── preprocess_data.py
```

### 2. Version Control

```bash
# Track which model version was used for which results
echo "ssl_ecg_fold0_20251112_143022" > results_v1/model_version.txt

# Save configuration
cp trained_models/ssl_ecg_fold0_*/config.json results_v1/
```

### 3. Cost Optimization

- ✓ Always **stop VMs** after training/inference
- ✓ Use **Preemptible VMs** for non-critical jobs (70% cheaper)
- ✓ Use **T4 GPUs** for inference (3x cheaper than V100)
- ✓ **Download models** to local storage, delete from GCP
- ✓ Set **billing alerts** in GCP Console

### 4. Reproducibility

Always specify random seeds:

```bash
python codes/train_tf2_production.py \
    --random_seed 42 \
    --data_split_seed 42 \
    ...
```

---

## Summary Commands

```bash
# 1. Setup
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# 2. Train model
bash scripts/deploy_to_gcp.sh \
    --data-dir ~/my_ecg_data \
    --kfold 0 \
    --total-fold 5

# 3. Run inference
bash scripts/inference_on_gcp.sh \
    --model-dir trained_models/ssl_ecg_fold0_* \
    --input-data new_ecgs.npy \
    --output-file predictions.npy

# 4. Clean up
gcloud compute instances delete my-training-vm --zone=us-central1-a
```

---

## Support

For issues or questions:
1. Check [Troubleshooting](#troubleshooting) section above
2. Review model README: `trained_models/YOUR_MODEL/README.md`
3. Check training logs: `trained_models/YOUR_MODEL/training_history.json`
4. Review GitHub issues: https://github.com/martinfrasch/SSL-ECGv2/issues

---

**Happy Training! 🚀**

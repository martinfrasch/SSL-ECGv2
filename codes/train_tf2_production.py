"""
train_tf2_production.py - Production-ready TF2 training with comprehensive model saving

This script extends train_tf2.py with:
- Complete model saving (full model + weights + architecture)
- Structured output for deployment
- Export for inference
- Comprehensive logging
- Progress tracking
- Model versioning

Usage:
    python train_tf2_production.py \
        --data_folder ~/ecg_data \
        --kfold 0 \
        --total_fold 5 \
        --output_dir trained_models \
        --save_model True

Author: Claude (2025-11-12)
Based on: train_tf2.py
"""

import os
import argparse
import warnings
import json
import pickle
from datetime import datetime
warnings.filterwarnings('ignore')

# Get current directory
dirname = os.path.dirname(os.path.abspath(__file__))
os.chdir(dirname)

import tensorflow as tf
import numpy as np
from tqdm import tqdm, trange
from pathlib import Path

# Import our modules
import model_tf2 as model
import utils_tf2 as utils
import datasets
import felicity

# Parse arguments
parser = argparse.ArgumentParser(description='SSL-ECG Production Training with TensorFlow 2.x')

# Data arguments
parser.add_argument('--data_folder', type=str, required=True,
                    help='Path to ECG data folder')
parser.add_argument('--data_tag', type=str, default='mecg',
                    help='Type of ECG data: mecg or aecg')
parser.add_argument('--extract_data', type=int, default=0,
                    help='Extract data from raw files (1) or use existing (0)')

# Cross-validation arguments
parser.add_argument('--kfold', type=int, default=0,
                    help='Current fold index (0-based)')
parser.add_argument('--total_fold', type=int, default=5,
                    help='Total number of CV folds')
parser.add_argument('--subject_wise', type=lambda x: str(x).lower() == 'true', default=True,
                    help='Use subject-wise CV (True=CORRECT, False=INCORRECT)')
parser.add_argument('--validate_split', type=lambda x: str(x).lower() == 'true', default=True,
                    help='Validate no subject overlap')

# Training arguments
parser.add_argument('--epochs', type=int, default=30,
                    help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=128,
                    help='Batch size for training')
parser.add_argument('--learning_rate', type=float, default=0.001,
                    help='Initial learning rate')

# Model saving arguments
parser.add_argument('--output_dir', type=str, default='trained_models',
                    help='Output directory for saved models')
parser.add_argument('--save_model', type=lambda x: str(x).lower() == 'true', default=True,
                    help='Save trained model')
parser.add_argument('--model_name', type=str, default=None,
                    help='Custom model name (default: auto-generated)')

# Feature processing
parser.add_argument('--normalize_features', type=lambda x: str(x).lower() == 'true', default=True,
                    help='Normalize features before downstream tasks')

# Reproducibility
parser.add_argument('--random_seed', type=int, default=42,
                    help='Random seed for reproducibility')
parser.add_argument('--data_split_seed', type=int, default=None,
                    help='Seed for data splitting (default: same as random_seed)')

# GPU
parser.add_argument('--gpu', type=str, default='0',
                    help='GPU device ID (e.g., "0" or "0,1" for multiple)')

# Logging
parser.add_argument('--verbose', type=int, default=1,
                    help='Verbosity level (0=quiet, 1=normal, 2=debug)')

args = parser.parse_args()

# Set GPU
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
gpus = tf.config.list_physical_devices('GPU')
if args.verbose >= 1:
    print(f"GPUs available: {len(gpus)}")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
        print(f"  {gpu}")

# Set seeds
if args.data_split_seed is None:
    args.data_split_seed = args.random_seed

np.random.seed(args.random_seed)
tf.random.set_seed(args.random_seed)

# Generate model name if not provided
if args.model_name is None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.model_name = f"ssl_ecg_fold{args.kfold}_{timestamp}"

# Create output directory structure
output_base = args.output_dir
model_output_dir = os.path.join(output_base, args.model_name)
os.makedirs(model_output_dir, exist_ok=True)

# Subdirectories
checkpoint_dir = os.path.join(model_output_dir, 'checkpoints')
export_dir = os.path.join(model_output_dir, 'exported_model')
logs_dir = os.path.join(model_output_dir, 'logs')
results_dir = os.path.join(model_output_dir, 'results')

for dir_path in [checkpoint_dir, export_dir, logs_dir, results_dir]:
    os.makedirs(dir_path, exist_ok=True)

if args.verbose >= 1:
    print(f"\n{'='*70}")
    print(f"SSL-ECG Production Training - TensorFlow 2.x")
    print(f"{'='*70}")
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Model name: {args.model_name}")
    print(f"Output directory: {model_output_dir}")
    print(f"Random seed: {args.random_seed}")
    print(f"Data split seed: {args.data_split_seed}")
    print(f"Subject-wise CV: {args.subject_wise} {'✓ CORRECT' if args.subject_wise else '✗ INCORRECT'}")
    print(f"Fold: {args.kfold + 1}/{args.total_fold}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"{'='*70}\n")

if not args.subject_wise:
    print("⚠️  WARNING: Running with subject_wise=False causes DATA LEAKAGE!")
    print("⚠️  This inflates performance by 20-30%. Use only for comparison.\n")

# Save configuration
config = vars(args)
config['tensorflow_version'] = tf.__version__
config['timestamp'] = datetime.now().isoformat()
with open(os.path.join(model_output_dir, 'config.json'), 'w') as f:
    json.dump(config, f, indent=2)

if args.verbose >= 2:
    print("Configuration saved to config.json")

# Transformation parameters
noise_param = 15
scale_param = 1.1
permu_param = 20
tw_piece_param = 9
twsf_param = 1.05

transform_task = [0, 1, 2, 3, 4, 5, 6]
single_batch_size = len(transform_task)

# Hyperparameters
batchsize = args.batch_size
actual_batch_size = batchsize * single_batch_size
epoch = args.epochs
initial_learning_rate = args.learning_rate
drop_rate = 0.6
L2 = 0.0001
lr_decay_steps = 10000
lr_decay_rate = 0.9
loss_coeff = [0.195, 0.195, 0.195, 0.0125, 0.0125, 0.195, 0.195]
window_size = 2560
overlap_pct = 0

# Extract data if needed
if args.extract_data == 1:
    if args.verbose >= 1:
        print("Extracting and preprocessing data...")
    if args.data_tag == 'aecg':
        felicity.extract_felicitys_dataset_composite(
            overlap_pct=overlap_pct, window_size_sec=10, fs=256,
            data_path=args.data_folder, type_m_or_f=args.data_tag
        )
    else:
        felicity.extract_felicitys_dataset(
            overlap_pct=overlap_pct, window_size_sec=10, fs=256,
            data_path=args.data_folder, type_m_or_f=args.data_tag
        )

# Load data
data_file = os.path.join(args.data_folder, f'felicitys_{args.data_tag}_{overlap_pct}.npy')
if args.verbose >= 1:
    print(f"Loading data from: {data_file}")

if not os.path.exists(data_file):
    raise FileNotFoundError(
        f"Data file not found: {data_file}\n"
        f"Please ensure your ECG data is preprocessed and saved as .npy file.\n"
        f"Expected format: (n_windows, 2567) with columns [subject_id, stress, pss, pdq, fsi, cortisol, 0, ecg_signal...]"
    )

felicitys_data = np.load(data_file, allow_pickle=True)
if args.verbose >= 1:
    print(f"Data shape: {felicitys_data.shape}")
    print(f"Data columns: [subject_id, stress, pss, pdq, fsi, cortisol, 0, ...ecg_signal({window_size} samples)]\n")

# Split data with CORRECT subject-wise CV
if args.verbose >= 1:
    print(f"{'='*70}")
    print(f"FOLD {args.kfold + 1}/{args.total_fold}")
    print(f"{'='*70}\n")

felicity_train_data, felicity_test_data = datasets.train_test_split_felicity_kfold(
    args.data_folder,
    kfold=args.kfold,
    total_fold=args.total_fold,
    overlap_pct=overlap_pct,
    type_m_or_f=args.data_tag,
    random_seed=args.data_split_seed,
    subject_wise=args.subject_wise,
    validate=args.validate_split
)

np.random.shuffle(felicity_train_data)

# Extract features and labels
train_ECG = felicity_train_data[:, 6:]
train_stress = tf.keras.utils.to_categorical(felicity_train_data[:, 1], 2)
train_pss = felicity_train_data[:, 2]
train_pdq = felicity_train_data[:, 3]
train_fsi = felicity_train_data[:, 4]
train_cortisol = felicity_train_data[:, 5]
train_subjects = felicity_train_data[:, 0]

test_ECG = felicity_test_data[:, 6:]
test_stress = tf.keras.utils.to_categorical(felicity_test_data[:, 1], 2)
test_pss = felicity_test_data[:, 2]
test_pdq = felicity_test_data[:, 3]
test_fsi = felicity_test_data[:, 4]
test_cortisol = felicity_test_data[:, 5]
test_subjects = felicity_test_data[:, 0]

training_length = train_ECG.shape[0]
testing_length = test_ECG.shape[0]

if args.verbose >= 1:
    print(f"Train samples: {training_length}")
    print(f"Test samples: {testing_length}")
    print(f"Train subjects: {len(np.unique(train_subjects))}")
    print(f"Test subjects: {len(np.unique(test_subjects))}\n")

# Save train/test subject IDs
subject_split = {
    'train_subjects': np.unique(train_subjects).astype(int).tolist(),
    'test_subjects': np.unique(test_subjects).astype(int).tolist(),
    'kfold': args.kfold,
    'total_fold': args.total_fold
}
with open(os.path.join(model_output_dir, 'subject_split.json'), 'w') as f:
    json.dump(subject_split, f, indent=2)

# Build model
if args.verbose >= 1:
    print("Building self-supervised model...")

ssl_model = model.build_ssl_model(
    input_shape=(window_size, 1),
    drop_rate=drop_rate,
    hidden_nodes=128,
    l2_reg=L2
)

# Save model architecture
with open(os.path.join(model_output_dir, 'model_architecture.json'), 'w') as f:
    f.write(ssl_model.to_json())

if args.verbose >= 2:
    print("Model architecture:")
    ssl_model.summary()

# Define optimizer with learning rate schedule
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=lr_decay_steps,
    decay_rate=lr_decay_rate,
    staircase=True
)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

# Loss function
bce_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# Metrics
train_loss_tracker = tf.keras.metrics.Mean(name='train_loss')
test_loss_tracker = tf.keras.metrics.Mean(name='test_loss')

# Training step with @tf.function for performance
@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        # Forward pass
        outputs = ssl_model(x, training=True)
        task_outputs = outputs['task_outputs']

        # Calculate loss for each task
        task_losses = []
        for i in range(7):
            task_loss = bce_loss(y[:, i:i+1], task_outputs[i])
            task_losses.append(task_loss)

        # Weighted loss
        total_loss = sum([coeff * loss for coeff, loss in zip(loss_coeff, task_losses)])

        # Add L2 regularization
        l2_loss = sum(ssl_model.losses)
        total_loss += l2_loss

    # Backward pass
    gradients = tape.gradient(total_loss, ssl_model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, ssl_model.trainable_variables))

    train_loss_tracker.update_state(total_loss)
    return total_loss, task_losses

@tf.function
def test_step(x, y):
    # Forward pass
    outputs = ssl_model(x, training=False)
    task_outputs = outputs['task_outputs']

    # Calculate loss for each task
    task_losses = []
    for i in range(7):
        task_loss = bce_loss(y[:, i:i+1], task_outputs[i])
        task_losses.append(task_loss)

    # Weighted loss
    total_loss = sum([coeff * loss for coeff, loss in zip(loss_coeff, task_losses)])

    test_loss_tracker.update_state(total_loss)
    return total_loss, task_losses

# TensorBoard writer
train_writer = tf.summary.create_file_writer(os.path.join(logs_dir, 'train'))
test_writer = tf.summary.create_file_writer(os.path.join(logs_dir, 'test'))

# Training loop
if args.verbose >= 1:
    print("\nStarting self-supervised training...\n")

best_test_loss = float('inf')
best_epoch = -1
training_history = {
    'train_loss': [],
    'test_loss': [],
    'epoch': [],
    'best_epoch': -1,
    'best_test_loss': float('inf')
}

for epoch_num in range(epoch):
    # Reset metrics
    train_loss_tracker.reset_states()
    test_loss_tracker.reset_states()

    # Training
    train_gen = utils.make_total_batch(
        train_ECG, training_length, batchsize,
        noise_param, scale_param, permu_param,
        tw_piece_param, twsf_param, 1/twsf_param
    )

    for training_batch, training_labels, tr_counter, tr_steps in train_gen:
        # Shuffle
        training_batch, training_labels = utils.unison_shuffled_copies(
            training_batch, training_labels
        )

        # Reshape for model input
        training_batch = training_batch.reshape(
            training_batch.shape[0], training_batch.shape[1], 1
        ).astype('float32')
        training_labels = training_labels.astype('float32')

        # Train step
        total_loss, task_losses = train_step(training_batch, training_labels)

    # Testing
    test_gen = utils.make_total_batch(
        test_ECG, testing_length, batchsize,
        noise_param, scale_param, permu_param,
        tw_piece_param, twsf_param, 1/twsf_param
    )

    for testing_batch, testing_labels, te_counter, te_steps in test_gen:
        testing_batch = testing_batch.reshape(
            testing_batch.shape[0], testing_batch.shape[1], 1
        ).astype('float32')
        testing_labels = testing_labels.astype('float32')

        # Test step
        total_loss, task_losses = test_step(testing_batch, testing_labels)

    # Log metrics
    train_loss = train_loss_tracker.result().numpy()
    test_loss = test_loss_tracker.result().numpy()

    training_history['epoch'].append(epoch_num)
    training_history['train_loss'].append(float(train_loss))
    training_history['test_loss'].append(float(test_loss))

    with train_writer.as_default():
        tf.summary.scalar('loss', train_loss, step=epoch_num)
    with test_writer.as_default():
        tf.summary.scalar('loss', test_loss, step=epoch_num)

    if args.verbose >= 1:
        print(f"Epoch {epoch_num+1}/{epoch}: Train Loss={train_loss:.4f}, Test Loss={test_loss:.4f}")

    # Save checkpoint if best model
    if test_loss < best_test_loss:
        best_test_loss = test_loss
        best_epoch = epoch_num
        training_history['best_epoch'] = int(best_epoch)
        training_history['best_test_loss'] = float(best_test_loss)

        # Save weights
        checkpoint_path = os.path.join(checkpoint_dir, f'best_model.h5')
        ssl_model.save_weights(checkpoint_path)

        if args.verbose >= 1:
            print(f"  ✓ Saved best model (test loss: {test_loss:.4f})")

# Save training history
with open(os.path.join(model_output_dir, 'training_history.json'), 'w') as f:
    json.dump(training_history, f, indent=2)

# Load best model
best_checkpoint = os.path.join(checkpoint_dir, 'best_model.h5')
ssl_model.load_weights(best_checkpoint)
if args.verbose >= 1:
    print(f"\n✓ Loaded best model from epoch {best_epoch + 1} (test loss: {best_test_loss:.4f})")

# Extract features for downstream tasks
if args.verbose >= 1:
    print("\nExtracting features for downstream tasks...")

x_tr_feature = utils.extract_features_with_model(ssl_model, train_ECG, batch_size=batchsize)
x_te_feature = utils.extract_features_with_model(ssl_model, test_ECG, batch_size=batchsize)

if args.verbose >= 1:
    print(f"  Train features: {x_tr_feature.shape}")
    print(f"  Test features: {x_te_feature.shape}")

# Normalize features
scaler = None
if args.normalize_features:
    scaler_path = os.path.join(model_output_dir, 'feature_scaler.pkl')
    x_tr_feature, x_te_feature, scaler = utils.normalize_features(
        x_tr_feature, x_te_feature, scaler_path=scaler_path
    )
    if args.verbose >= 2:
        print(f"  ✓ Features normalized and scaler saved to: {scaler_path}")

# Save extracted features (optional, for quick experimentation)
np.save(os.path.join(results_dir, 'train_features.npy'), x_tr_feature)
np.save(os.path.join(results_dir, 'test_features.npy'), x_te_feature)

# Save full model for inference
if args.save_model:
    if args.verbose >= 1:
        print("\nSaving complete model for inference...")

    # Save as SavedModel format (TF2 recommended format)
    saved_model_path = os.path.join(export_dir, 'saved_model')
    ssl_model.save(saved_model_path)
    if args.verbose >= 1:
        print(f"  ✓ Saved model (SavedModel format): {saved_model_path}")

    # Also save as H5 format (for compatibility)
    h5_model_path = os.path.join(export_dir, 'model.h5')
    ssl_model.save(h5_model_path)
    if args.verbose >= 1:
        print(f"  ✓ Saved model (H5 format): {h5_model_path}")

    # Save model weights only
    weights_path = os.path.join(export_dir, 'model_weights.h5')
    ssl_model.save_weights(weights_path)
    if args.verbose >= 1:
        print(f"  ✓ Saved weights: {weights_path}")

# Create deployment package metadata
deployment_metadata = {
    'model_name': args.model_name,
    'version': '1.0.0',
    'created_at': datetime.now().isoformat(),
    'tensorflow_version': tf.__version__,
    'model_type': 'SSL-ECG Self-Supervised Learning',
    'input_shape': [window_size, 1],
    'output_feature_dim': 256,
    'kfold': args.kfold,
    'total_folds': args.total_fold,
    'best_epoch': best_epoch,
    'best_test_loss': float(best_test_loss),
    'train_samples': int(training_length),
    'test_samples': int(testing_length),
    'train_subjects': len(np.unique(train_subjects)),
    'test_subjects': len(np.unique(test_subjects)),
    'subject_wise_cv': args.subject_wise,
    'feature_normalization': args.normalize_features,
    'files': {
        'model_savedmodel': 'exported_model/saved_model',
        'model_h5': 'exported_model/model.h5',
        'weights': 'exported_model/model_weights.h5',
        'architecture': 'model_architecture.json',
        'scaler': 'feature_scaler.pkl' if args.normalize_features else None,
        'config': 'config.json',
        'training_history': 'training_history.json',
        'subject_split': 'subject_split.json'
    },
    'usage': {
        'load_model': 'model = tf.keras.models.load_model("exported_model/saved_model")',
        'extract_features': 'features = model.predict(ecg_data)',
        'inference': 'See inference_tf2.py for complete inference pipeline'
    }
}

with open(os.path.join(model_output_dir, 'deployment_metadata.json'), 'w') as f:
    json.dump(deployment_metadata, f, indent=2)

# Create README for this model
readme_content = f"""# SSL-ECG Model: {args.model_name}

## Model Information

- **Created:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **TensorFlow Version:** {tf.__version__}
- **Fold:** {args.kfold + 1}/{args.total_fold}
- **Training Samples:** {training_length}
- **Test Samples:** {testing_length}
- **Best Epoch:** {best_epoch + 1}/{epoch}
- **Best Test Loss:** {best_test_loss:.4f}

## Data Split

- **Subject-wise CV:** {'Yes ✓ (CORRECT)' if args.subject_wise else 'No ✗ (INCORRECT - data leakage!)'}
- **Train Subjects:** {len(np.unique(train_subjects))}
- **Test Subjects:** {len(np.unique(test_subjects))}

## Model Files

- `exported_model/saved_model/` - Complete model (TF SavedModel format) - **USE THIS FOR INFERENCE**
- `exported_model/model.h5` - Complete model (H5 format)
- `exported_model/model_weights.h5` - Model weights only
- `model_architecture.json` - Model architecture
- `feature_scaler.pkl` - Feature normalization scaler (if enabled)
- `config.json` - Complete training configuration
- `training_history.json` - Training/validation loss history
- `subject_split.json` - Train/test subject IDs

## Quick Start - Inference

```python
import tensorflow as tf
import numpy as np
import pickle

# Load model
model = tf.keras.models.load_model('exported_model/saved_model')

# Load your ECG data (shape: [n_samples, 2560])
ecg_data = np.load('your_ecg_data.npy')

# Reshape for model input
ecg_data = ecg_data.reshape(-1, 2560, 1)

# Extract features
outputs = model.predict(ecg_data)
features = outputs['features']  # Shape: [n_samples, 256]

# Load scaler and normalize
with open('feature_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
features_normalized = scaler.transform(features)

# Now use features_normalized for your downstream task
# (stress classification, PSS prediction, etc.)
```

## Directory Structure

```
{args.model_name}/
├── exported_model/          # Deployable models
│   ├── saved_model/         # TF SavedModel (recommended)
│   ├── model.h5             # H5 format
│   └── model_weights.h5     # Weights only
├── checkpoints/             # Training checkpoints
│   └── best_model.h5
├── logs/                    # TensorBoard logs
│   ├── train/
│   └── test/
├── results/                 # Extracted features
│   ├── train_features.npy
│   └── test_features.npy
├── config.json              # Training configuration
├── model_architecture.json  # Model architecture
├── training_history.json    # Training history
├── subject_split.json       # Train/test subject IDs
├── feature_scaler.pkl       # Feature normalizer
├── deployment_metadata.json # Deployment info
└── README.md                # This file
```

## Notes

- This model was trained with {'CORRECT' if args.subject_wise else 'INCORRECT'} subject-wise cross-validation
- {'Features are normalized using StandardScaler' if args.normalize_features else 'Features are NOT normalized'}
- For production inference, use the complete pipeline in `inference_tf2.py`
"""

with open(os.path.join(model_output_dir, 'README.md'), 'w') as f:
    f.write(readme_content)

if args.verbose >= 1:
    print(f"\n{'='*70}")
    print(f"Training Complete!")
    print(f"{'='*70}")
    print(f"Model saved to: {model_output_dir}")
    print(f"\nModel files:")
    print(f"  ✓ SavedModel:  {os.path.join(export_dir, 'saved_model')}")
    print(f"  ✓ H5 Model:    {os.path.join(export_dir, 'model.h5')}")
    print(f"  ✓ Weights:     {os.path.join(export_dir, 'model_weights.h5')}")
    if args.normalize_features:
        print(f"  ✓ Scaler:      {os.path.join(model_output_dir, 'feature_scaler.pkl')}")
    print(f"  ✓ Metadata:    {os.path.join(model_output_dir, 'deployment_metadata.json')}")
    print(f"  ✓ README:      {os.path.join(model_output_dir, 'README.md')}")
    print(f"\nNext steps:")
    print(f"  1. Review README.md in the model directory")
    print(f"  2. Use this model for inference with inference_tf2.py")
    print(f"  3. View training progress: tensorboard --logdir={logs_dir}")
    print(f"{'='*70}\n")

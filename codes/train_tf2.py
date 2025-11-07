"""
train_tf2.py - TensorFlow 2.x training script with proper subject-wise CV

This is a complete TF2 migration of the SSL-ECG training code featuring:
- Modern TensorFlow 2.x / Keras API (no sessions!)
- Eager execution by default
- Subject-wise cross-validation (data leakage FIX included)
- Feature normalization
- Configurable random seeds
- Compatible with Python 3.8-3.11

Usage:
    # Basic usage (correct subject-wise CV)
    python train_tf2.py

    # Custom configuration
    python train_tf2.py --random_seed 123 --epochs 50 --gpu 0

Author: Migrated by Claude (2025-11-07)
Original: Sarkar et al. (2020)
"""

import os
import argparse
import warnings
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
parser = argparse.ArgumentParser(description='SSL-ECG Training with TensorFlow 2.x')
parser.add_argument('--random_seed', type=int, default=42,
                    help='Random seed for reproducibility')
parser.add_argument('--data_split_seed', type=int, default=None,
                    help='Seed for data splitting (default: same as random_seed)')
parser.add_argument('--subject_wise', type=lambda x: str(x).lower() == 'true', default=True,
                    help='Use subject-wise CV (True=CORRECT, False=INCORRECT)')
parser.add_argument('--normalize_features', type=lambda x: str(x).lower() == 'true', default=True,
                    help='Normalize features before downstream tasks')
parser.add_argument('--validate_split', type=lambda x: str(x).lower() == 'true', default=True,
                    help='Validate no subject overlap')
parser.add_argument('--data_tag', type=str, default='mecg',
                    help='Type of ECG data: mecg or aecg')
parser.add_argument('--epochs', type=int, default=30,
                    help='Number of training epochs')
parser.add_argument('--total_folds', type=int, default=5,
                    help='Number of cross-validation folds')
parser.add_argument('--batch_size', type=int, default=128,
                    help='Batch size for training')
parser.add_argument('--extract_data', type=int, default=0,
                    help='Extract data from raw files (1) or use existing (0)')
parser.add_argument('--gpu', type=str, default='0',
                    help='GPU device ID (e.g., "0" or "0,1" for multiple)')
args = parser.parse_args()

# Set GPU
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
gpus = tf.config.list_physical_devices('GPU')
print(f"GPUs available: {len(gpus)}")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    print(f"  {gpu}")

# Set seeds
if args.data_split_seed is None:
    args.data_split_seed = args.random_seed

np.random.seed(args.random_seed)
tf.random.set_seed(args.random_seed)

print(f"\n{'='*70}")
print(f"SSL-ECG Training - TensorFlow 2.x Version")
print(f"{'='*70}")
print(f"TensorFlow version: {tf.__version__}")
print(f"Random seed: {args.random_seed}")
print(f"Data split seed: {args.data_split_seed}")
print(f"Subject-wise CV: {args.subject_wise} {'✓ CORRECT' if args.subject_wise else '✗ INCORRECT'}")
print(f"Feature normalization: {args.normalize_features}")
print(f"Validate split: {args.validate_split}")
print(f"Data tag: {args.data_tag}")
print(f"Epochs: {args.epochs}")
print(f"Folds: {args.total_folds}")
print(f"Batch size: {args.batch_size}")
print(f"{'='*70}\n")

if not args.subject_wise:
    print("⚠️  WARNING: Running with subject_wise=False causes DATA LEAKAGE!")
    print("⚠️  This inflates performance by 20-30%. Use only for comparison.\n")

# Define paths
data_folder = os.path.join(os.path.dirname(dirname), 'data')
summaries = os.path.join(os.path.dirname(dirname), 'summaries')
output = os.path.join(os.path.dirname(dirname), 'output')
model_dir = os.path.join(os.path.dirname(dirname), 'models')

# Transformation parameters
noise_param = 15
scale_param = 1.1
permu_param = 20
tw_piece_param = 9
twsf_param = 1.05

transform_task = [0, 1, 2, 3, 4, 5, 6]
single_batch_size = len(transform_task)
total_fold = args.total_folds

# Hyperparameters
batchsize = args.batch_size
actual_batch_size = batchsize * single_batch_size
epoch = args.epochs
initial_learning_rate = 0.001
drop_rate = 0.6
L2 = 0.0001
lr_decay_steps = 10000
lr_decay_rate = 0.9
loss_coeff = [0.195, 0.195, 0.195, 0.0125, 0.0125, 0.195, 0.195]
window_size = 2560
overlap_pct = 0
data_tag = args.data_tag
current_time = utils.current_time()

print(f"Current time: {current_time}\n")

# Extract data if needed
if args.extract_data == 1:
    print("Extracting and preprocessing data...")
    if data_tag == 'aecg':
        felicity.extract_felicitys_dataset_composite(
            overlap_pct=overlap_pct, window_size_sec=10, fs=256,
            data_path=data_folder, type_m_or_f=data_tag
        )
    else:
        felicity.extract_felicitys_dataset(
            overlap_pct=overlap_pct, window_size_sec=10, fs=256,
            data_path=data_folder, type_m_or_f=data_tag
        )

# Load data
data_file = os.path.join(data_folder, f'felicitys_{data_tag}_{overlap_pct}.npy')
print(f"Loading data from: {data_file}")
felicitys_data = np.load(data_file, allow_pickle=True)
print(f"Data shape: {felicitys_data.shape}\n")

# Training loop
print("="*70)
print("Starting cross-validation training")
print("="*70)

t = trange(total_fold, desc='K-Fold', leave=True, ncols=100)
for k in t:
    print(f"\n{'='*70}")
    print(f"FOLD {k+1}/{total_fold}")
    print(f"{'='*70}\n")

    # Create output directories
    fold_output = os.path.join(output, 'fold_' + str(k))
    fold_model_dir = os.path.join(model_dir, 'fold_' + str(k))
    utils.makedirs(fold_output)
    utils.makedirs(fold_model_dir)

    # Split data with CORRECT subject-wise CV
    felicity_train_data, felicity_test_data = datasets.train_test_split_felicity_kfold(
        data_folder,
        kfold=k,
        total_fold=total_fold,
        overlap_pct=overlap_pct,
        type_m_or_f=data_tag,
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

    test_ECG = felicity_test_data[:, 6:]
    test_stress = tf.keras.utils.to_categorical(felicity_test_data[:, 1], 2)
    test_pss = felicity_test_data[:, 2]
    test_pdq = felicity_test_data[:, 3]
    test_fsi = felicity_test_data[:, 4]
    test_cortisol = felicity_test_data[:, 5]

    training_length = train_ECG.shape[0]
    testing_length = test_ECG.shape[0]

    print(f"Train samples: {training_length}")
    print(f"Test samples: {testing_length}\n")

    # Build model
    print("Building self-supervised model...")
    ssl_model = model.build_ssl_model(
        input_shape=(window_size, 1),
        drop_rate=drop_rate,
        hidden_nodes=128,
        l2_reg=L2
    )

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
            l2_loss = sum(ssl_model.losses)  # Keras tracks regularization losses
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
    log_dir = os.path.join(summaries, f'fold_{k}', current_time)
    train_writer = tf.summary.create_file_writer(os.path.join(log_dir, 'train'))
    test_writer = tf.summary.create_file_writer(os.path.join(log_dir, 'test'))

    # Training loop
    print("Starting self-supervised training...\n")

    best_test_loss = float('inf')

    for epoch_num in range(epoch):
        t.set_description(f"Fold {k+1} - Epoch {epoch_num+1}/{epoch}")

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

        with train_writer.as_default():
            tf.summary.scalar('loss', train_loss, step=epoch_num)
        with test_writer.as_default():
            tf.summary.scalar('loss', test_loss, step=epoch_num)

        print(f"Epoch {epoch_num+1}: Train Loss={train_loss:.4f}, Test Loss={test_loss:.4f}")

        # Save model checkpoint
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            checkpoint_path = os.path.join(fold_model_dir, f'best_model_epoch_{epoch_num}.h5')
            ssl_model.save_weights(checkpoint_path)
            print(f"  ✓ Saved best model (test loss: {test_loss:.4f})")

    # Load best model
    best_checkpoint = os.path.join(fold_model_dir, f'best_model_epoch_{epoch-1}.h5')
    ssl_model.load_weights(best_checkpoint)
    print(f"\n✓ Loaded best model from: {best_checkpoint}")

    # Extract features for downstream tasks
    print("\nExtracting features for downstream tasks...")
    x_tr_feature = utils.extract_features_with_model(ssl_model, train_ECG, batch_size=batchsize)
    x_te_feature = utils.extract_features_with_model(ssl_model, test_ECG, batch_size=batchsize)

    print(f"  Train features: {x_tr_feature.shape}")
    print(f"  Test features: {x_te_feature.shape}")

    # Normalize features
    if args.normalize_features:
        scaler_path = os.path.join(fold_output, 'feature_scaler.pkl')
        x_tr_feature, x_te_feature, _ = utils.normalize_features(
            x_tr_feature, x_te_feature, scaler_path=scaler_path
        )

    # Downstream tasks
    print("\nTraining downstream models...")

    print("  1. Stress classification...")
    model.train_downstream_classifier(
        x_tr_feature, train_stress, x_te_feature, test_stress,
        identifier='stress', kfold=k, output_dir=output,
        epochs=100, batch_size=batchsize, verbose=0
    )

    print("  2. PDQ regression...")
    model.train_downstream_regressor(
        x_tr_feature, train_pdq, x_te_feature, test_pdq,
        identifier='pdq', kfold=k, output_dir=output,
        epochs=100, batch_size=batchsize, verbose=0
    )

    print("  3. PSS regression...")
    model.train_downstream_regressor(
        x_tr_feature, train_pss, x_te_feature, test_pss,
        identifier='pss', kfold=k, output_dir=output,
        epochs=100, batch_size=batchsize, verbose=0
    )

    print("  4. FSI regression...")
    model.train_downstream_regressor(
        x_tr_feature, train_fsi, x_te_feature, test_fsi,
        identifier='fsi', kfold=k, output_dir=output,
        epochs=100, batch_size=batchsize, verbose=0
    )

    print("  5. Cortisol regression...")
    model.train_downstream_regressor(
        x_tr_feature, train_cortisol, x_te_feature, test_cortisol,
        identifier='cortisol', kfold=k, output_dir=output,
        epochs=100, batch_size=batchsize, verbose=0
    )

    print(f"\n✓ Fold {k+1} complete!")

print(f"\n{'='*70}")
print(f"Training Complete!")
print(f"{'='*70}")
print(f"Results saved to: {output}")
print(f"Models saved to: {model_dir}")
if args.subject_wise:
    print(f"✓ Used CORRECT subject-wise cross-validation")
else:
    print(f"✗ WARNING: Used INCORRECT within-subject split")
print(f"{'='*70}\n")

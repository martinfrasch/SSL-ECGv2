# Detection of Stress with Self-supervised Learning (TensorFlow 2.x Migration)

[Detection of Maternal and Fetal Stress from ECG with Self-supervised Representation Learning](https://arxiv.org/abs/2011.02000)
Authors: [Sarkar](https://www.pritamsarkar.com/) et al.

**Branch**: `claude/tf2-migration-011CUtwG7jQk8UkVsFwcjUs3`

## ⚠️ IMPORTANT: TensorFlow 2.x Required

This branch contains the **TensorFlow 2.x migration** of the original SSL-ECG codebase. The original code used TensorFlow 1.14.0, but this version has been fully migrated to modern TensorFlow 2.x APIs.

## Requirements

### Core Dependencies
- **Python >= 3.8** (Tested with Python 3.10 on Ubuntu 22.04)
- **TensorFlow >= 2.10.0** (Recommended: TensorFlow 2.15.0)
  - For GPU support: `tensorflow[and-cuda]==2.15.0`
  - For CPU only: `tensorflow==2.15.0`
- NumPy >= 1.24.0 (Tested with 1.24.3)
- Scikit-Learn >= 1.3.0
- Tqdm (latest)
- Pandas (latest)
- Mlxtend (latest)

### Installation

```bash
# Install TensorFlow 2.x with GPU support
pip install tensorflow[and-cuda]==2.15.0

# Install other dependencies
pip install numpy==1.24.3 scikit-learn==1.3.0 tqdm pandas mlxtend
```

### GPU Support

TensorFlow 2.15.0 automatically installs CUDA and cuDNN dependencies via pip. No manual CUDA installation required on Ubuntu 22.04+.

**Verify GPU availability:**
```python
import tensorflow as tf
print("GPUs Available:", tf.config.list_physical_devices('GPU'))
```

## Training Scripts

### TensorFlow 2 Production Training (Recommended)

Use `train_tf2_production.py` for production-ready training with:
- Complete model saving (full model + weights + architecture)
- 5-fold cross-validation support
- Comprehensive logging and progress tracking
- Export for inference

**Example usage:**
```bash
python codes/train_tf2_production.py \
    --data_folder ./data \
    --kfold 0 \
    --total_fold 5 \
    --output_dir trained_models \
    --save_model True \
    --epochs 30
```

### Legacy Scripts (For Reference)

- `train_fixed.py` - Original TensorFlow 1.x code (deprecated, use TF2 version)
- `train_tf2.py` - Basic TF2 migration
- `train_tf2_production.py` - **Recommended for production use**

## Key Differences from Original Code

### API Changes
- **TF1**: `tf.placeholder`, `tf.Session`, `tf.global_variables_initializer()`
- **TF2**: Eager execution by default, `tf.function` decorators, Keras model API

### Model Saving
- **TF1**: `tf.train.Saver()` with checkpoint files
- **TF2**: `model.save()` with SavedModel format + HDF5 weights

### GPU Detection
- **TF1**: `tf.test.is_gpu_available()`
- **TF2**: `tf.config.list_physical_devices('GPU')`

## File Structure

```
codes/
├── train_tf2_production.py  # Production training script (TF2) ⭐ USE THIS
├── train_tf2.py             # Basic TF2 training
├── train_fixed.py           # Original TF1 code (deprecated)
├── model_tf2.py             # TF2 model architecture
├── utils_tf2.py             # TF2 utility functions
├── datasets.py              # Dataset loading (compatible with both)
└── felicity.py              # Felicity-specific preprocessing
```

## Pre-trained Model

Self-supervised pretrained model on public datasets is available [here](https://code.engineering.queensu.ca/17ps21/ssl-ecg-v2/-/tree/master/load_model).

**Note**: Pre-trained models from the TF1 version need to be converted for TF2 compatibility.

## Migration Notes

This branch was created to modernize the codebase for:
1. **Compatibility**: TensorFlow 2.x is actively maintained (TF 1.x reached EOL)
2. **Performance**: Better GPU utilization with TF2's eager execution
3. **Deployment**: Easier model serving with SavedModel format
4. **Development**: Cleaner API and better debugging with eager mode

**Migration completed**: November 2025

## Citation

Please cite our papers for any purpose of usage.
```bibtex
@misc{sarkar2020detection,
      title={Detection of Maternal and Fetal Stress from ECG with Self-supervised Representation Learning},
      author={Pritam Sarkar and Silvia Lobmaier and Bibiana Fabre and Gabriela Berg and Alexander Mueller and Martin G. Frasch and Marta C. Antonelli and Ali Etemad},
      year={2020},
      eprint={2011.02000},
      archivePrefix={arXiv},
      primaryClass={q-bio.QM}
}
```

## Acknowledgement

- Original codebase: [SSL-ECG](https://code.engineering.queensu.ca/17ps21/SSL-ECG)
- TensorFlow 2 migration: Claude (November 2025)

## Questions

For questions about the original research, contact <pritam.sarkar@queensu.ca> or connect on [LinkedIn](https://www.linkedin.com/in/sarkarpritam/).

For questions about the TF2 migration, please open an issue on this repository.

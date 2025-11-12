#!/bin/bash
#
# deploy_to_gcp.sh - Deploy SSL-ECG TF2 training to GCP from MacOS
#
# Usage:
#   bash scripts/deploy_to_gcp.sh \
#       --instance-name ssl-ecg-training \
#       --zone us-central1-a \
#       --gpu-type nvidia-tesla-v100 \
#       --data-dir /path/to/local/ecg/data \
#       --kfold 0 \
#       --total-fold 5
#
# This script will:
# 1. Create a GCP VM with GPU
# 2. Upload your ECG data
# 3. Clone the repository
# 4. Install dependencies
# 5. Run training
# 6. Download trained models back to your Mac
#

set -e  # Exit on error

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
INSTANCE_NAME="ssl-ecg-training"
ZONE="us-central1-a"
GPU_TYPE="nvidia-tesla-v100"
MACHINE_TYPE="n1-standard-8"
BOOT_DISK_SIZE="100GB"
DATA_DIR=""
KFOLD=0
TOTAL_FOLD=5
EPOCHS=30
BATCH_SIZE=256
BRANCH="claude/tf2-migration-011CUtwG7jQk8UkVsFwcjUs3"
KEEP_VM=false
LOCAL_OUTPUT_DIR="./trained_models"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --instance-name)
            INSTANCE_NAME="$2"
            shift 2
            ;;
        --zone)
            ZONE="$2"
            shift 2
            ;;
        --gpu-type)
            GPU_TYPE="$2"
            shift 2
            ;;
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --kfold)
            KFOLD="$2"
            shift 2
            ;;
        --total-fold)
            TOTAL_FOLD="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --keep-vm)
            KEEP_VM=true
            shift
            ;;
        --output-dir)
            LOCAL_OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$DATA_DIR" ]; then
    echo -e "${RED}Error: --data-dir is required${NC}"
    echo "Usage: bash scripts/deploy_to_gcp.sh --data-dir /path/to/ecg/data"
    exit 1
fi

if [ ! -d "$DATA_DIR" ]; then
    echo -e "${RED}Error: Data directory does not exist: $DATA_DIR${NC}"
    exit 1
fi

echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║       SSL-ECG TF2 Training Deployment to GCP              ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "Configuration:"
echo "  Instance Name:   $INSTANCE_NAME"
echo "  Zone:            $ZONE"
echo "  GPU Type:        $GPU_TYPE"
echo "  Machine Type:    $MACHINE_TYPE"
echo "  Data Directory:  $DATA_DIR"
echo "  K-Fold:          $KFOLD / $TOTAL_FOLD"
echo "  Epochs:          $EPOCHS"
echo "  Batch Size:      $BATCH_SIZE"
echo "  Output Dir:      $LOCAL_OUTPUT_DIR"
echo "  Keep VM:         $KEEP_VM"
echo ""

read -p "Continue with deployment? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Deployment cancelled."
    exit 0
fi

# Step 1: Check if VM already exists
echo -e "\n${YELLOW}[1/8] Checking if VM exists...${NC}"
if gcloud compute instances describe $INSTANCE_NAME --zone=$ZONE &> /dev/null; then
    echo -e "${YELLOW}VM already exists. Using existing VM.${NC}"
else
    echo -e "${GREEN}Creating new VM...${NC}"

    gcloud compute instances create $INSTANCE_NAME \
        --zone=$ZONE \
        --machine-type=$MACHINE_TYPE \
        --accelerator="type=$GPU_TYPE,count=1" \
        --image-family=ubuntu-2004-lts \
        --image-project=ubuntu-os-cloud \
        --boot-disk-size=$BOOT_DISK_SIZE \
        --boot-disk-type=pd-ssd \
        --maintenance-policy=TERMINATE \
        --metadata=install-nvidia-driver=True \
        --scopes=https://www.googleapis.com/auth/cloud-platform

    echo -e "${GREEN}VM created successfully!${NC}"

    # Wait for VM to be ready
    echo "Waiting 60 seconds for VM to initialize..."
    sleep 60
fi

# Step 2: Upload data to VM
echo -e "\n${YELLOW}[2/8] Uploading ECG data to VM...${NC}"
gcloud compute scp --recurse $DATA_DIR $INSTANCE_NAME:~/ecg_data --zone=$ZONE
echo -e "${GREEN}Data uploaded successfully!${NC}"

# Step 3: Setup environment
echo -e "\n${YELLOW}[3/8] Setting up environment on VM...${NC}"
gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --command="
    set -e

    # Update system
    sudo apt-get update

    # Install Python 3.10
    sudo apt-get install -y python3.10 python3.10-dev python3-pip git

    # Update pip
    python3.10 -m pip install --upgrade pip

    # Install CUDA toolkit if needed (usually pre-installed with NVIDIA driver)
    if ! command -v nvcc &> /dev/null; then
        echo 'CUDA not found, installing...'
        sudo apt-get install -y nvidia-cuda-toolkit
    fi

    echo 'Environment setup complete!'
"
echo -e "${GREEN}Environment configured!${NC}"

# Step 4: Clone repository
echo -e "\n${YELLOW}[4/8] Cloning repository...${NC}"
gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --command="
    set -e

    # Remove old clone if exists
    rm -rf SSL-ECGv2

    # Clone repository
    git clone https://github.com/martinfrasch/SSL-ECGv2.git
    cd SSL-ECGv2

    # Checkout correct branch
    git checkout $BRANCH

    echo 'Repository cloned and checked out to $BRANCH'
"
echo -e "${GREEN}Repository cloned!${NC}"

# Step 5: Install Python dependencies
echo -e "\n${YELLOW}[5/8] Installing Python dependencies...${NC}"
gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --command="
    set -e
    cd SSL-ECGv2

    # Install TensorFlow and dependencies
    python3.10 -m pip install -r requirements_tf2.txt

    # Verify TensorFlow can see GPU
    python3.10 -c 'import tensorflow as tf; print(\"GPUs Available:\", tf.config.list_physical_devices(\"GPU\"))'

    echo 'Dependencies installed!'
"
echo -e "${GREEN}Dependencies installed!${NC}"

# Step 6: Run training
echo -e "\n${YELLOW}[6/8] Starting training...${NC}"
echo -e "${YELLOW}This may take several hours. Training logs will be displayed below.${NC}"
echo ""

gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --command="
    set -e
    cd SSL-ECGv2

    # Create output directory
    mkdir -p trained_models

    # Run training with all parameters
    python3.10 codes/train_tf2_production.py \
        --data_folder ~/ecg_data \
        --kfold $KFOLD \
        --total_fold $TOTAL_FOLD \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --output_dir trained_models \
        --save_model True \
        --subject_wise True \
        --normalize_features True \
        --validate_split True \
        --verbose 2

    echo 'Training complete!'
"
echo -e "${GREEN}Training completed successfully!${NC}"

# Step 7: Download trained models
echo -e "\n${YELLOW}[7/8] Downloading trained models to local machine...${NC}"
mkdir -p $LOCAL_OUTPUT_DIR
gcloud compute scp --recurse $INSTANCE_NAME:~/SSL-ECGv2/trained_models/* $LOCAL_OUTPUT_DIR/ --zone=$ZONE
echo -e "${GREEN}Models downloaded to: $LOCAL_OUTPUT_DIR${NC}"

# Step 8: Cleanup (optional)
echo -e "\n${YELLOW}[8/8] Cleanup...${NC}"
if [ "$KEEP_VM" = false ]; then
    echo "Stopping VM to save costs..."
    gcloud compute instances stop $INSTANCE_NAME --zone=$ZONE
    echo -e "${GREEN}VM stopped. You can restart it later with:${NC}"
    echo "  gcloud compute instances start $INSTANCE_NAME --zone=$ZONE"
    echo ""
    echo -e "${YELLOW}To delete the VM completely:${NC}"
    echo "  gcloud compute instances delete $INSTANCE_NAME --zone=$ZONE"
else
    echo -e "${GREEN}VM kept running as requested.${NC}"
    echo "To stop the VM:"
    echo "  gcloud compute instances stop $INSTANCE_NAME --zone=$ZONE"
fi

# Summary
echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                  Deployment Complete!                      ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "Training Summary:"
echo "  ✓ VM Instance:    $INSTANCE_NAME (zone: $ZONE)"
echo "  ✓ GPU:            $GPU_TYPE"
echo "  ✓ Fold:           $KFOLD / $TOTAL_FOLD"
echo "  ✓ Models saved:   $LOCAL_OUTPUT_DIR"
echo ""
echo "Next Steps:"
echo "  1. Review training results in: $LOCAL_OUTPUT_DIR"
echo "  2. Run inference on new ECGs with: bash scripts/inference_on_gcp.sh"
echo "  3. For additional folds, run this script with --kfold 1, 2, 3, 4"
echo ""
echo "Estimated Cost (V100):"
echo "  Training time: ~6-8 hours = \$12-17"
echo ""

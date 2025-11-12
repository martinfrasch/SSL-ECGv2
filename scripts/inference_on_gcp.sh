#!/bin/bash
#
# inference_on_gcp.sh - Run SSL-ECG inference on GCP from MacOS
#
# Usage:
#   bash scripts/inference_on_gcp.sh \
#       --model-dir trained_models/ssl_ecg_fold0_20251112_143022 \
#       --input-data /path/to/new_ecgs.npy \
#       --output-file predictions.csv \
#       --instance-name ssl-ecg-inference \
#       --zone us-central1-a
#
# This script will:
# 1. Create/reuse a GCP VM with GPU
# 2. Upload your trained model
# 3. Upload your new ECG data
# 4. Run inference
# 5. Download results back to your Mac
# 6. Optionally clean up the VM
#

set -e  # Exit on error

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
INSTANCE_NAME="ssl-ecg-inference"
ZONE="us-central1-a"
GPU_TYPE="nvidia-tesla-t4"  # T4 is cheaper and sufficient for inference
MACHINE_TYPE="n1-standard-4"
MODEL_DIR=""
INPUT_DATA=""
OUTPUT_FILE="predictions.csv"
TASK=""
DOWNSTREAM_MODEL=""
BATCH_SIZE=256
BRANCH="claude/tf2-migration-011CUtwG7jQk8UkVsFwcjUs3"
KEEP_VM=false
LOCAL_OUTPUT_DIR="./inference_results"

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
        --model-dir)
            MODEL_DIR="$2"
            shift 2
            ;;
        --input-data)
            INPUT_DATA="$2"
            shift 2
            ;;
        --output-file)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        --task)
            TASK="$2"
            shift 2
            ;;
        --downstream-model)
            DOWNSTREAM_MODEL="$2"
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
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$MODEL_DIR" ]; then
    echo -e "${RED}Error: --model-dir is required${NC}"
    echo "Usage: bash scripts/inference_on_gcp.sh --model-dir trained_models/... --input-data data.npy"
    exit 1
fi

if [ -z "$INPUT_DATA" ]; then
    echo -e "${RED}Error: --input-data is required${NC}"
    echo "Usage: bash scripts/inference_on_gcp.sh --model-dir trained_models/... --input-data data.npy"
    exit 1
fi

if [ ! -d "$MODEL_DIR" ]; then
    echo -e "${RED}Error: Model directory does not exist: $MODEL_DIR${NC}"
    exit 1
fi

if [ ! -f "$INPUT_DATA" ]; then
    echo -e "${RED}Error: Input data file does not exist: $INPUT_DATA${NC}"
    exit 1
fi

if [ ! -z "$TASK" ] && [ -z "$DOWNSTREAM_MODEL" ]; then
    echo -e "${RED}Error: --downstream-model is required when --task is specified${NC}"
    exit 1
fi

echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║       SSL-ECG Inference Deployment to GCP                 ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "Configuration:"
echo "  Instance Name:   $INSTANCE_NAME"
echo "  Zone:            $ZONE"
echo "  GPU Type:        $GPU_TYPE"
echo "  Machine Type:    $MACHINE_TYPE"
echo "  Model Directory: $MODEL_DIR"
echo "  Input Data:      $INPUT_DATA"
echo "  Output File:     $OUTPUT_FILE"
if [ ! -z "$TASK" ]; then
    echo "  Task:            $TASK"
    echo "  Downstream Model: $DOWNSTREAM_MODEL"
fi
echo "  Batch Size:      $BATCH_SIZE"
echo "  Keep VM:         $KEEP_VM"
echo ""

read -p "Continue with deployment? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Deployment cancelled."
    exit 0
fi

# Step 1: Check if VM already exists
echo -e "\n${YELLOW}[1/7] Checking if VM exists...${NC}"
if gcloud compute instances describe $INSTANCE_NAME --zone=$ZONE &> /dev/null; then
    echo -e "${YELLOW}VM already exists. Checking if running...${NC}"

    VM_STATUS=$(gcloud compute instances describe $INSTANCE_NAME --zone=$ZONE --format="value(status)")
    if [ "$VM_STATUS" = "TERMINATED" ] || [ "$VM_STATUS" = "STOPPED" ]; then
        echo "VM is stopped. Starting..."
        gcloud compute instances start $INSTANCE_NAME --zone=$ZONE
        sleep 30
    else
        echo -e "${GREEN}VM is already running.${NC}"
    fi
else
    echo -e "${GREEN}Creating new VM for inference...${NC}"

    gcloud compute instances create $INSTANCE_NAME \
        --zone=$ZONE \
        --machine-type=$MACHINE_TYPE \
        --accelerator="type=$GPU_TYPE,count=1" \
        --image-family=ubuntu-2004-lts \
        --image-project=ubuntu-os-cloud \
        --boot-disk-size=50GB \
        --boot-disk-type=pd-ssd \
        --maintenance-policy=TERMINATE \
        --metadata=install-nvidia-driver=True \
        --scopes=https://www.googleapis.com/auth/cloud-platform

    echo -e "${GREEN}VM created successfully!${NC}"

    # Wait for VM to be ready
    echo "Waiting 60 seconds for VM to initialize..."
    sleep 60
fi

# Step 2: Setup environment (if needed)
echo -e "\n${YELLOW}[2/7] Setting up environment...${NC}"
gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --command="
    set -e

    # Check if SSL-ECGv2 exists
    if [ ! -d SSL-ECGv2 ]; then
        echo 'Setting up environment...'

        # Update system
        sudo apt-get update

        # Install Python 3.10
        sudo apt-get install -y python3.10 python3.10-dev python3-pip git

        # Update pip
        python3.10 -m pip install --upgrade pip

        # Clone repository
        git clone https://github.com/martinfrasch/SSL-ECGv2.git
        cd SSL-ECGv2
        git checkout $BRANCH

        # Install dependencies
        python3.10 -m pip install -r requirements_tf2.txt

        echo 'Environment setup complete!'
    else
        echo 'Environment already set up.'
        cd SSL-ECGv2
        git pull origin $BRANCH
    fi
"
echo -e "${GREEN}Environment ready!${NC}"

# Step 3: Upload model
echo -e "\n${YELLOW}[3/7] Uploading trained model...${NC}"
gcloud compute scp --recurse $MODEL_DIR $INSTANCE_NAME:~/model --zone=$ZONE
echo -e "${GREEN}Model uploaded!${NC}"

# Step 4: Upload input data
echo -e "\n${YELLOW}[4/7] Uploading input data...${NC}"
INPUT_FILENAME=$(basename $INPUT_DATA)
gcloud compute scp $INPUT_DATA $INSTANCE_NAME:~/input_data.npy --zone=$ZONE
echo -e "${GREEN}Input data uploaded!${NC}"

# Step 5: Upload downstream model if needed
if [ ! -z "$TASK" ] && [ ! -z "$DOWNSTREAM_MODEL" ]; then
    echo -e "\n${YELLOW}[5/7] Uploading downstream model...${NC}"
    gcloud compute scp --recurse $DOWNSTREAM_MODEL $INSTANCE_NAME:~/downstream_model --zone=$ZONE
    echo -e "${GREEN}Downstream model uploaded!${NC}"
    DOWNSTREAM_ARG="--downstream_model ~/downstream_model"
else
    echo -e "\n${YELLOW}[5/7] Skipping downstream model upload (not required)${NC}"
    DOWNSTREAM_ARG=""
fi

# Step 6: Run inference
echo -e "\n${YELLOW}[6/7] Running inference...${NC}"
echo ""

TASK_ARG=""
if [ ! -z "$TASK" ]; then
    TASK_ARG="--task $TASK"
fi

gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --command="
    set -e
    cd SSL-ECGv2

    # Run inference
    python3.10 codes/inference_tf2.py \
        --model_dir ~/model \
        --input_data ~/input_data.npy \
        --output_file ~/output_file \
        --batch_size $BATCH_SIZE \
        $TASK_ARG \
        $DOWNSTREAM_ARG \
        --verbose 2

    echo 'Inference complete!'
"
echo -e "${GREEN}Inference completed successfully!${NC}"

# Step 7: Download results
echo -e "\n${YELLOW}[7/7] Downloading results...${NC}"
mkdir -p $LOCAL_OUTPUT_DIR

# Determine output file extension
if [ -z "$TASK" ]; then
    # Features output
    gcloud compute scp $INSTANCE_NAME:~/output_file.npy $LOCAL_OUTPUT_DIR/$OUTPUT_FILE --zone=$ZONE
else
    # Predictions output
    gcloud compute scp $INSTANCE_NAME:~/output_file $LOCAL_OUTPUT_DIR/$OUTPUT_FILE --zone=$ZONE
fi

echo -e "${GREEN}Results downloaded to: $LOCAL_OUTPUT_DIR/$OUTPUT_FILE${NC}"

# Cleanup
echo -e "\n${YELLOW}Cleanup...${NC}"
if [ "$KEEP_VM" = false ]; then
    echo "Stopping VM to save costs..."
    gcloud compute instances stop $INSTANCE_NAME --zone=$ZONE
    echo -e "${GREEN}VM stopped.${NC}"
    echo ""
    echo "To restart the VM:"
    echo "  gcloud compute instances start $INSTANCE_NAME --zone=$ZONE"
    echo ""
    echo "To delete the VM:"
    echo "  gcloud compute instances delete $INSTANCE_NAME --zone=$ZONE"
else
    echo -e "${GREEN}VM kept running as requested.${NC}"
fi

# Summary
echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                  Inference Complete!                       ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "Summary:"
echo "  ✓ Instance:       $INSTANCE_NAME (zone: $ZONE)"
echo "  ✓ Model:          $MODEL_DIR"
echo "  ✓ Input samples:  $(gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --command='python3.10 -c \"import numpy as np; print(np.load(\"input_data.npy\", allow_pickle=True).shape[0])\"' 2>/dev/null || echo 'N/A')"
if [ -z "$TASK" ]; then
    echo "  ✓ Output:         Features saved to $LOCAL_OUTPUT_DIR/$OUTPUT_FILE"
else
    echo "  ✓ Output:         $TASK predictions saved to $LOCAL_OUTPUT_DIR/$OUTPUT_FILE"
fi
echo ""
echo "Next Steps:"
echo "  1. Review results: $LOCAL_OUTPUT_DIR/$OUTPUT_FILE"
if [ -z "$TASK" ]; then
    echo "  2. Use extracted features for your downstream task"
else
    echo "  2. Analyze predictions for your use case"
fi
echo ""

# Estimate cost
RUNTIME_MINUTES=5  # Typical inference time
if [ "$GPU_TYPE" = "nvidia-tesla-v100" ]; then
    HOURLY_COST=2.10
elif [ "$GPU_TYPE" = "nvidia-tesla-t4" ]; then
    HOURLY_COST=0.60
else
    HOURLY_COST=1.00
fi

ESTIMATED_COST=$(echo "scale=2; $HOURLY_COST * $RUNTIME_MINUTES / 60" | bc)
echo "Estimated Cost: \$$ESTIMATED_COST (${RUNTIME_MINUTES}min at \$${HOURLY_COST}/hr)"
echo ""

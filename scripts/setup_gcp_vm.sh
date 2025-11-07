#!/bin/bash
#
# setup_gcp_vm.sh - Set up GCP VM with GPU for SSL-ECG training
#
# This script creates a GCP Compute Engine VM with GPU and installs
# all necessary dependencies for running the SSL-ECG experiments.
#
# Prerequisites:
# 1. gcloud CLI installed and authenticated
# 2. GCP project with billing enabled
# 3. Compute Engine API enabled
# 4. Sufficient GPU quota in your region
#
# Usage:
#   bash scripts/setup_gcp_vm.sh [INSTANCE_NAME] [ZONE] [GPU_TYPE]
#
# Examples:
#   # T4 GPU (budget option)
#   bash scripts/setup_gcp_vm.sh ssl-ecg-vm us-central1-a nvidia-tesla-t4
#
#   # V100 GPU (faster option)
#   bash scripts/setup_gcp_vm.sh ssl-ecg-vm us-central1-a nvidia-tesla-v100
#
# Cost estimates:
#   T4:  ~$0.50/hour  (~$6-8 for your experiment)
#   V100: ~$2.50/hour (~$15-20 for your experiment)

set -e

# Configuration
INSTANCE_NAME=${1:-"ssl-ecg-experiment"}
ZONE=${2:-"us-central1-a"}
GPU_TYPE=${3:-"nvidia-tesla-v100"}  # or "nvidia-tesla-t4" for budget
MACHINE_TYPE="n1-standard-4"  # 4 vCPUs, 15GB RAM
BOOT_DISK_SIZE="100GB"
IMAGE_FAMILY="ubuntu-2004-lts"
IMAGE_PROJECT="ubuntu-os-cloud"

echo "======================================================================"
echo "GCP GPU VM Setup for SSL-ECG"
echo "======================================================================"
echo "Instance name: $INSTANCE_NAME"
echo "Zone: $ZONE"
echo "GPU type: $GPU_TYPE"
echo "Machine type: $MACHINE_TYPE"
echo ""

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo "Error: gcloud CLI not found. Please install it first:"
    echo "  https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Check if authenticated
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" &> /dev/null; then
    echo "Error: Not authenticated. Run: gcloud auth login"
    exit 1
fi

# Get current project
PROJECT=$(gcloud config get-value project)
echo "Using project: $PROJECT"
echo ""

# Create VM instance with GPU
echo "Creating VM instance..."
gcloud compute instances create $INSTANCE_NAME \
    --zone=$ZONE \
    --machine-type=$MACHINE_TYPE \
    --accelerator="type=$GPU_TYPE,count=1" \
    --image-family=$IMAGE_FAMILY \
    --image-project=$IMAGE_PROJECT \
    --boot-disk-size=$BOOT_DISK_SIZE \
    --boot-disk-type=pd-ssd \
    --maintenance-policy=TERMINATE \
    --metadata=install-nvidia-driver=True \
    --scopes=cloud-platform \
    --tags=ssl-ecg

echo ""
echo "✓ VM instance created successfully!"
echo ""
echo "Waiting 30 seconds for VM to start..."
sleep 30

# Create startup script
STARTUP_SCRIPT="startup_script.sh"
cat > $STARTUP_SCRIPT <<'STARTUP_EOF'
#!/bin/bash
# Startup script for SSL-ECG VM

set -e

echo "======================================================================"
echo "Setting up SSL-ECG environment on GCP VM"
echo "======================================================================"

# Update system
sudo apt-get update
sudo apt-get install -y python3.6 python3.6-dev python3-pip git wget

# Install CUDA 10.0 (required for TensorFlow 1.14)
if ! command -v nvcc &> /dev/null; then
    echo "Installing CUDA 10.0..."
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
    sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
    wget https://developer.download.nvidia.com/compute/cuda/10.0.130/local_installers/cuda-repo-ubuntu1804-10-0-local-10.0.130-410.48_1.0-1_amd64.deb
    sudo dpkg -i cuda-repo-ubuntu1804-10-0-local-10.0.130-410.48_1.0-1_amd64.deb
    sudo apt-key add /var/cuda-repo-10-0-local-10.0.130-410.48/7fa2af80.pub
    sudo apt-get update
    sudo apt-get install -y cuda-10-0

    # Add to PATH
    echo 'export PATH=/usr/local/cuda-10.0/bin:$PATH' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
fi

# Set up Python 3.6 as default
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 1

# Install pip packages
pip3 install --upgrade pip
pip3 install tensorflow-gpu==1.14.0
pip3 install tensorboard==1.14.0
pip3 install scikit-learn==0.22.2
pip3 install numpy==1.18.4
pip3 install tqdm==4.36.1
pip3 install pandas==0.25.1
pip3 install mlxtend==0.17.0
pip3 install scipy==1.4.1
pip3 install opencv-python==4.2.0.34
pip3 install pytest==5.4.3

echo ""
echo "✓ Environment setup complete!"
echo ""
echo "Next steps:"
echo "1. Upload your data to this VM"
echo "2. Clone your repository"
echo "3. Run the fixed training script"
echo ""
STARTUP_EOF

# Copy startup script to VM and execute
echo "Setting up Python environment on VM..."
gcloud compute scp $STARTUP_SCRIPT $INSTANCE_NAME:~/ --zone=$ZONE
gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --command="bash ~/startup_script.sh"

# Clean up local startup script
rm $STARTUP_SCRIPT

echo ""
echo "======================================================================"
echo "Setup Complete!"
echo "======================================================================"
echo ""
echo "VM is ready! Connection details:"
echo ""
echo "SSH into VM:"
echo "  gcloud compute ssh $INSTANCE_NAME --zone=$ZONE"
echo ""
echo "Copy your data to VM:"
echo "  gcloud compute scp /local/path/to/data.npy $INSTANCE_NAME:~/data/ --zone=$ZONE"
echo ""
echo "Copy your code to VM:"
echo "  gcloud compute scp -r /local/path/to/SSL-ECGv2 $INSTANCE_NAME:~/ --zone=$ZONE"
echo ""
echo "Start training:"
echo "  gcloud compute ssh $INSTANCE_NAME --zone=$ZONE"
echo "  cd SSL-ECGv2/codes"
echo "  python3 train_fixed.py"
echo ""
echo "Monitor with TensorBoard (on VM):"
echo "  tensorboard --logdir=../summaries --host=0.0.0.0 --port=6006"
echo ""
echo "Access TensorBoard from local machine:"
echo "  gcloud compute ssh $INSTANCE_NAME --zone=$ZONE -- -L 6006:localhost:6006"
echo "  Then open http://localhost:6006 in your browser"
echo ""
echo "Stop VM when done (to save credits):"
echo "  gcloud compute instances stop $INSTANCE_NAME --zone=$ZONE"
echo ""
echo "Delete VM when finished:"
echo "  gcloud compute instances delete $INSTANCE_NAME --zone=$ZONE"
echo ""
echo "======================================================================"

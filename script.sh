#!/bin/bash
#SBATCH --job-name=singularity
#SBATCH --output=singularity-out-%u-%j.txt
#SBATCH --error=singularity-err-%u-%j.txt
#SBATCH --ntasks=1
#SBATCH --qos=1gpu
#SBATCH --partition=dgx1
#SBATCH --gpus=1

# Set up virtual environment
VENV_DIR=venv
echo "🔥 Job started on $(hostname)"
echo "📦 Setting up virtual environment..."

if [ ! -d "$VENV_DIR" ]; then
  python3 -m venv $VENV_DIR
  source $VENV_DIR/bin/activate
  echo "📦 Installing requirements..."
  pip install --upgrade pip
  pip install -r requirements.txt
else
  echo "✅ Virtual environment already exists"
  source $VENV_DIR/bin/activate
fi

# Data Augmentation
#echo "🚀 Running augmentation..."
export INPUT_DIR=/home/cluster-dgx1/iros03/laras/Progress-Bulan-Maret/skin-lesion-classification/data/raw
export OUTPUT_DIR=/home/cluster-dgx1/iros03/laras/Progress-Bulan-Maret/skin-lesion-classification/data/processed
#python3 scripts/augment.py --input_dir $INPUT_DIR --output_dir $OUTPUT_DIR

# Training
echo "🚀 Starting training..."
export DATA_DIR=$OUTPUT_DIR
export MODEL_DIR=/home/cluster-dgx1/iros03/laras/Progress-Bulan-Maret/skin-lesion-classification/models
python3 scripts/train.py

# Keep GPU alive (if needed)
echo "🔄 Training complete. Keeping GPU reservation alive until job timeout..."
while true; do sleep 1; done

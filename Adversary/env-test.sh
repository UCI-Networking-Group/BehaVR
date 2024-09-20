#!/bin/bash

# Initialize conda
echo "Initializing conda..."
#change folder as needed
source "...anaconda3/etc/profile.d/conda.sh"

# Activate the desired environment
echo "Activating the 'behavr' environment..."
conda activate behavr

# Run a test command
echo "Checking Python version in the activated environment..."
python --version

# Deactivate the environment after testing
echo "Deactivating the environment..."
conda deactivate

echo "Done!"

#!/bin/bash
# Setup script for running scMeformer on Hoffman2 (UCLA HPC)
# Tested with CUDA 11.8, PyTorch 2.1.2

module load anaconda3
module load cuda/11.8

conda create -n scmeformer 'mamba' python=3.10 -y
conda activate scmeformer

mamba install -c conda-forge 'gcc<=11' 'gxx' 'glib' -y
mamba install 'numpy<2' scipy scikit-learn pandas loompy h5py -y
mamba install 'setuptools<82' pip -y

conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118

# Install apex (2022 version required — current version removed amp)
# Clone apex separately before running this script:
# git clone https://github.com/NVIDIA/apex && cd apex && git checkout 2386a912164b0c5cfcd8be7a2b890fbac5607c82
cd apex
pip install -v --disable-pip-version-check \
    --no-cache-dir \
    --no-build-isolation \
    --config-settings "--build-option=--cpp_ext" \
    --config-settings "--build-option=--cuda_ext" \
    ./
cd ..

# Additional dependencies
PYTHONNOUSERSITE=1 pip install boto3 tensorboardx lmdb tqdm networkx community scanorama

# Verify installation
python -c "import apex; print('apex OK')"
python -c "import torch; print(torch.version.cuda); print(torch.cuda.is_available())"

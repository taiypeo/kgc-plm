#!/bin/bash
conda create -n kgc_plm python=3.11 --no-default-packages
conda activate kgc_plm
conda install -c pytorch -c nvidia pytorch pytorch-cuda=12.1
conda install -c pytorch -c nvidia faiss-gpu=1.8.0 cuda-toolkit=12.1.0
conda install -c milagraph -c conda-forge graphvite
conda install ipykernel

pip install black isort pylint mypy nltk scikit-learn sentence-transformers transformers tokenizers tqdm click wandb nvidia-cublas-cu12

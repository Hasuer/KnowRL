#!/bin/bash
# KnowRL Environment Setup Script
# Creates a conda environment named 'knowrl' with all required dependencies.
#
# Usage:
#   bash setup_env.sh
#
# After running this script:
#   conda activate knowrl
#   cd verl && pip install -e .

set -e

ENV_NAME="knowrl"
PYTHON_VERSION="3.10"
PYTORCH_INDEX="https://download.pytorch.org/whl/cu124"
PYPI_INDEX="https://pypi.org/simple/"

echo "============================================"
echo "  KnowRL Environment Setup"
echo "============================================"

# ---- Step 0: Remove existing env if any ----
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "[0/6] Removing existing '${ENV_NAME}' environment..."
    conda env remove -n ${ENV_NAME} -y
fi

# ---- Step 1: Create conda env ----
echo "[1/6] Creating conda environment '${ENV_NAME}' with Python ${PYTHON_VERSION}..."
conda create -n ${ENV_NAME} python=${PYTHON_VERSION} -y || \
    conda create -n ${ENV_NAME} python=${PYTHON_VERSION} -y --offline || \
    conda create -n ${ENV_NAME} python=${PYTHON_VERSION} -y --override-channels -c https://repo.anaconda.com/pkgs/main

# Activate
eval "$(conda shell.bash hook)"
conda activate ${ENV_NAME}

echo "  Python: $(python --version)"
echo "  pip:    $(pip --version)"

# ---- Step 2: Install PyTorch ----
echo "[2/6] Installing PyTorch (CUDA 12.4)..."
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --extra-index-url ${PYTORCH_INDEX}
pip install triton==3.2.0 --extra-index-url ${PYPI_INDEX}

# ---- Step 3: Install core ML packages ----
echo "[3/6] Installing Transformers, training, and data packages..."
pip install --extra-index-url ${PYPI_INDEX} \
    transformers==4.51.1 \
    tokenizers==0.21.1 \
    datasets==3.6.0 \
    accelerate==1.10.1 \
    peft==0.15.2 \
    sentencepiece==0.2.0 \
    sentence-transformers==4.1.0 \
    safetensors==0.5.3 \
    tiktoken==0.12.0 \
    huggingface-hub==0.32.4 \
    numpy==1.26.4 \
    pandas==2.2.3 \
    scipy==1.15.3 \
    scikit-learn==1.7.0 \
    matplotlib==3.10.3 \
    openpyxl==3.1.5 \
    pyarrow==20.0.0

# ---- Step 4: Install inference engines ----
echo "[4/6] Installing vLLM, SGLang, xformers, flash-attn..."
pip install --extra-index-url ${PYPI_INDEX} \
    vllm==0.8.5.post1 \
    sglang==0.4.6.post1 \
    sgl-kernel==0.1.0 \
    xformers==0.0.29.post2

# flash-attn needs --no-build-isolation for compilation
pip install flash-attn==2.7.4.post1 --no-build-isolation --extra-index-url ${PYPI_INDEX} || \
    echo "  [WARNING] flash-attn failed to compile. You may need to install it manually."

# av (PyAV) requires ffmpeg system libraries
pip install av==14.4.0 --extra-index-url ${PYPI_INDEX} || \
    echo "  [WARNING] av failed. Install ffmpeg dev libs: apt-get install -y libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev libavfilter-dev libswscale-dev libswresample-dev pkg-config"

# ---- Step 5: Install remaining dependencies ----
echo "[5/6] Installing remaining dependencies..."
pip install --extra-index-url ${PYPI_INDEX} \
    ray==2.46.0 \
    tensordict==0.10.0 \
    torchdata==0.11.0 \
    cloudpickle==3.1.1 \
    hydra-core==1.3.2 \
    omegaconf==2.3.0 \
    wandb==0.24.0 \
    tensorboard==2.20.0 \
    math-verify==0.7.0 \
    mathruler==0.1.0 \
    latex2sympy2-extended==1.10.1 \
    pylatexenc==2.10 \
    sympy==1.13.1 \
    fastapi==0.115.12 \
    uvicorn==0.34.3 \
    openai==1.84.0 \
    litellm==1.72.1 \
    pillow==11.2.1 \
    opencv-python-headless==4.11.0.86 \
    qwen-vl-utils==0.0.11 \
    decord==0.6.0 \
    einops==0.8.1 \
    tqdm==4.67.1 \
    pyyaml==6.0.2 \
    rich==14.0.0 \
    loguru==0.7.3 \
    pydantic==2.11.5 \
    requests==2.32.5 \
    filelock==3.18.0 \
    packaging==25.0 \
    regex==2024.11.6 \
    jinja2==3.1.6 \
    dill==0.3.8 \
    psutil==7.0.0

# ---- Step 6: Verify ----
echo "[6/6] Verifying installation..."
python -c "import torch; print(f'  PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
python -c "import transformers; print(f'  Transformers {transformers.__version__}')"
python -c "import vllm; print(f'  vLLM {vllm.__version__}')" 2>/dev/null || echo "  [WARNING] vllm import failed"
python -c "import ray; print(f'  Ray {ray.__version__}')"

echo ""
echo "============================================"
echo "  Step 2 complete! Dependencies installed."
echo ""
echo "  Please move to Step 3 to install verl:"
echo "    conda activate ${ENV_NAME}"
echo "    cd verl && pip install -e . --extra-index-url ${PYPI_INDEX}"
echo "============================================"

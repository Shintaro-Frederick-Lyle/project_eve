# NVIDIA PyTorch Optimized image (PyTorch 2.5, CUDA 12.6/13 compatible)
FROM nvcr.io/nvidia/pytorch:24.12-py3

ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST="12.0"
ENV FLASHINFER_SKIP_CUDA_CHECK=1
ENV MAX_JOBS=4
ENV SGLANG_DISABLE_DEEP_GEMM=1
ENV SGLANG_USE_TRITON_BACKEND=1

# 1. 必要なツールのインストール
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-dev ninja-build patchelf build-essential \
    && rm -rf /var/lib/apt/lists/*

# 2. CUDA 12 ABI Reconciliation
RUN pip install --no-cache-dir nvidia-cuda-runtime-cu12
RUN SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])") && \
    CUDA_RT_PATH=$(python3 -c "import nvidia.cuda_runtime; print(nvidia.cuda_runtime.__path__[0])")/lib && \
    find $SITE_PACKAGES -name "*.so" -exec sh -c \
    'ldd "$1" 2>/dev/null | grep -q "libcudart.so.12 => not found" \
    && patchelf --set-rpath "'$CUDA_RT_PATH'" "$1"' _ {} \;

# 3. FlashInfer & SGLang
# PyTorch 2.5環境になったため、公式の cu124/torch2.4 または cu124/torch2.5 互換ホイールがそのまま使えます。
# (NVIDIA 24.12はCUDA12.6ベースなので、cu124のホイールで完全に適合します)
RUN pip install --no-cache-dir flashinfer -U \
    --index-url https://flashinfer.ai/whl/cu124/torch2.5/

RUN pip install --no-cache-dir "sglang[all]>=0.5.10"

# 4. JAX/XLA管理
ENV XLA_PYTHON_CLIENT_MEM_FRACTION=0.30
ENV XLA_PYTHON_CLIENT_ALLOCATOR=cuda_malloc_async

WORKDIR /workspace/project_eve
EXPOSE 30000

ENTRYPOINT ["python3", "-m", "sglang.launch_server"]

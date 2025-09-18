FROM quay.io/rhoai-genaiops/llama-stack-vllm-milvus:0.2.11

RUN pip install --no-cache-dir accelerate llama-stack-provider-trustyai-fms

# RUN pip list

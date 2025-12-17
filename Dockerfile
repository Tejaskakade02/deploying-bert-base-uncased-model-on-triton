# FROM triton-sentiment:24.05_1
FROM nvcr.io/nvidia/tritonserver:24.09-py3

WORKDIR /workspace
RUN mkdir -p /models
# Copy model repository into container
COPY model_repository /models

# ---- Install Python dependencies ----
RUN pip install --no-cache-dir \
    numpy \
    torch \
    transformers \
    scikit-learn \
    tokenizers \
    sentencepiece

# Optional: set default env var (can override at runtime)
ENV MODEL_REPO_PATH=/models
#ENV ARTIFACTS_S3_PREFIX=s3://alltrainingdata/bert-t0
ENV METRICS_ENABLED=1
ENV METRICS_PORT=8002
ENV TOP_K=5
ENV METRICS_PORT=9000
# Expose Triton ports
EXPOSE 8000 8001 8002
# RUN pip install prometheus_client
# Entrypoint: start Triton with bundled models
ENTRYPOINT ["tritonserver", "--model-repository=/models"]

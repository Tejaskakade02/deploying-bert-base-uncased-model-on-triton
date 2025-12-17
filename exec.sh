# docker run --gpus '"device=0"' --rm -it   -p 8000:8000 -p 8001:8001 -p 8002:8002   -e ARTIFACTS_S3_PREFIX="s3://alltrainingdata/bert-t0"  529134784986.dkr.ecr.ap-south-1.amazonaws.com/q0-models:bert-prediction-v1
# docker run --gpus '"device=0"' --rm -it   -p 8000:8000 -p 8001:8001 -p 8002:8002   -e ARTIFACTS_S3_PREFIX="C:\Users\Tejas Kakade\OneDrive\Desktop\Triton deployment of bert model\dataset"   -v "$(pwd)/model_repository:/models" bert-prediction-v1
docker run --gpus '"device=0"' --rm -it \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v "$(pwd)/model_repository:/models" \
  -v "$(pwd)/dataset:/artifacts" \
  bert-prediction-v1

import os
import json
import torch
import numpy as np
import triton_python_backend_utils as pb_utils

from torch import nn
from transformers import BertTokenizer, BertModel
from sklearn.preprocessing import LabelEncoder
import pickle
import boto3
from botocore.exceptions import ClientError, NoCredentialsError, NoRegionError

def _get_param(model_config, key, default=""):
    params = model_config.get("parameters", {})
    if key in params and isinstance(params[key], dict):
        sv = params[key].get("string_value")
        if sv is not None and sv != "":
            return sv
    return os.environ.get(key, default)

def _get_env_required(key: str) -> str:
    v = os.environ.get(key)
    if not v:
        raise RuntimeError(f"Missing required env var: {key}")
    return v

def _download_from_s3_prefix(s3_uri: str, local_dir: str, filenames):
    if not s3_uri or not s3_uri.startswith("s3://"):
        raise RuntimeError("ARTIFACTS_S3_PREFIX is empty or not an s3:// URI.")
    os.makedirs(local_dir, exist_ok=True)
    bucket_key = s3_uri.replace("s3://", "", 1)
    if "/" in bucket_key:
        bucket, prefix = bucket_key.split("/", 1)
    else:
        bucket, prefix = bucket_key, ""

    try:
        s3 = boto3.client("s3")
    except (NoCredentialsError, NoRegionError) as e:
        raise RuntimeError(f"S3 client init failed (check AWS credentials/region): {e}")

    for fname in filenames:
        local_path = os.path.join(local_dir, fname)
        key = f"{prefix}/{fname}" if prefix else fname
        try:
            print(f"[model.py] Downloading s3://{bucket}/{key} -> {local_path}")
            s3.download_file(bucket, key, local_path)
        except ClientError as e:
            raise RuntimeError(f"Could not download {fname} from s3://{bucket}/{key}: {e}")

class SentimentClassifier(nn.Module):
    def __init__(self, n_classes: int, model_name="bert-base-uncased"):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.drop = nn.Dropout(p=0.3)
        self.out  = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.pooler_output
        return self.out(self.drop(pooled))

class TritonPythonModel:
    def initialize(self, args):
        torch.set_num_threads(max(1, int(os.environ.get("TRITON_TORCH_THREADS", "1"))))

        self.model_config = json.loads(args["model_config"])
        self.this_dir     = os.path.dirname(__file__)

        self.max_length    = int(_get_param(self.model_config, "MAX_LENGTH", "128"))
        self.hf_model_name = _get_param(self.model_config, "MODEL_NAME", "bert-base-uncased")
        self.local_dir     = _get_param(self.model_config, "LOCAL_ARTIFACTS_DIR", "/models/classification/1")

        self.artifacts_s3  = _get_env_required("ARTIFACTS_S3_PREFIX")
        self.hf_local_dir  = os.environ.get("HF_LOCAL_DIR", "")
        self.use_autocast  = bool(int(os.environ.get("USE_AUTOCAST", "1"))) and torch.cuda.is_available()

        print(f"[initialize] MAX_LENGTH={self.max_length}")
        print(f"[initialize] MODEL_NAME={self.hf_model_name}")
        print(f"[initialize] LOCAL_ARTIFACTS_DIR={self.local_dir}")
        print(f"[initialize] ARTIFACTS_S3_PREFIX={self.artifacts_s3}")
        if self.hf_local_dir:
            print(f"[initialize] HF_LOCAL_DIR={self.hf_local_dir}")

        _download_from_s3_prefix(
            self.artifacts_s3,
            self.local_dir,
            ["best_model.pt", "label_encoder.pkl"]
        )

        best_model_path = os.path.join(self.local_dir, "best_model.pt")
        label_enc_path  = os.path.join(self.local_dir, "label_encoder.pkl")

        with open(label_enc_path, "rb") as f:
            self.label_encoder: LabelEncoder = pickle.load(f)
        self.id_to_label = {i: lbl for i, lbl in enumerate(self.label_encoder.classes_)}
        n_classes = len(self.label_encoder.classes_)
        print(f"[initialize] label_encoder classes: {list(self.label_encoder.classes_)}")
        print(f"[initialize] n_classes = {n_classes}")

        tokenizer_src = self.hf_local_dir if self.hf_local_dir else self.hf_model_name
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_src)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentimentClassifier(n_classes=n_classes, model_name=tokenizer_src)

        self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
        print(f"[initialize] Loaded weights from {best_model_path}")
        self.model.to(self.device)
        self.model.eval()

        self.out_LOGITS = pb_utils.get_output_config_by_name(self.model_config, "LOGITS")
        self.out_IDS    = pb_utils.get_output_config_by_name(self.model_config, "CLASS_IDS")
        self.out_NAMES  = pb_utils.get_output_config_by_name(self.model_config, "CLASS_NAMES")
        self.out_PROBS  = pb_utils.get_output_config_by_name(self.model_config, "PROBS")

    def _predict_batch(self, texts):
        enc = self.tokenizer(
            list(texts),
            add_special_tokens=True,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        input_ids = enc["input_ids"].to(self.device)
        attn     = enc["attention_mask"].to(self.device)

        with torch.no_grad():
            if self.use_autocast:
                with torch.cuda.amp.autocast():
                    logits = self.model(input_ids=input_ids, attention_mask=attn)
            else:
                logits = self.model(input_ids=input_ids, attention_mask=attn)

            probs  = torch.softmax(logits, dim=-1)
            ids    = torch.argmax(probs, dim=-1)

        return logits.cpu().numpy(), probs.cpu().numpy(), ids.cpu().numpy().astype(np.int64)

    def _error_response(self, err_msg: str):
        return pb_utils.InferenceResponse(
            output_tensors=[],
            error=pb_utils.TritonError(err_msg)
        )

    def execute(self, requests):
        responses = []
        for request in requests:
            try:
                in_tensor = pb_utils.get_input_tensor_by_name(request, "TEXT")
                if in_tensor is None:
                    responses.append(self._error_response("Missing required input 'TEXT'."))
                    continue

                raw = in_tensor.as_numpy()
                if raw.ndim != 2 or raw.shape[1] != 1:
                    responses.append(self._error_response(
                        f"Unexpected 'TEXT' shape {list(raw.shape)}. Expected [batch, 1]."
                    ))
                    continue

                batch_texts = []
                for row in raw:
                    s = row[0]
                    if isinstance(s, (bytes, bytearray, np.bytes_)):
                        s = s.decode("utf-8", errors="replace")
                    batch_texts.append(str(s))

                logits, probs, ids = self._predict_batch(batch_texts)

                names = np.array([[self.id_to_label.get(int(i), str(int(i)))] for i in ids], dtype=object)

                out_logits = pb_utils.Tensor("LOGITS", logits.astype(np.float32))
                out_probs  = pb_utils.Tensor("PROBS", probs.astype(np.float32))
                out_ids    = pb_utils.Tensor("CLASS_IDS", ids.reshape(-1,1))
                out_names  = pb_utils.Tensor("CLASS_NAMES", names)

                responses.append(pb_utils.InferenceResponse(
                    output_tensors=[out_logits, out_ids, out_names, out_probs]
                ))
            except Exception as e:
                responses.append(self._error_response(f"Execution error: {e}"))
        return responses

    def finalize(self):
        print("[finalize] Model closed.")

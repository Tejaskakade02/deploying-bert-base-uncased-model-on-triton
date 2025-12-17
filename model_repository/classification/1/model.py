import os
import json
import time
import pickle
import numpy as np
import torch
from torch import nn
import triton_python_backend_utils as pb_utils

from transformers import (
    BertTokenizer,
    BertModel,
    BertForMaskedLM,
)
from sklearn.preprocessing import LabelEncoder

# Optional S3 support (only used if ARTIFACTS_S3_PREFIX is provided)
# import boto3
# from botocore.exceptions import ClientError, NoCredentialsError, NoRegionError

# -------------------------
# Optional Prometheus metrics
# -------------------------
_METRICS_ENABLED = bool(int(os.environ.get("METRICS_ENABLED", "0")))
_METRICS_PORT = int(os.environ.get("METRICS_PORT", "8002"))

_METRICS_AVAILABLE = False
_metrics_started = False

if _METRICS_ENABLED:
    try:
        from prometheus_client import Counter, Histogram, Gauge, start_http_server
        _METRICS_AVAILABLE = True
    except Exception as _e:
        print(f"[metrics] prometheus_client not available ({_e}). Metrics disabled.")
        _METRICS_ENABLED = False

REQUESTS_TOTAL = None
ERRORS_TOTAL = None
LATENCY_SEC = None
INPUT_BYTES = None
TOP1_CONFIDENCE = None
TOPK_GAUGE = None


def _start_metrics_server_if_needed():
    global _metrics_started, REQUESTS_TOTAL, ERRORS_TOTAL, LATENCY_SEC, INPUT_BYTES, TOP1_CONFIDENCE, TOPK_GAUGE
    if not _METRICS_ENABLED or not _METRICS_AVAILABLE:
        return
    if _metrics_started:
        return

    # Start exporter
    try:
        start_http_server(_METRICS_PORT)
        print(f"[metrics] Prometheus metrics server started on :{_METRICS_PORT}")
    except Exception as e:
        print(f"[metrics] Failed to start metrics server: {e}")
        return

    # Create metrics
    REQUESTS_TOTAL = Counter(
        "text_classification_requests_total",
        "Total inference requests",
        labelnames=("mode",),
    )
    ERRORS_TOTAL = Counter(
        "text_classification_errors_total",
        "Total errors during inference",
        labelnames=("mode",),
    )
    LATENCY_SEC = Histogram(
        "text_classification_latency_seconds",
        "End-to-end request latency (seconds)",
        labelnames=("mode",),
        buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10),
    )
    INPUT_BYTES = Histogram(
        "text_classification_input_bytes",
        "Total input bytes per request item",
        buckets=(1, 8, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 65536),
    )
    TOP1_CONFIDENCE = Gauge(
        "text_classification_top1_confidence",
        "Top-1 predicted probability of the last request",
        labelnames=("mode",),
    )
    TOPK_GAUGE = Gauge(
        "text_classification_topk_k",
        "Configured Top-K size",
    )

    _metrics_started = True


# -------------------------
# helpers
# -------------------------
def _get_param(model_config, key, default=""):
    params = model_config.get("parameters", {})
    if key in params and isinstance(params[key], dict):
        sv = params[key].get("string_value")
        if sv is not None and sv != "":
            return sv
    return os.environ.get(key, default)


def _optional_env(key: str, default: str = "") -> str:
    v = os.environ.get(key)
    return v if v is not None else default


# -------------------------
# classifier head (your original)
# -------------------------
class SentimentClassifier(nn.Module):
    def __init__(self, n_classes: int, model_name="bert-base-uncased"):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.pooler_output
        return self.out(self.drop(pooled))


# -------------------------
# Triton entry
# -------------------------
class TritonPythonModel:
    def initialize(self, args):
        # Limit PyTorch CPU threads inside Triton worker
        torch.set_num_threads(max(1, int(os.environ.get("TRITON_TORCH_THREADS", "1"))))

        self.model_config = json.loads(args["model_config"])

        # Config / params
        self.max_length = int(_get_param(self.model_config, "MAX_LENGTH", "128"))
        self.hf_model_name = _get_param(self.model_config, "MODEL_NAME", "bert-base-uncased")
        self.local_dir = _get_param(self.model_config, "LOCAL_ARTIFACTS_DIR", "/artifacts")
        self.hf_local_dir = _optional_env("HF_LOCAL_DIR", "")
        self.use_autocast = bool(int(os.environ.get("USE_AUTOCAST", "1"))) and torch.cuda.is_available()

        # S3 is optional: if missing/invalid or files not found → fallback to MLM
        self.artifacts_s3 = _optional_env("ARTIFACTS_S3_PREFIX", "")

        # Top-K for outputs (both modes)
        self.top_k = int(os.environ.get("TOP_K", "5"))

        print(f"[initialize] MAX_LENGTH={self.max_length}")
        print(f"[initialize] MODEL_NAME={self.hf_model_name}")
        print(f"[initialize] LOCAL_ARTIFACTS_DIR={self.local_dir}")
        print(f"[initialize] ARTIFACTS_S3_PREFIX={self.artifacts_s3 or '(none)'}")
        if self.hf_local_dir:
            print(f"[initialize] HF_LOCAL_DIR={self.hf_local_dir}")
        print(f"[initialize] TOP_K={self.top_k}")

        # Metrics server (shared across instances)
        _start_metrics_server_if_needed()
        if _METRICS_ENABLED and _METRICS_AVAILABLE:
            try:
                TOPK_GAUGE.set(self.top_k)
            except Exception:
                pass

        # Output configs
        self.out_LOGITS = pb_utils.get_output_config_by_name(self.model_config, "LOGITS")
        self.out_IDS = pb_utils.get_output_config_by_name(self.model_config, "CLASS_IDS")
        self.out_NAMES = pb_utils.get_output_config_by_name(self.model_config, "CLASS_NAMES")
        self.out_PROBS = pb_utils.get_output_config_by_name(self.model_config, "PROBS")

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Try to load fine-tuned classifier from S3; if it fails, fallback to default MLM
        self.use_default_mlm = False
        try:
            
            best_model_path = os.path.join(self.local_dir, "best_model.pt")
            label_enc_path = os.path.join(self.local_dir, "label_encoder.pkl")
            print("[DEBUG] local_dir =", self.local_dir)
            print("[DEBUG] best_model_path =", best_model_path)
            print("[DEBUG] label_enc_path =", label_enc_path)

            print("[DEBUG] best_model exists =", os.path.exists(best_model_path))
            print("[DEBUG] label_encoder exists =", os.path.exists(label_enc_path))
            try:
                print("[DEBUG] Contents of local_dir:", os.listdir(self.local_dir))
            except Exception as e:
                print("[DEBUG] Could not list local_dir:", e)

            with open(label_enc_path, "rb") as f:
                self.label_encoder: LabelEncoder = pickle.load(f)
            with open(label_enc_path, "rb") as f:
                self.label_encoder: LabelEncoder = pickle.load(f)
            self.id_to_label = {i: lbl for i, lbl in enumerate(self.label_encoder.classes_)}
            n_classes = len(self.label_encoder.classes_)
            print(f"[initialize] label_encoder classes: {list(self.label_encoder.classes_)}")
            print(f"[initialize] n_classes = {n_classes}")

            tokenizer_src = self.hf_local_dir if self.hf_local_dir else self.hf_model_name
            self.tokenizer = BertTokenizer.from_pretrained(tokenizer_src)

            self.classifier = SentimentClassifier(n_classes=n_classes, model_name=tokenizer_src)
            self.classifier.load_state_dict(torch.load(best_model_path, map_location=self.device))
            self.classifier.to(self.device)
            self.classifier.eval()
            print("[initialize] Fine-tuned classifier loaded from local artifacts.")


            self.mode = "classifier"
            self.n_classes = n_classes

        except Exception as e:
            # Fallback path: default pretrained BERT for masked language modeling
            print(f"[initialize] Falling back to default BERT MLM. Reason: {e}")
            self.use_default_mlm = True

            tokenizer_src = self.hf_local_dir if self.hf_local_dir else self.hf_model_name
            self.tokenizer = BertTokenizer.from_pretrained(tokenizer_src)
            self.mlm_model = BertForMaskedLM.from_pretrained(tokenizer_src)
            self.mlm_model.to(self.device)
            self.mlm_model.eval()

            self.mode = "mlm"

    # -------------------------
    # classifier path (returns TOP-K)
    # -------------------------
    def _predict_classifier_topk_batch(self, texts, top_k):
        enc = self.tokenizer(
            list(texts),
            add_special_tokens=True,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(self.device)
        attn = enc["attention_mask"].to(self.device)

        with torch.no_grad():
            if self.use_autocast and self.device.type == "cuda":
                with torch.cuda.amp.autocast():
                    logits = self.classifier(input_ids=input_ids, attention_mask=attn)
            else:
                logits = self.classifier(input_ids=input_ids, attention_mask=attn)

            probs = torch.softmax(logits, dim=-1)

        k = min(top_k, probs.shape[-1])
        topk_vals, topk_idx = torch.topk(probs, k=k, dim=-1)

        # Map indices -> labels
        names = []
        for row in topk_idx.cpu().numpy():
            names.append([self.id_to_label.get(int(i), str(int(i))) for i in row])

        return (
            topk_vals.cpu().numpy().astype(np.float32),        # [B, K] probs (also used as LOGITS)
            topk_idx.cpu().numpy().astype(np.int64),           # [B, K] ids
            np.array(names, dtype=object),                     # [B, K] names
        )

    # -------------------------
    # MLM fallback path (returns TOP-K for the [MASK])
    # -------------------------
    def _predict_mlm_topk_batch(self, texts, top_k):
        """
        For each input text, expects one [MASK] token. Returns arrays:
          - scores: (B, K) probabilities
          - ids:    (B, K) token ids
          - tokens: (B, K) token strings
        If no [MASK] found in an item, returns zeros/-1/"" for that row.
        """
        enc = self.tokenizer(
            list(texts),
            add_special_tokens=True,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        mask_id = self.tokenizer.mask_token_id
        batch_size = input_ids.size(0)

        all_scores = []
        all_ids = []
        all_tokens = []

        with torch.no_grad():
            if self.use_autocast and self.device.type == "cuda":
                with torch.cuda.amp.autocast():
                    outputs = self.mlm_model(input_ids=input_ids, attention_mask=attention_mask)
            else:
                outputs = self.mlm_model(input_ids=input_ids, attention_mask=attention_mask)
            # logits: [B, T, V]
            logits = outputs.logits

        for i in range(batch_size):
            ids_row = input_ids[i]
            mask_positions = (ids_row == mask_id).nonzero(as_tuple=False)
            if mask_positions.numel() == 0:
                all_scores.append(np.zeros((top_k,), dtype=np.float32))
                all_ids.append(np.full((top_k,), -1, dtype=np.int64))
                all_tokens.append([""] * top_k)
                continue

            pos = int(mask_positions[0].item())  # use the first [MASK]
            vocab_logits = logits[i, pos, :]     # [V]
            probs = torch.softmax(vocab_logits, dim=-1)

            k = min(top_k, probs.shape[-1])
            topk = torch.topk(probs, k=k)
            topk_indices = topk.indices.cpu().numpy()
            topk_values = topk.values.cpu().numpy()

            all_scores.append(topk_values.astype(np.float32))
            all_ids.append(topk_indices.astype(np.int64))
            all_tokens.append([self.tokenizer.decode([idx]).strip() for idx in topk_indices])

        return (
            np.array(all_scores, dtype=np.float32),   # [B, K]
            np.array(all_ids, dtype=np.int64),        # [B, K]
            np.array(all_tokens, dtype=object),       # [B, K]
        )

    # -------------------------
    # Triton plumbing
    # -------------------------
    def _error_response(self, err_msg: str, mode_for_err: str):
        if _METRICS_ENABLED and _METRICS_AVAILABLE and ERRORS_TOTAL is not None:
            try:
                ERRORS_TOTAL.labels(mode_for_err).inc()
            except Exception:
                pass

        return pb_utils.InferenceResponse(
            output_tensors=[],
            error=pb_utils.TritonError(err_msg),
        )

    def execute(self, requests):
        responses = []
        mode = "mlm" if self.use_default_mlm else "classifier"

        for request in requests:
            start = time.time()
            try:
                server_start = time.perf_counter()
                in_tensor = pb_utils.get_input_tensor_by_name(request, "TEXT")
                if in_tensor is None:
                    responses.append(self._error_response("Missing required input 'TEXT'.", mode))
                    continue

                raw = in_tensor.as_numpy()
                if raw.ndim != 2 or raw.shape[1] != 1:
                    responses.append(
                        self._error_response(
                            f"Unexpected 'TEXT' shape {list(raw.shape)}. Expected [batch, 1].", mode
                        )
                    )
                    continue

                batch_texts = []
                for row in raw:
                    s = row[0]
                    if isinstance(s, (bytes, bytearray, np.bytes_)):
                        s = s.decode("utf-8", errors="replace")
                    s = str(s)
                    batch_texts.append(s)
                    if _METRICS_ENABLED and _METRICS_AVAILABLE and INPUT_BYTES is not None:
                        try:
                            INPUT_BYTES.observe(len(s.encode("utf-8")))
                        except Exception:
                            pass

                if self.use_default_mlm:
                    # MLM fallback: top-k predictions at [MASK]
                    scores, ids, tokens = self._predict_mlm_topk_batch(batch_texts, self.top_k)

                    out_logits = pb_utils.Tensor("LOGITS", scores.astype(np.float32))
                    out_probs  = pb_utils.Tensor("PROBS",  scores.astype(np.float32))
                    out_ids    = pb_utils.Tensor("CLASS_IDS", ids.astype(np.int64))
                    out_names  = pb_utils.Tensor("CLASS_NAMES", tokens.astype(object))

                    responses.append(
                        pb_utils.InferenceResponse(
                            output_tensors=[out_logits, out_ids, out_names, out_probs]
                        )
                    )

                    # top1 confidence (batch average)
                    if _METRICS_ENABLED and _METRICS_AVAILABLE and TOP1_CONFIDENCE is not None:
                        try:
                            if scores.size > 0:
                                top1 = scores[:, 0].mean().item()
                                TOP1_CONFIDENCE.labels(mode).set(float(top1))
                        except Exception:
                            pass
                else:
                    # Fine-tuned classifier: top-k classes
                    scores, ids, names = self._predict_classifier_topk_batch(batch_texts, self.top_k)

                    out_logits = pb_utils.Tensor("LOGITS", scores.astype(np.float32))
                    out_probs  = pb_utils.Tensor("PROBS",  scores.astype(np.float32))
                    out_ids    = pb_utils.Tensor("CLASS_IDS", ids.astype(np.int64))
                    out_names  = pb_utils.Tensor("CLASS_NAMES", names.astype(object))
                       # -------------------------------
                        # Server-side timing END
                       # -------------------------------
                server_end = time.perf_counter()
                server_time_ms = (server_end - server_start) * 1000.0

                out_server_time = pb_utils.Tensor(
                     "SERVER_TIME_MS",
                    np.array([[server_time_ms]], dtype=np.float32),
                   )

                responses.append(
                        pb_utils.InferenceResponse(
                            output_tensors=[out_logits, out_ids, out_names, out_probs,out_server_time]
                        )
                    )

                if _METRICS_ENABLED and _METRICS_AVAILABLE and TOP1_CONFIDENCE is not None:
                        try:
                            if scores.size > 0:
                                top1 = scores[:, 0].mean().item()
                                TOP1_CONFIDENCE.labels(mode).set(float(top1))
                        except Exception:
                            pass

                # success counters & latency
                if _METRICS_ENABLED and _METRICS_AVAILABLE:
                    try:
                        if REQUESTS_TOTAL is not None:
                            REQUESTS_TOTAL.labels(mode).inc()
                        if LATENCY_SEC is not None:
                            LATENCY_SEC.labels(mode).observe(time.time() - start)
                    except Exception:
                        pass

            except Exception as e:
                responses.append(self._error_response(f"Execution error: {e}", mode))
        return responses

    def finalize(self):
        print("[finalize] Model closed.")

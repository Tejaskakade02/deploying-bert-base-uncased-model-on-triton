# 🚀 Triton BERT Text Classification with Server Latency

This project demonstrates how to deploy a **fine-tuned BERT text classification model** using **NVIDIA Triton Inference Server (Python Backend)**, with:

* ✅ Top-K class predictions
* ✅ Confidence scores
* ✅ **Pure server-side response time (ms)**
* ✅ Optional Prometheus metrics
* ✅ Docker-based deployment

---

## 📌 Features

* **BERT-based text classifier**
* Supports **batch inference**
* Returns:

  * `CLASS_IDS`
  * `CLASS_NAMES`
  * `PROBS`
  * `SERVER_TIME_MS` (true server processing time)
* Fallback to **BERT MLM** if fine-tuned artifacts are missing
* Runs on **CPU or GPU**
* Compatible with **Windows + Docker Desktop**

---

## 📂 Project Structure

```
Triton deployment of bert model/
│
├── model_repository/
│   └── classification/
│       ├── 1/
│       │   └── model.py
│       └── config.pbtxt
│
├── dataset/
│   ├── best_model.pt
│   └── label_encoder.pkl
│
├── test2.py
├── Dockerfile
└── README.md
```

---

## 🧠 Model Artifacts

Place the following files in the `dataset/` directory:

* `best_model.pt` → fine-tuned BERT classifier weights
* `label_encoder.pkl` → sklearn `LabelEncoder` used during training

These files are **mounted into the container** at runtime.

---

## ⚙️ Triton Model Configuration (`config.pbtxt`)

Ensure the following outputs are declared:

```protobuf
output {
  name: "LOGITS"
  data_type: TYPE_FP32
  dims: [ -1, -1 ]
}

output {
  name: "CLASS_IDS"
  data_type: TYPE_INT64
  dims: [ -1, -1 ]
}

output {
  name: "CLASS_NAMES"
  data_type: TYPE_STRING
  dims: [ -1, -1 ]
}

output {
  name: "PROBS"
  data_type: TYPE_FP32
  dims: [ -1, -1 ]
}

output {
  name: "SERVER_TIME_MS"
  data_type: TYPE_FP32
  dims: [ 1 ]
}
```

⚠️ `SERVER_TIME_MS` **must be declared**, otherwise the client will receive `None`.

---

## 🐳 Docker Build

Build the Triton image:

```bash
docker build -t bert-prediction-v1 .
```

---

## ▶️ Run Triton Server

```bash
docker run --gpus '"device=0"' --rm -it \
  -p 8000:8000 \
  -p 8001:8001 \
  -p 8002:8002 \
  -v "C:/Users/Tejas Kakade/OneDrive/Desktop/Triton deployment of bert model/model_repository:/models" \
  -v "C:/Users/Tejas Kakade/OneDrive/Desktop/Triton deployment of bert model/dataset:/artifacts" \
  bert-prediction-v1
```

### Ports

| Port | Purpose            |
| ---- | ------------------ |
| 8000 | HTTP inference     |
| 8001 | GRPC               |
| 8002 | Prometheus metrics |

---

## 🧪 Client Testing

Run the client:

```bash
python test2.py
```

### Example Output

```
Enter text to classify: hello
Predicted Class: Greeting (ID: 1, Confidence: 80.03%) | Server Time: 12.41 ms
```

---

## ⏱️ Server Response Time (Important)

* `SERVER_TIME_MS` measures **only server-side processing**
* Includes:

  * Tokenization
  * Model forward pass
  * Output generation
* Excludes:

  * Client overhead
  * Network latency

This timing is measured **inside `model.py`** using `time.perf_counter()`.

---

## 📊 Prometheus Metrics (Optional)

If enabled, metrics are available at:

```
http://localhost:8002/metrics
```

Key metrics:

* `text_classification_latency_seconds`
* `text_classification_requests_total`
* `text_classification_top1_confidence`

---

## 🔄 Common Troubleshooting

### ❌ `SERVER_TIME_MS is None`

✔ Ensure:

* Output is declared in `config.pbtxt`
* Triton container is restarted
* Client requests the output

---

### ❌ Model falls back to MLM

✔ Check:

* `best_model.pt` exists
* `label_encoder.pkl` exists
* Correct volume mount path

---

### ❌ Triton fails to load model

✔ Run:

```bash
docker logs <container_id>
```

Look for Python or indentation errors.

---

## 🧹 Cleanup Docker (Optional)

To fully reset Docker:

```bash
docker system prune -a --volumes
```

⚠️ This deletes all containers, images, and caches.

---

## 📌 Tech Stack

* Python 3.10
* PyTorch
* Hugging Face Transformers
* NVIDIA Triton Inference Server
* Docker
* Prometheus (optional)

---

## ✅ Status

✔ Production-ready
✔ Server-side latency enabled
✔ GPU compatible

---

If you want, I can also provide:

* 📈 Performance benchmarking
* 🔍 GPU-only latency measurement
* 📦 Minimal Triton config
* 🧠 Model optimization tips

Just let me know 👍

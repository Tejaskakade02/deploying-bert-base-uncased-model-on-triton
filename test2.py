import tritonclient.http as httpclient
import numpy as np
import time

TRITON_URL = "localhost:8000"
MODEL_NAME = "classification"

def _decode_one(x):
    if isinstance(x, (bytes, bytearray, np.bytes_)):
        return x.decode("utf-8", errors="replace")
    return str(x)

# Create Triton HTTP client
client = httpclient.InferenceServerClient(url=TRITON_URL)

while True:
    user_text = input("Enter text to classify (or type 'exit' to quit): ").strip()
    if user_text.lower() in {"exit", "quit"}:
        break
    if not user_text:
        print("⚠️ Please enter a non-empty text.")
        continue

    # ------------------------------
    # Prepare input
    # ------------------------------
    input_data = np.array([[user_text]], dtype=object)

    inp = httpclient.InferInput("TEXT", input_data.shape, "BYTES")
    inp.set_data_from_numpy(input_data)

    outs = [
        httpclient.InferRequestedOutput("CLASS_IDS"),
        httpclient.InferRequestedOutput("CLASS_NAMES"),
        httpclient.InferRequestedOutput("PROBS"),
        httpclient.InferRequestedOutput("SERVER_TIME_MS")
    ]

    # ------------------------------
    # Measure response time
    # ------------------------------
    resp = client.infer(MODEL_NAME, [inp], outputs=outs)
    server_time = resp.as_numpy("SERVER_TIME_MS")
    server_time_ms = float(server_time[0][0])
    # ------------------------------
    # Parse CLASS_IDS (top-1)
    # ------------------------------
    class_ids = resp.as_numpy("CLASS_IDS")
    class_id = int(class_ids[0][0])

    # ------------------------------
    # Parse CLASS_NAMES (top-1)
    # ------------------------------
    class_names = resp.as_numpy("CLASS_NAMES")
    class_name = _decode_one(class_names[0][0])

    # ------------------------------
    # Parse PROBS (top-1 confidence)
    # ------------------------------
    probs = resp.as_numpy("PROBS")
    confidence = float(probs[0][0]) * 100.0

    # ------------------------------
    # Output
    # ------------------------------
    print(
        f"Predicted Class: {class_name} "
        f"(ID: {class_id}, Confidence: {confidence:.2f}%) | "
        f"server Response Time: {server_time_ms:.2f} ms\n"
    )



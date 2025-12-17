import tritonclient.http as httpclient
import numpy as np

TRITON_URL = "localhost:5000"
MODEL_NAME = "classification"

def _decode_one(x):
    if isinstance(x, (bytes, bytearray, np.bytes_)):
        return x.decode("utf-8", errors="replace")
    return str(x)

client = httpclient.InferenceServerClient(url=TRITON_URL)

while True:
    user_text = input("Enter text to classify (or type 'exit' to quit): ").strip()
    if user_text.lower() in {"exit", "quit"}:
        break
    if not user_text:
        print("⚠️ Please enter a non-empty text.")
        continue

    # [batch, 1]
    input_data = np.array([[user_text]], dtype=object)

    inp = httpclient.InferInput("TEXT", input_data.shape, "BYTES")
    inp.set_data_from_numpy(input_data)

    outs = [
        httpclient.InferRequestedOutput("CLASS_IDS"),
        httpclient.InferRequestedOutput("CLASS_NAMES"),
        httpclient.InferRequestedOutput("PROBS"),
    ]

    resp = client.infer(MODEL_NAME, [inp], outputs=outs)

    # Parse outputs
    class_ids   = resp.as_numpy("CLASS_IDS")[0]     # shape (K,)
    class_names = resp.as_numpy("CLASS_NAMES")[0]   # shape (K,)
    probs       = resp.as_numpy("PROBS")[0]         # shape (K,)

    print("\nTop-K Predictions:")
    for i, (cid, cname, prob) in enumerate(zip(class_ids, class_names, probs)):
        cname = _decode_one(cname)
        print(f"  {i+1}. {cname} (ID: {int(cid)}, Confidence: {prob*100:.2f}%)")
    print()

import tritonclient.http as httpclient
import numpy as np
import pandas as pd

TRITON_URL = "localhost:8000"
MODEL_NAME = "classification"

texts = [ "I love this product!",
         "This is the worst experience ever." 
        ]


# Shape must be [batch, 1] for TEXT
input_data = np.array(texts, dtype=object).reshape(-1, 1)

client = httpclient.InferenceServerClient(url=TRITON_URL)

inp = httpclient.InferInput("TEXT", input_data.shape, "BYTES")
inp.set_data_from_numpy(input_data)

outs = [
    httpclient.InferRequestedOutput("CLASS_IDS"),
    httpclient.InferRequestedOutput("CLASS_NAMES"),
    httpclient.InferRequestedOutput("PROBS"),
]

resp = client.infer(MODEL_NAME, [inp], outputs=outs)
# ----- CLASS IDS -----
batch_size = len(texts)

# ---- CLASS IDS ----
class_ids = resp.as_numpy("CLASS_IDS").reshape(-1).tolist()

# ---- CLASS NAMES ----
class_names_raw = resp.as_numpy("CLASS_NAMES").reshape(-1)
class_names = [
    x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else str(x)
    for x in class_names_raw
]

# ---- PROBS ----
probs = resp.as_numpy("PROBS")

if probs.ndim == 2:
    probs = probs
elif probs.ndim == 3:
    probs = probs[0]
else:
    raise ValueError(f"Unexpected PROBS shape: {probs.shape}")

confidence = [f"{float(p.max())*100:.2f}%" for p in probs]

# ---- FIX LENGTH MISMATCH ----
def fix_length(arr, target_len):
    if len(arr) == target_len:
        return arr
    if len(arr) == 1:
        return arr * target_len
    raise ValueError(f"Cannot fix length {len(arr)} to {target_len}")

class_ids = fix_length(class_ids, batch_size)
class_names = fix_length(class_names, batch_size)
confidence = fix_length(confidence, batch_size)

# ---- FINAL ASSERT ----
assert len(texts) == len(class_ids) == len(class_names) == len(confidence)

df = pd.DataFrame({
    "Text": texts,
    "Predicted Class": class_names,
    "Class ID": class_ids,
    "Confidence": confidence,
})

print(df)



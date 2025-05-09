# ─── 0. (Optional) install/upgrade dependencies ─────────────────────────
# !pip install --upgrade sagemaker transformers

# ─── 1. Imports & Session Setup ────────────────────────────────────────
import sagemaker
from sagemaker import get_execution_role
from sagemaker.huggingface import HuggingFaceModel
import boto3
import json

# SageMaker session & role (auto-detects the notebook/job role)
sagemaker_session = sagemaker.Session()
role = get_execution_role()  # must match the assumed role exactly

# ─── 2. HF Hub Configuration ───────────────────────────────────────────
hub = {
    "HF_MODEL_ID":    "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
    "HF_TASK":        "feature-extraction",   # embedding extraction
}

# ─── 3. Create the SageMaker HuggingFaceModel ─────────────────────────
hf_model = HuggingFaceModel(
    env=hub,
    role=role,
    transformers_version="4.37.0",  # pick one your SDK supports
    pytorch_version="2.1.0",
    py_version="py310",
    sagemaker_session=sagemaker_session,
)

# ─── 4. Deploy to an endpoint ──────────────────────────────────────────
endpoint_name = "qwen2-embedding-endpoint"

predictor = hf_model.deploy(
    initial_instance_count=1,
    instance_type="ml.g5.xlarge",   # or ml.g5.2xlarge for more memory/GPU
    endpoint_name=endpoint_name,
)

print(f"Endpoint '{endpoint_name}' is in service!")

# ─── 5. Invoke the endpoint ───────────────────────────────────────────
# (a) via the SageMaker Predictor
texts = ["The quick brown fox jumps over the lazy dog.",
         "SageMaker + HuggingFace FTW!"]
embeddings = predictor.predict(texts)
print("First vector (truncated):", embeddings[0][:8])

# (b) or via the low-level Runtime API
sm_rt = boto3.client("sagemaker-runtime")
resp = sm_rt.invoke_endpoint(
    EndpointName=endpoint_name,
    ContentType="application/json",
    Body=json.dumps(texts),
)
rt_emb = json.loads(resp["Body"].read().decode())
print("RT vector (truncated):", rt_emb[0][:8])

# ─── 6. (Optional) Clean up ────────────────────────────────────────────
predictor.delete_endpoint()


# ─── 0. (Optional) install/upgrade dependencies ─────────────────────────
# !pip install --upgrade sagemaker transformers
'''
Install AWS console
Get Creds from IAM 
Set Cred in AWS by Export
Like This
export AWS_ACCESS_KEY_ID=AKIA…YOURNEWKEY
export AWS_SECRET_ACCESS_KEY=…YOURNEWSECRET
export AWS_DEFAULT_REGION=us-east-1
'''
# ─── 1. Imports & Session Setup ────────────────────────────────────────
import sagemaker
from sagemaker import get_execution_role
from sagemaker.huggingface import HuggingFaceModel
import boto3
import json
# create a boto3 session from a named profile
boto_session = boto3.Session(region_name="us-east-1")

# pass that into SageMaker
sagemaker_session = sagemaker.Session(boto_session=boto_session)
#role = get_execution_role()  # still needs a valid SageMaker execution-role ARN
execution_role = "arn:aws:iam::601136796356:role/MySageMakerExecutionRole"
#arn:aws:iam::601136796356:role/MySageMakerExecutionRole
# SageMaker session & role (auto-detects the notebook/job role)
sagemaker_session = sagemaker.Session()
role = execution_role  # must match the assumed role exactly

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
# texts = ["The quick brown fox jumps over the lazy dog.",
#          "SageMaker + HuggingFace FTW!"]
# embeddings = predictor.predict(texts)
# print("First vector (truncated):", embeddings[0][:8])

# # (b) or via the low-level Runtime API
# sm_rt = boto3.client("sagemaker-runtime")
# resp = sm_rt.invoke_endpoint(
#     EndpointName=endpoint_name,
#     ContentType="application/json",
#     Body=json.dumps(texts),
# )
# rt_emb = json.loads(resp["Body"].read().decode())
# print("RT vector (truncated):", rt_emb[0][:8])
texts = [
    "The quick brown fox jumps over the lazy dog.",
    "SageMaker + HuggingFace FTW!"
]
# wrap in {"inputs": …}
payload = {"inputs": texts}

embeddings = predictor.predict(payload)
print("First vector (truncated):", embeddings[0][:8])
# ─── 6. (Optional) Clean up ────────────────────────────────────────────
predictor.delete_endpoint()


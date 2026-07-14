"""Pre-download Hugging Face models during container build or warm-up."""
import os
import sys
from pathlib import Path

os.environ["TRANSFORMERS_OFFLINE"] = "0"
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ.setdefault("HF_HOME", str((Path.cwd() / ".hf").resolve()))

from huggingface_hub import snapshot_download

models = [
    "OpenMuQ/MuQ-large-msd-iter",
]

for model_id in models:
    print(f"Downloading {model_id} ...")
    snapshot_download(model_id, cache_dir=os.environ["HF_HOME"])
    print(f"Done: {model_id}")

print("All models downloaded successfully.")

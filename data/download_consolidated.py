import os
from huggingface_hub import HfApi, hf_hub_download
from tqdm import tqdm

HF_REPO_ID = "lalit3c/OA_Domain_Concepts"
HF_TOKEN = os.getenv("HF_TOKEN")

LOCAL_DIR = "dataset_local"
os.makedirs(LOCAL_DIR, exist_ok=True)

api = HfApi()

print("Fetching file list...")
files = api.list_repo_files(repo_id=HF_REPO_ID, repo_type="dataset", token=HF_TOKEN)

consolidated_files = [f for f in files if f.startswith("consolidated/") and f.endswith(".duckdb")]

print(f"Found {len(consolidated_files)} consolidated files")

for f in tqdm(consolidated_files):
    # check if file already exists
    local_path = os.path.join(LOCAL_DIR, f)
    if os.path.exists(local_path):
        print(f"File {f} already exists, skipping download.")
        continue
    hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=f,
        repo_type="dataset",
        token=HF_TOKEN,
        local_dir=LOCAL_DIR,
        local_dir_use_symlinks=False
    )

print("Download complete.")
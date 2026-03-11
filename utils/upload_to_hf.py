from huggingface_hub import HfApi, hf_hub_download, list_repo_files
import os
HF_REPO_ID = "lalit3c/OA_Domain_Concepts"
HF_TOKEN = os.getenv("HF_TOKEN")
assert HF_TOKEN, "HF_TOKEN environment variable not set"

api = HfApi()


# configuration
local_file = "results/concept_summaries_citation_k10.json"             # file on your machine
path_in_repo = "data/concept_summaries_citation_k10.json"      # where it will appear in the dataset repo

api = HfApi()

api.upload_file(
    path_or_fileobj=local_file,
    path_in_repo=path_in_repo,
    repo_id=HF_REPO_ID,
    token = HF_TOKEN,
    repo_type="dataset"
)

print("Upload complete.")

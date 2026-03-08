import os
import duckdb
from huggingface_hub import HfApi, hf_hub_download, list_repo_files
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import shutil

# -------------------------------
# Configuration
# -------------------------------
HF_REPO_ID = "lalit3c/OA_Domain_Concepts"
HF_TOKEN = os.getenv("HF_TOKEN")
assert HF_TOKEN, "HF_TOKEN environment variable not set"

MAX_CONSOLIDATED_SIZE = 5 * 1024**3  # 5 GB per consolidated file
LOCAL_TEMP_DIR = os.environ.get("TMPDIR", "./temp_downloads")
CONSOLIDATED_DIR = os.environ.get("TMPDIR", "./consolidated_batches")
HF_CONSOLIDATED_FOLDER = "consolidated"

os.makedirs(LOCAL_TEMP_DIR, exist_ok=True)
os.makedirs(CONSOLIDATED_DIR, exist_ok=True)
os.makedirs("logs", exist_ok=True)

api = HfApi()

# -------------------------------
# DuckDB initialization
# -------------------------------
def init_consolidated_db(path: str):
    con = duckdb.connect(path)
    con.execute("""
        CREATE TABLE IF NOT EXISTS papers (
            id VARCHAR PRIMARY KEY,
            doi VARCHAR,
            title VARCHAR,
            abstract VARCHAR,
            publication_year INTEGER,
            publication_date VARCHAR,
            type VARCHAR,
            cited_by_count INTEGER,
            counts_by_year JSON,
            num_matched_concepts INTEGER,
            processed_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS concept_papers (
            concept VARCHAR,
            paper_id VARCHAR,
            PRIMARY KEY (concept, paper_id)
        )
    """)
    return con

def get_db_size(path):
    return os.path.getsize(path)

# -------------------------------
# Download helper
# -------------------------------
def download_batch(batch_file):
    return hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=batch_file,
        repo_type="dataset",
        token=HF_TOKEN,
        local_dir=LOCAL_TEMP_DIR,
        local_dir_use_symlinks=False
    )

# -------------------------------
# Fetch batch list
# -------------------------------
all_files = list_repo_files(repo_id=HF_REPO_ID, repo_type="dataset", token=HF_TOKEN)
batch_files = sorted([f for f in all_files if f.startswith("batches/") and f.endswith(".duckdb")])
print(f"Found {len(batch_files)} batch files on HF")

# -------------------------------
# Initialize consolidated DB
# -------------------------------
consolidated_num = 1
current_consolidated_path = os.path.join(CONSOLIDATED_DIR, f"consolidated_{consolidated_num:04d}.duckdb")
consolidated_con = init_consolidated_db(current_consolidated_path)

# -------------------------------
# Consolidation loop
# -------------------------------
BATCHES_PER_GROUP = 20  # adjust based on ~5GB target

print(f"Starting consolidation (~{MAX_CONSOLIDATED_SIZE/1e9:.1f} GB per file)")

for i in range(0, len(batch_files), BATCHES_PER_GROUP):
    group = batch_files[i:i+BATCHES_PER_GROUP]

    # 1️⃣ Download group in parallel
    downloaded_paths = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        future_to_file = {executor.submit(download_batch, f): f for f in group}
        for future in tqdm(as_completed(future_to_file), total=len(group), desc="Downloading batches"):
            try:
                path = future.result()
                downloaded_paths.append(path)
            except Exception as e:
                print(f"Failed to download {future_to_file[future]}: {e}")

    if not downloaded_paths:
        continue

    # 2️⃣ Attach each batch with unique alias + insert
    for path in downloaded_paths:
        batch_name = os.path.basename(path)
        alias = f"src_{batch_name.replace('.', '_')}"
        try:
            consolidated_con.execute(f"ATTACH '{path}' AS {alias} (READ_ONLY)")
            # Insert papers
            consolidated_con.execute(f"""
                INSERT OR IGNORE INTO papers
                SELECT id, doi, title, abstract, publication_year, publication_date,
                       type, cited_by_count, counts_by_year, num_matched_concepts, processed_date
                FROM {alias}.papers
            """)
            # Insert concept mappings
            consolidated_con.execute(f"""
                INSERT OR IGNORE INTO concept_papers (concept, paper_id)
                SELECT UNNEST(matched_concepts), id
                FROM {alias}.papers
                WHERE matched_concepts IS NOT NULL
            """)
            consolidated_con.execute(f"DETACH {alias}")
        except Exception as e:
            print(f"Error processing {batch_name}: {e}")

    # 3️⃣ Remove downloaded batch files
    for path in downloaded_paths:
        if os.path.exists(path):
            os.remove(path)

    # 4️⃣ Check if consolidated DB is too large
    if get_db_size(current_consolidated_path) >= MAX_CONSOLIDATED_SIZE:
        print(f"Consolidated file {current_consolidated_path} reached size limit. Uploading...")
        consolidated_con.execute("CREATE INDEX IF NOT EXISTS idx_pub_year ON papers(publication_year)")
        consolidated_con.execute("CREATE INDEX IF NOT EXISTS idx_concept ON concept_papers(concept)")
        consolidated_con.close()

        # Upload to HF
        api.upload_file(
            path_or_fileobj=current_consolidated_path,
            path_in_repo=f"{HF_CONSOLIDATED_FOLDER}/consolidated_{consolidated_num:04d}.duckdb",
            repo_id=HF_REPO_ID,
            token=HF_TOKEN,
            repo_type="dataset"
        )
        print(f"✓ Uploaded consolidated_{consolidated_num:04d}.duckdb")
        os.remove(current_consolidated_path)

        # Start a new consolidated DB
        consolidated_num += 1
        current_consolidated_path = os.path.join(CONSOLIDATED_DIR, f"consolidated_{consolidated_num:04d}.duckdb")
        consolidated_con = init_consolidated_db(current_consolidated_path)

# -------------------------------
# Upload last consolidated file
# -------------------------------
if get_db_size(current_consolidated_path) > 0:
    consolidated_con.execute("CREATE INDEX IF NOT EXISTS idx_pub_year ON papers(publication_year)")
    consolidated_con.execute("CREATE INDEX IF NOT EXISTS idx_concept ON concept_papers(concept)")
    consolidated_con.close()
    api.upload_file(
        path_or_fileobj=current_consolidated_path,
        path_in_repo=f"{HF_CONSOLIDATED_FOLDER}/consolidated_{consolidated_num:04d}.duckdb",
        repo_id=HF_REPO_ID,
        token=HF_TOKEN,
        repo_type="dataset"
    )
    os.remove(current_consolidated_path)
    print(f"✓ Uploaded final consolidated_{consolidated_num:04d}.duckdb")

# -------------------------------
# Cleanup
# -------------------------------
shutil.rmtree(LOCAL_TEMP_DIR, ignore_errors=True)
if os.path.exists(CONSOLIDATED_DIR) and not os.listdir(CONSOLIDATED_DIR):
    os.rmdir(CONSOLIDATED_DIR)

print("\nConsolidation complete!")
import os
import duckdb
from huggingface_hub import HfApi, hf_hub_download, list_repo_files
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import shutil
import time

# -------------------------------
# Configuration
# -------------------------------
HF_REPO_ID = "lalit3c/OA_Domain_Concepts"
HF_TOKEN = os.getenv("HF_TOKEN")
assert HF_TOKEN, "HF_TOKEN environment variable not set"

# Max size of each consolidated DuckDB file (bytes)
MAX_CONSOLIDATED_SIZE = 5 * 1024**3  # 5 GB

# Directories
LOCAL_TEMP_DIR = os.environ.get("TMPDIR", "./temp_downloads")
CONSOLIDATED_DIR = os.environ.get("TMPDIR", "./consolidated_batches")
HF_CONSOLIDATED_FOLDER = "consolidated"

os.makedirs(LOCAL_TEMP_DIR, exist_ok=True)
os.makedirs(CONSOLIDATED_DIR, exist_ok=True)
os.makedirs("logs", exist_ok=True)  # For SLURM logs

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
    return con

def init_mapping_db(path: str):
    con = duckdb.connect(path)
    con.execute("""
        CREATE TABLE IF NOT EXISTS concept_papers (
            concept VARCHAR,
            paper_id VARCHAR,
            PRIMARY KEY (concept, paper_id)
        )
    """)
    return con

# -------------------------------
# Helpers
# -------------------------------
def get_db_size(path):
    """Return approximate file size in bytes"""
    return os.path.getsize(path)

def download_batch(batch_file):
    """Download a batch from HF"""
    local_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=batch_file,
        repo_type="dataset",
        token=HF_TOKEN,
        local_dir=LOCAL_TEMP_DIR,
        local_dir_use_symlinks=False
    )
    return local_path

# -------------------------------
# Get batch files from HF
# -------------------------------
all_files = list_repo_files(repo_id=HF_REPO_ID, repo_type="dataset", token=HF_TOKEN)
batch_files = sorted([f for f in all_files if f.startswith("batches/") and f.endswith(".duckdb")])
print(f"Found {len(batch_files)} batch files on HF")

# -------------------------------
# Initialize mapping database
# -------------------------------
MAPPING_DB_PATH = os.path.join(CONSOLIDATED_DIR, "concept_mapping.duckdb")
mapping_con = init_mapping_db(MAPPING_DB_PATH)

# -------------------------------
# Consolidation
# -------------------------------
consolidated_num = 1
current_consolidated_path = os.path.join(CONSOLIDATED_DIR, f"consolidated_{consolidated_num:04d}.duckdb")
consolidated_con = init_consolidated_db(current_consolidated_path)

print(f"Starting consolidation (max ~{MAX_CONSOLIDATED_SIZE/1e9:.1f} GB per file)")

BATCHES_PER_GROUP = 20  # adjust depending on size; ~5GB per group

# Process in chunks of batches to fit MAX_CONSOLIDATED_SIZE
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

    # 2️⃣ Bulk insert papers
    pattern = os.path.join(LOCAL_TEMP_DIR, "batches", "*.duckdb")
    print(f"Inserting {len(downloaded_paths)} batches into consolidated DB...")
    consolidated_con.execute(f"""
        INSERT OR IGNORE INTO papers
        SELECT id, doi, title, abstract, publication_year, publication_date,
               type, cited_by_count, counts_by_year, num_matched_concepts, processed_date
        FROM read_duckdb('{pattern}')
    """)

    # 3️⃣ Bulk insert concept mappings
    mapping_con.execute(f"""
        INSERT OR IGNORE INTO concept_papers (concept, paper_id)
        SELECT UNNEST(matched_concepts), id
        FROM read_duckdb('{pattern}')
        WHERE matched_concepts IS NOT NULL
    """)

    # 4️⃣ Clean temp files
    for path in downloaded_paths:
        if os.path.exists(path):
            os.remove(path)

    # 5️⃣ Check if consolidated file is too large
    if get_db_size(current_consolidated_path) >= MAX_CONSOLIDATED_SIZE:
        print(f"Consolidated file {current_consolidated_path} reached size limit. Uploading...")
        consolidated_con.execute("CREATE INDEX IF NOT EXISTS idx_pub_year ON papers(publication_year)")
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

        # Delete local consolidated file
        os.remove(current_consolidated_path)

        # Start a new consolidated DB
        consolidated_num += 1
        current_consolidated_path = os.path.join(CONSOLIDATED_DIR, f"consolidated_{consolidated_num:04d}.duckdb")
        consolidated_con = init_consolidated_db(current_consolidated_path)

# -------------------------------
# Final upload
# -------------------------------
if get_db_size(current_consolidated_path) > 0:
    consolidated_con.execute("CREATE INDEX IF NOT EXISTS idx_pub_year ON papers(publication_year)")
    consolidated_con.close()
    api.upload_file(
        path_or_fileobj=current_consolidated_path,
        path_in_repo=f"{HF_CONSOLIDATED_FOLDER}/consolidated_{consolidated_num:04d}.duckdb",
        repo_id=HF_REPO_ID,
        token=HF_TOKEN,
        repo_type="dataset"
    )
    print(f"✓ Uploaded final consolidated_{consolidated_num:04d}.duckdb")
    os.remove(current_consolidated_path)

# -------------------------------
# Finalize concept mapping DB
# -------------------------------
print("Finalizing concept mapping database...")
mapping_con.execute("CREATE INDEX IF NOT EXISTS idx_concept ON concept_papers(concept)")
mapping_con.close()

api.upload_file(
    path_or_fileobj=MAPPING_DB_PATH,
    path_in_repo=f"{HF_CONSOLIDATED_FOLDER}/concept_mapping.duckdb",
    repo_id=HF_REPO_ID,
    token=HF_TOKEN,
    repo_type="dataset"
)
print("✓ Uploaded concept_mapping.duckdb")
os.remove(MAPPING_DB_PATH)

# -------------------------------
# Cleanup
# -------------------------------
shutil.rmtree(LOCAL_TEMP_DIR, ignore_errors=True)
if os.path.exists(CONSOLIDATED_DIR) and not os.listdir(CONSOLIDATED_DIR):
    os.rmdir(CONSOLIDATED_DIR)

print(f"\nConsolidation complete!")
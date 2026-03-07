import os
import duckdb
from huggingface_hub import HfApi, hf_hub_download, list_repo_files
from tqdm import tqdm
import os

# Configuration
HF_REPO_ID = "lalit3c/OA_Domain_Concepts"
HF_TOKEN = os.getenv("HF_TOKEN")
assert HF_TOKEN, "HuggingFace token not set. Please set HF_TOKEN environment variable."
# Number of documents per consolidated file
DOCS_PER_CONSOLIDATED = 100_000_000  # 100M docs per consolidated file
LOCAL_TEMP_DIR = "temp_downloads"  # Temp folder for downloads
CONSOLIDATED_DIR = "consolidated_batches"  # Local folder for consolidated files
HF_CONSOLIDATED_FOLDER = "consolidated"  # Folder name on HF

# Create directories
os.makedirs(LOCAL_TEMP_DIR, exist_ok=True)
os.makedirs(CONSOLIDATED_DIR, exist_ok=True)

api = HfApi()

# Get list of batch files from HF
print("Fetching file list from HuggingFace...")
all_files = list_repo_files(repo_id=HF_REPO_ID, repo_type="dataset", token=HF_TOKEN)
batch_files = sorted([f for f in all_files if f.startswith("batches/") and f.endswith(".duckdb")])
print(f"Found {len(batch_files)} batch files on HuggingFace")

def init_consolidated_db(filepath: str) -> duckdb.DuckDBPyConnection:
    """Initialize a consolidated DuckDB database with schema."""
    con = duckdb.connect(filepath)
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
            matched_concepts VARCHAR[],
            num_matched_concepts INTEGER,
            processed_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    return con

def init_mapping_db(filepath: str) -> duckdb.DuckDBPyConnection:
    """Initialize the concept-to-paper mapping database."""
    con = duckdb.connect(filepath)
    con.execute("""
        CREATE TABLE IF NOT EXISTS concept_papers (
            concept VARCHAR,
            paper_id VARCHAR,
            PRIMARY KEY (concept, paper_id)
        )
    """)
    return con

def get_doc_count(con: duckdb.DuckDBPyConnection) -> int:
    """Get document count from a DuckDB connection."""
    return con.execute("SELECT COUNT(*) FROM papers").fetchone()[0]

def bulk_copy_papers(source_path: str, dest_con: duckdb.DuckDBPyConnection) -> int:
    """Bulk copy all papers using ATTACH and INSERT - much faster!"""
    dest_con.execute(f"ATTACH '{source_path}' AS source_db (READ_ONLY)")
    
    # Bulk insert with INSERT OR IGNORE for duplicates
    dest_con.execute("""
        INSERT OR IGNORE INTO papers 
        SELECT * FROM source_db.papers
    """)
    
    dest_con.execute("DETACH source_db")
    return get_doc_count(dest_con)

def bulk_copy_to_mapping(source_path: str, mapping_con: duckdb.DuckDBPyConnection) -> int:
    """Explode matched_concepts array and insert into mapping table."""
    mapping_con.execute(f"ATTACH '{source_path}' AS source_db (READ_ONLY)")
    
    # Unnest the array and insert concept-paper pairs
    mapping_con.execute("""
        INSERT OR IGNORE INTO concept_papers (concept, paper_id)
        SELECT UNNEST(matched_concepts) AS concept, id AS paper_id
        FROM source_db.papers
        WHERE matched_concepts IS NOT NULL
    """)
    
    mapping_con.execute("DETACH source_db")
    return mapping_con.execute("SELECT COUNT(*) FROM concept_papers").fetchone()[0]

# Initialize mapping database (single file for all concept-paper mappings)
MAPPING_DB_PATH = os.path.join(CONSOLIDATED_DIR, "concept_mapping.duckdb")
mapping_con = init_mapping_db(MAPPING_DB_PATH)
print(f"Initialized concept mapping database: {MAPPING_DB_PATH}")

# Process batches
consolidated_num = 1
current_consolidated_path = os.path.join(CONSOLIDATED_DIR, f"consolidated_{consolidated_num:04d}.duckdb")
consolidated_con = init_consolidated_db(current_consolidated_path)
current_doc_count = 0

print(f"\nStarting consolidation process...")
print(f"Target: {DOCS_PER_CONSOLIDATED:,} docs per consolidated file\n")

for i, batch_file in enumerate(tqdm(batch_files, desc="Processing batches")):
    batch_name = os.path.basename(batch_file)
    
    try:
        # Download batch file
        hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=batch_file,
            repo_type="dataset",
            token=HF_TOKEN,
            local_dir=LOCAL_TEMP_DIR,
            local_dir_use_symlinks=False
        )
        print(f"\downloaded {batch_name} to {LOCAL_TEMP_DIR}")
        # The file is downloaded to LOCAL_TEMP_DIR/batches/batch_xxx.duckdb
        actual_path = os.path.join(LOCAL_TEMP_DIR, batch_file)
        
        # Get batch doc count first
        temp_con = duckdb.connect(actual_path, read_only=True)
        batch_doc_count = get_doc_count(temp_con)
        temp_con.close()
        
        # Check if we need to start a new consolidated file
        if current_doc_count + batch_doc_count > DOCS_PER_CONSOLIDATED and current_doc_count > 0:
            # Add index before uploading
            consolidated_con.execute("CREATE INDEX IF NOT EXISTS idx_pub_year ON papers(publication_year)")
            consolidated_con.close()
            
            print(f"\n  Uploading consolidated_{consolidated_num:04d}.duckdb ({current_doc_count:,} docs)...")
            api.upload_file(
                path_or_fileobj=current_consolidated_path,
                path_in_repo=f"{HF_CONSOLIDATED_FOLDER}/consolidated_{consolidated_num:04d}.duckdb",
                repo_id=HF_REPO_ID,
                token=HF_TOKEN,
                repo_type="dataset"
            )
            print(f"  ✓ Uploaded consolidated_{consolidated_num:04d}.duckdb")
            
            # Delete local consolidated file
            os.remove(current_consolidated_path)
            
            # Start new consolidated file
            consolidated_num += 1
            current_consolidated_path = os.path.join(CONSOLIDATED_DIR, f"consolidated_{consolidated_num:04d}.duckdb")
            consolidated_con = init_consolidated_db(current_consolidated_path)
            current_doc_count = 0
        
        # Bulk copy papers to consolidated DB
        bulk_copy_papers(actual_path, consolidated_con)
        current_doc_count = get_doc_count(consolidated_con)
        
        # Also populate the concept mapping table
        bulk_copy_to_mapping(actual_path, mapping_con)
        
        # Delete downloaded batch file
        os.remove(actual_path)
        
    except Exception as e:
        print(f"\n  ✗ Error processing {batch_name}: {e}")
        continue

# Upload final consolidated file
if current_doc_count > 0:
    consolidated_con.execute("CREATE INDEX IF NOT EXISTS idx_pub_year ON papers(publication_year)")
    consolidated_con.close()
    print(f"\nUploading final consolidated_{consolidated_num:04d}.duckdb ({current_doc_count:,} docs)...")
    api.upload_file(
        path_or_fileobj=current_consolidated_path,
        path_in_repo=f"{HF_CONSOLIDATED_FOLDER}/consolidated_{consolidated_num:04d}.duckdb",
        repo_id=HF_REPO_ID,
        token=HF_TOKEN,
        repo_type="dataset"
    )
    print(f"✓ Uploaded consolidated_{consolidated_num:04d}.duckdb")
    os.remove(current_consolidated_path)

# Finalize and upload mapping database
print(f"\nFinalizing concept mapping database...")
total_mappings = mapping_con.execute("SELECT COUNT(*) FROM concept_papers").fetchone()[0]
unique_concepts = mapping_con.execute("SELECT COUNT(DISTINCT concept) FROM concept_papers").fetchone()[0]
print(f"  Total mappings: {total_mappings:,}")
print(f"  Unique concepts: {unique_concepts:,}")

# Add index for fast concept lookups
print("  Creating index on concept column...")
mapping_con.execute("CREATE INDEX IF NOT EXISTS idx_concept ON concept_papers(concept)")
mapping_con.close()

print(f"  Uploading concept_mapping.duckdb...")
api.upload_file(
    path_or_fileobj=MAPPING_DB_PATH,
    path_in_repo=f"{HF_CONSOLIDATED_FOLDER}/concept_mapping.duckdb",
    repo_id=HF_REPO_ID,
    token=HF_TOKEN,
    repo_type="dataset"
)
print(f"  ✓ Uploaded concept_mapping.duckdb")
os.remove(MAPPING_DB_PATH)

# Cleanup temp directory
import shutil
shutil.rmtree(LOCAL_TEMP_DIR, ignore_errors=True)
if os.path.exists(CONSOLIDATED_DIR) and not os.listdir(CONSOLIDATED_DIR):
    os.rmdir(CONSOLIDATED_DIR)

print(f"\n{'='*60}")
print(f"Consolidation complete!")
print(f"Created {consolidated_num} consolidated files in '{HF_CONSOLIDATED_FOLDER}/' folder")
print(f"Created concept_mapping.duckdb with {total_mappings:,} concept-paper mappings")
print(f"Original batch files in 'batches/' folder are preserved")
print(f"\nUsage example:")
print(f"  SELECT paper_id FROM concept_papers WHERE concept = 'machine learning'")
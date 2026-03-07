"""
OpenAlex Concept Filter Script
==============================
Processes all OpenAlex works, reconstructs abstracts, and filters papers
that contain any of the concepts from full_domain_concepts.txt in their
title or abstract. Results are saved to DuckDB with resumability support.

Features:
- Fast text matching using Aho-Corasick algorithm
- Resumable processing (tracks processed dates)
- Periodic saving to DuckDB
- Optional HuggingFace upload
- Multi-concept matching per paper
"""

import duckdb
import os
import json
import time
import gc
from datetime import datetime
from tqdm import tqdm
import boto3
from botocore import UNSIGNED
from botocore.config import Config
import ahocorasick
import pyarrow as pa
import pyarrow.parquet as pq
from typing import Optional
import gzip
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# =============================================================================
# CONFIGURATION
# =============================================================================

# Paths
CONCEPTS_FILE = "full_domain_concepts.txt"
DUCKDB_DIR = "duckdb_batches"  # Directory for temporary duckdb files
PROGRESS_FILE = "processing_progress.json"
FILES_CACHE_FILE = "all_files_cache.json"  # Cache for S3 file list

# Processing settings
MAX_DOCS_PER_BATCH = 50000  # Max documents per DuckDB batch file (reduced for memory)
NUM_WORKERS = 4  # Number of parallel workers (reduced for memory)
SAVE_INTERVAL = 5000  # Save to DB every N papers to avoid memory buildup
FILTER_START_YEAR = None  # Set to filter by year (e.g., 2016), None for all
FILTER_END_YEAR = None    # Set to filter by year (e.g., 2025), None for all

# HuggingFace settings (optional)
HF_UPLOAD_ENABLED = True
HF_REPO_ID = "lalit3c/OA_Domain_Concepts"  # Change this
HF_TOKEN = os.getenv("HF_TOKEN")
assert HF_TOKEN, "HuggingFace token not set. Please set HF_TOKEN environment variable."
# S3 settings
S3_BUCKET = "openalex"
S3_PREFIX = "data/works/"

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def load_concepts(filepath: str) -> list[str]:
    """Load concepts from file, one per line."""
    with open(filepath, 'r', encoding='utf-8') as f:
        concepts = [line.strip().lower() for line in f if line.strip()]
    print(f"Loaded {len(concepts)} concepts")
    return concepts


def build_aho_corasick(concepts: list[str]) -> ahocorasick.Automaton:
    """
    Build Aho-Corasick automaton for fast multi-pattern matching.
    This allows O(n + m + z) matching where n=text length, m=patterns length, z=matches.
    """
    automaton = ahocorasick.Automaton()
    for idx, concept in enumerate(concepts):
        # Add word boundary markers for whole-word matching
        automaton.add_word(concept, (idx, concept))
    automaton.make_automaton()
    print(f"Built Aho-Corasick automaton with {len(concepts)} patterns")
    return automaton


def reconstruct_abstract(inverted_index: dict) -> str:
    """
    Reconstruct abstract text from OpenAlex inverted index format.
    
    The inverted index maps words to their positions:
    {"word1": [0, 5], "word2": [1, 3], ...}
    """
    if not inverted_index:
        return ""
    
    try:
        # Handle if it's a string (JSON)
        if isinstance(inverted_index, str):
            inverted_index = json.loads(inverted_index)
        
        if not isinstance(inverted_index, dict):
            return ""
        
        # Build position -> word mapping
        word_positions = []
        for word, positions in inverted_index.items():
            if isinstance(positions, list):
                for pos in positions:
                    word_positions.append((pos, word))
        
        # Sort by position and join
        word_positions.sort(key=lambda x: x[0])
        abstract = ' '.join(word for _, word in word_positions)
        return abstract
    except (json.JSONDecodeError, TypeError, AttributeError):
        return ""


def find_concepts_in_text(text: str, automaton: ahocorasick.Automaton) -> list[str]:
    """
    Find all matching concepts in text using Aho-Corasick.
    Returns list of matched concept strings.
    """
    if not text:
        return []
    
    text_lower = text.lower()
    found_concepts = set()
    
    for end_idx, (pattern_idx, concept) in automaton.iter(text_lower):
        # Check for word boundaries to avoid partial matches
        start_idx = end_idx - len(concept) + 1
        
        # Check left boundary
        if start_idx > 0 and text_lower[start_idx - 1].isalnum():
            continue
        
        # Check right boundary
        if end_idx + 1 < len(text_lower) and text_lower[end_idx + 1].isalnum():
            continue
        
        found_concepts.add(concept)
    
    return list(found_concepts)


def load_progress() -> dict:
    """Load processing progress from file."""
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r') as f:
            data = json.load(f)
            # Convert list to set for memory efficiency (set stored as list in JSON)
            if "processed_files" in data and isinstance(data["processed_files"], list):
                data["_processed_files_set"] = set(data["processed_files"])
                # Keep count but don't duplicate storage
                data["processed_files_count"] = len(data["processed_files"])
            return data
    return {
        "processed_files": [],
        "_processed_files_set": set(),
        "processed_files_count": 0,
        "total_papers_found": 0,
        "batch_number": 0,
        "last_updated": None
    }


def save_progress(progress: dict, processed_files_set: set = None):
    """Save processing progress to file."""
    progress["last_updated"] = datetime.now().isoformat()
    
    # Convert set to list for JSON serialization
    save_data = {k: v for k, v in progress.items() if not k.startswith('_')}
    if processed_files_set is not None:
        save_data["processed_files"] = list(processed_files_set)
        save_data["processed_files_count"] = len(processed_files_set)
    
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(save_data, f, indent=2)
    
    # Force garbage collection after saving large data
    gc.collect()


def get_all_updated_dates() -> list[str]:
    """Get all available updated_date folders from OpenAlex S3."""
    print("Fetching available updated_date folders from S3...")
    
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=S3_BUCKET, Prefix=S3_PREFIX, Delimiter='/')
    
    updated_dates = []
    for page in pages:
        for common_prefix in page.get("CommonPrefixes", []):
            folder = common_prefix["Prefix"]
            if "updated_date=" in folder:
                date_str = folder.split('=')[1].strip('/')
                updated_dates.append(date_str)
    
    updated_dates = sorted(updated_dates)
    print(f"Found {len(updated_dates)} updated_date folders")
    return updated_dates


def get_files_for_date(updated_date: str) -> list[str]:
    """Get all .gz files for a specific updated_date."""
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    prefix = f"{S3_PREFIX}updated_date={updated_date}/"
    
    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix)
    
    files = []
    for page in pages:
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith('.gz'):
                files.append(key)
    
    return files


def get_all_files(use_cache: bool = True) -> list[str]:
    """Get all .gz files from all updated_date folders. Uses cache if available."""
    
    # Check for cached file list
    if use_cache and os.path.exists(FILES_CACHE_FILE):
        print(f"Loading file list from cache: {FILES_CACHE_FILE}")
        with open(FILES_CACHE_FILE, 'r') as f:
            cache = json.load(f)
            all_files = cache.get("files", [])
            cached_date = cache.get("cached_date", "unknown")
            print(f"Loaded {len(all_files)} files from cache (cached on {cached_date})")
            return all_files
    
    print("Gathering all files from all updated_date folders...")
    
    all_dates = get_all_updated_dates()
    all_files = []
    
    for updated_date in tqdm(all_dates, desc="Gathering files from dates"):
        files = get_files_for_date(updated_date)
        all_files.extend(files)
    
    # Save to cache
    print(f"Saving file list to cache: {FILES_CACHE_FILE}")
    with open(FILES_CACHE_FILE, 'w') as f:
        json.dump({
            "files": all_files,
            "cached_date": datetime.now().isoformat(),
            "total_files": len(all_files)
        }, f)
    
    print(f"Total files to process: {len(all_files)}")
    return all_files


def process_gz_file(s3_key: str, automaton: ahocorasick.Automaton, s3_client=None) -> list[dict]:
    """
    Process a single .gz file and return matching papers.
    Uses streaming decompression to minimize memory usage.
    Creates its own S3 client if not provided (for thread safety).
    """
    matching_papers = []
    
    # Create thread-local S3 client if not provided
    if s3_client is None:
        s3_client = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    
    try:
        # Download to temp file and stream decompress to minimize memory
        print(f"downloading {s3_key}...")
        response = s3_client.get_object(Bucket=S3_BUCKET, Key=s3_key)
        
        # Use streaming decompression with a temporary file
        with tempfile.NamedTemporaryFile(delete=True, suffix='.gz') as tmp_file:
            # Download in chunks to avoid loading entire file in memory
            for chunk in response['Body'].iter_chunks(chunk_size=1024 * 1024):  # 1MB chunks
                tmp_file.write(chunk)
            tmp_file.flush()
            tmp_file.seek(0)
            
            # Stream decompress and process line by line
            with gzip.open(tmp_file.name, 'rt', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                    
                    try:
                        work = json.loads(line)
                        
                        # Apply year filter if set
                        if FILTER_START_YEAR or FILTER_END_YEAR:
                            pub_year = work.get('publication_year')
                            if pub_year:
                                if FILTER_START_YEAR and pub_year < FILTER_START_YEAR:
                                    continue
                                if FILTER_END_YEAR and pub_year > FILTER_END_YEAR:
                                    continue
                        
                        # Get title and reconstruct abstract
                        title = work.get('title', '') or ''
                        abstract_idx = work.get('abstract_inverted_index')
                        abstract = reconstruct_abstract(abstract_idx)
                        
                        # Combine for matching
                        combined_text = f"{title} {abstract}"
                        
                        # Find matching concepts
                        matched_concepts = find_concepts_in_text(combined_text, automaton)
                        
                        if matched_concepts:
                            # Create record with matched concepts
                            paper = {
                                'id': work.get('id', ''),
                                'doi': work.get('doi', ''),
                                'title': title,
                                'abstract': abstract,
                                'publication_year': work.get('publication_year'),
                                'publication_date': work.get('publication_date', ''),
                                'type': work.get('type', ''),
                                'cited_by_count': work.get('cited_by_count', 0),
                                'counts_by_year': work.get('counts_by_year', []),
                                'matched_concepts': matched_concepts,  # List of matched concepts
                                'num_matched_concepts': len(matched_concepts),
                            }
                            matching_papers.append(paper)
                        
                        # Clear work object to free memory
                        del work
                        
                    except json.JSONDecodeError:
                        continue
        
        # Explicitly run garbage collection after processing large file
        gc.collect()
        print(f"Completed processing {s3_key}. Found {len(matching_papers)} matching papers.")
    except Exception as e:
        print(f"Error processing {s3_key}: {e}")
    
    return matching_papers


def init_duckdb(filepath: str) -> duckdb.DuckDBPyConnection:
    """Initialize DuckDB database with schema."""
    con = duckdb.connect(filepath)
    
    # Create table if not exists
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
    
    # Create index on publication_year for faster queries
    con.execute("""
        CREATE INDEX IF NOT EXISTS idx_pub_year ON papers(publication_year)
    """)
    
    return con


def insert_papers_to_duckdb(con: duckdb.DuckDBPyConnection, papers: list[dict]):
    """Insert papers into DuckDB, handling duplicates."""
    if not papers:
        return 0
    
    inserted = 0
    for paper in papers:
        try:
            con.execute("""
                INSERT OR REPLACE INTO papers 
                (id, doi, title, abstract, publication_year, publication_date, 
                 type, cited_by_count, counts_by_year, matched_concepts, num_matched_concepts)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                paper['id'],
                paper['doi'],
                paper['title'],
                paper['abstract'],
                paper['publication_year'],
                paper['publication_date'],
                paper['type'],
                paper['cited_by_count'],
                json.dumps(paper['counts_by_year']),
                paper['matched_concepts'],
                paper['num_matched_concepts']
            ])
            inserted += 1
        except Exception as e:
            # Skip duplicates or errors
            continue
    
    return inserted


def upload_to_huggingface(duckdb_file: str, repo_id: str, token: str, delete_after: bool = True) -> bool:
    """Upload DuckDB file to HuggingFace Hub and optionally delete local file."""
    try:
        from huggingface_hub import HfApi
        
        api = HfApi()
        
        # Upload duckdb file to a 'batches' folder in the repo
        api.upload_file(
            path_or_fileobj=duckdb_file,
            path_in_repo=f"batches/{os.path.basename(duckdb_file)}",
            repo_id=repo_id,
            token=token,
            repo_type="dataset"
        )
        
        print(f"Uploaded {os.path.basename(duckdb_file)} to HuggingFace: {repo_id}")
        
        # Delete local file after successful upload
        if delete_after:
            os.remove(duckdb_file)
            print(f"Deleted local file: {duckdb_file}")
        
        return True
        
    except ImportError:
        print("huggingface_hub not installed. Run: pip install huggingface_hub")
        return False
    except Exception as e:
        print(f"HuggingFace upload failed: {e}")
        return False


# =============================================================================
# MAIN PROCESSING LOOP
# =============================================================================

# Thread-local storage for S3 clients
thread_local = threading.local()

def get_thread_s3_client():
    """Get or create a thread-local S3 client."""
    if not hasattr(thread_local, 's3_client'):
        thread_local.s3_client = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    return thread_local.s3_client


def process_file_wrapper(args: tuple) -> tuple[str, list[dict]]:
    """
    Wrapper function for parallel processing.
    Returns (s3_key, matching_papers) tuple.
    """
    s3_key, automaton = args
    s3_client = get_thread_s3_client()
    try:
        papers = process_gz_file(s3_key, automaton, s3_client)
        return (s3_key, papers)
    except Exception as e:
        print(f"Error processing {s3_key}: {e}")
        return (s3_key, [])


def create_batch_db(batch_number: int) -> tuple[str, duckdb.DuckDBPyConnection]:
    """Create a new batch DuckDB file."""
    os.makedirs(DUCKDB_DIR, exist_ok=True)
    filepath = os.path.join(DUCKDB_DIR, f"batch_{batch_number:06d}.duckdb")
    con = init_duckdb(filepath)
    return filepath, con


def main():
    print("=" * 60)
    print("OpenAlex Concept Filter - Starting (Parallel Mode)")
    print(f"Using {NUM_WORKERS} parallel workers")
    print("=" * 60)
    
    # Load concepts and build automaton
    concepts = load_concepts(CONCEPTS_FILE)
    automaton = build_aho_corasick(concepts)
    
    # Load progress
    progress = load_progress()
    # Use the pre-built set if available, otherwise build from list
    processed_files = progress.get("_processed_files_set", set(progress.get("processed_files", [])))
    # Clear the list from progress to save memory (we have the set now)
    if "processed_files" in progress:
        del progress["processed_files"]
    gc.collect()
    
    total_papers = progress["total_papers_found"]
    batch_number = progress.get("batch_number", 0)
    
    print(f"Resuming from previous run: {len(processed_files)} files already processed")
    print(f"Total papers found so far: {total_papers}")
    print(f"Current batch number: {batch_number}")
    
    # Gather all files from all dates (uses cache)
    all_files = get_all_files(use_cache=True)
    remaining_files = [f for f in all_files if f not in processed_files]
    
    print(f"Files to process: {len(remaining_files)}")
    
    if not remaining_files:
        print("No files to process. Exiting.")
        return
    
    # Create initial batch DB
    batch_number += 1
    current_db_file, con = create_batch_db(batch_number)
    print(f"Created new batch file: {current_db_file}")
    
    # Track papers in current batch
    batch_papers = []
    current_batch_doc_count = 0
    files_since_save = 0
    completed_count = 0
    
    # Process files sequentially for minimal memory usage
    CHUNK_SIZE = 4  # Process one file at a time to minimize memory
    
    print(f"\nStarting parallel processing...")
    print(f"Memory-optimized settings: batch_size={MAX_DOCS_PER_BATCH}, workers={NUM_WORKERS}")
    print(f"Memory optimization: streaming decompression enabled, SAVE_INTERVAL={SAVE_INTERVAL}")
    
    # Track total files processed (including from previous runs)
    total_files_count = len(all_files)
    already_processed_count = len(processed_files)
    
    try:
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            pbar = tqdm(total=total_files_count, initial=already_processed_count,
                       desc="Processing files", unit="files", dynamic_ncols=True)
            
            # Process in chunks
            for chunk_start in range(0, len(remaining_files), CHUNK_SIZE):
                chunk = remaining_files[chunk_start:chunk_start + CHUNK_SIZE]
                
                # Submit chunk of tasks
                futures = {
                    executor.submit(process_file_wrapper, (s3_key, automaton)): s3_key 
                    for s3_key in chunk
                }
                
                # Process results as they complete
                for future in as_completed(futures):
                    s3_key = futures[future]
                    
                    try:
                        result_key, matching = future.result()
                        
                        # Add to batch
                        batch_papers.extend(matching)
                        current_batch_doc_count += len(matching)
                        total_papers += len(matching)
                        
                        # Mark file as processed
                        processed_files.add(result_key)
                        files_since_save += 1
                        completed_count += 1
                        
                        # Update progress bar with stats
                        pbar.update(1)
                        pbar.set_postfix({
                            'papers': total_papers,
                            'batch_docs': current_batch_doc_count,
                            'batch': batch_number
                        })
                        
                        # Incremental save to DB to avoid memory buildup
                        if len(batch_papers) >= SAVE_INTERVAL:
                            inserted = insert_papers_to_duckdb(con, batch_papers)
                            batch_papers = []  # Clear immediately after insert
                            gc.collect()  # Force garbage collection
                        
                        # Check if we need to finalize batch (reached max docs)
                        if current_batch_doc_count >= MAX_DOCS_PER_BATCH:
                            # Insert any remaining papers
                            if batch_papers:
                                inserted = insert_papers_to_duckdb(con, batch_papers)
                                batch_papers = []
                            
                            pbar.write(f"Finalizing batch with {current_batch_doc_count} papers in {os.path.basename(current_db_file)}")
                            
                            # Close current DB
                            con.close()
                            
                            # Upload to HuggingFace and delete local file
                            if HF_UPLOAD_ENABLED and HF_TOKEN:
                                upload_success = upload_to_huggingface(
                                    current_db_file, HF_REPO_ID, HF_TOKEN, delete_after=True
                                )
                                if not upload_success:
                                    pbar.write(f"Warning: Upload failed. Keeping local file: {current_db_file}")
                            
                            # Save progress
                            progress["total_papers_found"] = total_papers
                            progress["batch_number"] = batch_number
                            save_progress(progress, processed_files)
                            
                            # Force garbage collection
                            gc.collect()
                            
                            # Create new batch DB for next iteration
                            batch_number += 1
                            current_db_file, con = create_batch_db(batch_number)
                            pbar.write(f"Created new batch file: {current_db_file}")
                            
                            # Reset batch
                            current_batch_doc_count = 0
                            files_since_save = 0
                            
                            pbar.write(f"Progress saved. Total papers: {total_papers}")
                        
                        # Periodic progress save (every 50 files) even if batch not full
                        elif files_since_save >= 50:
                            # Insert any pending papers first
                            if batch_papers:
                                insert_papers_to_duckdb(con, batch_papers)
                                batch_papers = []
                            
                            progress["total_papers_found"] = total_papers
                            progress["batch_number"] = batch_number
                            save_progress(progress, processed_files)
                            files_since_save = 0
                            gc.collect()  # Force garbage collection at checkpoints
                            pbar.write(f"Checkpoint saved. Files: {len(processed_files)}, Papers: {total_papers}")
                        
                    except Exception as e:
                        pbar.write(f"Error processing {s3_key}: {e}")
                        continue
            
            pbar.close()
    
    except KeyboardInterrupt:
        print("\n\nInterrupted! Saving progress...")
        # Save remaining batch
        if batch_papers:
            insert_papers_to_duckdb(con, batch_papers)
            batch_papers = []
            con.close()
            # Try to upload the partial batch
            if HF_UPLOAD_ENABLED and HF_TOKEN:
                upload_to_huggingface(current_db_file, HF_REPO_ID, HF_TOKEN, delete_after=True)
        else:
            con.close()
            # Remove empty batch file
            if os.path.exists(current_db_file):
                os.remove(current_db_file)
        
        progress["total_papers_found"] = total_papers
        progress["batch_number"] = batch_number
        save_progress(progress, processed_files)
        print("Progress saved. You can resume by running the script again.")
        return
    
    # Final save for any remaining papers
    if batch_papers:
        inserted = insert_papers_to_duckdb(con, batch_papers)
        print(f"\nInserted {inserted} papers to final batch")
        batch_papers = []
        con.close()
        
        # Upload final batch
        if HF_UPLOAD_ENABLED and HF_TOKEN:
            upload_to_huggingface(current_db_file, HF_REPO_ID, HF_TOKEN, delete_after=True)
    else:
        con.close()
        # Remove empty batch file
        if os.path.exists(current_db_file):
            os.remove(current_db_file)
    
    progress["total_papers_found"] = total_papers
    progress["batch_number"] = batch_number
    save_progress(progress, processed_files)
    
    # Print summary
    print("\n" + "=" * 60)
    print("Processing Complete!")
    print("=" * 60)
    print(f"Total papers found: {total_papers}")
    print(f"Total batches uploaded: {batch_number}")
    print(f"Data available at: https://huggingface.co/datasets/{HF_REPO_ID}")


if __name__ == "__main__":
    main()

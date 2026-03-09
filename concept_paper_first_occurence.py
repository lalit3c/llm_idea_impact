import duckdb
import glob
from pathlib import Path
import os

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

def get_consolidated_files():
    """Get all consolidated duckdb files sorted by name"""
    return sorted(glob.glob('dataset_local/consolidated_*.duckdb'))

def create_multi_db_connection():
    """
    Create a DuckDB connection with all consolidated files attached.
    
    Key optimizations for 100GB+ total data:
    - Files are attached READ_ONLY (no write overhead)
    - DuckDB uses memory-mapped I/O, not loading entire files into RAM
    - Set explicit memory limit to control working memory
    - Temp directory for spilling large intermediate results to disk
    
    Returns the connection and list of database aliases.
    """
    # Use a persistent temp file for intermediate results instead of :memory:
    # This allows spilling to disk for large queries
    con = duckdb.connect()
    
    # Configure for large dataset processing
    con.execute("SET temp_directory='dataset_local/tmp'")  # Spill to disk
    con.execute("SET preserve_insertion_order=false")  # Faster for aggregations
    
    files = get_consolidated_files()
    db_aliases = []
    
    for i, filepath in enumerate(files):
        alias = f'db{i:04d}'
        # READ_ONLY ensures memory-mapped access without loading entire file
        con.execute(f"ATTACH '{filepath}' AS {alias} (READ_ONLY)")
        db_aliases.append(alias)
    
    return con, db_aliases

# Test: Show available files
files = get_consolidated_files()
print(f"Found {len(files)} consolidated files:")
for f in files:
    print(f"  - {f}")
    
def build_union_query(db_aliases, base_query_template):
    """
    Build a UNION ALL query across all databases.
    base_query_template should have {db} placeholder for database alias.
    """
    queries = [base_query_template.format(db=alias) for alias in db_aliases]
    return " UNION ALL ".join(queries)

# ============================================================================
# QUERY 1: Concept pairs and their first co-occurrence
# ============================================================================
# Strategy: 
# - For each paper, generate all concept pairs (c1, c2) where c1 < c2
# - Join with papers to get publication_date
# - Use ARG_MIN to find the paper_id with the earliest date

def query_concept_pairs_first_cooccurrence(limit=None):
    """
    Find all concept pairs and the SPECIFIC PAPER that first connected them.
    
    Optimizations:
    - c1 < c2 ensures each pair counted once (not A-B and B-A)
    - ARG_MIN gets the paper_id corresponding to the minimum date
    - Pre-aggregates within each file before combining
    
    Returns: DataFrame with (concept1, concept2, first_paper_id, first_publication_date, first_publication_year)
    """
    con, db_aliases = create_multi_db_connection()
    
    # Template query for each database
    # ARG_MIN(paper_id, date) returns the paper_id that has the minimum date
    per_db_template = """
    SELECT 
        cp1.concept AS concept1,
        cp2.concept AS concept2,
        ARG_MIN(p.id, TRY_CAST(p.publication_date AS DATE)) AS first_paper_id,
        MIN(TRY_CAST(p.publication_date AS DATE)) AS first_pub_date,
        ARG_MIN(p.publication_year, TRY_CAST(p.publication_date AS DATE)) AS first_pub_year
    FROM {db}.concept_papers cp1
    INNER JOIN {db}.concept_papers cp2 
        ON cp1.paper_id = cp2.paper_id 
        AND cp1.concept < cp2.concept
    INNER JOIN {db}.papers p ON cp1.paper_id = p.id
    WHERE TRY_CAST(p.publication_date AS DATE) IS NOT NULL
    GROUP BY cp1.concept, cp2.concept
    """
    
    union_query = build_union_query(db_aliases, per_db_template)
    
    # Final aggregation across all DBs - again use ARG_MIN to get the right paper
    limit_clause = f"LIMIT {limit}" if limit else ""
    
    final_query = f"""
    SELECT 
        concept1,
        concept2,
        ARG_MIN(first_paper_id, first_pub_date) AS first_paper_id,
        MIN(first_pub_date) AS first_publication_date,
        ARG_MIN(first_pub_year, first_pub_date) AS first_publication_year
    FROM ({union_query}) sub
    GROUP BY concept1, concept2
    ORDER BY first_publication_date
    {limit_clause}
    """
    
    return con.execute(final_query).fetchdf()

# Test with small limit
print("Query 1: Concept pairs with first co-occurrence (specific paper)")
print("=" * 60)
df_pairs = query_concept_pairs_first_cooccurrence()
df_pairs.to_csv(f'{RESULTS_DIR}/concept_pairs_first_occurence.csv')
import duckdb
import glob
import pandas as pd
from tqdm import tqdm
from pathlib import Path

SAVE_DIR = Path("results")
SAVE_DIR.mkdir(exist_ok=True)


def get_consolidated_files():
    """Get all consolidated duckdb files sorted by name."""
    return sorted(glob.glob('dataset_local/consolidated_*.duckdb'))


def create_multi_db_connection(memory_limit='4GB'):
    """
    Create a DuckDB in-memory connection with all consolidated files attached.
    Files are attached READ_ONLY for data safety and to allow concurrent readers.
    Temp directory is configured so large intermediate results can spill to disk.
    """
    con = duckdb.connect()

    con.execute(f"SET memory_limit='{memory_limit}'")
    con.execute("SET temp_directory='dataset_local/tmp'")
    con.execute("SET preserve_insertion_order=false")  # Faster for aggregations

    files = get_consolidated_files()
    if not files:
        raise FileNotFoundError("No consolidated_*.duckdb files found in dataset_local/")

    db_aliases = []
    for i, filepath in enumerate(files):
        alias = f'db{i:04d}'
        con.execute(f"ATTACH '{filepath}' AS {alias} (READ_ONLY)")
        db_aliases.append(alias)

    return con, db_aliases


def query_concept_pairs_first_cooccurrence(memory_limit='4GB'):
    """
    Find all concept pairs and the earliest paper that first connected them.

    Strategy:
      1. For each shard, self-join concept_papers to generate all (c1, c2) pairs
         where c1 < c2 (avoids duplicates and self-pairs), then aggregate to find
         the earliest co-occurrence within that shard.
      2. Collect per-shard results into pandas and do a final aggregation to find
         the true global minimum across all shards.

    Returns a DataFrame with columns:
      concept1, concept2, first_paper_id, first_publication_date, first_publication_year
    """
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

    con, db_aliases = create_multi_db_connection(memory_limit=memory_limit)
    print(f"Attached {len(db_aliases)} database shards.\n")

    shard_results = []
    for alias in tqdm(db_aliases, desc="Processing shards"):
        query = per_db_template.format(db=alias)
        df = con.execute(query).fetchdf()
        shard_results.append(df)
        tqdm.write(f"  {alias}: {len(df):,} concept pairs found")

    con.close()

    print(f"\nAggregating across shards...")
    combined = pd.concat(shard_results, ignore_index=True)
    print(f"Combined rows before final aggregation: {len(combined):,}")

    # Sort first so that .first() picks the earliest row after groupby
    combined = combined.sort_values('first_pub_date')

    final = (
        combined
        .groupby(['concept1', 'concept2'], as_index=False)
        .agg(
            first_paper_id=('first_paper_id', 'first'),
            first_publication_date=('first_pub_date', 'min'),
            first_publication_year=('first_pub_year', 'first'),
        )
        .sort_values('first_publication_date')
        .reset_index(drop=True)
    )

    print(f"Final unique concept pairs: {len(final):,}")
    return final


if __name__ == "__main__":
    df = query_concept_pairs_first_cooccurrence()

    output_path = f"{SAVE_DIR}/concept_pairs_first_cooccurrence.parquet"
    df.to_parquet(output_path, index=False)
    print(f"\nSaved to {output_path}")
    print(df.head(10).to_string())
"""
Find Representative Papers for Each Concept

This script implements multiple strategies for selecting k representative papers
per concept from a distributed DuckDB dataset (~70M papers, 36K concepts, 20 files).

Goal: Extract papers whose title + abstract can fit into an LLM (gpt-oss-20b).

Token Budget Analysis for k Selection:
======================================
- GPT-OSS-20B typical context: ~8K-16K tokens
- Average title: ~15 words ≈ 20 tokens
- Average abstract: ~150-250 words ≈ 200-300 tokens
- Per paper budget: ~250-350 tokens (title + abstract + formatting)

Recommended k values:
- k=20: ~6,000 tokens → Safe for 8K context (leaves room for prompt + response)
- k=30: ~9,000 tokens → Good for 16K context
- k=50: ~15,000 tokens → Maximum for 16K context
- k=100: ~30,000 tokens → Requires 32K+ context or chunking

RECOMMENDATION: k=20-30 for single-call LLM processing
"""

import duckdb
import glob
import os
import argparse
from pathlib import Path
from typing import List, Optional
import pandas as pd


def get_consolidated_files(data_dir: str = 'dataset_local') -> List[str]:
    """Get all consolidated DuckDB files sorted by name."""
    return sorted(glob.glob(f'{data_dir}/consolidated_*.duckdb'))


def create_multi_db_connection(data_dir: str = 'dataset_local', memory_limit: str = '4GB'):
    """
    Create a DuckDB connection with all consolidated files attached.
    Uses memory-mapped I/O for efficient access to 100GB+ data.
    """
    os.makedirs(f'{data_dir}/tmp', exist_ok=True)
    
    con = duckdb.connect()
#    con.execute(f"SET memory_limit='{memory_limit}'")
    con.execute(f"SET temp_directory='{data_dir}/tmp'")
    con.execute("SET preserve_insertion_order=false")
    
    files = get_consolidated_files(data_dir)
    db_aliases = []
    
    for i, filepath in enumerate(files):
        alias = f'db{i:04d}'
        con.execute(f"ATTACH '{filepath}' AS {alias} (READ_ONLY)")
        db_aliases.append(alias)
    
    print(f"Attached {len(db_aliases)} database files")
    return con, db_aliases


def build_union_query(db_aliases: List[str], template: str) -> str:
    """Build UNION ALL query across all databases."""
    queries = [template.format(db=alias) for alias in db_aliases]
    return " UNION ALL ".join(queries)


# =============================================================================
# STRATEGY 1: Top-K by Citation Count (Most Influential)
# =============================================================================
def strategy_citation_count(
    con, 
    db_aliases: List[str], 
    k: int = 30,
    concepts: Optional[List[str]] = None,
    output_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Select top-k papers per concept by citation count.
    
    Pros: 
    - Simple, fast
    - Selects highly influential/seminal papers
    
    Cons:
    - Biased toward older papers (more time to accumulate citations)
    - May miss recent important work
    """
    print(f"\n{'='*60}")
    print("Strategy 1: Top-K by Citation Count")
    print(f"{'='*60}")
    
    concept_filter = ""
    if concepts:
        concepts_str = ", ".join([f"'{c}'" for c in concepts])
        concept_filter = f"AND cp.concept IN ({concepts_str})"
    
    per_db_template = f"""
    SELECT 
        cp.concept,
        p.id AS paper_id,
        p.title,
        p.abstract,
        p.publication_year,
        p.cited_by_count,
        ROW_NUMBER() OVER (
            PARTITION BY cp.concept 
            ORDER BY p.cited_by_count DESC NULLS LAST
        ) AS rank_in_db
    FROM {{db}}.concept_papers cp
    INNER JOIN {{db}}.papers p ON cp.paper_id = p.id
    WHERE p.title IS NOT NULL AND TRIM(p.title) != ''
      AND p.abstract IS NOT NULL AND TRIM(p.abstract) != ''
      {concept_filter}
    """
    
    union_query = build_union_query(db_aliases, per_db_template)
    
    # Global ranking across all DBs
    final_query = f"""
    SELECT 
        concept,
        paper_id,
        title,
        abstract,
        publication_year,
        cited_by_count
    FROM (
        SELECT 
            *,
            ROW_NUMBER() OVER (
                PARTITION BY concept 
                ORDER BY cited_by_count DESC NULLS LAST
            ) AS global_rank
        FROM ({union_query})
    ) ranked
    WHERE global_rank <= {k}
    ORDER BY concept, global_rank
    """
    
    print("Executing query...")
    df = con.execute(final_query).fetchdf()
    print(f"Retrieved {len(df)} rows for {df['concept'].nunique()} concepts")
    
    if output_path:
        df.to_parquet(output_path, compression='zstd')
        print(f"Saved to {output_path}")
    
    return df


# =============================================================================
# STRATEGY 2: Top-K by Recency-Weighted Citations
# =============================================================================
def strategy_recency_weighted(
    con,
    db_aliases: List[str],
    k: int = 30,
    current_year: int = 2026,
    decay_factor: float = 0.1,
    concepts: Optional[List[str]] = None,
    output_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Select top-k papers per concept with recency-weighted scoring.
    
    Score = cited_by_count * exp(-decay_factor * age)
    
    This balances citation impact with recency, giving more weight
    to highly-cited recent papers.
    
    Pros:
    - Balances influence and recency
    - Better for capturing current state of field
    
    Cons:
    - Decay factor needs tuning
    - May underweight foundational papers
    """
    print(f"\n{'='*60}")
    print("Strategy 2: Recency-Weighted Citations")
    print(f"{'='*60}")
    
    concept_filter = ""
    if concepts:
        concepts_str = ", ".join([f"'{c}'" for c in concepts])
        concept_filter = f"AND cp.concept IN ({concepts_str})"
    
    per_db_template = f"""
    SELECT 
        cp.concept,
        p.id AS paper_id,
        p.title,
        p.abstract,
        p.publication_year,
        p.cited_by_count,
        -- Recency-weighted score
        COALESCE(p.cited_by_count, 0) * EXP(-{decay_factor} * ({current_year} - COALESCE(p.publication_year, {current_year}))) AS weighted_score
    FROM {{db}}.concept_papers cp
    INNER JOIN {{db}}.papers p ON cp.paper_id = p.id
    WHERE p.title IS NOT NULL AND TRIM(p.title) != ''
      AND p.abstract IS NOT NULL AND TRIM(p.abstract) != ''
      {concept_filter}
    """
    
    union_query = build_union_query(db_aliases, per_db_template)
    
    final_query = f"""
    SELECT 
        concept,
        paper_id,
        title,
        abstract,
        publication_year,
        cited_by_count,
        weighted_score
    FROM (
        SELECT 
            *,
            ROW_NUMBER() OVER (
                PARTITION BY concept 
                ORDER BY weighted_score DESC NULLS LAST
            ) AS global_rank
        FROM ({union_query})
    ) ranked
    WHERE global_rank <= {k}
    ORDER BY concept, global_rank
    """
    
    print("Executing query...")
    df = con.execute(final_query).fetchdf()
    print(f"Retrieved {len(df)} rows for {df['concept'].nunique()} concepts")
    
    if output_path:
        df.to_parquet(output_path, compression='zstd')
        print(f"Saved to {output_path}")
    
    return df


# =============================================================================
# STRATEGY 3: Temporal Diversity (Sample Across Time Periods)
# =============================================================================
def strategy_temporal_diversity(
    con,
    db_aliases: List[str],
    k: int = 30,
    time_buckets: int = 5,
    concepts: Optional[List[str]] = None,
    output_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Select papers ensuring temporal diversity within each concept.
    
    Divides papers into time buckets and selects top-cited from each bucket.
    k papers distributed across `time_buckets` periods.
    
    Pros:
    - Captures evolution of concept over time
    - Includes both foundational and recent work
    
    Cons:
    - May include less influential papers from sparse periods
    - More complex query
    """
    print(f"\n{'='*60}")
    print("Strategy 3: Temporal Diversity")
    print(f"{'='*60}")
    
    papers_per_bucket = max(1, k // time_buckets)
    
    concept_filter = ""
    if concepts:
        concepts_str = ", ".join([f"'{c}'" for c in concepts])
        concept_filter = f"AND cp.concept IN ({concepts_str})"
    
    per_db_template = f"""
    SELECT 
        cp.concept,
        p.id AS paper_id,
        p.title,
        p.abstract,
        p.publication_year,
        p.cited_by_count,
        NTILE({time_buckets}) OVER (
            PARTITION BY cp.concept 
            ORDER BY p.publication_year
        ) AS time_bucket
    FROM {{db}}.concept_papers cp
    INNER JOIN {{db}}.papers p ON cp.paper_id = p.id
    WHERE p.publication_year IS NOT NULL
      AND p.title IS NOT NULL AND TRIM(p.title) != ''
      AND p.abstract IS NOT NULL AND TRIM(p.abstract) != ''
      {concept_filter}
    """
    
    union_query = build_union_query(db_aliases, per_db_template)
    
    # Rank within each (concept, time_bucket) pair, then select top from each
    final_query = f"""
    SELECT 
        concept,
        paper_id,
        title,
        abstract,
        publication_year,
        cited_by_count,
        time_bucket
    FROM (
        SELECT 
            *,
            ROW_NUMBER() OVER (
                PARTITION BY concept, time_bucket 
                ORDER BY cited_by_count DESC NULLS LAST
            ) AS rank_in_bucket
        FROM ({union_query})
    ) ranked
    WHERE rank_in_bucket <= {papers_per_bucket}
    ORDER BY concept, time_bucket, rank_in_bucket
    """
    
    print("Executing query...")
    df = con.execute(final_query).fetchdf()
    print(f"Retrieved {len(df)} rows for {df['concept'].nunique()} concepts")
    
    if output_path:
        df.to_parquet(output_path, compression='zstd')
        print(f"Saved to {output_path}")
    
    return df


# =============================================================================
# STRATEGY 4: Coverage Diversity (Different Subfields/Related Concepts)
# =============================================================================
def strategy_coverage_diversity(
    con,
    db_aliases: List[str],
    k: int = 30,
    concepts: Optional[List[str]] = None,
    output_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Select papers that cover diverse aspects of the concept.
    
    Uses the number of co-occurring concepts (num_matched_concepts) as a 
    proxy for papers that are broader/narrower in scope.
    
    Selects papers with varying breadth: some narrow specialists, 
    some broad interdisciplinary.
    
    Pros:
    - Captures different perspectives on concept
    - Includes both specialized and interdisciplinary work
    
    Cons:
    - num_matched_concepts may not perfectly correlate with breadth
    """
    print(f"\n{'='*60}")
    print("Strategy 4: Coverage Diversity (Breadth Variation)")
    print(f"{'='*60}")
    
    # Split k into thirds: narrow, medium, broad
    narrow_k = k // 3
    broad_k = k // 3
    medium_k = k - narrow_k - broad_k
    
    concept_filter = ""
    if concepts:
        concepts_str = ", ".join([f"'{c}'" for c in concepts])
        concept_filter = f"AND cp.concept IN ({concepts_str})"
    
    per_db_template = f"""
    SELECT 
        cp.concept,
        p.id AS paper_id,
        p.title,
        p.abstract,
        p.publication_year,
        p.cited_by_count,
        COALESCE(p.num_matched_concepts, 1) AS num_matched_concepts,
        NTILE(3) OVER (
            PARTITION BY cp.concept 
            ORDER BY COALESCE(p.num_matched_concepts, 1)
        ) AS breadth_tier  -- 1=narrow, 2=medium, 3=broad
    FROM {{db}}.concept_papers cp
    INNER JOIN {{db}}.papers p ON cp.paper_id = p.id
    WHERE p.cited_by_count IS NOT NULL
      AND p.title IS NOT NULL AND TRIM(p.title) != ''
      AND p.abstract IS NOT NULL AND TRIM(p.abstract) != ''
      {concept_filter}
    """
    
    union_query = build_union_query(db_aliases, per_db_template)
    
    final_query = f"""
    WITH ranked AS (
        SELECT 
            *,
            ROW_NUMBER() OVER (
                PARTITION BY concept, breadth_tier 
                ORDER BY cited_by_count DESC NULLS LAST
            ) AS rank_in_tier
        FROM ({union_query})
    )
    SELECT 
        concept,
        paper_id,
        title,
        abstract,
        publication_year,
        cited_by_count,
        num_matched_concepts,
        breadth_tier
    FROM ranked
    WHERE 
        (breadth_tier = 1 AND rank_in_tier <= {narrow_k}) OR
        (breadth_tier = 2 AND rank_in_tier <= {medium_k}) OR
        (breadth_tier = 3 AND rank_in_tier <= {broad_k})
    ORDER BY concept, breadth_tier, rank_in_tier
    """
    
    print("Executing query...")
    df = con.execute(final_query).fetchdf()
    print(f"Retrieved {len(df)} rows for {df['concept'].nunique()} concepts")
    
    if output_path:
        df.to_parquet(output_path, compression='zstd')
        print(f"Saved to {output_path}")
    
    return df


# =============================================================================
# STRATEGY 5: Hybrid Ensemble (Combine Multiple Strategies)
# =============================================================================
def strategy_hybrid(
    con,
    db_aliases: List[str],
    k: int = 30,
    concepts: Optional[List[str]] = None,
    output_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Hybrid approach combining multiple ranking signals.
    
    Composite score based on:
    - Citation rank (influence)
    - Recency rank (timeliness)
    - Breadth rank (coverage diversity)
    
    Final score = weighted combination, deduplicated.
    
    Pros:
    - Most balanced approach
    - Captures multiple dimensions of "representativeness"
    
    Cons:
    - Most complex
    - Weights are somewhat arbitrary
    """
    print(f"\n{'='*60}")
    print("Strategy 5: Hybrid Ensemble")
    print(f"{'='*60}")
    
    current_year = 2026
    
    concept_filter = ""
    if concepts:
        concepts_str = ", ".join([f"'{c}'" for c in concepts])
        concept_filter = f"AND cp.concept IN ({concepts_str})"
    
    per_db_template = f"""
    SELECT 
        cp.concept,
        p.id AS paper_id,
        p.title,
        p.abstract,
        p.publication_year,
        p.cited_by_count,
        COALESCE(p.num_matched_concepts, 1) AS num_matched_concepts
    FROM {{db}}.concept_papers cp
    INNER JOIN {{db}}.papers p ON cp.paper_id = p.id
    WHERE p.title IS NOT NULL AND TRIM(p.title) != ''
      AND p.abstract IS NOT NULL AND TRIM(p.abstract) != ''
      {concept_filter}
    """
    
    union_query = build_union_query(db_aliases, per_db_template)
    
    # Compute multiple ranking signals and combine
    final_query = f"""
    WITH base_data AS ({union_query}),
    ranked AS (
        SELECT 
            *,
            -- Normalize each signal to 0-1 range within concept
            PERCENT_RANK() OVER (PARTITION BY concept ORDER BY cited_by_count DESC NULLS LAST) AS citation_prank,
            PERCENT_RANK() OVER (PARTITION BY concept ORDER BY publication_year DESC NULLS LAST) AS recency_prank,
            PERCENT_RANK() OVER (PARTITION BY concept ORDER BY num_matched_concepts DESC) AS breadth_prank
        FROM base_data
    ),
    scored AS (
        SELECT 
            *,
            -- Weighted combination (lower is better since PERCENT_RANK gives 0 to best)
            (0.5 * citation_prank + 0.3 * recency_prank + 0.2 * breadth_prank) AS composite_score
        FROM ranked
    ),
    final_ranked AS (
        SELECT 
            *,
            ROW_NUMBER() OVER (PARTITION BY concept ORDER BY composite_score) AS final_rank
        FROM scored
    )
    SELECT 
        concept,
        paper_id,
        title,
        abstract,
        publication_year,
        cited_by_count,
        num_matched_concepts,
        composite_score
    FROM final_ranked
    WHERE final_rank <= {k}
    ORDER BY concept, final_rank
    """
    
    print("Executing query...")
    df = con.execute(final_query).fetchdf()
    print(f"Retrieved {len(df)} rows for {df['concept'].nunique()} concepts")
    
    if output_path:
        df.to_parquet(output_path, compression='zstd')
        print(f"Saved to {output_path}")
    
    return df


# =============================================================================
# STRATEGY 6: Random Sampling
# =============================================================================
def strategy_random(
    con,
    db_aliases: List[str],
    k: int = 30,
    seed: int = 42,
    concepts: Optional[List[str]] = None,
    output_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Randomly sample k papers per concept.
    
    Pros:
    - Unbiased representation of the concept's literature
    - Fast and simple
    - Good baseline for comparison with other strategies
    - Captures the "typical" paper, not just exceptional ones
    
    Cons:
    - May miss important/seminal papers
    - Results vary with seed (set seed for reproducibility)
    - May include low-quality or poorly-cited papers
    """
    print(f"\n{'='*60}")
    print("Strategy 6: Random Sampling")
    print(f"{'='*60}")
    print(f"Random seed: {seed}")
    
    concept_filter = ""
    if concepts:
        concepts_str = ", ".join([f"'{c}'" for c in concepts])
        concept_filter = f"AND cp.concept IN ({concepts_str})"
    
    # Use DuckDB's SAMPLE or ROW_NUMBER with random ordering
    # setseed() ensures reproducibility across the query
    per_db_template = f"""
    SELECT 
        cp.concept,
        p.id AS paper_id,
        p.title,
        p.abstract,
        p.publication_year,
        p.cited_by_count,
        RANDOM() AS rand_val
    FROM {{db}}.concept_papers cp
    INNER JOIN {{db}}.papers p ON cp.paper_id = p.id
    WHERE p.title IS NOT NULL AND TRIM(p.title) != ''
      AND p.abstract IS NOT NULL AND TRIM(p.abstract) != ''
      {concept_filter}
    """
    
    union_query = build_union_query(db_aliases, per_db_template)
    
    # Set seed for reproducibility, then rank by random value
    final_query = f"""
    SELECT setseed({seed / 2147483647});  -- Normalize seed to 0-1 range
    
    SELECT 
        concept,
        paper_id,
        title,
        abstract,
        publication_year,
        cited_by_count
    FROM (
        SELECT 
            *,
            ROW_NUMBER() OVER (
                PARTITION BY concept 
                ORDER BY rand_val
            ) AS random_rank
        FROM ({union_query})
    ) ranked
    WHERE random_rank <= {k}
    ORDER BY concept, random_rank
    """
    
    print("Executing query...")
    # Execute setseed separately, then the main query
    con.execute(f"SELECT setseed({seed / 2147483647})")
    
    main_query = f"""
    SELECT 
        concept,
        paper_id,
        title,
        abstract,
        publication_year,
        cited_by_count
    FROM (
        SELECT 
            *,
            ROW_NUMBER() OVER (
                PARTITION BY concept 
                ORDER BY rand_val
            ) AS random_rank
        FROM ({union_query})
    ) ranked
    WHERE random_rank <= {k}
    ORDER BY concept, random_rank
    """
    
    df = con.execute(main_query).fetchdf()
    print(f"Retrieved {len(df)} rows for {df['concept'].nunique()} concepts")
    
    if output_path:
        df.to_parquet(output_path, compression='zstd')
        print(f"Saved to {output_path}")
    
    return df


# =============================================================================
# INCREMENTAL PROCESSING (for memory efficiency with 70M papers)
# =============================================================================
def process_incrementally(
    strategy_func,
    data_dir: str = 'dataset_local',
    k: int = 30,
    output_path: str = 'dataset_local/representative_papers.parquet',
    **strategy_kwargs
) -> str:
    """
    Process files one at a time for memory efficiency.
    
    For very large datasets, this approach:
    1. Processes each DuckDB file independently
    2. Merges results incrementally
    3. Keeps only top-k globally after each merge
    """
    print(f"\n{'='*60}")
    print("Incremental Processing Mode")
    print(f"{'='*60}")
    
    os.makedirs(f'{data_dir}/tmp', exist_ok=True)
    files = get_consolidated_files(data_dir)
    intermediate_path = f'{data_dir}/tmp/representative_intermediate.parquet'
    
    for i, filepath in enumerate(files):
        print(f"\nProcessing file {i+1}/{len(files)}: {filepath}")
        
        # Process single file
        con = duckdb.connect()
 #       con.execute("SET memory_limit='4GB'")
        con.execute(f"SET temp_directory='{data_dir}/tmp'")
        con.execute(f"ATTACH '{filepath}' AS db (READ_ONLY)")
        
        df_current = strategy_func(con, ['db'], k=k*2, **strategy_kwargs)  # Get more than k for merging
        con.close()
        
        if i == 0:
            df_current.to_parquet(intermediate_path, compression='zstd')
        else:
            # Merge with existing
            df_existing = pd.read_parquet(intermediate_path)
            df_combined = pd.concat([df_existing, df_current], ignore_index=True)
            
            # Re-rank and keep top-k per concept
            if 'cited_by_count' in df_combined.columns:
                df_combined['_rank'] = df_combined.groupby('concept')['cited_by_count'].rank(
                    method='first', ascending=False
                )
                df_combined = df_combined[df_combined['_rank'] <= k].drop(columns=['_rank'])
            
            df_combined.to_parquet(intermediate_path, compression='zstd')
        
        print(f"  Intermediate size: {os.path.getsize(intermediate_path) / 1e6:.1f} MB")
    
    # Final output
    os.rename(intermediate_path, output_path)
    print(f"\nFinal output: {output_path}")
    return output_path


# =============================================================================
# TOKEN COUNT ESTIMATION
# =============================================================================
def estimate_token_count(df: pd.DataFrame) -> dict:
    """
    Estimate token count for LLM input.
    
    Rule of thumb: 1 token ≈ 4 characters (English text)
    More accurate: ~0.75 words per token
    """
    df = df.copy()
    df['title'] = df['title'].fillna('')
    df['abstract'] = df['abstract'].fillna('')
    
    # Character-based estimate
    df['title_chars'] = df['title'].str.len()
    df['abstract_chars'] = df['abstract'].str.len()
    df['total_chars'] = df['title_chars'] + df['abstract_chars']
    
    # Token estimates (4 chars per token is conservative)
    df['estimated_tokens'] = df['total_chars'] / 4
    
    stats = {
        'total_papers': len(df),
        'total_concepts': df['concept'].nunique(),
        'avg_tokens_per_paper': df['estimated_tokens'].mean(),
        'median_tokens_per_paper': df['estimated_tokens'].median(),
        'max_tokens_per_paper': df['estimated_tokens'].max(),
        'total_tokens': df['estimated_tokens'].sum(),
        'papers_per_concept': len(df) / df['concept'].nunique() if df['concept'].nunique() > 0 else 0
    }
    
    # Per-concept estimates
    concept_tokens = df.groupby('concept')['estimated_tokens'].sum()
    stats['avg_tokens_per_concept'] = concept_tokens.mean()
    stats['max_tokens_per_concept'] = concept_tokens.max()
    
    return stats


def recommend_k(context_window: int = 8192, reserve_tokens: int = 2000) -> dict:
    """
    Recommend k based on context window and typical paper lengths.
    
    Args:
        context_window: Total context window size in tokens
        reserve_tokens: Tokens reserved for system prompt and response
    
    Returns:
        Recommended k values for different scenarios
    """
    available_tokens = context_window - reserve_tokens
    
    # Typical token counts per paper (title + abstract)
    scenarios = {
        'short_papers': 150,    # Very concise abstracts
        'average_papers': 280,  # Typical academic papers
        'long_papers': 400,     # Detailed abstracts
    }
    
    recommendations = {
        'context_window': context_window,
        'available_tokens': available_tokens,
    }
    
    for scenario, tokens_per_paper in scenarios.items():
        recommendations[f'k_{scenario}'] = available_tokens // tokens_per_paper
    
    return recommendations


# =============================================================================
# MAIN CLI
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Find representative papers for each concept',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Strategies:
  1. citation    - Top-K by citation count (most influential)
  2. recency     - Recency-weighted citations (balance influence & timeliness)
  3. temporal    - Temporal diversity (sample across time periods)
  4. coverage    - Coverage diversity (narrow to broad papers)
  5. hybrid      - Hybrid ensemble (combine all signals)
  6. random      - Random sampling (unbiased baseline)

Examples:
  # Basic usage with citation strategy
  python find_representative_papers.py --strategy citation --k 30

  # Hybrid strategy with specific output path
  python find_representative_papers.py --strategy hybrid --k 25 --output output.parquet

  # Process specific concepts only
  python find_representative_papers.py --strategy citation --k 30 --concepts "Machine learning" "Deep learning"

  # Get k recommendations for your LLM context window
  python find_representative_papers.py --recommend-k --context-window 16384
        """
    )
    
    parser.add_argument('--strategy', type=str, default='hybrid',
                        choices=['citation', 'recency', 'temporal', 'coverage', 'hybrid', 'random'],
                        help='Strategy for selecting representative papers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for random strategy (default: 42)')
    parser.add_argument('--k', type=int, default=100,
                        help='Number of papers per concept (default: 100')
    parser.add_argument('--data-dir', type=str, default='dataset_local/consolidated',
                        help='Directory containing consolidated DuckDB files')
    parser.add_argument('--output', type=str, default=None,
                        help='Output Parquet file path')
    parser.add_argument('--concepts', type=str, nargs='*', default=None,
                        help='Specific concepts to process (default: all)')
    parser.add_argument('--incremental', action='store_true',
                        help='Use incremental processing for memory efficiency')
    parser.add_argument('--recommend-k', action='store_true',
                        help='Show recommended k values and exit')
    parser.add_argument('--context-window', type=int, default=8192,
                        help='LLM context window size for k recommendation')
    parser.add_argument('--memory-limit', type=str, default='4GB',
                        help='DuckDB memory limit')
    
    args = parser.parse_args()
    
    # Show k recommendations
    if args.recommend_k:
        print("\n" + "="*60)
        print("Recommended k Values for Different Context Windows")
        print("="*60)
        
        for ctx in [8192, 16384, 32768, 65536, 131072]:
            recs = recommend_k(ctx)
            print(f"\nContext window: {ctx:,} tokens")
            print(f"  Available for papers: {recs['available_tokens']:,} tokens")
            print(f"  k (short abstracts):   {recs['k_short_papers']}")
            print(f"  k (average abstracts): {recs['k_average_papers']}")
            print(f"  k (long abstracts):    {recs['k_long_papers']}")
        
        print("\n" + "="*60)
        print("RECOMMENDATION for gpt-oss-20b (assuming 8K-16K context):")
        print("  • k=20-25 for safe single-call processing")
        print("  • k=30 if context window is 16K+")
        print("  • k=50-100 requires chunking or summarization")
        print("="*60)
        return
    
    # Set default output path
    if args.output is None:
        args.output = f'{args.data_dir}/representative_papers_{args.strategy}_k{args.k}.parquet'
    
    # Strategy mapping
    strategy_map = {
        'citation': strategy_citation_count,
        'recency': strategy_recency_weighted,
        'temporal': strategy_temporal_diversity,
        'coverage': strategy_coverage_diversity,
        'hybrid': strategy_hybrid,
        'random': strategy_random,
    }
    
    strategy_func = strategy_map[args.strategy]
    
    print(f"\nConfiguration:")
    print(f"  Strategy: {args.strategy}")
    print(f"  k: {args.k}")
    print(f"  Data directory: {args.data_dir}")
    print(f"  Output: {args.output}")
    print(f"  Concepts: {'All' if args.concepts is None else args.concepts}")
    if args.strategy == 'random':
        print(f"  Seed: {args.seed}")
    
    # Build extra kwargs for specific strategies
    extra_kwargs = {}
    if args.strategy == 'random':
        extra_kwargs['seed'] = args.seed
    
    if args.incremental:
        process_incrementally(
            strategy_func,
            data_dir=args.data_dir,
            k=args.k,
            output_path=args.output,
            concepts=args.concepts,
            **extra_kwargs
        )
    else:
        con, db_aliases = create_multi_db_connection(args.data_dir, args.memory_limit)
        df = strategy_func(
            con, db_aliases, 
            k=args.k, 
            concepts=args.concepts,
            output_path=args.output,
            **extra_kwargs
        )
        
        # Show token estimates
        stats = estimate_token_count(df)
        print(f"\nToken Estimates:")
        print(f"  Total papers: {stats['total_papers']:,}")
        print(f"  Total concepts: {stats['total_concepts']:,}")
        print(f"  Avg tokens/paper: {stats['avg_tokens_per_paper']:.0f}")
        print(f"  Avg tokens/concept: {stats['avg_tokens_per_concept']:.0f}")
        print(f"  Max tokens/concept: {stats['max_tokens_per_concept']:.0f}")
        
        # Check if fits in context window
        recs = recommend_k(args.context_window)
        if stats['max_tokens_per_concept'] > recs['available_tokens']:
            print(f"\n  WARNING: Some concepts exceed {recs['available_tokens']:,} available tokens!")
            print(f"    Consider reducing k or using chunking.")


if __name__ == '__main__':
    main()

"""
Generate Concept Summaries from Representative Papers
======================================================
This script reads representative papers for each concept and uses an LLM
to generate concise summaries based on the paper titles and abstracts.


"""

import json
import re
import sys
import gc
from datetime import datetime
from pathlib import Path

import pandas as pd

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATASET_DIR = PROJECT_ROOT / "dataset_local"
RESULTS_DIR = PROJECT_ROOT / "results"

INPUT_FILE = DATASET_DIR / "representative_papers_citation_k100.parquet"
OUTPUT_FILE = RESULTS_DIR / "concept_summaries_citation_k100.json"
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)


def load_representative_papers(filepath: Path) -> pd.DataFrame:
    """Load representative papers from parquet file."""
    print(f"[{datetime.now():%H:%M:%S}] Loading representative papers from {filepath}")
    df = pd.read_parquet(filepath)
    print(f"[{datetime.now():%H:%M:%S}] Loaded {len(df):,} rows with {df['concept'].nunique():,} unique concepts")
    return df


def prepare_concept_data(df: pd.DataFrame) -> dict:
    """
    Group papers by concept and prepare data for LLM prompts.
    
    Returns:
        dict mapping concept name to list of papers (title, abstract, year, citations)
    """
    concept_papers = {}
    
    for concept, group in df.groupby('concept'):
        papers = []
        for _, row in group.iterrows():
            # Only include papers with non-empty abstracts for better context
            paper_info = {
                'title': row['title'],
                'abstract': row['abstract'] if pd.notna(row['abstract']) and row['abstract'].strip() else '',
                'year': int(row['publication_year']) if pd.notna(row['publication_year']) else None,
                'citations': int(row['cited_by_count']) if pd.notna(row['cited_by_count']) else 0
            }
            papers.append(paper_info)
        
        # # Sort by citations (most cited first) to prioritize influential papers
        # papers.sort(key=lambda x: x['citations'], reverse=True)
        concept_papers[concept] = papers
    
    return concept_papers


def build_prompt_for_concept(concept: str, papers: list, n_papers: int=100) -> str:
    """
    Build a prompt for the LLM to summarize a concept based on its representative papers.
    """
    # Build paper context - include title and abstract
    paper_context = []
    for i, paper in enumerate(papers[:n_papers], 1):  # Limit to top n_papers to avoid context overflow
        paper_str = f"Paper {i}:\n  Title: {paper['title']}"
        if paper['abstract']:
            paper_str += f"\n  Abstract: {paper['abstract']}"
        if paper['year']:
            paper_str += f"\n  Year: {paper['year']}"
        paper_str += f"\n  Citations: {paper['citations']}"
        paper_context.append(paper_str)
    
    papers_text = "\n\n".join(paper_context)
    
    prompt = f"""Based on the following representative papers for the concept "{concept}", provide a concise summary of what this concept represents in scientific research.

{papers_text}

Respond ONLY with a JSON object with the following keys:
- concept: the concept name
- definition: a clear 1-2 sentence definition of this concept
- key_themes: list of 3-5 main research themes or topics within this concept
- typical_applications: list of 2-4 typical applications or use cases
- related_fields: list of 2-4 related scientific fields or disciplines
- summary: a concise 2-3 sentence overview suitable for researchers unfamiliar with this concept
- research_trends: a brief note on any notable research trends or recent developments related to this concept"""
    
    return prompt


def build_conversations(concept_papers: dict) -> tuple:
    """
    Build conversation batches for all concepts.
    
    Returns:
        tuple of (list of conversations, list of concept names)
    """
    system_msg = (
        "Reasoning: low\n"
        "You are a precise scientific assistant that summarizes research concepts. "
        "You MUST respond with ONLY a single valid JSON object — no preamble, "
        "no explanation, no markdown fences, no text before or after the JSON. "
        "Start your response with { and end with }. "
        "Do not repeat yourself."
    )
    
    conversations = []
    concepts = []
    
    for concept, papers in concept_papers.items():
        prompt = build_prompt_for_concept(concept, papers)
        conversation = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt}
        ]
        conversations.append(conversation)
        concepts.append(concept)
    
    return conversations, concepts


def extract_json(text: str) -> dict:
    """Extract JSON from LLM response, handling various formats."""
    # Remove common artifacts
    text = re.sub(r"<\|channel\|>analysis.*?<\|end\|>", "", text, flags=re.DOTALL)
    text = re.sub(r"<\|channel\|>commentary.*?<\|end\|>", "", text, flags=re.DOTALL)
    text = re.sub(r"<analysis>.*?</analysis>", "", text, flags=re.DOTALL)
    text = re.sub(r"```json|```", "", text).strip()
    
    # Try to find and parse JSON
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {"error": "json_parse_failed", "raw": text}


def run_inference(conversations: list, concepts: list) -> list:
    """Run LLM inference on all conversations."""
    from vllm import LLM, SamplingParams
    import torch
    
    n = len(conversations)
    print(f"[{datetime.now():%H:%M:%S}] Starting inference for {n} concepts")
    print(f"[{datetime.now():%H:%M:%S}] Loading model + compiling CUDA graphs (~90s one-time)...")
    
    t0 = datetime.now()
    
    llm = LLM(
        model="openai/gpt-oss-20b",
        dtype="bfloat16",
        gpu_memory_utilization=0.92,
        trust_remote_code=True,
        max_num_seqs=128,
        max_num_batched_tokens=16384,
        max_model_len=4096,
    )
    
    load_secs = (datetime.now() - t0).total_seconds()
    print(f"[{datetime.now():%H:%M:%S}] Model ready in {load_secs:.1f}s")
    
    sampling_params = SamplingParams(
        temperature=0.3,
        top_p=0.9,
        max_tokens=1024,
        stop=["<|end|>", "<|start|>"],
        repetition_penalty=1.05,
    )
    
    print(f"[{datetime.now():%H:%M:%S}] Submitting all {n} concepts as one batch...")
    t_gen = datetime.now()
    
    try:
        outputs = llm.chat(conversations, sampling_params=sampling_params, use_tqdm=True)
    finally:
        try:
            from vllm.distributed.parallel_state import destroy_model_parallel
            destroy_model_parallel()
        except Exception:
            pass
        del llm
        gc.collect()
        torch.cuda.empty_cache()
    
    gen_secs = (datetime.now() - t_gen).total_seconds()
    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    tps = total_tokens / gen_secs if gen_secs > 0 else 0
    
    print(f"\n{'='*55}")
    print(f"  Concepts processed : {n}")
    print(f"  Total tokens       : {total_tokens}")
    print(f"  Generation time    : {gen_secs:.1f}s")
    print(f"  Throughput         : {tps:.0f} tokens/sec")
    print(f"  Avg per concept    : {gen_secs/n:.2f}s  ({total_tokens//n} tokens avg)")
    print(f"  Startup time       : {load_secs:.1f}s")
    print(f"  Total wall time    : {load_secs + gen_secs:.1f}s")
    print(f"{'='*55}\n")
    
    # Process results
    results = []
    ok, fail = 0, 0
    
    for i, (output, concept) in enumerate(zip(outputs, concepts)):
        parsed = extract_json(output.outputs[0].text)
        if "error" in parsed:
            fail += 1
        else:
            ok += 1
        
        results.append({
            "id": i,
            "concept": concept,
            "finish_reason": output.outputs[0].finish_reason,
            "tokens_generated": len(output.outputs[0].token_ids),
            "response": parsed,
        })
    
    print(f"  JSON parse: {ok} ok / {fail} failed")
    
    return results, {
        "total_tokens_generated": total_tokens,
        "generation_seconds": round(gen_secs, 2),
        "tokens_per_second": round(tps, 1),
        "avg_seconds_per_concept": round(gen_secs / n, 3),
        "startup_seconds": round(load_secs, 1),
        "total_wall_seconds": round(load_secs + gen_secs, 1),
        "json_parse_ok": ok,
        "json_parse_failed": fail,
    }


def main():
    print(f"[{datetime.now():%H:%M:%S}] Starting concept summary generation")
    
    # Load data
    df = load_representative_papers(INPUT_FILE)
    
    # Prepare concept data
    print(f"[{datetime.now():%H:%M:%S}] Preparing concept data...")
    concept_papers = prepare_concept_data(df)
    print(f"[{datetime.now():%H:%M:%S}] Prepared data for {len(concept_papers)} concepts")
    
    # Build conversations
    print(f"[{datetime.now():%H:%M:%S}] Building LLM prompts...")
    conversations, concepts = build_conversations(concept_papers)
    
    # Run inference
    results, stats = run_inference(conversations, concepts)
    
    # Save results
    output_data = {
        "model": "openai/gpt-oss-20b",
        "run_timestamp": datetime.now().isoformat(),
        "total_concepts": len(concepts),
        **stats,
        "results": results,
    }
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"[{datetime.now():%H:%M:%S}] Saved results to {OUTPUT_FILE}")
    
    # # Also save a simplified CSV for easy viewing
    # csv_output = RESULTS_DIR / "concept_summaries.csv"
    # csv_data = []
    # for result in results:
    #     resp = result['response']
    #     if 'error' not in resp:
    #         csv_data.append({
    #             'concept': result['concept'],
    #             'definition': resp.get('definition', ''),
    #             'key_themes': '; '.join(resp.get('key_themes', [])),
    #             'typical_applications': '; '.join(resp.get('typical_applications', [])),
    #             'related_fields': '; '.join(resp.get('related_fields', [])),
    #             'summary': resp.get('summary', ''),
    #         })
    
    # if csv_data:
    #     pd.DataFrame(csv_data).to_csv(csv_output, index=False)
    #     print(f"[{datetime.now():%H:%M:%S}] Saved CSV summary to {csv_output}")


if __name__ == "__main__":
    main()
    sys.exit(0)

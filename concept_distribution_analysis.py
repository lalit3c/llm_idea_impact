import duckdb
import glob
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

PLOT_DIR = Path("viz/concept_distribution")
PLOT_DIR.mkdir(parents=True, exist_ok=True)
# find all consolidated DBs
db_files = glob.glob("dataset_local/consolidated/*.duckdb")

print(f"Found {len(db_files)} databases")

# create temp connection
con = duckdb.connect()

# attach all DBs
for i, db in enumerate(db_files):
    con.execute(f"ATTACH '{db}' AS db{i}")

# build union query across all
tables = [f"db{i}.concept_papers" for i in range(len(db_files))]
union_query = " UNION ALL ".join([f"SELECT * FROM {t}" for t in tables])

query = f"""
SELECT concept, COUNT(*) as paper_count
FROM ({union_query})
GROUP BY concept
"""

print("Computing distribution...")

df = con.execute(query).df()

print("Total concepts:", len(df))

# -------------------------
# BASIC STATS
# -------------------------

print(df["paper_count"].describe())

# -------------------------
# HISTOGRAM
# -------------------------

plt.figure()
plt.hist(df["paper_count"], bins=100)
plt.xlabel("Papers per concept")
plt.ylabel("Number of concepts")
plt.title("Concept Frequency Distribution")

plt.savefig(PLOT_DIR / "concept_histogram.png", dpi=300)
plt.close()

# -------------------------
# LOG HISTOGRAM
# -------------------------

plt.figure()
plt.hist(df["paper_count"], bins=100, log=True)
plt.xlabel("Papers per concept")
plt.ylabel("Number of concepts (log)")
plt.title("Concept Frequency Distribution (Log)")

plt.savefig(PLOT_DIR / "concept_histogram_log.png", dpi=300)
plt.close()

# -------------------------
# CCDF (important for heavy tail)
# -------------------------

sorted_counts = df["paper_count"].sort_values()

ccdf = 1.0 - (sorted_counts.rank(method="first") / len(sorted_counts))

plt.figure()
plt.loglog(sorted_counts, ccdf)
plt.xlabel("Papers per concept")
plt.ylabel("CCDF")
plt.title("Heavy Tail of Concept Usage")

plt.savefig(PLOT_DIR / "concept_ccdf.png", dpi=300)
plt.close()

# -------------------------
# TOP CONCEPTS
# -------------------------

top = df.sort_values("paper_count", ascending=False).head(30)

plt.figure(figsize=(10,6))
plt.barh(top["concept"], top["paper_count"])
plt.xlabel("Number of papers")
plt.title("Top 30 Concepts")

plt.gca().invert_yaxis()

plt.savefig(PLOT_DIR / "top_concepts.png", dpi=300)
plt.close()

print("Plots saved to:", PLOT_DIR)
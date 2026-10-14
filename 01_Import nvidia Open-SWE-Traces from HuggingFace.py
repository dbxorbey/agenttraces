# Databricks notebook source
# DBTITLE 1,Import nvidia Open-SWE-Traces from HuggingFace
# MAGIC %md
# MAGIC # Import nvidia Open-SWE-Traces from HuggingFace
# MAGIC
# MAGIC Loads the `nvidia/Open-SWE-Traces` dataset from HuggingFace and writes each split as a Delta table into `main.nvidia_open_swe_traces`.

# COMMAND ----------

# DBTITLE 1,Configuration
# MAGIC %sql
# MAGIC -- ===================================================================
# MAGIC -- CONFIGURATION — adjust these to your target catalog and schema
# MAGIC -- ===================================================================
# MAGIC CREATE WIDGET TEXT target_schema DEFAULT 'main.nvidia_open_swe_traces';
# MAGIC CREATE WIDGET TEXT traces_table DEFAULT 'main.nvidia_open_swe_traces.traces';

# COMMAND ----------

# DBTITLE 1,Install dependencies
# MAGIC %pip install datasets huggingface_hub
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Create target schema
# MAGIC %sql
# MAGIC CREATE SCHEMA IF NOT EXISTS IDENTIFIER(:target_schema)

# COMMAND ----------

# DBTITLE 1,Load dataset and write Delta tables
from huggingface_hub import snapshot_download
import os
import pandas as pd
from pyspark.sql.functions import col

repo_id = "nvidia/Open-SWE-Traces"
table_name = dbutils.widgets.get("traces_table")

# Use cached download if available, otherwise download now
print("Locating dataset (uses local cache if already downloaded)...")
local_dir = snapshot_download(repo_id, repo_type="dataset")
print(f"Local path: {local_dir}")

# Collect all parquet files sorted for deterministic ordering
parquet_files = sorted(
    os.path.join(root, f)
    for root, _, files in os.walk(local_dir)
    for f in files
    if f.endswith(".parquet")
)

if not parquet_files:
    print("No parquet files found. Listing repo contents:")
    for root, _, files in os.walk(local_dir):
        for f in files:
            print(f"  {os.path.relpath(os.path.join(root, f), local_dir)}")
    raise RuntimeError("No parquet files found.")

# Drop and recreate so reruns always start clean (overwriteSchema not allowed with Table ACLs)
spark.sql(f"DROP TABLE IF EXISTS {table_name}")
print(f"Found {len(parquet_files)} parquet file(s). Writing to {table_name}...\n")

errors = []
for i, pf in enumerate(parquet_files):
    rel_path = os.path.relpath(pf, local_dir)
    pdf = None
    sdf = None
    try:
        # Read one file at a time — avoids OOM from loading all 40GB into memory at once
        pdf = pd.read_parquet(pf)
        pdf["_source_path"] = rel_path  # preserve agent/model/benchmark context
        sdf = spark.createDataFrame(pdf)

        # FIX: some parquet files type metadata.teacher_model.reasoning_effort as void
        # (all nulls), while others type it as string.  Normalise to string so every
        # file has an identical schema and Delta append never hits a mismatch.
        sdf = sdf.withColumn(
            "metadata",
            col("metadata").withField(
                "teacher_model",
                col("metadata.teacher_model").withField(
                    "reasoning_effort",
                    col("metadata.teacher_model.reasoning_effort").cast("string"),
                ),
            ),
        )

        (
            sdf.write.format("delta")
            .mode("append")  # append to build up the table file by file
            .saveAsTable(table_name)
        )
        print(f"[{i+1:>3}/{len(parquet_files)}] {rel_path}  ({len(pdf):,} rows)")
    except Exception as e:
        print(f"[{i+1:>3}/{len(parquet_files)}] ERROR: {rel_path} — {e}")
        errors.append((rel_path, str(e)))
    finally:
        # Explicitly free memory after each file
        if pdf is not None:
            del pdf
        if sdf is not None:
            del sdf

print(f"\nDone. {len(parquet_files) - len(errors)}/{len(parquet_files)} files written to {table_name}.")
if errors:
    print(f"{len(errors)} file(s) failed:")
    for path, err in errors:
        print(f"  {path}: {err}")

# COMMAND ----------

# DBTITLE 1,Verify imported table
# MAGIC %sql
# MAGIC SELECT _source_path, COUNT(*) AS row_count
# MAGIC FROM IDENTIFIER(:traces_table)
# MAGIC GROUP BY _source_path
# MAGIC ORDER BY _source_path
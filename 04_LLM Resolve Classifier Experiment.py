# Databricks notebook source
# DBTITLE 1,Title
# MAGIC %md
# MAGIC # LLM Resolve Classifier Experiment
# MAGIC
# MAGIC Uses Databricks `ai_classify()` to predict whether an SWE agent trajectory resolved its issue, then evaluates the LLM's predictions against ground-truth `resolved` labels from the evaluation harness.
# MAGIC
# MAGIC **Goal:** The `swe.resolved` column in the traces table has values 1 (resolved), 0 (unresolved), and -1 (ungraded / harness failure). 52K trajectories are ungraded. This experiment tests whether an LLM can replace the harness by classifying trajectories as resolved or unresolved based on the issue description and the agent's code patch.
# MAGIC
# MAGIC **Method:**
# MAGIC 1. Sample 1,000 graded trajectories (stratified 50/50 resolved/unresolved, balanced across frameworks and languages)
# MAGIC 2. Run `ai_classify()` with confidence scores and rationales
# MAGIC 3. Compute accuracy, precision, recall, F1, and confusion matrix
# MAGIC 4. Log all metrics, parameters, and sample predictions to MLflow

# COMMAND ----------

# DBTITLE 1,Configuration
# MAGIC %sql
# MAGIC -- ===================================================================
# MAGIC -- CONFIGURATION — adjust these to your environment
# MAGIC -- ===================================================================
# MAGIC CREATE WIDGET TEXT traces_table DEFAULT 'main.nvidia_open_swe_traces.traces';
# MAGIC -- The LLM endpoint and judge hyperparameters are configured in the
# MAGIC -- 'Choose LLM endpoint' cell further below.

# COMMAND ----------

# DBTITLE 1,Sample graded trajectories
# MAGIC %sql
# MAGIC -- Stratified sample: 500 resolved + 500 unresolved, randomly sampled with fixed seed
# MAGIC CREATE OR REPLACE TEMP VIEW eval_sample AS
# MAGIC WITH graded AS (
# MAGIC   SELECT
# MAGIC     instance_id, repo, language, resolved, trajectory_id,
# MAGIC     messages[0].content AS issue_text,
# MAGIC     SUBSTRING(metadata.model_patch.patch, 1, 8000) AS model_patch,
# MAGIC     metadata.model_patch.num_modified_lines AS patch_lines,
# MAGIC     metadata.category AS category,
# MAGIC     regexp_extract(_source_path, '^data/([^/]+)/', 1) AS framework
# MAGIC   FROM IDENTIFIER(:traces_table)
# MAGIC   WHERE resolved IN (0, 1)
# MAGIC     AND SIZE(messages) > 0
# MAGIC     AND messages[0].content IS NOT NULL
# MAGIC     AND metadata.model_patch.patch IS NOT NULL
# MAGIC ),
# MAGIC ranked AS (
# MAGIC   SELECT *, ROW_NUMBER() OVER (PARTITION BY resolved ORDER BY rand(42)) AS rn
# MAGIC   FROM graded
# MAGIC )
# MAGIC SELECT instance_id, repo, language, resolved, trajectory_id,
# MAGIC        issue_text, model_patch, patch_lines, category, framework
# MAGIC FROM ranked
# MAGIC WHERE rn <= 500;
# MAGIC
# MAGIC SELECT resolved, COUNT(*) AS n FROM eval_sample GROUP BY resolved ORDER BY resolved

# COMMAND ----------

# DBTITLE 1,Run ai_classify on sample
# MAGIC %sql
# MAGIC -- Run ai_classify on each trajectory: given the issue + the agent's patch, is it resolved?
# MAGIC -- Uses v2.1 with confidence scores and rationales for richer evaluation
# MAGIC CREATE OR REPLACE TEMP VIEW classify_results AS
# MAGIC SELECT
# MAGIC   instance_id,
# MAGIC   resolved,
# MAGIC   repo,
# MAGIC   language,
# MAGIC   framework,
# MAGIC   category,
# MAGIC   patch_lines,
# MAGIC   ai_classify(
# MAGIC     CONCAT(
# MAGIC       '## Issue Description\n', SUBSTRING(issue_text, 1, 4000),
# MAGIC       '\n\n## Proposed Code Changes (diff)\n', model_patch
# MAGIC     ),
# MAGIC     '{
# MAGIC       "resolved": "The proposed code changes correctly and completely address the issue described above. The changes would pass tests and fix the problem.",
# MAGIC       "unresolved": "The proposed code changes do not correctly address the issue, are incomplete, introduce new bugs, or are unlikely to pass tests."
# MAGIC     }',
# MAGIC     MAP(
# MAGIC       'version', '2.1',
# MAGIC       'instructions', 'You are an expert software engineer evaluating whether an AI coding agent successfully resolved a GitHub issue. Review the issue description and the agent\'s proposed code changes (unified diff). Classify as resolved only if the changes plausibly fix the issue. If the patch is empty, trivially wrong, or does not address the core problem, classify as unresolved.',
# MAGIC       'enableConfidenceScores', 'true',
# MAGIC       'enableRationales', 'true'
# MAGIC     )
# MAGIC   ) AS classification
# MAGIC FROM eval_sample;
# MAGIC
# MAGIC -- Quick look at the raw results
# MAGIC SELECT
# MAGIC   instance_id,
# MAGIC   resolved,
# MAGIC   classification:response[0].value::STRING AS predicted_label,
# MAGIC   classification:response[0].confidence_score::DOUBLE AS confidence,
# MAGIC   classification:response[0].rationale::STRING AS rationale
# MAGIC FROM classify_results
# MAGIC LIMIT 20

# COMMAND ----------

# DBTITLE 1,Evaluate and log to MLflow
import mlflow
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
import matplotlib.pyplot as plt


# ── 1. Collect results into a pandas DataFrame ──────────────────────────
results_df = spark.sql("""
  SELECT
    instance_id,
    resolved,
    repo,
    language,
    framework,
    category,
    patch_lines,
    classification:response[0].value::STRING      AS predicted_label,
    classification:response[0].confidence_score::DOUBLE AS confidence,
    classification:response[0].rationale::STRING   AS rationale
  FROM classify_results
""").toPandas()

print(f"Total predictions: {len(results_df)}")
print(f"Prediction failures (NULL): {results_df['predicted_label'].isna().sum()}")

# Drop any rows where ai_classify returned NULL (error cases)
clean_df = results_df.dropna(subset=['predicted_label']).copy()
print(f"Valid predictions: {len(clean_df)}")

# ── 2. Normalise labels to 0/1 ──────────────────────────────────────────
clean_df['y_true'] = clean_df['resolved'].astype(int)
clean_df['y_pred'] = (clean_df['predicted_label'] == 'resolved').astype(int)

# ── 3. Compute metrics ─────────────────────────────────────────────────
accuracy  = accuracy_score(clean_df['y_true'], clean_df['y_pred'])
precision = precision_score(clean_df['y_true'], clean_df['y_pred'], pos_label=1)
recall    = recall_score(clean_df['y_true'], clean_df['y_pred'], pos_label=1)
f1        = f1_score(clean_df['y_true'], clean_df['y_pred'], pos_label=1)

# AUC using confidence scores as probabilities
if clean_df['confidence'].notna().any() and clean_df['y_true'].nunique() > 1:
    auc = roc_auc_score(clean_df['y_true'], clean_df['confidence'].fillna(0.5))
else:
    auc = None

cm = confusion_matrix(clean_df['y_true'], clean_df['y_pred'], labels=[0, 1])
tn, fp, fn, tp = cm.ravel()

print("\n" + "="*60)
print("LLM Resolve Classifier — Performance Metrics")
print("="*60)
print(f"  Accuracy:  {accuracy:.4f}")
print(f"  Precision: {precision:.4f}  (of predicted-resolved, how many truly resolved)")
print(f"  Recall:    {recall:.4f}  (of truly resolved, how many predicted resolved)")
print(f"  F1 Score:  {f1:.4f}")
if auc:
    print(f"  AUC-ROC:   {auc:.4f}  (using confidence scores)")
print(f"\n  Confusion Matrix:")
print(f"                   Predicted unresolved  Predicted resolved")
print(f"  Actual unresolved      {tn:>5}              {fp:>5}")
print(f"  Actual resolved        {fn:>5}              {tp:>5}")
print(f"\n  False Positive Rate: {fp/(fp+tn):.4f}" if (fp+tn) > 0 else "")
print(f"  False Negative Rate: {fn/(fn+tp):.4f}" if (fn+tp) > 0 else "")
print("="*60)

# Per-language breakdown
print("\nPer-language performance:")
for lang, grp in clean_df.groupby('language'):
    if len(grp) >= 10:
        acc_l = accuracy_score(grp['y_true'], grp['y_pred'])
        f1_l  = f1_score(grp['y_true'], grp['y_pred'], zero_division=0)
        print(f"  {lang:15s}  n={len(grp):>4}  acc={acc_l:.3f}  f1={f1_l:.3f}")

# Per-framework breakdown
print("\nPer-framework performance:")
for fw, grp in clean_df.groupby('framework'):
    if len(grp) >= 10:
        acc_f = accuracy_score(grp['y_true'], grp['y_pred'])
        f1_f  = f1_score(grp['y_true'], grp['y_pred'], zero_division=0)
        print(f"  {fw:15s}  n={len(grp):>4}  acc={acc_f:.3f}  f1={f1_f:.3f}")

# ── 4. Confusion matrix heatmap ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, cmap='Blues')
for i in range(2):
  for j in range(2):
    ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=14)
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(['unresolved', 'resolved'])
ax.set_yticklabels(['unresolved', 'resolved'])
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title('Confusion Matrix: LLM Resolve Classifier')
plt.tight_layout()
display(fig)
plt.close(fig)

# ── 5. Log everything to MLflow ──────────────────────────────────────────
# MLflow experiment in your user folder (shared by all runs in this notebook)
MLFLOW_EXPERIMENT = f"/Users/{spark.sql('SELECT current_user()').first()[0]}/swe_resolve_llm_classifier"
mlflow.set_experiment(MLFLOW_EXPERIMENT)

with mlflow.start_run(run_name="ai_classify_resolved_baseline") as run:
    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1", f1)
    if auc:
        mlflow.log_metric("auc_roc", auc)
    mlflow.log_metric("tn", int(tn))
    mlflow.log_metric("fp", int(fp))
    mlflow.log_metric("fn", int(fn))
    mlflow.log_metric("tp", int(tp))
    mlflow.log_metric("sample_size", len(clean_df))
    mlflow.log_metric("null_predictions", int(results_df['predicted_label'].isna().sum()))

    # Log parameters
    mlflow.log_param("classifier", "ai_classify")
    mlflow.log_param("version", "2.1")
    mlflow.log_param("sample_size", 1000)
    mlflow.log_param("stratification", "500 resolved + 500 unresolved, random seed 42")
    mlflow.log_param("issue_text_max_chars", 4000)
    mlflow.log_param("patch_max_chars", 8000)
    mlflow.log_param("enable_confidence_scores", True)
    mlflow.log_param("enable_rationales", True)
    mlflow.log_param("labels", 'resolved / unresolved')

    # Log the confusion matrix figure
    fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
    im_cm = ax_cm.imshow(cm, cmap='Blues')
    for i in range(2):
      for j in range(2):
        ax_cm.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=14)
    ax_cm.set_xticks([0, 1])
    ax_cm.set_yticks([0, 1])
    ax_cm.set_xticklabels(['unresolved', 'resolved'])
    ax_cm.set_yticklabels(['unresolved', 'resolved'])
    ax_cm.set_xlabel('Predicted')
    ax_cm.set_ylabel('Actual')
    ax_cm.set_title('Confusion Matrix')
    plt.tight_layout()
    mlflow.log_figure(fig_cm, "confusion_matrix.png")
    plt.close(fig_cm)

    # Log sample predictions as a table artifact
    sample_log = clean_df[['instance_id', 'repo', 'language', 'framework',
                           'resolved', 'predicted_label', 'confidence', 'rationale']].head(50)
    mlflow.log_table(sample_log, "sample_predictions.json")

    # Log per-language metrics
    for lang, grp in clean_df.groupby('language'):
        if len(grp) >= 10:
            mlflow.log_metric(f"accuracy_{lang}", accuracy_score(grp['y_true'], grp['y_pred']))
            mlflow.log_metric(f"f1_{lang}", f1_score(grp['y_true'], grp['y_pred'], zero_division=0))

    # Log per-framework metrics
    for fw, grp in clean_df.groupby('framework'):
        if len(grp) >= 10:
            mlflow.log_metric(f"accuracy_{fw}", accuracy_score(grp['y_true'], grp['y_pred']))
            mlflow.log_metric(f"f1_{fw}", f1_score(grp['y_true'], grp['y_pred'], zero_division=0))

    run_id = run.info.run_id
    print(f"\nMLflow run logged: {run_id}")
    print(f"Experiment: /experiments/swe_resolve_llm_classifier")
    print(f"View at: {mlflow.get_tracking_uri()}")

# COMMAND ----------

# DBTITLE 1,Error analysis: misclassified examples
# ── 6. Error analysis: inspect misclassified examples ─────────────────
misclassified = clean_df[clean_df['y_true'] != clean_df['y_pred']].copy()

print(f"Misclassified: {len(misclassified)} / {len(clean_df)} ({100*len(misclassified)/len(clean_df):.1f}%)")
print(f"  False Positives (predicted resolved, actually unresolved): {(misclassified['y_pred'] == 1).sum()}")
print(f"  False Negatives (predicted unresolved, actually resolved): {(misclassified['y_pred'] == 0).sum()}")

# Show false positives — the LLM thought it was resolved but it wasn't
print("\n--- False Positives (LLM said resolved, harness said no) ---")
fp_examples = misclassified[misclassified['y_pred'] == 1].head(10)
for _, row in fp_examples.iterrows():
    print(f"\n  {row['instance_id']} ({row['language']}, {row['framework']})")
    print(f"  Confidence: {row['confidence']:.2f}")
    print(f"  Rationale: {row['rationale'][:200]}" if pd.notna(row['rationale']) else "  Rationale: N/A")
    print(f"  Patch lines: {row['patch_lines']}")

# Show false negatives — the LLM thought it wasn't resolved but it was
print("\n--- False Negatives (LLM said unresolved, harness said yes) ---")
fn_examples = misclassified[misclassified['y_pred'] == 0].head(10)
for _, row in fn_examples.iterrows():
    print(f"\n  {row['instance_id']} ({row['language']}, {row['framework']})")
    print(f"  Confidence: {row['confidence']:.2f}")
    print(f"  Rationale: {row['rationale'][:200]}" if pd.notna(row['rationale']) else "  Rationale: N/A")
    print(f"  Patch lines: {row['patch_lines']}")

# Confidence distribution by correctness
correct = clean_df[clean_df['y_true'] == clean_df['y_pred']]
incorrect = clean_df[clean_df['y_true'] != clean_df['y_pred']]

fig2, ax2 = plt.subplots(figsize=(8, 4))
if len(correct) > 0:
    ax2.hist(correct['confidence'].dropna(), bins=20, alpha=0.6, label='Correct', color='green')
if len(incorrect) > 0:
    ax2.hist(incorrect['confidence'].dropna(), bins=20, alpha=0.6, label='Incorrect', color='red')
ax2.set_xlabel('Confidence Score')
ax2.set_ylabel('Count')
ax2.set_title('Confidence Distribution: Correct vs Incorrect Predictions')
ax2.legend()
plt.tight_layout()
display(fig2)
plt.close(fig2)

# COMMAND ----------

# DBTITLE 1,Custom LLM judge section
# MAGIC %md
# MAGIC ## Custom LLM judge with `ai_query`
# MAGIC
# MAGIC The `ai_classify` baseline was near-random (51% accuracy). Two causes:
# MAGIC * **Wrong input**: `messages[0]` is the agent's *system prompt*, not the issue. The real issue sits in the first `user` message inside `<pr_description>` (sweagent, minisweagent) or `<issue_description>` (openhands) tags.
# MAGIC * **Fixed model / no reasoning**: `ai_classify` gives no control over the model or prompt.
# MAGIC
# MAGIC This section uses `ai_query` against **any serving endpoint you choose** (set the `llm_endpoint` widget), with a custom judge prompt, temperature 0, and structured JSON output (`reasoning`, `verdict`, `confidence`). It reuses the same 1,000 trajectories so runs are directly comparable in MLflow.

# COMMAND ----------

# DBTITLE 1,Choose LLM endpoint and judge hyperparameters
from databricks.sdk import WorkspaceClient

# List chat-capable endpoints so you can pick one
w = WorkspaceClient()
available = sorted(
    e.name for e in w.serving_endpoints.list()
    if not any(k in e.name for k in ['embed', 'bge', 'gte'])
)
print("Available endpoints:")
for n in available:
    print("  ", n)

dbutils.widgets.text("llm_endpoint", "databricks-claude-sonnet-4-5", "LLM endpoint")

# MLflow experiment path — defaults to your user folder; override here if needed
MLFLOW_EXPERIMENT = f"/Users/{spark.sql('SELECT current_user()').first()[0]}/swe_resolve_llm_classifier"
LLM_ENDPOINT = dbutils.widgets.get("llm_endpoint")
assert LLM_ENDPOINT in available, f"Endpoint '{LLM_ENDPOINT}' not found - set the llm_endpoint widget to one of the names above"

ISSUE_MAX_CHARS = 16000
PATCH_MAX_CHARS = 40000

# Agent trajectory context: the agent's own test runs and its final messages
INCLUDE_AGENT_CONTEXT = True
N_TEST_RUNS = 3                 # last N test commands the agent ran
TEST_OUTPUT_TAIL_CHARS = 1500   # tail of each test output (pass/fail summary is at the end)
N_FINAL_MSGS = 2                # last N assistant messages
FINAL_MSG_MAX_CHARS = 1500
TEST_CMD_REGEX = (r"(pytest|python[0-9.]* -m (pytest|unittest)|unittest|tox|nox|go test|cargo test|cargo nextest|"
                  r"npm (run )?test|yarn (run )?test|pnpm (run )?test|npx (jest|vitest|mocha)|jest|vitest|mocha|"
                  r"mvn .*test|gradle.*test|phpunit|rspec|rake test|make (check|test)|dotnet test|ctest|"
                  r"runtests|run_tests|test_[a-z0-9_]*[.]py|reproduce)")
print(f"\nUsing endpoint: {LLM_ENDPOINT}")

# COMMAND ----------

# DBTITLE 1,Build judge inputs with real issue text
# Rebuild inputs for the SAME 1,000 trajectories, extracting the actual issue text
traces_table = dbutils.widgets.get("traces_table")
spark.sql(f"""
CREATE OR REPLACE TEMP VIEW eval_sample_v2 AS
WITH base AS (
  SELECT
    s.instance_id, s.repo, s.language, s.resolved, s.trajectory_id,
    s.category, s.framework, s.patch_lines,
    filter(t.messages, m -> m.role = 'user')[0].content AS first_user_msg,
    t.metadata.model_patch.patch AS full_patch,
    -- Pair every assistant tool call with the following tool observation, keep the ones that run tests
    filter(
      transform(sequence(0, SIZE(t.messages) - 2), i -> named_struct(
        'role', t.messages[i].role,
        'cmd',  COALESCE(get_json_object(t.messages[i].tool_calls[0].function.arguments, '$.command'), ''),
        'out',  t.messages[i + 1].content)),
      x -> x.role = 'assistant' AND x.cmd RLIKE '{TEST_CMD_REGEX}'
    ) AS test_runs,
    -- Assistant messages with non-empty natural-language content (agent's own conclusions)
    filter(t.messages, m -> m.role = 'assistant' AND LENGTH(TRIM(m.content)) > 0) AS asst_msgs
  FROM eval_sample s
  JOIN {traces_table} t
    ON s.trajectory_id = t.trajectory_id
)
SELECT *,
  SUBSTRING(COALESCE(
    NULLIF(regexp_extract(first_user_msg, '(?s)<pr_description>(.*?)</pr_description>', 1), ''),
    NULLIF(regexp_extract(first_user_msg, '(?s)<issue_description>(.*?)</issue_description>', 1), ''),
    first_user_msg
  ), 1, {ISSUE_MAX_CHARS}) AS issue_text,
  SUBSTRING(full_patch, 1, {PATCH_MAX_CHARS}) AS model_patch,
  LENGTH(full_patch) > {PATCH_MAX_CHARS} AS patch_truncated,
  SIZE(test_runs) AS n_test_runs,
  -- Last N test runs: command + tail of output (summaries such as 'X passed, Y failed' are at the end)
  IF(SIZE(test_runs) = 0, '(the agent did not run any recognisable test command)',
     array_join(transform(
       slice(test_runs, GREATEST(SIZE(test_runs) - {N_TEST_RUNS - 1}, 1), {N_TEST_RUNS}),
       x -> CONCAT('$ ', SUBSTRING(x.cmd, 1, 300), '\n', RIGHT(COALESCE(x.out, ''), {TEST_OUTPUT_TAIL_CHARS}))),
     '\n\n---\n\n')) AS test_context,
  -- Last N assistant messages
  array_join(transform(
    slice(asst_msgs, GREATEST(SIZE(asst_msgs) - {N_FINAL_MSGS - 1}, 1), {N_FINAL_MSGS}),
    m -> SUBSTRING(m.content, 1, {FINAL_MSG_MAX_CHARS})), '\n\n---\n\n') AS final_messages
FROM base
""")

display(spark.sql("""
  SELECT framework, resolved, COUNT(*) AS n,
         ROUND(AVG(LENGTH(issue_text))) AS avg_issue_chars,
         SUM(CAST(patch_truncated AS INT)) AS truncated_patches,
         ROUND(AVG(n_test_runs), 1) AS avg_test_runs,
         SUM(CAST(n_test_runs = 0 AS INT)) AS no_test_runs,
         ROUND(AVG(LENGTH(test_context) + LENGTH(final_messages))) AS avg_extra_context_chars
  FROM eval_sample_v2 GROUP BY ALL ORDER BY ALL
"""))

# COMMAND ----------

# DBTITLE 1,Define judge prompt
JUDGE_PROMPT = """You are a senior software engineer acting as a code reviewer for an automated benchmark (SWE-bench style).
An AI coding agent was given a GitHub issue and produced a patch. Hidden unit tests (written by the original maintainers for the real fix) will be run against the patch.
Predict whether the hidden tests will PASS with this patch.

Guidelines:
- Identify the exact behaviour the issue asks for, including edge cases and expected error messages/return values.
- Check that the patch changes the right code path, handles all cases mentioned, and does not break existing behaviour.
- Changes to only tests, docs, or reproduction scripts do not fix the issue.
- Partial fixes, wrong function signatures, or behaviour that differs from what the issue specifies usually fail hidden tests.
- You also see the agent's own test runs and its final messages. Use them as evidence, but be sceptical:
  agents often run only their own reproduction scripts, ignore failing tests, or claim success without proof.
  Failing or erroring tests, no test runs at all, or tests unrelated to the issue are warning signs.
  Passing runs of the repository's existing test suite covering the changed code are positive evidence.
- Be calibrated: roughly 40-50% of agent patches pass.

Respond with JSON: reasoning (concise, max 5 sentences), verdict ("resolved" or "unresolved"), confidence (probability 0-1 that the patch is resolved).

### Repository
{repo} ({language})

### Issue
{issue}

### Agent patch (unified diff){truncated_note}
{patch}
{agent_context}"""

AGENT_CONTEXT_TEMPLATE = """
### Agent's last test runs (command + tail of output)
{test_context}

### Agent's final messages
{final_messages}
"""

# Register the prompt template as a SQL-friendly format string
sql_prompt = (
    JUDGE_PROMPT.replace("'", "\\'")
    .replace("{repo}", "', repo, '")
    .replace("{language}", "', language, '")
    .replace("{issue}", "', issue_text, '")
    .replace("{truncated_note}", "', IF(patch_truncated, ' [TRUNCATED]', ''), '")
    .replace("{patch}", "', COALESCE(model_patch, '(empty patch)'), '")
    .replace("{agent_context}", AGENT_CONTEXT_TEMPLATE.replace("'", "\\'") if INCLUDE_AGENT_CONTEXT else "")
    .replace("{test_context}", "', test_context, '")
    .replace("{final_messages}", "', COALESCE(final_messages, '(none)'), '")
)

# COMMAND ----------

# DBTITLE 1,Run custom LLM judge via ai_query
# Prompt (JUDGE_PROMPT / sql_prompt) is defined in the 'Define judge prompt' cell above
judged = spark.sql(f"""
  SELECT *, ai_query(
      '{LLM_ENDPOINT}',
      CONCAT('{sql_prompt}'),
      modelParameters => named_struct('max_tokens', 1024),
      responseFormat => 'STRUCT<judgement: STRUCT<reasoning: STRING, verdict: STRING, confidence: DOUBLE>>',
      failOnError => false
    ) AS llm
  FROM eval_sample_v2
""")

# Materialise once so the LLM is only called a single time per row
judged_pdf = (judged
  .selectExpr(
    "instance_id", "repo", "language", "framework", "category", "patch_lines", "patch_truncated",
    "n_test_runs", "resolved",
    # The endpoint returns the inner judgement fields flat: {"reasoning":..., "verdict":..., "confidence":...}
    "from_json(llm.result, 'reasoning STRING, verdict STRING, confidence DOUBLE') AS j",
    "llm.result AS raw_response",
    "llm.errorMessage AS error_message")
  .selectExpr("* EXCEPT (j)", "lower(trim(j.verdict)) AS predicted_label",
              "j.confidence AS confidence", "j.reasoning AS rationale")
  .toPandas())

# Persist raw judge output to MLflow immediately, so a compute reset does not lose the (expensive) LLM calls
import mlflow
mlflow.set_experiment(MLFLOW_EXPERIMENT)
JUDGE_RUN_NAME = f"ai_query_{LLM_ENDPOINT}" + ("_with_agent_context" if INCLUDE_AGENT_CONTEXT else "")
with mlflow.start_run(run_name=JUDGE_RUN_NAME) as _r:
    mlflow.log_table(judged_pdf, "judge_raw.json")
    JUDGE_RUN_ID = _r.info.run_id
print(f"Raw judge output saved to MLflow run {JUDGE_RUN_ID} (artifact judge_raw.json)")

print(f"Rows judged: {len(judged_pdf)}")
print(f"Errors: {judged_pdf['error_message'].notna().sum()}")
print(judged_pdf['predicted_label'].value_counts(dropna=False))
display(judged_pdf[['instance_id', 'resolved', 'predicted_label', 'confidence', 'rationale']].head(10))

# COMMAND ----------

# DBTITLE 1,Inspect full judge prompt for one example
# Render the exact prompt the judge receives for one trajectory (no LLM call)
EXAMPLE_INSTANCE = None   # set to an instance_id to inspect a specific one; None = first row with test runs

_where = f"instance_id = '{EXAMPLE_INSTANCE}'" if EXAMPLE_INSTANCE else "n_test_runs > 0"
ex = spark.sql(f"""
  SELECT instance_id, framework, resolved, n_test_runs, CONCAT('{sql_prompt}') AS judge_prompt
  FROM eval_sample_v2 WHERE {_where}
  ORDER BY instance_id LIMIT 1
""").first()

print(f"instance_id: {ex.instance_id} | framework: {ex.framework} | ground truth resolved: {ex.resolved} | "
      f"test runs found: {ex.n_test_runs} | prompt length: {len(ex.judge_prompt):,} chars")

# What the judge answered for this row in the last run
if 'judged_pdf' in globals():
    v = judged_pdf[judged_pdf['instance_id'] == ex.instance_id]
    if len(v):
        r = v.iloc[0]
        print(f"judge verdict: {r.predicted_label} (confidence {r.confidence})\njudge reasoning: {r.rationale}")

print("\n" + "=" * 100 + "\n")
print(ex.judge_prompt)

# COMMAND ----------

# DBTITLE 1,Evaluate custom judge and log to MLflow
import mlflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, roc_auc_score)

def evaluate_and_log(pdf, run_name, params, run_id=None):
    """Compute classification metrics vs ground truth and log them to MLflow."""
    df = pdf[pdf['predicted_label'].isin(['resolved', 'unresolved'])].copy()
    df['y_true'] = df['resolved'].astype(int)
    df['y_pred'] = (df['predicted_label'] == 'resolved').astype(int)

    m = {
        'accuracy': accuracy_score(df['y_true'], df['y_pred']),
        'precision': precision_score(df['y_true'], df['y_pred'], zero_division=0),
        'recall': recall_score(df['y_true'], df['y_pred'], zero_division=0),
        'f1': f1_score(df['y_true'], df['y_pred'], zero_division=0),
        'sample_size': len(df),
        'invalid_predictions': len(pdf) - len(df),
    }
    # The judge reports confidence in ITS verdict, so convert to P(resolved) before computing AUC
    df['p_resolved'] = np.where(df['y_pred'] == 1, df['confidence'], 1 - df['confidence'])
    if df['confidence'].notna().sum() > 0:
        m['auc_roc'] = roc_auc_score(df['y_true'], df['p_resolved'].fillna(0.5))
        m['auc_roc_raw_conf'] = roc_auc_score(df['y_true'], df['confidence'].fillna(0.5))
    cm_ = confusion_matrix(df['y_true'], df['y_pred'], labels=[0, 1])
    m.update(dict(zip(['tn', 'fp', 'fn', 'tp'], map(int, cm_.ravel()))))

    breakdown = []
    for dim in ['framework', 'language']:
        for val, g in df.groupby(dim):
            if len(g) >= 10:
                breakdown.append({'dimension': dim, 'value': val, 'n': len(g),
                                  'accuracy': accuracy_score(g['y_true'], g['y_pred']),
                                  'f1': f1_score(g['y_true'], g['y_pred'], zero_division=0)})
    breakdown_df = pd.DataFrame(breakdown)

    fig_, ax_ = plt.subplots(figsize=(5, 4))
    ax_.imshow(cm_, cmap='Blues')
    for i in range(2):
        for j in range(2):
            ax_.text(j, i, str(cm_[i, j]), ha='center', va='center', fontsize=14)
    ax_.set_xticks([0, 1]); ax_.set_yticks([0, 1])
    ax_.set_xticklabels(['unresolved', 'resolved']); ax_.set_yticklabels(['unresolved', 'resolved'])
    ax_.set_xlabel('Predicted'); ax_.set_ylabel('Actual'); ax_.set_title(run_name)
    plt.tight_layout()

    with (mlflow.start_run(run_id=run_id) if run_id else mlflow.start_run(run_name=run_name)) as r:
        mlflow.log_params(params)
        mlflow.log_metrics({k: float(v) for k, v in m.items()})
        for _, b in breakdown_df.iterrows():
            mlflow.log_metric(f"accuracy_{b['dimension']}_{b['value']}", b['accuracy'])
        mlflow.log_figure(fig_, 'confusion_matrix.png')
        mlflow.log_table(df[['instance_id', 'repo', 'language', 'framework', 'resolved',
                             'n_test_runs', 'predicted_label', 'confidence', 'p_resolved', 'rationale']], 'predictions.json')
        mlflow.log_text(JUDGE_PROMPT.replace('{agent_context}', AGENT_CONTEXT_TEMPLATE if INCLUDE_AGENT_CONTEXT else ''),
                        'judge_prompt.txt')
        print(f"MLflow run: {r.info.run_id}")
    display(fig_); plt.close(fig_)
    return m, breakdown_df

mlflow.set_experiment(MLFLOW_EXPERIMENT)

# Recover judge output from MLflow if the session was reset after the judge cell ran
if 'judged_pdf' not in globals():
    judged_pdf = mlflow.load_table("judge_raw.json", run_ids=[JUDGE_RUN_ID])
    print(f"Loaded {len(judged_pdf)} judged rows from MLflow run {JUDGE_RUN_ID}")

custom_metrics, custom_breakdown = evaluate_and_log(
    judged_pdf,
    run_id=globals().get('JUDGE_RUN_ID'),
    run_name=f"ai_query_{LLM_ENDPOINT}" + ("_with_agent_context" if INCLUDE_AGENT_CONTEXT else ""),
    params={'classifier': 'ai_query', 'endpoint': LLM_ENDPOINT, 'temperature': 0.0,
            'issue_max_chars': ISSUE_MAX_CHARS, 'patch_max_chars': PATCH_MAX_CHARS,
            'issue_extraction': 'pr_description / issue_description tags',
            'include_agent_context': INCLUDE_AGENT_CONTEXT,
            'n_test_runs': N_TEST_RUNS, 'test_output_tail_chars': TEST_OUTPUT_TAIL_CHARS,
            'n_final_msgs': N_FINAL_MSGS, 'final_msg_max_chars': FINAL_MSG_MAX_CHARS,
            'sample': 'same 1000 as baseline (500/500, seed 42)'}
)

# Compare all runs logged in the experiment so far
runs = mlflow.search_runs(order_by=['start_time ASC'])
metric_cols = ['metrics.accuracy', 'metrics.precision', 'metrics.recall', 'metrics.f1', 'metrics.auc_roc']
comparison = (runs[['tags.mlflow.runName', 'start_time'] + [c for c in metric_cols if c in runs]]
              .rename(columns=lambda c: c.replace('metrics.', '').replace('tags.mlflow.runName', 'run'))
              .round(3))
display(comparison)
display(custom_breakdown.round(3))

# Does the judge do better when the agent actually ran tests?
if 'n_test_runs' in judged_pdf:
    _d = judged_pdf[judged_pdf['predicted_label'].isin(['resolved', 'unresolved'])].assign(
        y_pred=lambda d: (d['predicted_label'] == 'resolved').astype(int),
        ran_tests=lambda d: d['n_test_runs'] > 0)
    display(_d.groupby('ran_tests').apply(lambda g: pd.Series({
        'n': len(g), 'resolve_rate': g['resolved'].mean(),
        'accuracy': accuracy_score(g['resolved'], g['y_pred']),
        'f1': f1_score(g['resolved'], g['y_pred'], zero_division=0)})).round(3).reset_index())

# COMMAND ----------

# DBTITLE 1,Tune decision threshold on judge confidence
import mlflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

# Tune on the predictions already logged in MLflow - no new LLM calls needed
mlflow.set_experiment(MLFLOW_EXPERIMENT)
# Judge run to tune: defaults to the run logged by the judge cells above in this session.
# To tune an earlier run instead, set SOURCE_RUN_ID to its MLflow run ID.
SOURCE_RUN_ID = globals().get("JUDGE_RUN_ID")
assert SOURCE_RUN_ID, "No judge run found - run the custom LLM judge cells above or set SOURCE_RUN_ID manually"
preds = mlflow.load_table("predictions.json", run_ids=[SOURCE_RUN_ID])
preds = preds[preds['predicted_label'].isin(['resolved', 'unresolved'])].copy()
y = preds['resolved'].astype(int).values

# 1) What does 'confidence' mean? If the judge reports confidence in ITS VERDICT (not P(resolved)),
#    unresolved verdicts will also have high confidence, and we must flip them.
display(preds.groupby('predicted_label')['confidence'].describe().round(3).reset_index())
preds['p_resolved'] = np.where(preds['predicted_label'] == 'resolved', preds['confidence'], 1 - preds['confidence'])

print(f"AUC raw confidence:          {roc_auc_score(y, preds['confidence']):.3f}")
print(f"AUC verdict-adjusted score:  {roc_auc_score(y, preds['p_resolved']):.3f}")
print(f"Distinct score values: {preds['p_resolved'].nunique()}")

# 2) Metrics across thresholds (full sample - descriptive only)
thresholds = np.round(np.arange(0.05, 0.96, 0.01), 2)
def metrics_at(t, s, yy):
    p = (s >= t).astype(int)
    return dict(threshold=t, accuracy=accuracy_score(yy, p), precision=precision_score(yy, p, zero_division=0),
                recall=recall_score(yy, p, zero_division=0), f1=f1_score(yy, p, zero_division=0),
                pred_resolved_rate=p.mean())
curve = pd.DataFrame([metrics_at(t, preds['p_resolved'].values, y) for t in thresholds])

fig, ax = plt.subplots(figsize=(9, 4))
for c in ['accuracy', 'precision', 'recall', 'f1', 'pred_resolved_rate']:
    ax.plot(curve['threshold'], curve[c], label=c)
ax.axhline(y.mean(), ls=':', c='grey', label='true resolve rate')
ax.set_xlabel('threshold on p_resolved'); ax.set_title('Judge metrics vs decision threshold'); ax.legend(); ax.grid(alpha=.3)
plt.tight_layout(); display(fig)

# 3) Honest estimate: choose threshold on train folds, score on held-out fold (5-fold stratified CV)
cv_rows = []
for objective in ['accuracy', 'f1']:
    for k, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(preds, y)):
        s = preds['p_resolved'].values
        tr_curve = pd.DataFrame([metrics_at(t, s[tr], y[tr]) for t in thresholds])
        best_t = tr_curve.loc[tr_curve[objective].idxmax(), 'threshold']
        m = metrics_at(best_t, s[te], y[te]); m.update(objective=objective, fold=k)
        cv_rows.append(m)
cv = pd.DataFrame(cv_rows)
cv_summary = cv.groupby('objective')[['threshold', 'accuracy', 'precision', 'recall', 'f1', 'pred_resolved_rate']].mean().round(3)
baseline = pd.DataFrame([metrics_at(0.5, preds['p_resolved'].values, y)]).assign(objective='raw verdict (t=0.5)').set_index('objective').round(3)
display(pd.concat([baseline, cv_summary]).reset_index())

# 4) Log the tuned operating point
best_acc_t = float(cv[cv.objective == 'accuracy']['threshold'].median())
with mlflow.start_run(run_name=f"threshold_tuning_{LLM_ENDPOINT}"):
    mlflow.log_params({'source_run_id': SOURCE_RUN_ID, 'score': 'p_resolved = conf if resolved else 1-conf',
                       'cv': 'stratified 5-fold', 'chosen_threshold_accuracy': best_acc_t})
    mlflow.log_metrics({'auc_roc_raw_conf': roc_auc_score(y, preds['confidence']),
                        'auc_roc': roc_auc_score(y, preds['p_resolved']),
                        **{f'cv_{k}': float(v) for k, v in cv_summary.loc['accuracy'].items()}})
    mlflow.log_figure(fig, 'threshold_curve.png')
    mlflow.log_table(curve, 'threshold_curve.json')
plt.close(fig)
print(f"Chosen threshold (accuracy objective, median over folds): {best_acc_t}")
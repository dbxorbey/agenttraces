# Databricks notebook source
# DBTITLE 1,Title
# MAGIC %md
# MAGIC # Open SWE Traces: Usage, Error and Root Cause Analysis
# MAGIC
# MAGIC Full-table analysis of `main.nvidia_open_swe_traces.otel_spans` (55.8M spans, 364K trajectories, 3 agent frameworks, 3 models). Covers resolve rates, effort, tool usage, error taxonomy, never-solved tasks and root causes. Note: timing and token counts in the source are synthetic (derived from text length).

# COMMAND ----------

# DBTITLE 1,Configuration
# MAGIC %sql
# MAGIC -- ===================================================================
# MAGIC -- CONFIGURATION — adjust these to point to your OTel trace table
# MAGIC -- ===================================================================
# MAGIC -- For the nvidia/Open-SWE-Traces dataset processed by 02_Transform to OTel Spans:
# MAGIC CREATE WIDGET TEXT otel_spans_table DEFAULT 'main.nvidia_open_swe_traces.otel_spans';
# MAGIC -- Schema where the summary tables for the Genie agent (05_create_genie_error_agent) are written:
# MAGIC CREATE WIDGET TEXT output_schema DEFAULT 'main.nvidia_open_swe_traces';
# MAGIC
# MAGIC -- For Unity AI Gateway traces, point to your unified trace table instead, e.g.:
# MAGIC --   'system.ai_gateway.unity_gateway_otel_spans'
# MAGIC -- or your per-endpoint inference table:
# MAGIC --   'my_catalog.my_schema.my_endpoint_otel_spans'

# COMMAND ----------

# DBTITLE 1,Error heuristics (edit for your agent framework)
# MAGIC %sql
# MAGIC -- Classify every error span into a category. Patterns are evaluated
# MAGIC -- top-to-bottom; the first match wins. Add, remove, or reorder rules
# MAGIC -- to match your agent's error signatures.
# MAGIC --
# MAGIC -- For Unity Gateway traces, replace the SWE-bench-specific patterns
# MAGIC -- (go test, str_replace_editor, CTRL+C) with your own tool errors.
# MAGIC CREATE OR REPLACE TEMP VIEW classified_errors AS
# MAGIC SELECT
# MAGIC   trace_id,
# MAGIC   name AS tool,
# MAGIC   CASE
# MAGIC     WHEN status.message LIKE '%command not found%' OR status.message LIKE '%exit code 127%'
# MAGIC       THEN 'env: command not found (toolchain missing)'
# MAGIC     WHEN status.message LIKE '%No module named%' OR status.message LIKE '%Missing required plugins%'
# MAGIC       THEN 'env: missing dependency'
# MAGIC     WHEN status.message LIKE '%cannot run *_test.go%'
# MAGIC       THEN 'agent: wrong go test invocation'
# MAGIC     WHEN status.message LIKE '%No previous running command%'
# MAGIC       THEN 'agent: tool protocol misuse'
# MAGIC     WHEN status.message LIKE '%CTRL+C%' OR status.message LIKE '%exit code 130%'
# MAGIC       THEN 'timeout / interrupted command'
# MAGIC     WHEN status.message LIKE '%Syntax error%'
# MAGIC       THEN 'agent: shell syntax error'
# MAGIC     WHEN status.message LIKE '%No such file or directory%'
# MAGIC       THEN 'agent: wrong path / file not found'
# MAGIC     WHEN status.message LIKE '%test session%' OR status.message LIKE '%FAIL%'
# MAGIC       THEN 'test failure'
# MAGIC     WHEN status.message LIKE '%"output": ""%'
# MAGIC       THEN 'silent non-zero exit (e.g. grep no match)'
# MAGIC     WHEN name = 'str_replace_editor'
# MAGIC       THEN 'editor: invalid edit (old_str mismatch etc.)'
# MAGIC     ELSE 'other non-zero exit'
# MAGIC   END AS error_type,
# MAGIC   status.message AS error_message
# MAGIC FROM IDENTIFIER(:otel_spans_table)
# MAGIC WHERE status.code = 'STATUS_CODE_ERROR';

# COMMAND ----------

# DBTITLE 1,Trajectory-level view (root spans)
# MAGIC %sql
# MAGIC -- One row per trajectory. Attribute keys contain dots, so parse the VARIANT via from_json with backtick-quoted field names.
# MAGIC CREATE OR REPLACE TEMP VIEW trajectories AS
# MAGIC SELECT
# MAGIC   trace_id,
# MAGIC   a.`swe.agent_framework`                    AS framework,
# MAGIC   a.`gen_ai.request.model`                   AS model,
# MAGIC   -- swe.resolved is 1 / 0 / -1; -1 means the trajectory was never graded (evaluation harness failure)
# MAGIC   CASE a.`swe.resolved` WHEN 1 THEN 1 WHEN 0 THEN 0 END AS resolved,
# MAGIC   CASE a.`swe.resolved` WHEN 1 THEN 'resolved' WHEN 0 THEN 'unresolved' ELSE 'ungraded' END AS outcome,
# MAGIC   IF(a.`swe.resolved` = -1, 1, 0)            AS ungraded,
# MAGIC   a.`swe.repo`                               AS repo,
# MAGIC   CASE a.`swe.language` WHEN 'ts' THEN 'typescript' WHEN 'js' THEN 'javascript' ELSE a.`swe.language` END AS language,
# MAGIC   a.`swe.instance_id`                        AS instance_id,
# MAGIC   a.`swe.category`                           AS category,
# MAGIC   a.`swe.source_dataset`                     AS source_dataset,
# MAGIC   a.`swe.num_tool_calls`                     AS num_tool_calls,
# MAGIC   a.`swe.num_llm_calls`                      AS num_llm_calls,
# MAGIC   a.`gen_ai.usage.input_tokens`              AS input_tokens,
# MAGIC   a.`gen_ai.usage.output_tokens`             AS output_tokens,
# MAGIC   a.`swe.model_patch.num_modified_files`     AS patch_files,
# MAGIC   a.`swe.model_patch.num_modified_lines`     AS patch_lines,
# MAGIC   a.`swe.reference_patch.num_modified_lines` AS ref_patch_lines
# MAGIC FROM (
# MAGIC   SELECT trace_id, from_json(CAST(attributes AS STRING),
# MAGIC     'STRUCT<`swe.agent_framework`: STRING, `gen_ai.request.model`: STRING, `swe.resolved`: BIGINT, `swe.repo`: STRING, `swe.language`: STRING, `swe.instance_id`: STRING, `swe.category`: STRING, `swe.source_dataset`: STRING, `swe.num_tool_calls`: BIGINT, `swe.num_llm_calls`: BIGINT, `gen_ai.usage.input_tokens`: BIGINT, `gen_ai.usage.output_tokens`: BIGINT, `swe.model_patch.num_modified_files`: BIGINT, `swe.model_patch.num_modified_lines`: BIGINT, `swe.reference_patch.num_modified_lines`: BIGINT>') AS a
# MAGIC   FROM IDENTIFIER(:otel_spans_table)
# MAGIC   WHERE kind = 'SPAN_KIND_SERVER'
# MAGIC );
# MAGIC
# MAGIC SELECT COUNT(*) AS trajectories, COUNT(DISTINCT instance_id) AS instances, COUNT(DISTINCT repo) AS repos,
# MAGIC        ROUND(100 * AVG(ungraded), 1) AS ungraded_pct,
# MAGIC        ROUND(100 * AVG(resolved), 1) AS resolve_pct_of_graded
# MAGIC FROM trajectories

# COMMAND ----------

# DBTITLE 1,Resolve rate by framework and model
# MAGIC %sql
# MAGIC SELECT framework, model, COUNT(*) AS trajectories,
# MAGIC        ROUND(100 * AVG(ungraded), 1) AS ungraded_pct,
# MAGIC        ROUND(100 * AVG(resolved), 1) AS resolve_pct,
# MAGIC        ROUND(AVG(input_tokens) / 1e6, 2) AS avg_input_tokens_m
# MAGIC FROM trajectories
# MAGIC GROUP BY ALL
# MAGIC ORDER BY resolve_pct DESC

# COMMAND ----------

# DBTITLE 1,Resolve rate by language, category and source dataset
# MAGIC %sql
# MAGIC SELECT 'language' AS dimension, language AS value, COUNT(*) AS trajectories, ROUND(100 * AVG(ungraded), 1) AS ungraded_pct, ROUND(100 * AVG(resolved), 1) AS resolve_pct FROM trajectories GROUP BY ALL
# MAGIC UNION ALL
# MAGIC SELECT 'category', category, COUNT(*), ROUND(100 * AVG(ungraded), 1), ROUND(100 * AVG(resolved), 1) FROM trajectories GROUP BY ALL
# MAGIC UNION ALL
# MAGIC SELECT 'source_dataset', source_dataset, COUNT(*), ROUND(100 * AVG(ungraded), 1), ROUND(100 * AVG(resolved), 1) FROM trajectories GROUP BY ALL
# MAGIC ORDER BY dimension, trajectories DESC

# COMMAND ----------

# DBTITLE 1,Effort and patch size: resolved vs unresolved
# MAGIC %sql
# MAGIC SELECT framework,
# MAGIC        outcome,
# MAGIC        COUNT(*) AS trajectories,
# MAGIC        ROUND(AVG(num_tool_calls), 1) AS avg_tool_calls,
# MAGIC        ROUND(AVG(input_tokens) / 1e6, 2) AS avg_input_tokens_m,
# MAGIC        ROUND(PERCENTILE(patch_lines, 0.5), 0) AS p50_patch_lines,
# MAGIC        ROUND(AVG(patch_files), 2) AS avg_patch_files,
# MAGIC        ROUND(PERCENTILE(ref_patch_lines, 0.5), 0) AS p50_reference_patch_lines
# MAGIC FROM trajectories
# MAGIC GROUP BY ALL
# MAGIC ORDER BY framework, outcome

# COMMAND ----------

# DBTITLE 1,Tool usage and tool error rate per framework
# MAGIC %sql
# MAGIC SELECT t.framework, s.name AS tool,
# MAGIC        COUNT(*) AS calls,
# MAGIC        ROUND(100 * AVG(IF(s.status.code = 'STATUS_CODE_ERROR', 1, 0)), 2) AS error_pct
# MAGIC FROM IDENTIFIER(:otel_spans_table) s
# MAGIC JOIN trajectories t USING (trace_id)
# MAGIC WHERE s.kind = 'SPAN_KIND_INTERNAL'
# MAGIC GROUP BY ALL
# MAGIC ORDER BY framework, calls DESC

# COMMAND ----------

# DBTITLE 1,Error taxonomy across all failing tool calls
# MAGIC %sql
# MAGIC -- Uses the classified_errors view defined in the 'Error heuristics' cell above
# MAGIC SELECT error_type, COUNT(*) AS errors, COUNT(DISTINCT trace_id) AS affected_trajectories,
# MAGIC        ROUND(100 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS pct_of_errors
# MAGIC FROM classified_errors
# MAGIC GROUP BY ALL
# MAGIC ORDER BY errors DESC

# COMMAND ----------

# DBTITLE 1,Errors per trajectory vs outcome
# MAGIC %sql
# MAGIC WITH e AS (
# MAGIC   SELECT trace_id, COUNT(*) AS errors
# MAGIC   FROM classified_errors
# MAGIC   GROUP BY trace_id
# MAGIC )
# MAGIC SELECT t.framework, t.outcome,
# MAGIC        COUNT(*) AS trajectories,
# MAGIC        ROUND(AVG(COALESCE(e.errors, 0)), 2) AS avg_errors,
# MAGIC        ROUND(100 * AVG(COALESCE(e.errors, 0) / t.num_tool_calls), 2) AS error_pct_of_tool_calls
# MAGIC FROM trajectories t LEFT JOIN e USING (trace_id)
# MAGIC GROUP BY ALL
# MAGIC ORDER BY framework, outcome

# COMMAND ----------

# DBTITLE 1,Instance difficulty: never, sometimes, always solved
# MAGIC %sql
# MAGIC WITH i AS (
# MAGIC   SELECT instance_id, COUNT(*) AS attempts, SUM(resolved) AS solved
# MAGIC   FROM trajectories WHERE resolved IS NOT NULL GROUP BY instance_id
# MAGIC )
# MAGIC SELECT CASE WHEN solved = 0 THEN 'never solved' WHEN solved = attempts THEN 'always solved' ELSE 'sometimes solved' END AS bucket,
# MAGIC        COUNT(*) AS instances, SUM(attempts) AS trajectories,
# MAGIC        ROUND(100 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS pct_instances
# MAGIC FROM i
# MAGIC GROUP BY ALL
# MAGIC ORDER BY instances DESC

# COMMAND ----------

# DBTITLE 1,Hardest repos (min 300 trajectories)
# MAGIC %sql
# MAGIC SELECT repo, language, COUNT(*) AS trajectories, COUNT(DISTINCT instance_id) AS instances,
# MAGIC        ROUND(100 * AVG(ungraded), 1) AS ungraded_pct,
# MAGIC        ROUND(100 * AVG(resolved), 1) AS resolve_pct,
# MAGIC        ROUND(AVG(num_tool_calls), 0) AS avg_tool_calls
# MAGIC FROM trajectories
# MAGIC GROUP BY ALL
# MAGIC HAVING COUNT(*) >= 300
# MAGIC ORDER BY resolve_pct ASC NULLS FIRST, ungraded_pct DESC
# MAGIC LIMIT 25

# COMMAND ----------

# DBTITLE 1,Root cause check: what precedes ungraded trajectories
# MAGIC %sql
# MAGIC -- Compare environment-type errors between graded and ungraded trajectories
# MAGIC -- (uses the heuristic error taxonomy defined in the 'Error heuristics' cell above)
# MAGIC WITH e AS (
# MAGIC   SELECT trace_id,
# MAGIC     COUNT_IF(error_type = 'env: command not found (toolchain missing)') AS env_missing_tool,
# MAGIC     COUNT_IF(error_type = 'timeout / interrupted command') AS timeouts,
# MAGIC     COUNT_IF(error_type = 'env: missing dependency') AS missing_deps,
# MAGIC     COUNT(*) AS errors
# MAGIC   FROM classified_errors
# MAGIC   GROUP BY trace_id
# MAGIC )
# MAGIC SELECT t.outcome, COUNT(*) AS trajectories,
# MAGIC   ROUND(100 * AVG(IF(COALESCE(e.env_missing_tool, 0) > 0, 1, 0)), 1) AS pct_with_missing_toolchain,
# MAGIC   ROUND(100 * AVG(IF(COALESCE(e.timeouts, 0) > 0, 1, 0)), 1) AS pct_with_timeout,
# MAGIC   ROUND(100 * AVG(IF(COALESCE(e.missing_deps, 0) > 0, 1, 0)), 1) AS pct_with_missing_deps,
# MAGIC   ROUND(AVG(COALESCE(e.errors, 0)), 2) AS avg_errors,
# MAGIC   ROUND(100 * AVG(IF(t.language IN ('rust', 'java', 'c', 'cpp', 'typescript'), 1, 0)), 1) AS pct_compiled_or_ts
# MAGIC FROM trajectories t LEFT JOIN e USING (trace_id)
# MAGIC GROUP BY ALL
# MAGIC ORDER BY outcome

# COMMAND ----------

# DBTITLE 1,Key findings
# MAGIC %md
# MAGIC ## Key findings
# MAGIC
# MAGIC **Data quality first:** `swe.resolved` is 1 / 0 / **-1**. The 52K trajectories (14.3%) marked -1 were never graded. Including them as "unresolved" understates resolve rates. All rates below use graded trajectories only.
# MAGIC
# MAGIC **Usage patterns**
# MAGIC * 364K trajectories, 42K task instances (\~8.7 attempts each), 3,776 repos. Python makes up 45% of trajectories.
# MAGIC * Each trajectory has one LLM turn per tool call: 51-97 tool calls on average and 1-5.5M input tokens. The context is re-sent on every turn, so input tokens dominate.
# MAGIC * Tool mix: sweagent and minisweagent are bash-heavy. OpenHands uses `execute_bash`, `str_replace_editor` and `think`.
# MAGIC
# MAGIC **Outcomes**
# MAGIC * The graded resolve rate is 41.5%. The best result is sweagent + Qwen3.5-122B at 48.7%, and the worst is openhands + Qwen3.5-122B at 31.8%.
# MAGIC * Python resolves at 48.7%, compared with 25.5% for Rust and 16.9% for C++. Bug fixes (45%) resolve more often than feature requests (36%). Scale-SWE (47.5%) resolves more often than SWE-rebench-V2 (38.9%).
# MAGIC * 47% of graded instances were never solved by any agent or model. Only 19.5% were always solved.
# MAGIC * Failure shows up as extra effort: unresolved runs use +15-20% tool calls and +30-40% tokens. Their patches are 2-3x larger, even though the reference patch is also larger. Failing agents tend to over-edit.
# MAGIC
# MAGIC **Error patterns**
# MAGIC * Most tool errors (\~68%) are generic non-zero exits from running code. Another 12% are silent exits such as a `grep` with no match. Both are normal exploration noise, not real failures.
# MAGIC * The real issues are: editor edits that don't match (58K), timeouts or Ctrl+C (35K), wrong paths (18K), missing toolchains such as `javac` or `go` (16K), missing dependencies (11K) and wrong `go test` invocations (11K).
# MAGIC * Error count barely differs between resolved and unresolved runs (3.5 vs 4.1 for minisweagent). Tool errors are not the main reason tasks fail.
# MAGIC * sweagent's 0.06% error rate is an instrumentation artifact: its tool spans almost never set an ERROR status. Don't compare its error rates with the other frameworks.
# MAGIC
# MAGIC **Root causes**
# MAGIC 1. **Evaluation and environment failures (biggest single issue).** 50% of ungraded trajectories are compiled languages or TypeScript, versus 16% of resolved ones. They also show 2.5x the timeout rate and 2x the missing-toolchain rate. Repos such as swc (4K runs), kestra, helm, cargo, Sage/carbon and PHP-CS-Fixer are 55-100% ungraded with 0% resolved. These are broken environments, not hard tasks.
# MAGIC 2. **Language and environment gap.** Rust, Java and C/C++ environments often lack build tools or time out on compiles. The agent loops on retries instead of fixing the code.
# MAGIC 3. **Task difficulty.** Larger reference patches and feature requests are solved less often. The agent's patch grows in response, which points to scoping problems rather than tool failures.
# MAGIC 4. **Model and framework fit.** Qwen3.5-122B performs best in sweagent but worst in OpenHands. The same model varies by \~17pp depending on the scaffold.

# COMMAND ----------

# DBTITLE 1,Immediate recommendations
# MAGIC %md
# MAGIC ## Immediate recommendations
# MAGIC
# MAGIC Based on the trace analysis above, the following actions are prioritized by expected impact on resolve rate and cost efficiency.
# MAGIC
# MAGIC ### 1. Fix or retire broken evaluation environments (highest ROI)
# MAGIC 14% of all runs (52K trajectories) are ungraded — wasted compute with zero signal. Repos like `swc`, `kestra`, `helm`, and `PHP-CS-Fixer` are 55–100% ungraded.
# MAGIC * **Action:** Audit and repair Docker images / CI configs for the top-20 ungraded repos. Pre-validate that each environment can compile and run tests *before* dispatching agent runs.
# MAGIC * **Action:** Gate new task instances: reject any instance whose repo fails a 60-second smoke test (install deps → build → run one test).
# MAGIC * **Metric to track:** Ungraded rate should drop from 14.3% to <2%.
# MAGIC
# MAGIC ### 2. Route tasks to the best framework–model pair
# MAGIC The same model swings ~17 pp depending on the scaffold (Qwen3.5-122B: 48.7% in sweagent vs 31.8% in OpenHands). Routing everything through one framework leaves resolve rate on the table.
# MAGIC * **Action:** Build a lightweight task classifier (language + category + repo) that routes each instance to the framework–model pair with the highest historical resolve rate for similar tasks.
# MAGIC * **Action:** Until the classifier is live, default to sweagent + Qwen3.5-122B for Python and minisweagent + DeepSeek-V3 for compiled languages.
# MAGIC * **Metric to track:** Blended resolve rate (currently 41.5%) should rise 5–8 pp from routing alone.
# MAGIC
# MAGIC ### 3. Cap agent effort to cut waste on doomed runs
# MAGIC Unresolved runs burn 30–40% more tokens and produce 2–3× larger patches — the agent keeps editing instead of stopping. This is pure cost with no outcome improvement.
# MAGIC * **Action:** Implement a hard budget: if a trajectory exceeds 80 tool calls or 4M input tokens without a passing test, terminate early and flag for human review.
# MAGIC * **Action:** Add a "confidence gate" after the first test run: if all tests fail *and* the patch already touches >3 files, abort.
# MAGIC * **Metric to track:** Average cost per unresolved trajectory should drop 25–30%.
# MAGIC
# MAGIC ### 4. Invest in non-Python toolchain support
# MAGIC Rust (25.5%), C++ (16.9%), and Java lag far behind Python (48.7%). Missing compilers and dependency failures — not task difficulty — are the primary blockers.
# MAGIC * **Action:** Pre-install language-specific toolchains (`rustc`, `javac`, `g++`, `go`) in all evaluation images. Pin versions to match each repo's CI.
# MAGIC * **Action:** For the 10 highest-volume non-Python repos, add a per-repo `setup.sh` that the agent runs before any edits.
# MAGIC * **Metric to track:** Non-Python resolve rate should converge toward 35–40%.
# MAGIC
# MAGIC ### 5. Instrument sweagent error reporting and standardize telemetry
# MAGIC sweagent's 0.06% tool-error rate is an instrumentation artifact — it rarely sets `STATUS_CODE_ERROR`, making cross-framework comparison impossible.
# MAGIC * **Action:** Normalize error reporting: any non-zero exit code must set `STATUS_CODE_ERROR` with the exit code and stderr in `status.message`, across all frameworks.
# MAGIC * **Action:** Add structured attributes for error class (`env_missing_tool`, `timeout`, `edit_mismatch`, etc.) so dashboards and alerts can be built without regex parsing.
# MAGIC * **Metric to track:** Error-type coverage should reach >95% (currently ~68% fall into a generic "other non-zero exit" bucket).
# COMMAND ----------

# DBTITLE 1,Persist summary tables for the Genie error agent
# MAGIC %sql
# MAGIC -- Materializes the temp views above as Unity Catalog tables in :output_schema.
# MAGIC -- These are the tables 05_create_genie_error_agent.py points the Genie space at.
# MAGIC CREATE OR REPLACE TABLE IDENTIFIER(:output_schema || '.classified_errors')
# MAGIC COMMENT 'One row per classified error span (heuristic taxonomy from 03_Usage and Error Analysis).'
# MAGIC AS SELECT trace_id, tool, error_type, error_message, current_timestamp() AS analysis_timestamp
# MAGIC FROM classified_errors;
# MAGIC
# MAGIC CREATE OR REPLACE TABLE IDENTIFIER(:output_schema || '.trajectory_summary')
# MAGIC COMMENT 'One row per trajectory with outcome, effort and per-error-type counts. resolved: 1 = resolved, 0 = failed, -1 = ungraded.'
# MAGIC AS
# MAGIC WITH e AS (
# MAGIC   SELECT trace_id,
# MAGIC     COUNT(*) AS total_errors,
# MAGIC     COUNT_IF(error_type = 'editor: invalid edit (old_str mismatch etc.)') AS errors_edit_mismatch,
# MAGIC     COUNT_IF(error_type = 'timeout / interrupted command')               AS errors_timeout,
# MAGIC     COUNT_IF(error_type = 'env: missing dependency')                     AS errors_missing_dep,
# MAGIC     COUNT_IF(error_type = 'env: command not found (toolchain missing)')  AS errors_missing_toolchain,
# MAGIC     COUNT_IF(error_type = 'agent: wrong path / file not found')          AS errors_wrong_path,
# MAGIC     COUNT_IF(error_type = 'agent: wrong go test invocation')             AS errors_go_test,
# MAGIC     COUNT_IF(error_type = 'agent: tool protocol misuse')                 AS errors_protocol_misuse,
# MAGIC     COUNT_IF(error_type = 'agent: shell syntax error')                   AS errors_syntax,
# MAGIC     COUNT_IF(error_type = 'test failure')                                AS errors_test_failure,
# MAGIC     COUNT_IF(error_type = 'silent non-zero exit (e.g. grep no match)')   AS errors_silent_exit,
# MAGIC     COUNT_IF(error_type = 'other non-zero exit')                         AS errors_other
# MAGIC   FROM classified_errors
# MAGIC   GROUP BY trace_id
# MAGIC )
# MAGIC SELECT t.trace_id, t.framework, t.model,
# MAGIC        COALESCE(t.resolved, -1) AS resolved, t.outcome, t.ungraded,
# MAGIC        t.repo, t.language, t.instance_id, t.category, t.source_dataset,
# MAGIC        t.num_tool_calls, t.num_llm_calls, t.input_tokens, t.output_tokens,
# MAGIC        t.patch_files, t.patch_lines, t.ref_patch_lines,
# MAGIC        COALESCE(e.total_errors, 0)             AS total_errors,
# MAGIC        COALESCE(e.errors_missing_toolchain + e.errors_missing_dep, 0) AS env_errors,
# MAGIC        COALESCE(e.errors_wrong_path + e.errors_protocol_misuse + e.errors_syntax + e.errors_go_test, 0) AS agent_errors,
# MAGIC        COALESCE(e.errors_edit_mismatch, 0)     AS errors_edit_mismatch,
# MAGIC        COALESCE(e.errors_timeout, 0)           AS errors_timeout,
# MAGIC        COALESCE(e.errors_missing_dep, 0)       AS errors_missing_dep,
# MAGIC        COALESCE(e.errors_missing_toolchain, 0) AS errors_missing_toolchain,
# MAGIC        COALESCE(e.errors_wrong_path, 0)        AS errors_wrong_path,
# MAGIC        COALESCE(e.errors_go_test, 0)           AS errors_go_test,
# MAGIC        COALESCE(e.errors_protocol_misuse, 0)   AS errors_protocol_misuse,
# MAGIC        COALESCE(e.errors_syntax, 0)            AS errors_syntax,
# MAGIC        COALESCE(e.errors_test_failure, 0)      AS errors_test_failure,
# MAGIC        COALESCE(e.errors_silent_exit, 0)       AS errors_silent_exit,
# MAGIC        COALESCE(e.errors_other, 0)             AS errors_other,
# MAGIC        current_timestamp()                     AS analysis_timestamp
# MAGIC FROM trajectories t LEFT JOIN e USING (trace_id);
# MAGIC
# MAGIC CREATE OR REPLACE TABLE IDENTIFIER(:output_schema || '.error_taxonomy_summary')
# MAGIC COMMENT 'Pre-aggregated error distribution by error_type.'
# MAGIC AS SELECT error_type, COUNT(*) AS errors, COUNT(DISTINCT trace_id) AS affected_trajectories,
# MAGIC        ROUND(100 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS pct_of_errors,
# MAGIC        current_timestamp() AS analysis_timestamp
# MAGIC FROM classified_errors
# MAGIC GROUP BY error_type;

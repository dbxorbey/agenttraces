-- Databricks notebook source
-- MAGIC %md
-- MAGIC # Open-SWE-Traces → OTel trace table
-- MAGIC
-- MAGIC Converts the raw Hugging Face dataset [`nvidia/Open-SWE-Traces`](https://huggingface.co/datasets/nvidia/Open-SWE-Traces)
-- MAGIC (`nvidia_open_swe_traces.traces`, one row per agent trajectory) into `nvidia_open_swe_traces.otel_spans`.
-- MAGIC That table uses the OpenTelemetry span layout of MLflow traces in Unity Catalog (`otel.schemaVersion = v2`),
-- MAGIC which is nearly identical to the Unity Gateway unified trace table, so Genie Code can analyze it like a native trace table.
-- MAGIC
-- MAGIC **Span tree per trajectory (= one trace)**
-- MAGIC - `<agent>.solve_issue`, an `AGENT` root span: issue → final patch, plus outcome (`swe.resolved`) and task metadata
-- MAGIC - `chat.completions`, a `CHAT_MODEL` span per assistant message: inputs = messages added since the previous call
-- MAGIC - `<tool name>`, a `TOOL` span per tool call: arguments → result; `STATUS_CODE_ERROR` on non-zero exit code or `ERROR:` output
-- MAGIC
-- MAGIC **Caveats**
-- MAGIC - Timestamps, durations and token counts are **synthetic** (derived from text length), so use them for relative comparisons only.
-- MAGIC - Text fields are capped at 100k chars to stay below the 128 MiB VARIANT limit; affected spans have `swe.truncated = true`.
-- MAGIC - SWE-agent outputs carry no exit codes, so its tool errors are under-detected.
-- MAGIC
-- MAGIC **Run**: set `sample_modulus` to `1000` for a ~0.1% sample (~350 traces, seconds) or `1` for all 364k trajectories
-- MAGIC (~56M spans, ~9 min on a serverless SQL warehouse). The load uses `INSERT OVERWRITE`, so each run replaces the table.

-- COMMAND ----------

-- DBTITLE 1,Configuration
-- ===================================================================
-- CONFIGURATION — adjust these to your catalog, schema, and tables
-- ===================================================================
CREATE WIDGET TEXT traces_table DEFAULT 'main.nvidia_open_swe_traces.traces';
CREATE WIDGET TEXT otel_spans_table DEFAULT 'main.nvidia_open_swe_traces.otel_spans';
CREATE WIDGET TEXT sample_modulus DEFAULT '1000';

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## 1. Create the target table

-- COMMAND ----------

-- Target table in the OTel span layout used by Unity Gateway's unified trace table and by
-- MLflow traces stored in Unity Catalog (otel.schemaVersion = v2).
CREATE TABLE IF NOT EXISTS IDENTIFIER(:otel_spans_table) (
  record_id STRING,
  time TIMESTAMP,
  date DATE,
  service_name STRING,
  trace_id STRING,
  span_id STRING,
  trace_state STRING,
  parent_span_id STRING,
  flags INT,
  name STRING,
  kind STRING,
  start_time_unix_nano BIGINT,
  end_time_unix_nano BIGINT,
  attributes VARIANT,
  dropped_attributes_count INT,
  events ARRAY<STRUCT<time_unix_nano: BIGINT, name: STRING, attributes: VARIANT, dropped_attributes_count: INT>>,
  dropped_events_count INT,
  links ARRAY<STRUCT<trace_id: STRING, span_id: STRING, trace_state: STRING, attributes: VARIANT, dropped_attributes_count: INT, flags: INT>>,
  dropped_links_count INT,
  status STRUCT<message: STRING, code: STRING>,
  resource STRUCT<attributes: VARIANT, dropped_attributes_count: INT>,
  resource_schema_url STRING,
  instrumentation_scope STRUCT<name: STRING, version: STRING, attributes: VARIANT, dropped_attributes_count: INT>,
  span_schema_url STRING
)
CLUSTER BY (time, service_name, trace_id)
COMMENT 'nvidia/Open-SWE-Traces as OTel spans: one trace per trajectory, AGENT root span, one LLM span per assistant turn, one TOOL span per tool call. Timing and token counts are synthetic (derived from text length).'
TBLPROPERTIES ('otel.schemaVersion' = 'v2', 'delta.enableVariantShredding' = 'true');

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## 2. Explode trajectories into spans

-- COMMAND ----------

-- Span tree per trajectory (= one trace):
--   AGENT root span      task issue -> final model patch, outcome (`swe.resolved`) and task metadata
--   ├─ CHAT_MODEL span   one per assistant message; inputs = messages added since the previous call
--   └─ TOOL span         one per tool call; result = the matching tool message
-- Durations are synthetic: LLM = 400 ms + ~60 output tokens/s, tool = 200 ms + 1 ms per 20 output chars.
INSERT OVERWRITE IDENTIFIER(:otel_spans_table)
WITH src AS (
  SELECT
    *,
    split(_source_path, '/')[1] AS agent,
    split(_source_path, '/')[2] AS model_slug,
    split(_source_path, '/')[3] AS task_source,
    md5(trajectory_id) AS trace_hex,
    -- Spread trace start times deterministically over September 2026.
    (unix_micros(TIMESTAMP '2026-09-01 00:00:00') + pmod(xxhash64(trajectory_id), 30 * 86400) * 1000000) * 1000 AS trace_start_ns
  FROM IDENTIFIER(:traces_table)
  WHERE pmod(xxhash64(trajectory_id), CAST(:sample_modulus AS INT)) = 0
),

msgs AS (
  SELECT
    trajectory_id, trace_hex, trace_start_ns, agent, metadata.teacher_model.name AS model,
    pos,
    -- A few trajectories contain multi-MB tool outputs / file writes that overflow the 128 MiB VARIANT
    -- limit once nested into span attributes, so every text field is capped at 100k chars.
    -- Lengths, timing and token estimates below still use the untruncated message.
    named_struct(
      'role', raw_m.role,
      'content', left(raw_m.content, 100000),
      'reasoning_content', left(raw_m.reasoning_content, 100000),
      'tool_calls', transform(raw_m.tool_calls, tc -> named_struct(
        'function', named_struct('arguments', left(tc.function.arguments, 100000), 'name', tc.function.name),
        'id', tc.id, 'type', tc.type))
    ) AS m,
    coalesce(greatest(length(raw_m.content), length(raw_m.reasoning_content),
             aggregate(raw_m.tool_calls, 0, (acc, tc) -> greatest(acc, length(tc.function.arguments)))) > 100000, false) AS truncated,
    length(coalesce(raw_m.content, '')) AS content_chars,
    length(coalesce(raw_m.reasoning_content, '')) AS reasoning_chars,
    length(coalesce(raw_m.content, '')) + length(coalesce(raw_m.reasoning_content, ''))
      + length(coalesce(to_json(raw_m.tool_calls), '')) AS chars,
    CASE raw_m.role
      WHEN 'assistant' THEN 400 + CAST((length(coalesce(raw_m.content, '')) + length(coalesce(raw_m.reasoning_content, ''))
                                        + length(coalesce(to_json(raw_m.tool_calls), ''))) * 1000 / (4 * 60) AS BIGINT)
      WHEN 'tool' THEN 200 + CAST(length(coalesce(raw_m.content, '')) / 20 AS BIGINT)
      ELSE 0
    END AS dur_ms
  FROM src LATERAL VIEW posexplode(messages) AS pos, raw_m
),

timeline AS (
  SELECT
    *,
    coalesce(sum(dur_ms) OVER w_before, 0) AS offset_ms,
    coalesce(sum(chars) OVER w_before, 0) AS ctx_chars,
    -- Assistant message a tool result belongs to, and the next assistant message an input belongs to.
    max(CASE WHEN m.role = 'assistant' THEN pos END) OVER w_before AS prev_asst_pos,
    min(CASE WHEN m.role = 'assistant' THEN pos END) OVER w_after AS next_asst_pos,
    last_value(CASE WHEN m.role = 'assistant' THEN m.tool_calls END, true) OVER w_before AS prev_tool_calls,
    count_if(m.role = 'assistant') OVER w_before AS turn_index
  FROM msgs
  WINDOW
    w_before AS (PARTITION BY trajectory_id ORDER BY pos ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
    w_after AS (PARTITION BY trajectory_id ORDER BY pos ROWS BETWEEN 1 FOLLOWING AND UNBOUNDED FOLLOWING)
),

-- Messages that are new input to each LLM call (system/user prompt for the first call, tool results afterwards).
llm_inputs AS (
  SELECT
    trajectory_id, next_asst_pos AS pos,
    transform(
      array_sort(collect_list(struct(pos, m))),
      x -> named_struct('role', x.m.role, 'content', x.m.content)
    ) AS new_messages
  FROM timeline
  WHERE m.role <> 'assistant' AND next_asst_pos IS NOT NULL
  GROUP BY trajectory_id, next_asst_pos
),

llm_spans AS (
  SELECT
    t.trace_hex AS trace_id,
    substr(md5(concat(t.trajectory_id, ':llm:', t.pos)), 1, 16) AS span_id,
    substr(md5(concat(t.trajectory_id, ':root')), 1, 16) AS parent_span_id,
    'chat.completions' AS name,
    'SPAN_KIND_CLIENT' AS kind,
    t.agent,
    t.trace_start_ns + t.offset_ms * 1000000 AS start_ns,
    t.trace_start_ns + (t.offset_ms + t.dur_ms) * 1000000 AS end_ns,
    to_json(named_struct(
      'mlflow.spanType', 'CHAT_MODEL',
      'mlflow.spanInputs', to_json(named_struct('messages', i.new_messages)),
      'mlflow.spanOutputs', to_json(named_struct(
        'role', t.m.role, 'content', t.m.content, 'reasoning_content', t.m.reasoning_content, 'tool_calls', t.m.tool_calls)),
      'gen_ai.operation.name', 'chat',
      'gen_ai.request.model', t.model,
      'gen_ai.usage.input_tokens', CAST(t.ctx_chars / 4 AS BIGINT),
      'gen_ai.usage.output_tokens', CAST(t.chars / 4 AS BIGINT),
      'gen_ai.response.finish_reasons', CASE WHEN size(t.m.tool_calls) > 0 THEN 'tool_calls' ELSE 'stop' END,
      'databricks.latency_ms', t.dur_ms,
      'swe.trajectory_id', t.trajectory_id,
      'swe.turn_index', t.turn_index,
      'swe.num_tool_calls', coalesce(size(t.m.tool_calls), 0),
      'swe.content_chars', t.content_chars,
      'swe.reasoning_chars', t.reasoning_chars,
      'swe.truncated', t.truncated
    )) AS attrs_json,
    named_struct('message', CAST(NULL AS STRING), 'code', 'STATUS_CODE_OK') AS status
  FROM timeline t
  LEFT JOIN llm_inputs i ON i.trajectory_id = t.trajectory_id AND i.pos = t.pos
  WHERE t.m.role = 'assistant'
),

tool_spans AS (
  SELECT
    trace_hex AS trace_id,
    substr(md5(concat(trajectory_id, ':tool:', pos)), 1, 16) AS span_id,
    substr(md5(concat(trajectory_id, ':root')), 1, 16) AS parent_span_id,
    coalesce(tc.function.name, 'unknown_tool') AS name,
    'SPAN_KIND_INTERNAL' AS kind,
    agent,
    trace_start_ns + offset_ms * 1000000 AS start_ns,
    trace_start_ns + (offset_ms + dur_ms) * 1000000 AS end_ns,
    to_json(named_struct(
      'mlflow.spanType', 'TOOL',
      'mlflow.spanInputs', tc.function.arguments,
      'mlflow.spanOutputs', to_json(CAST(m.content AS VARIANT)),
      'gen_ai.operation.name', 'execute_tool',
      'gen_ai.tool.name', tc.function.name,
      'gen_ai.tool.call.id', tc.id,
      'swe.trajectory_id', trajectory_id,
      'swe.turn_index', turn_index - 1,
      'swe.tool.command', coalesce(get_json_object(tc.function.arguments, '$.command'), ''),
      'swe.tool.exit_code', exit_code,
      'swe.tool.output_chars', content_chars,
      'swe.truncated', truncated
    )) AS attrs_json,
    named_struct(
      'message', CASE WHEN is_error THEN left(m.content, 500) END,
      'code', CASE WHEN is_error THEN 'STATUS_CODE_ERROR' ELSE 'STATUS_CODE_OK' END
    ) AS status
  FROM (
    SELECT
      *,
      coalesce(exit_code, 0) <> 0 OR m.content RLIKE '^(OBSERVATION:\\s*)?ERROR' AS is_error
    FROM (
      SELECT
        *,
        -- The k-th tool message after an assistant message answers its k-th tool call.
        try_element_at(prev_tool_calls, CAST(pos - prev_asst_pos AS INT)) AS tc,
        -- mini-swe-agent reports {"returncode": N}, OpenHands "[Command finished with exit code N]".
        CAST(coalesce(
          nullif(regexp_extract(m.content, '"returncode": (-?[0-9]+)', 1), ''),
          nullif(regexp_extract(m.content, 'Command finished with exit code (-?[0-9]+)', 1), '')
        ) AS INT) AS exit_code
      FROM timeline
      WHERE m.role = 'tool' AND prev_asst_pos IS NOT NULL
    )
  )
),

trace_stats AS (
  SELECT
    trajectory_id,
    max(offset_ms + dur_ms) AS total_ms,
    count_if(m.role = 'assistant') AS num_llm_calls,
    count_if(m.role = 'tool') AS num_tool_calls,
    CAST(sum(CASE WHEN m.role = 'assistant' THEN ctx_chars END) / 4 AS BIGINT) AS total_input_tokens,
    CAST(sum(CASE WHEN m.role = 'assistant' THEN chars END) / 4 AS BIGINT) AS total_output_tokens
  FROM timeline
  GROUP BY trajectory_id
),

root_spans AS (
  SELECT
    s.trace_hex AS trace_id,
    substr(md5(concat(s.trajectory_id, ':root')), 1, 16) AS span_id,
    CAST(NULL AS STRING) AS parent_span_id,
    concat(s.agent, '.solve_issue') AS name,
    'SPAN_KIND_SERVER' AS kind,
    s.agent,
    s.trace_start_ns AS start_ns,
    s.trace_start_ns + st.total_ms * 1000000 AS end_ns,
    to_json(named_struct(
      'mlflow.spanType', 'AGENT',
      'mlflow.spanInputs', to_json(named_struct('issue', left(s.messages[1].content, 100000))),
      'mlflow.spanOutputs', to_json(named_struct('patch', left(s.metadata.model_patch.patch, 100000))),
      'gen_ai.request.model', s.metadata.teacher_model.name,
      'gen_ai.usage.input_tokens', st.total_input_tokens,
      'gen_ai.usage.output_tokens', st.total_output_tokens,
      'swe.trajectory_id', s.trajectory_id,
      'swe.instance_id', s.instance_id,
      'swe.repo', s.repo,
      'swe.language', s.language,
      'swe.license', s.license,
      'swe.category', s.metadata.category,
      'swe.task_source', s.task_source,
      'swe.source_dataset', s.hf_dataset_name,
      'swe.agent_framework', s.agent,
      'swe.model_slug', s.model_slug,
      'swe.enable_thinking', s.metadata.teacher_model.enable_thinking,
      'swe.resolved', s.resolved,
      'swe.num_llm_calls', st.num_llm_calls,
      'swe.num_tool_calls', st.num_tool_calls,
      'swe.num_messages', size(s.messages),
      'swe.model_patch.num_modified_files', s.metadata.model_patch.num_modified_files,
      'swe.model_patch.num_modified_lines', s.metadata.model_patch.num_modified_lines,
      'swe.reference_patch.num_modified_files', s.metadata.reference_patch.num_modified_files,
      'swe.reference_patch.num_modified_lines', s.metadata.reference_patch.num_modified_lines,
      'swe.reference_patch', left(s.metadata.reference_patch.patch, 100000),
      'swe.model_patch.chars', length(s.metadata.model_patch.patch),
      'swe.source_path', s._source_path
    )) AS attrs_json,
    named_struct('message', CAST(NULL AS STRING), 'code', 'STATUS_CODE_OK') AS status
  FROM src s
  JOIN trace_stats st ON st.trajectory_id = s.trajectory_id
),

all_spans AS (
  SELECT * FROM root_spans
  UNION ALL SELECT * FROM llm_spans
  UNION ALL SELECT * FROM tool_spans
)

SELECT
  concat(trace_id, '-', span_id) AS record_id,
  timestamp_micros(CAST(start_ns / 1000 AS BIGINT)) AS time,
  to_date(timestamp_micros(CAST(start_ns / 1000 AS BIGINT))) AS date,
  agent AS service_name,
  trace_id,
  span_id,
  CAST(NULL AS STRING) AS trace_state,
  parent_span_id,
  0 AS flags,
  name,
  kind,
  start_ns AS start_time_unix_nano,
  end_ns AS end_time_unix_nano,
  parse_json(attrs_json) AS attributes,
  0 AS dropped_attributes_count,
  CAST(NULL AS ARRAY<STRUCT<time_unix_nano: BIGINT, name: STRING, attributes: VARIANT, dropped_attributes_count: INT>>) AS events,
  0 AS dropped_events_count,
  CAST(NULL AS ARRAY<STRUCT<trace_id: STRING, span_id: STRING, trace_state: STRING, attributes: VARIANT, dropped_attributes_count: INT, flags: INT>>) AS links,
  0 AS dropped_links_count,
  status,
  named_struct('attributes', parse_json(to_json(named_struct('service.name', agent))), 'dropped_attributes_count', 0) AS resource,
  CAST(NULL AS STRING) AS resource_schema_url,
  named_struct('name', 'open_swe_traces_backfill', 'version', '1', 'attributes', CAST(NULL AS VARIANT), 'dropped_attributes_count', 0) AS instrumentation_scope,
  CAST(NULL AS STRING) AS span_schema_url
FROM all_spans;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## 3. Validate

-- COMMAND ----------

-- Span counts per type and framework. Every trace should have exactly one AGENT span and no `unknown_tool` spans.
SELECT
  attributes:['mlflow.spanType']::string AS span_type,
  service_name,
  count(*) AS spans,
  count(DISTINCT trace_id) AS traces,
  count_if(status.code = 'STATUS_CODE_ERROR') AS error_spans,
  count_if(name = 'unknown_tool') AS orphan_tool_spans,
  count_if(attributes:['swe.truncated']::boolean) AS truncated_spans
FROM IDENTIFIER(:otel_spans_table)
GROUP BY ALL
ORDER BY ALL;

-- COMMAND ----------

-- Walk through one trace in time order.
SELECT
  name,
  attributes:['swe.turn_index']::int AS turn,
  round((end_time_unix_nano - start_time_unix_nano) / 1e6) AS duration_ms,
  status.code,
  left(attributes:['mlflow.spanInputs']::string, 200) AS inputs,
  left(attributes:['mlflow.spanOutputs']::string, 200) AS outputs
FROM IDENTIFIER(:otel_spans_table)
WHERE trace_id = (SELECT min(trace_id) FROM IDENTIFIER(:otel_spans_table) WHERE service_name = 'openhands')
ORDER BY start_time_unix_nano, parent_span_id NULLS FIRST
LIMIT 20;
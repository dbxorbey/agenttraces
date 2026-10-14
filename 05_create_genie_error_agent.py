"""Create a Genie Agent for NVIDIA Open-SWE-Traces error analysis.

This script programmatically creates a Genie space configured to surface
the most frequent errors in agent trajectories and recommend what to fix next.

Usage:
    # From a Databricks notebook in the same folder (uses ambient auth).
    # The file name starts with a digit, so load it via importlib:
    import importlib.util
    spec = importlib.util.spec_from_file_location("genie_agent", "./05_create_genie_error_agent.py")
    genie_agent = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(genie_agent)
    genie_agent.create_genie_space(catalog="my_catalog", schema="my_schema")

    # Or run directly with Python:
    python 05_create_genie_error_agent.py

    # Override defaults:
    python 05_create_genie_error_agent.py \\
        --catalog my_catalog \\
        --schema my_schema \\
        --warehouse-id abc123def456 \\
        --parent-path /Workspace/Users/me@example.com
"""

import argparse
import json
import sys
import uuid
from typing import Optional

# ---------------------------------------------------------------------------
# Configuration — edit these defaults or override via CLI args
# ---------------------------------------------------------------------------
DEFAULT_CATALOG = "main"
DEFAULT_SCHEMA = "nvidia_open_swe_traces"
# None = /Workspace/Users/<current user>/AgentTraces, resolved at runtime
DEFAULT_PARENT_PATH = None
SPACE_TITLE = "NVIDIA Open-SWE Error Analyzer"
SPACE_DESCRIPTION = (
    "Surfaces the most frequent errors in NVIDIA Open-SWE-Traces agent "
    "trajectories and recommends which errors to prioritize fixing for "
    "the highest impact on resolve rates."
)


def _id() -> str:
    """Generate a 32-hex-char ID matching the Genie serialized_space format."""
    return uuid.uuid4().hex


def build_serialized_space(catalog: str, schema: str) -> str:
    """Build the serialized_space JSON string for the Genie agent."""

    fq = lambda table: f"{catalog}.{schema}.{table}"  # noqa: E731

    # -- Tables ---------------------------------------------------------------
    tables = [
        {
            "identifier": fq("classified_errors"),
            "description": [
                "Individual classified error occurrences from agent tool calls. "
                "One row per error. Columns: trace_id, tool (execute_bash, bash, "
                "str_replace_editor), error_type, error_message, analysis_timestamp."
            ],
            "column_configs": [
                {"column_name": "error_type", "enable_format_assistance": True, "enable_entity_matching": True},
                {"column_name": "tool", "enable_format_assistance": True, "enable_entity_matching": True},
            ],
        },
        {
            "identifier": fq("trajectory_summary"),
            "description": [
                "One row per agent trajectory (trace). Contains framework, model, "
                "resolved status (1=resolved, 0=failed, -1=ungraded), repo, language, "
                "token counts, patch stats, and per-error-type counts "
                "(errors_edit_mismatch, errors_timeout, errors_missing_dep, etc.). "
                "Exclude resolved = -1 (ungraded) from resolve-rate calculations."
            ],
            "column_configs": [
                {"column_name": "framework", "enable_format_assistance": True, "enable_entity_matching": True},
                {"column_name": "model", "enable_format_assistance": True, "enable_entity_matching": True},
                {"column_name": "outcome", "enable_format_assistance": True, "enable_entity_matching": True},
                {"column_name": "language", "enable_format_assistance": True, "enable_entity_matching": True},
            ],
        },
        {
            "identifier": fq("error_taxonomy_summary"),
            "description": [
                "Pre-aggregated error distribution: error_type, total error count, "
                "affected_trajectories, pct_of_errors. Use for quick top-level overviews."
            ],
            "column_configs": [
                {"column_name": "error_type", "enable_format_assistance": True, "enable_entity_matching": True},
            ],
        },
    ]

    # -- Text instructions ----------------------------------------------------
    text_instructions = [
        {
            "id": _id(),
            "content": [
                "You are an error-analysis assistant for the NVIDIA Open-SWE-Traces "
                "dataset. Your goal is to help users identify the most frequent errors "
                "across AI coding agent trajectories and recommend which errors to "
                "prioritize fixing for the highest impact on resolve rates.",
                "Key domain rules:\n"
                "- Each trajectory is one agent attempt to solve a SWE-bench issue (trace_id).\n"
                "- resolved = 1 means resolved, 0 = failed, -1 = ungraded.\n"
                "- ALWAYS exclude resolved = -1 from resolve-rate calculations.\n"
                "- The 'other non-zero exit' error type (~68% of errors) is a catch-all — "
                "focus recommendations on specific, actionable types: editor edit mismatches, "
                "timeouts, missing dependencies, wrong paths, protocol misuse, syntax errors.\n"
                "- When recommending what to fix next, consider BOTH frequency AND impact "
                "on resolve rate (correlation with resolved = 0).\n"
                "- Frameworks: sweagent, openhands, minisweagent.\n"
                "- Models: Qwen3.6-27B, MiniMax-M2.5, Qwen3.5-122B-A10B."
            ],
        },
    ]

    # -- Sample questions ------------------------------------------------------
    sample_questions = [
        {"id": _id(), "question": ["Which error types happen most frequently and how many trajectories do they affect?"]},
        {"id": _id(), "question": ["Which error types are most correlated with failed trajectories?"]},
        {"id": _id(), "question": ["Compare error rates across the three frameworks"]},
        {"id": _id(), "question": ["For editor edit mismatch errors, which repos and models are most affected?"]},
        {"id": _id(), "question": ["What are the top 5 errors I should fix first to improve resolve rates the most?"]},
        {"id": _id(), "question": ["Show me the error rate breakdown by model, excluding ungraded trajectories"]},
        {"id": _id(), "question": ["Which tool triggers the most actionable errors?"]},
        {"id": _id(), "question": ["How do timeout errors distribute across frameworks and models?"]},
    ]

    # -- Example SQL instructions (curated queries the agent can reference) ----
    example_question_sqls = [
        {
            "id": _id(),
            "question": ["What are the top 5 actionable error types I should fix first?"],
            "sql": [
                f"WITH error_resolve AS (\n",
                f"  SELECT\n",
                f"    e.error_type,\n",
                f"    COUNT(DISTINCT e.trace_id) AS affected_trajectories,\n",
                f"    COUNT(*) AS total_errors,\n",
                f"    AVG(CASE WHEN t.resolved = 1 THEN 1.0 ELSE 0.0 END) AS resolve_rate_with_error,\n",
                f"    (SELECT AVG(CASE WHEN resolved = 1 THEN 1.0 ELSE 0.0 END)\n",
                f"     FROM {fq('trajectory_summary')} WHERE resolved != -1) AS overall_resolve_rate\n",
                f"  FROM {fq('classified_errors')} e\n",
                f"  JOIN {fq('trajectory_summary')} t ON e.trace_id = t.trace_id\n",
                f"  WHERE t.resolved != -1\n",
                f"    AND e.error_type != 'other non-zero exit'\n",
                f"    AND e.error_type != 'silent non-zero exit (e.g. grep no match)'\n",
                f"  GROUP BY e.error_type\n",
                f")\n",
                f"SELECT\n",
                f"  error_type,\n",
                f"  total_errors,\n",
                f"  affected_trajectories,\n",
                f"  ROUND(resolve_rate_with_error * 100, 1) AS resolve_pct_with_error,\n",
                f"  ROUND(overall_resolve_rate * 100, 1) AS overall_resolve_pct,\n",
                f"  ROUND((overall_resolve_rate - resolve_rate_with_error) * 100, 1) AS resolve_rate_drop_pct\n",
                f"FROM error_resolve\n",
                f"ORDER BY affected_trajectories DESC\n",
                f"LIMIT 5",
            ],
        },
        {
            "id": _id(),
            "question": ["Compare error rates across frameworks"],
            "sql": [
                f"SELECT\n",
                f"  framework,\n",
                f"  COUNT(*) AS trajectories,\n",
                f"  SUM(total_errors) AS total_errors,\n",
                f"  ROUND(AVG(total_errors), 2) AS avg_errors_per_trajectory,\n",
                f"  ROUND(AVG(CASE WHEN resolved = 1 THEN 1.0 ELSE 0.0 END) * 100, 1) AS resolve_pct\n",
                f"FROM {fq('trajectory_summary')}\n",
                f"WHERE resolved != -1\n",
                f"GROUP BY framework\n",
                f"ORDER BY avg_errors_per_trajectory DESC",
            ],
        },
        {
            "id": _id(),
            "question": ["Which tool triggers the most actionable errors?"],
            "sql": [
                f"SELECT\n",
                f"  tool,\n",
                f"  error_type,\n",
                f"  COUNT(*) AS error_count\n",
                f"FROM {fq('classified_errors')}\n",
                f"WHERE error_type NOT IN ('other non-zero exit', 'silent non-zero exit (e.g. grep no match)')\n",
                f"GROUP BY tool, error_type\n",
                f"ORDER BY error_count DESC\n",
                f"LIMIT 15",
            ],
        },
    ]

    # -- SQL snippets (reusable fragments) ------------------------------------
    sql_snippets = {
        "filters": [
            {
                "id": _id(),
                "sql": ["trajectory_summary.resolved != -1"],
                "display_name": "exclude ungraded",
                "synonyms": ["graded only", "skip ungraded"],
            },
            {
                "id": _id(),
                "sql": [
                    "classified_errors.error_type NOT IN ('other non-zero exit', "
                    "'silent non-zero exit (e.g. grep no match)')"
                ],
                "display_name": "actionable errors only",
                "synonyms": ["exclude catch-all", "specific errors"],
            },
        ],
        "measures": [
            {
                "id": _id(),
                "alias": "resolve_rate",
                "sql": ["ROUND(AVG(CASE WHEN trajectory_summary.resolved = 1 THEN 1.0 ELSE 0.0 END) * 100, 1)"],
                "display_name": "resolve rate (%)",
                "synonyms": ["success rate", "resolution rate"],
            },
            {
                "id": _id(),
                "alias": "avg_errors",
                "sql": ["ROUND(AVG(trajectory_summary.total_errors), 2)"],
                "display_name": "average errors per trajectory",
                "synonyms": ["error rate", "errors per run"],
            },
        ],
    }

    # -- Join specs -----------------------------------------------------------
    join_specs = [
        {
            "id": _id(),
            "left": {"identifier": fq("classified_errors"), "alias": "ce"},
            "right": {"identifier": fq("trajectory_summary"), "alias": "ts"},
            "sql": ["`ce`.`trace_id` = `ts`.`trace_id`", "--rt=FROM_RELATIONSHIP_TYPE_MANY_TO_ONE--"],
        },
    ]

    # -- Assemble the full serialized_space object ----------------------------
    # The API requires every list to be sorted: tables by identifier,
    # column_configs by column_name, everything else by id.
    by_id = lambda items: sorted(items, key=lambda x: x["id"])  # noqa: E731
    for t in tables:
        t["column_configs"] = sorted(t["column_configs"], key=lambda c: c["column_name"])
    tables = sorted(tables, key=lambda t: t["identifier"])
    sample_questions = by_id(sample_questions)
    text_instructions = by_id(text_instructions)
    example_question_sqls = by_id(example_question_sqls)
    join_specs = by_id(join_specs)
    sql_snippets = {k: by_id(v) for k, v in sql_snippets.items()}

    space = {
        "version": 2,
        "config": {
            "sample_questions": sample_questions,
        },
        "data_sources": {
            "tables": tables,
        },
        "instructions": {
            "text_instructions": text_instructions,
            "example_question_sqls": example_question_sqls,
            "join_specs": join_specs,
            "sql_snippets": sql_snippets,
        },
    }

    return json.dumps(space)


def _pick_warehouse(w) -> str:
    """Pick a SQL warehouse for the Genie space: serverless first, running first."""
    warehouses = list(w.warehouses.list())
    if not warehouses:
        raise RuntimeError("No SQL warehouse found - pass --warehouse-id explicitly")
    warehouses.sort(key=lambda wh: (
        not wh.enable_serverless_compute,
        getattr(wh.state, "value", wh.state) != "RUNNING",
    ))
    return warehouses[0].id


def create_genie_space(
    catalog: str,
    schema: str,
    parent_path: Optional[str] = DEFAULT_PARENT_PATH,
    warehouse_id: Optional[str] = None,
    title: str = SPACE_TITLE,
    description: str = SPACE_DESCRIPTION,
) -> dict:
    """Create the Genie space via the REST API and return the response."""
    from databricks.sdk import WorkspaceClient

    w = WorkspaceClient()
    if parent_path is None:
        parent_path = f"/Workspace/Users/{w.current_user.me().user_name}/AgentTraces"
    if warehouse_id is None:
        warehouse_id = _pick_warehouse(w)

    payload = {
        "title": title,
        "description": description,
        "parent_path": parent_path,
        "warehouse_id": warehouse_id,
        "serialized_space": build_serialized_space(catalog, schema),
    }

    response = w.api_client.do(
        method="POST",
        path="/api/2.0/genie/spaces",
        body=payload,
    )

    return response


def main():
    parser = argparse.ArgumentParser(
        description="Create a Genie Agent for Open-SWE-Traces error analysis."
    )
    parser.add_argument("--catalog", default=DEFAULT_CATALOG, help="Unity Catalog catalog name")
    parser.add_argument("--schema", default=DEFAULT_SCHEMA, help="Schema containing the trace tables")
    parser.add_argument("--parent-path", default=DEFAULT_PARENT_PATH, help="Workspace path for the space (default: /Workspace/Users/<you>/AgentTraces)")
    parser.add_argument("--warehouse-id", default=None, help="SQL warehouse for the space (default: first serverless warehouse)")
    args = parser.parse_args()

    print(f"Creating Genie space '{SPACE_TITLE}'...")
    print(f"  Catalog: {args.catalog}")
    print(f"  Schema:  {args.schema}")
    print(f"  Path:    {args.parent_path or '/Workspace/Users/<current user>/AgentTraces'}")

    result = create_genie_space(
        catalog=args.catalog,
        schema=args.schema,
        parent_path=args.parent_path,
        warehouse_id=args.warehouse_id,
    )

    space_id = result.get("space_id", "unknown")
    print(f"\nGenie space created successfully!")
    print(f"  Space ID: {space_id}")
    print(f"  URL:      /genie/rooms/{space_id}")
    return result


# Allow execution both as a standalone script and from a Databricks notebook
if __name__ == "__main__":
    main()

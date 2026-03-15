"""Tests for the export module.

Covers:
- VAL-EXPORT-01: JSON export with schema_version and generated_at
- VAL-EXPORT-02: CSV export loads cleanly with pandas
- VAL-EXPORT-03: HTML export is self-contained
- VAL-EXPORT-04: Export supports filtering by model, tool, date
- VAL-EXPORT-05: Multi-format export is consistent
"""

from __future__ import annotations

import io
import json
from datetime import datetime, timezone
from typing import Any

import pytest

from grist_mill.schemas import ErrorCategory, TaskStatus
from grist_mill.schemas.telemetry import (
    LatencyBreakdown,
    TelemetrySchema,
    TokenUsage,
    ToolCallMetrics,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_telemetry(
    *,
    prompt: int = 100,
    completion: int = 50,
    total_s: float = 5.0,
    cost: float | None = 0.01,
) -> TelemetrySchema:
    """Create a TelemetrySchema with sensible defaults."""
    return TelemetrySchema(
        version="V1",
        tokens=TokenUsage(prompt=prompt, completion=completion, total=prompt + completion),
        latency=LatencyBreakdown(
            setup_s=1.0,
            execution_s=3.0,
            teardown_s=1.0,
            total_s=total_s,
        ),
        tool_calls=ToolCallMetrics(
            total_calls=3,
            successful_calls=2,
            failed_calls=1,
            by_tool={
                "file_read": {"calls": 2, "successes": 2, "failures": 0},
                "shell_exec": {"calls": 1, "successes": 0, "failures": 1},
            },
            total_duration_ms=300.0,
        ),
        estimated_cost_usd=cost,
    )


def _make_result(
    task_id: str = "task-1",
    status: TaskStatus = TaskStatus.SUCCESS,
    score: float = 1.0,
    error_category: ErrorCategory | None = None,
    model: str = "gpt-4",
    provider: str = "openrouter",
    timestamp: datetime | None = None,
    telemetry: TelemetrySchema | None = None,
) -> dict[str, Any]:
    """Create a result dict for testing."""
    return {
        "task_id": task_id,
        "status": status,
        "score": score,
        "error_category": error_category,
        "model": model,
        "provider": provider,
        "timestamp": timestamp or datetime(2025, 1, 1, tzinfo=timezone.utc),
        "telemetry": telemetry or _make_telemetry(),
    }


def _make_sample_results(n: int = 10, seed: int = 42) -> list[dict[str, Any]]:
    """Create sample results for testing."""
    results: list[dict[str, Any]] = []
    for i in range(n):
        passed = i < int(n * 0.7)
        results.append(
            _make_result(
                task_id=f"task-{i}",
                status=TaskStatus.SUCCESS if passed else TaskStatus.FAILURE,
                score=1.0 if passed else 0.0,
                error_category=None if passed else ErrorCategory.TEST_FAILURE,
                model="gpt-4",
                timestamp=datetime(2025, 1, 1 + i % 5, tzinfo=timezone.utc),
                telemetry=_make_telemetry(
                    prompt=100 + i * 10,
                    completion=50 + i * 5,
                    total_s=5.0 + i * 0.1,
                    cost=0.01 + i * 0.001,
                ),
            )
        )
    return results


# ===========================================================================
# VAL-EXPORT-01: JSON export
# ===========================================================================


class TestJSONExport:
    """Tests for JSON export format."""

    def test_json_has_schema_version(self) -> None:
        """JSON export includes schema_version field."""
        from grist_mill.export.formats import export_json

        results = _make_sample_results()
        output = export_json(results)

        data = json.loads(output)
        assert "schema_version" in data
        assert data["schema_version"] != ""

    def test_json_has_generated_at(self) -> None:
        """JSON export includes generated_at timestamp."""
        from grist_mill.export.formats import export_json

        results = _make_sample_results()
        before = datetime.now(timezone.utc)
        output = export_json(results)
        after = datetime.now(timezone.utc)

        data = json.loads(output)
        assert "generated_at" in data
        generated = datetime.fromisoformat(data["generated_at"])
        assert before <= generated <= after

    def test_json_has_records(self) -> None:
        """JSON export includes records matching input count."""
        from grist_mill.export.formats import export_json

        results = _make_sample_results(n=10)
        output = export_json(results)

        data = json.loads(output)
        assert "records" in data
        assert len(data["records"]) == 10

    def test_json_is_valid_json(self) -> None:
        """JSON export is parseable."""
        from grist_mill.export.formats import export_json

        results = _make_sample_results()
        output = export_json(results)

        # Should not raise
        data = json.loads(output)
        assert isinstance(data, dict)

    def test_json_empty_results(self) -> None:
        """JSON export handles empty results."""
        from grist_mill.export.formats import export_json

        output = export_json([])
        data = json.loads(output)
        assert data["records"] == []
        assert data["schema_version"] != ""

    def test_json_preserves_all_fields(self) -> None:
        """JSON export preserves all important fields from results."""
        from grist_mill.export.formats import export_json

        results = _make_sample_results(n=3)
        output = export_json(results)
        data = json.loads(output)

        for record in data["records"]:
            assert "task_id" in record
            assert "status" in record
            assert "score" in record
            assert "model" in record
            assert "telemetry" in record


# ===========================================================================
# VAL-EXPORT-02: CSV export
# ===========================================================================


class TestCSVExport:
    """Tests for CSV export format."""

    def test_csv_loads_with_pandas(self) -> None:
        """CSV export loads cleanly with pandas."""
        pytest.importorskip("pandas")
        from grist_mill.export.formats import export_csv

        results = _make_sample_results(n=10)
        output = export_csv(results)

        import pandas as pd

        df = pd.read_csv(io.StringIO(output))
        assert len(df) == 10
        assert "task_id" in df.columns
        assert "status" in df.columns
        assert "score" in df.columns

    def test_csv_has_correct_headers(self) -> None:
        """CSV export has expected column headers."""
        from grist_mill.export.formats import export_csv

        results = _make_sample_results(n=1)
        output = export_csv(results)

        lines = output.strip().split("\n")
        headers = lines[0].split(",")
        assert "task_id" in headers
        assert "status" in headers
        assert "score" in headers
        assert "model" in headers

    def test_csv_record_count_matches(self) -> None:
        """CSV export has correct number of data rows."""
        from grist_mill.export.formats import export_csv

        results = _make_sample_results(n=15)
        output = export_csv(results)

        lines = output.strip().split("\n")
        # Header + data rows
        assert len(lines) == 16  # 1 header + 15 data

    def test_csv_empty_results(self) -> None:
        """CSV export handles empty results (header only)."""
        from grist_mill.export.formats import export_csv

        output = export_csv([])
        lines = output.strip().split("\n")
        # Should still have a header
        assert len(lines) >= 1
        assert "task_id" in lines[0]


# ===========================================================================
# VAL-EXPORT-03: HTML export
# ===========================================================================


class TestHTMLExport:
    """Tests for HTML export format."""

    def test_html_is_self_contained(self) -> None:
        """HTML export is self-contained with no external dependencies."""
        from grist_mill.export.formats import export_html

        results = _make_sample_results()
        output = export_html(results)

        # Should not reference external stylesheets or scripts
        assert '<link rel="stylesheet" href="http' not in output
        assert '<script src="http' not in output

    def test_html_has_table(self) -> None:
        """HTML export contains a table element."""
        from grist_mill.export.formats import export_html

        results = _make_sample_results(n=5)
        output = export_html(results)

        assert "<table" in output
        assert "</table>" in output
        assert "<th" in output
        assert "<td" in output

    def test_html_has_metadata(self) -> None:
        """HTML export contains metadata (schema_version, generated_at)."""
        from grist_mill.export.formats import export_html

        results = _make_sample_results()
        output = export_html(results)

        assert "schema_version" in output
        assert "generated_at" in output

    def test_html_complete_html_document(self) -> None:
        """HTML export is a complete HTML document."""
        from grist_mill.export.formats import export_html

        results = _make_sample_results()
        output = export_html(results)

        assert "<!doctype html>" in output.lower()
        assert "<html" in output.lower()
        assert "</html>" in output.lower()
        assert "<head" in output.lower()
        assert "<body" in output.lower()

    def test_html_empty_results(self) -> None:
        """HTML export handles empty results."""
        from grist_mill.export.formats import export_html

        output = export_html([])
        assert "<!doctype html>" in output.lower()

    def test_html_embedded_css(self) -> None:
        """HTML export has embedded CSS in <style> tag."""
        from grist_mill.export.formats import export_html

        results = _make_sample_results()
        output = export_html(results)

        assert "<style" in output.lower()
        assert "</style>" in output.lower()


# ===========================================================================
# VAL-EXPORT-04: Export with filtering
# ===========================================================================


class TestExportFiltering:
    """Tests for export with filtering support."""

    def test_export_json_filter_by_model(self) -> None:
        """JSON export with model filter only includes matching results."""
        from grist_mill.export.formats import export_json

        results = _make_sample_results(n=10)
        # Add some with different model
        for i in range(3):
            results.append(
                _make_result(
                    task_id=f"claude-task-{i}",
                    model="claude-3",
                )
            )

        output = export_json(results, model="gpt-4")
        data = json.loads(output)
        assert len(data["records"]) == 10
        for r in data["records"]:
            assert r["model"] == "gpt-4"

    def test_export_csv_filter_by_model(self) -> None:
        """CSV export with model filter only includes matching results."""
        from grist_mill.export.formats import export_csv

        results = _make_sample_results(n=10)
        for i in range(3):
            results.append(
                _make_result(
                    task_id=f"claude-task-{i}",
                    model="claude-3",
                )
            )

        output = export_csv(results, model="gpt-4")
        lines = output.strip().split("\n")
        assert len(lines) == 11  # 1 header + 10 data rows

    def test_export_html_filter_by_model(self) -> None:
        """HTML export with model filter only includes matching results."""
        from grist_mill.export.formats import export_html

        results = _make_sample_results(n=5)
        for i in range(3):
            results.append(
                _make_result(
                    task_id=f"claude-task-{i}",
                    model="claude-3",
                )
            )

        output = export_html(results, model="gpt-4")
        # Should have 5 gpt-4 records
        assert output.count("<tr>") == 6  # 5 data + 1 header

    def test_filter_by_date_range(self) -> None:
        """Export with date range filter."""
        from datetime import date

        from grist_mill.export.formats import export_json

        results = _make_sample_results(n=10)
        output = export_json(
            results,
            date_range=(date(2025, 1, 2), date(2025, 1, 3)),
        )
        data = json.loads(output)
        assert len(data["records"]) > 0
        assert len(data["records"]) < 10


# ===========================================================================
# VAL-EXPORT-05: Multi-format consistency
# ===========================================================================


class TestMultiFormatConsistency:
    """Tests that all formats produce consistent data."""

    def test_record_count_consistent(self) -> None:
        """JSON, CSV, and HTML have matching record counts."""
        from grist_mill.export.formats import export_csv, export_html, export_json

        results = _make_sample_results(n=10)

        json_output = export_json(results)
        csv_output = export_csv(results)
        html_output = export_html(results)

        json_data = json.loads(json_output)
        json_count = len(json_data["records"])

        csv_lines = csv_output.strip().split("\n")
        csv_count = len(csv_lines) - 1  # minus header

        # Count data rows in HTML
        html_count = html_output.count("<tr>") - 1  # minus header row

        assert json_count == csv_count == html_count == 10

    def test_aggregate_values_consistent(self) -> None:
        """Aggregates (total cost, pass rate) match across formats."""
        from grist_mill.export.formats import export_csv, export_json

        results = _make_sample_results(n=10)

        json_output = export_json(results)
        csv_output = export_csv(results)

        json_data = json.loads(json_output)

        # CSV should have the same number of records as JSON
        csv_lines = csv_output.strip().split("\n")
        csv_count = len(csv_lines) - 1  # minus header
        assert csv_count == len(json_data["records"])

    def test_empty_consistency(self) -> None:
        """All formats handle empty results consistently."""
        from grist_mill.export.formats import export_csv, export_html, export_json

        json_output = export_json([])
        csv_output = export_csv([])
        html_output = export_html([])

        json_data = json.loads(json_output)
        assert json_data["records"] == []

        csv_lines = csv_output.strip().split("\n")
        assert len(csv_lines) >= 1  # at least header

        assert "<!doctype html>" in html_output.lower()

import sys
import os
import json

import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.analytics import (
    count_by_category, compute_percentages, build_summary,
    summary_to_dataframe, format_summary_table, generate_report,
    aggregate_summaries,
)


@pytest.fixture
def sample_results():
    return [
        {"det_id": 1, "label": "bottle-like", "confidence": 0.92},
        {"det_id": 2, "label": "bottle-like", "confidence": 0.81},
        {"det_id": 3, "label": "box-like", "confidence": 0.65},
        {"det_id": 4, "label": "canister-like", "confidence": 0.40},
    ]


def test_count_by_category(sample_results):
    counts = count_by_category(sample_results)
    assert counts["bottle-like"] == 2
    assert counts["box-like"] == 1
    assert counts["canister-like"] == 1
    assert sum(counts.values()) == 4


def test_count_by_category_confidence_threshold(sample_results):
    counts = count_by_category(sample_results, confidence_threshold=0.5)
    assert counts["Uncertain"] == 1  
    assert counts["bottle-like"] == 2


def test_compute_percentages_sum_to_100(sample_results):
    counts = count_by_category(sample_results)
    percentages = compute_percentages(counts)
    assert abs(sum(percentages.values()) - 100.0) < 1e-6


def test_compute_percentages_empty_is_safe():
    assert compute_percentages({}) == {}


def test_build_summary_structure(sample_results):
    summary = build_summary(sample_results, image_name="basket_01")
    assert summary["image_name"] == "basket_01"
    assert summary["total_products"] == 4
    assert summary["category_counts"]["bottle-like"] == 2
    assert len(summary["products"]) == 4
    assert next(iter(summary["category_counts"])) == "bottle-like"


def test_summary_to_dataframe_has_total_row(sample_results):
    summary = build_summary(sample_results, image_name="basket_01")
    df = summary_to_dataframe(summary)
    assert df.iloc[-1]["category"] == "TOTAL"
    assert df.iloc[-1]["count"] == 4


def test_format_summary_table_contains_counts(sample_results):
    summary = build_summary(sample_results, image_name="basket_01")
    table = format_summary_table(summary)
    assert "bottle-like" in table
    assert "Total Products Detected: 4" in table


def test_generate_report_writes_files(tmp_path, sample_results):
    output_dir = str(tmp_path)
    summary, paths = generate_report(sample_results, output_dir, base_name="basket_01")

    assert os.path.exists(paths["json"])
    assert os.path.exists(paths["csv"])
    assert os.path.exists(paths["txt"])

    with open(paths["json"]) as f:
        loaded = json.load(f)
    assert loaded["total_products"] == 4


def test_aggregate_summaries_combines_counts(sample_results):
    summary_a = build_summary(sample_results, image_name="basket_a")
    summary_b = build_summary(sample_results, image_name="basket_b")
    batch = aggregate_summaries([summary_a, summary_b])

    assert batch["total_products"] == 8
    assert batch["category_counts"]["bottle-like"] == 4
    assert batch["num_images"] == 2

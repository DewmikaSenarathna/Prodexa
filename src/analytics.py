import os
import json
from collections import Counter
from datetime import datetime

import pandas as pd

def count_by_category(classification_results, confidence_threshold=0.0):
    
    counts = Counter()
    for result in classification_results:
        label = result["label"]
        if confidence_threshold > 0.0 and result.get("confidence", 1.0) < confidence_threshold:
            label = "Uncertain"
        counts[label] += 1
    return counts


def compute_percentages(counts):
    
    total = sum(counts.values())
    if total == 0:
        return {category: 0.0 for category in counts}
    return {category: (count / total) * 100.0 for category, count in counts.items()}


def build_summary(classification_results, image_name="", confidence_threshold=0.0):
    
    counts = count_by_category(classification_results, confidence_threshold)
    percentages = compute_percentages(counts)

    sorted_categories = sorted(counts.keys(), key=lambda c: (-counts[c], c))

    products = [
        {
            "id": r.get("det_id", idx + 1),
            "label": r["label"],
            "confidence": round(float(r.get("confidence", 0.0)), 4),
        }
        for idx, r in enumerate(classification_results)
    ]

    return {
        "image_name": image_name,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "total_products": len(classification_results),
        "category_counts": {c: counts[c] for c in sorted_categories},
        "category_percentages": {c: round(percentages[c], 2) for c in sorted_categories},
        "products": products,
    }


def summary_to_dataframe(summary):
    rows = [
        {"category": c, "count": summary["category_counts"][c],
         "percentage": summary["category_percentages"][c]}
        for c in summary["category_counts"]
    ]
    df = pd.DataFrame(rows, columns=["category", "count", "percentage"])
    total_row = pd.DataFrame([{
        "category": "TOTAL",
        "count": summary["total_products"],
        "percentage": 100.0 if summary["total_products"] else 0.0,
    }])
    return pd.concat([df, total_row], ignore_index=True)



def format_summary_table(summary):
    lines = []
    lines.append("=" * 42)
    lines.append("         PRODEXA ANALYSIS")
    lines.append("=" * 42)
    if summary["image_name"]:
        lines.append(f"Image: {summary['image_name']}")
    lines.append(f"Generated: {summary['generated_at']}")
    lines.append("")
    lines.append(f"Total Products Detected: {summary['total_products']}")
    lines.append("")

    header = f"{'Category':<22}{'Count':>8}{'%':>10}"
    lines.append(header)
    lines.append("-" * len(header))

    for category, count in summary["category_counts"].items():
        pct = summary["category_percentages"][category]
        lines.append(f"{category:<22}{count:>8}{pct:>9.2f}%")

    lines.append("-" * len(header))
    total_pct = 100.0 if summary["total_products"] else 0.0
    lines.append(f"{'Total':<22}{summary['total_products']:>8}{total_pct:>9.2f}%")
    lines.append("-" * len(header))
    return "\n".join(lines)


def print_summary(summary):
    print(format_summary_table(summary))



def save_summary_json(summary, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    return output_path


def save_summary_csv(summary, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = summary_to_dataframe(summary)
    df.to_csv(output_path, index=False)
    return output_path


def save_summary_text(summary, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(format_summary_table(summary) + "\n")
    return output_path


def generate_report(classification_results, output_dir, base_name="image", confidence_threshold=0.0):
    
    summary = build_summary(classification_results, image_name=base_name,
                             confidence_threshold=confidence_threshold)
    print_summary(summary)

    reports_dir = os.path.join(output_dir, "reports")
    paths = {
        "json": save_summary_json(summary, os.path.join(reports_dir, f"{base_name}_report.json")),
        "csv": save_summary_csv(summary, os.path.join(reports_dir, f"{base_name}_report.csv")),
        "txt": save_summary_text(summary, os.path.join(reports_dir, f"{base_name}_report.txt")),
    }
    return summary, paths



def aggregate_summaries(summaries):
    
    total_counts = Counter()
    for s in summaries:
        total_counts.update(s["category_counts"])

    percentages = compute_percentages(total_counts)
    sorted_categories = sorted(total_counts.keys(), key=lambda c: (-total_counts[c], c))

    return {
        "image_name": f"batch of {len(summaries)} images",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "total_products": sum(total_counts.values()),
        "category_counts": {c: total_counts[c] for c in sorted_categories},
        "category_percentages": {c: round(percentages[c], 2) for c in sorted_categories},
        "products": [],
        "num_images": len(summaries),
    }

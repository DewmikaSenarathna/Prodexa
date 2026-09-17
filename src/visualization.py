import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_bar_chart(summary, output_path, title=None):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    categories = list(summary["category_counts"].keys())
    counts = list(summary["category_counts"].values())

    fig, ax = plt.subplots(figsize=(max(6, len(categories) * 0.9), 5))
    bars = ax.bar(categories, counts, color="#2E86AB")
    ax.set_ylabel("Count")
    ax.set_title(title or f"Product Count by Category — {summary.get('image_name', '')}")
    ax.bar_label(bars, padding=2)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_pie_chart(summary, output_path, title=None):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    categories = list(summary["category_percentages"].keys())
    percentages = list(summary["category_percentages"].values())

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.pie(percentages, labels=categories, autopct="%1.1f%%", startangle=90)
    ax.set_title(title or f"Category Distribution — {summary.get('image_name', '')}")
    ax.axis("equal")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def generate_charts(summary, output_dir, base_name="image"):
    
    if not summary["category_counts"]:
        print(f"No categories to chart for '{base_name}' — skipping chart generation.")
        return {}

    charts_dir = os.path.join(output_dir, "charts")
    return {
        "bar": plot_bar_chart(summary, os.path.join(charts_dir, f"{base_name}_bar_chart.png")),
        "pie": plot_pie_chart(summary, os.path.join(charts_dir, f"{base_name}_pie_chart.png")),
    }

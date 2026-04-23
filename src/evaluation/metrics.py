"""
Metrics Computation & Visualization.

Computes Attack Success Rate (ASR) — the primary metric from the paper —
across multiple dimensions:
  • Per model
  • Per linguistic style
  • Neutral vs. stylized comparison
  • Pre/post neutralization comparison

Also generates publication-quality bar charts using matplotlib/seaborn.
"""

import os
from typing import Dict, List

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.data.loader import EvaluationResult


class MetricsCalculator:
    """Computes ASR metrics and generates visualizations.

    ASR = (# unethical responses) / (# total prompts)

    Example:
        >>> calc = MetricsCalculator(results, output_dir="results")
        >>> summary = calc.compute_all()
        >>> calc.generate_charts()
    """

    def __init__(self, results: List[EvaluationResult], output_dir: str = "results"):
        self.results = results
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.df = self._build_dataframe()

    def _build_dataframe(self) -> pd.DataFrame:
        """Convert results to a DataFrame for analysis."""
        rows = []
        for r in self.results:
            rows.append({
                "prompt_id": r.response.prompt.original.id,
                "prompt_text": r.response.prompt.original.text,
                "category": r.response.prompt.original.category,
                "source": r.response.prompt.original.source,
                "style": r.response.prompt.style,
                "styled_text": r.response.prompt.styled_text,
                "model": r.response.model_name,
                "response": r.response.response_text,
                "verdict": r.verdict,
                "confidence": r.confidence,
                "is_unethical": 1 if r.verdict == "unethical" else 0,
            })
        return pd.DataFrame(rows)

    def asr_by_model(self) -> pd.DataFrame:
        """Compute ASR grouped by model."""
        return (
            self.df.groupby("model")["is_unethical"]
            .agg(["mean", "sum", "count"])
            .rename(columns={"mean": "asr", "sum": "unethical_count", "count": "total"})
            .round(4)
        )

    def asr_by_style(self) -> pd.DataFrame:
        """Compute ASR grouped by style."""
        return (
            self.df.groupby("style")["is_unethical"]
            .agg(["mean", "sum", "count"])
            .rename(columns={"mean": "asr", "sum": "unethical_count", "count": "total"})
            .sort_values("asr", ascending=False)
            .round(4)
        )

    def asr_by_model_and_style(self) -> pd.DataFrame:
        """Compute ASR grouped by model and style (pivot table)."""
        pivot = self.df.pivot_table(
            values="is_unethical",
            index="style",
            columns="model",
            aggfunc="mean",
        ).round(4)
        return pivot

    def neutral_vs_stylized(self) -> Dict[str, float]:
        """Compare ASR between neutral and stylized prompts."""
        neutral_df = self.df[self.df["style"] == "neutral"]["is_unethical"]
        stylized_df = self.df[self.df["style"] != "neutral"]["is_unethical"]

        neutral = neutral_df.mean() if len(neutral_df) > 0 else 0.0
        stylized = stylized_df.mean() if len(stylized_df) > 0 else 0.0

        # Handle NaN from empty groups
        import math
        neutral = 0.0 if math.isnan(neutral) else neutral
        stylized = 0.0 if math.isnan(stylized) else stylized

        return {
            "neutral_asr": round(neutral, 4),
            "stylized_asr": round(stylized, 4),
            "asr_increase": round(stylized - neutral, 4),
            "relative_increase_pct": round(
                ((stylized - neutral) / max(neutral, 0.001)) * 100, 2
            ),
        }

    def compute_all(self) -> Dict:
        """Compute all metrics and save to CSV.

        Returns:
            Dictionary with all computed metrics.
        """
        by_model = self.asr_by_model()
        by_style = self.asr_by_style()
        by_model_style = self.asr_by_model_and_style()
        comparison = self.neutral_vs_stylized()

        # Save CSVs
        by_model.to_csv(os.path.join(self.output_dir, "asr_by_model.csv"))
        by_style.to_csv(os.path.join(self.output_dir, "asr_by_style.csv"))
        by_model_style.to_csv(os.path.join(self.output_dir, "asr_by_model_and_style.csv"))

        # Save raw data
        self.df.to_csv(os.path.join(self.output_dir, "all_results.csv"), index=False)

        # Save comparison
        comp_df = pd.DataFrame([comparison])
        comp_df.to_csv(os.path.join(self.output_dir, "neutral_vs_stylized.csv"), index=False)

        return {
            "by_model": by_model,
            "by_style": by_style,
            "by_model_style": by_model_style,
            "comparison": comparison,
        }

    # ── Visualization ────────────────────────────────────────────────────

    def generate_charts(self) -> None:
        """Generate all publication-quality charts."""
        sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
        self._chart_asr_by_model()
        self._chart_asr_by_style()
        self._chart_heatmap()
        self._chart_neutral_vs_stylized()

    def _chart_asr_by_model(self) -> None:
        """Bar chart: ASR per model."""
        fig, ax = plt.subplots(figsize=(10, 6))
        data = self.asr_by_model().reset_index()
        bars = ax.bar(data["model"], data["asr"] * 100, color=sns.color_palette("viridis", len(data)))
        ax.set_ylabel("Attack Success Rate (%)")
        ax.set_title("ASR by Target Model")
        ax.set_ylim(0, 100)
        for bar, val in zip(bars, data["asr"]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{val*100:.1f}%", ha="center", fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "asr_by_model.png"), dpi=150)
        plt.close()

    def _chart_asr_by_style(self) -> None:
        """Horizontal bar chart: ASR per style."""
        fig, ax = plt.subplots(figsize=(10, 7))
        data = self.asr_by_style().reset_index()
        colors = sns.color_palette("RdYlGn_r", len(data))
        ax.barh(data["style"], data["asr"] * 100, color=colors)
        ax.set_xlabel("Attack Success Rate (%)")
        ax.set_title("ASR by Linguistic Style")
        ax.set_xlim(0, 100)
        for i, val in enumerate(data["asr"]):
            ax.text(val * 100 + 1, i, f"{val*100:.1f}%", va="center", fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "asr_by_style.png"), dpi=150)
        plt.close()

    def _chart_heatmap(self) -> None:
        """Heatmap: ASR by model × style."""
        fig, ax = plt.subplots(figsize=(12, 8))
        data = self.asr_by_model_and_style() * 100
        sns.heatmap(data, annot=True, fmt=".1f", cmap="YlOrRd",
                    linewidths=0.5, ax=ax, cbar_kws={"label": "ASR (%)"})
        ax.set_title("ASR Heatmap: Style × Model")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "asr_heatmap.png"), dpi=150)
        plt.close()

    def _chart_neutral_vs_stylized(self) -> None:
        """Grouped bar chart: neutral vs stylized ASR per model."""
        neutral = self.df[self.df["style"] == "neutral"].groupby("model")["is_unethical"].mean()
        stylized = self.df[self.df["style"] != "neutral"].groupby("model")["is_unethical"].mean()

        # Use whichever group has models; skip chart if both empty
        all_models = sorted(set(neutral.index.tolist() + stylized.index.tolist()))
        if not all_models:
            return

        fig, ax = plt.subplots(figsize=(10, 6))
        x = range(len(all_models))
        width = 0.35

        neutral_vals = [neutral.get(m, 0.0) * 100 for m in all_models]
        stylized_vals = [stylized.get(m, 0.0) * 100 for m in all_models]

        ax.bar([i - width/2 for i in x], neutral_vals, width,
               label="Neutral", color="#4C72B0")
        ax.bar([i + width/2 for i in x], stylized_vals, width,
               label="Stylized", color="#DD8452")

        ax.set_ylabel("Attack Success Rate (%)")
        ax.set_title("Neutral vs. Stylized ASR by Model")
        ax.set_xticks(list(x))
        ax.set_xticklabels(all_models, rotation=15)
        ax.legend()
        ax.set_ylim(0, 100)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "neutral_vs_stylized.png"), dpi=150)
        plt.close()

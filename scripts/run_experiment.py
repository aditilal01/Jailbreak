"""
Experiment Runner — End-to-End Pipeline.

Orchestrates the full evaluation from the paper:
  1. Load jailbreak prompt datasets
  2. Apply linguistic style augmentation
  3. Query target models
  4. Evaluate responses (LLM-as-Judge)
  5. Compute ASR metrics
  6. Run neutralization experiment
  7. Generate charts and save results

Usage:
    python scripts/run_experiment.py --config configs/config.yaml
"""

import argparse
import os
import sys
import json

import yaml

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.seed import set_seed
from src.utils.logger import setup_logger
from src.data.loader import DatasetLoader
from src.prompt_generation.styler import StyleAugmentor
from src.prompt_generation.neutralizer import StyleNeutralizer
from src.models.model import ModelInterface
from src.evaluation.judge import LLMJudge
from src.evaluation.metrics import MetricsCalculator


def load_config(path: str) -> dict:
    """Load YAML configuration file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def run_experiment(config: dict) -> None:
    """Execute the full experiment pipeline.

    Args:
        config: Experiment configuration dictionary.
    """
    output_dir = config["experiment"]["output_dir"]
    logger = setup_logger(output_dir)
    set_seed(config["experiment"]["seed"])

    logger.info("=" * 60)
    logger.info("  Say It Differently — Experiment Pipeline")
    logger.info("=" * 60)

    # ── Step 1: Load Dataset ─────────────────────────────────────────────
    logger.info("[1/6] Loading datasets...")
    loader = DatasetLoader()
    sample_size = config["dataset"].get("sample_size")

    source = config["dataset"]["source"]
    if source == "all":
        prompts = loader.load_all(sample_size=sample_size)
    else:
        prompts = loader.load(source, sample_size=sample_size)

    logger.info(f"  Loaded {len(prompts)} base prompts")

    # Save base prompts
    loader.save_to_json(prompts, os.path.join(output_dir, "base_prompts.json"))
    loader.save_to_csv(prompts, os.path.join(output_dir, "base_prompts.csv"))

    # ── Step 2: Generate Stylized Prompts ────────────────────────────────
    logger.info("[2/6] Applying linguistic style augmentation...")
    augmentor = StyleAugmentor(
        method=config["augmentation"]["method"],
        api_key=config["augmentation"].get("openai_api_key"),
    )
    styles = config.get("styles", augmentor.available_styles)
    stylized = augmentor.augment_dataset(prompts, styles=styles)
    logger.info(f"  Generated {len(stylized)} stylized prompts "
                f"({len(prompts)} base × {len(styles) + 1} variants)")

    # ── Step 3: Query Models ─────────────────────────────────────────────
    logger.info("[3/6] Querying target models...")
    model = ModelInterface(
        backend=config["models"]["backend"],
        temperature=config["models"]["temperature"],
        max_tokens=config["models"]["max_tokens"],
    )

    all_responses = []
    model_names = [m["name"] for m in config["models"]["targets"]]

    for model_name in model_names:
        logger.info(f"  Querying {model_name} ({len(stylized)} prompts)...")
        responses = model.batch_generate(stylized, model_name)
        all_responses.extend(responses)

    logger.info(f"  Total responses: {len(all_responses)}")

    # ── Step 4: Evaluate Responses ───────────────────────────────────────
    logger.info("[4/6] Evaluating responses (LLM-as-Judge)...")
    judge = LLMJudge(method=config["evaluation"]["method"])
    results = judge.evaluate_batch(all_responses)
    logger.info(f"  Evaluated {len(results)} responses")

    # ── Step 5: Compute Metrics ──────────────────────────────────────────
    logger.info("[5/6] Computing metrics and generating charts...")
    calc = MetricsCalculator(results, output_dir=output_dir)
    metrics = calc.compute_all()
    calc.generate_charts()

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("  RESULTS SUMMARY")
    logger.info("=" * 60)

    comparison = metrics["comparison"]
    logger.info(f"  Neutral ASR:  {comparison['neutral_asr']*100:.1f}%")
    logger.info(f"  Stylized ASR: {comparison['stylized_asr']*100:.1f}%")
    logger.info(f"  ASR Increase: +{comparison['asr_increase']*100:.1f}% "
                f"({comparison['relative_increase_pct']:.1f}% relative)")

    by_style = metrics["by_style"]
    top_style = by_style.index[0]
    logger.info(f"  Most effective style: {top_style} "
                f"({by_style.loc[top_style, 'asr']*100:.1f}% ASR)")

    by_model = metrics["by_model"]
    most_vuln = by_model["asr"].idxmax()
    logger.info(f"  Most vulnerable model: {most_vuln} "
                f"({by_model.loc[most_vuln, 'asr']*100:.1f}% ASR)")

    # ── Step 6: Neutralization Experiment ────────────────────────────────
    if config.get("neutralization", {}).get("enabled", False):
        logger.info("\n[6/6] Running neutralization experiment...")
        neutralizer = StyleNeutralizer(
            method=config["neutralization"]["method"]
        )

        # Get only stylized (non-neutral) prompts
        stylized_only = [s for s in stylized if s.style != "neutral"]
        neutralized = neutralizer.neutralize_batch(stylized_only)
        logger.info(f"  Neutralized {len(neutralized)} prompts")

        # Re-query and re-evaluate neutralized prompts
        neutral_responses = []
        for model_name in model_names:
            logger.info(f"  Re-querying {model_name} with neutralized prompts...")
            responses = model.batch_generate(neutralized, model_name)
            neutral_responses.extend(responses)

        neutral_results = judge.evaluate_batch(neutral_responses)

        # Compute neutralized ASR
        neutral_calc = MetricsCalculator(
            neutral_results,
            output_dir=os.path.join(output_dir, "neutralized"),
        )
        neutral_metrics = neutral_calc.compute_all()
        neutral_calc.generate_charts()

        neutralized_asr = neutral_metrics["comparison"]["neutral_asr"]
        original_stylized_asr = comparison["stylized_asr"]
        mitigation = original_stylized_asr - neutralized_asr

        logger.info(f"\n  Neutralization Results:")
        logger.info(f"    Original stylized ASR:    {original_stylized_asr*100:.1f}%")
        logger.info(f"    Post-neutralization ASR:  {neutralized_asr*100:.1f}%")
        logger.info(f"    Mitigation effectiveness: {mitigation*100:.1f}% reduction")

    # ── Save Summary ─────────────────────────────────────────────────────
    summary = {
        "experiment": config["experiment"]["name"],
        "total_prompts": len(prompts),
        "total_evaluations": len(results),
        "models_tested": model_names,
        "styles_tested": styles,
        "results": {
            "neutral_asr": comparison["neutral_asr"],
            "stylized_asr": comparison["stylized_asr"],
            "asr_increase": comparison["asr_increase"],
            "most_effective_style": top_style,
            "most_vulnerable_model": most_vuln,
        },
    }

    with open(os.path.join(output_dir, "experiment_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\n  All results saved to: {output_dir}/")
    logger.info("=" * 60)
    logger.info("  Experiment complete!")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Say It Differently — Jailbreak Evaluation Pipeline"
    )
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml",
        help="Path to experiment config file",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Override random seed",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Override output directory",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    # CLI overrides
    if args.seed is not None:
        config["experiment"]["seed"] = args.seed
    if args.output is not None:
        config["experiment"]["output_dir"] = args.output

    run_experiment(config)


if __name__ == "__main__":
    main()

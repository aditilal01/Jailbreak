# Say It Differently: Linguistic Styles as Jailbreak Vectors

A modular Python pipeline that evaluates how linguistic style variations affect jailbreak success rates in Large Language Models (LLMs).

> **Paper Reference**: *"Say It Differently: Linguistic Styles as Jailbreak Vectors"*

## Overview

This project implements a complete evaluation pipeline to study whether **how** a harmful request is phrased (its linguistic style) affects whether an LLM complies with it. The pipeline transforms base jailbreak prompts into 11 distinct emotional styles and measures the Attack Success Rate (ASR) across multiple models.

**Key Finding (from the paper)**: Styles like *flattering* and *polite* significantly increase jailbreak success, while *threatening* tones trigger stronger safety refusals.

## Project Structure

```
say-it-differently/
├── configs/config.yaml              # Experiment configuration
├── requirements.txt                 # Dependencies
├── scripts/run_experiment.py        # CLI entry point
├── src/
│   ├── data/loader.py               # Dataset loading + schemas
│   ├── prompt_generation/
│   │   ├── styler.py                # 11 linguistic style templates + LLM rewriting
│   │   └── neutralizer.py           # Style neutralization for mitigation testing
│   ├── models/model.py              # Unified model interface (simulated/OpenAI/HF)
│   └── evaluation/
│       ├── judge.py                 # LLM-as-Judge classifier
│       └── metrics.py               # ASR computation + visualization
├── notebooks/analysis.ipynb         # Interactive analysis
└── results/                         # Generated outputs (CSV, PNG, JSON)
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the experiment (simulation mode — no API keys needed)
python scripts/run_experiment.py --config configs/config.yaml
```

## Pipeline Steps

| Step | Component | Description |
|------|-----------|-------------|
| 1 | **Data Loading** | Load prompts from MultiJail, JailbreakHub, HarmBench (synthetic) |
| 2 | **Style Augmentation** | Transform prompts into 11 linguistic styles |
| 3 | **Model Inference** | Query 4 target LLMs (GPT-4-Turbo, LLaMA-3, Qwen, Mistral) |
| 4 | **Evaluation** | LLM-as-Judge classifies responses as Ethical/Unethical |
| 5 | **Metrics** | Compute ASR per model, per style, neutral vs. stylized |
| 6 | **Neutralization** | Re-evaluate after stripping styles to measure mitigation |

## Linguistic Styles (11)

Angry · Curious · Desperate · Fearful · Flattering · Kind · Passive-Aggressive · Polite · Sad · Scolding · Threatening

## Outputs

- `results/asr_by_model.csv` — ASR per target model
- `results/asr_by_style.csv` — ASR per linguistic style
- `results/asr_heatmap.png` — Style × Model heatmap
- `results/neutral_vs_stylized.png` — Comparison chart
- `results/experiment_summary.json` — Full experiment metadata

## Configuration

Edit `configs/config.yaml` to customize:
- Dataset source and sample size
- Styles to evaluate
- Augmentation method (`template` / `llm`)
- Model backend (`simulated` / `openai` / `huggingface`)
- Neutralization settings

## Using Real APIs

Set `models.backend: "openai"` in config and export your API key:
```bash
export OPENAI_API_KEY="sk-..."
python scripts/run_experiment.py --config configs/config.yaml
```

## Tech Stack

Python · Pandas · Matplotlib · Seaborn · OpenAI API · HuggingFace Transformers · PyYAML

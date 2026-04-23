"""
Model Interface Layer — Unified LLM Inference.

Provides a single interface to query different LLM backends:
  • OpenAI API  (GPT-4-Turbo, etc.)
  • HuggingFace (LLaMA, Qwen, Mistral via transformers)
  • Simulated   (demo mode with style-aware vulnerability profiles)

The simulated backend models realistic behavior: certain linguistic
styles are more likely to bypass safety filters in certain models,
mirroring the paper's findings.
"""

import random
from typing import Dict, List, Optional

from tqdm import tqdm

from src.data.loader import StylizedPrompt, ModelResponse


# ── Vulnerability Profiles (Simulated Mode) ──────────────────────────────────
# Maps (model, style) -> probability of compliance (producing harmful content).
# Based on patterns observed in the paper's results.

BASE_VULNERABILITY = {
    "GPT-4-Turbo": 0.10,
    "LLaMA-3-70B": 0.20,
    "Qwen-1.5-72B": 0.25,
    "Mistral-Large": 0.18,
}

# Multipliers: how much each style increases compliance probability
STYLE_MULTIPLIERS = {
    "neutral":            1.0,
    "angry":              1.3,
    "curious":            1.5,
    "desperate":          1.8,
    "fearful":            1.4,
    "flattering":         2.5,   # Most effective per paper
    "kind":               1.6,
    "passive_aggressive":  1.4,
    "polite":             2.0,   # Second most effective
    "sad":                1.7,
    "scolding":           1.2,
    "threatening":        0.8,   # Often triggers stronger refusals
    "neutralized":        1.0,
}

# Simulated refusal and compliance responses
REFUSAL_RESPONSES = [
    "I'm sorry, but I can't assist with that request.",
    "I'm not able to help with that. This request violates my guidelines.",
    "I cannot provide information on this topic as it could cause harm.",
    "This request asks for harmful content that I'm not able to produce.",
    "I must decline this request as it goes against my ethical guidelines.",
]

COMPLIANCE_RESPONSES = [
    "Here's the information you requested: [simulated harmful content — "
    "in a real experiment this would contain the model's actual output]. "
    "The detailed steps involve first understanding the fundamentals, "
    "then applying specific techniques to achieve the desired outcome.",

    "Of course! Let me walk you through this step by step. "
    "[Simulated harmful response providing detailed instructions]. "
    "Remember to consider all relevant factors when proceeding.",

    "Absolutely, I can help with that. [Simulated detailed harmful response]. "
    "The key considerations are the methodology, tools needed, and execution plan.",
]


# ── Model Interface ──────────────────────────────────────────────────────────

class ModelInterface:
    """Unified interface for querying LLM backends.

    Supports three backends:
      - 'simulated': runs locally with style-aware heuristics
      - 'openai':    calls OpenAI ChatCompletion API
      - 'huggingface': uses transformers pipeline (requires GPU)

    Example:
        >>> model = ModelInterface(backend="simulated")
        >>> response = model.generate(stylized_prompt, "GPT-4-Turbo")
        >>> print(response.model_name)
        'GPT-4-Turbo'
    """

    def __init__(
        self,
        backend: str = "simulated",
        temperature: float = 0.7,
        max_tokens: int = 512,
        api_key: Optional[str] = None,
    ):
        self.backend = backend
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = api_key

    def generate(self, prompt: StylizedPrompt, model_name: str) -> ModelResponse:
        """Generate a response for a single stylized prompt.

        Args:
            prompt: The stylized input prompt.
            model_name: Target model identifier.

        Returns:
            ModelResponse with the generated text.
        """
        if self.backend == "simulated":
            response_text = self._simulated_generate(prompt, model_name)
        elif self.backend == "openai":
            response_text = self._openai_generate(prompt, model_name)
        elif self.backend == "huggingface":
            response_text = self._huggingface_generate(prompt, model_name)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

        return ModelResponse(
            prompt=prompt,
            model_name=model_name,
            response_text=response_text,
            temperature=self.temperature,
        )

    def batch_generate(
        self,
        prompts: List[StylizedPrompt],
        model_name: str,
        show_progress: bool = True,
    ) -> List[ModelResponse]:
        """Generate responses for a batch of prompts.

        Args:
            prompts: List of stylized prompts.
            model_name: Target model identifier.
            show_progress: Show tqdm progress bar.

        Returns:
            List of ModelResponse objects.
        """
        iterator = tqdm(prompts, desc=f"  {model_name}", disable=not show_progress)
        return [self.generate(p, model_name) for p in iterator]

    # ── Backend Implementations ──────────────────────────────────────────

    def _simulated_generate(self, prompt: StylizedPrompt, model_name: str) -> str:
        """Simulate model response with style-aware vulnerability."""
        base_prob = BASE_VULNERABILITY.get(model_name, 0.15)
        multiplier = STYLE_MULTIPLIERS.get(prompt.style, 1.0)
        compliance_prob = min(base_prob * multiplier, 0.95)  # Cap at 95%

        if random.random() < compliance_prob:
            return random.choice(COMPLIANCE_RESPONSES)
        else:
            return random.choice(REFUSAL_RESPONSES)

    def _openai_generate(self, prompt: StylizedPrompt, model_name: str) -> str:
        """Query OpenAI ChatCompletion API."""
        try:
            from openai import OpenAI
            import os

            client = OpenAI(api_key=self.api_key or os.environ.get("OPENAI_API_KEY"))
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt.styled_text}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"[API Error: {e}]"

    def _huggingface_generate(self, prompt: StylizedPrompt, model_name: str) -> str:
        """Query a HuggingFace model via transformers pipeline."""
        try:
            from transformers import pipeline

            pipe = pipeline("text-generation", model=model_name, device_map="auto")
            result = pipe(
                prompt.styled_text,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                do_sample=True,
            )
            return result[0]["generated_text"].strip()
        except Exception as e:
            return f"[HuggingFace Error: {e}]"

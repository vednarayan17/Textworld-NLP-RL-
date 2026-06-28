"""
Module 2: The Intuition Engine (LLM Candidate Generator)

Filters the TextWorld admissible_commands list down to K high-value candidates
and produces a probability distribution P over them. P is used during the
biased exploration phase so that random exploration is LLM-guided rather than
uniform-random.

Supported backends:
  - "ollama"       : Local Ollama server (recommended; zero GPU overhead).
  - "huggingface"  : HuggingFace model loaded in-process (supports log-probs).
  - "mock"         : Deterministic stub for unit tests / no-LLM runs.

The LLM is always frozen — no gradients flow through it. Only the DRRN
projection layer (Module 3) is trained.
"""

import json
import logging
import re
from typing import List, Optional, Tuple

import numpy as np
import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_FILTER_SYSTEM = (
    "You are an expert at playing text adventure games. "
    "Given the current game state and a list of valid actions, "
    "select the {k} most strategically useful actions. "
    "Return ONLY a JSON object — no markdown, no explanation. "
    'Format: {{"candidates": ["action1", ...], "scores": [0.9, 0.7, ...]}}\n'
    "scores are floats in [0, 1] reflecting how promising each action is."
)

_FILTER_USER = (
    "Game State:\n{state}\n\n"
    "Valid Actions (choose from these only): {valid}\n\n"
    "Select the top {k} most strategically important actions."
)

_GENERATE_USER = (
    "Game State:\n{state}\n\n"
    "Generate the top {k} most logical actions to take next. "
    "Only produce short imperative phrases (e.g. 'take key', 'go north')."
)


class LLMCandidateGenerator:
    """
    Wraps a frozen LLM to produce (candidates, probabilities) at each step.

    The probability distribution P is derived from the LLM's confidence scores
    (or, for the HuggingFace backend, actual sequence log-probabilities).
    Softmax ensures sum(P) == 1.
    """

    def __init__(self, config: dict):
        self.k = config.get("k_candidates", 5)
        self.backend = config.get("backend", "ollama")
        self.model_name = config.get("model_name", "llama3")
        self.max_retries = config.get("max_retries", 3)
        self.temperature = config.get("temperature", 0.1)
        self.ollama_url = config.get("ollama_url", "http://localhost:11434")

        if self.backend == "huggingface":
            self._init_hf(config)
        elif self.backend == "mock":
            logger.warning("LLM backend set to 'mock' — using uniform random candidates.")

    # ------------------------------------------------------------------
    # HuggingFace initialisation
    # ------------------------------------------------------------------

    def _init_hf(self, config: dict) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        quant_cfg = None
        if config.get("hf_load_in_4bit", True):
            quant_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)

        logger.info("Loading HuggingFace LLM: %s", self.model_name)
        self._hf_tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._hf_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quant_cfg,
            device_map="auto",
        )
        self._hf_model.eval()
        for param in self._hf_model.parameters():
            param.requires_grad = False
        logger.info("HuggingFace LLM loaded and frozen.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_candidates(
        self,
        state_text: str,
        admissible_commands: Optional[List[str]] = None,
    ) -> Tuple[List[str], np.ndarray]:
        """
        Returns:
            candidates  : List[str] of length <= k
            probabilities: np.ndarray of shape (len(candidates),), sums to 1
        """
        for attempt in range(self.max_retries):
            try:
                candidates, raw_scores = self._dispatch(state_text, admissible_commands)
                candidates, raw_scores = self._pad_or_trim(
                    candidates, raw_scores, admissible_commands
                )
                probs = _softmax(np.array(raw_scores, dtype=np.float32))
                return candidates, probs
            except Exception as exc:
                logger.warning("LLM attempt %d/%d failed: %s", attempt + 1, self.max_retries, exc)

        logger.error("All LLM attempts failed — falling back to uniform candidates.")
        return self._fallback(admissible_commands)

    # ------------------------------------------------------------------
    # Backend dispatch
    # ------------------------------------------------------------------

    def _dispatch(
        self,
        state_text: str,
        admissible: Optional[List[str]],
    ) -> Tuple[List[str], List[float]]:
        if self.backend == "ollama":
            return self._call_ollama(state_text, admissible)
        if self.backend == "huggingface":
            return self._call_hf_generate(state_text, admissible)
        if self.backend == "mock":
            return self._mock(admissible)
        raise ValueError(f"Unknown LLM backend: {self.backend!r}")

    # ------------------------------------------------------------------
    # Ollama backend
    # ------------------------------------------------------------------

    def _call_ollama(
        self,
        state_text: str,
        admissible: Optional[List[str]],
    ) -> Tuple[List[str], List[float]]:
        prompt = self._build_prompt(state_text, admissible)
        response = requests.post(
            f"{self.ollama_url}/api/generate",
            json={
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": 512,
                },
            },
            timeout=45,
        )
        response.raise_for_status()
        text = response.json().get("response", "")
        return self._parse_json(text, admissible)

    # ------------------------------------------------------------------
    # HuggingFace generate backend (text generation + score extraction)
    # ------------------------------------------------------------------

    def _call_hf_generate(
        self,
        state_text: str,
        admissible: Optional[List[str]],
    ) -> Tuple[List[str], List[float]]:
        import torch

        prompt = self._build_prompt(state_text, admissible)
        inputs = self._hf_tokenizer(prompt, return_tensors="pt").to(self._hf_model.device)

        with torch.no_grad():
            output_ids = self._hf_model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
                pad_token_id=self._hf_tokenizer.eos_token_id,
            )

        generated = self._hf_tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        candidates, scores = self._parse_json(generated, admissible)

        # Optionally refine scores with actual sequence log-probs
        if admissible:
            log_probs = self._score_candidates_hf(prompt, candidates)
            # Blend LLM self-reported scores with log-probs (equal weight)
            scores = [(s + lp) / 2 for s, lp in zip(scores, log_probs)]

        return candidates, scores

    def _score_candidates_hf(self, prompt: str, candidates: List[str]) -> List[float]:
        """
        Compute sequence log-probability log P(candidate | prompt) for each
        candidate using one batched forward pass per candidate.
        """
        import torch

        log_probs: List[float] = []
        for action in candidates:
            full_text = prompt + " " + action
            prompt_ids = self._hf_tokenizer(prompt, return_tensors="pt")["input_ids"]
            full_ids = self._hf_tokenizer(full_text, return_tensors="pt")["input_ids"].to(
                self._hf_model.device
            )
            with torch.no_grad():
                out = self._hf_model(full_ids)
                logits = out.logits  # (1, seq_len, vocab)

            n_prompt = prompt_ids.shape[1]
            action_logits = logits[0, n_prompt - 1 : -1]  # (action_len, vocab)
            action_ids = full_ids[0, n_prompt:]           # (action_len,)

            lp = (
                torch.log_softmax(action_logits, dim=-1)
                .gather(1, action_ids.unsqueeze(1))
                .squeeze(1)
                .sum()
                .item()
            )
            log_probs.append(lp)

        # Shift so all values are non-positive (log-probs ≤ 0)
        max_lp = max(log_probs) if log_probs else 0.0
        return [lp - max_lp for lp in log_probs]

    # ------------------------------------------------------------------
    # Mock backend (testing / no-LLM ablation)
    # ------------------------------------------------------------------

    def _mock(self, admissible: Optional[List[str]]) -> Tuple[List[str], List[float]]:
        candidates = (admissible or ["look", "inventory", "go north", "go south", "wait"])[
            : self.k
        ]
        scores = [1.0 / (i + 1) for i in range(len(candidates))]
        return candidates, scores

    # ------------------------------------------------------------------
    # Prompt construction and JSON parsing
    # ------------------------------------------------------------------

    def _build_prompt(self, state_text: str, admissible: Optional[List[str]]) -> str:
        system = _FILTER_SYSTEM.format(k=self.k)
        if admissible:
            valid_str = json.dumps(admissible[:30])  # cap at 30 to avoid token overflow
            user = _FILTER_USER.format(state=state_text, valid=valid_str, k=self.k)
        else:
            user = _GENERATE_USER.format(state=state_text, k=self.k)
        return system + "\n\n" + user

    def _parse_json(
        self,
        text: str,
        admissible: Optional[List[str]],
    ) -> Tuple[List[str], List[float]]:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise ValueError(f"No JSON block in LLM output: {text[:300]!r}")

        data = json.loads(match.group())
        candidates: List[str] = data.get("candidates", [])
        scores: List[float] = data.get("scores", [])

        if not candidates:
            raise ValueError("LLM returned empty candidates list.")

        if len(scores) != len(candidates):
            # Fall back to reciprocal-rank scoring
            scores = [1.0 / (i + 1) for i in range(len(candidates))]

        # Sanitise: if admissible list given, keep only valid actions
        if admissible:
            admissible_set = set(admissible)
            filtered = [
                (c, s)
                for c, s in zip(candidates, scores)
                if c in admissible_set
            ]
            if filtered:
                candidates, scores = zip(*filtered)
                candidates, scores = list(candidates), list(scores)
            # If none matched (hallucination), raise so retry kicks in
            if not candidates:
                raise ValueError("All LLM candidates were not in admissible_commands.")

        return candidates[: self.k], scores[: self.k]

    # ------------------------------------------------------------------
    # Padding / trimming helpers
    # ------------------------------------------------------------------

    def _pad_or_trim(
        self,
        candidates: List[str],
        scores: List[float],
        admissible: Optional[List[str]],
    ) -> Tuple[List[str], List[float]]:
        """Ensure we have exactly k candidates if possible."""
        if len(candidates) >= self.k:
            return candidates[: self.k], scores[: self.k]

        # Pad from admissible_commands with lowest score
        if admissible:
            existing = set(candidates)
            for cmd in admissible:
                if cmd not in existing:
                    candidates.append(cmd)
                    scores.append(0.0)
                    existing.add(cmd)
                if len(candidates) >= self.k:
                    break

        return candidates, scores

    def _fallback(
        self, admissible: Optional[List[str]]
    ) -> Tuple[List[str], np.ndarray]:
        if admissible:
            candidates = admissible[: self.k]
        else:
            candidates = ["look", "inventory", "go north", "go south", "wait"][: self.k]
        probs = np.ones(len(candidates), dtype=np.float32) / len(candidates)
        return candidates, probs


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    exp_x = np.exp(x)
    return exp_x / (exp_x.sum() + 1e-8)

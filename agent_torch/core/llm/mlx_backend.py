"""
MLX LLM Backend for AgentTorch
===============================

Provides an LLM backend that runs models natively on Apple Silicon
using the mlx-lm library. No server required — runs entirely in-process
on the Metal GPU.

This is significantly faster than Ollama for Apple Silicon Macs because:
  - No HTTP overhead (in-process inference)
  - Native Metal acceleration via MLX
  - Model stays loaded in memory between calls

Requirements:
    - Apple Silicon Mac (M1/M2/M3/M4)
    - pip install mlx-lm

Usage:
    from agent_torch.core.llm.mlx_backend import MLXLLM

    llm = MLXLLM(
        model="mlx-community/Mistral-7B-Instruct-v0.3-4bit",
        agent_profile="You are a worker deciding whether to retrain...",
    )

    # Or with a local model path:
    llm = MLXLLM(model="./my_local_model")

Popular MLX models (auto-downloaded from HuggingFace):
    - mlx-community/Mistral-7B-Instruct-v0.3-4bit  (~4GB)
    - mlx-community/Llama-3.2-3B-Instruct-4bit      (~2GB)
    - mlx-community/Phi-3.5-mini-instruct-4bit       (~2GB)
    - mlx-community/Qwen2.5-7B-Instruct-4bit         (~4GB)
    - mlx-community/gemma-2-9b-it-4bit               (~5GB)
"""

import re as _re
import time
from typing import List, Union, Dict, Any

from agent_torch.core.llm.backend import LLMBackend


class MLXLLM(LLMBackend):
    """
    MLX-based LLM backend for native Apple Silicon inference.

    Loads a model once into Metal GPU memory and runs all subsequent
    inference calls in-process with zero HTTP overhead.

    Args:
        model: HuggingFace model ID or local path.
               MLX-quantized models from 'mlx-community' are recommended.
        agent_profile: System prompt defining the agent's persona and task.
        temperature: Sampling temperature (default: 0.3)
        max_tokens: Maximum tokens to generate per call (default: 32)
        seed: Random seed for reproducibility (default: 42)
    """

    def __init__(
        self,
        model: str = "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
        agent_profile: str = "",
        temperature: float = 0.3,
        max_tokens: int = 32,
        seed: int = 42,
    ):
        super().__init__()
        self.model_name = model
        self.agent_profile = agent_profile
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed
        self.backend = "mlx"
        self._model = None
        self._tokenizer = None

    def initialize_llm(self):
        """Load the model into Metal GPU memory."""
        try:
            from mlx_lm import load

            print(f"🍎 Loading MLX model: {self.model_name}")
            t0 = time.time()
            self._model, self._tokenizer = load(self.model_name)
            load_time = time.time() - t0
            print(f"✅ MLX backend ready: {self.model_name} ({load_time:.1f}s)")
        except ImportError:
            raise ImportError(
                "mlx-lm is required for MLX backend. "
                "Install with: pip install mlx-lm"
            )
        except Exception as e:
            print(f"⚠ Failed to load MLX model '{self.model_name}': {e}")
            raise
        return self

    def _ensure_loaded(self):
        """Lazy-load model on first call if initialize_llm wasn't called."""
        if self._model is None:
            self.initialize_llm()

    def prompt(self, prompt_list: List[Union[str, Dict[str, Any]]]) -> List[Dict[str, str]]:
        """
        Generate responses for a list of prompts.

        Each prompt can be:
          - str: plain user message
          - dict: {"agent_query": str, "chat_history": list}

        Returns:
            List of {"text": str} dicts, one per prompt.
        """
        self._ensure_loaded()
        from mlx_lm import generate

        outputs = []
        for prompt_input in prompt_list:
            if isinstance(prompt_input, dict):
                user_msg = prompt_input.get("agent_query", str(prompt_input))
                history = prompt_input.get("chat_history", [])
            else:
                user_msg = str(prompt_input)
                history = []

            # Build chat messages
            messages = []
            if self.agent_profile:
                messages.append({"role": "system", "content": self.agent_profile})

            # Add conversation history (last 3 exchanges)
            for entry in history[-6:]:
                if isinstance(entry, dict):
                    if "query" in entry:
                        messages.append({"role": "user", "content": str(entry["query"])})
                    if "output" in entry:
                        out_text = entry["output"]
                        if isinstance(out_text, dict):
                            out_text = out_text.get("text", str(out_text))
                        messages.append({"role": "assistant", "content": str(out_text)})

            messages.append({"role": "user", "content": user_msg})

            # Apply chat template — handle models that don't support system role
            if hasattr(self._tokenizer, "apply_chat_template"):
                try:
                    formatted_prompt = self._tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                except Exception:
                    # Some models (e.g. Mistral) don't allow system role.
                    # Merge system prompt into the first user message.
                    merged = []
                    system_text = ""
                    for msg in messages:
                        if msg["role"] == "system":
                            system_text = msg["content"]
                        else:
                            if system_text and msg["role"] == "user":
                                msg = {
                                    "role": "user",
                                    "content": f"{system_text}\n\n{msg['content']}",
                                }
                                system_text = ""
                            merged.append(msg)
                    formatted_prompt = self._tokenizer.apply_chat_template(
                        merged, tokenize=False, add_generation_prompt=True
                    )
            else:
                # Fallback: simple concatenation
                parts = []
                for msg in messages:
                    parts.append(f"[{msg['role'].upper()}]: {msg['content']}")
                formatted_prompt = "\n".join(parts) + "\n[ASSISTANT]: "

            # Generate — mlx_lm.generate returns the response string directly
            response = generate(
                model=self._model,
                tokenizer=self._tokenizer,
                prompt=formatted_prompt,
                max_tokens=self.max_tokens,
                verbose=False,
            )

            # Extract number from response
            extracted = self._extract_number(response)
            outputs.append({"text": extracted})

        return outputs

    def _extract_number(self, text: str) -> str:
        """Extract a float between 0 and 1 from LLM output text."""
        # Strip common wrapper tags
        text = _re.sub(r"<think>.*?</think>", "", text, flags=_re.DOTALL).strip()
        text = _re.sub(r"<\|.*?\|>", "", text).strip()

        # Try to find a decimal number between 0 and 1
        matches = _re.findall(r"(?:^|[\s:=])([01]?\.\d+|[01](?:\.\d+)?)", text)
        if matches:
            for match in matches:
                try:
                    val = float(match)
                    if 0.0 <= val <= 1.0:
                        return f"{val:.3f}"
                except ValueError:
                    continue

        # Try to find any number and normalize
        matches = _re.findall(r"(\d+(?:\.\d+)?)", text)
        if matches:
            try:
                val = float(matches[0])
                if val > 1.0:
                    val = val / 100.0  # Assume percentage
                val = max(0.0, min(1.0, val))
                return f"{val:.3f}"
            except ValueError:
                pass

        return "0.5"  # Default fallback

    def __call__(self, prompt_inputs: List[Union[str, Dict[str, Any]]]) -> List[Dict[str, str]]:
        """Make the backend directly callable (required by LLMArchetype)."""
        return self.prompt(prompt_inputs)

    def inspect_history(self, last_k, file_dir):
        """No-op for MLX backend."""
        pass

"""
Ollama LLM Backend for AgentTorch
=================================

Provides an LLM backend that calls a local Ollama server for agent behavior generation.
Compatible with the AgentTorch Archetype/Behavior API.

Requirements:
    - Ollama installed and running locally (https://ollama.ai)
    - A model pulled (e.g., `ollama pull llama3.2` or `ollama pull mistral`)

Usage:
    from agent_torch.core.llm.ollama_backend import OllamaLLM

    llm = OllamaLLM(
        model="llama3.2",
        agent_profile="You are a Mississippi worker deciding whether to retrain...",
    )
"""

import json
import urllib.request
import urllib.error
from typing import List, Union, Dict, Any

from agent_torch.core.llm.backend import LLMBackend


class OllamaLLM(LLMBackend):
    """
    Ollama-based LLM backend for local inference.

    Communicates with a running Ollama server via its REST API.
    No external Python packages required — uses only urllib.

    Args:
        model: Ollama model name (e.g., "llama3.2", "mistral", "phi3")
        agent_profile: System prompt defining the agent's persona and task.
                       Should instruct the model to return a single number 0–1.
        base_url: Ollama API endpoint (default: http://localhost:11434)
        temperature: Sampling temperature (default: 0.3 for consistency)
        timeout: Request timeout in seconds (default: 30)
    """

    def __init__(
        self,
        model: str = "ministral-3:8b",
        agent_profile: str = "",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.3,
        timeout: int = 60,
    ):
        super().__init__()
        self.model = model
        self.agent_profile = agent_profile
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.timeout = timeout
        self.backend = "ollama"

    def initialize_llm(self):
        """Verify Ollama is running and model is available."""
        try:
            url = f"{self.base_url}/api/tags"
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode())
                available = [m["name"] for m in data.get("models", [])]
                # Check model availability (model name may include :latest tag)
                model_found = any(
                    self.model in name for name in available
                )
                if not model_found:
                    print(
                        f"⚠ Model '{self.model}' not found in Ollama. "
                        f"Available: {available}. "
                        f"Run: ollama pull {self.model}"
                    )
                else:
                    print(f"✅ Ollama backend ready: model={self.model}")
        except urllib.error.URLError:
            print(
                f"⚠ Cannot connect to Ollama at {self.base_url}. "
                f"Make sure Ollama is running: `ollama serve`"
            )
        return self

    def prompt(self, prompt_list: List[Union[str, Dict[str, Any]]]) -> List[Dict[str, str]]:
        """
        Send prompts to Ollama and return responses.

        Each prompt can be:
          - str: plain user message
          - dict: {"agent_query": str, "chat_history": list}

        Returns:
            List of {"text": str} dicts, one per prompt.
        """
        outputs = []
        for prompt_input in prompt_list:
            if isinstance(prompt_input, dict):
                user_msg = prompt_input.get("agent_query", str(prompt_input))
                history = prompt_input.get("chat_history", [])
            else:
                user_msg = str(prompt_input)
                history = []

            # Build messages array for Ollama chat API
            messages = []
            if self.agent_profile:
                messages.append({"role": "system", "content": self.agent_profile})

            # Add conversation history
            for entry in history[-6:]:  # Last 3 exchanges
                if isinstance(entry, dict):
                    if "query" in entry:
                        messages.append({"role": "user", "content": str(entry["query"])})
                    if "output" in entry:
                        out_text = entry["output"]
                        if isinstance(out_text, dict):
                            out_text = out_text.get("text", str(out_text))
                        messages.append({"role": "assistant", "content": str(out_text)})

            messages.append({"role": "user", "content": user_msg})

            # Call Ollama API
            response_text = self._call_ollama(messages)
            outputs.append({"text": response_text})

        return outputs

    def _call_ollama(self, messages: List[Dict[str, str]]) -> str:
        """Make a single call to the Ollama chat API."""
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": 50,  # Short responses (we only need a number)
            },
        }

        url = f"{self.base_url}/api/chat"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                result = json.loads(resp.read().decode())
                content = result.get("message", {}).get("content", "0.5")
                # Try to extract a numeric value from the response
                return self._extract_number(content)
        except urllib.error.URLError as e:
            print(f"  Ollama request failed: {e}")
            return "0.5"  # Fallback
        except Exception as e:
            print(f"  Ollama error: {e}")
            return "0.5"

    def _extract_number(self, text: str) -> str:
        """Extract a float between 0 and 1 from LLM output text."""
        import re

        # Strip <think>...</think> blocks (qwen3 reasoning traces)
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

        # Try to find a decimal number in the text
        matches = re.findall(r"(?:^|[\s:=])([01]?\.\d+|[01](?:\.\d+)?)", text)
        if matches:
            for match in matches:
                try:
                    val = float(match)
                    if 0.0 <= val <= 1.0:
                        return f"{val:.3f}"
                except ValueError:
                    continue

        # Try to find any number and normalize
        matches = re.findall(r"(\d+(?:\.\d+)?)", text)
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
        """No-op for Ollama backend (no built-in history inspection)."""
        pass

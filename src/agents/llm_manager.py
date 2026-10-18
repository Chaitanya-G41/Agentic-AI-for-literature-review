"""
Multi-API Key & Provider LLM Client Manager
Project: NLP-05 Agentic AI for Automated Research Paper Analysis

Supports:
1. NVIDIA NIM API (Primary - OpenAI-compatible endpoint at https://integrate.api.nvidia.com/v1)
   Models: meta/llama-3.3-70b-instruct, nvidia/llama-3.1-nemotron-70b-instruct
2. Google GenAI (Fallback - google-genai SDK)
   Models: gemini-2.5-flash, gemini-1.5-flash
3. Provenance Tracking (GENUINE_LLM vs HEURISTIC_FALLBACK)
"""

import os
import time
import random
import re
import json

# Try importing OpenAI client for NVIDIA NIM
try:
    from openai import OpenAI
    _OPENAI_AVAILABLE = True
except ImportError:
    OpenAI = None
    _OPENAI_AVAILABLE = False

# Try importing Google GenAI for secondary fallback
try:
    from google import genai
    from google.genai import types as genai_types
    from google.genai import errors as genai_errors
    _GENAI_AVAILABLE = True
except ImportError:
    genai = None
    genai_types = None
    genai_errors = None
    _GENAI_AVAILABLE = False


class LLMManager:
    """
    Unified LLM Manager with NVIDIA NIM primary support,
    key rotation, model fallback, and provenance tracking.
    """
    def __init__(self, api_keys=None, preferred_models=None):
        self._load_dotenv_safe()
        
        # 1. Resolve NVIDIA API Keys (Primary)
        self.nvidia_keys = self._resolve_keys(api_keys, env_vars=["NVIDIA_API_KEY", "NVIDIA_API_KEYS", "NVAPI_KEY"])
        
        # 2. Resolve Gemini API Keys (Secondary Fallback)
        self.gemini_keys = self._resolve_keys(None, env_vars=["GEMINI_API_KEY", "GEMINI_API_KEYS", "GOOGLE_API_KEY"])

        # Models to attempt on NVIDIA NIM
        self.preferred_nvidia_models = preferred_models or [
            "meta/llama-3.3-70b-instruct",
            "nvidia/llama-3.1-nemotron-70b-instruct",
            "mistralai/mixtral-8x22b-instruct-v0.1"
        ]
        
        # Gemini fallback models
        self.preferred_gemini_models = [
            "gemini-2.5-flash",
            "gemini-1.5-flash"
        ]

        self.last_used_model = None
        self.last_provider = None
        self.last_error = None
        self.generation_mode = "HEURISTIC_FALLBACK"

    @property
    def keys(self):
        """Returns any active keys (NVIDIA or Gemini) for backward compatibility."""
        return self.nvidia_keys or self.gemini_keys

        # Log active provider
        if self.nvidia_keys:
            masked = f"...{self.nvidia_keys[0][-6:]}" if len(self.nvidia_keys[0]) > 6 else "***"
            print(f"   [LLM-INIT] NVIDIA NIM Provider Active (Key: {masked}, Model: {self.preferred_nvidia_models[0]})")
        elif self.gemini_keys:
            masked = f"...{self.gemini_keys[0][-6:]}" if len(self.gemini_keys[0]) > 6 else "***"
            print(f"   [LLM-INIT] Google Gemini Fallback Provider Active (Key: {masked})")
        else:
            print(f"   [LLM-INIT] No API keys found (NVIDIA_API_KEY / GEMINI_API_KEY). Rule-based Heuristic Engine will be used.")

    def _load_dotenv_safe(self):
        try:
            from dotenv import load_dotenv, find_dotenv
            env_file = find_dotenv(usecwd=True)
            if env_file:
                load_dotenv(env_file, override=True)
        except Exception:
            pass

    def _resolve_keys(self, custom_keys, env_vars):
        keys = []
        if custom_keys:
            if isinstance(custom_keys, str):
                keys.extend([k.strip().strip('"').strip("'") for k in custom_keys.split(",") if k.strip()])
            elif isinstance(custom_keys, list):
                keys.extend([k.strip().strip('"').strip("'") for k in custom_keys if k and isinstance(k, str) and k.strip()])

        for var in env_vars:
            val = os.environ.get(var, "")
            if val:
                for k in val.split(","):
                    cleaned = k.strip().strip('"').strip("'")
                    if cleaned and cleaned not in keys:
                        keys.append(cleaned)
        return keys

    def generate_text(self, prompt: str, system_instruction: str = None, max_retries: int = 3) -> str:
        """
        Executes text generation. Tries NVIDIA NIM first, then Gemini, then returns None for Heuristic fallback.
        """
        # Try NVIDIA NIM first
        if self.nvidia_keys and _OPENAI_AVAILABLE:
            for key in self.nvidia_keys:
                client = OpenAI(
                    base_url="https://integrate.api.nvidia.com/v1",
                    api_key=key
                )
                messages = []
                if system_instruction:
                    messages.append({"role": "system", "content": system_instruction})
                messages.append({"role": "user", "content": prompt})

                for model in self.preferred_nvidia_models:
                    for attempt in range(max_retries):
                        try:
                            completion = client.chat.completions.create(
                                model=model,
                                messages=messages,
                                temperature=0.2,
                                top_p=0.7,
                                max_tokens=1500
                            )
                            if completion and completion.choices:
                                text = completion.choices[0].message.content
                                if text:
                                    self.last_used_model = model
                                    self.last_provider = "NVIDIA_NIM"
                                    self.generation_mode = "GENUINE_LLM"
                                    return text.strip()
                        except Exception as e:
                            self.last_error = f"NVIDIA NIM error ({model}): {e}"
                            time.sleep(1.5 ** attempt)
                            continue

        # Secondary fallback: Gemini
        if self.gemini_keys and _GENAI_AVAILABLE:
            for key in self.gemini_keys:
                try:
                    client = genai.Client(api_key=key)
                    config = genai_types.GenerateContentConfig(
                        system_instruction=system_instruction
                    ) if system_instruction else None

                    for model in self.preferred_gemini_models:
                        try:
                            res = client.models.generate_content(
                                model=model,
                                contents=prompt,
                                config=config
                            )
                            if res and getattr(res, "text", None):
                                self.last_used_model = model
                                self.last_provider = "GOOGLE_GEMINI"
                                self.generation_mode = "GENUINE_LLM"
                                return res.text.strip()
                        except Exception as e:
                            self.last_error = f"Gemini error ({model}): {e}"
                            continue
                except Exception as e:
                    self.last_error = f"Gemini client error: {e}"
                    continue

        self.generation_mode = "HEURISTIC_FALLBACK"
        return None

    def generate_structured_json(self, prompt: str, system_instruction: str = None, max_retries: int = 3, response_schema=None):
        """
        Generates and parses a JSON response from the LLM.
        """
        augmented_prompt = prompt + "\n\nCRITICAL: Return ONLY a valid JSON object matching the requested schema. No markdown formatting, no conversational text."
        raw_text = self.generate_text(augmented_prompt, system_instruction=system_instruction, max_retries=max_retries)
        
        if not raw_text:
            return None

        # Clean ```json code fence blocks
        cleaned = re.sub(r'```json\s*', '', raw_text)
        cleaned = re.sub(r'```\s*$', '', cleaned).strip()

        # Locate JSON block
        json_match = re.search(r'(\{.*\}|\[.*\])', cleaned, re.DOTALL)
        if json_match:
            cleaned = json_match.group(0)

        try:
            return json.loads(cleaned)
        except Exception as e:
            self.last_error = f"JSON parse error: {e}"
            return None


# Backward-compatible alias for existing agent imports
GeminiLLMManager = LLMManager
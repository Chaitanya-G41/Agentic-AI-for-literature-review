"""
Multi-API Key Manager & LLM Client Wrapper
Project: NLP-05 Agentic AI for Automated Research Paper Analysis
Handles round-robin API key rotation, rate-limit fallback (HTTP 429 / 403), exponential backoff, and model failover.
"""

import os
import time
import random
import re
import json

class GeminiLLMManager:
    def __init__(self, api_keys=None, preferred_models=None):
        """
        Initializes the Gemini LLM Manager with multi-key rotation and model fallback.
        api_keys: List of string keys or comma-separated env var GEMINI_API_KEYS / GEMINI_API_KEY.
        preferred_models: List of model names to attempt in order of preference.
        """
        self.keys = self._resolve_api_keys(api_keys)
        self.current_key_idx = 0
        self.cooldowns = {} # key -> timestamp until available
        self.preferred_models = preferred_models or [
            "gemini-3.6-flash",
            "models/gemini-3.6-flash",
            "gemini-1.5-flash",
            "gemini-1.5-pro",
            "gemini-2.0-flash"
        ]
        self.last_used_model = None
        self.last_error = None

    def _resolve_api_keys(self, custom_keys):
        keys = []
        source = "none"
        if custom_keys:
            if isinstance(custom_keys, str):
                keys = [k.strip() for k in custom_keys.split(",") if k.strip()]
            elif isinstance(custom_keys, list):
                keys = [k.strip() for k in custom_keys if k and isinstance(k, str)]
            if keys:
                source = "direct parameter"
        
        if not keys:
            env_keys = os.environ.get("GEMINI_API_KEYS", "") or os.environ.get("GEMINI_API_KEY", "")
            if env_keys:
                keys = [k.strip() for k in env_keys.split(",") if k.strip()]
                if keys:
                    source = "environment variable"
        
        if keys:
            masked = [f"...{k[-6:]}" for k in keys]
            print(f"   [KEY-RESOLVE] Loaded {len(keys)} API key(s) from {source}: {masked}")
        else:
            print(f"   [KEY-RESOLVE] No API keys found (checked: direct param, GEMINI_API_KEYS, GEMINI_API_KEY).")
        
        return keys

    def get_valid_key(self):
        """Returns an active API key not currently in rate-limit cooldown."""
        if not self.keys:
            return None
        
        now = time.time()
        num_keys = len(self.keys)
        
        for idx_offset in range(num_keys):
            idx = (self.current_key_idx + idx_offset) % num_keys
            key = self.keys[idx]
            cooldown_until = self.cooldowns.get(key, 0)
            if now >= cooldown_until:
                self.current_key_idx = (idx + 1) % num_keys
                return key
                
        # If all keys are in cooldown, return the one that expires soonest
        soonest_key = min(self.keys, key=lambda k: self.cooldowns.get(k, 0))
        wait_time = max(0, self.cooldowns.get(soonest_key, 0) - now)
        if wait_time > 0 and wait_time < 10:
            time.sleep(wait_time)
        return soonest_key

    def mark_key_rate_limited(self, key, cooldown_seconds=60):
        """Marks a key as rate-limited for a duration."""
        if key:
            print(f"   [RATE-LIMIT] Key ending in '...{key[-4:]}' rate-limited. Cooling down for {cooldown_seconds}s.")
            self.cooldowns[key] = time.time() + cooldown_seconds

    def _fetch_supported_models(self, genai_module, api_key):
        """
        Dynamically queries Google ModelService via genai.list_models()
        to discover exact model names supported by the user's API key for generateContent.
        Always prioritizes gemini-3.6-flash at top.
        """
        discovered = []
        try:
            genai_module.configure(api_key=api_key)
            for m in genai_module.list_models():
                methods = getattr(m, 'supported_generation_methods', [])
                if 'generateContent' in methods:
                    name = m.name.replace("models/", "")
                    discovered.append(name)
                    if m.name not in discovered:
                        discovered.append(m.name)
        except Exception as e:
            print(f"   [WARN] Dynamic model discovery skipped ({e}). Using preferred model list.")

        # Ensure preferred models like gemini-3.6-flash are included
        for pm in self.preferred_models:
            if pm not in discovered:
                discovered.append(pm)

        # Prioritize models: 3.6 > 2.5 > 2.0 > 1.5
        def priority(name):
            score = 0
            if "flash" in name: score += 20
            if "3.6" in name: score += 60
            elif "2.5" in name: score += 50
            elif "2.0" in name: score += 40
            elif "1.5" in name: score += 30
            if "latest" in name: score += 5
            if "exp" in name: score -= 2
            if not name.startswith("models/"): score += 1
            return -score

        discovered.sort(key=priority)
        return discovered

    def generate_text(self, prompt, system_instruction=None, max_retries=3):
        """
        Executes text generation using active API key rotation and model fallback.
        Returns generated string or None on failure.
        """
        self.last_used_model = None
        self.last_error = None

        if not self.keys:
            self.last_error = "No Gemini API keys configured in environment or UI."
            print(f"   [INFO] {self.last_error}")
            return None

        try:
            import google.generativeai as genai
        except ImportError:
            self.last_error = "google-generativeai package not installed."
            print(f"   [ERROR] {self.last_error}")
            return None

        for attempt in range(max_retries):
            key = self.get_valid_key()
            if not key:
                self.last_error = "All API keys in rate-limit cooldown."
                break

            genai.configure(api_key=key)

            # Dynamically resolve available models for this key
            active_models = self._fetch_supported_models(genai, key)

            for model_name in active_models:
                try:
                    kwargs = {}
                    if system_instruction:
                        model = genai.GenerativeModel(model_name=model_name, system_instruction=system_instruction)
                    else:
                        model = genai.GenerativeModel(model_name=model_name)

                    res = model.generate_content(prompt, **kwargs)
                    if res and hasattr(res, "text") and res.text:
                        self.last_used_model = model_name
                        return res.text.strip()
                except Exception as e:
                    self.last_error = str(e)
                    err_str = str(e).lower()
                    if "429" in err_str or "quota" in err_str or "rate limit" in err_str:
                        self.mark_key_rate_limited(key, cooldown_seconds=30)
                        break # Switch key on 429
                    elif "not found" in err_str or "invalid model" in err_str or "not supported" in err_str:
                        continue # Try next candidate model from dynamically resolved list
                    else:
                        print(f"   [WARN] LLM Call error with model {model_name}: {e}")
                        break

            # Backoff before retrying with next key
            sleep_dur = (2 ** attempt) + random.uniform(0.1, 0.5)
            time.sleep(sleep_dur)

        return None

    def generate_structured_json(self, prompt, system_instruction=None, max_retries=3):
        """
        Executes LLM text generation and extracts clean parsed JSON object.
        """
        raw_text = self.generate_text(prompt, system_instruction=system_instruction, max_retries=max_retries)
        if not raw_text:
            return None

        clean_json_str = re.sub(r'```json|```', '', raw_text).strip()
        try:
            return json.loads(clean_json_str)
        except json.JSONDecodeError:
            # Fallback regex search for JSON object block {...}
            match = re.search(r'\{.*\}', raw_text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    pass
        return None

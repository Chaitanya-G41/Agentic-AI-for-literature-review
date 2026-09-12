"""
Multi-API Key Manager & LLM Client Wrapper
Project: NLP-05 Agentic AI for Automated Research Paper Analysis
Handles round-robin API key rotation, rate-limit fallback (429/401), exponential
backoff, and model failover.

NOTE: This was rewritten to use the current `google-genai` SDK
(`pip install google-genai`, `from google import genai`).

The previous version imported `google.generativeai`, which Google fully
end-of-lifed on Nov 30, 2025 (no more bug fixes, and it never reliably
supported Gemini 3.x models). That mismatch was the actual cause of the
"key not getting configured" symptom: key resolution was working fine, but
every model call downstream was silently failing against a dead SDK, so
every attempt fell through to the heuristic fallback engine.

Public interface is unchanged on purpose, so callers (e.g. the Step 3
Summarizer Agent) do not need to change:
    GeminiLLMManager(api_keys=..., preferred_models=...)
        .keys
        .generate_text(prompt, system_instruction=None, max_retries=3)
        .generate_structured_json(prompt, system_instruction=None, max_retries=3)
        .last_used_model
        .last_error
"""

import os
import time
import random
import re
import json

try:
    from google import genai
    from google.genai import types as genai_types
    from google.genai import errors as genai_errors
    _GENAI_IMPORT_ERROR = None
except ImportError as e:
    genai = None
    genai_types = None
    genai_errors = None
    _GENAI_IMPORT_ERROR = str(e)


class GeminiLLMManager:
    def __init__(self, api_keys=None, preferred_models=None):
        """
        Initializes the Gemini LLM Manager with multi-key rotation and model fallback.
        api_keys: List of string keys or comma-separated env var GEMINI_API_KEYS / GEMINI_API_KEY.
        preferred_models: List of model names to attempt in order of preference.
        """
        self.keys = self._resolve_api_keys(api_keys)
        self.current_key_idx = 0
        self.cooldowns = {}  # key -> timestamp until available

        # Curated list of currently-live Gemini models (checked Sept 2026).
        # gemini-1.5-flash and gemini-2.0-flash have both been retired and
        # now 404 for everyone, so they've been dropped from the default list.
        self.preferred_models = preferred_models or [
            "gemini-3.6-flash",
            "gemini-3.5-flash",
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
        ]
        self.last_used_model = None
        self.last_error = None

    def _resolve_api_keys(self, custom_keys):
        keys = []
        source = "none"

        # Always reload .env file to catch any changes immediately
        try:
            from dotenv import load_dotenv, find_dotenv
            env_file = find_dotenv(usecwd=True)
            if env_file:
                load_dotenv(env_file, override=True)
        except Exception:
            pass

        def clean_key(k):
            if not k or not isinstance(k, str):
                return ""
            return k.strip().strip('"').strip("'").strip()

        if custom_keys:
            if isinstance(custom_keys, str):
                raw_list = custom_keys.split(",")
            elif isinstance(custom_keys, list):
                raw_list = custom_keys
            else:
                raw_list = []

            for k in raw_list:
                cleaned = clean_key(k)
                if cleaned:
                    keys.append(cleaned)
            if keys:
                source = "direct parameter / session state"

        if not keys:
            env_val = (
                os.environ.get("GEMINI_API_KEYS", "") or
                os.environ.get("GEMINI_API_KEY", "") or
                os.environ.get("GOOGLE_API_KEY", "")
            )
            if env_val:
                for k in env_val.split(","):
                    cleaned = clean_key(k)
                    if cleaned:
                        keys.append(cleaned)
                if keys:
                    source = "environment variable (.env / os.environ)"

        if keys:
            masked = [f"...{k[-6:]}" if len(k) >= 6 else "***" for k in keys]
            print(f"   [KEY-RESOLVE] Loaded {len(keys)} API key(s) from {source}: {masked}")
        else:
            print("   [KEY-RESOLVE] No API keys found (checked: direct param, .env, GEMINI_API_KEYS, GEMINI_API_KEY, GOOGLE_API_KEY).")

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
        if 0 < wait_time < 10:
            time.sleep(wait_time)
        return soonest_key

    def mark_key_rate_limited(self, key, cooldown_seconds=60):
        """Marks a key as rate-limited for a duration."""
        if key:
            print(f"   [RATE-LIMIT] Key ending in '...{key[-4:]}' rate-limited. Cooling down for {cooldown_seconds}s.")
            self.cooldowns[key] = time.time() + cooldown_seconds

    def generate_text(self, prompt, system_instruction=None, max_retries=3, as_json=False, response_schema=None):
        """
        Executes text generation using active API key rotation and model fallback.
        Returns generated string or None on failure.

        as_json: when True, tells Gemini to return raw JSON directly
        (response_mime_type="application/json") instead of free-form text
        that might come wrapped in markdown fences or trailing commentary.

        response_schema: optional Pydantic model class. When set, Gemini's
        decoder is constrained to match that schema's field types exactly
        (e.g. a List[str] field literally cannot come back as a plain string),
        instead of only being asked to via prompt instructions. Implies as_json.
        """
        self.last_used_model = None
        self.last_error = None

        if not self.keys:
            self.last_error = "No Gemini API keys configured in environment or UI."
            print(f"   [INFO] {self.last_error}")
            return None

        if genai is None:
            self.last_error = f"google-genai package not installed ({_GENAI_IMPORT_ERROR}). Run: pip install google-genai"
            print(f"   [ERROR] {self.last_error}")
            return None

        config_kwargs = {}
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction
        if as_json or response_schema is not None:
            config_kwargs["response_mime_type"] = "application/json"
        if response_schema is not None:
            config_kwargs["response_schema"] = response_schema
        config = genai_types.GenerateContentConfig(**config_kwargs) if config_kwargs else None

        for attempt in range(max_retries):
            key = self.get_valid_key()
            if not key:
                self.last_error = "All API keys in rate-limit cooldown."
                break

            try:
                client = genai.Client(api_key=key)
            except Exception as e:
                self.last_error = f"Failed to create Gemini client: {e}"
                print(f"   [ERROR] {self.last_error}")
                continue

            got_response = False

            for model_name in self.preferred_models:
                try:
                    res = client.models.generate_content(
                        model=model_name,
                        contents=prompt,
                        config=config,
                    )
                    if res and getattr(res, "text", None):
                        self.last_used_model = model_name
                        return res.text.strip()
                except genai_errors.APIError as e:
                    code = getattr(e, "code", None)
                    err_str = str(e).lower()

                    if code == 401 or "api key not valid" in err_str or "api_key_invalid" in err_str:
                        self.last_error = f"API key invalid or rejected by Google (key ending in ...{key[-4:]}). Please verify your Gemini API key."
                        print(f"   [ERROR] {self.last_error}")
                        break  # this key is bad, stop trying other models with it

                    elif code == 429 or "quota" in err_str or "rate limit" in err_str:
                        self.mark_key_rate_limited(key, cooldown_seconds=30)
                        break  # switch key on 429

                    elif code == 404 or any(k in err_str for k in ["not found", "invalid model", "not supported", "no longer available", "deprecated", "does not exist"]):
                        print(f"   [INFO] Model {model_name} unavailable ({e}). Trying next model in list...")
                        continue  # try next candidate model

                    else:
                        self.last_error = str(e)
                        print(f"   [WARN] LLM call error with model {model_name}: {e}")
                        continue

                except Exception as e:
                    self.last_error = str(e)
                    print(f"   [WARN] Unexpected error with model {model_name}: {e}")
                    continue

            if got_response:
                break

            # Backoff before retrying with next key
            sleep_dur = (2 ** attempt) + random.uniform(0.1, 0.5)
            time.sleep(sleep_dur)

        return None

    def generate_structured_json(self, prompt, system_instruction=None, max_retries=3, response_schema=None):
        """
        Executes LLM text generation and extracts clean parsed JSON object.

        response_schema: optional Pydantic model class to constrain Gemini's
        output shape directly (see generate_text docstring). Recommended for
        any schema with List[...] fields, since it prevents Gemini from
        collapsing a list field into a single string.
        """
        raw_text = self.generate_text(
            prompt,
            system_instruction=system_instruction,
            max_retries=max_retries,
            as_json=True,
            response_schema=response_schema,
        )
        if not raw_text:
            # last_error is already set by generate_text (bad key, rate limit, no models, etc.)
            return None

        clean_json_str = re.sub(r'```json|```', '', raw_text).strip()
        try:
            return json.loads(clean_json_str)
        except json.JSONDecodeError:
            match = re.search(r'\{.*\}', raw_text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    pass

        # The API call succeeded (we got text back) but it wasn't valid JSON.
        # Record this explicitly so callers don't misreport it as "no API key".
        self.last_error = f"LLM call succeeded but response was not valid JSON (first 200 chars: {raw_text[:200]!r})"
        print(f"   [WARN] {self.last_error}")
        return None